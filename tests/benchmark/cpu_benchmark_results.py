# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Analyze CPU thread scalability benchmarks from p3s benchmark results.

This script reads benchmark results from JSON files and analyzes thread scalability
by extracting methods matching the pattern cpp-{N}thr where N is a power of 2.

The script calculates:
  - Speedup: ratio of execution time for 1 thread vs N threads
  - Efficiency: speedup / number of threads (ideal = 1.0)
  - Absolute execution times per thread count

Usage:
    python cpu_benchmark_results.py --input-files file1.json file2.json
    python cpu_benchmark_results.py --input-files file1.json file2.json --output-dir plots
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    from compile_results import COLORS
except ImportError:
    COLORS = []

try:
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


# Regex pattern to match thread count in method names like "cpp-1thr", "cpp-128thr"
THREAD_PATTERN = re.compile(r"cpp-(\d+)thr$")


def extract_thread_count(method_name: str) -> int | str | None:
    """Extract thread count from method name matching cpp-{N}thr pattern.

    Returns None for non-matching patterns, the thread count for cpp-{N}thr patterns,
    and the string "N" for the special cpp-Nthr pattern (all threads).
    """
    match = THREAD_PATTERN.match(method_name)
    if match:
        return int(match.group(1))
    return None


def load_result_file(filepath: Path) -> dict:
    """Load a single result JSON file."""
    with open(filepath) as f:
        return json.load(f)


def extract_times(method_data: dict, T_values: list[int]) -> list[float] | None:
    """Extract the time value from method results for a given T_values length."""
    time_ms = method_data.get("time_ms", [])
    if not time_ms:
        return None

    if len(time_ms) == len(T_values):
        return time_ms
    elif len(time_ms) == 1:
        return [time_ms[0]] * len(T_values)
    return None


def mean_time(time_list: list[float]) -> float:
    """Calculate mean of time values, ignoring None."""
    valid_times = [t for t in time_list if t is not None]
    if not valid_times:
        return 0.0
    return sum(valid_times) / len(valid_times)


def analyze_scalability(combined_results: dict) -> dict:
    """
    Analyze thread scalability for each case.

    Returns a dict with:
      - per_case: {case_name: {T_value: {thread_count: time_ms}}}
      - speedup: {case_name: {T_value: {thread_count: speedup}}}
      - efficiency: {case_name: {T_value: {thread_count: efficiency}}}
    """
    analysis = {
        "per_case": {},
        "speedup": {},
        "efficiency": {},
    }

    for case_name, result in combined_results.items():
        T_values = result.get("T_values", [])
        methods = result.get("methods", {})

        # Methods are already filtered in load_and_combine_results, use directly
        thread_methods = methods

        # Get sorted thread counts
        # Sort thread counts numerically only (no "N" at this point)
        thread_counts = thread_methods.keys()
        thread_counts = sorted(thread_counts, key=lambda tc: tc if isinstance(tc, int) else float("inf"))

        # Extract times for each thread count and T value
        per_case_times = defaultdict(dict)
        for t_idx, T in enumerate(T_values):
            for tc in thread_counts:
                method_data = thread_methods[tc]
                times = extract_times(method_data, T_values)
                if times and times[t_idx] is not None:
                    per_case_times[T][tc] = times[t_idx]

        analysis["per_case"][case_name] = dict(per_case_times)

        # Calculate speedup and efficiency
        case_speedup = {}
        case_efficiency = {}

        for T in per_case_times:
            # Get reference time (1-thread or minimum thread count)
            int_thread_counts = [tc for tc in per_case_times[T].keys() if isinstance(tc, int)]
            base_tc = min(int_thread_counts) if int_thread_counts else None
            base_time = per_case_times[T][base_tc]

            case_speedup[T] = {}
            case_efficiency[T] = {}

            for tc, time in per_case_times[T].items():
                # Speedup = time_1thread / time_Nthreads
                speedup = base_time / time if time > 0 else None
                # Efficiency = speedup / num_threads (ideal = 1.0)
                efficiency = speedup / tc if speedup is not None and isinstance(tc, int) and tc > 0 else None

                case_speedup[T][tc] = speedup
                case_efficiency[T][tc] = efficiency

        analysis["speedup"][case_name] = case_speedup
        analysis["efficiency"][case_name] = case_efficiency

    return analysis


def generate_absolute_times_graph(
    analysis: dict,
    output_dir: str,
) -> None:
    """Generate absolute execution time vs thread count graphs for each T value."""
    if not _HAS_MATPLOTLIB:
        print("matplotlib not available, skipping graph generation")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["legend.fontsize"] = 14

    times_data = analysis.get("per_case", {})

    all_T_values = set()
    for case_times in times_data.values():
        all_T_values.update(case_times.keys())
    all_T_values = sorted(all_T_values)

    cases_to_plot = sorted(times_data.keys())

    color_map = {}
    for i, case_name in enumerate(cases_to_plot):
        color_map[case_name] = COLORS[i % len(COLORS)] if COLORS else None

    for T in all_T_values:
        plt.figure(figsize=(12, 8))

        for case_name in cases_to_plot:
            if T not in times_data.get(case_name, {}):
                continue

            case_times = times_data[case_name][T]
            thread_counts = sorted(case_times.keys(), key=lambda tc: tc if isinstance(tc, int) else float("inf"))
            times = [case_times[tc] / 1000 for tc in thread_counts]

            plt.plot(
                thread_counts,
                times,
                marker="D",
                linewidth=2,
                markersize=8,
                color=color_map.get(case_name),
                label=f"{case_name}",
            )

        plt.xlabel("Number of threads", fontsize=18)
        plt.ylabel("Execution time (seconds)", fontsize=18)
        plt.title(f"Execution Time vs Thread Count (T={T} timesteps)", fontsize=24)
        plt.xscale("log", base=2)

        max_threads = max(tc for tc in thread_counts if isinstance(tc, int))
        xticks = []
        xtick_labels = []
        exponent = 0
        while (2**exponent) <= max_threads:
            xticks.append(2**exponent)
            xtick_labels.append(str(2**exponent))
            exponent += 1
        plt.xticks(xticks, xtick_labels)

        plt.yscale("log")
        plt.grid(True, alpha=0.3, which="both")

        if len(cases_to_plot) <= 3:
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=min(len(cases_to_plot), 3))
        else:
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()

        filepath = output_path / f"time_T{T}.png"
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {filepath}")


def print_summary(analysis: dict) -> None:
    """Print a summary of the analysis results."""
    print("\n" + "=" * 80)
    print("CPU Thread Scalability Analysis Summary")
    print("=" * 80)

    speedup_data = analysis.get("speedup", {})
    efficiency_data = analysis.get("efficiency", {})
    times_data = analysis.get("per_case", {})

    for case_name in sorted(speedup_data.keys()):
        print(f"\nCase: {case_name}")
        print("-" * 40)

        for T in sorted(speedup_data[case_name].keys()):
            case_speedup = speedup_data[case_name][T]
            case_efficiency = efficiency_data[case_name].get(T, {})
            case_times = times_data.get(case_name, {}).get(T, {})

            print(f"  T={T}:")
            print(f"    {'Threads':<10} {'Time (ms)':<12} {'Speedup':<12} {'Efficiency':<12}")
            print(f"    {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 12}")

            # Sort thread counts for display
            for tc in sorted(case_speedup.keys(), key=lambda tc: tc if isinstance(tc, int) else float("inf")):
                time_ms = case_times.get(tc, "N/A")
                speedup = case_speedup.get(tc, "N/A")
                efficiency = case_efficiency.get(tc, "N/A")

                if isinstance(time_ms, (int, float)):
                    time_str = f"{time_ms:.1f}"
                else:
                    time_str = str(time_ms)

                if isinstance(speedup, (int, float)):
                    speedup_str = f"{speedup:.2f}x"
                else:
                    speedup_str = str(speedup)

                if isinstance(efficiency, (int, float)):
                    eff_str = f"{efficiency:.2%}"
                else:
                    eff_str = str(efficiency)

                print(f"    {tc:<10} {time_str:<12} {speedup_str:<12} {eff_str:<12}")


def load_and_combine_results(input_files: list[Path]) -> dict:
    """Load specified result files and combine them by case+T_values.

    For duplicate methods (same thread count) in same case with identical T_values,
    the mean of time_ms values is computed.

    cpp-Nthr methods are filtered out and reported as warnings.
    """
    print(f"Loading {len(input_files)} input file(s)")

    combined_results = {}

    for filepath in input_files:
        try:
            result = load_result_file(filepath)

            case_name = result.get("case", "")
            T_values = tuple(result.get("T_values", []))
            methods = result.get("methods", {})

            # Extract thread-scalable methods only (filter out cpp-Nthr)
            thread_methods = {}
            for method_name, method_data in methods.items():
                if method_name == "cpp-Nthr":
                    print(f"Warning: {filepath.name} has cpp-Nthr method (excluded from plotting)")
                    continue
                thread_count = extract_thread_count(method_name)
                if thread_count is not None:
                    thread_methods[thread_count] = method_data

            if not thread_methods:
                print(f"Warning: No thread-scalable methods in {filepath.name}")
                continue

            case_key = (case_name, T_values)
            if case_key not in combined_results:
                combined_results[case_key] = {"case": case_name, "T_values": list(T_values), "methods": {}}

            existing_methods = combined_results[case_key]["methods"]

            for tc, method_data in thread_methods.items():
                if tc not in existing_methods:
                    existing_methods[tc] = method_data
                else:
                    existing_time_ms = existing_methods[tc].get("time_ms", [])
                    new_time_ms = method_data.get("time_ms", [])

                    if len(existing_time_ms) == len(new_time_ms):
                        merged_time_ms = [
                            mean_time([e, n]) for e, n in zip(existing_time_ms, new_time_ms, strict=False)
                        ]
                        existing_methods[tc]["time_ms"] = merged_time_ms
                        print(f"  Averaged {tc}thr across {filepath.name} and existing")

        except Exception as e:
            print(f"Warning: Failed to load {filepath.name}: {e}")
            continue

    result_dict = {}
    for (case_name, _), result in combined_results.items():
        result_dict[case_name] = result

    print(f"Loaded {len(result_dict)} unique case(s)")
    return result_dict


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CPU thread scalability benchmarks from p3s results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cpu_benchmark_results.py --input-files results/file1.json results/file2.json
    python cpu_benchmark_results.py --input-files results/file1.json --output-dir plots
        """,
    )

    parser.add_argument(
        "--input-files",
        "-f",
        type=str,
        nargs="+",
        required=True,
        help="Input JSON files to analyze",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./plots",
        help="Output directory for generated graphs",
    )
    parser.add_argument(
        "--no-graphs",
        action="store_true",
        help="Skip graph generation (only print summary)",
    )

    args = parser.parse_args()

    input_paths = [Path(f) for f in args.input_files]
    for p in input_paths:
        if not p.exists():
            print(f"Error: File not found: {p}")
            return 1

    try:
        combined_results = load_and_combine_results(input_paths)
    except ValueError as e:
        print(f"Failed to load results: {e}")
        return 1

    if not combined_results:
        print("No results found with thread-scalable methods (cpp-{N}thr)")
        return 1

    print("\nAnalyzing thread scalability...")
    analysis = analyze_scalability(combined_results)

    print_summary(analysis)

    if not args.no_graphs:
        print(f"\nGenerating graphs to: {args.output_dir}")
        generate_absolute_times_graph(analysis, args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
