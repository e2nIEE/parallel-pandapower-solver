#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compile benchmark results from multiple job runs into a single summary table.

Searches OUTPUT_DIR for JSON result files and aggregates them into:
  - A comprehensive CSV with all benchmark results
  - Per-method summary tables
  - Per-case summary tables

Usage:
        python compile_results.py --output-dir /path/to/results
        python compile_results.py --output-dir results
        python compile_results.py --method gpu
        python compile_results.py --include-partial --include-error-unknown
        python compile_results.py --use-global-extrema
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from benchmark_batched import BenchmarkResults, MethodResults

COLORS = [
    (57 / 256, 55 / 256, 139 / 256),  # fh-8
    (226 / 256, 0 / 256, 26 / 256),  # fh-10
    (143 / 256, 164 / 256, 2 / 256),  # fh-24
    (37 / 256, 186 / 256, 226 / 256),  # fh-34
    (253 / 256, 195 / 256, 0 / 256),  # fh-20
    (235 / 256, 106 / 256, 10 / 256),  # fh-16
]


def method_results_compare(m1: MethodResults, m2: MethodResults, ts1: str, ts2: str) -> MethodResults:
    """Compare two MethodResults and return the preferred one.

    Priority:
    1. Status: success > partial > fail
    2. If one has errors and other does not, prefer without errors
    3. Compare time_ms values with 1e-6 precision, if match keep both
    4. If don't match, use newer timestamp
    """
    status_order = {"success": 3, "partial": 2, "fail": 1, "not_available": 0}

    status1 = m1.get("status", "not_available")
    status2 = m2.get("status", "not_available")

    if status_order.get(status1, 0) > status_order.get(status2, 0):
        return m1
    if status_order.get(status1, 0) < status_order.get(status2, 0):
        return m2

    errors1 = m1.get("errors", [])
    errors2 = m2.get("errors", [])

    if errors1 and not errors2:
        return m2
    if errors2 and not errors1:
        return m1

    time1 = m1.get("time_ms", [])
    time2 = m2.get("time_ms", [])

    if len(time1) == len(time2):
        match = True
        for t1, t2 in zip(time1, time2, strict=False):
            if not (np.isnan(t1) and np.isnan(t2)):
                if abs(t1 - t2) > 1e-6:
                    match = False
                    break
        if match:
            max_err1 = m1.get("max_vm_error", [])
            max_err2 = m2.get("max_vm_error", [])
            if len(max_err1) == len(max_err2):
                for e1, e2 in zip(max_err1, max_err2, strict=False):
                    if e1 is not None and e2 is not None:
                        if abs(e1 - e2) > 1e-6:
                            match = False
                            break
            if match:
                return m1

    ts_obj1 = datetime.fromisoformat(ts1)
    ts_obj2 = datetime.fromisoformat(ts2)

    if ts_obj1 > ts_obj2:
        return m1
    return m2


def find_result_files(output_dir):
    """Find all result JSON files in output directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        raise ValueError(f"Output directory does not exist: {output_dir}")

    json_files = list(output_path.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {output_dir}")

    return json_files


def load_result_file(filepath) -> BenchmarkResults:
    """Load a single result JSON file."""
    with open(filepath) as f:
        return json.load(f)


def combine_benchmark_results(
    results_list: list[BenchmarkResults],
) -> dict[str, BenchmarkResults]:
    """Combine BenchmarkResults with same case and T_values.

    Group results by (case, T_values). For each group, combine methods:
    - prefer status success > partial > fail
    - if one has errors and other doesn't, prefer without errors
    - compare time_ms with 1e-6 precision; if match keep both
    - otherwise use newer timestamp
    """
    groups: dict[tuple, list[BenchmarkResults]] = defaultdict(list)

    for result in results_list:
        key = (result["case"], tuple(result["T_values"]))
        groups[key].append(result)

    combined: dict[str, BenchmarkResults] = {}

    if len(groups) == 1:
        case, _ = groups.keys()[0]
        combined[case] = groups[case][0]
        return combined

    for (case, t_values), group in groups.items():
        newest_ts = max(r["timestamp"] for r in group)
        newest_job_id = next(r["job_id"] for r in group if r["timestamp"] == newest_ts)
        bus_count = group[0]["bus_count"]  # should be identical for all BenchmarkResults where case is identical

        combined_methods: dict = {}

        for result in group:
            for method, method_results in result["methods"].items():
                if method not in combined_methods:
                    combined_methods[method] = (method_results, result["timestamp"])
                else:
                    existing, existing_ts = combined_methods[method]
                    better = method_results_compare(existing, method_results, existing_ts, result["timestamp"])
                    combined_methods[method] = (better, result["timestamp"])

        combined_result: BenchmarkResults = BenchmarkResults(
            case=case,
            job_id=newest_job_id,
            bus_count=bus_count,
            timestamp=newest_ts,
            T_values=list(t_values),
            methods={m: mr for m, (mr, _) in combined_methods.items()},
        )
        combined[case] = combined_result

    return combined


def generate_time_vs_timesteps_graph(
    combined_results: dict[str, BenchmarkResults],
    output_dir: str,
    include_partial=False,
    include_error_unknown=False,
    include_fail=False,
    include_not_available=False,
    use_global_extrema=False,
) -> None:
    """Generate time vs timesteps graph for all methods on single graph.

    Y-axis: time_ms converted to seconds (log scale)
    X-axis: T_values
    All methods on one graph, each method has 2 lines:
    - Best case: solid line
    - Worst case: dotted line
    Same color for both lines of a method.

    Extrema behavior:
    - Per-method (default): For each method, find that method's best and worst case
    - Global (--use-global-extrema): Find single best and worst across all methods
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.rcParams["axes.titlesize"] = 32
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["xtick.labelsize"] = 24
    plt.rcParams["ytick.labelsize"] = 24
    plt.rcParams["legend.fontsize"] = 24

    methods_in_data: set[str] = set()
    for _, result in combined_results.items():
        methods_in_data.update(result["methods"].keys())

    color_map = {}
    colors = COLORS
    for i, method in enumerate(sorted(methods_in_data)):
        color_map[method] = colors[i % len(colors)]

    plt.figure(figsize=(18, 8))

    best_all_t_values = set()
    worst_all_t_values = set()

    all_times = []

    all_case_method_times = []

    for case, result in combined_results.items():
        t_values = result["T_values"]
        if not t_values:
            continue
        max_t = max(t_values)
        max_t_idx = t_values.index(max_t)

        for method in sorted(methods_in_data):
            if method not in result["methods"]:
                continue

            method_data = result["methods"][method]
            status = method_data.get("status", "not_available")
            if status == "not_available" and not include_not_available:
                continue
            if status == "partial" and not include_partial:
                continue
            if status == "error_unknown" and not include_error_unknown:
                continue
            if status == "fail" and not include_fail:
                continue

            time_ms = method_data.get("time_ms", [])
            if not time_ms or len(time_ms) != len(t_values):
                continue

            time_val = time_ms[max_t_idx]
            if time_val is None:
                continue

            time_sec = time_val / 1000
            all_case_method_times.append((method, case, time_sec))
            best_all_t_values.update(t_values)
            worst_all_t_values.update(t_values)

    all_case_method_times.sort(key=lambda x: x[2])

    best_case = all_case_method_times[0][1]
    worst_case = all_case_method_times[-1][1]

    for method in sorted(methods_in_data):
        if use_global_extrema:
            for method_, case, time in all_case_method_times:
                if method_ != method:
                    continue
                if case == best_case:
                    best_time = time
                if case == worst_case:
                    worst_time = time
        else:
            # get method specific best/worst case and time
            case_times_for_method = [(case, time) for method_, case, time in all_case_method_times if method_ == method]
            case_times_for_method.sort(key=lambda x: x[1])
            best_case, best_time = case_times_for_method[0]
            worst_case, worst_time = case_times_for_method[-1]

        best_data = None
        worst_data = None
        for case, result in {best_case: combined_results[best_case], worst_case: combined_results[worst_case]}.items():
            if method not in result["methods"]:
                print(f"{method} could not be found in {case} results.")
                continue

            method_data = result["methods"][method]
            status = method_data.get("status", "not_available")
            if status == "not_available" and not include_not_available:
                continue
            if status == "partial" and not include_partial:
                continue
            if status == "error_unknown" and not include_error_unknown:
                continue
            if status == "fail" and not include_fail:
                continue

            t_values = result["T_values"]
            time_ms = method_data.get("time_ms", [])

            if not time_ms or len(time_ms) != len(t_values):
                continue

            time_min = []
            valid = True
            for t in time_ms:
                if t is None or (isinstance(t, float) and t != t):
                    valid = False
                    break
                time_min.append(t / 1000)

            if not valid:
                continue

            if case == best_case:
                best_data = {
                    "case": case,
                    "t_values": t_values,
                    "time_min": time_min,
                }

            if case == worst_case:
                worst_data = {
                    "case": case,
                    "t_values": t_values,
                    "time_min": time_min,
                }

        if best_data is None and worst_data is None:
            continue

        color = color_map[method]

        if best_data is not None:
            plt.plot(
                best_data["t_values"],
                best_data["time_min"],
                marker="o",
                color=color,
                linestyle="--",
                label=f"{method} ({best_case[4:]})",
            )
            all_times.extend(best_data["time_min"])

        if worst_data is not None:
            plt.plot(
                worst_data["t_values"],
                worst_data["time_min"],
                marker="o",
                color=color,
                linestyle="-",
                label=f"{method} ({worst_case[4:]})",
            )
            all_times.extend(worst_data["time_min"])

    all_t_values = sorted(best_all_t_values | worst_all_t_values)
    plt.xscale("log")
    plt.xticks(all_t_values, [str(t) for t in all_t_values], rotation=45)

    if all_times:
        min_time = min(all_times)
        max_time = max(all_times)

        import math

        min_exp = math.floor(math.log10(min_time))
        max_exp = math.ceil(math.log10(max_time))

        y_ticks = []
        y_tick_labels = []

        for exp in range(int(min_exp), int(max_exp) + 1):
            base = 10**exp
            for mult in [2, 5]:
                val = base * mult
                if val >= min_time and val <= max_time:
                    y_ticks.append(val)
                    if val >= 1:
                        y_tick_labels.append(f"{val:.0f}s")
                    else:
                        y_tick_labels.append(f"{val * 1000:.0f}ms")

    plt.xlabel("timesteps")
    plt.ylabel("time (s)")
    plt.title("Execution time for different number of time steps (extrema)")
    plt.yscale("log")
    if all_times:
        plt.yticks(y_ticks, y_tick_labels)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.5)
    plt.savefig(
        output_path / "execution-time_timesteps.png",
        dpi=150,
    )
    plt.close()


def generate_time_vs_bus_graph(
    combined_results: list[BenchmarkResults],
    output_dir: str,
    include_partial=False,
    include_error_unknown=False,
    include_fail=False,
    include_not_available=False,
) -> None:
    """Generate time vs bus count graphs.

    Y-axis: time_ms converted to seconds (log scale)
    X-axis: bus count (from network)
    Two graphs: one for lowest T, one for highest T
    All methods on each graph, sorted by bus_count
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.rcParams["axes.titlesize"] = 32
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["xtick.labelsize"] = 24
    plt.rcParams["ytick.labelsize"] = 24
    plt.rcParams["legend.fontsize"] = 24

    methods_in_data: set[str] = set()
    for result in combined_results.values():
        methods_in_data.update(result["methods"].keys())

    color_map = {}
    colors = COLORS
    for i, method in enumerate(sorted(methods_in_data)):
        color_map[method] = colors[i % len(colors)]

    all_cases = set()
    for result in combined_results.values():
        all_cases.add(result["case"])

    bus_counts: dict[str, int] = {c: r["bus_count"] for c, r in combined_results.items()}

    all_t_values: list[int] = []
    for result in combined_results.values():
        all_t_values.extend(result["T_values"])
    all_t_values = sorted(set(all_t_values))

    lowest_t = all_t_values[0]
    highest_t = all_t_values[-1]

    for target_t, suffix in [(lowest_t, "lowest_T"), (highest_t, "highest_T")]:
        plt.figure(figsize=(18, 8))

        for method in sorted(methods_in_data):
            method_data_by_case: dict[str, float] = {}

            for result in combined_results.values():
                if method not in result["methods"]:
                    continue

                method_data = result["methods"][method]
                status = method_data.get("status", "not_available")
                if status == "not_available" and not include_not_available:
                    continue
                if status == "partial" and not include_partial:
                    continue
                if status == "error_unknown" and not include_error_unknown:
                    continue
                if status == "fail" and not include_fail:
                    continue

                case = result["case"]
                t_values = result["T_values"]
                time_ms = method_data.get("time_ms", [])

                if not time_ms or len(time_ms) != len(t_values):
                    continue

                if target_t in t_values:
                    t_idx = t_values.index(target_t)
                    time_val = time_ms[t_idx]
                    if time_val is None or (isinstance(time_val, float) and time_val != time_val):
                        continue
                    method_data_by_case[case] = time_val / 1000

            sorted_cases = sorted(method_data_by_case.keys(), key=lambda c: bus_counts.get(c, 0))

            bus_list = [bus_counts.get(c, 0) for c in sorted_cases]
            time_list = [method_data_by_case[c] for c in sorted_cases]

            color = color_map[method]
            plt.plot(
                bus_list,
                time_list,
                marker="o",
                color=color,
                label=f"{method}",
            )

        plt.xlabel("bus count (cases)")
        plt.ylabel("time (s)")
        plt.title(f"Time vs Bus Count for {target_t} timesteps")

        plt.xscale("log")
        plt.yscale("log")

        case_bus_list = sorted(bus_counts.keys(), key=lambda c: bus_counts.get(c, 0))
        case_bus_counts = sorted({bus_counts[c] for c in case_bus_list})
        purged_bus_counts = [case_bus_counts[0]]
        for c in case_bus_counts[1:]:  # remove any ticks that are closer than 5 together.
            if c - purged_bus_counts[-1] >= int(c * 0.25):
                purged_bus_counts.append(c)
        plt.xticks(purged_bus_counts, purged_bus_counts, rotation=45)

        all_times_bus = []
        for method in sorted(methods_in_data):
            for result in combined_results.values():
                if method not in result["methods"]:
                    continue
                method_data = result["methods"][method]
                status = method_data.get("status", "not_available")
                if status == "not_available" and not include_not_available:
                    continue
                if status == "partial" and not include_partial:
                    continue
                if status == "error_unknown" and not include_error_unknown:
                    continue
                if status == "fail" and not include_fail:
                    continue
                t_values = result["T_values"]
                time_ms = method_data.get("time_ms", [])
                if not time_ms or len(time_ms) != len(t_values):
                    continue
                if target_t in t_values:
                    t_idx = t_values.index(target_t)
                    time_val = time_ms[t_idx]
                    if time_val is not None and not (isinstance(time_val, float) and time_val != time_val):
                        all_times_bus.append(time_val / 1000)

        if all_times_bus:
            min_time = min(all_times_bus)
            max_time = max(all_times_bus)

            import math

            min_exp = math.floor(math.log10(min_time))
            max_exp = math.ceil(math.log10(max_time))

            y_ticks = []
            y_tick_labels = []

            for exp in range(int(min_exp), int(max_exp) + 1):
                base = 10**exp
                for mult in [2, 5]:
                    val = base * mult
                    if val >= min_time and val <= max_time:
                        y_ticks.append(val)
                        if val >= 1:
                            y_tick_labels.append(f"{val:.0f}s")
                        else:
                            y_tick_labels.append(f"{val * 1000:.0f}ms")

            plt.yticks(y_ticks, y_tick_labels)

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout(pad=0.5)
        plt.savefig(
            output_path / f"time_vs_bus_{suffix}.png",
            dpi=150,
        )
        plt.close()


def compile_results(
    output_dir,
    method_filter=None,
    generate_graphs=True,
    include_partial=False,
    include_error_unknown=False,
    include_fail=False,
    include_not_available=False,
    use_global_extrema=False,
):
    """Compile all results into a pandas DataFrame.

    Args:
        output_dir: Directory containing result JSON files
        method_filter: Filter by method name
        generate_graphs: Whether to generate graphs
        include_partial: Include partial status results in graph
        include_error_unknown: Include error_unknown status results in graph
        include_fail: Include fail status results in graph
        include_not_available: Include not_available status results in graph
        use_global_extrema: Use global extrema for timesteps graph (default: per-method)
    """
    json_files = find_result_files(output_dir)

    results: list[BenchmarkResults] = []

    for filepath in json_files:
        try:
            result: BenchmarkResults = load_result_file(filepath)
            if method_filter:
                methods = result.get("methods", {})
                if method_filter not in methods:
                    continue
            results.append(result)
        except Exception as e:
            print(f"Failed to load results from {filepath}: {e}")

    if not results:
        raise ValueError("No results found after filtering")

    combined: dict[str, BenchmarkResults] = combine_benchmark_results(results)

    print(f"\nLoaded and combined {len(results)} result files")
    print(f"Combined into {len(combined)} unique (case, T_values) groups")

    if generate_graphs:
        print("\nGenerating graphs...")
        generate_time_vs_timesteps_graph(
            combined,
            output_dir,
            include_partial=include_partial,
            include_error_unknown=include_error_unknown,
            include_fail=include_fail,
            include_not_available=include_not_available,
            use_global_extrema=use_global_extrema,
        )
        generate_time_vs_bus_graph(
            combined,
            output_dir,
            include_partial=include_partial,
            include_error_unknown=include_error_unknown,
            include_fail=include_fail,
            include_not_available=include_not_available,
        )
        print(f"Graphs saved to: {output_dir}")

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Compile benchmark results from multiple job runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python compile_results.py --output-dir /mnt/home/user/p3s/results
    python compile_results.py --output-dir results
    python compile_results.py --method gpu
    python compile_results.py --include-partial --include-error-unknown
    python compile_results.py --use-global-extrema
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory containing result JSON files",
    )
    parser.add_argument(
        "--method",
        type=str,
        help="Filter by method (pp,grav,gpu,cpp-1thr,cpp-Nthr)",
    )
    parser.add_argument(
        "--no-graphs",
        action="store_true",
        help="Skip graph generation",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Include partial status results in graph",
    )
    parser.add_argument(
        "--include-error-unknown",
        action="store_true",
        help="Include error_unknown status results in graph",
    )
    parser.add_argument(
        "--include-fail",
        action="store_true",
        help="Include fail status results in graph",
    )
    parser.add_argument(
        "--include-not-available",
        action="store_true",
        help="Include not_available status results in graph",
    )
    parser.add_argument(
        "--use-global-extrema",
        action="store_true",
        help="Use global extrema for timesteps graph (default: per-method extrema)",
    )

    args = parser.parse_args()

    print(f"Loading results from: {args.output_dir}")

    try:
        compile_results(
            args.output_dir,
            method_filter=args.method,
            generate_graphs=not args.no_graphs,
            include_partial=args.include_partial,
            include_error_unknown=args.include_error_unknown,
            include_fail=args.include_fail,
            include_not_available=args.include_not_available,
            use_global_extrema=args.use_global_extrema,
        )
    except ValueError as e:
        print(f"Failed to compile the results: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
