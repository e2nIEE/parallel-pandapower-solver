# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified cross-method batched power-flow benchmark.

Compares the throughput of every batched / time-series power-flow path in p3s on
shared networks, per operating point and end-to-end:

  * scipy loop         -- NewtonPowerflow.calculate x T               (CPU baseline)
  * C++ batch (1 thr)  -- NewtonPowerflowCpp.calculate_timeseries_cpp, n_threads=1
  * C++ batch (N thr)  -- ... n_threads=0 (all cores)
  * GPU batch          -- NewtonPowerflowCuda.calculate_timeseries_cuda  (if CUDA)

Newton-family methods (scipy / C++ / GPU) run on case9241pegase (PV+slack).

Run as a script:
    python tests/benchmark_batched.py --case case9241pegase --methods pp,p3s cpp --T 64 256 1024
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from functools import partial
from typing import TypedDict, Literal, NotRequired

import numpy as np
from pandapower import pandapowerNet

try:
    import pandapower.networks.power_system_test_cases as pstc

    _PANDAPOWER_CASES = True
except ImportError:
    pstc = None
    _PANDAPOWER_CASES = False

from pandapower.run import runpp
from p3s.calculateTrafoTapTable import calculateTrafoCharacteristic
from p3s.NewtonPowerflow import NewtonPowerflow

# The C++ batch needs the compiled nr_klu extension (Linux .so / Windows .pyd). If it
# is not built, the C++ columns are skipped rather than failing the whole benchmark.
try:
    from p3s.NewtonPowerflowCpp import NewtonPowerflow as NewtonPowerflowCpp

    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False

# The GPU path is opt-in: accessing a Windows GPU from WSL can segfault the whole
# process (not catchable), so it is OFF unless P3S_BENCH_CUDA=1 is set (use it on a
# native CUDA host where the cusolverRf batch was validated).
_HAS_CUDA = False
if os.environ.get("P3S_BENCH_CUDA") == "1":
    try:
        from p3s.cuda.NewtonPowerflowCuda import NewtonPowerflowCUDA

        _HAS_CUDA = True
    except ImportError:
        _HAS_CUDA = False

import warnings

warnings.filterwarnings('ignore', message="Matrix is exactly singular")

type methods_type = Literal["pp", "p3s", "cpp-1thr", "cpp-Nthr", "gpu"]
ALL_METHODS: list[methods_type] = ["pp", "p3s", "cpp-1thr", "cpp-Nthr", "gpu"]  # removed "sam" for now

class MethodResults(TypedDict):
    status: Literal["success", "partial", "fail", "not_available", "error_unknown"]
    time_ms: list[float | int | None]
    results: NotRequired[list[list[list[complex | float | int]] | None]]
    max_vm_error: list[float | int | None]
    errors: list[Exception]

class BenchmarkResults(TypedDict):
    case: str
    job_id: str
    timestamp: str
    bus_count: int
    T_values: list[int]
    methods: dict[methods_type, MethodResults]


def _make_timeseries(net, T, seed=0):
    """
    Mild per-load p/q jitter -> a timeseries dict + the per-load scale matrix.

    A global rescale pushes pegase out of convergence; small per-load jitter keeps every
    step convergent, which is the regime where a throughput comparison is meaningful.
    """
    rng = np.random.default_rng(seed)
    n_load = len(net.load)
    scale = 1.0 + 0.05 * rng.standard_normal((n_load, T))
    p0 = net.load.p_mw.to_numpy()[:, None]
    q0 = net.load.q_mvar.to_numpy()[:, None]
    ts = {("load", "p_mw"): p0 * scale, ("load", "q_mvar"): q0 * scale}
    return ts, scale


def _scipy_loop(net, scale):
    """
    Per-operating-point CPU Newton (baseline). Returns (n_bus, T) voltages.
    """
    base_p = net.load.p_mw.to_numpy().copy()
    base_q = net.load.q_mvar.to_numpy().copy()
    T = scale.shape[1]
    out = np.empty((len(net.bus), T), dtype=np.complex128)
    try:
        for t in range(T):
            net.load.p_mw = base_p * scale[:, t]
            net.load.q_mvar = base_q * scale[:, t]
            npf = NewtonPowerflow(net)
            npf.calculate(net)
            out[:, t] = (net.res_bus.vm_pu.values
                         * np.exp(1j * np.radians(net.res_bus.va_degree.values)))
    finally:
        # Restore the base load values. Downstream methods re-derive their time-series
        # from net.load via _make_timeseries, and build_sbus_matrix applies time-series
        # values as a *delta* against the net's base -- so leaving the last time step's
        # values here would silently make later methods solve a different operating
        # point than this baseline.
        net.load.p_mw = base_p
        net.load.q_mvar = base_q
    return out


def _pp_loop(net, scale):
    """Per-operating-point pandapower (baseline). Returns (n_bus, T) voltages."""
    base_p = net.load.p_mw.to_numpy().copy()
    base_q = net.load.q_mvar.to_numpy().copy()
    T = scale.shape[1]
    out = np.empty((len(net.bus), T), dtype=np.complex128)
    try:
        for t in range(T):
            net.load.p_mw = base_p * scale[:, t]
            net.load.q_mvar = base_q * scale[:, t]
            runpp(net)
            out[:, t] = (net.res_bus.vm_pu.values
                         * np.exp(1j * np.radians(net.res_bus.va_degree.values)))
    finally:
        # Restore the base load values -- see _scipy_loop for why this matters.
        net.load.p_mw = base_p
        net.load.q_mvar = base_q
    return out


def _vm_err(ab: tuple[list[complex | float | int], list[complex | float | int]]) -> float:
    return float(np.abs(np.abs(ab[0]) - np.abs(ab[1])).max())


def _compute_vm_errors(
    baseline: list[list[list[complex | float | int]] | None],
    method_results: list[list[list[complex | float | int]] | None]
) -> list[float | None]:
    """Compute max VM error between baseline and method results using _vm_err pattern."""
    if not baseline or not method_results or len(baseline) != len(method_results):
        return []

    errors = []
    a: list[list[complex | float | int]]
    b: list[list[complex | float | int]]
    for a, b in zip(baseline, method_results):
        if a is None or b is None:
            errors.append(None)
        else:
            try:
                errors.append(max(map(_vm_err, zip(a, b))))
            except IndexError as ie:
                print(f"IndexError occurred during error computation: {ie}")
                errors.append(None)
    return errors


CASE_NAME_TO_FUNC = {
    "case4gs": pstc.case4gs,
    "case5": pstc.case5,
    "case6ww": pstc.case6ww,
    "case9": pstc.case9,
    "case11_iwamoto": pstc.case11_iwamoto,
    "case14": pstc.case14,
    "case24_ieee_rts": pstc.case24_ieee_rts,
    "GBreducednetwork": pstc.GBreducednetwork,
    "case30": pstc.case30,
    "case_ieee30": pstc.case_ieee30,
    "case33bw": pstc.case33bw,
    "case39": pstc.case39,
    "case57": pstc.case57,
    "case89pegase": pstc.case89pegase,
    "case118": pstc.case118,
    "case145": pstc.case145,
    "iceland": pstc.iceland,
    "case_illinois200": pstc.case_illinois200,
    "case300": pstc.case300,
    "case1354pegase": pstc.case1354pegase,
    "case1888rte": pstc.case1888rte,
    "GBnetwork": pstc.GBnetwork,
    "case2848rte": pstc.case2848rte,
    "case2869pegase": pstc.case2869pegase,
    "case3120sp": pstc.case3120sp,
    "case6470rte": pstc.case6470rte,
    "case6515rte": pstc.case6515rte,
    "case9241pegase": pstc.case9241pegase,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified cross-method batched power-flow benchmark"
    )
    parser.add_argument(
        "--case", "-c",
        type=str,
        default="case9241pegase",
        help="Case name (e.g., case4gs, case9241pegase)",
    )
    parser.add_argument(
        "--methods", "-m",
        type=str,
        default="all",
        help="Comma-separated list of methods to run: pp, p3s, cpp, cpp-1thr, cpp-Nthr",
    )
    parser.add_argument(
        "--T", "-t",
        type=int,
        nargs="+",
        default=None,
        help="T values for timeseries (e.g., 64 256 1024)",
    )
    parser.add_argument(
        "--num-threads", "-n",
        type=int,
        nargs="+",
        default=None,
        help="Number of threads to use for parallelization (only supported for method cpp)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for results JSON",
    )
    parser.add_argument(
        "--job-id", "-j",
        type=str,
        default=None,
        help="Job ID for output file naming",
    )
    parser.add_argument(
        "--drop-results", "-dr",
        action="store_true",
        help="Drop results from JSON",
    )
    return parser.parse_args()


def get_methods_to_run(methods_str) -> list[methods_type]:
    if methods_str.lower() == "all":
        return ALL_METHODS

    selected = [m.strip() for m in methods_str.split(",")]
    if "cpp" in selected:
        selected.remove("cpp")
        selected.append("cpp-1thr")
        selected.append("cpp-Nthr")
    invalid = [m for m in selected if m not in ALL_METHODS]
    if invalid:
        raise ValueError(f"Invalid methods: {invalid}. Valid: {ALL_METHODS}")
    return selected


def select_case(case_name):
    if not _PANDAPOWER_CASES:
        raise ImportError("pandapower.networks.power_system_test_cases not available")
    if case_name not in CASE_NAME_TO_FUNC:
        available = ", ".join(CASE_NAME_TO_FUNC.keys())
        raise ValueError(f"Unknown case '{case_name}'. Available: {available}")
    return CASE_NAME_TO_FUNC[case_name]()


def run_pp_benchmark(net: pandapowerNet, timesteps: list[int]) -> MethodResults:
    """Run pandapower benchmark and return timing results per T."""
    result: MethodResults = MethodResults(
        status="success",
        time_ms=[],
        results=[],
        max_vm_error=[],
        errors=[],
    )

    for T in timesteps:
        try:
            ts, scale = _make_timeseries(net, T)
            t0 = time.perf_counter()
            res = _pp_loop(net, scale)
            elapsed = time.perf_counter() - t0
            result["time_ms"].append(elapsed * 1000)
            result["results"].append(res.tolist())
        except Exception as err:
            result["time_ms"].append(None)
            result["results"].append(None)
            result["status"] = "partial"
            result["errors"].append(err)

    if all(t is None for t in result["time_ms"]):
        result["status"] = "fail"
    return result


def run_p3s_benchmark(net: pandapowerNet, timesteps: list[int]) -> MethodResults:
    """Run p3s Newton benchmark and return timing results per T."""
    result: MethodResults = MethodResults(
        status="success",
        time_ms=[],
        results=[],
        max_vm_error=[],
        errors=[],
    )

    for T in timesteps:
        try:
            ts, scale = _make_timeseries(net, T)
            t0 = time.perf_counter()
            res = _scipy_loop(net, scale)
            elapsed = time.perf_counter() - t0
            result["time_ms"].append(elapsed * 1000)
            result["results"].append(res.tolist())
        except Exception as err:
            result["time_ms"].append(None)
            result["results"].append(None)
            result["errors"].append(err)
            result["status"] = "partial"

    if all(t is None for t in result["time_ms"]):
        result["status"] = "fail"
    return result


def run_cpp_benchmark(net: pandapowerNet, timesteps: list[int], n_threads: int) -> MethodResults:
    """Run C++ benchmark and return timing results per T."""
    if not _HAS_CPP:
        return MethodResults(
            status="not_available",
            time_ms=[None],
            results=[None],
            max_vm_error=[],
            errors=[NotImplementedError("nr_klu (C++ batch) not available")],
        )

    result: MethodResults = MethodResults(
        status="success",
        time_ms=[],
        results=[],
        max_vm_error=[],
        errors=[],
    )

    cpp = NewtonPowerflowCpp(net)

    for T in timesteps:
        ts, scale = _make_timeseries(net, T)
        try:
            t0 = time.perf_counter()
            res = cpp.calculate_timeseries_cpp(net, ts, n_threads=n_threads)
            elapsed = time.perf_counter() - t0
            result["time_ms"].append(elapsed * 1000)
            result["results"].append(res.tolist())
        except Exception as err:
            result["time_ms"].append(None)
            result["results"].append(None)
            result["errors"].append(err)
            result["status"] = "partial"

    if all(t is None for t in result["time_ms"]):
        result["status"] = "fail"
    return result


def run_gpu_benchmark(net: pandapowerNet, timesteps: list[int]) -> MethodResults:
    """Run GPU benchmark and return timing results per T."""
    if not _HAS_CUDA:
        return MethodResults(
            status="not_available",
            time_ms=[None],
            results=[None],
            max_vm_error=[],
            errors=[NotImplementedError("CUDA not available (set P3S_BENCH_CUDA=1)")],
        )

    result: MethodResults = MethodResults(
        status="success",
        time_ms=[],
        results=[],
        max_vm_error=[],
        errors=[],
    )

    gpu = NewtonPowerflowCUDA(net)

    for T in timesteps:
        ts, scale = _make_timeseries(net, T)
        try:
            t0 = time.perf_counter()
            res = gpu.calculate_timeseries_cudss(net, ts, tolerance=1e-6)
            elapsed = time.perf_counter() - t0
            result["time_ms"].append(elapsed * 1000)
            result["results"].append(res.tolist())
        except Exception as err:
            result["time_ms"].append(None)
            result["results"].append(None)
            result["errors"].append(err)

    if all(t is None for t in result["time_ms"]):
        result["status"] = "fail"
    return result


def benchmark_newton(
    net: pandapowerNet,
    methods: list[str],
    timesteps: list[int],
) -> BenchmarkResults:
    """Run Newton-family benchmarks and return structured results."""
    net.trafo.shift_degree = 0.0
    calculateTrafoCharacteristic(net, inplace=True)

    benchmark_res: BenchmarkResults = BenchmarkResults(
            case=net.name,
            job_id="",
            timestamp=datetime.now().isoformat(),
            bus_count=len(net.bus),
            T_values=timesteps,
            methods={}
    )

    method_functions = {
        "pp": run_pp_benchmark,
        "p3s": run_p3s_benchmark,
        "cpp-1thr": partial(run_cpp_benchmark, n_threads=1),
        "cpp-Nthr": partial(run_cpp_benchmark, n_threads=0),
        "gpu": run_gpu_benchmark,
    }

    for m in methods:
        if m not in method_functions:
            raise ValueError(f"method {m} not found")
        benchmark_res["methods"][m] = method_functions[m](net, timesteps)

    return benchmark_res


def benchmark_cpp_multicore(
    net: pandapowerNet,
    timesteps: list[int],
    n_threads: list[int]
) -> BenchmarkResults:
    """Run Newton-family benchmarks and return structured results."""
    net.trafo.shift_degree = 0.0
    calculateTrafoCharacteristic(net, inplace=True)

    benchmark_res: BenchmarkResults = BenchmarkResults(
            case=net.name,
            job_id="",
            timestamp=datetime.now().isoformat(),
            bus_count=len(net.bus),
            T_values=timesteps,
            methods={}
    )

    method_functions = {f"cpp-{n}thr": partial(run_cpp_benchmark, n_threads=n) for n in n_threads}

    for m, func in method_functions.items():
        benchmark_res["methods"][m] = func(net, timesteps)

    return benchmark_res


def write_results(results: BenchmarkResults, output_dir) -> str:
    """Write results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{results["job_id"]}_{results["case"]}.json"
    filepath_ = os.path.join(output_dir, filename)

    with open(filepath_, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return filepath_


if __name__ == "__main__":
    args = parse_args()

    t_list: list[int] = args.T or [64, 256, 1024, 2048, 4096, 8760, 17520, 35040]
    num_threads: list[int] | None = args.num_threads or None

    try:
        test_case = select_case(args.case)
    except (ImportError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    if num_threads is not None and args.methods == "all":
        methods_to_run = ["cpp"]
    else:
        try:
            methods_to_run = get_methods_to_run(args.methods)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(2)

    if num_threads is not None:
        if len(methods_to_run) != 1 or methods_to_run[0] != "cpp":
            print(f"n_threads argument not supported for methods other than cpp")
            sys.exit(3)

        newton_results = benchmark_cpp_multicore(test_case, t_list, num_threads)
    elif set(methods_to_run) & {"pp", "p3s", "cpp", "cpp-1thr", "cpp-Nthr", "gpu"}:
            newton_results = benchmark_newton(test_case, methods_to_run, t_list)
    else:
        print("could not run benchmark")
        sys.exit(4)

    newton_results["case"] = args.case
    if args.job_id:
        newton_results["job_id"] = args.job_id
    else:
        newton_results["job_id"] = "no_job_id"

    # calculate error and add to BenchmarkResults
    if "pp" in newton_results["methods"]:
        pp_res = newton_results["methods"]["pp"]["results"]
        for method_name, method_data in newton_results["methods"].items():
            if method_name == "pp":
                method_data["max_vm_error"] = [0.0] * len(pp_res)
            else:
                method_data["max_vm_error"] = _compute_vm_errors(pp_res, method_data["results"])
            # Set status to failure if any of the max_vm_error is greater than 10^^-6
            max_vm_err = [x for x in method_data["max_vm_error"] if x is not None]
            if max_vm_err and max(max_vm_err) > 1e-6:
                method_data["status"] = "fail"
            if None in method_data["max_vm_error"]:
                method_data["status"] = "error_unknown"

    if args.drop_results:
        for k in newton_results["methods"]:
            del newton_results["methods"][k]["results"]

    if args.output_dir:
        filepath = write_results(newton_results, args.output_dir)
        print(f"Results written to: {filepath}")

    print("\n=== Benchmark summary ===")
    if "methods" in newton_results:
        for method, data in newton_results["methods"].items():
            if isinstance(data, list):
                print(f"{method}: {len(data)} runs")
                for i, run in enumerate(data):
                    status = run.get("status", "unknown")
                    print(f"  T={t_list[i]}: {status}")
                    if run.get("errors"):
                        print(f"    Errors: {run['errors']}")
                    if run.get("max_vm_error") is not None:
                        print(f"    Max VM error: {run['max_vm_error']}")
            else:
                status = data.get("status", "unknown")
                print(f"{method}: {status}")
                if data.get("errors"):
                    print(f"  Errors: {data['errors']}")
                if data.get("max_vm_error") is not None:
                    print(f"  Max VM error: {data['max_vm_error']}")
