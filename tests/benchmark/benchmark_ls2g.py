# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""p3s nr_klu vs. lightsim2grid: which one solves AC power flow faster.

Both libraries wrap a KLU-backed sparse Newton-Raphson solver in C++, so this compares
solver implementations, not "KLU vs. some other algorithm" -- lightsim2grid is pinned to
its ``NR_KLU`` algorithm (its default, ``NR_SparseLU``, is ~15x slower on pegase; see
BENCHMARK.md) so neither side wins by using a better linear solver.

Two comparable regimes, mirroring ``benchmark_batched.py``'s cpp-1thr / cpp-Nthr split:

  * ``ls2g-1thr`` / ``grav-1thr``  -- single-threaded batch of T independent solves.
  * ``ls2g-Nthr`` / ``grav-Nthr``  -- same batch, all CPU cores.

"Independent" matters: p3s's ``calculate_timeseries_cpp`` DC-initialises every step
from scratch (so steps can run in any order / any thread) rather than warm-starting step
t from step t-1's converged voltage. lightsim2grid's sequential ``TimeSeriesCPP`` warm
starts and is therefore a different algorithm (usually fewer Newton iterations per step);
its ``InjectionSweepCPP`` solves independent steps from one shared ``Vinit`` like p3s
does, and is the one used here.

Run:
    python tests/benchmark/benchmark_lightsim2grid.py --case case9241pegase --T 512
    python tests/benchmark/benchmark_lightsim2grid.py --list-cases

Needs an interpreter where both packages import: as of 2026-09-26 the system Python 3.13
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Literal, NotRequired, TypedDict

import numpy as np
import pandapower.networks.power_system_test_cases as pstc
from pandapower import pandapowerNet

from p3s.calculateTrafoTapTable import calculate_trafo_characteristic
from p3s.NewtonPowerflowCpp import NewtonPowerflow as NewtonPowerflowCpp

try:
    from lightsim2grid._solver_type import AlgorithmType
    from lightsim2grid.injectionSweep import InjectionSweepCPP
    from lightsim2grid.network import init_from_pandapower

    _HAS_LS2G = True
except ImportError:
    _HAS_LS2G = False

warnings.filterwarnings("ignore", message="Matrix is exactly singular")

type methods_type = Literal["grav-1thr", "grav-Nthr", "ls2g-1thr", "ls2g-Nthr"]
ALL_METHODS: list[methods_type] = ["grav-1thr", "grav-Nthr", "ls2g-1thr", "ls2g-Nthr"]


class MethodResults(TypedDict):
    status: Literal["success", "partial", "fail", "not_available", "error_unknown"]
    time_ms: list[float | int | None]
    results: NotRequired[list[list[list[complex | float | int]] | None]]
    max_vm_error: list[float | int | None]
    errors: list[str]


class BenchmarkResults(TypedDict):
    case: str
    job_id: str
    timestamp: str
    bus_count: int
    T_values: list[int]
    methods: dict[methods_type, MethodResults]


CASE_NAME_TO_FUNC = {
    "case9": pstc.case9,
    "case14": pstc.case14,
    "case30": pstc.case30,
    "case33bw": pstc.case33bw,
    "case39": pstc.case39,
    "case57": pstc.case57,
    "case89pegase": pstc.case89pegase,
    "case118": pstc.case118,
    "case145": pstc.case145,
    "case300": pstc.case300,
    "case1354pegase": pstc.case1354pegase,
    "case1888rte": pstc.case1888rte,
    "case2848rte": pstc.case2848rte,
    "case2869pegase": pstc.case2869pegase,
    "case3120sp": pstc.case3120sp,
    "case6470rte": pstc.case6470rte,
    "case6515rte": pstc.case6515rte,
    "case9241pegase": pstc.case9241pegase,
}


def _make_timeseries(net: pandapowerNet, T: int, seed: int = 0):
    """Mild per-load p/q jitter -> (n_load, T) scale matrix, same recipe as
    benchmark_batched.py so the two benchmarks are directly comparable."""
    rng = np.random.default_rng(seed)
    n_load = len(net.load)
    scale = 1.0 + 0.05 * rng.standard_normal((n_load, T))
    return scale


def _vm_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.abs(a) - np.abs(b)).max())


def run_grav_benchmark(net: pandapowerNet, timesteps: list[int], n_threads: int) -> MethodResults:
    """p3s nr_klu batch (independent DC-initialised steps, OpenMP over n_threads)."""
    result: MethodResults = MethodResults(status="success", time_ms=[], results=[], max_vm_error=[], errors=[])

    cpp = NewtonPowerflowCpp(net)
    p0 = net.load.p_mw.to_numpy()[:, None]
    q0 = net.load.q_mvar.to_numpy()[:, None]

    # Warm up the KLU symbolic analyze so the timed call only pays for numeric solves.
    warm_ts = {("load", "p_mw"): p0[:, :2], ("load", "q_mvar"): q0[:, :2]}
    try:
        cpp.calculate_timeseries_cpp(net, warm_ts, n_threads=n_threads)
    except Exception:
        pass

    for T in timesteps:
        scale = _make_timeseries(net, T)
        ts = {("load", "p_mw"): p0 * scale, ("load", "q_mvar"): q0 * scale}
        try:
            t0 = time.perf_counter()
            res = cpp.calculate_timeseries_cpp(net, ts, n_threads=n_threads)
            elapsed = time.perf_counter() - t0
            result["time_ms"].append(elapsed * 1000)
            result["results"].append(res.tolist())
        except Exception as err:
            result["time_ms"].append(None)
            result["results"].append(None)
            result["errors"].append(repr(err))
            result["status"] = "partial"

    if all(t is None for t in result["time_ms"]):
        result["status"] = "fail"
    return result


def run_ls2g_benchmark(net: pandapowerNet, timesteps: list[int], n_threads: int) -> MethodResults:
    """lightsim2grid InjectionSweepCPP batch, pinned to NR_KLU, independent steps from a
    shared flat-start Vinit -- the same regime as run_grav_benchmark."""
    if not _HAS_LS2G:
        return MethodResults(
            status="not_available",
            time_ms=[None],
            results=[None],
            max_vm_error=[],
            errors=["lightsim2grid not installed"],
        )

    result: MethodResults = MethodResults(status="success", time_ms=[], results=[], max_vm_error=[], errors=[])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid = init_from_pandapower(net, pp_orig_file="pandapower_v3")
        grid.change_solver(AlgorithmType.NR_KLU)
    except Exception as err:
        return MethodResults(
            status="not_available",
            time_ms=[None] * len(timesteps),
            results=[None] * len(timesteps),
            max_vm_error=[],
            errors=[f"grid conversion failed: {err!r}"],
        )

    n_bus = grid.total_bus()
    V0 = np.ones(n_bus, dtype=complex)
    n_load = len(grid.get_loads())
    if n_load != len(net.load):
        result["errors"].append(f"load count mismatch after conversion: ls2g={n_load} pp={len(net.load)}")
    p0 = np.array([load.target_p_mw for load in grid.get_loads()])
    q0 = np.array([load.target_q_mvar for load in grid.get_loads()])
    gen_p0 = np.array([g.target_p_mw for g in grid.get_generators()])
    sgen_p0 = np.array([g.target_p_mw for g in grid.get_static_generators()])

    for T in timesteps:
        scale = _make_timeseries(net, T).T  # (T, n_load), matches p3s's seed/recipe
        load_p = p0[None, :] * scale
        load_q = q0[None, :] * scale
        gen_p = np.tile(gen_p0, (T, 1))
        sgen_p = np.tile(sgen_p0, (T, 1))
        try:
            sweep = InjectionSweepCPP(grid)
            sweep.change_algorithm(AlgorithmType.NR_KLU)
            sweep.nb_thread = n_threads if n_threads else os.cpu_count()
            t0 = time.perf_counter()
            status = sweep.compute_Vs(gen_p, sgen_p, load_p, load_q, V0, 30, 1e-8)
            elapsed = time.perf_counter() - t0
            if status != 1 or sweep.nb_converged() != T:
                result["errors"].append(f"T={T}: status={status} nb_converged={sweep.nb_converged()}/{T}")
                result["status"] = "partial"
            result["time_ms"].append(elapsed * 1000)
            Vs = sweep.get_voltages()  # (T, n_bus) complex, bus order == pp bus order
            result["results"].append(Vs.tolist())
        except Exception as err:
            result["time_ms"].append(None)
            result["results"].append(None)
            result["errors"].append(repr(err))
            result["status"] = "partial"

    if all(t is None for t in result["time_ms"]):
        result["status"] = "fail"
    return result


def _compute_vm_errors(baseline, method_results) -> list[float | None]:
    """Max |Vm| error per T against the grav-1thr baseline. p3s returns (n_bus, T)
    per step; lightsim2grid returns (T, n_bus) -- both compared as |V| over the same
    pandapower bus ordering (bus indices are 0..n-1 and contiguous for every case here)."""
    if not baseline or not method_results or len(baseline) != len(method_results):
        return []
    errors = []
    for base, other in zip(baseline, method_results, strict=False):
        if base is None or other is None:
            errors.append(None)
            continue
        try:
            base_arr = np.asarray(base)  # (n_bus, T)
            other_arr = np.asarray(other)  # (T, n_bus)
            if other_arr.shape == base_arr.T.shape:
                other_arr = other_arr.T
            errors.append(_vm_err(base_arr, other_arr))
        except Exception:
            errors.append(None)
    return errors


def benchmark(net: pandapowerNet, methods: list[methods_type], timesteps: list[int]) -> BenchmarkResults:
    net.trafo.shift_degree = 0.0
    calculate_trafo_characteristic(net, inplace=True)

    benchmark_res: BenchmarkResults = BenchmarkResults(
        case=net.name,
        job_id="",
        timestamp=datetime.now().isoformat(),
        bus_count=len(net.bus),
        T_values=timesteps,
        methods={},
    )

    method_functions = {
        "grav-1thr": lambda n, t: run_grav_benchmark(n, t, n_threads=1),
        "grav-Nthr": lambda n, t: run_grav_benchmark(n, t, n_threads=0),
        "ls2g-1thr": lambda n, t: run_ls2g_benchmark(n, t, n_threads=1),
        "ls2g-Nthr": lambda n, t: run_ls2g_benchmark(n, t, n_threads=0),
    }

    for m in methods:
        if m not in method_functions:
            raise ValueError(f"method {m} not found")
        benchmark_res["methods"][m] = method_functions[m](net, timesteps)

    return benchmark_res


def parse_args():
    parser = argparse.ArgumentParser(description="p3s nr_klu vs. lightsim2grid AC power-flow speed benchmark")
    parser.add_argument("--case", "-c", type=str, default="case9241pegase")
    parser.add_argument(
        "--methods",
        "-m",
        type=str,
        default="all",
        help="Comma-separated: grav-1thr, grav-Nthr, ls2g-1thr, ls2g-Nthr, or 'all'",
    )
    parser.add_argument("--T", "-t", type=int, nargs="+", default=None)
    parser.add_argument("--output-dir", "-o", type=str, default=None)
    parser.add_argument("--job-id", "-j", type=str, default=None)
    parser.add_argument("--drop-results", "-dr", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args()


def write_results(results: BenchmarkResults, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{results['job_id']}_{results['case']}.json"
    filepath_ = os.path.join(output_dir, filename)
    with open(filepath_, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return filepath_


if __name__ == "__main__":
    args = parse_args()

    if args.list_cases:
        for name in CASE_NAME_TO_FUNC:
            print(name)
        sys.exit(0)

    if not _HAS_LS2G:
        print("lightsim2grid is not importable in this interpreter -- `pip install lightsim2grid` into it first.")

    if args.case not in CASE_NAME_TO_FUNC:
        print(f"Unknown case '{args.case}'. Use --list-cases.")
        sys.exit(1)
    net = CASE_NAME_TO_FUNC[args.case]()

    t_list: list[int] = args.T or [64, 256, 1024, 2048]
    methods_to_run: list[methods_type] = (
        ALL_METHODS if args.methods.lower() == "all" else [m.strip() for m in args.methods.split(",")]
    )
    invalid = [m for m in methods_to_run if m not in ALL_METHODS]
    if invalid:
        print(f"Invalid methods: {invalid}. Valid: {ALL_METHODS}")
        sys.exit(2)

    results = benchmark(net, methods_to_run, t_list)
    results["job_id"] = args.job_id or "no_job_id"

    # Validate every method against grav-1thr (the reference: same DC-init-per-step
    # convention as pandapower/p3s's own scipy-loop baseline in benchmark_batched.py).
    if "grav-1thr" in results["methods"] and results["methods"]["grav-1thr"]["results"]:
        baseline = results["methods"]["grav-1thr"]["results"]
        for method_name, method_data in results["methods"].items():
            if method_name == "grav-1thr":
                method_data["max_vm_error"] = [0.0] * len(baseline)
                continue
            if not method_data.get("results"):
                continue
            method_data["max_vm_error"] = _compute_vm_errors(baseline, method_data["results"])
            max_err = [x for x in method_data["max_vm_error"] if x is not None]
            if max_err and max(max_err) > 1e-6:
                method_data["status"] = "fail"

    if args.drop_results:
        for k in results["methods"]:
            results["methods"][k].pop("results", None)

    if args.output_dir:
        filepath = write_results(results, args.output_dir)
        print(f"Results written to: {filepath}")

    print(f"\n=== Benchmark summary: {args.case} ({results['bus_count']} buses) ===")
    for method, data in results["methods"].items():
        print(f"\n{method}: {data['status']}")
        for i, T in enumerate(t_list):
            tm = data["time_ms"][i] if i < len(data["time_ms"]) else None
            per_step = f"{tm / T:.4f} ms/step" if tm is not None else "n/a"
            err = data.get("max_vm_error", [None] * len(t_list))
            err_i = err[i] if i < len(err) else None
            err_str = f"max_vm_err={err_i:.2e}" if err_i is not None else ""
            tm_str = f"{tm:.1f} ms" if tm is not None else "FAILED"
            print(f"  T={T:>6}: {tm_str:>12} ({per_step:>16})  {err_str}")
        if data.get("errors"):
            for e in data["errors"][:5]:
                print(f"    error: {e}")
