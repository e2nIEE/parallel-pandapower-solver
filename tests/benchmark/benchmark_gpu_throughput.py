# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Throughput of the GPU polar Newton solver in power flows per second.

Times ``PolarNewtonSolverCUDA.solve_batch`` on a time-series batch (one grid, T operating
points with +-5 % per-load jitter, each solved from the same start voltage to 1e-8) and
reports power flows per second for each batch size and linear-solve backend, next to the
all-core C++ solver (``nr_klu``) on the same batch. Only the solve is timed: building the
net, the injections and the solver (symbolic work, kernel compilation) is one-off setup.

    python -m tests.benchmark.benchmark_gpu_throughput                       # pegase 9241
    python -m tests.benchmark.benchmark_gpu_throughput --backends lean cudss
    python -m tests.benchmark.benchmark_gpu_throughput --case case118 --T 1024 16384
    python -m tests.benchmark.benchmark_gpu_throughput --profile             # phase split

``--profile`` synchronizes the device after every phase of the Newton loop and prints where
the time of the largest batch goes (assembly, linear solve, transfers); the forced
synchronization makes that run slightly slower than the timed ones.

Needs pycuda, a CUDA GPU and nvcc on PATH (the kernels compile on first use); the cuDSS
backend additionally needs ``nvidia-cudss-cuXX``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandapower.networks as pn

from p3s.calculateTrafoTapTable import calculate_trafo_characteristic
from p3s.NewtonPowerflowCpp import NewtonPowerflow
from p3s.timeseries import build_sbus_matrix, dc_initial_voltage


def build_problem(case: str, T: int, seed: int = 0):
    """(Yp, Yj, Yx, pv, pq, Sbus (n, T), V0 (n,)) of a pandapower case with jittered loads."""
    net = getattr(pn, case)()
    calculate_trafo_characteristic(net, inplace=True)
    pf = NewtonPowerflow(net)
    rng = np.random.default_rng(seed)
    scale = 1.0 + 0.05 * rng.standard_normal((len(net.load), T))
    ts = {
        ("load", "p_mw"): net.load.p_mw.to_numpy()[:, None] * scale,
        ("load", "q_mvar"): net.load.q_mvar.to_numpy()[:, None] * scale,
    }
    Y = pf._YBus.tocsr()
    Y.sort_indices()
    return (
        Y.indptr.astype(np.int32),
        Y.indices.astype(np.int32),
        Y.data.astype(np.complex128),
        np.asarray(pf.busses["pv"], dtype=np.int32),
        np.asarray(pf.busses["pq"], dtype=np.int32),
        np.ascontiguousarray(build_sbus_matrix(pf, net, ts), dtype=np.complex128),
        np.ascontiguousarray(dc_initial_voltage(pf), dtype=np.complex128),
    )


def _best_of(fn, repeat: int):
    best, res = np.inf, None
    for _ in range(repeat):
        t0 = time.perf_counter()
        res = fn()
        best = min(best, time.perf_counter() - t0)
    return best, res


def _profile(solver, S, V0, tol):
    """Per-phase wall time of one solve, with a device synchronization after every phase."""
    import pycuda.driver as cuda

    phases = ("_bind", "_eval", "_decide", "_step", "_linear_solve", "_solve_on_cpu")
    spent: dict[str, float] = {}
    calls: dict[str, int] = {}
    cls = type(solver)
    originals = {name: getattr(cls, name) for name in phases}

    def timed(name):
        orig = originals[name]

        def wrapper(self, *a, **k):
            cuda.Context.synchronize()
            t0 = time.perf_counter()
            out = orig(self, *a, **k)
            cuda.Context.synchronize()
            spent[name] = spent.get(name, 0.0) + time.perf_counter() - t0
            calls[name] = calls.get(name, 0) + 1
            return out

        return wrapper

    try:
        for name in phases:
            setattr(cls, name, timed(name))
        t0 = time.perf_counter()
        solver.solve_batch(S, V0, tol=tol)
        total = time.perf_counter() - t0
    finally:
        for name, orig in originals.items():
            setattr(cls, name, orig)
    label = {
        "_bind": "upload + state init",
        "_eval": "mismatch + Jacobian",
        "_decide": "convergence / line search",
        "_step": "voltage update",
        "_linear_solve": "factorize + solve",
        "_solve_on_cpu": "CPU safety net",
    }
    T = S.shape[1]
    print(f"    phase split at T={T} ({total * 1e3:.0f} ms, {solver.last_stats['chunks']} chunk(s)):")
    for name in phases:
        if name in spent:
            ms, share = spent[name] * 1e3, 100 * spent[name] / total
            per = spent[name] * 1e6 * solver.last_stats["chunks"] / T / calls[name]
            print(f"      {label[name]:26s} {ms:9.1f} ms {share:5.1f} %   {per:8.1f} us/system/call")
    rest = total - sum(spent.values())
    print(f"      {'download + host':26s} {rest * 1e3:9.1f} ms {100 * rest / total:5.1f} %")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="case9241pegase", help="pandapower.networks case name")
    ap.add_argument("--T", type=int, nargs="+", default=[256, 1024, 4096], help="batch sizes (operating points)")
    ap.add_argument("--backends", nargs="+", default=["lean"], choices=("lean", "cudss", "rf", "qr"))
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--repeat", type=int, default=3, help="timed runs per configuration (best is reported)")
    ap.add_argument("--max-chunk", type=int, default=None, help="cap the systems per GPU chunk")
    ap.add_argument("--precision", choices=("double", "mixed"), default="double", help="lean LU factor precision")
    ap.add_argument("--threads", type=int, default=0, help="nr_klu threads (0 = all cores)")
    ap.add_argument("--skip-cpu", action="store_true", help="do not time the C++ reference")
    ap.add_argument("--profile", action="store_true", help="print the phase split of the largest batch")
    args = ap.parse_args()

    import pycuda.driver as cuda

    from p3s.cuda.nr_polar_solver import PolarNewtonSolverCUDA

    Tmax = max(args.T)
    Yp, Yj, Yx, pv, pq, S, V0 = build_problem(args.case, Tmax)
    dev = cuda.Context.get_device()
    print(f"{args.case}: {len(V0)} buses, {len(Yj)} Ybus nonzeros | {dev.name()}, {dev.total_memory() >> 20} MB")

    cpu = None
    try:
        from p3s.cpp import nr_klu

        cpu = nr_klu.Solver(Yp, Yj, Yx, pv, pq)
    except ImportError:
        print("nr_klu not installed: no CPU reference, no result check")
    ref = {}
    if cpu is not None:

        def solve_cpu(Sc):
            return cpu.solve_batch(Sc, V0, tol=args.tol, n_threads=args.threads)

        for T in args.T:
            Sc = np.ascontiguousarray(S[:, :T])
            if args.skip_cpu:
                ref[T] = solve_cpu(Sc)
                continue
            dt, ref[T] = _best_of(lambda Sc=Sc: solve_cpu(Sc), args.repeat)
            it = ref[T]["iterations"]
            print(
                f"  cpu nr_klu  T={T:6d}: {dt * 1e3:10.1f} ms  {T / dt:10.1f} PF/s   iterations {it.min()}-{it.max()}"
            )

    for backend in args.backends:
        t0 = time.perf_counter()
        solver = PolarNewtonSolverCUDA(Yp, Yj, Yx, pv, pq, backend=backend)
        solver.max_chunk = args.max_chunk
        solver.precision = args.precision
        note = ""
        if backend == "lean":
            note = f", {solver.plan.n_pairs} multiply-subtracts in {solver.plan.n_levels} levels"
        print(f"  gpu {backend}: solver built in {time.perf_counter() - t0:.2f} s{note}")
        solver.solve_batch(np.ascontiguousarray(S[:, : min(args.T)]), V0, tol=args.tol)  # warm-up
        for T in args.T:
            Sc = np.ascontiguousarray(S[:, :T])
            dt, r = _best_of(lambda Sc=Sc, solver=solver: solver.solve_batch(Sc, V0, tol=args.tol), args.repeat)
            st = solver.last_stats
            it = r["iterations"]
            line = (
                f"  gpu {backend:5s}  T={T:6d}: {dt * 1e3:10.1f} ms  {T / dt:10.1f} PF/s"
                f"   iterations {it.min()}-{it.max()}"
                f"   converged {int(r['converged'].sum())}/{T}   chunk {st['chunk']}"
            )
            if st["handed_off"] or st["failed_on_gpu"]:
                line += f"   on CPU: {st['handed_off']} handed off, {st['failed_on_gpu']} rescued"
            if T in ref:
                both = r["converged"] & ref[T]["converged"]
                line += f"   max |dV| vs cpu {np.abs(r['V'][:, both] - ref[T]['V'][:, both]).max():.1e}"
            print(line)
        if args.profile:
            _profile(solver, np.ascontiguousarray(S[:, :Tmax]), V0, args.tol)
        solver.free()


if __name__ == "__main__":
    main()
