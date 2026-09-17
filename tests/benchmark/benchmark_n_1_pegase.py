# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone N-1 benchmark: nr_klu (multithreaded) vs pandapower on case9241pegase.

Times a full single-element N-1 contingency analysis -- every line AND every transformer
taken out one at a time -- comparing:

  * p3s's batched C++/KLU solver (``solve_contingencies_cpp``, all cores), which
    shares one symbolic factorization across the whole batch, against
  * a pandapower per-contingency loop (a fresh ``runpp`` per outage), the reference an
    N-1 study would otherwise run.

This is a SCRIPT, not a pytest test (a full pegase N-1 takes minutes and would interfere
with the test pipeline). Run it directly:

    python -m tests.benchmark.benchmark_n_1_pegase                # all lines + trafos
    python -m tests.benchmark.benchmark_n_1_pegase --limit 200    # first 200 contingencies
    python -m tests.benchmark.benchmark_n_1_pegase --skip-pandapower  # p3s only
    python -m tests.benchmark.benchmark_n_1_pegase --threads 8 --validate 25

GPU backend (fully-resident polar cuSolverRf path):

    python -m tests.benchmark.benchmark_n_1_pegase --backend gpu --limit 2000
    python -m tests.benchmark.benchmark_n_1_pegase --backend gpu --gpu-max-chunk 512
    python -m tests.benchmark.benchmark_n_1_pegase --backend both --limit 2000  # CPU vs GPU

``--backend gpu`` needs pycuda + a CUDA GPU + nvcc on PATH (the polar kernels compile at
import). ``--backend both`` times CPU and GPU on the same net and reports the GPU/CPU
speedup and their max voltage disagreement on mutually-converged cases. On a small GPU
(e.g. 4 GB), cap the per-chunk batch with ``--gpu-max-chunk`` (the solver also chunks by
free memory automatically). ``--backend cpp`` (default) is the original nr_klu path.

Notes
-----
* ``init="flat"`` is used: p3s's simplified DC model does a flat start and converges
  pegase in ~6 iterations.
* The pandapower loop dominates wall-clock (~0.25 s/contingency). For the full ~16k
  contingencies that is ~1.1 hour, so use ``--limit`` for a quick estimate or
  ``--skip-pandapower`` to time only p3s.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from datetime import datetime
from typing import Literal, NotRequired, TypedDict

import numpy as np
from pandapower.networks import case9241pegase

from p3s.calculateTrafoTapTable import calculate_trafo_characteristic
from p3s.contingency.ground_truth import enumerate_contingencies, solve_contingency
from p3s.contingency.solver_cpp import solve_contingencies_cpp

type methods_type = Literal["cpp", "gpu"]


class MethodResults(TypedDict):
    status: Literal["success", "partial", "fail", "not_available", "error_unknown"]
    time_ms: list[float | int | None]
    results: NotRequired[list[list[list[complex | float | int]] | None]]
    ms_per_cont: list[float | int | None]
    errors: list[Exception]
    threads: int | None


class BenchmarkResults(TypedDict):
    case: str
    job_id: str
    timestamp: str
    bus_count: int
    T_values: list[int]
    methods: dict[methods_type, list[MethodResults]]


def build_net(limit: int | None = None):
    """case9241pegase with every line + transformer assigned its own outage_group.

    If ``limit`` is given, only the first ``limit`` branches (lines first, then trafos)
    become contingencies; the rest stay in service (no outage_group).
    """
    net = case9241pegase()
    calculate_trafo_characteristic(net, inplace=True)
    net.line["outage_group"] = None
    net.trafo["outage_group"] = None

    # (table, index) for every branch, lines first then trafos
    branches = [("line", i) for i in net.line.index] + [("trafo", i) for i in net.trafo.index]
    if limit is not None:
        branches = branches[:limit]
    for k, (tbl, idx) in enumerate(branches):
        net[tbl].loc[idx, "outage_group"] = f"{tbl[0].upper()}{k}"
    return net, len(branches)


def _solve_in_chunks(net, threads, chunk):
    """Run the contingency solve over the net's outage groups in chunks of ``chunk`` to
    bound peak memory. The full result table is (n_bus x L) complex128 -- for pegase over
    all ~16k branches that is ~2.4 GB, plus a similarly large per-case Ybus matrix.
    Chunking caps both to one chunk at a time; per-chunk we keep only the served counts +
    converged flags (not the voltages).

    A fresh ``solve_contingencies_cpp`` per chunk re-does the (cheap) symbolic
    factorization once per chunk -- negligible next to ``chunk`` Newton solves, so the
    timing stays representative as long as chunk is reasonably large (>= a few hundred).
    Returns (n_converged, n_total).
    """
    all_groups = enumerate_contingencies(net)
    n_conv = 0
    for start in range(0, len(all_groups), chunk):
        sub = set(all_groups[start : start + chunk])
        work = copy.deepcopy(net)
        # deactivate outage groups outside this chunk (they stay in service, untested)
        for tbl in ("line", "trafo"):
            if tbl in work and "outage_group" in work[tbl].columns:
                col = work[tbl]["outage_group"]
                work[tbl].loc[~col.isin(sub), "outage_group"] = None
        res = solve_contingencies_cpp(work, reslack_islands=False, n_threads=threads, init="flat")
        n_conv += int(res.converged.sum())
        del res  # free this chunk's table before the next
    return n_conv, len(all_groups)


def time_p3s(net, threads: int, chunk: int | None = None):
    """Warm up (build pattern / fill caches), then time the batched contingency solve.

    When ``chunk`` is given, solve in memory-bounded chunks and return only converged
    counts (no result table). Otherwise, solve the whole batch at once and return the
    full ``ContingencyResultTable`` (needed for the --validate spot-check).
    """
    if chunk is not None:
        # warm up on the first chunk only (cheap), then time the full chunked sweep
        _ = _solve_in_chunks(net, threads, chunk=min(chunk, 64))
        t0 = time.perf_counter()
        n_conv, n_total = _solve_in_chunks(net, threads, chunk)
        dt = time.perf_counter() - t0
        return None, dt, n_conv, n_total

    _ = solve_contingencies_cpp(net, reslack_islands=False, n_threads=threads, init="flat")
    t0 = time.perf_counter()
    res = solve_contingencies_cpp(net, reslack_islands=False, n_threads=threads, init="flat", max_iter=10)
    dt = time.perf_counter() - t0
    return res, dt, int(res.converged.sum()), len(res.groups)


def time_gpu(net, max_chunk: int | None = None, backend: str = "cudss"):
    """Warm up, then time the fully-resident polar GPU contingency solve.

    Unlike the CPU path, the GPU solver chunks internally (by free-memory budget, capped by
    ``max_chunk``), so we call ``solve_contingencies_cuda`` on the whole net directly and
    return the full ``ContingencyResultTable`` (for the --validate spot-check) + timing.
    Import is local so the CPU-only benchmark never requires pycuda/CUDA.

    ``backend``: "rf" = cusolverRf batched (fast, default); "qr" = cusolverSp per-system QR
    (robust fallback when cusolverRf segfaults, e.g. CUDA 12.x).
    """
    from p3s.contingency.solver_cuda import solve_contingencies_cuda

    # warm up: kernel compile + cuSolverRf symbolic setup + first factor happen once here,
    # so they don't pollute the timed run. Use a tiny subnet (first outage group) to keep
    # the warmup cheap while still exercising the full code path.
    warm = copy.deepcopy(net)
    all_groups = enumerate_contingencies(net)
    if all_groups:
        keep = {all_groups[0]}
        for tbl in ("line", "trafo"):
            if tbl in warm and "outage_group" in warm[tbl].columns:
                col = warm[tbl]["outage_group"]
                warm[tbl].loc[~col.isin(keep), "outage_group"] = None
        _ = solve_contingencies_cuda(warm, reslack_islands=False, init="flat", max_chunk=max_chunk, backend=backend)

    t0 = time.perf_counter()
    res = solve_contingencies_cuda(
        net, reslack_islands=False, init="flat", max_iter=10, max_chunk=max_chunk, backend=backend
    )
    dt = time.perf_counter() - t0
    return res, dt, int(res.converged.sum()), len(res.groups)


def time_pandapower(net, groups):
    """Time a pandapower per-contingency loop (one full runpp per outage group)."""
    n_conv = 0
    t0 = time.perf_counter()
    for g in groups:
        gt = solve_contingency(net, g, reslack_islands=False)
        n_conv += int(gt.converged)
    dt = time.perf_counter() - t0
    return dt, n_conv


def validate(net, res, groups, n_sample: int):
    """Spot-check p3s vs pandapower on a sample of contingencies; return max errors
    over served buses (these are pandapower full-topology solves)."""
    if n_sample <= 0:
        return None
    idx = np.linspace(0, len(groups) - 1, num=min(n_sample, len(groups)), dtype=int)
    vm_max = va_max = 0.0
    mask_mismatch = 0
    for c in idx:
        g = groups[c]
        gt = solve_contingency(net, g, reslack_islands=False)
        served = res.served[:, c]
        if not np.array_equal(served, gt.served):
            mask_mismatch += 1
            continue
        if served.any():
            vm_max = max(vm_max, float(np.nanmax(np.abs(res.vm[served, c] - gt.vm[served]))))
            va_max = max(va_max, float(np.nanmax(np.abs(res.va[served, c] - gt.va[served]))))
    return vm_max, va_max, mask_mismatch, len(idx)


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--limit", type=int, default=None, help="only benchmark the first N branches (default: all lines+trafos)"
    )
    ap.add_argument("--threads", type=int, default=0, help="nr_klu OpenMP threads (0=all cores, 1=serial; default 0)")
    ap.add_argument(
        "--num-threads",
        type=int,
        nargs="+",
        default=None,
        help="Number of threads to use for parallelization (only supported for method cpp)",
    )
    ap.add_argument("--skip-pandapower", action="store_true", help="time p3s only (skip the slow pandapower loop)")
    ap.add_argument(
        "--validate",
        type=int,
        default=10,
        help="spot-check this many contingencies vs pandapower (default 10; "
        "ignored with --chunk, which keeps no result table)",
    )
    ap.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="CPU only: solve in memory-bounded chunks of this many "
        "contingencies (recommended for the full pegase sweep: the full "
        "result table is ~2.4 GB). Disables --validate.",
    )
    ap.add_argument(
        "--backend",
        choices=("cpp", "gpu", "both"),
        default="both",
        help="which p3s solver to time: cpp=nr_klu (default), "
        "gpu=fully-resident polar cuSolverRf, both=run both and compare",
    )
    ap.add_argument(
        "--gpu-max-chunk",
        type=int,
        default=None,
        help="GPU: cap the per-chunk batch size (default: memory-budget only). "
        "On a small GPU (e.g. 4 GB A500) ~128 is the RF-solve sweet spot; "
        "leave unset on large GPUs (A100) so the memory budget decides.",
    )
    ap.add_argument(
        "--gpu-backend",
        choices=("rf", "qr", "cudss"),
        default="cudss",
        help="GPU linear-solve backend: cudss=NVIDIA cuDSS batched direct solver "
        "(DEFAULT; needs nvidia-cudss-cuXX); rf=cusolverRf batched (legacy, "
        "segfaults on CUDA 12.4); qr=cusolverSp per-system QR (robust but "
        "slow at scale). Run diagnose_gpu.py to see which work in your env.",
    )
    ap.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for results JSON",
    )
    ap.add_argument(
        "--job-id",
        "-j",
        type=str,
        default="no-job-id",
        help="Job ID for output file naming",
    )
    return ap.parse_args()


def main():
    args = _parse_args()
    print("Building case9241pegase with per-branch outage groups ...")
    net, n_branches = build_net(args.limit)
    groups = enumerate_contingencies(net)
    n_cont = len(groups)
    n_line = int((net.line["outage_group"].notna()).sum())
    n_trafo = int((net.trafo["outage_group"].notna()).sum())
    print(f"  {len(net.bus)} buses | {n_cont} contingencies ({n_line} lines + {n_trafo} transformers)")

    do_cpp = args.backend in ("cpp", "both")
    do_gpu = args.backend in ("gpu", "both")

    res_cpp = res_gpu = None
    t_cpp = t_gpu = None

    count = len(net.bus)
    if args.limit:
        count = int(args.limit)
    elif args.chunk:
        count = int(args.chunk)

    contingency_results: BenchmarkResults = BenchmarkResults(
        case=net.name,
        job_id=args.job_id,
        timestamp=datetime.now().isoformat(),
        bus_count=count,
        T_values=[],
        methods={},
    )

    if do_cpp:
        threads: list[int]
        if args.num_threads:
            threads = args.num_threads
        else:
            threads = [args.threads]

        results = []
        for thread in threads:
            chunk_note = f", chunk={args.chunk}" if args.chunk else ""
            print(f"\nRunning p3s CPU batch (nr_klu, threads={thread or 'all'}{chunk_note}) ...")
            res_cpp, t_cpp, n_conv_c, _ = time_p3s(net, thread, chunk=args.chunk)
            print(
                f"  nr_klu:    {t_cpp:.3f} s total | {t_cpp / n_cont * 1e3:.3f} "
                f"ms/contingency | converged {n_conv_c}/{n_cont}"
            )
            result: MethodResults = MethodResults(
                status="success",
                time_ms=[t_cpp * 1e3],
                results=[],
                ms_per_cont=[float(t_cpp / n_cont)],
                errors=[],
                threads=thread,
            )
            results.append(result)
        contingency_results["methods"]["cpp"] = results

    if do_gpu:
        mc_note = f", max_chunk={args.gpu_max_chunk}" if args.gpu_max_chunk else ""
        print(f"\nRunning p3s GPU batch (polar, backend={args.gpu_backend}{mc_note}) ...")
        res_gpu, t_gpu, n_conv_g, _ = time_gpu(net, max_chunk=args.gpu_max_chunk, backend=args.gpu_backend)
        print(
            f"  polar GPU: {t_gpu:.3f} s total | {t_gpu / n_cont * 1e3:.3f} "
            f"ms/contingency | converged {n_conv_g}/{n_cont}"
        )
        result: MethodResults = MethodResults(
            status="success",
            time_ms=[t_gpu * 1e3],
            results=[],
            ms_per_cont=[float(t_gpu / n_cont)],
            errors=[],
            threads=1,
        )
        contingency_results["methods"]["gpu"] = [result]

    # CPU vs GPU agreement (both have full result tables here)
    if do_cpp and do_gpu and res_cpp is not None and res_gpu is not None:
        served_match = np.array_equal(res_cpp.served, res_gpu.served)
        conv_match = np.array_equal(res_cpp.converged, res_gpu.converged)
        both_conv = res_cpp.converged & res_gpu.converged
        m = (~np.isnan(res_cpp.V)) & (~np.isnan(res_gpu.V)) & both_conv[None, :]
        dV = float(np.abs(res_cpp.V[m] - res_gpu.V[m]).max()) if m.any() else 0.0
        print(
            f"\n  CPU vs GPU: served-mask match={served_match} | converged match="
            f"{conv_match} | max |dV| on {int(both_conv.sum())} mutually-converged "
            f"cases = {dV:.2e}"
        )

    # spot-check the available result table(s) vs pandapower ground truth
    res_for_val = res_gpu if do_gpu else res_cpp
    if args.validate and res_for_val is not None:
        print(f"\nValidating {args.validate} sampled contingencies vs pandapower ...")
        v = validate(net, res_for_val, groups, args.validate)
        if v:
            vm_max, va_max, mism, n_chk = v
            print(
                f"  over {n_chk} sampled: max vm err {vm_max:.2e} pu | "
                f"max va err {va_max:.2e} deg | served-mask mismatches {mism}"
            )
    elif args.validate and res_for_val is None:
        print("\n(--validate skipped: --chunk keeps no result table to compare)")

    if args.output_dir:
        filepath = write_results(contingency_results, args.output_dir)
        print(f"Results written to: {filepath}")

    if args.skip_pandapower:
        print("\nSkipping pandapower loop (--skip-pandapower). Done.")
        return

    print(f"\nRunning pandapower per-contingency loop ({n_cont} runpp solves; this is the slow part) ...")
    t_pp, n_conv_pp = time_pandapower(net, groups)
    print(
        f"  pandapower: {t_pp:.3f} s total | {t_pp / n_cont * 1e3:.3f} ms/contingency | converged {n_conv_pp}/{n_cont}"
    )

    print("\n" + "=" * 60)
    print(f"  N-1 on case9241pegase: {n_cont} contingencies")
    if t_cpp is not None:
        print(
            f"  p3s CPU (nr_klu, {args.threads or 'all'} threads): {t_cpp:.2f} s  ({t_pp / t_cpp:.1f}x vs pandapower)"
        )
    if t_gpu is not None:
        print(f"  p3s GPU (polar cuSolverRf):            {t_gpu:.2f} s  ({t_pp / t_gpu:.1f}x vs pandapower)")
    print(f"  pandapower loop:                            {t_pp:.2f} s")
    print("=" * 60)


def write_results(results: BenchmarkResults, output_dir: str | os.PathLike) -> str:
    """Write results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{results['job_id']}_{results['case']}.json"
    filepath_ = os.path.join(output_dir, filename)

    with open(filepath_, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return filepath_


if __name__ == "__main__":
    main()
