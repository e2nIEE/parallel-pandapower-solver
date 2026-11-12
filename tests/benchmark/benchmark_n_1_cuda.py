# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Top-level phase timing for the GPU contingency path (run on the cluster).

Decomposes the wall clock of the contingency pipeline into: build_net, case generation
(host), input assembly, V0/pin build, solver init, and the full solve call. Useful for
seeing where wall-clock goes at a given grid size / contingency count (e.g. when profiling
larger grids in future). Most of the non-solve time is serial Python/numpy host prep, part
of which the CPU path shares.

    python -m tests.benchmark.benchmark_n_1_cuda --limit 2000
    python -m tests.benchmark.benchmark_n_1_cuda --limit 2000 --gpu-backend cudss --gpu-max-chunk 256

Prints a phase breakdown (ms + % of total) and ms/contingency.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from p3s.contingency.case_generator import ContingencyBatch, ContingencyCaseGenerator
from p3s.timeseries import dc_initial_voltage
from tests.benchmark.benchmark_n_1_pegase import build_net


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--gpu-max-chunk", type=int, default=None)
    ap.add_argument("--gpu-backend", choices=("lean", "cudss", "rf", "qr"), default="lean")
    ap.add_argument("--init", choices=("dc", "flat"), default="flat")
    args = ap.parse_args()

    import pycuda.driver as cuda

    from p3s.cuda.nr_polar_solver import PolarNewtonSolverCUDA

    def sync_t():
        cuda.Context.synchronize()
        return time.perf_counter()

    T = {}
    t0 = time.perf_counter()

    net, _ = build_net(args.limit)
    t = time.perf_counter()
    T["build_net"] = t - t0
    t0 = t

    gen = ContingencyCaseGenerator(net, reslack_islands=False)
    batch: ContingencyBatch = gen.build()
    n = batch.n_bus
    L = len(batch.cases)
    t = time.perf_counter()
    T["case_generator.build"] = t - t0
    t0 = t

    solver = PolarNewtonSolverCUDA(
        batch.Yp,
        batch.Yj,
        np.ascontiguousarray(batch.Yx_base, dtype=np.complex128),
        np.ascontiguousarray(batch.pv, dtype=np.int32),
        np.ascontiguousarray(batch.pq, dtype=np.int32),
        backend=args.gpu_backend,
    )
    solver.max_chunk = args.gpu_max_chunk
    t = time.perf_counter()
    T["solver_init(kernels+topo)"] = t - t0
    t0 = t

    v_start = dc_initial_voltage(gen._npf) if args.init == "dc" else gen._npf._initial_voltage.copy()
    # The per-case Ybus goes to the GPU as the sparse outage stamps the generator already
    # holds (expanded on the device), so there is no dense (nnz, L) matrix to build here.
    Sbus = np.ascontiguousarray(gen._npf._sBus, dtype=np.complex128)

    V0 = np.empty((n, L), dtype=np.complex128)
    pin = np.zeros((n, L), dtype=np.uint8)
    for c, case in enumerate(batch.cases):
        v0 = v_start.copy()
        for bus, _kind in case.pinned_refs:
            v0[bus] = np.abs(v0[bus]) + 0.0j
            pin[bus, c] = 1
        unserved = ~case.served
        pin[unserved, c] = 1
        v0[unserved] = 1.0 + 0.0j
        V0[:, c] = v0
    t = time.perf_counter()
    T["build V0/pin (per-case loop)"] = t - t0
    t0 = t

    # the actual solve (host->device transfer + Newton loop on the GPU + stragglers on the CPU)
    _ = sync_t()
    tS = time.perf_counter()
    res = solver.solve_batch_contingency_stamp(batch._stamp, Sbus, V0, pin, max_iter=10)
    T["solve (transfer + GPU Newton loop + CPU stragglers)"] = sync_t() - tS

    total = sum(T.values())
    print(
        f"\npegase N-1: L={L} contingencies, n={n}, backend={args.gpu_backend}, "
        f"converged {int(res['converged'].sum())}/{L}, {solver.last_stats}"
    )
    print("=" * 66)
    for k, v in T.items():
        print(f"  {k:48s} {v * 1e3:9.1f} ms  {100 * v / total:5.1f}%")
    print("=" * 66)
    print(f"  {'TOTAL':48s} {total * 1e3:9.1f} ms  ({total / L * 1e3:.3f} ms/contingency)")


if __name__ == "__main__":
    main()
