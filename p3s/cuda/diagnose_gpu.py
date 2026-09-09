# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU environment diagnostic for the fully resident polar solver.

Run this FIRST on a new machine (esp. a cluster) when the GPU path segfaults. It walks the
whole stack in small steps and prints a PASS line after each, so the last line printed
before a crash tells you exactly which layer failed -- CUDA driver init, library load,
kernel compile (nvcc), the cuSolverRf host-LU symbols, a device solve, or the full Newton
batch. A bare segfault gives no Python traceback, so this stepwise probe is the way to
localize it.

    python -m p3s.cuda.diagnose_gpu

Each step flushes stdout immediately, so nothing is lost when the process dies.
"""

from __future__ import annotations

import ctypes
import os
import sys


def _ok(msg):
    print(f"  PASS: {msg}", flush=True)


def _step(msg):
    print(f"[step] {msg} ...", flush=True)


def main():
    print("=" * 70, flush=True)
    print("GPU diagnostic for p3s polar solver", flush=True)
    print("=" * 70, flush=True)
    print(f"python      : {sys.version.split()[0]}  ({sys.executable})", flush=True)
    print(f"platform    : {sys.platform}", flush=True)
    print(f"CUDA_HOME   : {os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')}", flush=True)
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')[:200]}", flush=True)
    print(f"nvcc on PATH: {_which('nvcc')}", flush=True)
    print("", flush=True)

    # 1. driver init (NO context yet) ----------------------------------------
    _step("import pycuda.driver + cuda.init()")
    import pycuda.driver as cuda

    cuda.init()
    ndev = cuda.Device.count()
    _ok(f"CUDA driver init, {ndev} device(s)")
    if ndev == 0:
        print(
            "  FATAL: no CUDA devices visible.\n"
            "  -> check CUDA_VISIBLE_DEVICES and that the job actually allocated a GPU "
            "(e.g. srun --gres=gpu:1). pycuda.autoinit would segfault here.",
            flush=True,
        )
        return
    for i in range(ndev):
        d = cuda.Device(i)
        cc = d.compute_capability()
        print(f"    dev {i}: {d.name()}  cc={cc[0]}.{cc[1]}  {d.total_memory() // (1024**2)} MB", flush=True)

    # 2. context via the shared primary-context setup -- EXACTLY what the solver does at
    # import (p3s.cuda._ctx uses pycuda.autoprimaryctx so pycuda + cuDSS share the
    # device primary context). If the cluster segfaults here it is a GPU-allocation problem
    # (no device assigned by the scheduler). Isolated as its own step.
    _step("import p3s.cuda._ctx (retains device primary context -- as the solver does)")
    from p3s.cuda import _ctx  # noqa: F401

    free, total = cuda.mem_get_info()
    _ok(f"context up; free {free // (1024**2)}/{total // (1024**2)} MB")

    # 3. cuSolver / cuSparse library load ------------------------------------
    _step("load libcusolver / libcusparse (CuSolverWrapper)")
    from p3s.cuda import CuSolverWrapper as W  # noqa: F401

    _ok("cuSolver + cuSparse loaded and symbols bound")

    # 3b. probe the version-sensitive host-LU symbols actually EXIST ----------
    _step("probe cusolverRf + host-LU symbols exist")
    need = [
        "cusolverSpCreate",
        "cusolverSpCreateCsrluInfoHost",
        "cusolverSpDcsrluExtractHost",
        "cusolverSpXcsrluNnzHost",
        "cusolverRfCreate",
        "cusolverRfBatchSetupHost",
        "cusolverRfBatchRefactor",
        "cusolverRfBatchSolve",
    ]
    missing = [s for s in need if not hasattr(W._libcusolver, s)]
    if missing:
        print(
            f"  FATAL: cuSolver lib is MISSING symbols {missing}.\n"
            "  -> the cluster's cuSolver dropped the low-level host-LU / cusolverRf batch "
            "API. This solver needs them. Load a cuSolver that still exports them "
            "(CUDA 11.x, or a 12.x that retained cusolverRf).",
            flush=True,
        )
        return
    _ok("all required cuSolver symbols present")

    # 4. kernel compile (nvcc) -----------------------------------------------
    _step("compile polar kernels via pycuda SourceModule (needs nvcc on PATH)")
    from pycuda.compiler import SourceModule

    src_path = os.path.join(os.path.dirname(__file__), "nr_polar_kernels.cu")
    mod = SourceModule(open(src_path).read(), no_extern_c=True)
    for k in ("eval_F_and_J_polar", "gather_J_polar", "negate_F", "update_voltage_polar", "inf_norm_per_column"):
        mod.get_function(k)
    _ok("kernels compiled and all functions found")

    # 5. cusolverSp device QR solve -- the cusolverRf-FREE fallback path ------
    # Run this BEFORE the cusolverRf probe: on CUDA 12.4 the cusolverRf step segfaults,
    # which would prevent us from ever learning whether the QR fallback works. QR
    # (cusolverSpDcsrlsvqr) is a single on-device sparse solve, no host-LU / no cusolverRf,
    # so it is robust across CUDA versions. If this PASSES and step 6 segfaults, use
    # backend="qr".
    _step("cusolverSp device QR solve on a 3x3 (cusolverRf-FREE fallback viability)")
    _tiny_qr_solve()
    _ok("cusolverSpDcsrlsvqr device solve OK -> fallback backend 'qr' usable")

    # 5c. cuDSS batched solve (the RECOMMENDED cusolverRf replacement) --------
    # If libcudss is installed this exercises the full analyze->factor->solve on a 3x3
    # batch via CudssBatch. Skipped (not failed) if libcudss is absent, so the rest of the
    # diagnostic still runs. A PASS here means backend="cudss" is usable.
    _step("cuDSS batched solve on a 3x3 (recommended backend; skipped if libcudss absent)")
    _tiny_cudss_solve()

    # 6. cuSolverRf symbolic setup in isolation ------------------------------
    # The step most likely to segfault on CUDA 12.x: low-level host-LU +
    # cusolverRfBatchSetupHost/Analyze. On a trivial 3x3 BEFORE the full solve, so a crash
    # here pins the blame on cusolverRf (vs. our kernels).
    _step("cuSolverRf symbolic setup on a 3x3 (isolates the cusolverRf host-LU path)")
    _tiny_rf_setup()
    _ok("cusolverRf batch setup + refactor + solve OK on 3x3")

    # 7. tiny end-to-end solve on case14 -------------------------------------
    _step("build + solve a tiny batch (case14 time-series) end-to-end")
    _tiny_solve()
    _ok("tiny end-to-end GPU solve converged")

    print("\nALL PASS -- the GPU polar path works in this environment.\n", flush=True)


def _tiny_cudss_solve():
    """Solve the 3x3 as a batch of 2 via CudssBatch (analyze->factor->solve). Reports
    SKIP if libcudss can't load, so it never aborts the diagnostic."""
    import numpy as np
    import pycuda.driver as cuda

    try:
        from p3s.cuda import CudssWrapper as C
        from p3s.cuda.cudss_batch import CudssBatch
    except OSError as e:
        print(f"  SKIP: libcudss not loadable ({e}). pip install nvidia-cudss-cu12 to use backend='cudss'.", flush=True)
        return
    ver = C.version()
    if ver:
        print(f"    cuDSS version {ver[0]}.{ver[1]}.{ver[2]}", flush=True)

    Jp = np.array([0, 2, 5, 7], dtype=np.int32)
    Jj = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int32)
    Jx = np.array([4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0], dtype=np.float64)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    n, B = 3, 2

    s = CudssBatch(Jp, Jj, batch_size=B)
    s.symbolic_setup(Jx)
    # fill per-system values + rhs on device, run refactor + solve
    cuda.memcpy_htod(s.d_A_batch, np.tile(Jx, B))
    cuda.memcpy_htod(s.d_rhs_batch, np.tile(b, B))
    s.reset_refactor_device()
    s.batch_solve_device()
    cuda.Context.synchronize()
    x = np.empty(B * n, np.float64)
    cuda.memcpy_dtoh(x, s.d_X_batch)
    A = np.array([[4, 1, 0], [1, 4, 1], [0, 1, 4]], float)
    ref = np.linalg.solve(A, b)
    err = max(np.linalg.norm(x[:n] - ref, np.inf), np.linalg.norm(x[n:] - ref, np.inf))
    s.free()
    if err < 1e-9:
        print(f"  PASS: cuDSS 3x3 batch solve OK (err {err:.1e}) -> backend 'cudss' usable", flush=True)
    else:
        print(f"  FAIL: cuDSS 3x3 wrong (err {err:.2e})", flush=True)


def _tiny_qr_solve():
    """Solve the same 3x3 via cusolverSpDcsrlsvqr (on-device sparse QR, NO cusolverRf).
    This is the version-robust fallback the resident solver can use when cusolverRf is
    broken (as on CUDA 12.4). Exercises the exact ctypes call the fallback would make."""
    import numpy as np
    import pycuda.driver as cuda

    from p3s.cuda.CuSolverWrapper import _libcusolver, _libcusparse

    Jp = np.array([0, 2, 5, 7], dtype=np.int32)
    Jj = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int32)
    Jx = np.array([4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0], dtype=np.float64)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    n, nnz = 3, 7

    spH = ctypes.c_void_p()
    _libcusolver.cusolverSpCreate(ctypes.byref(spH))
    descr = ctypes.c_void_p()
    _libcusparse.cusparseCreateMatDescr(ctypes.byref(descr))
    _libcusparse.cusparseSetMatType(descr, 0)  # GENERAL
    _libcusparse.cusparseSetMatIndexBase(descr, 0)  # base-0

    d_Jp = cuda.mem_alloc(Jp.nbytes)
    cuda.memcpy_htod(d_Jp, Jp)
    d_Jj = cuda.mem_alloc(Jj.nbytes)
    cuda.memcpy_htod(d_Jj, Jj)
    d_Jx = cuda.mem_alloc(Jx.nbytes)
    cuda.memcpy_htod(d_Jx, Jx)
    d_b = cuda.mem_alloc(b.nbytes)
    cuda.memcpy_htod(d_b, b)
    d_x = cuda.mem_alloc(b.nbytes)
    singular = ctypes.c_int(-1)
    st = _libcusolver.cusolverSpDcsrlsvqr(
        spH,
        n,
        nnz,
        descr,
        ctypes.c_void_p(int(d_Jx)),
        ctypes.c_void_p(int(d_Jp)),
        ctypes.c_void_p(int(d_Jj)),
        ctypes.c_void_p(int(d_b)),
        ctypes.c_double(0.0),
        ctypes.c_int(0),
        ctypes.c_void_p(int(d_x)),
        ctypes.byref(singular),
    )
    cuda.Context.synchronize()
    if st != 0:
        raise RuntimeError(f"cusolverSpDcsrlsvqr status {st}")
    x = np.empty(3, np.float64)
    cuda.memcpy_dtoh(x, d_x)
    A = np.array([[4, 1, 0], [1, 4, 1], [0, 1, 4]], float)
    err = np.linalg.norm(x - np.linalg.solve(A, b), np.inf)
    assert err < 1e-9, f"QR 3x3 wrong: err {err:.2e}"
    _libcusolver.cusolverSpDestroy(spH)


def _tiny_rf_setup():
    """Drive CusolverRfBatch on a 3x3 SPD-ish system: reorder -> host LU -> extract ->
    batch setup/analyze -> reset/refactor -> solve. If the cluster's cusolverRf is broken,
    this segfaults HERE (before any custom kernel), pinning the blame on cusolverRf."""
    import numpy as np

    from p3s.cuda.cusolver_rf_batch import CusolverRfBatch

    # A = [[4,1,0],[1,4,1],[0,1,4]] in CSR
    Jp = np.array([0, 2, 5, 7], dtype=np.int32)
    Jj = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int32)
    Jx = np.array([4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0], dtype=np.float64)
    rhs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    solver = CusolverRfBatch(Jp, Jj, batch_size=1)
    solver.symbolic_setup(Jx)
    x = solver.solve(Jx.reshape(1, -1), rhs.reshape(1, -1))[0]
    # verify against a dense solve
    A = np.array([[4, 1, 0], [1, 4, 1], [0, 1, 4]], float)
    err = np.linalg.norm(x - np.linalg.solve(A, rhs), np.inf)
    assert err < 1e-9, f"cusolverRf 3x3 wrong: err {err:.2e}"
    solver.free()


def _tiny_solve():
    import numpy as np
    from pandapower.networks import case14

    from p3s.calculateTrafoTapTable import calculate_trafo_characteristic
    from p3s.cuda.nr_polar_solver import PolarNewtonSolverCUDA
    from p3s.NewtonPowerflow import NewtonPowerflow

    net = case14()
    calculate_trafo_characteristic(net, inplace=True)
    npf = NewtonPowerflow(net)
    yb = npf._YBus.tocsr()
    yb.sort_indices()
    Yp = yb.indptr.astype(np.int32)
    Yj = yb.indices.astype(np.int32)
    Yx = yb.data.astype(np.complex128)
    pv = np.asarray(npf.busses["pv"], np.int32)
    pq = np.asarray(npf.busses["pq"], np.int32)
    V0 = npf._initial_voltage.astype(np.complex128)
    Sbus = npf._sBus.astype(np.complex128)
    Sbus_mat = np.ascontiguousarray(np.repeat(Sbus[:, None], 4, axis=1))

    solver = PolarNewtonSolverCUDA(Yp, Yj, Yx, pv, pq)
    out = solver.solve_batch(Sbus_mat, V0, tol=1e-8)
    assert out["converged"].all(), f"did not converge: {out['converged']}"


def _which(exe):
    from shutil import which

    return which(exe) or "NOT FOUND"


if __name__ == "__main__":
    main()
