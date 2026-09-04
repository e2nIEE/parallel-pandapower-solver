# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched sparse direct solver for many systems that share one sparsity pattern.

This wraps the NVIDIA cuSolverRf batched refactorization API. The intended use is a
time-series Newton-Raphson powerflow: every operating point (and every Newton
iteration) produces a Jacobian with an *identical* CSR sparsity pattern but different
numeric values. cuSolverRf lets us pay for the symbolic factorization (reorder + host
LU + L/U extraction) exactly once, then for each batch only

    reset values -> batch refactor -> batch solve

runs on the GPU for all systems at once.

The implementation mirrors NVIDIA's reference sample ``cuSolverRfBatch.cpp``.
All CSR inputs must be **base-0** and have **row-sorted column indices**, which
is what cuSolver's host LU expects.

Usage
-----
    solver = CusolverRfBatch(Jp, Jj, batch_size=B)
    solver.symbolic_setup(Jx0)            # Jx0: representative values (n_nnz,)
    dx = solver.solve(Jx_batch, rhs_batch)
        # Jx_batch:  (B, nnz)  float64   per-system CSR values (same Jp/Jj)
        # rhs_batch: (B, n)    float64
        # returns    (B, n)    float64

``solve`` internally does reset_values + refactor + batch-solve. ``reset_refactor`` and
``batch_solve`` are also exposed separately for callers that want to reuse a
factorization across multiple right-hand sides.
"""

import ctypes

import numpy as np
from numpy.typing import NDArray
import pycuda.driver as cuda

from p3s.cuda import _ctx  # noqa: F401  (CUDA primary context; shared with cuDSS/kernels)
from p3s.cuda.CuSolverWrapper import _libcusolver, _libcusparse

# Raw pycuda.driver allocations are used throughout (not pycuda.gpuarray): gpuarray
# triggers a just-in-time nvcc compile of helper kernels on first use, which needs the
# MSVC host compiler on PATH. This solver runs no custom kernels, so plain
# mem_alloc / memcpy keeps it free of any nvcc dependency.

_F64 = np.dtype(np.float64).itemsize  # 8
_I32 = np.dtype(np.int32).itemsize  # 4

# cusparse matrix-type / index-base enums
_CUSPARSE_MATRIX_TYPE_GENERAL = 0
_CUSPARSE_INDEX_BASE_ZERO = 0

# cusolverRf enums (see cusolverRf.h)
_CUSOLVERRF_MATRIX_FORMAT_CSR = 0
_CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L = 2
_CUSOLVERRF_RESET_VALUES_FAST_MODE_ON = 1
_CUSOLVERRF_FACTORIZATION_ALG0 = 0
_CUSOLVERRF_TRIANGULAR_SOLVE_ALG1 = 1


def _check(status, what):
    if status != 0:
        raise RuntimeError(f"{what} failed with cusolver status {status}")


def _as_c_int_p(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def _as_c_double_p(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _as_void_p(arr):
    return arr.ctypes.data_as(ctypes.c_void_p)


class CusolverRfBatch:
    """Batched LU refactor+solve for a fixed CSR pattern via cuSolverRf."""

    def __init__(
        self,
        Jp: NDArray,
        Jj: NDArray,
        batch_size: int,
        reorder: str = "symrcm",
        numeric_zero: float = 0.0,
        numeric_boost: float = 0.0,
    ):
        self.n = int(len(Jp) - 1)
        self.nnz = int(len(Jj))
        self.batch_size = int(batch_size)
        self.reorder = reorder
        # Numeric pivot boosting (cusolverRfSetNumericProperties). When a pivot magnitude
        # falls below ``numeric_zero`` it is replaced by ``numeric_boost``, keeping the
        # batch refactor non-singular. Needed for near-islanded N-1 cases where the
        # batched RF factor is more fragile than KLU's per-system factor and would
        # otherwise emit a NaN solution. Defaults 0/0 = OFF (bit-identical to before).
        self.numeric_zero = float(numeric_zero)
        self.numeric_boost = float(numeric_boost)

        # base-0 int32 CSR pattern, kept on host (host LU + setup need host arrays)
        self._Ap = np.ascontiguousarray(Jp, dtype=np.int32)
        self._Aj = np.ascontiguousarray(Jj, dtype=np.int32)

        # handles / descriptors created lazily in symbolic_setup
        self._spH: ctypes.c_void_p | None = None  # cusolverSp handle
        self._rfH: ctypes.c_void_p | None = None  # cusolverRf handle
        self._descr: ctypes.c_void_p | None = None  # cusparse matrix descriptor

        # device-side persistent buffers (allocated in symbolic_setup)
        self._d_Ap = None  # int32 (n+1)
        self._d_Aj = None  # int32 (nnz)
        self._d_P = None  # int32 (n)
        self._d_Q = None  # int32 (n)
        self._d_A_batch = None  # float64 (B*nnz)
        self._d_A_array = None  # device array of B pointers into d_A_batch
        self._d_X_batch = None  # float64 (B*n)
        self._d_X_array = None  # device array of B pointers into d_X_batch
        self._d_T = None  # float64 (2*B*n) cusolverRf scratch

        self._setup_done = False

    # ------------------------------------------------------------------ setup
    def symbolic_setup(self, Jx0: NDArray):
        """Reorder, host-LU factor a representative matrix, extract L/U, and assemble
        the cusolverRf batch handle. Allocates all persistent device buffers.

        ``Jx0`` are representative CSR values (same pattern) used only to compute the
        symbolic structure; their exact numeric content does not affect later batches
        beyond pivot selection, so any non-degenerate Jacobian is fine.
        """
        n, nnz = self.n, self.nnz
        Ax0 = np.ascontiguousarray(Jx0, dtype=np.float64)

        # -- handles + descriptor (step: create cusolverSp, descrA = GENERAL, base-0) --
        spH = ctypes.c_void_p()
        _check(_libcusolver.cusolverSpCreate(ctypes.byref(spH)), "cusolverSpCreate")
        self._spH = spH

        descr = ctypes.c_void_p()
        _check(_libcusparse.cusparseCreateMatDescr(ctypes.byref(descr)), "cusparseCreateMatDescr")
        _libcusparse.cusparseSetMatType(descr, _CUSPARSE_MATRIX_TYPE_GENERAL)
        _libcusparse.cusparseSetMatIndexBase(descr, _CUSPARSE_INDEX_BASE_ZERO)
        self._descr = descr

        # -- step 2: Qreorder = symrcm(A) / symamd(A) --
        Qreorder: NDArray = np.empty(n, dtype=np.int32)
        if self.reorder == "symrcm":
            fn = _libcusolver.cusolverSpXcsrsymrcmHost
        elif self.reorder == "symamd":
            fn = _libcusolver.cusolverSpXcsrsymamdHost
        else:
            raise ValueError(f"unknown reorder '{self.reorder}'")
        _check(
            fn(spH, n, nnz, descr, _as_c_int_p(self._Ap), _as_c_int_p(self._Aj), _as_c_int_p(Qreorder)),
            f"cusolverSpXcsr{self.reorder}Host",
        )

        # -- step 3: B = Q*A*Q^T (permute pattern, build map Bfrom A) --
        Bp = self._Ap.copy()
        Bj = self._Aj.copy()
        size_perm = ctypes.c_size_t(0)
        _check(
            _libcusolver.cusolverSpXcsrperm_bufferSizeHost(
                spH,
                n,
                n,
                nnz,
                descr,
                _as_c_int_p(Bp),
                _as_c_int_p(Bj),
                _as_c_int_p(Qreorder),
                _as_c_int_p(Qreorder),
                ctypes.byref(size_perm),
            ),
            "cusolverSpXcsrperm_bufferSizeHost",
        )
        perm_buf: NDArray = np.zeros(max(1, size_perm.value), dtype=np.uint8)
        mapBfromA: NDArray = np.arange(nnz, dtype=np.int32)
        _check(
            _libcusolver.cusolverSpXcsrpermHost(
                spH,
                n,
                n,
                nnz,
                descr,
                _as_c_int_p(Bp),
                _as_c_int_p(Bj),
                _as_c_int_p(Qreorder),
                _as_c_int_p(Qreorder),
                _as_c_int_p(mapBfromA),
                _as_void_p(perm_buf),
            ),
            "cusolverSpXcsrpermHost",
        )
        Bx = Ax0[mapBfromA].copy()

        # -- step 4: LU(B) with partial pivoting on host --
        info = ctypes.c_void_p()
        _check(_libcusolver.cusolverSpCreateCsrluInfoHost(ctypes.byref(info)), "cusolverSpCreateCsrluInfoHost")
        _check(
            _libcusolver.cusolverSpXcsrluAnalysisHost(spH, n, nnz, descr, _as_c_int_p(Bp), _as_c_int_p(Bj), info),
            "cusolverSpXcsrluAnalysisHost",
        )

        size_internal = ctypes.c_size_t(0)
        size_lu = ctypes.c_size_t(0)
        _check(
            _libcusolver.cusolverSpDcsrluBufferInfoHost(
                spH,
                n,
                nnz,
                descr,
                _as_c_double_p(Bx),
                _as_c_int_p(Bp),
                _as_c_int_p(Bj),
                info,
                ctypes.byref(size_internal),
                ctypes.byref(size_lu),
            ),
            "cusolverSpDcsrluBufferInfoHost",
        )
        lu_buf: NDArray = np.zeros(max(1, size_lu.value), dtype=np.uint8)

        _check(
            _libcusolver.cusolverSpDcsrluFactorHost(
                spH,
                n,
                nnz,
                descr,
                _as_c_double_p(Bx),
                _as_c_int_p(Bp),
                _as_c_int_p(Bj),
                info,
                ctypes.c_double(1.0),
                _as_void_p(lu_buf),
            ),
            "cusolverSpDcsrluFactorHost",
        )

        singularity = ctypes.c_int(-1)
        _check(
            _libcusolver.cusolverSpDcsrluZeroPivotHost(spH, info, ctypes.c_double(1e-14), ctypes.byref(singularity)),
            "cusolverSpDcsrluZeroPivotHost",
        )
        if singularity.value >= 0:
            raise RuntimeError(f"symbolic matrix is singular at pivot {singularity.value}")

        # -- step 5: extract P, Q, L, U --
        nnzL = ctypes.c_int(0)
        nnzU = ctypes.c_int(0)
        _check(
            _libcusolver.cusolverSpXcsrluNnzHost(spH, ctypes.byref(nnzL), ctypes.byref(nnzU), info),
            "cusolverSpXcsrluNnzHost",
        )
        nnzL_v, nnzU_v = nnzL.value, nnzU.value

        Plu: NDArray = np.empty(n, dtype=np.int32)
        Qlu: NDArray = np.empty(n, dtype=np.int32)
        Lp: NDArray = np.empty(n + 1, dtype=np.int32)
        Lj: NDArray = np.empty(nnzL_v, dtype=np.int32)
        Lx: NDArray = np.empty(nnzL_v, dtype=np.float64)
        Up: NDArray = np.empty(n + 1, dtype=np.int32)
        Uj: NDArray = np.empty(nnzU_v, dtype=np.int32)
        Ux: NDArray = np.empty(nnzU_v, dtype=np.float64)
        _check(
            _libcusolver.cusolverSpDcsrluExtractHost(
                spH,
                _as_c_int_p(Plu),
                _as_c_int_p(Qlu),
                descr,
                _as_c_double_p(Lx),
                _as_c_int_p(Lp),
                _as_c_int_p(Lj),
                descr,
                _as_c_double_p(Ux),
                _as_c_int_p(Up),
                _as_c_int_p(Uj),
                info,
                _as_void_p(lu_buf),
            ),
            "cusolverSpDcsrluExtractHost",
        )

        # -- step 6: P = Qreorder[Plu], Q = Qreorder[Qlu] --
        P = Qreorder[Plu].astype(np.int32)
        Q = Qreorder[Qlu].astype(np.int32)

        # -- step 7-8: create cusolverRf handle and set parameters --
        rfH = ctypes.c_void_p()
        _check(_libcusolver.cusolverRfCreate(ctypes.byref(rfH)), "cusolverRfCreate")
        self._rfH = rfH
        _check(
            _libcusolver.cusolverRfSetNumericProperties(
                rfH, ctypes.c_double(self.numeric_zero), ctypes.c_double(self.numeric_boost)
            ),
            "cusolverRfSetNumericProperties",
        )
        _check(
            _libcusolver.cusolverRfSetAlgs(rfH, _CUSOLVERRF_FACTORIZATION_ALG0, _CUSOLVERRF_TRIANGULAR_SOLVE_ALG1),
            "cusolverRfSetAlgs",
        )
        _check(
            _libcusolver.cusolverRfSetMatrixFormat(
                rfH, _CUSOLVERRF_MATRIX_FORMAT_CSR, _CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L
            ),
            "cusolverRfSetMatrixFormat",
        )
        _check(
            _libcusolver.cusolverRfSetResetValuesFastMode(rfH, _CUSOLVERRF_RESET_VALUES_FAST_MODE_ON),
            "cusolverRfSetResetValuesFastMode",
        )

        # -- step 9: cusolverRfBatchSetupHost (host arrays for A, L, U, P, Q) --
        B = self.batch_size
        # host A value array: B pointers into one contiguous host buffer
        h_A_batch = np.tile(Ax0, B).astype(np.float64)  # (B*nnz,)
        h_A_ptrs = (ctypes.c_void_p * B)()
        base = h_A_batch.ctypes.data
        for i in range(B):
            h_A_ptrs[i] = base + i * nnz * 8  # 8 bytes / float64
        _check(
            _libcusolver.cusolverRfBatchSetupHost(
                B,
                n,
                nnz,
                _as_c_int_p(self._Ap),
                _as_c_int_p(self._Aj),
                ctypes.cast(h_A_ptrs, ctypes.c_void_p),
                nnzL_v,
                _as_c_int_p(Lp),
                _as_c_int_p(Lj),
                _as_c_double_p(Lx),
                nnzU_v,
                _as_c_int_p(Up),
                _as_c_int_p(Uj),
                _as_c_double_p(Ux),
                _as_c_int_p(P),
                _as_c_int_p(Q),
                rfH,
            ),
            "cusolverRfBatchSetupHost",
        )

        # -- step 10: analyze to extract parallelism --
        _check(_libcusolver.cusolverRfBatchAnalyze(rfH), "cusolverRfBatchAnalyze")

        # -- persistent device buffers (raw allocations) --
        self._d_Ap = self._to_device(self._Ap)
        self._d_Aj = self._to_device(self._Aj)
        self._d_P = self._to_device(P)
        self._d_Q = self._to_device(Q)
        self._d_A_batch = cuda.mem_alloc(B * nnz * _F64)
        self._d_X_batch = cuda.mem_alloc(B * n * _F64)
        self._d_T = cuda.mem_alloc(2 * B * n * _F64)

        # device arrays of pointers into the contiguous batch buffers
        self._d_A_array = self._make_ptr_array(self._d_A_batch, nnz, B)
        self._d_X_array = self._make_ptr_array(self._d_X_batch, n, B)

        self._setup_done = True
        self.nnzL, self.nnzU = nnzL_v, nnzU_v
        return self

    @staticmethod
    def _to_device(arr: NDArray):
        """Allocate device memory and copy a contiguous host array onto it."""
        arr = np.ascontiguousarray(arr)
        d = cuda.mem_alloc(arr.nbytes)
        cuda.memcpy_htod(d, arr)
        return d

    @staticmethod
    def _make_ptr_array(d_batch, stride: int, B: int):
        """Build a device array of B pointers: ptr[i] = base + i*stride*8 bytes."""
        base = int(d_batch)
        host_ptrs = np.array([base + i * stride * _F64 for i in range(B)], dtype=np.uint64)
        d = cuda.mem_alloc(host_ptrs.nbytes)
        cuda.memcpy_htod(d, host_ptrs)
        return d

    # ----------------------------------------------------------------- solve
    def reset_refactor(self, Jx_batch: NDArray):
        """Upload new per-system values and run batch reset + refactor.

        ``Jx_batch`` is (B, nnz) float64 (or (nnz,) for a single broadcast system).
        """
        if not self._setup_done:
            raise RuntimeError("call symbolic_setup() first")
        B, nnz = self.batch_size, self.nnz
        vals = np.ascontiguousarray(np.asarray(Jx_batch, dtype=np.float64).reshape(B, nnz).reshape(-1))
        cuda.memcpy_htod(self._d_A_batch, vals)

        _check(
            _libcusolver.cusolverRfBatchResetValues(
                B,
                self.n,
                nnz,
                ctypes.c_void_p(int(self._d_Ap)),
                ctypes.c_void_p(int(self._d_Aj)),
                ctypes.c_void_p(int(self._d_A_array)),
                ctypes.c_void_p(int(self._d_P)),
                ctypes.c_void_p(int(self._d_Q)),
                self._rfH,
            ),
            "cusolverRfBatchResetValues",
        )
        _check(_libcusolver.cusolverRfBatchRefactor(self._rfH), "cusolverRfBatchRefactor")

    def batch_solve(self, rhs_batch: NDArray) -> NDArray:
        """Solve with the current factorization. ``rhs_batch`` is (B, n) float64.

        Returns (B, n) float64. The RHS is overwritten in-place on device by the
        solution (cusolverRfBatchSolve semantics).
        """
        B, n = self.batch_size, self.n
        rhs = np.ascontiguousarray(np.asarray(rhs_batch, dtype=np.float64).reshape(B, n).reshape(-1))
        cuda.memcpy_htod(self._d_X_batch, rhs)

        _check(
            _libcusolver.cusolverRfBatchSolve(
                self._rfH,
                ctypes.c_void_p(int(self._d_P)),
                ctypes.c_void_p(int(self._d_Q)),
                1,  # nrhs (only 1 supported)
                ctypes.c_void_p(int(self._d_T)),
                n,  # ldt
                ctypes.c_void_p(int(self._d_X_array)),
                n,
            ),  # ldxf
            "cusolverRfBatchSolve",
        )
        cuda.Context.synchronize()
        out: NDArray = np.empty(B * n, dtype=np.float64)
        cuda.memcpy_dtoh(out, self._d_X_batch)
        return out.reshape(B, n)

    def solve(self, Jx_batch: NDArray, rhs_batch: NDArray) -> NDArray:
        """Convenience: reset+refactor with Jx_batch, then solve rhs_batch."""
        self.reset_refactor(Jx_batch)
        return self.batch_solve(rhs_batch)

    # ----------------------------------------------------- device-resident API
    # The fully resident GPU Newton solver (nr_polar_solver.py) already has the batch
    # Jacobian values and RHS on the device (written by its assembly kernels straight into
    # THESE buffers), so it must NOT re-upload them. These variants operate purely on the
    # persistent device buffers: no htod/dtoh, no per-iteration host round-trip.
    #
    # The internal batch buffers are exposed so the kernels can target them directly:
    #   d_A_batch : float64 (B*nnz)  Jacobian values, system-major (system c at c*nnz)
    #   d_X_batch : float64 (B*n)    RHS on input, solution in place on output
    @property
    def d_A_batch(self):
        return self._d_A_batch

    @property
    def d_rhs_batch(self):
        # cusolverRf solves IN PLACE (rhs and solution share d_X_batch). Exposed so the
        # Newton loop can target d_rhs_batch uniformly across backends; here it aliases
        # d_X_batch (the cuDSS backend uses a distinct buffer -- out-of-place solve).
        return self._d_X_batch

    @property
    def d_X_batch(self):
        return self._d_X_batch

    def reset_refactor_device(self):
        """Batch reset + refactor using values ALREADY in ``d_A_batch`` (no upload)."""
        if not self._setup_done:
            raise RuntimeError("call symbolic_setup() first")
        _check(
            _libcusolver.cusolverRfBatchResetValues(
                self.batch_size,
                self.n,
                self.nnz,
                ctypes.c_void_p(int(self._d_Ap)),
                ctypes.c_void_p(int(self._d_Aj)),
                ctypes.c_void_p(int(self._d_A_array)),
                ctypes.c_void_p(int(self._d_P)),
                ctypes.c_void_p(int(self._d_Q)),
                self._rfH,
            ),
            "cusolverRfBatchResetValues",
        )
        _check(_libcusolver.cusolverRfBatchRefactor(self._rfH), "cusolverRfBatchRefactor")

    def batch_solve_device(self):
        """Solve in place with RHS ALREADY in ``d_X_batch``; solution left on device.

        No synchronize, no dtoh -- the caller's next kernel consumes ``d_X_batch``. This is
        the hot path of the resident Newton loop.
        """
        _check(
            _libcusolver.cusolverRfBatchSolve(
                self._rfH,
                ctypes.c_void_p(int(self._d_P)),
                ctypes.c_void_p(int(self._d_Q)),
                1,  # nrhs (only 1 supported)
                ctypes.c_void_p(int(self._d_T)),
                self.n,  # ldt
                ctypes.c_void_p(int(self._d_X_array)),
                self.n,
            ),  # ldxf
            "cusolverRfBatchSolve",
        )

    def free(self):
        """Deterministically release device buffers + destroy cusolver handles.

        Called between chunks of a chunked solve so device memory is reclaimed immediately
        rather than at GC time (nondeterministic GC would let successive chunks' buffers
        pile up and re-exhaust the memory that chunking exists to conserve). Idempotent.
        """
        for name in ("_d_Ap", "_d_Aj", "_d_P", "_d_Q", "_d_A_batch", "_d_A_array", "_d_X_batch", "_d_X_array", "_d_T"):
            buf = getattr(self, name, None)
            if buf is not None:
                try:
                    buf.free()
                except Exception:
                    pass
                setattr(self, name, None)
        try:
            if self._rfH is not None:
                _libcusolver.cusolverRfDestroy(self._rfH)
                self._rfH = None
            if self._spH is not None:
                _libcusolver.cusolverSpDestroy(self._spH)
                self._spH = None
        except Exception:
            pass
        self._setup_done = False

    def __del__(self):
        self.free()
