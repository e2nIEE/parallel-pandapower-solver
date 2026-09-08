# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched sparse direct solver via NVIDIA cuDSS.

Drop-in backend for the fully resident polar Newton solver, exposing the same device-buffer
interface as ``CusolverRfBatch`` / ``CusolverQRBatch`` so ``nr_polar_solver.py`` and the
assembly kernels are unchanged.

cuDSS is the modern, supported successor with the SAME amortization we need to analyze
(reorder and symbolic) ONCE, then rerun the factorization phase with values-only updates
every Newton iteration, and a batched solve over all systems that share the CSR pattern.

Interface (matches the other backends):
  d_A_batch   : float64 (B*nnz)  Jacobian values, system-major (system c at c*nnz)
  d_rhs_batch : float64 (B*n)    right-hand side (write target of the Newton loop)
  d_X_batch   : float64 (B*n)    solution (read by the Newton voltage update)
  symbolic_setup(Jx0)        -> cudssExecute(ANALYSIS) once
  reset_refactor_device()    -> cudssExecute(FACTORIZATION) on current d_A_batch
  batch_solve_device()       -> cudssExecute(SOLVE); solution in d_X_batch

ONE semantic difference from cusolverRf/QR: cuDSS solve is OUT-OF-PLACE (x != b), so this
backend keeps d_rhs_batch and d_X_batch as DISTINCT buffers. The rf/qr backends alias them
(in-place solve); nr_polar_solver handles both via the d_rhs_batch/d_X_batch split.

Uniform batch: all systems share Jp/Jj (same n, nnz, pattern), the time-series and N-1
batches. cuDSS batch matrices take arrays-of-device-pointers per system; since the pattern
is shared, every system's rowStart/colIndices pointer is the SAME (d_Jp / d_Jj) and only the
value pointer advances by nnz.
"""

import ctypes

import numpy as np
import pycuda.driver as cuda

from p3s.cuda import CudssWrapper as C
from p3s.cuda import _ctx  # noqa: F401  (CUDA primary context; required for cuDSS)

_F64 = 8
_I32 = 4
_PTR = 8  # device pointer size


def _to_device(arr):
    arr = np.ascontiguousarray(arr)
    d = cuda.mem_alloc(arr.nbytes)
    cuda.memcpy_htod(d, arr)
    return d


class CudssBatch:
    """Batched cuDSS solve for a fixed, shared CSR pattern."""

    def __init__(self, Jp, Jj, batch_size, reorder="symrcm", numeric_zero=0.0, numeric_boost=0.0, pivot="none"):
        # reorder / numeric_zero / numeric_boost accepted for signature parity with the
        # other backends. pivot: "auto" (cuDSS default) or "none" (disable pivot search --
        # much faster factor/refactor; safe for well-conditioned power-flow Jacobians, which
        # is our case, esp. with the numeric boost the contingency path already applies).
        self.pivot = pivot
        self.n = int(len(Jp) - 1)
        self.nnz = int(len(Jj))
        self.batch_size = int(batch_size)

        self._Ap = np.ascontiguousarray(Jp, dtype=np.int32)
        self._Aj = np.ascontiguousarray(Jj, dtype=np.int32)

        # --- handle / config / data ---
        self._h = ctypes.c_void_p()
        C.check(C._libcudss.cudssCreate(ctypes.byref(self._h)), "cudssCreate")
        self._cfg = ctypes.c_void_p()
        C.check(C._libcudss.cudssConfigCreate(ctypes.byref(self._cfg)), "cudssConfigCreate")
        self._data = ctypes.c_void_p()
        C.check(C._libcudss.cudssDataCreate(self._h, ctypes.byref(self._data)), "cudssDataCreate")

        # --- UNIFORM BATCH: all B systems share ONE sparsity pattern (n, nnz, Jp/Jj).
        # This is the critical config: with UBATCH_SIZE set, cuDSS reorders + symbolically
        # factors the pattern ONCE (not per-system) and runs the whole batch on the GPU.
        # WITHOUT it, cudssMatrixCreateBatchCsr treats the batch as B DISTINCT matrices and
        # reorders each on the HOST single-threaded -> CPU-bound, GPU idle (the 72 ms/cont
        # pathology). Uniform batch uses the SINGLE-matrix create API on contiguous
        # B*nnz / B*n buffers; the batch dimension is carried by UBATCH_SIZE, not the object.
        B, n, nnz = self.batch_size, self.n, self.nnz

        # shared pattern on device (one copy for the whole uniform batch)
        self._d_Jp = _to_device(self._Ap)  # (n+1) int32
        self._d_Jj = _to_device(self._Aj)  # (nnz) int32
        # contiguous per-system value / rhs / solution buffers (system c at c*nnz / c*n)
        self._d_A_batch = cuda.mem_alloc(B * nnz * _F64)
        self._d_rhs_batch = cuda.mem_alloc(B * n * _F64)
        self._d_X_batch = cuda.mem_alloc(B * n * _F64)

        # single-matrix create (uniform batch): pass the base device pointers of the
        # contiguous buffers; cuDSS strides by nnz / n internally using UBATCH_SIZE.
        self._A = ctypes.c_void_p()
        C.check(
            C._libcudss.cudssMatrixCreateCsr(
                ctypes.byref(self._A),
                ctypes.c_int64(n),
                ctypes.c_int64(n),
                ctypes.c_int64(nnz),
                ctypes.c_void_p(int(self._d_Jp)),
                ctypes.c_void_p(0),
                ctypes.c_void_p(int(self._d_Jj)),
                ctypes.c_void_p(int(self._d_A_batch)),
                C.CUDA_R_32I,
                C.CUDA_R_32I,
                C.CUDA_R_64F,  # offsetType, indexType, valueType
                C.CUDSS_MTYPE_GENERAL,
                C.CUDSS_MVIEW_FULL,
                C.CUDSS_BASE_ZERO,
            ),
            "cudssMatrixCreateCsr",
        )
        self._b = ctypes.c_void_p()
        C.check(
            C._libcudss.cudssMatrixCreateDn(
                ctypes.byref(self._b),
                ctypes.c_int64(n),
                ctypes.c_int64(1),
                ctypes.c_int64(n),
                ctypes.c_void_p(int(self._d_rhs_batch)),
                C.CUDA_R_64F,
                C.CUDSS_LAYOUT_COL_MAJOR,
            ),
            "cudssMatrixCreateDn(b)",
        )
        self._x = ctypes.c_void_p()
        C.check(
            C._libcudss.cudssMatrixCreateDn(
                ctypes.byref(self._x),
                ctypes.c_int64(n),
                ctypes.c_int64(1),
                ctypes.c_int64(n),
                ctypes.c_void_p(int(self._d_X_batch)),
                C.CUDA_R_64F,
                C.CUDSS_LAYOUT_COL_MAJOR,
            ),
            "cudssMatrixCreateDn(x)",
        )

        # Set UBATCH_SIZE AFTER the matrices exist (order matters -- per NVIDIA sample).
        # Value type is C `int` (int32), NOT int64; a wrong size makes cuDSS read B as 1 and
        # silently solve only system 0. sizeof(int)=4.
        ub = ctypes.c_int(B)
        C.check(
            C._libcudss.cudssConfigSet(
                self._cfg, C.CUDSS_CONFIG_UBATCH_SIZE, ctypes.byref(ub), ctypes.c_size_t(ctypes.sizeof(ub))
            ),
            "cudssConfigSet(UBATCH_SIZE)",
        )

        # Disable pivoting for well-conditioned Jacobians: pivot SEARCH is a big chunk of the
        # (re)factorization cost, and power-flow J needs none (profiled: refactor dominated
        # the GPU loop). PIVOT_NONE makes refactor much cheaper. "auto" keeps cuDSS default.
        if self.pivot == "none":
            pv = ctypes.c_int(C.CUDSS_PIVOT_NONE)
            C.check(
                C._libcudss.cudssConfigSet(
                    self._cfg, C.CUDSS_CONFIG_PIVOT_TYPE, ctypes.byref(pv), ctypes.c_size_t(ctypes.sizeof(pv))
                ),
                "cudssConfigSet(PIVOT_TYPE)",
            )

        self._setup_done = False
        self._factored_once = False

    # -------- interface --------
    @property
    def d_A_batch(self):
        return self._d_A_batch

    @property
    def d_rhs_batch(self):
        return self._d_rhs_batch

    @property
    def d_X_batch(self):
        return self._d_X_batch

    def _execute(self, phase, what):
        C.check(C._libcudss.cudssExecute(self._h, phase, self._cfg, self._data, self._A, self._x, self._b), what)

    def symbolic_setup(self, Jx0):
        """Upload representative values then run the ANALYSIS phase once (reorder +
        symbolic factorization; reused by every later factorization)."""
        Ax0 = np.ascontiguousarray(Jx0, dtype=np.float64)
        # broadcast the representative values to all systems so ANALYSIS sees valid data
        cuda.memcpy_htod(self._d_A_batch, np.tile(Ax0, self.batch_size))
        self._execute(C.CUDSS_PHASE_ANALYSIS, "cudssExecute(ANALYSIS)")
        self._setup_done = True
        return self

    def reset_refactor_device(self):
        """(Re)factorize on the current d_A_batch values, reusing the ONE analysis.

        First call: full FACTORIZATION (chooses pivots for the shared pattern). Subsequent
        calls: REFACTORIZATION -- reuses the pivoting/symbolic from the first factor and
        only redoes the numeric work for the new values. This is the amortization that
        makes the per-Newton-iteration re-solve cheap (the cusolverRf-refactor analog)."""
        if not self._setup_done:
            raise RuntimeError("call symbolic_setup() first")
        if not self._factored_once:
            self._execute(C.CUDSS_PHASE_FACTORIZATION, "cudssExecute(FACTORIZATION)")
            self._factored_once = True
        else:
            self._execute(C.CUDSS_PHASE_REFACTORIZATION, "cudssExecute(REFACTORIZATION)")

    def batch_solve_device(self):
        """Triangular solves; rhs in d_rhs_batch -> solution in d_X_batch (out-of-place)."""
        self._execute(C.CUDSS_PHASE_SOLVE, "cudssExecute(SOLVE)")

    def free(self):
        for m in ("_A", "_b", "_x"):
            h = getattr(self, m, None)
            if h is not None and h.value:
                try:
                    C._libcudss.cudssMatrixDestroy(h)
                except Exception:
                    pass
                setattr(self, m, ctypes.c_void_p())
        try:
            if getattr(self, "_data", None) is not None and self._data.value:
                C._libcudss.cudssDataDestroy(self._h, self._data)
                self._data = None
            if getattr(self, "_cfg", None) is not None and self._cfg.value:
                C._libcudss.cudssConfigDestroy(self._cfg)
                self._cfg = None
            if getattr(self, "_h", None) is not None and self._h.value:
                C._libcudss.cudssDestroy(self._h)
                self._h = None
        except Exception:
            pass
        for name in ("_d_Jp", "_d_Jj", "_d_A_batch", "_d_rhs_batch", "_d_X_batch"):
            buf = getattr(self, name, None)
            if buf is not None:
                try:
                    buf.free()
                except Exception:
                    pass
                setattr(self, name, None)
        self._setup_done = False

    def __del__(self):
        self.free()
