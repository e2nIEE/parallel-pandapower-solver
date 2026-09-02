# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""cusolverRf-FREE batched sparse solver via per-system on-device QR.

A drop-in alternative to :class:`CusolverRfBatch` for environments where cusolverRf's
low-level host-LU path is broken (notably CUDA 12.x, where cusolverRfBatchSetupHost /
the ``cusolverSpXcsrlu*Host`` routines segfault -- see ``p3s/cuda/diagnose_gpu.py``).

It exposes the SAME device-buffer interface the resident Newton loop uses:
  * ``d_A_batch`` : float64 (B*nnz)  Jacobian values, system-major (system c at c*nnz)
  * ``d_X_batch`` : float64 (B*n)    RHS on input, solution in place on output
  * ``symbolic_setup(Jx0)`` / ``reset_refactor_device()`` / ``batch_solve_device()``

...but under the hood ``batch_solve_device`` loops over the B systems and calls
``cusolverSpDcsrlsvqr`` (a single on-device sparse QR solve per system). Trade-off vs
cusolverRf: NO shared factorization reuse and NO batching -- each system re-factors from
scratch every Newton iteration, so it is slower per column. In exchange it is robust
across CUDA versions and needs none of the deprecated host-LU / cusolverRf API.

The CSR pattern (Jp/Jj) is shared and constant; only values (d_A_batch) change per call,
exactly like the resident loop already assumes.
"""
import ctypes

import numpy as np
import pycuda.driver as cuda

from p3s.cuda.CuSolverWrapper import _libcusolver, _libcusparse

_F64 = 8

_CUSPARSE_MATRIX_TYPE_GENERAL = 0
_CUSPARSE_INDEX_BASE_ZERO = 0


def _check(status, what):
    if status != 0:
        raise RuntimeError(f"{what} failed with cusolver status {status}")


class CusolverQRBatch:
    """Per-system on-device QR solver with the CusolverRfBatch device-buffer interface."""

    def __init__(self, Jp, Jj, batch_size, reorder="symrcm",
                 numeric_zero=0.0, numeric_boost=0.0, qr_reorder=3):
        # numeric_zero/boost/reorder accepted for signature-compatibility with
        # CusolverRfBatch (the resident solver passes them); QR ignores boost (it has no
        # persistent factor to protect) and uses its own `qr_reorder` fill-reducing scheme.
        self.n = int(len(Jp) - 1)
        self.nnz = int(len(Jj))
        self.batch_size = int(batch_size)
        # csrlsvqr reorder: 0=none,1=symrcm,2=symamd,3=csrmetis (3 is the usual default).
        self.qr_reorder = int(qr_reorder)

        self._Ap = np.ascontiguousarray(Jp, dtype=np.int32)
        self._Aj = np.ascontiguousarray(Jj, dtype=np.int32)

        self._spH = ctypes.c_void_p()
        _check(_libcusolver.cusolverSpCreate(ctypes.byref(self._spH)), "cusolverSpCreate")
        self._descr = ctypes.c_void_p()
        _check(_libcusparse.cusparseCreateMatDescr(ctypes.byref(self._descr)),
               "cusparseCreateMatDescr")
        _libcusparse.cusparseSetMatType(self._descr, _CUSPARSE_MATRIX_TYPE_GENERAL)
        _libcusparse.cusparseSetMatIndexBase(self._descr, _CUSPARSE_INDEX_BASE_ZERO)

        # persistent device buffers (same layout as CusolverRfBatch)
        self._d_Ap = self._to_device(self._Ap)
        self._d_Aj = self._to_device(self._Aj)
        self._d_A_batch = cuda.mem_alloc(self.batch_size * self.nnz * _F64)
        self._d_X_batch = cuda.mem_alloc(self.batch_size * self.n * _F64)
        # QR writes the solution to a separate output buffer (in==out is not allowed for
        # csrlsvqr); we ping-pong X (rhs) -> Xout (solution) then copy back.
        self._d_Xout = cuda.mem_alloc(self.batch_size * self.n * _F64)
        self._setup_done = False

    @staticmethod
    def _to_device(arr):
        arr = np.ascontiguousarray(arr)
        d = cuda.mem_alloc(arr.nbytes)
        cuda.memcpy_htod(d, arr)
        return d

    @property
    def d_A_batch(self):
        return self._d_A_batch

    @property
    def d_rhs_batch(self):
        # QR reads the rhs from d_X_batch and writes the solution back into it (see
        # batch_solve_device), so rhs and solution alias -- like cusolverRf, unlike cuDSS.
        return self._d_X_batch

    @property
    def d_X_batch(self):
        return self._d_X_batch

    def symbolic_setup(self, Jx0):
        """No-op for QR (no shared symbolic factorization). Present for interface parity;
        cusolverSpDcsrlsvqr does its own analyze+factor per solve."""
        self._setup_done = True
        return self

    def reset_refactor_device(self):
        """No-op: QR has no reusable factor. Values live in d_A_batch; the factorization
        happens inside batch_solve_device (per system, every call)."""
        if not self._setup_done:
            raise RuntimeError("call symbolic_setup() first")

    def batch_solve_device(self):
        """Solve each of the B systems in d_A_batch/d_X_batch via on-device sparse QR.
        Solution overwrites d_X_batch (to match CusolverRfBatch semantics)."""
        n, nnz, B = self.n, self.nnz, self.batch_size
        d_Ap = ctypes.c_void_p(int(self._d_Ap))
        d_Aj = ctypes.c_void_p(int(self._d_Aj))
        singular = ctypes.c_int(-1)
        for c in range(B):
            d_val = ctypes.c_void_p(int(self._d_A_batch) + c * nnz * _F64)
            d_b = ctypes.c_void_p(int(self._d_X_batch) + c * n * _F64)
            d_x = ctypes.c_void_p(int(self._d_Xout) + c * n * _F64)
            _check(_libcusolver.cusolverSpDcsrlsvqr(
                self._spH, n, nnz, self._descr,
                d_val, d_Ap, d_Aj, d_b,
                ctypes.c_double(0.0), ctypes.c_int(self.qr_reorder),
                d_x, ctypes.byref(singular)),
                "cusolverSpDcsrlsvqr")
        # copy solutions back into d_X_batch (device->device)
        cuda.memcpy_dtod(self._d_X_batch, self._d_Xout, B * n * _F64)

    def free(self):
        for name in ("_d_Ap", "_d_Aj", "_d_A_batch", "_d_X_batch", "_d_Xout"):
            buf = getattr(self, name, None)
            if buf is not None:
                try:
                    buf.free()
                except Exception:
                    pass
                setattr(self, name, None)
        try:
            if self._spH is not None:
                _libcusolver.cusolverSpDestroy(self._spH)
                self._spH = None
        except Exception:
            pass
        self._setup_done = False

    def __del__(self):
        self.free()
