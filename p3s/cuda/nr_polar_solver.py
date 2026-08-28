# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fully-resident batched polar Newton-Raphson on the GPU.

Design:
  * All Newton state (Vm/Va/Jx/F/rhs) lives on the device for the whole solve. The only
    per-iteration host interaction is copying a B-byte ``converged`` mask to decide the
    early-out -- NO per-iteration Jx/dx transfers (unlike NewtonPowerflowCuda.py).
  * The Jacobian is assembled on the GPU in POLAR form (constant Ym/Ya, one sincos per
    nonzero), reproducing p3s/cpp/nr_klu.cpp exactly.
  * The batched sparse solve reuses CusolverRfBatch (symbolic factorization once), fed
    directly from the resident device buffers.

Surface mirrors nr_klu.Solver: build once (symbolic analyze), then solve_batch(Sbus, V0).
This phase handles the time-series shape (Sbus varies per column, Ybus shared). Phase B
adds per-column Ybus + pin for N-1 contingencies.
"""
from __future__ import annotations

import os

import numpy as np
from p3s.cuda import _ctx  # noqa: F401  (retains the CUDA primary context; cuDSS-safe)
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from p3s.cuda.cusolver_rf_batch import CusolverRfBatch
from p3s.cuda.cusolver_qr_batch import CusolverQRBatch
from p3s.cuda.polar_topology import build_polar_topology, PolarTopology

# Linear-solve backends, keyed by name. cuDSS is imported lazily (its lib may be absent).
_BACKENDS = {"rf": CusolverRfBatch, "qr": CusolverQRBatch}

_F64 = 8
_I32 = 4
_U8 = 1

_KERNEL_SRC = os.path.join(os.path.dirname(__file__), "nr_polar_kernels.cu")


def _upload(arr) -> cuda.DeviceAllocation:
    arr = np.ascontiguousarray(arr)
    d = cuda.mem_alloc(arr.nbytes)
    cuda.memcpy_htod(d, arr)
    return d


class PolarNewtonSolverCUDA:
    """Device-resident batched polar Newton solver for one grid topology."""

    _mod = None  # compiled kernel module, shared across instances

    def __init__(self, Yp, Yj, Yx, pv, pq, reorder: str = "symrcm",
                 backend: str = "cudss"):
        self.topo: PolarTopology = build_polar_topology(Yp, Yj, Yx, pv, pq)
        self.reorder = reorder
        # Linear-solve backend for the batched Newton step:
        #   "cudss" = NVIDIA cuDSS batched direct solver -- the DEFAULT. Modern, supported,
        #             works on CUDA 12.x; analyze once + refactor/iter + batched GPU solve.
        #             Needs libcudss (pip nvidia-cudss-cuXX); imported lazily.
        #   "rf"    = cusolverRf batched refactor. LEGACY: deprecated API, segfaults on the
        #             CUDA 12.4 host-LU path (see diagnose_gpu.py). Kept for old CUDA 11 envs.
        #   "qr"    = per-system cusolverSpDcsrlsvqr (robust fallback, no factor reuse, slow
        #             at scale). Use when libcudss is unavailable and rf is broken.
        if backend not in ("rf", "qr", "cudss"):
            raise ValueError(f"backend must be 'rf', 'qr', or 'cudss', got {backend!r}")
        self.backend = backend
        # Numeric pivot boost for the cuSolverRf batch factor (see CusolverRfBatch). Off
        # for time-series; the contingency path sets it so near-islanded cases don't emit
        # NaN. Set per-solve in _run_newton_batch.
        self._numeric_zero = 0.0
        self._numeric_boost = 0.0
        # Optional hard cap on the per-chunk batch size, independent of the memory budget.
        # The batched cuSolverRf refactor+solve amortizes per-column cost only up to a
        # GPU-dependent sweet spot, then gets WORSE (on the RTX A500, ~B=128; profiled in
        # Phase C). Past that, more columns per chunk cost more per column, so cap here.
        # None = memory-budget only (right for big GPUs where the sweet spot is large).
        self.max_chunk = None
        # Correctness cap for the cuDSS uniform batch: cuDSS 0.8.0's UBATCH SOLVE
        # non-deterministically corrupts systems as the batch grows -- at B>=1024 it leaves
        # a random *few hundred* systems as NaN, and even at 512 a rare one slips through
        # across the many solves of a full Newton run (verified against scipy: with the
        # batch capped here the corruption disappears; the per-system KLU/scipy paths never
        # see it because they solve one system at a time). 256 sat cleanly through 30 trials
        # and the whole benchmark; it is well past the amortization sweet spot (~B=128 on the
        # RTX A500) so the throughput cost is negligible. Any residual non-converged column
        # is still retried serially by _retry_failed_columns below. Applied on top of
        # self.max_chunk and the memory budget; only for backend=="cudss". 0/None disables.
        self._cudss_safe_chunk = 256
        # Bounded number of retry passes over the columns cuDSS corrupted (see the retry loop
        # in _run_newton_batch). The corruption is non-deterministic and each pass re-solves
        # the survivors in safe-sized sub-batches, so a few passes clear it with overwhelming
        # probability; the loop also exits early as soon as a pass makes no progress (those
        # columns are then genuinely non-convergent, not corruption).
        self._cudss_max_retries = 4
        # Convergence-check cadence. The per-iteration device->host copy of the converged
        # mask is a blocking sync that serializes the async GPU pipeline. We check only every
        # `check_every` iters, and not before `min_iter` (power-flow Newton essentially never
        # converges in <3 iters, so earlier checks are pure stalls). Tune per workload.
        self.min_iter = 3
        self.check_every = 2
        # cuDSS pivot strategy: "none" disables the pivot search (profiled to dominate the
        # (re)factorization cost) -- safe for well-conditioned power-flow Jacobians. "auto"
        # restores cuDSS's default pivoting if a case ever needs it.
        self.pivot = "none"
        self._compile_kernels()

        t = self.topo
        # constant, topology-only device buffers (uploaded once)
        self.d_Yp = _upload(t.Yp)
        self.d_Yj = _upload(t.Yj)
        self.d_Ydiag = _upload(t.Ydiag)
        self.d_Ym = _upload(t.Ym)
        self.d_Ya = _upload(t.Ya)
        self.d_pvpq = _upload(t.pvpq)
        self.d_pq = _upload(t.pq)
        self.d_pvpq_pos = _upload(t.pvpq_pos)
        self.d_pq_pos = _upload(t.pq_pos)
        self.d_src_block = _upload(t.src_block)
        self.d_src_k = _upload(t.src_k)
        self.d_Jp = _upload(t.Jp)
        self.d_Jj = _upload(t.Jj)
        # J-row index of each Jacobian nonzero (for the pin identity rows; Phase B)
        row_of_nz = np.empty(t.nnzJ, dtype=np.int32)
        for r in range(t.m):
            row_of_nz[t.Jp[r]:t.Jp[r + 1]] = r
        self.d_row_of_nz = _upload(row_of_nz)

    @staticmethod
    def _resolve_backend(name):
        """Return the backend class for ``name``. cuDSS is imported lazily so that
        environments without libcudss can still use the rf/qr backends."""
        if name in _BACKENDS:
            return _BACKENDS[name]
        if name == "cudss":
            from p3s.cuda.cudss_batch import CudssBatch
            return CudssBatch
        raise ValueError(f"unknown backend {name!r}")

    @classmethod
    def _compile_kernels(cls):
        if cls._mod is not None:
            return
        src = open(_KERNEL_SRC).read()
        cls._mod = SourceModule(src, no_extern_c=True)
        cls._k_eval = cls._mod.get_function("eval_F_and_J_polar")
        cls._k_gather = cls._mod.get_function("gather_J_polar")
        cls._k_negate = cls._mod.get_function("negate_F")
        cls._k_update = cls._mod.get_function("update_voltage_polar")
        cls._k_infnorm = cls._mod.get_function("inf_norm_per_column")
        cls._k_yx_polar = cls._mod.get_function("yx_to_polar")

    # ------------------------------------------------------------------ solve
    def solve_batch(self, Sbus_mat, V0, max_iter: int = 30, tol: float = 1e-8):
        """Solve a batch of operating points sharing this topology (time-series).

        Sbus_mat : (n, T) complex  per-column injections
        V0       : (n,) or (n, T) complex  start voltage (broadcast if 1-D)
        Returns dict: V (n, T) complex, iterations (T,) int, converged (T,) bool.
        """
        t = self.topo
        n = t.n
        Sbus_mat = np.ascontiguousarray(Sbus_mat, dtype=np.complex128)
        if Sbus_mat.shape[0] != n:
            raise ValueError("Sbus rows must equal n_bus")
        T = Sbus_mat.shape[1]

        V0 = np.asarray(V0, dtype=np.complex128)
        if V0.ndim == 1:
            V0 = np.repeat(V0[:, None], T, axis=1)
        V0 = np.ascontiguousarray(V0)

        # time-series: shared Ybus (topo Ym/Ya), per-column Sbus, no pinning.
        Pspec = np.ascontiguousarray(Sbus_mat.real.T, dtype=np.float64)  # (T, n)
        Qspec = np.ascontiguousarray(Sbus_mat.imag.T, dtype=np.float64)
        return self._run_newton_batch(
            V0=V0, Pspec=Pspec, Qspec=Qspec,
            d_Ym=self.d_Ym, d_Ya=self.d_Ya, Ystride=0,
            pin=None, max_iter=max_iter, tol=tol)

    def solve_batch_contingency(self, Yx_mag, Yx_ang, Sbus, V0, pin,
                                max_iter: int = 30, tol: float = 1e-8,
                                numeric_zero: float = 1e-10,
                                numeric_boost: float = 1e-8):
        """Solve a batch of N-1 contingencies sharing this topology's pattern.

        Mirrors ``nr_klu.Solver.solve_batch_contingency``:
          Yx_mag, Yx_ang : (nnz, L) float64  per-case Ybus magnitude/angle (same Yp/Yj)
          Sbus           : (n,) complex128    injections (constant across cases)
          V0             : (n, L) complex128  per-case start (pinned buses pre-set)
          pin            : (n, L) uint8       per-case freeze mask (1 = pin)
        Returns dict: V (n, L) complex, iterations (L,) int, converged (L,) bool.
        """
        t = self.topo
        n, nnzY = t.n, t.nnzY
        Yx_mag = np.ascontiguousarray(Yx_mag, dtype=np.float64)
        Yx_ang = np.ascontiguousarray(Yx_ang, dtype=np.float64)
        if Yx_mag.shape[0] != nnzY or Yx_ang.shape[0] != nnzY:
            raise ValueError("Yx_mag/ang rows must equal Ybus nnz")
        L = Yx_mag.shape[1]
        if Yx_ang.shape[1] != L:
            raise ValueError("Yx_mag/ang shape mismatch")

        V0 = np.ascontiguousarray(np.asarray(V0, dtype=np.complex128))
        if V0.shape != (n, L):
            raise ValueError("V0 must be (n, L)")
        pin = np.ascontiguousarray(np.asarray(pin, dtype=np.uint8))
        if pin.shape != (n, L):
            raise ValueError("pin must be (n, L)")

        # per-case Ybus values -> system-major (L, nnzY). Sbus constant across cases -> a
        # SINGLE (n,) vector broadcast on device (Pshared=True), not L materialized copies.
        d_Ym = _upload(np.ascontiguousarray(Yx_mag.T))   # (L, nnzY)
        d_Ya = _upload(np.ascontiguousarray(Yx_ang.T))
        Sbus = np.asarray(Sbus, dtype=np.complex128).ravel()
        Pspec = np.ascontiguousarray(Sbus.real, dtype=np.float64)  # (n,)
        Qspec = np.ascontiguousarray(Sbus.imag, dtype=np.float64)
        d_pin = _upload(np.ascontiguousarray(pin.T))     # (L, n)

        # Numeric pivot boost keeps the batched RF factor non-singular on near-islanded
        # cases (KLU tolerates these per-system; the batch factor is more fragile).
        self._numeric_zero = numeric_zero
        self._numeric_boost = numeric_boost

        # Representative Ybus for the symbolic setup: the BASE intact grid (topo Ym/Ya),
        # NOT a case's outaged values -- an outaged/islanded case can be numerically
        # singular, which breaks the host LU pivot selection (the symbolic factorization
        # only needs a well-conditioned representative of the shared pattern). Evaluated at
        # a flat start for the same reason (Ym_rep/Ya_rep=None => topo values, and we pass
        # a flat V for the representative eval below).
        return self._run_newton_batch(
            V0=V0, Pspec=Pspec, Qspec=Qspec,
            d_Ym=d_Ym, d_Ya=d_Ya, Ystride=nnzY,
            pin=d_pin, max_iter=max_iter, tol=tol,
            Ym_rep=None, Ya_rep=None, flat_rep=True, Pshared=True)

    def solve_batch_contingency_cx(self, Yx, Sbus, V0, pin,
                                   max_iter: int = 30, tol: float = 1e-8,
                                   numeric_zero: float = 1e-10,
                                   numeric_boost: float = 1e-8):
        """Like solve_batch_contingency but takes COMPLEX per-case Ybus values ``Yx`` (nnz, L)
        and does the magnitude/angle conversion + transpose on the GPU (``yx_to_polar``),
        avoiding the ~4 s of single-threaded host np.abs/np.angle/.T on pegase. Upload the
        complex values in natural (nnz, L) layout (split re/im), convert once on-device to
        the system-major (L, nnz) Ym/Ya the Newton kernels expect.
        """
        t = self.topo
        n, nnzY = t.n, t.nnzY
        Yx = np.ascontiguousarray(Yx, dtype=np.complex128)
        if Yx.shape[0] != nnzY:
            raise ValueError("Yx rows must equal Ybus nnz")
        L = Yx.shape[1]
        V0 = np.ascontiguousarray(np.asarray(V0, dtype=np.complex128))
        pin = np.ascontiguousarray(np.asarray(pin, dtype=np.uint8))
        if V0.shape != (n, L) or pin.shape != (n, L):
            raise ValueError("V0/pin must be (n, L)")

        # upload complex Yx as split re/im in natural (nnz,L) layout (a plain view copy, no
        # transpose), then convert to system-major polar on the GPU.
        d_re = _upload(np.ascontiguousarray(Yx.real))
        d_im = _upload(np.ascontiguousarray(Yx.imag))
        d_Ym = cuda.mem_alloc(L * nnzY * _F64)
        d_Ya = cuda.mem_alloc(L * nnzY * _F64)
        bs = 256
        total = nnzY * L
        self._k_yx_polar(d_re, d_im, d_Ym, d_Ya, np.int32(nnzY), np.int32(L),
                         block=(bs, 1, 1), grid=((total + bs - 1) // bs, 1, 1))
        d_re.free(); d_im.free()

        Sbus = np.asarray(Sbus, dtype=np.complex128).ravel()
        Pspec = np.ascontiguousarray(Sbus.real, dtype=np.float64)
        Qspec = np.ascontiguousarray(Sbus.imag, dtype=np.float64)
        d_pin = _upload(np.ascontiguousarray(pin.T))
        self._numeric_zero = numeric_zero
        self._numeric_boost = numeric_boost
        out = self._run_newton_batch(
            V0=V0, Pspec=Pspec, Qspec=Qspec,
            d_Ym=np.uintp(int(d_Ym)), d_Ya=np.uintp(int(d_Ya)), Ystride=nnzY,
            pin=d_pin, max_iter=max_iter, tol=tol,
            Ym_rep=None, Ya_rep=None, flat_rep=True, Pshared=True)
        d_Ym.free(); d_Ya.free()
        return out

    # ------------------------------------------------------ chunk-size budgeting
    def _chunk_size(self, B, Ystride, has_pin, frac=0.4):
        """Largest per-chunk batch that fits in ``frac`` of free GPU memory.

        Per-column device bytes (f64=8): solver buffers Vm/Va/Pspec/Qspec (4n) + F (m) +
        dblocks (4*nnzY) [+ per-case Ym/Ya (2*nnzY) if Ystride] [+ pin (n) if has_pin]; plus
        cuSolverRf's d_A_batch (nnzJ) + d_X_batch (n) + d_T (2n). We also reserve a fixed
        headroom for the RF symbolic factors (L/U on device, batch-replicated) which the
        per-column model doesn't capture -- hence ``frac`` well under 1.

        Returns min(B, budget). One chunk when it fits (best: avoids re-paying host LU).
        """
        t = self.topo
        n, m, nnzY, nnzJ = t.n, t.m, t.nnzY, t.nnzJ
        per_col = (4 * n + m + 4 * nnzY) * _F64
        if Ystride:
            per_col += 2 * nnzY * _F64
        if has_pin:
            per_col += n * _U8
        per_col += (nnzJ + n + 2 * n) * _F64
        free, _ = cuda.mem_get_info()
        budget = max(1, int(free * frac) // per_col)
        cs = min(B, budget)
        if self.max_chunk is not None:
            cs = min(cs, self.max_chunk)
        # cuDSS uniform-batch correctness cap (see __init__): keep each batch in the region
        # where cuDSS 0.8.0's UBATCH solve is reliable.
        if self.backend == "cudss" and self._cudss_safe_chunk:
            cs = min(cs, self._cudss_safe_chunk)
        return cs

    # -------------------------------------------------------- chunked Newton
    def _run_newton_batch(self, V0, Pspec, Qspec, d_Ym, d_Ya, Ystride,
                          pin, max_iter, tol, Ym_rep=None, Ya_rep=None,
                          flat_rep=False, Pshared=False):
        """Device-resident batched Newton, CHUNKED to fit GPU memory. Shared by
        time-series and contingency.

        V0             : (n, B) complex128 start voltage
        Pspec, Qspec   : (B, n) float64 system-major injections
        d_Ym, d_Ya     : device Ybus polar values (shared or per-system per Ystride)
        Ystride        : 0 = shared Ybus; nnzY = per-system
        pin            : device (B, n) uint8 pin mask, or None
        Ym_rep, Ya_rep : (nnz,) representative Ybus for the symbolic setup (contingency)
        Pshared        : True => Pspec/Qspec are a single (1,n) vector shared by ALL systems
                         (N-1: Sbus constant); the kernel broadcasts it (Pstride=0) and we
                         upload it ONCE, not L copies. False => per-system (B,n) (time-series).

        A chunk is a contiguous column range solved as one cuSolverRf batch (its own RF
        factorization). Chunk size comes from the free-memory budget; one chunk when the
        whole batch fits (avoids re-paying the host LU per chunk). The shared topology
        device buffers (Yp/Yj/pvpq/... , uploaded once in __init__) are reused by every
        chunk; only the per-chunk column data + RF handle are (re)allocated.
        """
        t = self.topo
        n = t.n
        B = V0.shape[1]
        cs = self._chunk_size(B, Ystride, pin is not None)

        V_out = np.empty((n, B), dtype=np.complex128)
        iters_out = np.zeros(B, dtype=np.int32)
        conv_out = np.zeros(B, dtype=bool)

        for start in range(0, B, cs):
            stop = min(start + cs, B)
            # slice per-column Ybus device pointers for contingency (byte offset), or reuse
            # the shared pointer for time-series (Ystride==0).
            if Ystride:
                d_Ym_c = np.uintp(int(d_Ym) + start * Ystride * _F64)
                d_Ya_c = np.uintp(int(d_Ya) + start * Ystride * _F64)
            else:
                d_Ym_c, d_Ya_c = d_Ym, d_Ya
            d_pin_c = (int(pin) + start * n * _U8) if pin is not None else None

            # Pspec/Qspec: shared vector reused for every chunk (no slicing), else per-column.
            Pc = Pspec if Pshared else Pspec[start:stop]
            Qc = Qspec if Pshared else Qspec[start:stop]

            r = self._run_chunk(
                V0[:, start:stop],
                Pc, Qc, Pshared,
                d_Ym_c, d_Ya_c, Ystride, d_pin_c,
                max_iter, tol, Ym_rep, Ya_rep, flat_rep)
            V_out[:, start:stop] = r["V"]
            iters_out[start:stop] = r["iterations"]
            conv_out[start:stop] = r["converged"]

        # Safety net for the residual cuDSS UBATCH corruption (see _cudss_safe_chunk): a
        # corrupted column comes back non-converged (typically NaN) even though its operating
        # point is perfectly solvable, and it is NON-DETERMINISTIC, so a fresh attempt almost
        # always lands it. Re-solve the failed columns, and -- crucially -- re-solve them in
        # SAFE-SIZED sub-batches (cs), not one big gather: on a large GPU the failed set can
        # itself exceed the corruption threshold, so an uncapped single retry can re-corrupt
        # and leave residual failures (the "24 of 1024" symptom). Loop a bounded number of
        # passes because each pass is itself probabilistic. Only the time-series path is
        # handled (SHARED Ybus, Ystride==0, no pin -> columns can be gathered freely with the
        # same d_Ym/d_Ya); the contingency path has per-column Ybus + pin and does not exhibit
        # this failure, so it is left untouched.
        if self.backend == "cudss" and Ystride == 0 and pin is None:
            for _ in range(self._cudss_max_retries):
                bad = ~conv_out | ~np.isfinite(V_out).all(axis=0)
                idx = np.nonzero(bad)[0]
                if idx.size == 0:
                    break
                # gather failed columns; re-solve in safe-sized sub-batches
                Vbad = np.ascontiguousarray(V0[:, idx])
                Pbad = Pspec if Pshared else np.ascontiguousarray(Pspec[idx])
                Qbad = Qspec if Pshared else np.ascontiguousarray(Qspec[idx])
                any_progress = False
                for s in range(0, idx.size, cs):
                    e = min(s + cs, idx.size)
                    sub = idx[s:e]
                    r = self._run_chunk(
                        Vbad[:, s:e],
                        Pbad if Pshared else Pbad[s:e],
                        Qbad if Pshared else Qbad[s:e],
                        Pshared, d_Ym, d_Ya, 0, None,
                        max_iter, tol, Ym_rep, Ya_rep, flat_rep)
                    newly_ok = r["converged"] & np.isfinite(r["V"]).all(axis=0)
                    # only accept columns that actually improved to a finite converged
                    # solution, so a re-corrupted retry cannot overwrite a good result.
                    take = np.nonzero(newly_ok)[0]
                    if take.size:
                        V_out[:, sub[take]] = r["V"][:, take]
                        iters_out[sub[take]] = np.maximum(iters_out[sub[take]],
                                                          r["iterations"][take])
                        conv_out[sub[take]] = True
                        any_progress = True
                if not any_progress:
                    break  # nothing improved this pass -> genuinely non-convergent columns

        return {"V": np.ascontiguousarray(V_out),
                "iterations": iters_out, "converged": conv_out}

    # -------------------------------------------------------- one chunk (a batch)
    def _run_chunk(self, V0, Pspec, Qspec, Pshared, d_Ym, d_Ya, Ystride,
                   d_pin_ptr, max_iter, tol, Ym_rep, Ya_rep, flat_rep):
        """Solve ONE chunk (a single batch). Args mirror _run_newton_batch but for a column
        sub-range; d_Ym/d_Ya/d_pin_ptr are device pointers already offset to this chunk's
        first column. Pshared: Pspec/Qspec are a shared (n,) vector (Pstride=0), else (B,n)."""
        t = self.topo
        n, m = t.n, t.m
        B = V0.shape[1]

        Vm0 = np.ascontiguousarray(np.abs(V0).T, dtype=np.float64)      # (B, n)
        Va0 = np.ascontiguousarray(np.angle(V0).T, dtype=np.float64)

        # shared injections -> upload one n-vector and broadcast (Pstride=0); else (B,n).
        Pflat = np.ascontiguousarray(np.asarray(Pspec, dtype=np.float64).ravel())
        Qflat = np.ascontiguousarray(np.asarray(Qspec, dtype=np.float64).ravel())
        Pstride = 0 if Pshared else n

        d_Vm = _upload(Vm0)
        d_Va = _upload(Va0)
        d_Pspec = _upload(Pflat)
        d_Qspec = _upload(Qflat)
        d_F = cuda.mem_alloc(B * m * _F64)
        d_dblocks = cuda.mem_alloc(B * 4 * t.nnzY * _F64)
        d_resid = cuda.mem_alloc(B * _F64)
        d_conv = cuda.mem_alloc(B * _U8)
        cuda.memset_d8(d_conv, 0, B)

        d_pin = np.uintp(d_pin_ptr) if d_pin_ptr is not None else np.uintp(0)

        # -- symbolic factorization once (representative Jx) --
        Ym_r = t.Ym if Ym_rep is None else np.ascontiguousarray(Ym_rep, dtype=np.float64)
        Ya_r = t.Ya if Ya_rep is None else np.ascontiguousarray(Ya_rep, dtype=np.float64)
        # For contingency, evaluate the representative Jacobian at a FLAT voltage on the
        # base grid: an outaged/pinned case's own start can give a numerically singular J
        # that breaks the host-LU pivot selection, even though the shared pattern is fine.
        if flat_rep:
            Vm_r = np.ones(n); Va_r = np.zeros(n)
        else:
            Vm_r, Va_r = Vm0[0], Va0[0]
        # (Pspec/Qspec unused by _eval_Jx_host -- J is independent of S -- pass None.)
        Jx0 = self._eval_Jx_host(Vm_r, Va_r, None, None, Ym_r, Ya_r)
        Backend = self._resolve_backend(self.backend)
        kw = dict(reorder=self.reorder, numeric_zero=self._numeric_zero,
                  numeric_boost=self._numeric_boost)
        if self.backend == "cudss":
            kw["pivot"] = self.pivot     # PIVOT_NONE by default -- cheap refactor on well-cond J
        rf = Backend(t.Jp, t.Jj, batch_size=B, **kw)
        rf.symbolic_setup(Jx0)

        conv_host = np.zeros(B, dtype=np.uint8)
        iters = np.zeros(B, dtype=np.int32)

        bs = 256
        def grid(total): return ((total + bs - 1) // bs, 1, 1)
        g_bn = grid(B * n)
        g_bJ = grid(B * t.nnzJ)
        g_bm = grid(B * m)

        for it in range(max_iter):
            # 1. assemble F + derivative blocks (pre-update mismatch)
            self._k_eval(
                self.d_Yp, self.d_Yj, self.d_Ydiag, d_Ym, d_Ya, np.int64(Ystride),
                d_Vm, d_Va, d_Pspec, d_Qspec, np.int64(Pstride),
                self.d_pvpq, self.d_pq, self.d_pvpq_pos, self.d_pq_pos,
                d_pin,
                d_dblocks, d_F,
                np.int32(n), np.int32(t.nnzY), np.int32(t.npvpq),
                np.int32(t.npq), np.int32(m), np.int32(B),
                block=(bs, 1, 1), grid=g_bn)

            # 2. per-column residual + converged mask
            self._k_infnorm(
                d_F, d_resid, d_conv, np.float64(tol), np.int32(m), np.int32(B),
                block=(bs, 1, 1), grid=(B, 1, 1), shared=bs * _F64)

            # Convergence check requires a device->host copy of the mask, which is a
            # BLOCKING sync that stalls the async kernel pipeline every iteration. Power-flow
            # Newton is very predictable (~5-6 iters), so we skip the check for the first
            # `check_every-1` iters and only sync periodically. Worst case we run a couple of
            # already-converged iters (cheap: converged columns are near-fixed points), but
            # we remove most per-iter sync stalls -> big win on large batches. The kernels
            # already no-op converged columns (update_voltage skips them via d_conv).
            ce = self.check_every
            if it + 1 >= self.min_iter and (it + 1 - self.min_iter) % ce == 0:
                cuda.memcpy_dtoh(conv_host, d_conv)
                if not (conv_host == 0).any():
                    iters[:] = np.maximum(iters, it + 1)
                    break
            iters[:] = it + 1

            # 3. gather J into rf's device Jx buffer (applies pin identity rows)
            self._k_gather(
                d_dblocks, self.d_src_block, self.d_src_k,
                self.d_Jp, self.d_Jj, self.d_pvpq, self.d_pq,
                d_pin,
                rf.d_A_batch,
                np.int32(t.nnzY), np.int32(t.nnzJ), np.int32(t.npvpq),
                np.int32(t.npq), np.int32(n), np.int32(B), self.d_row_of_nz,
                block=(bs, 1, 1), grid=g_bJ)

            # rhs = -F into the backend's rhs buffer. For rf/qr this aliases d_X_batch
            # (in-place solve); for cuDSS it is a distinct buffer (out-of-place solve).
            self._k_negate(d_F, rf.d_rhs_batch, np.int32(m), np.int32(B),
                           block=(bs, 1, 1), grid=g_bm)

            rf.reset_refactor_device()
            rf.batch_solve_device()

            self._k_update(
                rf.d_X_batch, self.d_pvpq, self.d_pq, d_conv,
                d_Vm, d_Va,
                np.int32(t.npvpq), np.int32(t.npq), np.int32(n), np.int32(B),
                block=(bs, 1, 1), grid=g_bm)

        cuda.Context.synchronize()

        Vm_h = np.empty((B, n), dtype=np.float64)
        Va_h = np.empty((B, n), dtype=np.float64)
        cuda.memcpy_dtoh(Vm_h, d_Vm)
        cuda.memcpy_dtoh(Va_h, d_Va)
        V = (Vm_h * np.exp(1j * Va_h)).T                # (n, B)
        cuda.memcpy_dtoh(conv_host, d_conv)

        # free this chunk's device memory so the next chunk starts clean (chunking exists
        # precisely because the whole batch doesn't fit; leaking would defeat it). The RF
        # handle frees its own buffers + destroys the cusolverRf/Sp handles via __del__.
        for buf in (d_Vm, d_Va, d_Pspec, d_Qspec, d_F, d_dblocks, d_resid, d_conv):
            buf.free()
        rf.free()

        return {
            "V": np.ascontiguousarray(V),
            "iterations": iters.copy(),
            "converged": conv_host.astype(bool),
        }

    # -------------------------------------------------------------- helpers
    def _eval_Jx_host(self, Vm, Va, Pspec, Qspec, Ym, Ya) -> np.ndarray:
        """Representative Jacobian values (sorted CSR order) for the symbolic setup.

        Pure numpy replica of the polar derivative + gather (matches the GPU kernel and
        nr_klu.cpp). Ym/Ya are the (representative) Ybus polar values -- topo's for
        time-series, case 0's for contingency. Only run once per solve; speed irrelevant.
        Pspec/Qspec are accepted for signature symmetry but unused (J is independent of S).
        """
        t = self.topo
        n, nnzY = t.n, t.nnzY
        dP_dVa = np.zeros(nnzY); dQ_dVa = np.zeros(nnzY)
        dP_dVm = np.zeros(nnzY); dQ_dVm = np.zeros(nnzY)
        Yp, Yj, Yd = t.Yp, t.Yj, t.Ydiag
        for row in range(n):
            dix = Yd[row]
            for k in range(Yp[row], Yp[row + 1]):
                col = Yj[k]
                delta = Va[row] - Ya[k] - Va[col]
                s = np.sin(delta); c = np.cos(delta)
                VmrowYm = Vm[row] * Ym[k]
                vij = VmrowYm * Vm[col]
                if row != col:
                    dpva = vij * s; dqva = -vij * c
                    dP_dVa[k] = dpva; dQ_dVa[k] = dqva
                    dP_dVm[k] = VmrowYm * c; dQ_dVm[k] = VmrowYm * s
                    dP_dVa[dix] -= dpva; dQ_dVa[dix] -= dqva
                    dP_dVm[dix] += Ym[k] * Vm[col] * c
                    dQ_dVm[dix] += Ym[k] * Vm[col] * s
                else:
                    dP_dVm[dix] += 2.0 * Vm[row] * Ym[k] * c
                    dQ_dVm[dix] += 2.0 * Vm[row] * Ym[k] * s
        blocks = [dP_dVa, dP_dVm, dQ_dVa, dQ_dVm]
        Jx = np.empty(t.nnzJ, dtype=np.float64)
        for p in range(t.nnzJ):
            Jx[p] = blocks[t.src_block[p]][t.src_k[p]]
        return np.ascontiguousarray(Jx, dtype=np.float64)
