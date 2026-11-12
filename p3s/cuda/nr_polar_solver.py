# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fully-resident batched polar Newton-Raphson on the GPU.

The GPU counterpart of ``nr_klu.Solver``: build once per grid topology, then
``solve_batch`` (time-series: one Ybus, many injections) or ``solve_batch_contingency*``
(N-1: one injection, per-case Ybus values and pin mask). The iteration is the one of
``p3s/cpp/nr_klu.cpp`` -- polar mismatch and Jacobian, Armijo backtracking line search,
static-pivot LU on a frozen pattern -- restructured for a GPU:

* **Everything stays on the device.** Voltage state, mismatch, Jacobian, factors and the
  Newton step never leave it; per iteration the host only reads back one status word per
  system. Inputs and results are transferred in the numpy ``(n, B)`` layout the caller
  already has (the kernels are batch-interleaved, see ``nr_polar_kernels.cu``), so there
  are no host-side transposes, magnitude/angle conversions or per-column Python loops.

* **Linear solve: batched lean LU** (``backend="lean"``, the default; no library needed).
  The pivot order and the factor pattern are frozen once per topology
  (``lean_lu_plan.py``), like the C++ ``lean_lu.hpp``. The Jacobian assembly writes
  straight into the factor array and ONE kernel launch per Newton iteration factorizes and
  solves the whole chunk. On pegase 9241 this is ~4x faster per iteration than the cuDSS
  uniform batch (RTX A500). ``precision="mixed"`` factorizes in single precision while the
  iteration is far from the solution (the mismatch stays double, so the result does not
  change, only the path to it): another ~1.3x.

* **cuDSS** (``backend="cudss"``) stays available for comparison and as the pivoting
  alternative; ``"rf"`` (cusolverRf) and ``"qr"`` are the legacy/zero-dependency
  fallbacks. These receive the Jacobian in their own system-major CSR buffers; their
  symbolic analysis is done once per batch size and reused across chunks and calls.

* **CPU safety net** (when the ``nr_klu`` extension is installed). A system the frozen
  pivot order does not suit (pivot ratio check in the kernel) or that does not converge is
  re-solved by ``nr_klu`` (KLU with pivoting) -- the safety net ``nr_klu`` has for its own
  lean path. And the last few stragglers of a chunk are handed to it mid-iteration: a
  handful of systems cannot fill a GPU, so finishing them there would cost more than the
  rest of the chunk. ``last_stats`` reports how many systems took either route.
"""

from __future__ import annotations

import os

import numpy as np
import pycuda.driver as cuda
import scipy.sparse as sp
from numpy.typing import NDArray
from pycuda.compiler import SourceModule

from p3s.cuda import _ctx  # noqa: F401  (retains the CUDA primary context; cuDSS-safe)
from p3s.cuda.lean_lu_plan import LeanLUPlan, build_lean_lu_plan
from p3s.cuda.polar_topology import PolarTopology, build_polar_topology, jacobian_values_host

_F64 = 8
_C128 = 16
_I32 = 4

# status codes (nr_polar_kernels.cu)
_ACTIVE, _CONVERGED, _FAILED, _BACKTRACK, _NEEDJ = range(5)

# Accept a frozen-order factorization while its pivot ratio stays within this factor of
# the flat-start one (nr_klu.cpp: REUSE_RCOND_RATIO).
_RCOND_RATIO = 1e-3

_KERNEL_SRC = os.path.join(os.path.dirname(__file__), "nr_polar_kernels.cu")
_BLOCK = 256  # threads per block of the per-item kernels

# argument formats of the prepared kernels (P pointer, i int32, q int64, d double)
_KERNELS = {
    "init_state": "PiPPPPiii",
    "eval_FJ": "PPP" + "Pi" + "PPP" + "Pi" + "PP" + "P" + "Pii" + "PPqq" + "Pqq" + "P" + "iii",
    "newton_decide": "PPPPPPPPPP" + "dd" + "iiiii",
    "apply_step": "Pqq" + "PP" + "PiP" + "PPPP" + "iii",
    "finalize": "PPPPiii",
    "expand_Y": "PPiii",
    "apply_Y_delta": "PPPPii",
    "lean_factor_solve_f64": "P" * 17 + "PPPP" + "PPP" + "d" + "iiiiiiii",
    "lean_factor_solve_f32": "P" * 17 + "PPPP" + "PPP" + "d" + "iiiiiiii",
}
# LeanLUPlan arrays, in the order lean_factor_solve takes them
_PLAN_ARRAYS = "lptr tdst tcode tpp pa pb flptr frow fptr fcol fpos perm blptr brow base Lp Li".split()


def _upload(arr) -> cuda.DeviceAllocation:
    arr = np.ascontiguousarray(arr)
    d = cuda.mem_alloc(max(arr.nbytes, 8))
    if arr.nbytes:
        cuda.memcpy_htod(d, arr)
    return d


def _grid(items: int) -> tuple[int, int, int]:
    return (max(1, (items + _BLOCK - 1) // _BLOCK), 1, 1)


def _copy_columns(dev: int, host: NDArray, col0: int, B: int, to_device: bool):
    """Copy columns ``[col0, col0 + B)`` of the C-ordered host array ``(rows, B_total)``
    to / from a device buffer laid out as ``(rows, B)``.

    A column range of a C-ordered array is strided; the pitched copy lets the driver walk
    the rows, which spares a gather into (or scatter from) a contiguous temporary.
    """
    item = host.itemsize
    cp = cuda.Memcpy2D()
    if to_device:
        cp.set_src_host(host)
        cp.set_dst_device(dev)
        cp.src_x_in_bytes, cp.src_pitch, cp.dst_pitch = col0 * item, host.strides[0], B * item
    else:
        cp.set_src_device(dev)
        cp.set_dst_host(host)
        cp.dst_x_in_bytes, cp.dst_pitch, cp.src_pitch = col0 * item, host.strides[0], B * item
    cp.width_in_bytes = B * item
    cp.height = host.shape[0]
    cp(aligned=False)


class _Buffers:
    """Named device buffers that persist across chunks and calls and only ever grow."""

    def __init__(self):
        self._bufs: dict[str, tuple[cuda.DeviceAllocation, int]] = {}

    def get(self, name: str, nbytes: int) -> int:
        cur = self._bufs.get(name)
        if cur is None or cur[1] < nbytes:
            if cur is not None:
                cur[0].free()
                del self._bufs[name]
            self._bufs[name] = (cuda.mem_alloc(max(int(nbytes), 8)), int(nbytes))
        return int(self._bufs[name][0])

    @property
    def nbytes(self) -> int:
        return sum(b[1] for b in self._bufs.values())

    def free(self):
        for buf, _ in self._bufs.values():
            try:
                buf.free()
            except Exception:
                pass
        self._bufs = {}


class _Chunk:
    """Problem data of one chunk of ``B`` systems (host arrays).

    ``col0 is None``: the 2-D arrays hold exactly the chunk's systems. Otherwise they are
    the caller's full ``(rows, B_total)`` arrays and the chunk is their columns
    ``[col0, col0 + B)`` -- transferred as a pitched copy, without gathering them first.
    """

    def __init__(self, B, V0, Sbus, Yx=None, stamp=None, pin=None, col0=None):
        self.B = B
        self.col0 = col0
        self.V0 = V0  # (n,) shared or (n, B) complex128
        self.Sbus = Sbus  # (n,) shared or (n, B) complex128
        self.Yx = Yx  # None (topology Ybus) or (nnzY, B) complex128
        self.stamp = stamp  # None or sparse (nnzY, B): values to subtract from the base Ybus
        self.pin = pin  # None or (n, B) uint8


class PolarNewtonSolverCUDA:
    """Device-resident batched polar Newton solver for one grid topology."""

    _mod = None  # compiled kernel module, shared across instances
    _k: dict = {}

    def __init__(self, Yp, Yj, Yx, pv, pq, reorder: str = "symrcm", backend: str = "lean", slots: int = 32):
        """
        Parameters
        ----------
        Yp, Yj, Yx : CSR Ybus (sorted column indices), complex128 values.
        pv, pq : bus index sets.
        backend : linear solver of the Newton step
            ``"lean"``  batched static-pivot LU kernel (default; needs nothing but CUDA).
            ``"cudss"`` NVIDIA cuDSS uniform batch (``pip install nvidia-cudss-cuXX``).
            ``"rf"``    cusolverRf batch refactorization -- legacy, deprecated by NVIDIA and
                        broken on CUDA 12.4 (see ``diagnose_gpu.py``).
            ``"qr"``    per-system cusolverSp QR -- slow, dependency-free fallback.
        reorder : fill-reducing ordering of the ``"rf"`` backend (ignored by the others).
        slots : warps per thread block of the lean LU kernel (1..32).
        """
        if backend not in ("lean", "cudss", "rf", "qr"):
            raise ValueError(f"backend must be 'lean', 'cudss', 'rf' or 'qr', got {backend!r}")
        if not 1 <= slots <= 32:
            raise ValueError("slots must be in 1..32")
        self.topo: PolarTopology = build_polar_topology(Yp, Yj, Yx, pv, pq)
        self.backend = backend
        self.reorder = reorder
        self.slots = int(slots)

        # Hard cap on the systems solved as one chunk, on top of the memory budget.
        # None = memory budget only.
        self.max_chunk: int | None = None
        # Share of the free device memory a chunk may claim. Past ~60 % a Windows (WDDM)
        # GPU starts to slow down across the board.
        self.mem_fraction = 0.6
        # Precision of the lean LU factors. "double": every factorization in double, the
        # iteration nr_klu runs. "mixed": single-precision factors while a system's
        # mismatch is above `mixed_switch` (p.u.), double for the final steps (decided per
        # system and iteration in the newton_decide kernel). The mismatch itself is always
        # double, so a converged result satisfies the same tolerance; only the path to it
        # changes (iteration counts can differ by one). Costs a float copy of the factors
        # in device memory.
        self.precision = "double"
        self.mixed_switch = 1e-3
        self._rcond_ratio_single = _RCOND_RATIO  # pivot-ratio bound of the single kernel
        # Armijo backtracking line search (nr_klu's damped Newton). The in-place backends
        # ("rf", "qr") overwrite the Newton step with the next right-hand side, so they
        # always take full steps.
        self.line_search = True
        # What to do with systems that fail on the GPU (rejected factorization, no
        # convergence): "auto" re-solves them with nr_klu on the CPU if it is installed,
        # otherwise (cuDSS only) retries them on the GPU; "none" reports them as they are.
        self.fallback = "auto"
        # cuDSS pivot strategy: "none" disables the pivot search (the dominant cost of a
        # refactorization, and unnecessary for power-flow Jacobians), "auto" keeps cuDSS's.
        self.pivot = "none"
        # cuDSS 0.8's uniform-batch solve sporadically corrupts a few systems of a large
        # batch (see test_cudss_batch.py). Capping the batch keeps it clear of that, and
        # the fallback picks up anything that still slips through.
        self._cudss_safe_chunk = 256
        self._cudss_max_retries = 4
        # Stragglers: once at most this many systems of a chunk (and at most 1/8 of it) are
        # still iterating, they are finished by nr_klu on the CPU, starting from the GPU's
        # current iterate (needs the fallback; 0 = iterate everything on the GPU).
        self.cpu_handoff = 32
        # Pivot perturbation of the "rf" backend (set per call on the contingency path).
        self._numeric_zero = 0.0
        self._numeric_boost = 0.0
        # Filled by every solve: chunking and fallback counters.
        self.last_stats: dict = {}

        self._compile_kernels()
        self._buf = _Buffers()
        self._lib: dict = {}  # library backend objects by batch size
        self._cpu = None  # lazily built nr_klu.Solver (fallback)

        t = self.topo
        # constant, topology-only device buffers
        self.d_Yp = _upload(t.Yp)
        self.d_Yj = _upload(t.Yj)
        self.d_Ydiag = _upload(t.Ydiag)
        self.d_Ybase = _upload(t.Yx)
        self.d_row_p = _upload(t.row_p)
        self.d_row_q = _upload(t.row_q)

        self.plan: LeanLUPlan | None = None
        self._rcond0 = 0.0
        if backend == "lean":
            self._setup_lean()
        else:
            # variable index of each bus's angle / magnitude in the solver's x (CSR order)
            self.d_posmap = _upload(t.posJ)
            self.d_xva = _upload(t.row_p)
            self.d_xvm = _upload(t.row_q)

    # ------------------------------------------------------------------ setup
    @classmethod
    def _compile_kernels(cls):
        if cls._mod is not None:
            return
        with open(_KERNEL_SRC) as fh:
            src = fh.read()
        mod = SourceModule(src, no_extern_c=True)
        k = {}
        for name, fmt in _KERNELS.items():
            fn = mod.get_function(name)
            fn.prepare(fmt)
            k[name] = fn
        cls._mod, cls._k = mod, k

    def _setup_lean(self):
        """Freeze the lean LU plan and measure its reference pivot ratio."""
        t = self.topo
        plan = build_lean_lu_plan(t.Jp, t.Jj, slots=self.slots)
        self.plan = plan
        self._d_plan = [_upload(getattr(plan, name)) for name in _PLAN_ARRAYS]
        # Jacobian entry -> factor position; Newton step of a bus -> pivot-order index
        self.d_posmap = _upload(np.where(t.posJ >= 0, plan.apos[np.maximum(t.posJ, 0)], -1).astype(np.int32))
        row_p, row_q = t.row_p, t.row_q
        self.d_xva = _upload(np.where(row_p >= 0, plan.pinv[np.maximum(row_p, 0)], -1).astype(np.int32))
        self.d_xvm = _upload(np.where(row_q >= 0, plan.pinv[np.maximum(row_q, 0)], -1).astype(np.int32))

        self._rcond0 = self._lean_reference_rcond()

    def _lean_reference_rcond(self) -> float:
        """min|pivot| / max|pivot| of the frozen order on the base-grid Jacobian at a flat
        voltage -- an operating-point-free matrix of the right structure, as in nr_klu's
        build_lean_plan. Every later factorization is judged against it."""
        t = self.topo
        b = self._bind(1, _Chunk(1, np.ones(t.n, dtype=np.complex128), np.zeros(t.n, dtype=np.complex128)))
        b["mixed"] = False
        self._eval(b, 1 << _ACTIVE, 1 << _ACTIVE)
        self._lean_factor_solve(b, rcond_min=0.0, do_solve=False)
        status = np.empty(1, dtype=np.int32)
        pstat = np.empty(2, dtype=np.float64)
        cuda.memcpy_dtoh(status, b["status"])
        cuda.memcpy_dtoh(pstat, b["pstat"])
        if status[0] != _ACTIVE or not np.isfinite(pstat).all() or pstat[1] <= 0.0:
            raise RuntimeError(
                "lean LU: the Jacobian has a zero or non-finite diagonal pivot at flat start, so the static "
                "pivot order cannot be used for this grid; construct the solver with backend='cudss'."
            )
        return float(pstat[0] / pstat[1])

    def _library_backend(self, B: int):
        """cuDSS / cusolverRf / QR object for batch size ``B`` (analysis done once, cached)."""
        lib = self._lib.get(B)
        if lib is not None:
            return lib
        # these objects hold batch-sized device memory: keep the full-chunk one plus the
        # most recent odd size (the last chunk of a batch).
        for size in list(self._lib)[:-1]:
            self._lib.pop(size).free()
        t = self.topo
        if self.backend == "cudss":
            from p3s.cuda.cudss_batch import CudssBatch

            lib = CudssBatch(t.Jp, t.Jj, batch_size=B, pivot=self.pivot)
        elif self.backend == "rf":
            from p3s.cuda.cusolver_rf_batch import CusolverRfBatch

            lib = CusolverRfBatch(
                t.Jp,
                t.Jj,
                batch_size=B,
                reorder=self.reorder,
                numeric_zero=self._numeric_zero,
                numeric_boost=self._numeric_boost,
            )
        else:
            from p3s.cuda.cusolver_qr_batch import CusolverQRBatch

            lib = CusolverQRBatch(t.Jp, t.Jj, batch_size=B)
        # Representative values for the symbolic setup: the intact grid at a flat voltage.
        # A case's own values can be numerically singular (outage / pinned buses), which
        # would derail a host LU's pivot selection although the shared pattern is fine.
        lib.symbolic_setup(jacobian_values_host(t, np.ones(t.n), np.zeros(t.n)))
        self._lib[B] = lib
        return lib

    # ------------------------------------------------------------------ public API
    def solve_batch(self, Sbus_mat, V0, max_iter: int = 30, tol: float = 1e-8):
        """Solve a batch of operating points sharing this topology (time-series).

        Sbus_mat : (n, T) complex  per-column injections
        V0       : (n,) or (n, T) complex  start voltage (shared if 1-D)
        Returns dict: V (n, T) complex, iterations (T,) int, converged (T,) bool.
        """
        n = self.topo.n
        Sbus_mat = np.ascontiguousarray(Sbus_mat, dtype=np.complex128)
        if Sbus_mat.ndim != 2 or Sbus_mat.shape[0] != n:
            raise ValueError("Sbus must be (n_bus, T)")
        T = Sbus_mat.shape[1]
        V0 = np.ascontiguousarray(V0, dtype=np.complex128)
        if V0.shape not in ((n,), (n, T)):
            raise ValueError("V0 must be (n_bus,) or (n_bus, T)")
        return self._run_newton_batch(T, V0, Sbus_mat, None, None, None, max_iter, tol)

    def solve_batch_contingency_cx(
        self,
        Yx,
        Sbus,
        V0,
        pin,
        max_iter: int = 30,
        tol: float = 1e-8,
        numeric_zero: float = 1e-10,
        numeric_boost: float = 1e-8,
    ):
        """Solve a batch of N-1 contingencies sharing this topology's pattern.

        Mirrors ``nr_klu.Solver.solve_batch_contingency`` with complex Ybus values:
          Yx   : (nnz, L) complex128  per-case Ybus values (same Yp/Yj as the solver)
          Sbus : (n,) complex128      injections (constant across cases)
          V0   : (n, L) complex128    per-case start (pinned buses pre-set)
          pin  : (n, L) uint8         per-case freeze mask (1 = pin)
        ``numeric_zero`` / ``numeric_boost`` only affect ``backend="rf"`` (pivot boost).
        Returns dict: V (n, L) complex, iterations (L,) int, converged (L,) bool.
        """
        t = self.topo
        Yx = np.ascontiguousarray(Yx, dtype=np.complex128)
        if Yx.ndim != 2 or Yx.shape[0] != t.nnzY:
            raise ValueError("Yx must be (Ybus nnz, L)")
        L = Yx.shape[1]
        Sbus, V0, pin = self._check_contingency_inputs(L, Sbus, V0, pin)
        self._numeric_zero, self._numeric_boost = numeric_zero, numeric_boost
        return self._run_newton_batch(L, V0, Sbus, Yx, None, pin, max_iter, tol)

    def solve_batch_contingency(
        self,
        Yx_mag,
        Yx_ang,
        Sbus,
        V0,
        pin,
        max_iter: int = 30,
        tol: float = 1e-8,
        numeric_zero: float = 1e-10,
        numeric_boost: float = 1e-8,
    ):
        """``solve_batch_contingency_cx`` with the per-case Ybus given as magnitude/angle
        ``(nnz, L)`` arrays -- the exact signature of ``nr_klu.Solver.solve_batch_contingency``."""
        Yx = np.asarray(Yx_mag, dtype=np.float64) * np.exp(1j * np.asarray(Yx_ang, dtype=np.float64))
        return self.solve_batch_contingency_cx(Yx, Sbus, V0, pin, max_iter, tol, numeric_zero, numeric_boost)

    def solve_batch_contingency_stamp(
        self,
        stamp,
        Sbus,
        V0,
        pin,
        max_iter: int = 30,
        tol: float = 1e-8,
        numeric_zero: float = 1e-10,
        numeric_boost: float = 1e-8,
    ):
        """N-1 batch with the per-case Ybus given as sparse outage stamps.

        ``stamp`` is a scipy sparse ``(nnz, L)`` matrix; case ``c`` runs on
        ``Yx_base - stamp[:, c]`` (``ContingencyBatch``'s native form). The per-case Ybus
        is expanded on the device, so neither the dense ``(nnz, L)`` matrix nor its upload
        is needed -- for a full pegase N-1 that is ~10 GB of host memory and bus traffic.
        Other arguments and the result are those of ``solve_batch_contingency_cx``.
        """
        t = self.topo
        stamp = sp.csc_matrix(stamp, dtype=np.complex128)
        if stamp.shape[0] != t.nnzY:
            raise ValueError("stamp must be (Ybus nnz, L)")
        stamp.sum_duplicates()
        L = stamp.shape[1]
        Sbus, V0, pin = self._check_contingency_inputs(L, Sbus, V0, pin)
        self._numeric_zero, self._numeric_boost = numeric_zero, numeric_boost
        return self._run_newton_batch(L, V0, Sbus, None, stamp, pin, max_iter, tol)

    def _check_contingency_inputs(self, L, Sbus, V0, pin):
        n = self.topo.n
        Sbus = np.ascontiguousarray(np.asarray(Sbus, dtype=np.complex128).ravel())
        V0 = np.ascontiguousarray(V0, dtype=np.complex128)
        pin = np.ascontiguousarray(pin, dtype=np.uint8)
        if Sbus.shape != (n,):
            raise ValueError("Sbus must be (n_bus,)")
        if V0.shape != (n, L) or pin.shape != (n, L):
            raise ValueError("V0/pin must be (n_bus, L)")
        return Sbus, V0, pin

    def update_Y(self, Yx):
        """Refresh the Ybus values (same sparsity pattern), e.g. after a tap change.

        The counterpart of ``nr_klu.Solver.update_Y``: nothing symbolic is redone.
        """
        t = self.topo
        Yx = np.ascontiguousarray(Yx, dtype=np.complex128)
        if Yx.shape != t.Yx.shape:
            raise ValueError("update_Y: nnz mismatch (topology changed; rebuild the solver)")
        if np.array_equal(Yx, t.Yx):
            return
        t.Yx = Yx
        t.Ym = np.abs(Yx)
        t.Ya = np.angle(Yx)
        cuda.memcpy_htod(self.d_Ybase, Yx)
        self._cpu = None
        if self.backend == "lean":
            self._rcond0 = self._lean_reference_rcond()

    def free(self):
        """Release the device memory held for chunks and library backends."""
        self._buf.free()
        for lib in self._lib.values():
            lib.free()
        self._lib = {}

    # ------------------------------------------------------ chunk-size budgeting
    def _bytes_per_system(self, per_sys_S: bool, per_sys_Y: bool, has_pin: bool) -> int:
        """Device bytes one system of a chunk occupies."""
        t = self.topo
        n, m = t.n, t.m
        per = 4 * n * _F64 + n * _C128  # Vm/Va/cs/sn + start/result voltage
        if per_sys_S:
            per += n * _C128
        if per_sys_Y:
            per += t.nnzY * _C128
        if has_pin:
            per += n
        if self.backend == "lean":
            per += (2 * m + self.plan.nnz_lu) * _F64  # R, y, factors
            if self.precision == "mixed":
                per += self.plan.nnz_lu * 4  # single-precision factors
        else:
            # library buffers (A, rhs, x) plus the library's own factors and workspace,
            # which we can only estimate (cuDSS: ~2 nnz(J) factor entries and scratch)
            per += (t.nnzJ + 2 * m) * _F64 + 4 * t.nnzJ * _F64
        return per

    def _chunk_size(self, B: int, per_sys_S: bool, per_sys_Y: bool, has_pin: bool) -> int:
        """Largest chunk within ``mem_fraction`` of the device memory we can use."""
        free, _ = cuda.mem_get_info()
        usable = (free + self._buf.nbytes) * self.mem_fraction
        cs = max(1, int(usable // self._bytes_per_system(per_sys_S, per_sys_Y, has_pin)))
        if self.max_chunk is not None:
            cs = min(cs, self.max_chunk)
        if self.backend == "cudss" and self._cudss_safe_chunk:
            cs = min(cs, self._cudss_safe_chunk)
        if cs >= B:
            return B
        if cs < 32:
            return cs
        # Whole warps (every per-item kernel then runs one item per warp), and chunks of
        # equal size: a small remainder chunk cannot fill the GPU and would cost almost as
        # much wall time as a full one.
        cs -= cs % 32
        n_chunks = -(-B // cs)
        even = -(-B // n_chunks)  # ceil(B / n_chunks)
        return min(cs, (even + 31) // 32 * 32)

    # -------------------------------------------------------- chunked Newton
    def _run_newton_batch(self, B, V0, Sbus, Yx, stamp, pin, max_iter, tol):
        """Solve ``B`` systems in memory-sized chunks, then rescue the failures.

        V0    : (n,) shared or (n, B) complex128
        Sbus  : (n,) shared or (n, B) complex128
        Yx    : None (topology Ybus) | (nnzY, B) complex128 per-system values
        stamp : None | sparse (nnzY, B) outage stamps (per-system Ybus = base - stamp)
        pin   : None | (n, B) uint8
        """
        n = self.topo.n
        per_sys_Y = Yx is not None or stamp is not None
        cs = self._chunk_size(B, Sbus.ndim == 2, per_sys_Y, pin is not None)

        V_out = np.empty((n, B), dtype=np.complex128)
        iters_out = np.zeros(B, dtype=np.int32)
        conv_out = np.zeros(B, dtype=bool)

        def chunk(idx):
            """Problem data of the systems ``idx`` (a slice or an index array)."""
            st = None if stamp is None else stamp[:, idx]
            if isinstance(idx, slice):  # a column range: hand the full arrays over as they are
                size = idx.stop - idx.start
                return _Chunk(size, V0, Sbus, Yx, st, pin, col0=None if size == B else idx.start)

            def cols(a):
                return a if a is None or a.ndim == 1 else np.ascontiguousarray(a[:, idx])

            return _Chunk(len(idx), cols(V0), cols(Sbus), cols(Yx), st, cols(pin))

        cpu = self._cpu_solver() if self.fallback != "none" else None
        stats = {
            "systems": B,
            "chunk": cs,
            "chunks": -(-B // cs),
            "handed_off": 0,
            "failed_on_gpu": 0,
            "fallback": None,
            "single_factorizations": 0,
            "double_factorizations": 0,
        }
        self.last_stats = stats
        on_cpu = np.zeros(B, dtype=bool)  # finished by nr_klu already: nothing left to rescue
        tails: dict[int, list] = {}  # GPU iterations done -> [(system indices, current iterate)]
        for start in range(0, B, cs):
            sl = slice(start, min(start + cs, B))
            handoff = self.cpu_handoff if cpu is not None else 0
            r = self._run_chunk(chunk(sl), max_iter, tol, handoff=handoff, out=(V_out, start))
            iters_out[sl] = r["iterations"]
            conv_out[sl] = r["converged"]
            tail = r.get("handed_off")
            if tail is not None and tail.size:
                tails.setdefault(r["it"], []).append((start + tail, r["V"][:, tail]))

        # Stragglers of all chunks: continue from the GPU's iterate on the CPU, one parallel
        # batch per iteration count they were handed over at.
        for it, parts in tails.items():
            idx = np.concatenate([p[0] for p in parts])
            ch = chunk(idx)
            ch.V0 = np.ascontiguousarray(np.concatenate([p[1] for p in parts], axis=1))
            rc = self._solve_on_cpu(cpu, ch, max_iter - it, tol)
            V_out[:, idx] = rc["V"]
            iters_out[idx] = it + np.asarray(rc["iterations"])
            conv_out[idx] = rc["converged"]
            on_cpu[idx] = True
            stats["handed_off"] += int(idx.size)

        # A system that did not converge on the GPU carries a non-finite mismatch or ran out
        # of iterations; either way `converged` is the whole story (a NaN voltage cannot pass
        # the convergence test), so the result needs no scan for non-finite values.
        bad = np.nonzero(~conv_out & ~on_cpu)[0]
        if bad.size and self.fallback != "none":
            stats["failed_on_gpu"] = int(bad.size)
            if cpu is not None:
                stats["fallback"] = "nr_klu"
                self._merge(V_out, iters_out, conv_out, bad, self._solve_on_cpu(cpu, chunk(bad), max_iter, tol))
            elif self.backend == "cudss":
                # cuDSS's batch corruption is non-deterministic: a fresh attempt in
                # safe-sized sub-batches almost always lands; stop when a pass gains nothing.
                stats["fallback"] = "gpu-retry"
                for _ in range(self._cudss_max_retries):
                    progress = False
                    for s in range(0, bad.size, cs):
                        sub = bad[s : s + cs]
                        r = self._run_chunk(chunk(sub), max_iter, tol)
                        progress |= self._merge(V_out, iters_out, conv_out, sub, r)
                    bad = np.nonzero(~conv_out)[0]
                    if bad.size == 0 or not progress:
                        break
        return {"V": V_out, "iterations": iters_out, "converged": conv_out}

    @staticmethod
    def _merge(V_out, iters_out, conv_out, idx, r) -> bool:
        """Take over the systems of ``r`` that came back converged."""
        take = np.nonzero(np.asarray(r["converged"], dtype=bool) & np.isfinite(r["V"]).all(axis=0))[0]
        if take.size:
            V_out[:, idx[take]] = r["V"][:, take]
            iters_out[idx[take]] = np.asarray(r["iterations"])[take]
            conv_out[idx[take]] = True
        return bool(take.size)

    # ------------------------------------------------------------ CPU fallback
    def _cpu_solver(self):
        """``nr_klu.Solver`` for this topology, or None if the extension is not installed."""
        if self._cpu is None:
            try:
                from p3s import nr_klu  # type: ignore[attr-defined]
            except ImportError:
                try:
                    from p3s.cpp import nr_klu  # type: ignore[attr-defined]
                except ImportError:
                    self._cpu = False
                    return None
            t = self.topo
            self._cpu = nr_klu.Solver(t.Yp, t.Yj, t.Yx, np.ascontiguousarray(t.pvpq[: t.npvpq - t.npq]), t.pq)
        return self._cpu or None

    def _solve_on_cpu(self, cpu, ch: _Chunk, max_iter, tol):
        if ch.Yx is None and ch.stamp is None:
            S = ch.Sbus if ch.Sbus.ndim == 2 else np.repeat(ch.Sbus[:, None], ch.B, axis=1)
            return cpu.solve_batch(np.ascontiguousarray(S), ch.V0, max_iter=max_iter, tol=tol)
        Yx = ch.Yx if ch.Yx is not None else self.topo.Yx[:, None] - np.asarray(ch.stamp.todense())
        V0 = ch.V0 if ch.V0.ndim == 2 else np.repeat(ch.V0[:, None], ch.B, axis=1)
        pin = ch.pin if ch.pin is not None else np.zeros((self.topo.n, ch.B), dtype=np.uint8)
        return cpu.solve_batch_contingency(
            np.ascontiguousarray(np.abs(Yx)),
            np.ascontiguousarray(np.angle(Yx)),
            ch.Sbus,
            np.ascontiguousarray(V0),
            np.ascontiguousarray(pin),
            max_iter=max_iter,
            tol=tol,
        )

    # -------------------------------------------------------- one chunk (a batch)
    def _bind(self, B: int, ch: _Chunk) -> dict:
        """Size the device buffers for a chunk of ``B`` systems, upload its problem data
        and reset the Newton state. Returns the device pointers / kernel parameters."""
        t = self.topo
        n, m, S = t.n, t.m, B
        buf = self._buf
        b: dict = {"B": B, "S": S}
        for name in ("Vm", "Va", "cs", "sn"):
            b[name] = buf.get(name, n * S * _F64)
        b["V"] = buf.get("V", n * S * _C128)  # start voltage in, result out
        for name, size in (("nrm", _F64), ("alpha", _F64), ("dalpha", _F64), ("best_alpha", _F64), ("best_nrm", _F64)):
            b[name] = buf.get(name, S * size)
        for name in ("trial", "iters"):
            b[name] = buf.get(name, S * _I32)
        # status and precision vote sit back to back: one read-back per Newton pass
        b["status"] = buf.get("status", 2 * S * _I32)
        b["vote"] = b["status"] + S * _I32
        b["resid"] = buf.get("resid", S * 8)
        b["pstat"] = buf.get("pstat", 2 * S * _F64)
        if self.precision not in ("double", "mixed"):
            raise ValueError(f"precision must be 'double' or 'mixed', got {self.precision!r}")
        b["mixed"] = self.backend == "lean" and self.precision == "mixed"

        # linear solver: where the Jacobian, the residual and the Newton step live
        if self.backend == "lean":
            b["J"] = buf.get("val", self.plan.nnz_lu * S * _F64)
            b["fac32"] = buf.get("val32", self.plan.nnz_lu * S * 4) if b["mixed"] else 0
            b["R"] = buf.get("R", m * S * _F64)
            b["x"] = buf.get("y", m * S * _F64)
            b["sJ"] = b["sR"] = b["sX"] = (S, 1)
            b["lib"] = None
        else:
            lib = self._library_backend(B)
            b["J"], b["R"], b["x"] = int(lib.d_A_batch), int(lib.d_rhs_batch), int(lib.d_X_batch)
            b["sJ"], b["sR"], b["sX"] = (1, t.nnzJ), (1, m), (1, m)
            b["lib"] = lib

        # problem data
        def put(dev, a):
            if a.ndim == 1 or ch.col0 is None:
                cuda.memcpy_htod(dev, a)
            else:
                _copy_columns(dev, a, ch.col0, B, to_device=True)

        if ch.V0.ndim == 1:
            b["V0"], b["v0_shared"] = buf.get("V0s", n * _C128), 1
        else:
            b["V0"], b["v0_shared"] = b["V"], 0
        put(b["V0"], ch.V0)
        if ch.Sbus.ndim == 1:
            b["Ssp"], b["s_shared"] = buf.get("Ss", n * _C128), 1
        else:
            b["Ssp"], b["s_shared"] = buf.get("S", n * S * _C128), 0
        put(b["Ssp"], ch.Sbus)
        if ch.Yx is not None:
            b["Y"], b["y_shared"] = buf.get("Y", t.nnzY * S * _C128), 0
            put(b["Y"], ch.Yx)
        elif ch.stamp is not None:
            b["Y"], b["y_shared"] = buf.get("Y", t.nnzY * S * _C128), 0
            self._expand_Y(b, ch.stamp)
        else:
            b["Y"], b["y_shared"] = int(self.d_Ybase), 1
        if ch.pin is not None:
            b["pin"] = buf.get("pin", n * S)
            put(b["pin"], ch.pin)
        else:
            b["pin"] = 0

        cuda.memset_d32(b["status"], _ACTIVE, 2 * B)
        cuda.memset_d32(b["resid"], 0, 2 * B)
        self._k["init_state"].prepared_call(
            _grid(n * B), (_BLOCK, 1, 1), b["V0"], b["v0_shared"], b["Vm"], b["Va"], b["cs"], b["sn"], n, S, B
        )
        return b

    def _expand_Y(self, b: dict, stamp):
        """Per-system Ybus on the device: base values minus the sparse outage stamps."""
        t = self.topo
        B, S = b["B"], b["S"]
        self._k["expand_Y"].prepared_call(_grid(t.nnzY * B), (_BLOCK, 1, 1), int(self.d_Ybase), b["Y"], t.nnzY, S, B)
        coo = sp.coo_matrix(stamp)
        nd = coo.nnz
        if nd == 0:
            return
        d_k = self._buf.get("dk", nd * _I32)
        d_c = self._buf.get("dc", nd * _I32)
        d_v = self._buf.get("dv", nd * _C128)
        cuda.memcpy_htod(d_k, np.ascontiguousarray(coo.row, dtype=np.int32))
        cuda.memcpy_htod(d_c, np.ascontiguousarray(coo.col, dtype=np.int32))
        cuda.memcpy_htod(d_v, np.ascontiguousarray(coo.data, dtype=np.complex128))
        self._k["apply_Y_delta"].prepared_call(_grid(nd), (_BLOCK, 1, 1), d_k, d_c, d_v, b["Y"], nd, S)

    def _eval(self, b: dict, run_mask: int, j_mask: int):
        """Mismatch (and Jacobian for the statuses in ``j_mask``) of the systems in ``run_mask``."""
        t = self.topo
        self._k["eval_FJ"].prepared_call(
            _grid(t.n * b["B"]),
            (_BLOCK, 1, 1),
            int(self.d_Yp),
            int(self.d_Yj),
            int(self.d_Ydiag),
            b["Y"],
            b["y_shared"],
            b["Vm"],
            b["cs"],
            b["sn"],
            b["Ssp"],
            b["s_shared"],
            int(self.d_row_p),
            int(self.d_row_q),
            b["pin"],
            b["status"],
            run_mask,
            j_mask,
            int(self.d_posmap),
            b["J"],
            *b["sJ"],
            b["R"],
            *b["sR"],
            b["resid"],
            t.n,
            b["S"],
            b["B"],
        )

    def _decide(self, b: dict, tol: float, phase: int, it: int, line_search: bool) -> NDArray:
        """Per-system convergence / line-search / precision decision; returns the status
        array (the precision votes are kept in ``b["votes"]``)."""
        B = b["B"]
        self._k["newton_decide"].prepared_call(
            _grid(B),
            (_BLOCK, 1, 1),
            b["resid"],
            b["nrm"],
            b["alpha"],
            b["dalpha"],
            b["best_alpha"],
            b["best_nrm"],
            b["trial"],
            b["status"],
            b["iters"],
            b["vote"],
            tol,
            float(self.mixed_switch),
            int(line_search),
            int(b["mixed"]),
            phase,
            it,
            B,
        )
        both = np.empty(2 * B, dtype=np.int32)
        cuda.memcpy_dtoh(both, b["status"])  # the one blocking sync of a Newton pass
        b["votes"] = both[B:]
        return both[:B]

    def _step(self, b: dict, run_mask: int, full: bool):
        """V += dx (full Newton step) or V += dalpha * dx (line-search move)."""
        t = self.topo
        self._k["apply_step"].prepared_call(
            _grid(t.n * b["B"]),
            (_BLOCK, 1, 1),
            b["x"],
            *b["sX"],
            int(self.d_xva),
            int(self.d_xvm),
            b["status"],
            run_mask,
            0 if full else b["dalpha"],
            b["Vm"],
            b["Va"],
            b["cs"],
            b["sn"],
            t.n,
            b["S"],
            b["B"],
        )

    def _lean_factor_solve(self, b: dict, rcond_min: float, do_solve: bool = True, single: bool = False):
        """One launch of the lean LU kernel: double (in place in ``J``) or single (factors in
        ``fac32``). With mixed precision each launch only touches the systems that voted
        for its precision."""
        p = self.plan
        W = p.slots
        self._k["lean_factor_solve_f32" if single else "lean_factor_solve_f64"].prepared_call(
            ((b["B"] + 31) // 32, 1, 1),
            (32 * W, 1, 1),
            *(int(a) for a in self._d_plan),
            b["J"],
            b["fac32"] if single else b["J"],
            b["R"],
            b["x"],
            b["status"],
            b["vote"],
            b["pstat"],
            rcond_min,
            p.n_levels,
            p.n_blevels,
            W,
            b["S"],
            b["B"],
            1,
            int(do_solve),
            int(b["mixed"]),
        )

    def _linear_solve(self, b: dict, status: NDArray):
        """Newton step of the ACTIVE systems: J dx = R with the Jacobian eval_FJ just wrote."""
        if b["lib"] is not None:
            b["lib"].reset_refactor_device()
            b["lib"].batch_solve_device()
            return
        rcond_min = _RCOND_RATIO * self._rcond0
        active = status == _ACTIVE
        n_single = int(np.count_nonzero(active & (b["votes"] == 1))) if b["mixed"] else 0
        n_double = int(np.count_nonzero(active)) - n_single
        if n_single:
            self._lean_factor_solve(b, self._rcond_ratio_single * self._rcond0, single=True)
        # The double kernel also picks up the systems the single one just rejected, which
        # the host cannot know without a read-back -- so in mixed mode it always runs. It
        # costs next to nothing when no system asks for it.
        if n_double or n_single:
            self._lean_factor_solve(b, rcond_min)
        st = self.last_stats
        st["single_factorizations"] = st.get("single_factorizations", 0) + n_single
        st["double_factorizations"] = st.get("double_factorizations", 0) + n_double

    def _run_chunk(self, ch: _Chunk, max_iter: int, tol: float, handoff: int = 0, out=None) -> dict:
        """Solve ONE chunk: all of its systems advance together, each stops on its own.

        ``handoff`` > 0: stop as soon as at most that many systems (and at most 1/8 of the
        chunk) are still iterating and report them in ``handed_off`` -- the caller finishes
        them on the CPU. A handful of stragglers cannot fill the GPU: an iteration then
        costs the same latency-bound ~tens of ms as for a full block, while a CPU core
        needs ~1 ms per system. ``out = (V, col0)``: download the voltages into columns
        ``[col0, col0 + B)`` of the caller's ``(n, B_total)`` array.
        """
        t = self.topo
        B = ch.B
        b = self._bind(B, ch)
        # the in-place library backends lose the step when the next residual is written
        line_search = self.line_search and b["R"] != b["x"]
        A, BT, NJ = 1 << _ACTIVE, 1 << _BACKTRACK, 1 << _NEEDJ

        self._eval(b, A, A)
        status = self._decide(b, tol, 0, 0, line_search)
        it = 0
        tail = None
        while it < max_iter and (status == _ACTIVE).any():
            n_active = int(np.count_nonzero(status == _ACTIVE))
            if it > 0 and n_active <= handoff and 8 * n_active <= B:
                tail = np.nonzero(status == _ACTIVE)[0]
                break
            it += 1
            self._linear_solve(b, status)
            self._step(b, A, full=True)
            self._eval(b, A, A)
            status = self._decide(b, tol, 1, it, line_search)
            if (status == _BACKTRACK).any():
                # Full step rejected for some systems: halve their step (mismatch-only
                # trials), then rebuild their Jacobian at the accepted point.
                while (status == _BACKTRACK).any():
                    self._step(b, BT, full=False)
                    self._eval(b, BT, 0)
                    status = self._decide(b, tol, 2, it, line_search)
                self._step(b, NJ, full=False)
                self._eval(b, NJ, NJ)
                status = self._decide(b, tol, 3, it, line_search)

        self._k["finalize"].prepared_call(
            _grid(t.n * B), (_BLOCK, 1, 1), b["Vm"], b["cs"], b["sn"], b["V"], t.n, b["S"], B
        )
        if out is None:
            V = np.empty((t.n, B), dtype=np.complex128)
            cuda.memcpy_dtoh(V, b["V"])
        else:  # straight into the caller's result, columns [col0, col0 + B)
            V_full, col0 = out
            if V_full.shape[1] == B:
                cuda.memcpy_dtoh(V_full, b["V"])
            else:
                _copy_columns(b["V"], V_full, col0, B, to_device=False)
            V = V_full[:, col0 : col0 + B]
        iters = np.empty(B, dtype=np.int32)
        cuda.memcpy_dtoh(iters, b["iters"])
        return {"V": V, "iterations": iters, "converged": status == _CONVERGED, "handed_off": tail, "it": it}
