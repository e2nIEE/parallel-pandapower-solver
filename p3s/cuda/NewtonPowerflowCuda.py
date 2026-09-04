# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

import logging

import numpy as np
import pycuda.driver as cuda
from numpy.typing import NDArray
from pandapower import LoadflowNotConverged, pandapowerNet

from p3s.cuda import _ctx  # noqa: F401 (retains the CUDA primary context; cuDSS-safe)
from p3s.cuda.cusolver_rf_batch import CusolverRfBatch
from p3s.NewtonPowerflow import NewtonPowerflow
from p3s.timeseries import build_sbus_matrix, dc_initial_voltage

logger = logging.getLogger(__name__)


def _sort_csr_pattern(Jp: NDArray, Jj: NDArray):
    """Return (sorted_Jj, perm) so that columns within each CSR row are ascending.

    p3s's ``create_J`` emits entries in Ybus order, which is generally *not*
    column-sorted within a row, but cuSolver's host LU requires sorted columns.
    ``perm`` is the gather index that reorders any value array (Jx) aligned to the
    original pattern into the sorted layout: ``Jx_sorted = Jx[perm]``. The pattern is
    fixed across the whole solve, so this is computed once.
    """
    Jp = np.ascontiguousarray(Jp, dtype=np.int32)
    Jj = np.ascontiguousarray(Jj, dtype=np.int32)
    perm: NDArray = np.arange(len(Jj), dtype=np.int64)
    Jj_sorted = Jj.copy()
    for r in range(len(Jp) - 1):
        s, e = Jp[r], Jp[r + 1]
        order = np.argsort(Jj[s:e], kind="stable")
        perm[s:e] = s + order
        Jj_sorted[s:e] = Jj[s + order]
    return Jj_sorted, perm


class NewtonPowerflowCUDA(NewtonPowerflow):
    """
    CUDA-accelerated variant of the Newton-Raphson powerflow loop.

    Two GPU back-ends live on this class:

    * The legacy ``*_cuda`` methods (``calculate_cuda``, ``calculate_timeseries_cuda``)
      assemble the Jacobian in Python and offload only the sparse linear solve to the
      deprecated cuSolverRf batch path. Kept for old CUDA-11 environments.

    * The ``*_cudss`` methods (``calculate_cudss``, ``calculate_timeseries_cudss``,
      ``calculate_contingency_cudss``) drive the modern, fully device-resident polar
      solver ``p3s.cuda.nr_polar_solver.PolarNewtonSolverCUDA`` (cuDSS batched
      direct solve, Jacobian assembled on the GPU). These are the easy, ``net``-driven
      wrappers -- the CUDA counterparts of ``NewtonPowerflowCpp`` around
      ``nr_klu.Solver`` -- and are the preferred entry points for batched
      (time-series / N-1 contingency) calculations.

    Requires a working CUDA environment. Falls back by raising if the GPU is unavailable.
    """

    def __init__(self, net: pandapowerNet):
        super().__init__(net)
        # Cached device-resident polar solver (cuDSS back-end). The symbolic
        # factorization depends only on the Ybus sparsity pattern + pv/pq split, so a
        # single solver is reused across calculate_*_cudss calls while the topology is
        # unchanged. Invalidated (rebuilt) when the Ybus structure (nnz) changes.
        self._cudss_solver = None
        self._cudss_solver_nnz = None

    def calculate_cuda(self, net: pandapowerNet, tolerance: float = 1e-5, max_iterations: int = 30, **kwargs):
        """Single operating-point Newton-Raphson using the GPU cuSolverRf solver
        (batch size 1).

        This is the correctness stepping-stone toward the batched time-series solver:
        it reuses the CPU Jacobian assembly / mismatch / voltage-update logic from the
        base class and only swaps the sparse linear solve for ``CusolverRfBatch``. The
        Jacobian sparsity pattern is fixed across Newton iterations, so the symbolic
        factorization is built once and only values are refreshed each iteration.
        """
        voltage = self._initial_voltage.copy()
        pvpq_obj = self.pf_objects["PVPQ"]

        mismatch = self._calc_mismatch(self._sBus, voltage)

        # -- build the fixed (sorted) sparsity pattern once from the first Jacobian --
        Jx0, Jp, Jj = pvpq_obj.create_J(voltage)
        Jp = np.ascontiguousarray(Jp, dtype=np.int32)
        Jj0 = np.ascontiguousarray(Jj, dtype=np.int32)
        Jj_sorted, perm = _sort_csr_pattern(Jp, Jj0)

        solver = CusolverRfBatch(Jp, Jj_sorted, batch_size=1)
        solver.symbolic_setup(np.ascontiguousarray(Jx0, dtype=np.float64)[perm])

        i = 0
        converged = False
        while not converged:
            i += 1
            if i > max_iterations:
                raise LoadflowNotConverged(f"Loadflow did not converge in {max_iterations} iterations.")

            Jx, _Jp, _Jj = pvpq_obj.create_J(voltage)
            Jx_sorted = np.ascontiguousarray(Jx, dtype=np.float64)[perm]

            # J * dx = -mismatch  -> solver returns dx for rhs = -mismatch
            dx = solver.solve(Jx_sorted, (-mismatch).reshape(1, -1))[0]

            voltage = self._calc_results_dx(dx, voltage)
            mismatch = self._calc_mismatch(self._sBus, voltage)
            converged = np.linalg.norm(mismatch, np.inf) < tolerance

        return voltage

    # ------------------------------------------------------------------ cuDSS path
    def _get_cudss_solver(self, backend: str = "cudss", reorder: str = "symrcm"):
        """Build (or reuse) the device-resident polar solver for this grid topology.

        Mirrors the cached-``nr_klu.Solver`` pattern in ``NewtonPowerflowCpp``: the
        symbolic factorization depends only on the Ybus sparsity pattern + pv/pq split, so
        the solver is created once and reused across ``calculate_*_cudss`` calls. Unlike
        ``nr_klu.Solver`` there is no in-place ``update_Y``. ``PolarNewtonSolverCUDA``
        bakes the Ybus polar values into its topology at construction, so a change in the
        Ybus structure (nnz) rebuilds the solver.
        """
        from p3s.cuda.nr_polar_solver import PolarNewtonSolverCUDA

        Yp = np.ascontiguousarray(self._YBus.indptr, dtype=np.int32)
        Yj = np.ascontiguousarray(self._YBus.indices, dtype=np.int32)
        Yx = np.ascontiguousarray(self._YBus.data, dtype=np.complex128)
        pv_i = np.ascontiguousarray(self.busses["pv"], dtype=np.int32)
        pq_i = np.ascontiguousarray(self.busses["pq"], dtype=np.int32)

        nnz = Yj.shape[0]
        if self._cudss_solver is None or self._cudss_solver_nnz != nnz:
            self._cudss_solver = PolarNewtonSolverCUDA(Yp, Yj, Yx, pv_i, pq_i, reorder=reorder, backend=backend)  # type: ignore[assignment]
            self._cudss_solver_nnz = nnz
        return self._cudss_solver

    def calculate_cudss(
        self,
        net: pandapowerNet,
        tolerance: float = 1e-8,
        max_iterations: int = 30,
        init: str = "dc",
        backend: str = "cudss",
        **kwargs,
    ) -> NDArray:
        """Single operating-point Newton-Raphson via the device-resident polar solver.

        The GPU counterpart of ``NewtonPowerflowCpp.calculate``. A batch of size one on
        ``PolarNewtonSolverCUDA.solve_batch``. Returns the converged complex voltage
        vector ``(n_bus,)``.
        """
        if init == "dc":
            v_init = dc_initial_voltage(self)
        elif init == "flat":
            v_init = self._initial_voltage.copy()
        else:
            raise ValueError(f"init must be 'dc' or 'flat', got {init!r}")

        solver = self._get_cudss_solver(backend=backend)
        result = solver.solve_batch(
            np.ascontiguousarray(self._sBus, dtype=np.complex128).reshape(-1, 1),
            np.ascontiguousarray(v_init, dtype=np.complex128),
            max_iter=max_iterations,
            tol=tolerance,
        )
        if not bool(result["converged"][0]):
            raise LoadflowNotConverged(
                f"Loadflow did not converge in {max_iterations} iterations (reached {int(result['iterations'][0])})."
            )
        return result["V"][:, 0]

    def calculate_timeseries_cudss(
        self,
        net: pandapowerNet,
        timeseries: dict[tuple[str, str], NDArray],
        tolerance: float = 1e-8,
        max_iterations: int = 30,
        init: str = "dc",
        backend: str = "cudss",
        max_chunk: int | None = None,
    ) -> NDArray:
        """Batched (time-series) Newton-Raphson via the device-resident polar solver.

        The GPU counterpart of ``NewtonPowerflowCpp.calculate_timeseries_cpp``: every time
        step is an independent Newton solve on the *same* grid topology, so the symbolic
        factorization is built once (cached solver) and the whole batch is solved on the
        GPU by ``PolarNewtonSolverCUDA.solve_batch`` (cuDSS). The same Sbus delta matrix +
        DC-init are reused (``p3s.timeseries``), so results match the C++ / CPU paths
        and pandapower per step.

        Parameters
        ----------
        timeseries: dict[(element, var) -> (n_element, T) array]
            Per-element time-series of ``p_mw`` (always) and ``q_mvar`` (optional); see
            ``build_sbus_matrix``.
        init: {"dc", "flat"}, default "dc"
            Shared start-voltage strategy (see ``dc_initial_voltage``); prefer "flat" on
            very large meshed nets where p3s's DC model is inaccurate.
        max_chunk: int, optional
            Hard cap on the per-chunk batch size (columns solved as one cuDSS batch),
            independent of the GPU memory budget. ``None`` = memory-budget only.

        Returns
        -------
        voltages : complex128 (n_bus, T) -- one converged voltage vector per time step.
        """
        sbus_matrix = self._build_sbus_matrix(net, timeseries)  # (n_bus, T)
        T = sbus_matrix.shape[1]

        if init == "dc":
            v_init = dc_initial_voltage(self)
        elif init == "flat":
            v_init = self._initial_voltage.copy()
        else:
            raise ValueError(f"init must be 'dc' or 'flat', got {init!r}")

        solver = self._get_cudss_solver(backend=backend)
        solver.max_chunk = max_chunk
        result = solver.solve_batch(
            np.ascontiguousarray(sbus_matrix, dtype=np.complex128),
            np.ascontiguousarray(v_init, dtype=np.complex128),
            max_iter=max_iterations,
            tol=tolerance,
        )
        if not bool(np.all(result["converged"])):
            n_bad = int((~result["converged"]).sum())
            raise LoadflowNotConverged(
                f"cuDSS batch did not converge for {n_bad} of {T} time steps in {max_iterations} iterations."
            )
        return result["V"]  # (n_bus, T)

    def calculate_contingency_cudss(
        self,
        net: pandapowerNet,
        reslack_islands: bool = False,
        tolerance: float = 1e-8,
        max_iterations: int = 30,
        init: str = "dc",
        backend: str = "cudss",
        max_chunk: int | None = None,
    ):
        """Batched N-1 contingency analysis via the device-resident polar solver.

        Thin ``net``-driven wrapper over ``p3s.contingency.solve_contingencies_cuda``
        (which drives ``PolarNewtonSolverCUDA.solve_batch_contingency_cx``). Every
        contingency is the base Ybus with one outage group's branch stamps removed. A
        values-only change on a shared sparsity pattern -- so the symbolic factorization is
        built once and reused across the whole batch, exactly like the time-series path.

        Parameters mirror ``solve_contingencies_cuda`` (``n_threads`` is meaningless on the
        GPU). Returns a ``ContingencyResultTable`` (``.V``/``.vm``/``.va`` of shape
        ``(n_bus, L)``, plus per-case ``.converged`` / ``.iterations`` / ``.groups``).
        """
        from p3s.contingency.solver_cuda import solve_contingencies_cuda

        return solve_contingencies_cuda(
            net,
            reslack_islands=reslack_islands,
            tol=tolerance,
            max_iter=max_iterations,
            init=init,
            max_chunk=max_chunk,
            backend=backend,
        )

    def _dc_initial_voltage(self) -> NDArray:
        """DC-power-flow-initialized start voltage (see p3s.timeseries)."""
        return dc_initial_voltage(self)

    def _build_sbus_matrix(self, net: pandapowerNet, timeseries) -> NDArray:
        """Per-bus, per-timestep complex injection matrix (see p3s.timeseries)."""
        return build_sbus_matrix(self, net, timeseries)

    def calculate_timeseries_cuda(
        self,
        net: pandapowerNet,
        timeseries: dict[tuple[str, str], NDArray],
        tolerance: float = 1e-5,
        max_iterations: int = 30,
        batch_size: int | None = None,
        reorder: str = "symrcm",
    ):
        """Batched time-series Newton-Raphson on the GPU via cuSolverRf.

        Every time step is an independent Newton problem on the *same* grid, so all
        Jacobians share one CSR sparsity pattern. We factor that pattern symbolically
        once, then per Newton iteration refactor, and solve a whole batch of time steps
        at once on the GPU.

        Returns ``voltages`` of shape (n_bus, T) -- one converged complex voltage vector
        per time step.
        """
        pvpq_obj = self.pf_objects["PVPQ"]
        sbus_matrix = self._build_sbus_matrix(net, timeseries)
        n_bus, T = sbus_matrix.shape

        # DC-initialized start (constant across time steps); flat start diverges on
        # large nets like pegase 9241.
        v_init = self._dc_initial_voltage()

        # -- fixed (sorted) sparsity pattern from a representative Jacobian --
        Jx0, Jp, Jj = pvpq_obj.create_J(v_init)
        Jp = np.ascontiguousarray(Jp, dtype=np.int32)
        Jj_sorted, perm = _sort_csr_pattern(Jp, np.ascontiguousarray(Jj, dtype=np.int32))
        nnz = len(Jj_sorted)
        n = len(Jp) - 1

        # -- choose batch size from free GPU memory if not given --
        if batch_size is None:
            free_mem, _ = cuda.mem_get_info()
            # per batch member: nnz Jx (8B) + n rhs (8B) + ~2n cusolverRf scratch (8B)
            per_member = (nnz + 3 * n) * 8
            budget = int(free_mem * 0.6)
            batch_size = max(1, min(T, budget // per_member))
        logger.info(f"Batched timeseries: T={T}, batch_size={batch_size}, n={n}, nnz={nnz}")

        voltages = np.empty((n_bus, T), dtype=np.complex128)

        # process time steps in chunks of `batch_size`
        for start in range(0, T, batch_size):
            stop = min(start + batch_size, T)
            B = stop - start
            sbus_chunk = sbus_matrix[:, start:stop]  # (n_bus, B)

            # per-timestep voltage state (DC-initialized start), and mismatch
            V = np.repeat(v_init[:, None], B, axis=1)  # (n_bus, B)
            mismatch = np.empty((B, n), dtype=np.float64)
            for c in range(B):
                mismatch[c] = self._calc_mismatch(sbus_chunk[:, c], V[:, c])

            solver = CusolverRfBatch(Jp, Jj_sorted, batch_size=B, reorder=reorder)
            solver.symbolic_setup(np.ascontiguousarray(Jx0, dtype=np.float64)[perm])

            converged_mask = np.zeros(B, dtype=bool)
            Jx_batch = np.zeros((B, nnz), dtype=np.float64)
            rhs_batch = np.zeros((B, n), dtype=np.float64)

            it = 0
            while not converged_mask.all():
                it += 1
                if it > max_iterations:
                    raise LoadflowNotConverged(
                        f"timeseries chunk [{start}:{stop}] did not converge in "
                        f"{max_iterations} iterations ({(~converged_mask).sum()} left)"
                    )

                # assemble Jacobian values + rhs for each still-active time step
                for c in range(B):
                    if converged_mask[c]:
                        continue
                    Jx_c, _, _ = pvpq_obj.create_J(V[:, c])
                    Jx_batch[c] = np.ascontiguousarray(Jx_c, dtype=np.float64)[perm]
                    rhs_batch[c] = -mismatch[c]

                dx_batch = solver.solve(Jx_batch, rhs_batch)  # (B, n)

                for c in range(B):
                    if converged_mask[c]:
                        continue
                    V[:, c] = self._calc_results_dx(dx_batch[c], V[:, c])
                    mismatch[c] = self._calc_mismatch(sbus_chunk[:, c], V[:, c])
                    if np.linalg.norm(mismatch[c], np.inf) < tolerance:
                        converged_mask[c] = True

            voltages[:, start:stop] = V

            # Free solver device memory before next chunk
            solver.free()

        return voltages
