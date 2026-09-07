# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU N-1 contingency solver (fully resident polar path).

The GPU sibling of ``solver_cpp.py``. Reuses the *identical* backend-agnostic case
generator and returns the *identical* ``ContingencyResultTable``; only the
solver backend differs -- ``PolarNewtonSolverCUDA.solve_batch_contingency`` instead of
``nr_klu.Solver.solve_batch_contingency``. The per-case inputs (per-contingency Ybus
magnitude/angle, constant Sbus, per-case V0 + pin mask) are assembled exactly as in the
CPU path, so results match to solver tolerance.

See solver_cpp.py for the pin-and-flag served-mask semantics (same here).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from p3s.contingency.case_generator import (
    ContingencyBatch,
    ContingencyCaseGenerator,
)
from p3s.contingency.solver_cpp import ContingencyResultTable
from p3s.timeseries import dc_initial_voltage


def solve_contingencies_cuda(
    net,
    reslack_islands: bool = False,
    tol: float = 1e-8,
    max_iter: int = 30,
    init: str = "dc",
    max_chunk: int | None = None,
    backend: str = "cudss",
) -> ContingencyResultTable:
    """Solve every N-1 contingency in ``net`` on the GPU polar batch path.

    Signature and semantics mirror ``solve_contingencies_cpp`` (minus ``n_threads``,
    which is meaningless on the GPU). ``init`` picks the shared start voltage: "dc" seeds
    angles (needed for phase-shifting transformers), "flat" is the robust choice on very
    large meshed nets.
    """
    from p3s.cuda.nr_polar_solver import PolarNewtonSolverCUDA

    gen = ContingencyCaseGenerator(net, reslack_islands=reslack_islands)
    batch: ContingencyBatch = gen.build()
    n = batch.n_bus
    L = len(batch.cases)

    if L == 0:
        empty2 = np.empty((n, 0))
        return ContingencyResultTable(
            groups=[],
            V=empty2.astype(np.complex128),
            served=empty2.astype(bool),
            converged=np.empty(0, bool),
            iterations=np.empty(0, int),
        )

    solver = PolarNewtonSolverCUDA(
        batch.Yp,
        batch.Yj,
        np.ascontiguousarray(batch.Yx_base, dtype=np.complex128),
        np.ascontiguousarray(batch.pv, dtype=np.int32),
        np.ascontiguousarray(batch.pq, dtype=np.int32),
        backend=backend,
    )
    # Cap the per-chunk batch at the GPU's RF-solve amortization sweet spot (memory permits
    # bigger, but past the sweet spot per-column cost rises; ~128 on the RTX A500). None =
    # memory-budget only, right for large GPUs.
    solver.max_chunk = max_chunk

    # Start voltage (one vector reused across all cases) -- identical to solver_cpp.
    if init == "dc":
        v_start = dc_initial_voltage(gen._npf)
    elif init == "flat":
        v_start = gen._npf._initial_voltage.copy()
    else:
        raise ValueError(f"init must be 'dc' or 'flat', got {init!r}")

    # Per-case COMPLEX Ybus values (nnz, L). The magnitude/angle conversion + transpose is
    # done ON THE GPU (solve_batch_contingency_cx -> yx_to_polar kernel), avoiding ~seconds
    # of single-threaded host np.abs/np.angle/.T on large nets.
    Yx_mat: NDArray = np.ascontiguousarray(batch.Yx_matrix, dtype=np.complex128)  # (nnz, L)
    Sbus: NDArray = np.ascontiguousarray(gen._npf._sBus, dtype=np.complex128)

    V0: NDArray = np.empty((n, L), dtype=np.complex128)
    pin: NDArray = np.zeros((n, L), dtype=np.uint8)
    for c, case in enumerate(batch.cases):
        v0 = v_start.copy()
        for bus, _kind in case.pinned_refs:
            v0[bus] = np.abs(v0[bus]) + 0.0j
            pin[bus, c] = 1
        unserved = ~case.served
        pin[unserved, c] = 1
        v0[unserved] = 1.0 + 0.0j
        V0[:, c] = v0

    res = solver.solve_batch_contingency_cx(
        Yx_mat,
        Sbus,
        np.ascontiguousarray(V0, dtype=np.complex128),
        np.ascontiguousarray(pin, dtype=np.uint8),
        max_iter=max_iter,
        tol=tol,
    )

    V = res["V"].copy()  # (n, L)
    served = np.stack([c.served for c in batch.cases], axis=1)  # (n, L)
    V[~served] = np.nan
    return ContingencyResultTable(
        groups=batch.groups,
        V=V,
        served=served,
        converged=np.asarray(res["converged"], dtype=bool),
        iterations=np.asarray(res["iterations"], dtype=int),
    )
