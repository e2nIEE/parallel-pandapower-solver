# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU N-1 contingency solver (C++/KLU ``solve_batch_contingency``).

Ties the backend-agnostic case generator to the batched C++ solver. Each
contingency is the base Ybus with one outage group's branch stamps removed.
A values-only change on a shared sparsity pattern, so the KLU symbolic factorization is
built once and reused across the whole batch (the same amortization the time-series path
uses).

Pin-and-flag:
  * Original slack buses are already outside the Newton variable set, so they need no pinning.
  * Re-slack generator buses (optional island references) ARE pinned: held at their gen
    voltage setpoint, so they act as that island's reference.
  * Unserved buses are pinned at a placeholder (1+0j) so the Jacobian stays non-singular;
    their result voltage is overwritten with NaN afterwards via the ``served`` mask.

Result table (mirrors the ground-truth oracle):
  * ``vm`` / ``va`` : (n_bus, L) float, NaN at unserved buses
  * ``V``           : (n_bus, L) complex, NaN at unserved buses
  * ``served``      : (n_bus, L) bool
  * ``converged``   : (L,) bool
  * ``iterations``  : (L,) int
  * ``groups``      : list[str], column order
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    from p3s import nr_klu  # type: ignore[attr-defined]
except ImportError:
    from p3s.cpp import nr_klu  # type: ignore[attr-defined]
from p3s.contingency.case_generator import (
    ContingencyBatch,
    ContingencyCaseGenerator,
)
from p3s.timeseries import dc_initial_voltage


@dataclass
class ContingencyResultTable:
    groups: list
    V: NDArray  # (n_bus, L) complex, NaN at unserved
    served: NDArray  # (n_bus, L) bool
    converged: NDArray  # (L,) bool
    iterations: NDArray  # (L,) int

    @property
    def vm(self) -> NDArray:
        return np.abs(self.V)

    @property
    def va(self) -> NDArray:
        return np.degrees(np.angle(self.V))


def solve_contingencies_cpp(
    net, reslack_islands: bool = False, tol: float = 1e-8, max_iter: int = 30, n_threads: int = 0, init: str = "dc"
) -> ContingencyResultTable:
    """Solve every N-1 contingency in ``net`` on the CPU C++/KLU batch path.

    init: {"dc", "flat"}, default "dc"
        Start-voltage strategy (one vector reused across all cases). "dc" runs p3s's
        DC power flow to seed angles -- needed for phase-shifting transformers, where a
        flat start diverges. "flat" uses 1 pu / 0 deg with the known voltage setpoints.
        Prefer "flat" on very large meshed nets (e.g. case9241pegase) where p3s's
        simplified DC model produces inaccurate angles that fall outside Newton's basin
        and even fail the first Jacobian factorization. See ``p3s-dc-init-bug``.
    """
    gen = ContingencyCaseGenerator(net, reslack_islands=reslack_islands)
    batch: ContingencyBatch = gen.build()
    n = batch.n_bus
    L = len(batch.cases)

    # Shared symbolic factorization: one Solver for the whole batch.
    solver = nr_klu.Solver(
        batch.Yp,
        batch.Yj,
        np.ascontiguousarray(batch.Yx_base, dtype=np.complex128),
        np.ascontiguousarray(batch.pv, dtype=np.int32),
        np.ascontiguousarray(batch.pq, dtype=np.int32),
    )

    if L == 0:
        empty2 = np.empty((n, 0))
        return ContingencyResultTable(
            groups=[],
            V=empty2.astype(np.complex128),
            served=empty2.astype(bool),
            converged=np.empty(0, bool),
            iterations=np.empty(0, int),
        )

    # Start voltage (one vector reused across all cases). "dc" seeds angles for
    # phase-shifting transformers; "flat" is the robust choice on very large nets where
    # p3s's DC model is inaccurate (see the `init` docstring). Both carry the
    # ext_grid + pv/gen voltage setpoints (from _initial_voltage).
    if init == "dc":
        v_start = dc_initial_voltage(gen._npf)
    elif init == "flat":
        v_start = gen._npf._initial_voltage.copy()
    else:
        raise ValueError(f"init must be 'dc' or 'flat', got {init!r}")

    # per-case Ybus values (magnitude/angle) and pin/V0 matrices, columns = cases
    Yx_mat = batch.Yx_matrix  # (nnz, L) complex
    Yx_mag = np.ascontiguousarray(np.abs(Yx_mat), dtype=np.float64)
    Yx_ang = np.ascontiguousarray(np.angle(Yx_mat), dtype=np.float64)

    Sbus = np.ascontiguousarray(gen._npf._sBus, dtype=np.complex128)  # constant per case

    V0: NDArray = np.empty((n, L), dtype=np.complex128)
    pin: NDArray = np.zeros((n, L), dtype=np.uint8)
    for c, case in enumerate(batch.cases):
        v0 = v_start.copy()  # already holds ext_grid + pv/gen voltage setpoints
        # pin re-slack reference generators at (gen vm setpoint) angle 0 -- the island's
        # angle reference is arbitrary, so we match the slack convention (va=0) rather
        # than leaving the DC-init angle, which would rotate the whole island by a
        # constant (physically identical but differs from pandapower's va=0 ext_grid).
        for bus, _kind in case.pinned_refs:
            v0[bus] = np.abs(v0[bus]) + 0.0j
            pin[bus, c] = 1
        # pin unserved buses at a placeholder; overwritten with NaN after the solve
        unserved = ~case.served
        pin[unserved, c] = 1
        v0[unserved] = 1.0 + 0.0j
        V0[:, c] = v0

    res = solver.solve_batch_contingency(
        Yx_mag,
        Yx_ang,
        Sbus,
        np.ascontiguousarray(V0, dtype=np.complex128),
        np.ascontiguousarray(pin, dtype=np.uint8),
        max_iter=max_iter,
        tol=tol,
        n_threads=n_threads,
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
