# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-processing: per-contingency bus voltage-band violations.

The contingency solvers return a ``ContingencyResultTable`` whose ``V`` is ``(n_bus, L)``
(rows in positional ``net.bus`` order, one column per contingency). The second N-1
criterion next to branch loading (see ``line_loading.find_overloads``) is the voltage
band: in every contingency, each bus magnitude must stay inside ``[vm_min, vm_max]``.

This module screens the whole table at once and returns a SPARSE list of violations, in
the same spirit as ``OverloadReport``: a clean study returns almost nothing regardless of
how many contingencies were swept.

What is never reported as a voltage violation:

* **Non-converged columns.** The solver leaves Newton's last iterate in ``V``; it is not a
  solution, so the whole column is skipped.
* **Unserved (islanded) buses.** They carry ``NaN`` voltage. Loss of supply is its own
  finding (``res.served``), not a band violation.
* **Out-of-service buses** (``net.bus.in_service == False``). The solver still reports them
  as served, at a placeholder voltage (e.g. an attached generator's setpoint), which would
  otherwise show up as a spurious violation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VoltageViolationReport:
    """Buses outside their voltage band, as a flat list of violations.

    All arrays share one length ``n_violations`` and are sorted by descending
    ``deviation_pu``, so ``rep.bus[0]`` etc. is always the worst violation found.
    """

    bus: np.ndarray  # (n_viol,) pandapower bus index (net.bus row label)
    group: np.ndarray  # (n_viol,) contingency name that caused it
    case: np.ndarray  # (n_viol,) column index into res.groups
    kind: np.ndarray  # (n_viol,) "low" (below vm_min) / "high" (above vm_max)
    vm_pu: np.ndarray  # (n_viol,) float, the offending magnitude
    limit_pu: np.ndarray  # (n_viol,) float, the bound that was violated
    deviation_pu: np.ndarray  # (n_viol,) float > 0, distance beyond that bound

    def __len__(self) -> int:
        return len(self.bus)

    @property
    def n_violations(self) -> int:
        return len(self.bus)

    def to_dataframe(self):
        """pandas DataFrame of the violations (one row each), worst first."""
        import pandas as pd

        return pd.DataFrame(
            {
                "bus": self.bus,
                "group": self.group,
                "case": self.case,
                "kind": self.kind,
                "vm_pu": self.vm_pu,
                "limit_pu": self.limit_pu,
                "deviation_pu": self.deviation_pu,
            }
        )

    def worst_per_bus(self):
        """DataFrame with only each bus's single worst violation, worst first."""
        df = self.to_dataframe()
        if df.empty:
            return df
        return (
            df.sort_values("deviation_pu", ascending=False)
            .drop_duplicates(subset=["bus"], keep="first")
            .reset_index(drop=True)
        )


def _resolve_bound(net, value, column: str, n_bus: int) -> np.ndarray:
    """(n_bus,) float bound: explicit ``value`` wins, else ``net.bus[column]``, else NaN.

    NaN means "no bound on this side for this bus", matching pandapower's convention for
    an unset ``min_vm_pu`` / ``max_vm_pu``.
    """
    if value is None:
        if column in net.bus.columns:
            return net.bus[column].to_numpy(dtype=float)
        return np.full(n_bus, np.nan)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_bus, float(arr))
    if arr.shape != (n_bus,):
        raise ValueError(f"{column} must be a scalar or have shape ({n_bus},) in net.bus order, got shape {arr.shape}")
    return arr


def find_voltage_violations(net, res, vm_min_pu=None, vm_max_pu=None, tol_pu: float = 1e-9) -> VoltageViolationReport:
    """Every (bus, contingency) pair whose voltage magnitude leaves its band.

    Parameters
    ----------
    net : pandapowerNet
        The SAME net handed to the contingency solver.
    res : ContingencyResultTable
        Result of ``solve_contingencies_cpp`` / ``solve_contingencies_cuda``.
    vm_min_pu, vm_max_pu : float or array-like of shape (n_bus,), optional
        Lower / upper band in pu. A scalar applies to every bus; an array is per bus in
        positional ``net.bus`` order. When omitted, ``net.bus["min_vm_pu"]`` /
        ``net.bus["max_vm_pu"]`` are used. A NaN bound (or a missing column) leaves that
        side unchecked for that bus. N-1 studies often use a wider emergency band than the
        operating band stored on the net -- pass it explicitly in that case.
    tol_pu : float, default 1e-9
        A bus is flagged only when it is outside its band by MORE than this. The default
        only absorbs floating-point noise, so a PV bus held exactly at a setpoint equal to
        the bound is not reported; it is far below any engineering significance.

    Returns
    -------
    VoltageViolationReport
        One entry per violation, sorted worst-first by ``deviation_pu``. Empty when every
        bus stays in band. See the module docstring for what is never reported.
    """
    if not np.isfinite(tol_pu) or tol_pu < 0:
        raise ValueError(f"tol_pu must be finite and >= 0, got {tol_pu!r}")

    V = np.asarray(res.V)
    if V.ndim != 2:
        raise ValueError(f"res.V must be 2-D (n_bus, L), got shape {V.shape}")
    n_bus, L = V.shape
    if len(net.bus) != n_bus:
        raise ValueError(f"net has {len(net.bus)} buses but res.V has {n_bus} rows")

    vmin = _resolve_bound(net, vm_min_pu, "min_vm_pu", n_bus)
    vmax = _resolve_bound(net, vm_max_pu, "max_vm_pu", n_bus)
    if np.isnan(vmin).all() and np.isnan(vmax).all():
        raise ValueError("no voltage band: pass vm_min_pu / vm_max_pu or set net.bus min_vm_pu / max_vm_pu")
    with np.errstate(invalid="ignore"):
        inverted = vmin > vmax
    if inverted.any():
        bad = net.bus.index.to_numpy()[inverted]
        raise ValueError(f"vm_min_pu > vm_max_pu at bus(es) {bad[:10].tolist()}")

    vm = np.abs(V)  # (n_bus, L)
    # Exclude what is not a solved, in-service bus voltage (see module docstring).
    # Setting it to NaN makes both comparisons below False.
    vm[:, ~np.asarray(res.converged, dtype=bool)] = np.nan
    vm[~net.bus["in_service"].to_numpy(dtype=bool), :] = np.nan

    # NaN bounds and NaN voltages both compare False, so neither is ever flagged.
    with np.errstate(invalid="ignore"):
        low = vm < (vmin - tol_pu)[:, None]
        high = vm > (vmax + tol_pu)[:, None]

    bus_index = net.bus.index.to_numpy()
    groups = np.asarray(res.groups, dtype=object)
    lr, lc = np.nonzero(low)
    hr, hc = np.nonzero(high)
    rows = np.concatenate([lr, hr])
    cols = np.concatenate([lc, hc])
    vm_v = vm[rows, cols]
    limit = np.concatenate([vmin[lr], vmax[hr]])
    kind = np.array(["low"] * lr.size + ["high"] * hr.size, dtype=object)

    order = np.argsort(np.abs(vm_v - limit), kind="stable")[::-1]  # worst first
    rows, cols = rows[order], cols[order]
    limit = limit[order]
    vm_v = vm_v[order]
    return VoltageViolationReport(
        bus=bus_index[rows],
        group=groups[cols],
        case=cols,
        kind=kind[order],
        vm_pu=vm_v,
        limit_pu=limit,
        deviation_pu=np.abs(vm_v - limit),
    )
