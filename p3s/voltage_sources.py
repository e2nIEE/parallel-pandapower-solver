# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""ext_grid and gen as one table of voltage-controlling units.

pandapower does the same internally: both end up in ``ppc["gen"]``, and a bus is the
reference bus when any unit on it is a slack. Here an ext_grid is simply a unit with
``slack=True``, no scheduled P and an angle set point; a ``gen`` with ``slack=True`` (as
cim2pp writes the CGMES slack) is a reference unit too. Both solvers build their bus types,
start voltage and gen/ext_grid results from this one table.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandapower import pandapowerNet

UNIT_COLUMNS = ["et", "idx", "bus", "slack", "p_mw", "vm_pu", "va_degree", "min_q_mvar", "max_q_mvar", "slack_weight"]


def _column(df: pd.DataFrame, name: str, default: float) -> NDArray:
    if name not in df:
        return np.full(len(df), default)
    return df[name].to_numpy(dtype=float, na_value=default)


def voltage_sources(net: pandapowerNet, lookup: pd.Series) -> pd.DataFrame:
    """In-service ext_grids and gens as ONE table, ext_grids first.

    Order matters: where an ext_grid and a gen share a bus, the ext_grid's voltage set point
    wins (it comes first), as in pandapower. Out-of-service units are left out entirely --
    they neither define a bus type nor inject power -- and so are units on unsupplied buses.
    ``bus`` is the Ybus row (``lookup`` maps pandapower bus index -> Ybus row, -1 = unsupplied).
    """
    parts = []
    if "ext_grid" in net and len(net.ext_grid):
        eg = net.ext_grid[net.ext_grid.in_service.astype(bool)]
        parts.append(
            pd.DataFrame(
                {
                    "et": "ext_grid",
                    "idx": eg.index,
                    "bus": lookup[eg.bus].to_numpy(dtype=int),
                    "slack": True,
                    "p_mw": 0.0,
                    "vm_pu": eg.vm_pu.to_numpy(dtype=float),
                    "va_degree": eg.va_degree.to_numpy(dtype=float),
                    "min_q_mvar": _column(eg, "min_q_mvar", -np.inf),
                    "max_q_mvar": _column(eg, "max_q_mvar", np.inf),
                    "slack_weight": _column(eg, "slack_weight", 1.0),
                }
            )
        )
    if "gen" in net and len(net.gen):
        g = net.gen[net.gen.in_service.astype(bool)]
        slack = g.slack.fillna(False).to_numpy(dtype=bool) if "slack" in g else False
        parts.append(
            pd.DataFrame(
                {
                    "et": "gen",
                    "idx": g.index,
                    "bus": lookup[g.bus].to_numpy(dtype=int),
                    "slack": slack,
                    "p_mw": (g.p_mw * g.scaling).to_numpy(dtype=float),
                    "vm_pu": g.vm_pu.to_numpy(dtype=float),
                    "va_degree": 0.0,  # a gen has no angle set point; a slack gen is the 0 deg reference
                    "min_q_mvar": _column(g, "min_q_mvar", -np.inf),
                    "max_q_mvar": _column(g, "max_q_mvar", np.inf),
                    "slack_weight": _column(g, "slack_weight", 0),
                }
            )
        )
    if not parts:
        return pd.DataFrame({c: [] for c in UNIT_COLUMNS}).astype({"bus": int, "slack": bool})
    units = pd.concat(parts, ignore_index=True)
    return units[units.bus >= 0].reset_index(drop=True)


def bus_types(units: pd.DataFrame, n_bus: int, initial_voltage: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """ref / pv / pq Ybus rows from the unit table, and the set points written into ``initial_voltage``.

    A bus with any slack unit is a reference bus, a bus with only non-slack units is PV, every
    other bus is PQ -- so ref and pv never overlap. The first unit on a bus sets |V|; the first
    slack unit sets the angle.
    """
    per_bus = units.groupby("bus").agg(vm_pu=("vm_pu", "first"), slack=("slack", "any"))
    buses = per_bus.index.to_numpy(dtype=int)
    is_ref = per_bus.slack.to_numpy(dtype=bool)
    ref = buses[is_ref]
    pv = buses[~is_ref]
    pq = np.setdiff1d(np.arange(n_bus), buses)

    initial_voltage[buses] = per_bus.vm_pu.to_numpy(dtype=float)
    va = units[units.slack].groupby("bus").va_degree.first()
    initial_voltage[va.index.to_numpy(dtype=int)] *= np.exp(1j * np.deg2rad(va.to_numpy(dtype=float)))
    return ref, pv, pq


def _unit_results(units: pd.DataFrame, s_residual: NDArray) -> tuple[NDArray, NDArray]:
    """P and Q of every unit, given ``s_residual`` = computed minus scheduled bus injection (MW/MVAr).

    P: a unit delivers its scheduled P; on a reference bus the FIRST slack unit also takes the
    bus's whole residual P (pypower pfsoln, ext_grids come first).
    Q: a bus's residual Q is shared by all its units in proportion to their reactive range, as
    pypower pfsoln._update_q does:
        Q[i] = Qmin[i] + (Q_bus - Qmin_bus) / (Qmax_bus - Qmin_bus) * (Qmax[i] - Qmin[i])
    with an equal split on buses where any limit is missing (NaN/inf) or the total range is zero.
    """
    n_bus = len(s_residual)
    bus = units.bus.to_numpy(dtype=int)

    p = units.p_mw.to_numpy(dtype=float).copy()
    first_slack = units[units.slack].groupby("bus").head(1).index.to_numpy()
    p[first_slack] += s_residual.real[bus[first_slack]]

    q_bus = s_residual.imag[bus]
    n_units = np.bincount(bus, minlength=n_bus)[bus]
    q = q_bus / n_units

    q_min = units.min_q_mvar.to_numpy(dtype=float)
    q_max = units.max_q_mvar.to_numpy(dtype=float)
    finite = np.isfinite(q_min) & np.isfinite(q_max)
    bus_finite = np.bincount(bus, weights=finite, minlength=n_bus)[bus] == n_units
    q_min_f = np.where(finite, q_min, 0.0)
    q_max_f = np.where(finite, q_max, 0.0)
    qmin_tot = np.bincount(bus, weights=q_min_f, minlength=n_bus)[bus]
    qmax_tot = np.bincount(bus, weights=q_max_f, minlength=n_bus)[bus]
    proportional = bus_finite & ~np.isclose(qmax_tot, qmin_tot)
    eps = np.finfo(float).eps
    q_prop = q_min_f + (q_bus - qmin_tot) / (qmax_tot - qmin_tot + eps) * (q_max_f - q_min_f)
    q = np.where(proportional, q_prop, q)
    return p, q


def write_unit_results(
    net: pandapowerNet, units: pd.DataFrame, s_residual: NDArray, vm: NDArray, va: NDArray
) -> tuple[NDArray, NDArray]:
    """Fill res_gen / res_ext_grid. Out-of-service units get 0 everywhere, as pandapower does.
    Returns P and Q (MW, Mvar) per row of ``units``."""
    p, q = _unit_results(units, s_residual)
    bus = units.bus.to_numpy(dtype=int)
    for et in ("gen", "ext_grid"):
        if et not in net or not len(net[et]):
            continue
        res = net[f"res_{et}"]
        if len(res) != len(net[et]) or not res.index.equals(net[et].index):
            res = net[f"res_{et}"] = res.reindex(net[et].index)
        sel = (units.et == et).to_numpy()
        rows = net[et].index.get_indexer(units.idx[sel])
        cols = ["p_mw", "q_mvar"] + (["vm_pu", "va_degree"] if et == "gen" else [])
        values = np.zeros((len(net[et]), len(cols)))
        values[rows, 0] = p[sel]
        values[rows, 1] = q[sel]
        if et == "gen":
            values[rows, 2] = vm[bus[sel]]
            values[rows, 3] = va[bus[sel]]
        res[cols] = values
    return p, q
