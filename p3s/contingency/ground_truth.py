# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pandapower ground-truth oracle for N-1 contingency analysis.

This is the *reference* the p3s batch solvers are validated against.
It is deliberately simple and slow: for each contingency it deep-copies the net, takes
the group's branches out of service, runs a full ``runpp``, and records the per-bus
result. It does NOT share any factorization -- correctness over speed.

Contingency enumeration matches the confirmed scope:
  * a contingency = one distinct non-null ``outage_group`` value across
    ``net.line`` + ``net.trafo``;
  * branches with a null ``outage_group`` are never taken out.

Result schema (one :class:`ContingencyResult` per contingency), mirroring the batch
result table the solvers will produce:
  * ``vm`` / ``va``     : (n_bus,) float, NaN at unserved (islanded-from-reference) buses
  * ``served``          : (n_bus,) bool, False where vm is NaN
  * ``converged``       : bool, whether the referenced system solved

Islanding falls out of pandapower for free: a bus disconnected from every reference
gets NaN in ``res_bus`` while the case still converges (verified on case14). For the
optional re-slacking path, an islanded generator with short-circuit data is promoted
to an ext_grid before the reference solve (lowest-Z generator per island).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray
import pandapower as pp
import pandapower.topology as top
import pandas as pd


@dataclass
class ContingencyResult:
    group: str
    vm: NDArray  # (n_bus,) pu, NaN where unserved
    va: NDArray  # (n_bus,) deg, NaN where unserved
    served: NDArray  # (n_bus,) bool
    converged: bool


def enumerate_contingencies(net) -> list[str]:
    """Return the sorted list of distinct non-null ``outage_group`` values across
    ``net.line`` and ``net.trafo``. Each is one contingency."""
    groups: set = set()
    for tbl in ("line", "trafo"):
        if tbl in net and "outage_group" in net[tbl].columns:
            col = net[tbl]["outage_group"]
            groups.update(g for g in col.dropna().unique() if g is not None and g != "")
    return sorted(groups)


def _branches_in_group(net, group) -> dict[str, list]:
    """Indices of line/trafo rows whose ``outage_group`` == group."""
    out = {}
    for tbl in ("line", "trafo"):
        if tbl in net and "outage_group" in net[tbl].columns:
            mask = net[tbl]["outage_group"] == group
            idx = list(net[tbl].index[mask])
            if idx:
                out[tbl] = idx
    return out


def generator_z_pu(net) -> pd.Series:
    """Per-generator short-circuit impedance magnitude |Z| in pu, from the populated
    ``net.gen`` SC columns. Used to pick an island's reference (lowest Z = stiffest).

    Z_pu = sqrt(rdss_pu^2 + xdss_pu^2), with rdss converted from ohm to pu on the
    generator's own base (Z_base = vn_kv^2 / sn_mva). Generators missing ``xdss_pu``
    get NaN (not eligible to be a re-slack reference).
    """
    gen = net.gen
    if "xdss_pu" not in gen.columns:
        return pd.Series(np.nan, index=gen.index)
    xdss = gen["xdss_pu"].astype(float)
    rdss_ohm = gen.get("rdss_ohm", pd.Series(0.0, index=gen.index)).astype(float).fillna(0.0)
    vn_kv = gen.get("vn_kv")
    if vn_kv is None:
        vn_kv = net.bus.vn_kv.reindex(gen.bus.values).to_numpy()
    else:
        vn_kv = vn_kv.astype(float).to_numpy()
    sn = gen["sn_mva"].astype(float).to_numpy()
    z_base = np.where(sn > 0, (vn_kv**2) / sn, np.nan)
    rdss_pu = rdss_ohm.to_numpy() / z_base
    z = np.sqrt(rdss_pu**2 + xdss.to_numpy() ** 2)
    return pd.Series(z, index=gen.index)


def _connected_components(net) -> list[set]:
    """Connected components of the *current* in-service topology, as sets of bus
    indices (respects in_service of branches and buses)."""
    g = top.create_nxgraph(net, respect_switches=True, include_out_of_service=False)
    return [set(c) for c in nx.connected_components(g)]


def _reslack_islands(net):
    """Promote, in each generatorless+slackless island, the lowest-Z generator to an
    ext_grid so pandapower can solve that island. Mutates ``net`` in place; returns the
    list of (gen_idx) promoted (for diagnostics)."""
    ref_buses = set(net.ext_grid.bus[net.ext_grid.in_service].tolist())
    z = generator_z_pu(net)
    promoted = []
    for comp in _connected_components(net):
        if comp & ref_buses:
            continue  # island already has an original slack
        # candidate generators in this island with valid Z
        cand = [gi for gi in net.gen.index if net.gen.in_service[gi] and net.gen.bus[gi] in comp and np.isfinite(z[gi])]
        if not cand:
            continue  # truly unserved -> leave it; pandapower will NaN the buses
        best = min(cand, key=lambda gi: z[gi])
        gbus = net.gen.bus[best]
        vm = net.gen.vm_pu[best]
        # promote: drop the gen, add an ext_grid at its bus with the gen's vm setpoint
        net.gen.loc[best, "in_service"] = False
        pp.create_ext_grid(net, gbus, vm_pu=vm, va_degree=0.0, name=f"reslack_gen_{best}")
        promoted.append(best)
    return promoted


def solve_contingency(net, group, reslack_islands: bool = False) -> ContingencyResult:
    """Reference result for a single contingency (one outage_group) via pandapower."""
    work = copy.deepcopy(net)
    branches = _branches_in_group(work, group)
    for tbl, idx in branches.items():
        work[tbl].loc[idx, "in_service"] = False

    if reslack_islands:
        _reslack_islands(work)

    n_bus = len(work.bus)
    vm = np.full(n_bus, np.nan)
    va = np.full(n_bus, np.nan)
    converged = False
    try:
        pp.runpp(work, init="flat")
        converged = bool(work["converged"])
        # res_bus is indexed by bus id; align to positional order of net.bus
        vm = work.res_bus.vm_pu.reindex(work.bus.index).to_numpy()
        va = work.res_bus.va_degree.reindex(work.bus.index).to_numpy()
    except pp.LoadflowNotConverged:
        converged = False

    served = np.isfinite(vm)
    return ContingencyResult(group=group, vm=vm, va=va, served=served, converged=converged)


def ground_truth(net, reslack_islands: bool = False) -> dict[str, ContingencyResult]:
    """Reference results for ALL contingencies in ``net``, keyed by outage_group."""
    return {g: solve_contingency(net, g, reslack_islands=reslack_islands) for g in enumerate_contingencies(net)}
