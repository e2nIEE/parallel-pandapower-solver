# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Switches and connectivity: which pandapower buses become which node of the power flow.

* Closed bus-bus switches are zero-impedance connections. The buses they join are FUSED into one
  node (many-to-one mapping), which keeps the Newton system as small as possible.
* Open line / trafo / trafo3w switches disconnect that end of the branch. As in pandapower (which
  connects that end to an auxiliary bus) the branch stays energised from its other end(s): the
  open terminal is Kron-eliminated from the branch's stamp (see TwoPort / ThreePort
  ``apply_open_ends``), so line charging and transformer magnetising are kept.
* Buses that are out of service or not connected to any slack (ext_grid or gen with slack=True)
  are unsupplied: they get no node, and everything connected to them drops out.

After the solve the bus-bus switch currents follow from Kirchhoff's current law on each group of
fused buses (``switch_currents``).

All matrices of the power flow live on nodes; the element models still stamp on bus positions
(net.bus row = position), and ``Topology.aggregate`` maps bus-level quantities onto nodes:
    Y_node = C^T Y_bus C,   S_node = C^T S_bus,   V_bus = C V_node
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray
from pandapower import pandapowerNet
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

# Placeholder contact impedance for switches with z_ohm <= 0 when splitting currents in switch
# loops: much smaller than any real contact impedance, so such a switch takes the current against
# parallel switches with a z_ohm, and loops of z_ohm = 0 switches split equally.
Z_OHM_ZERO = 1e-6

# switch.et -> element table and its terminal columns (port order of the element model)
SWITCHED_ELEMENTS = {
    "l": ("line", ("from_bus", "to_bus")),
    "t": ("trafo", ("hv_bus", "lv_bus")),
    "t3": ("trafo3w", ("hv_bus", "mv_bus", "lv_bus")),
}


@dataclass
class Topology:
    node_of_bus: NDArray  # per net.bus row: node index, -1 = unsupplied
    n_node: int
    bus_index: pd.Index  # net.bus.index
    identity: bool  # node == bus row for every bus (no fusion, nothing unsupplied)
    open_ends: dict[str, NDArray] = field(default_factory=dict)  # element -> (n_element, n_ports) bool
    # closed bus-bus switches between supplied buses, for the switch currents
    bb_switch_rows: NDArray = field(default_factory=lambda: np.zeros(0, dtype=int))  # rows in net.switch
    bb_from: NDArray = field(default_factory=lambda: np.zeros(0, dtype=int))  # bus rows
    bb_to: NDArray = field(default_factory=lambda: np.zeros(0, dtype=int))
    bb_weight: NDArray = field(default_factory=lambda: np.zeros(0))  # 1 / z_ohm

    @property
    def aggregate(self) -> sp.csr_matrix:
        """C^T: (n_node x n_bus), sums bus rows into their node; unsupplied buses drop out."""
        rows = self.node_of_bus
        cols = np.arange(len(rows))
        keep = rows >= 0
        return sp.csr_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(self.n_node, len(rows)))

    def to_nodes(self, matrix: sp.spmatrix) -> sp.csr_matrix:
        if self.identity:
            return sp.csr_matrix(matrix)
        c_t = self.aggregate
        return (c_t @ matrix @ c_t.T).tocsr()

    def vector_to_nodes(self, vector: NDArray) -> NDArray:
        if self.identity:
            return np.asarray(vector)
        return self.aggregate @ np.asarray(vector)

    def bus_voltage(self, v_node: NDArray) -> NDArray:
        """Node voltages (n_node,) or (n_node, T) -> bus voltages per net.bus row, NaN where unsupplied."""
        if self.identity:
            return np.asarray(v_node)
        v_node = np.asarray(v_node)
        v_bus = np.full((len(self.node_of_bus),) + v_node.shape[1:], np.nan, dtype=complex)
        supplied = self.node_of_bus >= 0
        v_bus[supplied] = v_node[self.node_of_bus[supplied]]
        return v_bus

    def node_voltage(self, v_bus: NDArray) -> NDArray:
        """Bus voltages per net.bus row -> node voltages (the first bus of each node)."""
        if self.identity:
            return np.asarray(v_bus)
        first = np.full(self.n_node, -1)
        supplied = np.flatnonzero(self.node_of_bus >= 0)
        first[self.node_of_bus[supplied[::-1]]] = supplied[::-1]
        return np.asarray(v_bus)[first]

    @property
    def lookup(self) -> pd.Series:
        """pandapower bus index -> node (-1 = unsupplied)."""
        return pd.Series(self.node_of_bus, index=self.bus_index)

    @property
    def representative_bus(self) -> pd.Series:
        """node -> pandapower bus index of its first bus."""
        supplied = np.flatnonzero(self.node_of_bus >= 0)
        first = pd.Series(supplied).groupby(self.node_of_bus[supplied]).first()
        return pd.Series(self.bus_index[first.to_numpy()], index=first.index)


def _bus_rows(net: pandapowerNet, buses) -> NDArray:
    return net.bus.index.get_indexer(np.asarray(buses))


def _open_ends(net: pandapowerNet) -> dict[str, NDArray]:
    """Per switched element type: which terminals are disconnected by an open switch."""
    open_ends = {}
    sw = net.switch if "switch" in net else pd.DataFrame(columns=["bus", "element", "et", "closed"])
    for et, (element, ports) in SWITCHED_ELEMENTS.items():
        if element not in net or not len(net[element]):
            continue
        mask: NDArray = np.zeros((len(net[element]), len(ports)), dtype=bool)
        sel = sw[(sw.et == et) & ~sw.closed.astype(bool)]
        if len(sel):
            rows = net[element].index.get_indexer(sel.element.to_numpy())
            for k, port in enumerate(ports):
                at_port = (rows >= 0) & (net[element][port].to_numpy()[np.maximum(rows, 0)] == sel.bus.to_numpy())
                mask[rows[at_port], k] = True
        open_ends[element] = mask
    return open_ends


def build_topology(net: pandapowerNet) -> Topology:
    n_bus = len(net.bus)
    bus_on = net.bus.in_service.to_numpy(dtype=bool) if "in_service" in net.bus else np.ones(n_bus, bool)
    open_ends = _open_ends(net)

    # 1. fuse buses joined by closed bus-bus switches (both buses in service)
    if "switch" in net and len(net.switch):
        sw = net.switch
        is_bb = (sw.et == "b").to_numpy() & sw.closed.astype(bool).to_numpy()
        f, t = _bus_rows(net, sw.bus), _bus_rows(net, sw.element)
        is_bb &= (f >= 0) & (t >= 0)
        is_bb[is_bb] &= bus_on[f[is_bb]] & bus_on[t[is_bb]]
    else:
        is_bb = np.zeros(0, dtype=bool)
        f = t = np.zeros(0, dtype=int)
    bb_rows = np.flatnonzero(is_bb)
    fused = sp.coo_matrix((np.ones(len(bb_rows)), (f[bb_rows], t[bb_rows])), shape=(n_bus, n_bus))
    _, group = connected_components(fused, directed=False)

    # 2. connectivity of the groups through in-service branches with closed ends
    edges_f, edges_t = [], []

    def _connect(element: str, ports: tuple[str, ...]):
        if element not in net or not len(net[element]):
            return
        table = net[element]
        on = table.in_service.to_numpy(dtype=bool) if "in_service" in table else np.ones(len(table), bool)
        closed = ~open_ends.get(element, np.zeros((len(table), len(ports)), dtype=bool))
        rows = [_bus_rows(net, table[p]) for p in ports]
        for a in range(len(ports)):
            for b in range(a + 1, len(ports)):
                use = on & closed[:, a] & closed[:, b]
                edges_f.append(group[rows[a][use]])
                edges_t.append(group[rows[b][use]])

    _connect("line", ("from_bus", "to_bus"))
    _connect("trafo", ("hv_bus", "lv_bus"))
    _connect("trafo3w", ("hv_bus", "mv_bus", "lv_bus"))
    _connect("impedance", ("from_bus", "to_bus"))
    n_group = group.max() + 1 if n_bus else 0
    ef = np.concatenate(edges_f) if edges_f else np.zeros(0, dtype=int)
    et_ = np.concatenate(edges_t) if edges_t else np.zeros(0, dtype=int)
    _, island = connected_components(
        sp.coo_matrix((np.ones(len(ef)), (ef, et_)), shape=(n_group, n_group)), directed=False
    )

    # 3. supplied = island of a group that holds an in-service slack (ext_grid, gen with slack=True)
    slack_buses = []
    if "ext_grid" in net and len(net.ext_grid):
        slack_buses.append(net.ext_grid.bus[net.ext_grid.in_service.astype(bool)].to_numpy())
    if "gen" in net and len(net.gen) and "slack" in net.gen:
        g = net.gen[net.gen.in_service.astype(bool) & net.gen.slack.fillna(False).astype(bool)]
        slack_buses.append(g.bus.to_numpy())
    slack_rows = _bus_rows(net, np.concatenate(slack_buses)) if slack_buses else np.zeros(0, dtype=int)
    slack_rows = slack_rows[(slack_rows >= 0)]
    slack_rows = slack_rows[bus_on[slack_rows]]
    supplied_island: NDArray = np.zeros(island.max() + 1 if len(island) else 0, dtype=bool)
    supplied_island[island[group[slack_rows]]] = True
    supplied = bus_on & supplied_island[island[group]]

    # 4. number the nodes in bus order (identity mapping when nothing is fused or dropped)
    node_of_group = np.full(n_group, -1)
    groups_in_order = pd.unique(group[supplied])
    node_of_group[groups_in_order] = np.arange(len(groups_in_order))
    node_of_bus = np.where(supplied, node_of_group[group], -1)
    n_node = len(groups_in_order)
    identity = n_node == n_bus and bool((node_of_bus == np.arange(n_bus)).all())

    topo = Topology(node_of_bus, n_node, net.bus.index, identity, open_ends)
    keep = bb_rows[supplied[f[bb_rows]]]
    if len(keep):
        z = net.switch.z_ohm.to_numpy(dtype=float)[keep] if "z_ohm" in net.switch else np.zeros(len(keep))
        topo.bb_switch_rows = keep
        topo.bb_from, topo.bb_to = f[keep], t[keep]
        topo.bb_weight = 1.0 / np.where(np.isfinite(z) & (z > 0), z, Z_OHM_ZERO)
    return topo


def switch_currents(topo: Topology, i_into_switches: NDArray) -> tuple[NDArray, NDArray]:
    """Currents through the closed bus-bus switches (from switch.bus to switch.element), and the
    resulting net current each bus (net.bus row) sends into its switches.

    ``i_into_switches`` (per net.bus row) is the current each bus pushes into its switches: its
    elements' injection minus what flows into its branches and shunts. On every group of fused
    buses the switch currents satisfy KCL, A @ I_sw = i, with A the bus x switch incidence matrix.
    That fixes them on a tree; in switch loops the split follows the contact impedances (weighted
    minimum-norm solution, W = diag(1 / z_ohm)):
        I_sw = W A^T x,   (A W A^T) x = i     (one bus per group grounded)
    """
    n_sw = len(topo.bb_switch_rows)
    per_bus: NDArray = np.zeros(len(topo.node_of_bus), dtype=complex)
    if n_sw == 0:
        return np.zeros(0, dtype=complex), per_bus
    buses = np.unique(np.r_[topo.bb_from, topo.bb_to])
    pos = np.full(len(topo.node_of_bus), -1)
    pos[buses] = np.arange(len(buses))
    # incidence: +1 at the switch's bus (current leaves), -1 at its element bus (current arrives)
    rows = np.r_[pos[topo.bb_from], pos[topo.bb_to]]
    cols = np.r_[np.arange(n_sw), np.arange(n_sw)]
    a = sp.csr_matrix((np.r_[np.ones(n_sw), -np.ones(n_sw)], (rows, cols)), shape=(len(buses), n_sw))
    w = sp.diags(topo.bb_weight)
    laplacian = (a @ w @ a.T).tocsr()
    # ground the first bus of every group (the Laplacian of each group is singular by one)
    _, comp = connected_components(laplacian, directed=False)
    grounded: NDArray = np.zeros(len(buses), dtype=bool)
    grounded[pd.Series(np.arange(len(buses))).groupby(comp).first().to_numpy()] = True
    free = ~grounded
    x: NDArray = np.zeros(len(buses), dtype=complex)
    rhs = np.asarray(i_into_switches)[buses][free]
    if free.any():
        x[free] = np.atleast_1d(spsolve(laplacian[free][:, free].tocsc(), rhs))
    i_sw = w @ (a.T @ x)
    per_bus[buses] = a @ i_sw
    return i_sw, per_bus


def switch_flows(
    topo: Topology, ybus_bus: sp.spmatrix, v_bus: NDArray, s_elements: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """Bus-bus switch currents after a solve on fused nodes.

    ``v_bus`` are the bus voltages (0 where unsupplied), ``s_elements`` the per-bus injection of
    all node elements (loads, sgens, gens, ext_grids, wards; per unit, injection positive). What a
    bus injects but does not send into its branches and shunts (``ybus_bus @ v_bus``) goes into
    its switches.

    Returns (current into the network per bus, current into the switches per bus, switch currents),
    all per unit; the bus injection is V * conj(sum of both).
    """
    i_network = ybus_bus @ v_bus
    with np.errstate(divide="ignore", invalid="ignore"):
        i_elements = np.where(v_bus != 0, np.conj(s_elements / v_bus), 0.0)
    i_sw, i_switches = switch_currents(topo, i_elements - i_network)
    return i_network, i_switches, i_sw


# element table -> (bus column, current / p / q column suffix) per terminal, for switch results
_SWITCH_ENDS = {
    "line": (("from_bus", "from"), ("to_bus", "to")),
    "trafo": (("hv_bus", "hv"), ("lv_bus", "lv")),
    "trafo3w": (("hv_bus", "hv"), ("mv_bus", "mv"), ("lv_bus", "lv")),
}


def write_switch_results(net: pandapowerNet, topo: Topology, i_sw: NDArray, v_bus: NDArray, sn_mva: float) -> None:
    """res_switch: i_ka, loading_percent (i_ka / in_ka), p/q at both sides (into the switch).

    * closed bus-bus switch: the KCL current of ``switch_currents`` (lossless: p_to = -p_from)
    * line / trafo / trafo3w switch: the element's current and flow at the switch's end
    * open switch: 0; switch on an unsupplied bus: NaN
    """
    sw = net.switch
    n = len(sw)
    i_ka = np.full(n, np.nan)
    s_from: NDArray = np.full(n, np.nan, dtype=complex)
    supplied = topo.node_of_bus >= 0
    bus_rows = _bus_rows(net, sw.bus)
    on_supplied = (bus_rows >= 0) & supplied[np.maximum(bus_rows, 0)]
    closed = sw.closed.astype(bool).to_numpy()

    # bus-bus switches
    if len(topo.bb_switch_rows):
        rows = topo.bb_switch_rows
        v_from = v_bus[topo.bb_from]
        vn_kv = net.bus.vn_kv.to_numpy(dtype=float)[topo.bb_from]
        i_ka[rows] = np.abs(i_sw) * sn_mva / (np.sqrt(3) * vn_kv)
        s_from[rows] = v_from * np.conj(i_sw) * sn_mva

    # branch switches: the element's end at the switch bus
    for et, (element, _ports) in SWITCHED_ELEMENTS.items():
        res = net.get(f"res_{element}")
        if res is None or not len(res):
            continue
        mask = (sw.et == et).to_numpy() & closed
        if not mask.any():
            continue
        rows = net[element].index.get_indexer(sw.element.to_numpy()[mask])
        for bus_col, end in _SWITCH_ENDS[element]:
            at_end = (rows >= 0) & (net[element][bus_col].to_numpy()[np.maximum(rows, 0)] == sw.bus.to_numpy()[mask])
            target = np.flatnonzero(mask)[at_end]
            src = rows[at_end]
            i_ka[target] = res[f"i_{end}_ka"].to_numpy(dtype=float)[src]
            p = res[f"p_{end}_mw"].to_numpy(dtype=float)[src]
            q = res[f"q_{end}_mvar"].to_numpy(dtype=float)[src]
            s_from[target] = p + 1j * q

    i_ka[~closed] = 0.0
    s_from[~closed] = 0.0
    i_ka[~on_supplied] = np.nan
    s_from[~on_supplied] = np.nan
    in_ka = sw.in_ka.to_numpy(dtype=float) if "in_ka" in sw else np.full(n, np.nan)

    res_switch = net.res_switch.reindex(sw.index) if not net.res_switch.index.equals(sw.index) else net.res_switch
    res_switch["i_ka"] = i_ka
    with np.errstate(divide="ignore", invalid="ignore"):
        res_switch["loading_percent"] = i_ka / in_ka * 100.0
    res_switch["p_from_mw"] = s_from.real
    res_switch["q_from_mvar"] = s_from.imag
    res_switch["p_to_mw"] = -s_from.real
    res_switch["q_to_mvar"] = -s_from.imag
    net.res_switch = res_switch


_ELEMENT_BUSES = {
    "line": ("from_bus", "to_bus"),
    "trafo": ("hv_bus", "lv_bus"),
    "trafo3w": ("hv_bus", "mv_bus", "lv_bus"),
    "impedance": ("from_bus", "to_bus"),
    "ward": ("bus",),
}


def zero_unsupplied_flows(net: pandapowerNet, topo: Topology) -> None:
    """Elements with no supplied terminal, as pandapower reports them: p/q 0, currents, loading and
    voltages NaN. (A branch fed from one end, its open end at an unsupplied bus, is energised and
    keeps its results.)"""
    if topo.identity:
        return
    supplied = topo.node_of_bus >= 0
    for element, bus_cols in _ELEMENT_BUSES.items():
        res = net.get(f"res_{element}")
        if element not in net or not len(net[element]) or res is None or not len(res):
            continue
        dead: NDArray = np.ones(len(net[element]), dtype=bool)
        for col in bus_cols:
            pos = _bus_rows(net, net[element][col])
            dead &= (pos < 0) | ~supplied[np.maximum(pos, 0)]
        if not dead.any():
            continue
        rows = res.index[dead]
        power_cols = [c for c in res.columns if c.startswith(("p_", "q_", "pl_", "ql_"))]
        other_cols = [c for c in res.columns if c not in power_cols]
        res.loc[rows, power_cols] = 0.0
        res.loc[rows, other_cols] = np.nan
