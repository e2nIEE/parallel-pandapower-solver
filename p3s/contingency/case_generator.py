# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-agnostic N-1 contingency case generator.

Turns a pandapower net + its ``outage_group`` columns into the per-contingency inputs
the batch solvers consume:

  * a SHARED Ybus sparsity pattern (CSR Yp/Yj), and
  * per-contingency Ybus *values* (Yx), obtained by subtracting the outaged branches'
    stamps from the base Yx (values-only change -> symbolic factorization stays shared),
  * a per-contingency ``served`` mask + the chosen reference bus of every island.

Why stamp subtraction (and not zeroing matrix entries): each branch contributes a
2x2 stamp ``[[Y_ff, Y_ft], [Y_tf, Y_tt]]`` to the four bus positions
``(f,f),(f,t),(t,f),(t,t)``. Subtracting that exact stamp:
  * handles transformers where ``Y_ft != Y_tf`` (off-nominal tap / phase shift), and
  * handles parallel branches correctly -- the base off-diagonal is the SUM of the
    parallel stamps, so subtracting one leaves the others intact.

The stamp values are read straight off p3s's branch models
(``_Y_ff/_Y_ft/_Y_tf/_Y_tt``), i.e. exactly the values p3s stamped into Ybus, so
this never re-derives admittances from raw table parameters.

This module does NO power-flow solve -- it only produces inputs and masks. It is pure
NumPy/SciPy/pandas (no C++/CUDA), so it is unit-testable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from p3s.contingency.ground_truth import enumerate_contingencies, generator_z_pu

# Pure-Python NewtonPowerflow: used ONLY for its base-Ybus + bus-classification +
# element-model setup (make_ybus, _ybus_elements, busses, _lookup). Phase 1 is
# backend-agnostic, so we deliberately avoid NewtonPowerflowCpp here (that module forces
# importing the compiled nr_klu solver at load time, which the case generator never uses).
from p3s.NewtonPowerflow import NewtonPowerflow

# Bus reference kinds, recorded per island for diagnostics / solver pinning.
REF_SLACK = "slack"  # island contains an original ext_grid (slack)
REF_RESLACK_GEN = "gen"  # island re-slacked onto its lowest-Z generator
REF_NONE = "none"  # island has no reference -> all its buses unserved


@dataclass
class ContingencyCase:
    """Solver inputs for a single contingency (one outage_group).

    ``Yx`` (the base Ybus values with this group's branch stamps removed) is exposed as a
    property, computed on demand from the shared ``Yx_base`` minus this case's sparse stamp
    delta. It is NOT stored as a dense per-case array: for a full pegase N-1 that would be
    ~9.6 GB of copies. The vectorized :meth:`ContingencyCaseGenerator.build` fills the lazy
    fields (``_yx_base`` + ``_stamp_delta``); the reference :meth:`build_case` stores an
    explicit ``_yx`` instead. Either way ``case.Yx`` returns the same (nnz,) complex vector.
    """

    group: str
    served: NDArray  # (n_bus,) bool
    ref_bus: NDArray  # (n_bus,) int, the reference bus of each bus's island, or -1
    pinned_refs: list  # list of (bus, kind) added beyond the original slacks

    # Yx is lazy: either an explicit stored vector (_yx), or reconstructed from a shared
    # base + a sparse delta (_yx_base, _stamp_delta). Exactly one path is populated.
    _yx: NDArray = None  # explicit (nnz,) complex128, or None -> use lazy path
    _yx_base: NDArray = None  # shared base values (not copied per case)
    _stamp_delta: object = None  # sparse (nnz,1) column: values to SUBTRACT from base

    @property
    def Yx(self) -> NDArray:
        """(nnz,) complex128: base pattern with this group's stamps removed."""
        if self._yx is not None:
            return self._yx
        delta = self._stamp_delta
        if delta is None or getattr(delta, "nnz", 1) == 0:
            return self._yx_base.copy()
        return self._yx_base - np.asarray(delta.todense()).ravel()  # type: ignore[attr-defined]


@dataclass
class ContingencyBatch:
    """The shared pattern + all contingency cases for a net."""

    Yp: NDArray  # (n_bus+1,) int32  CSR indptr (shared)
    Yj: NDArray  # (nnz,)     int32  CSR indices (shared)
    Yx_base: NDArray  # (nnz,)     complex128  intact-grid values
    n_bus: int
    groups: list  # contingency names, order matches `cases`
    cases: list  # list[ContingencyCase]
    pv: NDArray  # base pv bus indices (p3s 0-based)
    pq: NDArray  # base pq bus indices
    ref: NDArray  # base ext_grid (slack) bus indices

    # Cached vectorized stamp matrix (nnz, L) set by the generator's build(); lets Yx_matrix
    # form all columns in one op instead of reconstructing/stacking per case. None -> stack.
    _stamp: object = None

    @property
    def Yx_matrix(self) -> NDArray:
        """(nnz, L) per-contingency values, column c = cases[c].Yx. Convenient for the
        batch solvers (one column per case, shared Yp/Yj).

        Built in one vectorized op from the shared base minus the sparse stamp matrix
        (``Yx_base[:,None] - stamp``), not by stacking L per-case reconstructions."""
        L = len(self.cases)
        if self._stamp is not None:
            M = np.repeat(self.Yx_base[:, None], L, axis=1)  # (nnz, L)

            # subtract all stamp deltas
            M -= np.asarray(self._stamp.todense())  # type: ignore[attr-defined]
            return M
        return np.stack([c.Yx for c in self.cases], axis=1)


def _branch_stamps(npf: NewtonPowerflow):
    """Yield (from_bus, to_bus, Yff, Yft, Ytf, Ytt, table, row_idx) for every branch
    p3s modelled, reading the per-branch stamp straight off the element models.

    `row_idx` is the position within that element's table (0..n_element-1), matching the
    order of the pandapower line/trafo table rows -- so we can map an outage_group (which
    is keyed by table row) to its stamp.
    """
    elements = npf._ybus_elements
    for table in ("line", "trafo"):
        model = elements.get(table)
        if model is None:
            continue
        fb = np.asarray(model._from_bus, dtype=np.intp)  # type: ignore[union-attr]
        tb = np.asarray(model._to_bus, dtype=np.intp)  # type: ignore[union-attr]
        yff = np.asarray(model._Y_ff, dtype=complex)  # type: ignore[union-attr]
        yft = np.asarray(model._Y_ft, dtype=complex)  # type: ignore[union-attr]
        ytf = np.asarray(model._Y_tf, dtype=complex)  # type: ignore[union-attr]
        ytt = np.asarray(model._Y_tt, dtype=complex)  # type: ignore[union-attr]
        for i in range(len(fb)):
            yield table, i, int(fb[i]), int(tb[i]), yff[i], yft[i], ytf[i], ytt[i]


def _csr_pos_lookup(Ybus_csr: sp.csr_matrix):
    """Return a function pos(r, c) -> index into Ybus_csr.data for entry (r, c).

    Built once from the (sorted) CSR structure; O(1) lookups via a per-row dict.
    Raises KeyError if (r, c) is not a structural nonzero (should never happen for a
    branch stamp, since the branch created that entry).
    """
    Yp, Yj = Ybus_csr.indptr, Ybus_csr.indices
    n = Ybus_csr.shape[0]
    row_maps: list[dict] = [{} for _ in range(n)]
    for r in range(n):
        for k in range(Yp[r], Yp[r + 1]):
            row_maps[r][Yj[k]] = k

    def pos(r, c):
        return row_maps[r][c]

    return pos


def _find_bridges(n: int, ea: NDArray, eb: NDArray) -> set:
    """Return the set of bridge edges (as (min,max) endpoint tuples) of the SIMPLE
    undirected graph on ``n`` vertices with edges ``(ea[i], eb[i])``.

    Iterative Tarjan low-link (recursion would overflow the stack on large grids). An
    edge (u,v) is a bridge iff low[v] > disc[u] in the DFS tree, i.e. v's subtree has no
    back-edge to u or an ancestor of u. Multi-edges are assumed already removed by the
    caller (a multi-edge is never a bridge).
    """
    if len(ea) == 0:
        return set()
    # adjacency list with edge ids (to skip the tree edge back to the parent correctly,
    # even though there are no multi-edges here)
    adj: list = [[] for _ in range(n)]
    for eid in range(len(ea)):
        u, v = int(ea[eid]), int(eb[eid])
        adj[u].append((v, eid))
        adj[v].append((u, eid))

    disc = [-1] * n
    low = [0] * n
    bridges = set()
    timer = 0
    for s in range(n):
        if disc[s] != -1:
            continue
        # stack frames: (node, parent_edge_id, iterator_index)
        stack = [(s, -1, 0)]
        disc[s] = low[s] = timer
        timer += 1
        while stack:
            u, pe, idx = stack[-1]
            if idx < len(adj[u]):
                stack[-1] = (u, pe, idx + 1)
                v, eid = adj[u][idx]
                if eid == pe:
                    continue  # don't go back over the edge we came in on
                if disc[v] == -1:
                    disc[v] = low[v] = timer
                    timer += 1
                    stack.append((v, eid, 0))
                else:
                    if disc[v] < low[u]:
                        low[u] = disc[v]
            else:
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    if low[u] < low[p]:
                        low[p] = low[u]
                    if low[u] > disc[p]:
                        bridges.add((min(p, u), max(p, u)))
    return bridges


class ContingencyCaseGenerator:
    """Builds the shared pattern + per-contingency Ybus values and served masks.

    Parameters
    ----------
    net : pandapowerNet
        Must carry ``outage_group`` on ``net.line`` / ``net.trafo`` (see fixtures).
    reslack_islands : bool, default False
        When True, an island with no original slack but >=1 generator with valid
        short-circuit data is referenced by its lowest-Z generator (the bus is recorded
        as a pinned reference and counts as served). When False, such islands are
        entirely unserved.
    """

    def __init__(self, net, reslack_islands: bool = False):
        self.net = net
        self.reslack_islands = reslack_islands

        # Reuse p3s's setup: base Ybus (CSR), bus classification, element models.
        self._npf = NewtonPowerflow(net)
        self._Ybus = self._npf._YBus.tocsr()
        self._Ybus.sort_indices()
        self.n_bus = self._Ybus.shape[0]

        self.Yp = np.ascontiguousarray(self._Ybus.indptr, dtype=np.int32)
        self.Yj = np.ascontiguousarray(self._Ybus.indices, dtype=np.int32)
        self.Yx_base = np.ascontiguousarray(self._Ybus.data, dtype=np.complex128)

        self.pv = np.asarray(self._npf.busses["pv"], dtype=np.int64)
        self.pq = np.asarray(self._npf.busses["pq"], dtype=np.int64)
        self.ref = np.asarray(self._npf.busses["ref"], dtype=np.int64)

        # bus<->p3s-index lookup (p3s is 0-based positional; pandapower bus ids
        # may differ). `_lookup` maps pandapower bus id -> p3s index.
        self._lookup = self._npf._lookup

        self._pos = _csr_pos_lookup(self._Ybus)

        # Pre-index each branch stamp into its 4 CSR data positions and its outage_group.
        self._branch_records = self._index_branches()

        # Precompute generator short-circuit Z (pu) once, mapped to p3s bus index.
        self._gen_z = self._gen_z_by_bus()

    # -- setup helpers -------------------------------------------------------

    def _index_branches(self):
        """For every modelled branch, record its outage_group and the 4 (csr_pos, value)
        stamp entries, so a contingency just subtracts the right values.

        Also builds the *vectorized* arrays used by :meth:`build` (one row per branch):
          ``_br_from`` / ``_br_to``   : (B,) int  endpoint bus indices
          ``_br_group``               : (B,) object  outage_group (or None)
          ``_br_pos``                 : (B, 4) int  CSR data positions of the 4 stamp
                                        entries (ff, ft, tf, tt)
          ``_br_val``                 : (B, 4) complex  the stamp values at those positions
        """
        records = []  # list of dict(group, entries=[(csr_pos, value), ...])
        net = self.net
        fb_l, tb_l, grp_l, pos_l, val_l = [], [], [], [], []
        for table, i, fb, tb, yff, yft, ytf, ytt in _branch_stamps(self._npf):
            group = None
            if "outage_group" in net[table].columns:
                # element model row i corresponds to net[table] row i (same order)
                group = net[table]["outage_group"].iloc[i]
                if group is not None and (isinstance(group, float) and np.isnan(group)):
                    group = None
            pos = (self._pos(fb, fb), self._pos(fb, tb), self._pos(tb, fb), self._pos(tb, tb))
            val = (yff, yft, ytf, ytt)
            entries = list(zip(pos, val, strict=False))
            records.append({"group": group, "from": fb, "to": tb, "entries": entries})
            fb_l.append(fb)
            tb_l.append(tb)
            grp_l.append(group)
            pos_l.append(pos)
            val_l.append(val)

        self._br_from = np.asarray(fb_l, dtype=np.int64)
        self._br_to = np.asarray(tb_l, dtype=np.int64)
        self._br_group = np.asarray(grp_l, dtype=object)
        self._br_pos = np.asarray(pos_l, dtype=np.int64).reshape(-1, 4) if pos_l else np.zeros((0, 4), dtype=np.int64)
        self._br_val = (
            np.asarray(val_l, dtype=np.complex128).reshape(-1, 4) if val_l else np.zeros((0, 4), dtype=np.complex128)
        )
        return records

    def _gen_z_by_bus(self) -> dict:
        """Map p3s bus index -> min generator Z (pu) at that bus, for re-slacking.
        Only generators with finite Z are eligible. Buses with several gens take the
        stiffest (lowest Z)."""
        net = self.net
        if "gen" not in net or len(net.gen) == 0:
            return {}
        z = generator_z_pu(net)
        out: dict = {}
        for gi in net.gen.index:
            if not bool(net.gen.in_service[gi]):
                continue
            zi = z[gi]
            if not np.isfinite(zi):
                continue

            assert self._lookup is not None, "lookup not initialized"
            gbus = int(self._lookup[net.gen.bus[gi]])
            if gbus not in out or zi < out[gbus]:
                out[gbus] = zi
        return out

    def _outaged_branch_indices(self, group) -> list:
        """Indices into self._branch_records for branches in `group`."""
        return [k for k, rec in enumerate(self._branch_records) if rec["group"] == group]

    # -- per-contingency assembly --------------------------------------------

    def _case_Yx(self, outaged: list) -> NDArray:
        """Base Yx with the outaged branches' stamps subtracted (shared pattern)."""
        Yx = self.Yx_base.copy()
        for k in outaged:
            for csr_pos, value in self._branch_records[k]["entries"]:
                Yx[csr_pos] -= value
        return Yx

    def _components_after_outage(self, outaged: list) -> tuple[NDArray, int]:
        """Connected components of the bus graph with the outaged branches removed.

        Vectorized: the base branch endpoint arrays (``_br_from``/``_br_to``) are constant,
        so we just drop the outaged rows with a boolean mask and build the adjacency in one
        scipy call -- no per-branch Python loop. Returns (labels (n_bus,), n_components).
        """
        n = self.n_bus
        f, t = self._br_from, self._br_to
        if len(f):
            keep: NDArray = np.ones(len(f), dtype=bool)
            if outaged:
                keep[np.asarray(outaged, dtype=np.int64)] = False
            fk, tk = f[keep], t[keep]
            # undirected: stamp both (f,t) and (t,f)
            rows = np.concatenate([fk, tk])
            cols = np.concatenate([tk, fk])
            adj = sp.coo_matrix((np.ones(rows.shape[0], dtype=np.int8), (rows, cols)), shape=(n, n)).tocsr()
        else:
            adj = sp.csr_matrix((n, n))
        n_comp, labels = sp.csgraph.connected_components(adj, directed=False)
        return labels, n_comp

    def _resolve_references(self, labels: NDArray, n_comp: int):
        """Per island, choose a reference: original slack > lowest-Z gen (if reslack) >
        none. Returns (served mask, ref_bus per bus, pinned_refs list).

        Vectorized over buses (no Python per-bus loop). For each component we pick the
        reference bus by computing a per-component minimum:
          * slack components -> the lowest-index slack bus in the component;
          * else (if reslack) -> the lowest-Z generator bus (ties -> lowest index);
          * else -> unserved (ref -1).
        ``ref_bus`` is then a per-bus gather of its component's chosen reference.
        """
        labels = np.asarray(labels)

        # comp_ref[c] = chosen reference bus for component c, or -1 if none.
        comp_ref: NDArray = np.full(n_comp, -1, dtype=np.int64)

        # 1) original slacks: per component, the minimum slack bus index.
        if len(self.ref):
            slack_comp = labels[self.ref]
            # np.minimum.at gives per-component min over slack bus indices
            tmp: NDArray = np.full(n_comp, np.iinfo(np.int64).max, dtype=np.int64)
            np.minimum.at(tmp, slack_comp, self.ref.astype(np.int64))
            has_slack = tmp != np.iinfo(np.int64).max
            comp_ref[has_slack] = tmp[has_slack]

        pinned_refs = []
        # 2) re-slack: components with no slack but >=1 eligible generator.
        if self.reslack_islands and self._gen_z:
            gen_buses = np.fromiter(self._gen_z.keys(), dtype=np.int64, count=len(self._gen_z))
            gen_zs = np.fromiter(self._gen_z.values(), dtype=np.float64, count=len(self._gen_z))
            gcomp = labels[gen_buses]
            # per component: the minimum Z, then the (lowest-index) bus achieving it.
            best_z = np.full(n_comp, np.inf)
            np.minimum.at(best_z, gcomp, gen_zs)
            # for components still without a reference and having a finite best_z, pick
            # the lowest bus index among generators that attain best_z.
            need = (comp_ref == -1) & np.isfinite(best_z)
            if need.any():
                # candidate gens whose Z equals their component's best
                is_best = np.isclose(gen_zs, best_z[gcomp]) & need[gcomp]
                cand_bus = np.where(is_best, gen_buses, np.iinfo(np.int64).max)
                chosen: NDArray = np.full(n_comp, np.iinfo(np.int64).max, dtype=np.int64)
                np.minimum.at(chosen, gcomp, cand_bus)
                picked = need & (chosen != np.iinfo(np.int64).max)
                comp_ref[picked] = chosen[picked]
                for c in np.nonzero(picked)[0]:
                    pinned_refs.append((int(chosen[c]), REF_RESLACK_GEN))

        # 3) per-bus gather of the component reference; served iff a reference exists.
        ref_bus = comp_ref[labels]
        served = ref_bus != -1
        return served, ref_bus, pinned_refs

    def build_case(self, group) -> ContingencyCase:
        """Reference (clarity-first) single-case build. :meth:`build` produces identical
        cases far faster via vectorization; this is kept for tests / debugging."""
        outaged = self._outaged_branch_indices(group)
        Yx = self._case_Yx(outaged)
        labels, n_comp = self._components_after_outage(outaged)
        served, ref_bus, pinned = self._resolve_references(labels, n_comp)
        # reference path: store an explicit Yx (the vectorized build() uses the lazy fields).
        return ContingencyCase(group=group, served=served, ref_bus=ref_bus, pinned_refs=pinned, _yx=Yx)

    # -- vectorized batch build ----------------------------------------------

    def _group_branch_map(self, groups: list) -> list[NDArray]:
        """For each group (in ``groups`` order), the array of branch-row indices in it.
        One vectorized pass over ``_br_group`` instead of an O(B) scan per group."""
        gpos = {g: i for i, g in enumerate(groups)}
        members: list[list] = [[] for _ in groups]
        for k, g in enumerate(self._br_group):
            j = gpos.get(g)
            if j is not None:
                members[j].append(k)
        return [np.asarray(m, dtype=np.int64) for m in members]

    def _base_bridges(self) -> set:
        """Branch-row indices whose REMOVAL disconnects the base graph (cut edges).

        A single-branch contingency can island a bus only if the branch is a bridge;
        non-bridge removals leave connectivity (and thus the served mask) unchanged, so
        we can skip the per-case connectivity solve for them entirely. Computed once via
        the difference in connected-component count when each parallel-class edge is
        removed -- but cheaply: we use the standard fact that an edge is a bridge iff it
        is not part of any cycle, found here by comparing component counts of the full
        graph vs. the graph with multi-edges collapsed. We use scipy on the simple graph.
        """
        n = self.n_bus
        f, t = self._br_from, self._br_to
        # collapse parallels: an edge with multiplicity > 1 is never a bridge
        if len(f) == 0:
            return set()
        # multiplicity per undirected endpoint pair
        a = np.minimum(f, t)
        b = np.maximum(f, t)
        key = a.astype(np.int64) * n + b
        uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
        mult = counts[inv]  # per-branch multiplicity
        # Build the simple graph (unique edges) and find bridges via DFS low-link.
        bridges_pair = _find_bridges(n, a[mult == 1], b[mult == 1])
        # map bridge endpoint pairs back to branch rows (only mult==1 candidates)
        out = set()
        for k in np.nonzero(mult == 1)[0]:
            if (int(a[k]), int(b[k])) in bridges_pair:
                out.add(int(k))
        return out

    def build(self) -> ContingencyBatch:
        """Vectorized batch build: produces the same cases as ``build_case`` per group,
        but avoids the per-contingency Python loops that dominated wall-clock.
        Three optimizations:

          * stamp subtraction is assembled as one sparse (nnz, L) matrix in a single
            vectorized pass, instead of copying the full Yx and looping per case;
          * connectivity is skipped for single, non-bridge branch removals (which cannot
            island anything) -- only bridge / multi-branch groups run a components solve;
          * reference resolution is vectorized over buses.
        """
        groups = enumerate_contingencies(self.net)
        L = len(groups)
        n = self.n_bus
        nnz = len(self.Yx_base)
        members = self._group_branch_map(groups)

        # --- vectorized stamp subtraction -> sparse (nnz, L) ---
        # For each group column j, accumulate its branches' 4 stamp values at their CSR
        # positions. Flatten (csr_pos, col, value) across all in-group branches at once.
        rows_l, cols_l, vals_l = [], [], []
        for j, m in enumerate(members):
            if len(m) == 0:
                continue
            pos = self._br_pos[m].reshape(-1)  # (4*|m|,)
            val = self._br_val[m].reshape(-1)
            rows_l.append(pos)
            cols_l.append(np.full(pos.shape, j, dtype=np.int64))
            vals_l.append(val)
        if rows_l:
            srow = np.concatenate(rows_l)
            scol = np.concatenate(cols_l)
            sval = np.concatenate(vals_l)
            stamp = sp.coo_matrix((sval, (srow, scol)), shape=(nnz, L)).tocsc()
        else:
            stamp = sp.csc_matrix((nnz, L), dtype=np.complex128)
        self._stamp = stamp  # cached sparse stamps (used by Yx_matrix)

        # --- connectivity: only bridge / multi-branch groups can island ---
        bridges = self._base_bridges()
        # all_served = np.ones(n, dtype=bool)
        # default reference for every bus in the fully-connected case
        base_served, base_ref, _ = self._resolve_references(np.zeros(n, dtype=np.int64), 1)

        cases: list[ContingencyCase] = []
        stamp_csc = stamp  # column slicing (cheap CSC column extraction, per case)
        for j, g in enumerate(groups):
            m = members[j]
            needs_conn = (len(m) > 1) or any(int(k) in bridges for k in m)
            if not needs_conn:
                served, ref_bus, pinned = base_served.copy(), base_ref.copy(), []
            else:
                labels, n_comp = self._components_after_outage(list(m))
                served, ref_bus, pinned = self._resolve_references(labels, n_comp)
            # Lazy Yx: no dense per-case copy (that materialized ~9.6 GB for full pegase N-1).
            # Store the shared base + this case's sparse stamp column; case.Yx reconstructs on
            # demand, and Yx_matrix builds the whole (nnz,L) in one vectorized op below.
            cases.append(
                ContingencyCase(
                    group=g,
                    served=served,
                    ref_bus=ref_bus,
                    pinned_refs=pinned,
                    _yx_base=self.Yx_base,
                    _stamp_delta=stamp_csc[:, j],
                )
            )

        return ContingencyBatch(
            Yp=self.Yp,
            Yj=self.Yj,
            Yx_base=self.Yx_base,
            n_bus=self.n_bus,
            groups=groups,
            cases=cases,
            pv=self.pv,
            pq=self.pq,
            ref=self.ref,
            _stamp=stamp,
        )


def generate_cases(net, reslack_islands: bool = False) -> ContingencyBatch:
    """Convenience wrapper: build the contingency batch for a net."""
    return ContingencyCaseGenerator(net, reslack_islands=reslack_islands).build()
