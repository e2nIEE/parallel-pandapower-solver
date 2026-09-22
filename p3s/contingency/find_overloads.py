# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause
"""Post-processing: per-contingency branch flows, currents and loadings.

The contingency solvers (``solver_cpp`` / ``solver_cuda``) return only BUS VOLTAGES --
a ``ContingencyResultTable`` with ``V`` of shape ``(n_bus, L)``. An N-1 study almost always
wants the branch answer instead: "which lines are overloaded in which contingency?" This
module turns that voltage table into per-branch results, vectorized over all L cases at
once (one sparse matmul per branch end, not a Python loop over contingencies).

The physics matches ``NewtonPowerflow._parse_results`` exactly, so a contingency column
reproduces what a single ``calculate()`` on the outaged net would write into ``res_line``::

    S_from  = conj(Yf @ V) * V[from_bus] * sn_mva          # MVA
    I_from  = |S_from| / (vn_kv * vm_from_pu * sqrt(3))    # kA
    loading = max(I_from, I_to) / max_i_ka * 100           # %

Two things make the contingency case different from a plain power flow, and both are
handled here:

* **Outaged branches.** ``Yf``/``Yt`` are built once from the INTACT grid, so a naive
  ``Yf @ V`` reports a flow on the very branch that is out of service in that column. Every
  branch belonging to a column's outage group is therefore forced to exactly zero flow
  (an out-of-service line is 0 %, not undefined).
* **Unserved buses.** Islanded buses carry ``NaN`` voltage (see ``solver_cpp``). Any branch
  touching one is genuinely undefined, so its results stay ``NaN`` rather than silently
  producing a number from a placeholder voltage.

Non-converged columns are NaN-filled wholesale: a non-converged voltage vector is
meaningless, so deriving a loading from it would be worse than reporting nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from p3s.models.TransmissionLineModel import TransmissionLineModel
from p3s.models.TwoPort import TwoPort
from p3s.models.TwoWindingTransformerModel import TwoWindingTransformerModel
from p3s.NewtonPowerflow import NewtonPowerflow


@dataclass
class BranchLoadingTable:
    """Per-branch, per-contingency flows/currents/loadings.

    Every array is ``(n_branch, L)`` with rows in pandapower ``net.line`` order (or
    ``net.trafo`` order for the trafo table) and columns matching
    ``ContingencyResultTable.groups``.

    NaN marks "undefined for this case": the column did not converge, or the branch touches
    an unserved (islanded) bus. An OUT-OF-SERVICE branch is not undefined -- it carries no
    current, so it reports 0.0 across the board.
    """

    groups: list  # contingency names, column order
    element: str  # "line" or "trafo"
    index: np.ndarray  # (n_branch,) pandapower element index (row labels)
    p_from_mw: np.ndarray  # (n_branch, L)
    q_from_mvar: np.ndarray
    p_to_mw: np.ndarray
    q_to_mvar: np.ndarray
    i_from_ka: np.ndarray
    i_to_ka: np.ndarray
    i_ka: np.ndarray  # max(|I_from|, |I_to|)
    loading_percent: np.ndarray
    outaged: np.ndarray  # (n_branch, L) bool -- branch is the outage in that column

    @property
    def max_loading_percent(self) -> np.ndarray:
        """(n_branch,) worst loading each branch ever sees, across all contingencies."""
        if self.loading_percent.size == 0:
            return np.empty(0)
        with np.errstate(invalid="ignore"):
            return np.nanmax(self.loading_percent, axis=1)

    @property
    def worst_case(self) -> np.ndarray:
        """(n_branch,) column index of each branch's worst contingency (-1 if all NaN)."""
        lp = self.loading_percent
        if lp.size == 0:
            return np.empty(0, dtype=int)
        allnan = np.isnan(lp).all(axis=1)
        idx = np.full(lp.shape[0], -1, dtype=int)
        if (~allnan).any():
            idx[~allnan] = np.nanargmax(lp[~allnan], axis=1)
        return idx

    def overloads(self, threshold: float = 100.0):
        """``(branch_row, column)`` index pairs where loading exceeds ``threshold`` %.

        NaN entries never count as an overload. Use the returned column indices against
        ``self.groups`` and the row indices against ``self.index``.
        """
        with np.errstate(invalid="ignore"):
            hit = self.loading_percent > threshold
        return np.nonzero(hit)


def _branch_groups(net, element: str, n_branch: int) -> np.ndarray:
    """(n_branch,) object array of each branch's outage_group (None where unset)."""
    if element not in net or "outage_group" not in net[element].columns:
        return np.full(n_branch, None, dtype=object)
    return net[element]["outage_group"].to_numpy(dtype=object)


def _empty_table(res, element: str, L: int) -> BranchLoadingTable:
    z = np.empty((0, L))
    return BranchLoadingTable(
        groups=list(res.groups),
        element=element,
        index=np.empty(0, dtype=int),
        p_from_mw=z,
        q_from_mvar=z.copy(),
        p_to_mw=z.copy(),
        q_to_mvar=z.copy(),
        i_from_ka=z.copy(),
        i_to_ka=z.copy(),
        i_ka=z.copy(),
        loading_percent=z.copy(),
        outaged=np.empty((0, L), dtype=bool),
    )


def compute_branch_loading(net, res, element: str = "line", npf=None) -> BranchLoadingTable:
    """Branch flows / currents / loadings for every contingency in ``res``.

    Parameters
    ----------
    net : pandapowerNet
        The SAME net handed to the contingency solver (its ``outage_group`` columns are
        read to zero out each column's outaged branches). Prepare it exactly as for the
        solve -- notably ``calculateTrafoCharacteristic`` -- so the Ybus matches.
    res : ContingencyResultTable
        Result of ``solve_contingencies_cpp`` / ``solve_contingencies_cuda``.
    element : {"line", "trafo"}, default "line"
        Which branch table to report on.
    npf : NewtonPowerflow, optional
        An already-built solver for ``net``, reused for its branch models. Building one
        re-derives the whole Ybus (~20 ms on pegase-1354), which is pure overhead when the
        caller already has one or is asking for both element types.

    Returns
    -------
    BranchLoadingTable
        ``(n_branch, L)`` arrays; see the dataclass docstring for the NaN convention.

    Notes
    -----
    Cost is one sparse ``(n_branch, n_bus) @ (n_bus, L)`` product per branch end, so the
    whole table is two matmuls regardless of L -- no per-contingency Python loop.
    """
    if element not in ("line", "trafo"):
        raise ValueError(f"element must be 'line' or 'trafo', got {element!r}")

    V = np.asarray(res.V)
    if V.ndim != 2:
        raise ValueError(f"res.V must be 2-D (n_bus, L), got shape {V.shape}")
    n_bus, L = V.shape
    if len(net.bus) != n_bus:
        raise ValueError(f"net has {len(net.bus)} buses but res.V has {n_bus} rows")

    n_branch = len(net[element]) if element in net else 0
    if n_branch == 0:
        return _empty_table(res, element, L)

    # Rebuild the intact-grid branch model: yf_matrix / yt_matrix are (n_branch, n_bus)
    # and row-aligned with net[element], which is what lets us matmul the whole table.
    if npf is None:
        npf = NewtonPowerflow(net)
    mdl = npf._ybus_elements.get(element)
    if mdl is None:
        return _empty_table(res, element, L)
    # _ybus_elements also holds a ThreePort (under "trafo3w"), which has no from/to branch
    # ends and so no yf_matrix/yt_matrix. ``element`` is validated to line/trafo above, so
    # this is unreachable in practice -- but it narrows the union for type checkers and
    # turns a would-be AttributeError into a clear message if the mapping ever changes.
    if not isinstance(mdl, TwoPort):
        raise TypeError(
            f"{element!r} is modelled by {type(mdl).__name__}, which has no two-port "
            "branch ends; per-branch loading is only defined for line/trafo elements"
        )

    # yf_matrix / yt_matrix are built as a side effect of NewtonPowerflow's make_ybus.
    # Only (re)build if that did not happen -- calling create_y_matrix unconditionally
    # trips its `if self.y_matrix and ...` sparse-truthiness guard.
    if mdl.yf_matrix is None or mdl.yt_matrix is None:
        mdl.create_y_matrix(n_bus=n_bus)
    assert mdl.yf_matrix is not None and mdl.yt_matrix is not None  # narrowed for mypy
    fb = np.asarray(mdl._from_bus, dtype=np.intp)
    tb = np.asarray(mdl._to_bus, dtype=np.intp)
    sn_mva = net.sn_mva

    # Complex power at each end, all contingencies at once: (n_branch, n_bus) @ (n_bus, L).
    # NaN voltages at unserved buses propagate through the matmul, which is what we want --
    # a branch touching an island has undefined flow.
    S_from = np.conj(mdl.yf_matrix @ V) * V[fb, :] * sn_mva  # (n_branch, L) MVA
    S_to = np.conj(mdl.yt_matrix @ V) * V[tb, :] * sn_mva

    vm = np.abs(V)  # (n_bus, L) pu
    sqrt3 = np.sqrt(3.0)
    # Nominal voltage per branch end. The line model carries ONE voltage per branch
    # (both ends share a vn_kv); the trafo model carries hv/lv separately. isinstance
    # rather than `element ==` so the attribute access is type-safe on each concrete model.
    if isinstance(mdl, TransmissionLineModel):
        vn_from = np.asarray(mdl.voltages, dtype=float)[:, None]
        vn_to = vn_from
    elif isinstance(mdl, TwoWindingTransformerModel):
        vn_from = np.asarray(mdl.voltages_from, dtype=float)[:, None]
        vn_to = np.asarray(mdl.voltages_to, dtype=float)[:, None]
    else:
        raise TypeError(f"unsupported branch model {type(mdl).__name__} for element {element!r}")

    with np.errstate(divide="ignore", invalid="ignore"):
        i_from = np.abs(S_from) / (vn_from * vm[fb, :] * sqrt3)  # kA
        i_to = np.abs(S_to) / (vn_to * vm[tb, :] * sqrt3)
        i_max = np.maximum(i_from, i_to)
        if element == "line":
            rating = net.line["max_i_ka"].to_numpy(dtype=float)[:, None]
            loading = i_max / rating * 100.0
        else:
            # Mirror _parse_results' trafo loading: apparent power vs the trafo MVA rating.
            s_max = np.maximum(i_from * vn_from * sqrt3, i_to * vn_to * sqrt3)
            loading = s_max / net.trafo["sn_mva"].to_numpy(dtype=float)[:, None] * 100.0

    # -- zero the branches that are OUT in each column ------------------------------
    # yf/yt come from the intact grid, so without this an outaged line reports the flow it
    # would have carried had it stayed in service.
    grp = _branch_groups(net, element, n_branch)
    col_of = {g: c for c, g in enumerate(res.groups)}
    outaged = np.zeros((n_branch, L), dtype=bool)
    for r, g in enumerate(grp):
        c = col_of.get(g)
        if c is not None:
            outaged[r, c] = True
    for arr in (S_from, S_to, i_from, i_to, i_max, loading):
        arr[outaged] = 0.0

    # -- NaN out non-converged columns ---------------------------------------------
    # A non-converged voltage vector is meaningless; a derived loading would be worse than
    # reporting nothing. Applied AFTER the outage zeroing so it wins.
    bad = ~np.asarray(res.converged, dtype=bool)
    if bad.any():
        for arr in (S_from, S_to, i_from, i_to, i_max, loading):
            arr[:, bad] = np.nan

    return BranchLoadingTable(
        groups=list(res.groups),
        element=element,
        index=net[element].index.to_numpy(),
        p_from_mw=S_from.real,
        q_from_mvar=S_from.imag,
        p_to_mw=S_to.real,
        q_to_mvar=S_to.imag,
        i_from_ka=i_from,
        i_to_ka=i_to,
        i_ka=i_max,
        loading_percent=loading,
        outaged=outaged,
    )


@dataclass
class OverloadReport:
    """Branches loaded above a user-set threshold, as a flat list of violations.

    An N-1 study's actual question is "what is overloaded, and in which contingency?" --
    a SPARSE answer. Rather than a dense ``(n_branch, L)`` table this holds one entry per
    violation, so a clean study returns almost nothing regardless of how big the sweep was.

    All arrays share one length ``n_violations`` and are sorted by descending loading, so
    ``rep.element[0]`` etc. is always the worst violation found.
    """

    threshold_percent: float  # the level that was screened against
    element: np.ndarray  # (n_viol,) "line" / "trafo"
    index: np.ndarray  # (n_viol,) pandapower element index (net.line/.trafo row label)
    group: np.ndarray  # (n_viol,) contingency name that caused it
    case: np.ndarray  # (n_viol,) column index into res.groups
    loading_percent: np.ndarray  # (n_viol,) float
    i_ka: np.ndarray  # (n_viol,) float, max(|I_from|, |I_to|)
    rating: np.ndarray  # (n_viol,) float, max_i_ka (line) or sn_mva (trafo)

    def __len__(self) -> int:
        return len(self.index)

    @property
    def n_violations(self) -> int:
        return len(self.index)

    def to_dataframe(self):
        """pandas DataFrame of the violations (one row each), worst first."""
        import pandas as pd

        return pd.DataFrame(
            {
                "element": self.element,
                "index": self.index,
                "group": self.group,
                "case": self.case,
                "loading_percent": self.loading_percent,
                "i_ka": self.i_ka,
                "rating": self.rating,
            }
        )

    def worst_per_branch(self):
        """DataFrame with only each branch's single worst violation, worst first."""
        df = self.to_dataframe()
        if df.empty:
            return df
        return (
            df.sort_values("loading_percent", ascending=False)
            .drop_duplicates(subset=["element", "index"], keep="first")
            .reset_index(drop=True)
        )


def find_overloads(net, res, threshold_percent: float = 100.0, elements=("line", "trafo"), npf=None) -> OverloadReport:
    """Every (branch, contingency) pair loaded above ``threshold_percent``.

    This is the screening entry point for an N-1 study: it answers "which lines and
    transformers exceed my limit, and in which contingency?" without materializing the
    dense loading table for the caller.

    Parameters
    ----------
    net, res : as in :func:`compute_branch_loading`.
    threshold_percent : float, default 100.0
        Loading level to flag, in PERCENT OF THE BRANCH RATING -- i.e. of ``max_i_ka``
        for lines and ``sn_mva`` for transformers. So ``70.0`` means "flag anything above
        70 % of the rating". This is the planning limit and is entirely the caller's
        choice; 100 is the thermal rating itself, and N-1 studies commonly screen lower.
    elements : sequence of {"line", "trafo"}, default both
        Which branch types to screen.
    npf : NewtonPowerflow, optional
        Reused across both element types when not given (built once, not twice).

    Returns
    -------
    OverloadReport
        One entry per violation, sorted worst-first. Empty when nothing exceeds the
        threshold. NaN loadings (non-converged case, or a branch on an islanded bus) are
        never reported as violations.

    Notes
    -----
    Out-of-service (outaged) branches carry zero flow and so can never be flagged.
    """
    if not np.isfinite(threshold_percent):
        raise ValueError(f"threshold_percent must be finite, got {threshold_percent!r}")

    bad = [e for e in elements if e not in ("line", "trafo")]
    if bad:
        raise ValueError(f"elements must be 'line'/'trafo', got {bad!r}")

    # One NewtonPowerflow for both element types (it re-derives the whole Ybus).
    if npf is None and len(elements):
        npf = NewtonPowerflow(net)

    el_l, idx_l, grp_l, case_l, load_l, ika_l, rate_l = [], [], [], [], [], [], []  # type: ignore[var-annotated]
    for element in elements:
        if element not in net or len(net[element]) == 0:
            continue
        tab = compute_branch_loading(net, res, element=element, npf=npf)
        if tab.loading_percent.size == 0:
            continue
        # NaN never compares True, so non-converged / islanded entries drop out here.
        with np.errstate(invalid="ignore"):
            rows, cols = np.nonzero(tab.loading_percent > threshold_percent)
        if rows.size == 0:
            continue
        rating = (
            net.line["max_i_ka"].to_numpy(dtype=float)
            if element == "line"
            else net.trafo["sn_mva"].to_numpy(dtype=float)
        )
        groups = np.asarray(tab.groups, dtype=object)
        el_l.append(np.full(rows.size, element, dtype=object))
        idx_l.append(tab.index[rows])
        grp_l.append(groups[cols])
        case_l.append(cols)
        load_l.append(tab.loading_percent[rows, cols])
        ika_l.append(tab.i_ka[rows, cols])
        rate_l.append(rating[rows])

    if not el_l:
        e_obj: np.typing.NDArray = np.empty(0, dtype=object)
        e_f: np.typing.NDArray = np.empty(0, dtype=float)
        return OverloadReport(
            threshold_percent=float(threshold_percent),
            element=e_obj,
            index=np.empty(0, dtype=int),
            group=e_obj.copy(),
            case=np.empty(0, dtype=int),
            loading_percent=e_f,
            i_ka=e_f.copy(),
            rating=e_f.copy(),
        )

    element_a = np.concatenate(el_l)
    index_a = np.concatenate(idx_l)
    group_a = np.concatenate(grp_l)
    case_a = np.concatenate(case_l)
    load_a = np.concatenate(load_l)
    ika_a = np.concatenate(ika_l)
    rate_a = np.concatenate(rate_l)

    order = np.argsort(load_a)[::-1]  # worst violation first
    return OverloadReport(
        threshold_percent=float(threshold_percent),
        element=element_a[order],
        index=index_a[order],
        group=group_a[order],
        case=case_a[order],
        loading_percent=load_a[order],
        i_ka=ika_a[order],
        rating=rate_a[order],
    )
