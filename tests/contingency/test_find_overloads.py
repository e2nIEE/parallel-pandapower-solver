# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause
"""Validate per-contingency branch loadings against pandapower.

``compute_branch_loading`` turns the contingency solvers' bus-voltage table into
per-branch flows/currents/loadings. The reference is a real pandapower ``runpp`` on the
net with that contingency's branches actually taken out of service -- i.e. each column of
the helper's output must reproduce what an individual outaged solve writes into
``res_line`` / ``res_trafo``.
"""

import copy

import numpy as np
import pandapower as pp
import pytest
from pandapower.networks import case14

from p3s.calculateTrafoTapTable import calculate_trafo_characteristic
from p3s.contingency.find_overloads import compute_branch_loading
from p3s.contingency.solver_cpp import solve_contingencies_cpp

pytest.importorskip("p3s.nr_klu", reason="compiled nr_klu not built")


def _net_with_line_outages():
    net = case14()
    calculate_trafo_characteristic(net, inplace=True)
    net.line["outage_group"] = None
    net.trafo["outage_group"] = None
    for k, i in enumerate(net.line.index):
        net.line.loc[i, "outage_group"] = f"L{k}"
    return net


def _pandapower_reference(net, group):
    """runpp on the net with ``group``'s lines actually out of service."""
    n2 = copy.deepcopy(net)
    n2.line.loc[n2.line["outage_group"] == group, "in_service"] = False
    pp.runpp(n2)
    return n2


@pytest.mark.parametrize("element", ["line", "trafo"])
def test_branch_loading_matches_pandapower(element):
    net = _net_with_line_outages()
    res = solve_contingencies_cpp(net, reslack_islands=False, init="flat")
    tab = compute_branch_loading(net, res, element=element)

    assert tab.loading_percent.shape == (len(net[element]), len(res.groups))
    assert list(tab.groups) == list(res.groups)

    n_checked = 0
    for c, g in enumerate(res.groups):
        if not res.converged[c]:
            continue
        ref = _pandapower_reference(net, g)
        ref_tbl = ref["res_" + element]
        ref_load = ref_tbl.loading_percent.to_numpy()
        mine = tab.loading_percent[:, c]

        mask = ~np.isnan(mine) & ~np.isnan(ref_load)
        if element == "line":
            # the outaged line itself: pandapower reports it as out of service, we
            # report an explicit 0 -- compared separately below.
            mask &= ref.line.in_service.to_numpy()
        if not mask.any():
            continue
        n_checked += 1
        np.testing.assert_allclose(mine[mask], ref_load[mask], rtol=0, atol=1e-6)

    assert n_checked > 0, "no contingency columns were validated"


def test_outaged_branches_report_zero_not_stale_flow():
    """The out-of-service branch must read exactly 0, not the intact-grid flow.

    Yf/Yt are built from the INTACT Ybus, so without explicit zeroing the outaged line
    would report the current it would have carried had it stayed in service.
    """
    net = _net_with_line_outages()
    res = solve_contingencies_cpp(net, reslack_islands=False, init="flat")
    tab = compute_branch_loading(net, res, element="line")

    assert tab.outaged.any(), "fixture should have outaged branches"
    conv = np.asarray(res.converged)
    hit = tab.outaged & conv[None, :]
    assert np.all(tab.loading_percent[hit] == 0.0)
    assert np.all(tab.i_ka[hit] == 0.0)
    assert np.all(tab.p_from_mw[hit] == 0.0)


def test_nonconverged_columns_are_nan():
    net = _net_with_line_outages()
    res = solve_contingencies_cpp(net, reslack_islands=False, init="flat")
    tab = compute_branch_loading(net, res, element="line")

    bad = ~np.asarray(res.converged)
    if not bad.any():
        pytest.skip("all contingencies converged on this net")
    assert np.isnan(tab.loading_percent[:, bad]).all()


def test_summary_helpers():
    net = _net_with_line_outages()
    res = solve_contingencies_cpp(net, reslack_islands=False, init="flat")
    tab = compute_branch_loading(net, res, element="line")

    worst = tab.max_loading_percent
    assert worst.shape == (len(net.line),)
    # max_loading_percent must equal the row max of the table (ignoring NaN)
    with np.errstate(invalid="ignore"):
        expected = np.nanmax(tab.loading_percent, axis=1)
    np.testing.assert_allclose(worst, expected, equal_nan=True)

    wc = tab.worst_case
    assert wc.shape == (len(net.line),)
    for r, c in enumerate(wc):
        if c >= 0 and not np.isnan(worst[r]):
            assert tab.loading_percent[r, c] == pytest.approx(worst[r])

    rows, cols = tab.overloads(threshold=0.0)
    assert len(rows) == len(cols)
    assert np.all(tab.loading_percent[rows, cols] > 0.0)
    # NaN must never be reported as an overload
    assert not np.isnan(tab.loading_percent[rows, cols]).any()


def test_rejects_mismatched_net():
    net = _net_with_line_outages()
    res = solve_contingencies_cpp(net, reslack_islands=False, init="flat")
    with pytest.raises(ValueError):
        compute_branch_loading(net, res, element="sgen")


# --------------------------------------------------------------------------- #
# Overload screening (the actual N-1 question: what exceeds my limit?)         #
# --------------------------------------------------------------------------- #
def _net_with_real_ratings():
    """case14 with a realistic line rating so loading_percent is meaningful.

    The shipped case14 uses a placeholder max_i_ka (99999), which would put every
    loading at ~0 % and make threshold tests vacuous.
    """
    net = _net_with_line_outages()
    net.line["max_i_ka"] = 0.3
    return net


@pytest.mark.parametrize("threshold", [70.0, 100.0, 150.0])
def test_find_overloads_matches_pandapower(threshold):
    """Every reported violation must reproduce a real pandapower outaged solve."""
    from p3s.contingency.find_overloads import find_overloads

    net = _net_with_real_ratings()
    res = solve_contingencies_cpp(net, reslack_islands=False, init="flat")
    rep = find_overloads(net, res, threshold_percent=threshold, elements=("line",))

    assert rep.threshold_percent == threshold
    assert len(rep) > 0, f"fixture should violate at {threshold} %"

    # spot-check a bounded sample (a full cross-check is one runpp per violation).
    # RELATIVE tolerance: the fixture's artificially low 0.3 kA rating scales loadings to
    # ~1e4 %, which scales the Newton-tolerance residual with them, so an absolute bound
    # would be comparing against the wrong magnitude.
    for k in range(0, len(rep), max(1, len(rep) // 12)):
        ref = _pandapower_reference(net, rep.group[k])
        got = ref.res_line.loading_percent.loc[rep.index[k]]
        assert rep.loading_percent[k] == pytest.approx(got, rel=1e-6)


def test_find_overloads_threshold_is_percent_of_rating():
    """A higher threshold must be a strict subset, and everything reported exceeds it."""
    from p3s.contingency.find_overloads import find_overloads

    net = _net_with_real_ratings()
    res = solve_contingencies_cpp(net, reslack_islands=False, init="flat")

    low = find_overloads(net, res, threshold_percent=70.0, elements=("line",))
    high = find_overloads(net, res, threshold_percent=150.0, elements=("line",))

    assert np.all(low.loading_percent > 70.0)
    assert np.all(high.loading_percent > 150.0)
    assert len(high) <= len(low)

    def keys(r):
        return set(zip(r.element.tolist(), r.index.tolist(), r.case.tolist(), strict=False))

    assert keys(high) <= keys(low), "higher threshold must be a subset"


def test_find_overloads_equals_dense_table_filter():
    """The sparse report must equal filtering the dense table at the same level."""
    from p3s.contingency.find_overloads import compute_branch_loading, find_overloads

    net = _net_with_real_ratings()
    res = solve_contingencies_cpp(net, reslack_islands=False, init="flat")

    tab = compute_branch_loading(net, res, element="line")
    with np.errstate(invalid="ignore"):
        n_dense = int((tab.loading_percent > 70.0).sum())
    rep = find_overloads(net, res, threshold_percent=70.0, elements=("line",))
    assert len(rep) == n_dense


def test_find_overloads_sorted_and_clean():
    from p3s.contingency.find_overloads import find_overloads

    net = _net_with_real_ratings()
    res = solve_contingencies_cpp(net, reslack_islands=False, init="flat")
    rep = find_overloads(net, res, threshold_percent=70.0)

    assert np.all(np.diff(rep.loading_percent) <= 0), "must be worst-first"
    assert not np.isnan(rep.loading_percent).any(), "NaN is never a violation"
    # an outaged branch carries no flow, so it can never be its own violation
    for el, idx, grp in zip(rep.element, rep.index, rep.group, strict=False):
        if el == "line":
            assert net.line.loc[idx, "outage_group"] != grp

    df = rep.to_dataframe()
    assert len(df) == len(rep)
    wpb = rep.worst_per_branch()
    assert len(wpb) <= len(df)
    assert not wpb.duplicated(subset=["element", "index"]).any()


def test_find_overloads_empty_when_threshold_unreachable():
    from p3s.contingency.find_overloads import find_overloads

    net = _net_with_real_ratings()
    res = solve_contingencies_cpp(net, reslack_islands=False, init="flat")
    rep = find_overloads(net, res, threshold_percent=1e9)

    assert len(rep) == 0
    assert rep.n_violations == 0
    assert rep.to_dataframe().empty
    assert rep.worst_per_branch().empty


def test_find_overloads_rejects_bad_input():
    from p3s.contingency.find_overloads import find_overloads

    net = _net_with_real_ratings()
    res = solve_contingencies_cpp(net, reslack_islands=False, init="flat")
    with pytest.raises(ValueError):
        find_overloads(net, res, threshold_percent=np.nan)
    with pytest.raises(ValueError):
        find_overloads(net, res, elements=("line", "bogus"))
