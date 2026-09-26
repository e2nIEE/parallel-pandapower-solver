# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate per-contingency voltage-band screening against pandapower.

``find_voltage_violations`` flags every (bus, contingency) whose magnitude leaves its
band. The reference for a flagged voltage is a real pandapower ``runpp`` on the net with
that contingency's branches actually taken out of service.

case14 ships a 0.94-1.06 pu band on every bus while several PV buses hold 1.07-1.09 pu
setpoints, so the stored band already produces genuine high-voltage violations.
"""

import copy

import numpy as np
import pandapower as pp
import pytest
from pandapower.networks import case14

from p3s.calculateTrafoTapTable import calculate_trafo_characteristic
from p3s.contingency.solver_cpp import solve_contingencies_cpp
from p3s.contingency.voltage_band import find_voltage_violations

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
    """runpp on the net with ``group``'s branches actually out of service."""
    n2 = copy.deepcopy(net)
    for el in ("line", "trafo"):
        n2[el].loc[n2[el]["outage_group"] == group, "in_service"] = False
    pp.runpp(n2)
    return n2


@pytest.fixture(scope="module")
def solved():
    net = _net_with_line_outages()
    return net, solve_contingencies_cpp(net, reslack_islands=False, init="flat")


def test_stored_band_matches_pandapower(solved):
    """Every reported violation must reproduce a real pandapower outaged solve."""
    net, res = solved
    rep = find_voltage_violations(net, res)

    assert len(rep) > 0, "case14's 1.06 pu band should be violated by its PV setpoints"
    assert set(rep.kind) == {"high"}
    np.testing.assert_allclose(rep.limit_pu, 1.06)

    for k in range(len(rep)):
        ref = _pandapower_reference(net, rep.group[k])
        got = ref.res_bus.vm_pu.loc[rep.bus[k]]
        assert rep.vm_pu[k] == pytest.approx(got, abs=1e-6)
        assert got > 1.06


def test_report_equals_dense_filter(solved):
    """The sparse report must hold exactly the out-of-band entries of res.vm."""
    net, res = solved
    rep = find_voltage_violations(net, res, vm_min_pu=1.0, vm_max_pu=1.08)
    vm = res.vm
    n_low = int((vm < 1.0 - 1e-9).sum())
    n_high = int((vm > 1.08 + 1e-9).sum())

    assert n_low > 0 and n_high > 0, "fixture should violate both sides of this band"
    assert (rep.kind == "low").sum() == n_low
    assert (rep.kind == "high").sum() == n_high
    for b, c, v in zip(rep.bus, rep.case, rep.vm_pu, strict=False):
        assert vm[net.bus.index.get_loc(b), c] == v


def test_explicit_band_overrides_net_columns(solved):
    net, res = solved
    wide = find_voltage_violations(net, res, vm_min_pu=0.9, vm_max_pu=1.1)
    assert len(wide) == 0, "case14 stays inside 0.9-1.1 pu in every line outage"

    # A per-bus array is honoured positionally; a NaN leaves that side unchecked.
    vmax = np.full(len(net.bus), np.nan)
    b = int(np.nanargmax(np.nanmax(res.vm, axis=1)))  # the highest-voltage bus
    vmax[b] = 1.0
    rep = find_voltage_violations(net, res, vm_min_pu=np.nan, vm_max_pu=vmax)
    assert len(rep) > 0
    assert set(rep.bus) == {net.bus.index[b]}


def test_sorted_worst_first_and_consistent(solved):
    net, res = solved
    rep = find_voltage_violations(net, res, vm_min_pu=1.0, vm_max_pu=1.08)

    assert np.all(np.diff(rep.deviation_pu) <= 0), "must be worst-first"
    assert np.all(rep.deviation_pu > 0)
    np.testing.assert_allclose(rep.deviation_pu, np.abs(rep.vm_pu - rep.limit_pu))
    assert np.all(rep.vm_pu[rep.kind == "low"] < rep.limit_pu[rep.kind == "low"])
    assert np.all(rep.vm_pu[rep.kind == "high"] > rep.limit_pu[rep.kind == "high"])
    assert [res.groups[c] for c in rep.case] == list(rep.group)

    df = rep.to_dataframe()
    assert len(df) == len(rep)
    wpb = rep.worst_per_bus()
    assert not wpb.duplicated(subset=["bus"]).any()
    assert len(wpb) == len(set(rep.bus))


def test_tolerance_absorbs_setpoint_noise(solved):
    """A PV bus held exactly at a bound equal to its setpoint is not a violation."""
    net, res = solved
    top = float(np.nanmax(res.vm))  # the 1.09 pu generator setpoint
    assert len(find_voltage_violations(net, res, vm_min_pu=np.nan, vm_max_pu=top)) == 0
    strict = find_voltage_violations(net, res, vm_min_pu=np.nan, vm_max_pu=top - 1e-6)
    assert len(strict) > 0


def test_islanded_bus_is_never_a_violation():
    """An unserved bus carries NaN voltage and must not be screened."""
    net = _net_with_line_outages()
    # bus 7 hangs radially off trafo 6-7: outaging it islands the bus
    t = net.trafo.index[(net.trafo.hv_bus == 7) | (net.trafo.lv_bus == 7)]
    net.trafo.loc[t, "outage_group"] = "T_island"
    res = solve_contingencies_cpp(net, reslack_islands=False, init="flat")
    c = res.groups.index("T_island")
    assert not res.served[7, c], "fixture should island bus 7"

    rep = find_voltage_violations(net, res, vm_min_pu=1.0, vm_max_pu=1.0)
    assert not np.any((rep.bus == net.bus.index[7]) & (rep.case == c))
    assert not np.isnan(rep.vm_pu).any()


def test_out_of_service_bus_is_never_a_violation(solved):
    """An out-of-service bus is skipped even when ``res.V`` holds a finite value there.

    The solver marks such a bus as served and leaves a placeholder voltage in ``V`` (an
    attached generator's setpoint). A net with an out-of-service bus does not currently
    converge in ``solve_contingencies_cpp`` at all, so the mask is exercised directly: the
    solved intact-bus result is screened against a net that flags bus 7 out of service.
    """
    net, res = solved
    assert np.all(np.isfinite(res.V[7]))
    before = find_voltage_violations(net, res, vm_min_pu=0.99, vm_max_pu=1.01)
    assert net.bus.index[7] in set(before.bus), "bus 7 (1.09 pu PV) is out of band"

    oos = copy.deepcopy(net)
    oos.bus.loc[7, "in_service"] = False
    rep = find_voltage_violations(oos, res, vm_min_pu=0.99, vm_max_pu=1.01)
    assert net.bus.index[7] not in set(rep.bus)
    assert len(rep) == len(before) - int((before.bus == net.bus.index[7]).sum())


def test_nonconverged_columns_are_skipped(solved):
    net, res = solved
    res2 = copy.copy(res)
    res2.converged = np.asarray(res.converged, dtype=bool).copy()
    res2.converged[0] = False

    rep = find_voltage_violations(net, res2, vm_min_pu=1.0, vm_max_pu=1.0)
    assert len(rep) > 0
    assert 0 not in set(rep.case)


def test_rejects_bad_input(solved):
    net, res = solved
    with pytest.raises(ValueError, match="no voltage band"):
        find_voltage_violations(net, res, vm_min_pu=np.nan, vm_max_pu=np.nan)
    with pytest.raises(ValueError, match="vm_min_pu > vm_max_pu"):
        find_voltage_violations(net, res, vm_min_pu=1.1, vm_max_pu=0.9)
    with pytest.raises(ValueError, match="shape"):
        find_voltage_violations(net, res, vm_max_pu=np.ones(3))
    with pytest.raises(ValueError, match="tol_pu"):
        find_voltage_violations(net, res, tol_pu=-1.0)

    bare = copy.deepcopy(net)
    bare.bus = bare.bus.drop(columns=["min_vm_pu", "max_vm_pu"])
    with pytest.raises(ValueError, match="no voltage band"):
        find_voltage_violations(bare, res)
