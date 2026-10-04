# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the p3s failures seen on grid-bench (https://m-mirz.github.io/grid-bench/).

grid-bench drives ``NewtonPowerflowCpp`` on nets read with pandapower's own importers
(``from_mpc``, ``from_cim``) and checks the result against the case's reference solution.
Every test here builds a minimal net that isolates ONE root cause behind those failures,
solves it the way the grid-bench adapter does (see ``_solve_p3s``) and compares against
pandapower's ``runpp`` on the same net.

Root causes, and the benchmark cases each one broke:

1. ``impedance`` not stamped into the C++ Ybus -- every MATPOWER case for which
   ``from_mpc`` creates an impedance (case18/118/300, case2869/9241pegase, RTE, mvlv*).
2. Out-of-service ``gen`` still turns its bus into a PV bus -- case3120sp, case2848rte.
3. Slack only taken from ``ext_grid``; cim2pp writes it as ``gen.slack=True`` -- every
   CGMES input (fixtures and the *@cimoxide conversions).
4. Switches are not modelled and unsupplied buses are not removed -- cgmes_smallgrid
   (+ svedala / realgrid).
5. Transformer rated voltages that differ from the bus nominal voltages are ignored
   -- the *@cimoxide conversions.
6. calculateTrafoTapTable:
   a) ``_symmetrical`` shadows ``n_taps`` -> IndexError (cgmes_microgrid_be),
   b) trafo3w get no characteristic -> "cannot convert NA to integer" (cgmes_minigrid),
   c) an existing (Tabular) ``trafo_characteristic_table`` is overwritten -> phase
      shifters lost (cgmes_powerflow, pegase@cimoxide).
7. Found while writing these tests, hidden on grid-bench behind 6b: the trafo3w Ybus
   stamp itself does not match pandapower (and is NaN without magnetising branch).

The tests for 1-6 were checked to be passable: with the matching workaround applied to
the net (drop oos gens, slack gen -> ext_grid, fuse/drop switched buses, re-express the
trafo rating as a tap, hand-made characteristic rows) p3s reproduces pandapower to 1e-6.
For 7 there is no workaround; the reference is pandapower's (Kron-reduced) Ybus.
"""

import copy

import numpy as np
import pandas as pd
import pytest
from pandapower.auxiliary import pandapowerNet
from pandapower.create import (
    create_bus,
    create_empty_network,
    create_ext_grid,
    create_gen,
    create_impedance,
    create_line_from_parameters,
    create_load,
    create_switch,
    create_transformer3w_from_parameters,
    create_transformer_from_parameters,
)
from pandapower.run import rundcpp, runpp

from p3s.calculateTrafoTapTable import calculate_trafo_characteristic
from p3s.NewtonPowerflowCpp import NewtonPowerflow

LINE = dict(r_ohm_per_km=0.12, x_ohm_per_km=0.38, c_nf_per_km=9.5, max_i_ka=1.0)


def _line(net, fb, tb, length_km=10.0):
    return create_line_from_parameters(net, fb, tb, length_km=length_km, **LINE)


def _reference(net: pandapowerNet) -> np.ndarray:
    """pandapower's flat-start solution as a complex voltage per row of net.bus (NaN = unsupplied)."""
    ref = copy.deepcopy(net)
    runpp(ref, init="flat", calculate_voltage_angles=True, tolerance_mva=1e-8, max_iteration=30)
    return ref.res_bus.vm_pu.values * np.exp(1j * np.deg2rad(ref.res_bus.va_degree.values))


def _solve_p3s(net: pandapowerNet) -> tuple[NewtonPowerflow, np.ndarray]:
    """Solve exactly like grid-bench's p3s adapter: flat start, no voltage band."""
    calculate_trafo_characteristic(net, inplace=True)
    npf = NewtonPowerflow(net)
    v0 = np.array(npf._initial_voltage, dtype=np.complex128)
    v0[npf.busses["pq"]] = 1.0
    v = npf.calculate(net, init="flat", voltage=v0, tolerance=1e-8, max_iterations=30, voltage_band=None)
    return npf, v


def _assert_matches_pandapower(net: pandapowerNet, atol: float = 1e-6):
    expected = _reference(net)
    _, v = _solve_p3s(net)
    supplied = np.isfinite(expected)
    assert supplied.any()
    np.testing.assert_allclose(np.abs(v[supplied]), np.abs(expected[supplied]), atol=atol, err_msg="vm_pu")
    np.testing.assert_allclose(
        np.angle(v[supplied], deg=True), np.angle(expected[supplied], deg=True), atol=atol * 100, err_msg="va_degree"
    )


# --- 1. impedance -------------------------------------------------------------------------


def _net_impedance() -> pandapowerNet:
    net = create_empty_network(sn_mva=1.0)
    b = [create_bus(net, vn_kv=110.0) for _ in range(4)]
    create_ext_grid(net, b[0])
    _line(net, b[0], b[1])
    # asymmetric (network equivalent) and symmetric impedance
    create_impedance(net, b[1], b[2], rft_pu=0.02, xft_pu=0.08, rtf_pu=0.03, xtf_pu=0.11, sn_mva=25.0)
    create_impedance(net, b[2], b[3], rft_pu=0.05, xft_pu=0.15, rtf_pu=0.05, xtf_pu=0.15, sn_mva=10.0)
    create_load(net, b[2], p_mw=4.0, q_mvar=1.5)
    create_load(net, b[3], p_mw=8.0, q_mvar=3.0)
    return net


def test_impedance_in_cpp_ybus():
    net = _net_impedance()
    ref = copy.deepcopy(net)
    runpp(ref)
    npf = NewtonPowerflow(net)
    np.testing.assert_allclose(npf._YBus.toarray(), ref._ppc["internal"]["Ybus"].toarray(), atol=1e-10)


def test_impedance_powerflow():
    _assert_matches_pandapower(_net_impedance())


# --- 2. out-of-service gen ----------------------------------------------------------------


def _net_feeder(n_bus: int = 4, vn_kv: float = 110.0) -> tuple[pandapowerNet, list[int]]:
    net = create_empty_network(sn_mva=1.0)
    b = [create_bus(net, vn_kv=vn_kv) for _ in range(n_bus)]
    for i in range(n_bus - 1):
        _line(net, b[i], b[i + 1])
    for i in range(1, n_bus):
        create_load(net, b[i], p_mw=10.0, q_mvar=4.0)
    return net, b


def test_out_of_service_gen_is_not_pv():
    """An out-of-service gen must not hold its bus voltage (pandapower: bus stays PQ)."""
    net, b = _net_feeder()
    create_ext_grid(net, b[0])
    create_gen(net, b[2], p_mw=5.0, vm_pu=1.05, in_service=False)
    _assert_matches_pandapower(net)


def test_out_of_service_gen_does_not_set_bus_voltage():
    """With an oos and an in-service gen on one bus, the in-service set point applies."""
    net, b = _net_feeder()
    create_ext_grid(net, b[0])
    create_gen(net, b[2], p_mw=0.0, vm_pu=1.10, in_service=False)  # first in table -> picked by .first()
    create_gen(net, b[2], p_mw=5.0, vm_pu=1.02)
    _assert_matches_pandapower(net)


# --- 3. slack gen without ext_grid ----------------------------------------------------------


def test_slack_gen_without_ext_grid():
    """cim2pp writes the CGMES slack as gen.slack=True and creates no ext_grid."""
    net, b = _net_feeder()
    create_gen(net, b[0], p_mw=0.0, vm_pu=1.02, slack=True)
    create_gen(net, b[3], p_mw=8.0, vm_pu=1.0)
    _assert_matches_pandapower(net)


def test_slack_gen_is_reference_bus():
    net, b = _net_feeder()
    create_gen(net, b[0], p_mw=0.0, vm_pu=1.02, slack=True)
    npf = NewtonPowerflow(net)
    assert list(npf.busses["ref"]) == [b[0]]
    assert b[0] not in npf.busses["pv"]


# --- 2./3. gen and ext_grid results ---------------------------------------------------------


def _net_results_oos_gen() -> pandapowerNet:
    net, b = _net_feeder()
    create_ext_grid(net, b[0])
    create_gen(net, b[2], p_mw=5.0, vm_pu=1.02)
    create_gen(net, b[3], p_mw=3.0, vm_pu=1.05, in_service=False)  # res_gen row must be all 0
    return net


def _net_results_slack_gen() -> pandapowerNet:
    net, b = _net_feeder()
    create_gen(net, b[0], p_mw=2.0, vm_pu=1.02, slack=True)  # P is a result, not the 2 MW set point
    create_gen(net, b[3], p_mw=8.0, vm_pu=1.0)
    return net


def _net_results_shared_bus() -> pandapowerNet:
    """Two gens on one bus share its Q in proportion to their reactive range (pfsoln._update_q)."""
    net, b = _net_feeder()
    create_ext_grid(net, b[0])
    create_gen(net, b[2], p_mw=4.0, vm_pu=1.03, min_q_mvar=-10.0, max_q_mvar=30.0)
    create_gen(net, b[2], p_mw=2.0, vm_pu=1.03, min_q_mvar=-5.0, max_q_mvar=5.0)
    return net


def _results_cpp(net: pandapowerNet):
    npf, v = _solve_p3s(net)
    npf._parse_results(net, v)  # the C++ calculate() only returns the voltages


def _results_python(net: pandapowerNet):
    from p3s.NewtonPowerflow import NewtonPowerflow as NewtonPowerflowPy

    calculate_trafo_characteristic(net, inplace=True)
    NewtonPowerflowPy(net).calculate(net, init="flat", tolerance=1e-8, max_iterations=30)


@pytest.mark.parametrize("solve", [_results_cpp, _results_python], ids=["cpp", "python"])
@pytest.mark.parametrize(
    "make_net",
    [_net_results_oos_gen, _net_results_slack_gen, _net_results_shared_bus],
    ids=["oos_gen", "slack_gen", "shared_bus"],
)
def test_gen_and_ext_grid_results(make_net, solve):
    net = make_net()
    ref = copy.deepcopy(net)
    runpp(ref, init="flat", calculate_voltage_angles=True, tolerance_mva=1e-8)
    solve(net)
    cols = ["p_mw", "q_mvar", "vm_pu", "va_degree"]
    np.testing.assert_allclose(net.res_gen[cols].to_numpy(float), ref.res_gen[cols].to_numpy(float), atol=1e-5)
    if len(net.ext_grid):
        cols = ["p_mw", "q_mvar"]
        np.testing.assert_allclose(
            net.res_ext_grid[cols].to_numpy(float), ref.res_ext_grid[cols].to_numpy(float), atol=1e-5
        )


# --- 4. switches / unsupplied buses ---------------------------------------------------------


def test_closed_bus_bus_switch_connects_buses():
    """CGMES node-breaker models join buses with closed bus-bus switches (cgmes_smallgrid)."""
    net, b = _net_feeder(3)
    create_ext_grid(net, b[0])
    b_sw = create_bus(net, vn_kv=110.0)
    create_switch(net, b[2], b_sw, et="b", closed=True)
    create_load(net, b_sw, p_mw=5.0, q_mvar=2.0)
    _assert_matches_pandapower(net)


def test_open_bus_bus_switch_leaves_unsupplied_bus():
    """A bus behind an open switch is unsupplied; pandapower drops it, p3s must not go singular."""
    net, b = _net_feeder(3)
    create_ext_grid(net, b[0])
    b_iso = create_bus(net, vn_kv=110.0)
    create_switch(net, b[2], b_iso, et="b", closed=False)
    create_load(net, b_iso, p_mw=5.0, q_mvar=2.0)
    _assert_matches_pandapower(net)


def test_open_line_switch_disconnects_line():
    """A line with an open line switch carries no flow (ring opened at one end)."""
    net, b = _net_feeder(4)
    create_ext_grid(net, b[0])
    ring = _line(net, b[0], b[3])
    create_switch(net, b[3], ring, et="l", closed=False)
    _assert_matches_pandapower(net)


# Design (p3s.topology): buses joined by closed bus-bus switches are fused into one node; the
# switch currents follow from KCL after the solve (weighted by z_ohm in switch loops); an open
# line/trafo/trafo3w switch Kron-eliminates that terminal, so the branch stays energised from its
# other end(s) as with pandapower's auxiliary bus.

SOLVERS = pytest.mark.parametrize("solve", [_results_cpp, _results_python], ids=["cpp", "python"])


def _net_switch_feeder(n_switches: int = 1, z_ohm=(0.0,)) -> tuple[pandapowerNet, list[int]]:
    """ext_grid - line - b1 =switch(es)= b2 - line - b3, loads at b2 and b3."""
    net = create_empty_network(sn_mva=1.0)
    b = [create_bus(net, vn_kv=110.0) for _ in range(4)]
    create_ext_grid(net, b[0])
    _line(net, b[0], b[1])
    _line(net, b[2], b[3])
    create_load(net, b[2], p_mw=20.0, q_mvar=8.0)
    create_load(net, b[3], p_mw=10.0, q_mvar=4.0)
    for k in range(n_switches):
        create_switch(net, b[1], b[2], et="b", closed=True, z_ohm=z_ohm[k], in_ka=0.5)
    return net, b


def _current_into_b2_side(ref: pandapowerNet, b: list[int]) -> complex:
    """kA flowing from b1 to b2 according to pandapower's (fused) solution: everything the elements
    at b2 draw, i.e. its load, the line to b3 and minus a generator at b2 if any."""
    s = (
        ref.res_load.p_mw[0]
        + 1j * ref.res_load.q_mvar[0]
        + ref.res_line.p_from_mw[1]
        + 1j * ref.res_line.q_from_mvar[1]
    )
    gens_at_b2 = ref.gen.index[ref.gen.bus == b[2]] if len(ref.gen) else []
    for g in gens_at_b2:
        s -= ref.res_gen.p_mw[g] + 1j * ref.res_gen.q_mvar[g]
    v_kv = ref.res_bus.vm_pu[b[2]] * np.exp(1j * np.deg2rad(ref.res_bus.va_degree[b[2]])) * ref.bus.vn_kv[b[2]]
    return np.conj(s / (np.sqrt(3) * v_kv))


def _reference_pp(net: pandapowerNet) -> pandapowerNet:
    ref = copy.deepcopy(net)
    runpp(ref, init="flat", calculate_voltage_angles=True, tolerance_mva=1e-8)
    return ref


def _assert_res_matches(net, ref, table, columns=None, rtol=1e-6, atol=1e-6):
    columns = columns or list(ref[table].columns)
    for column in columns:
        np.testing.assert_allclose(
            net[table][column].to_numpy(float),
            ref[table][column].to_numpy(float),
            rtol=rtol,
            atol=atol,
            err_msg=f"{table}.{column}",
        )


@SOLVERS
def test_bus_bus_switch_current_and_fused_bus_results(solve):
    """Tree: the KCL switch current equals what flows downstream of the switch, and every fused bus
    reports its own elements (pandapower's res_bus)."""
    net, b = _net_switch_feeder()
    ref = _reference_pp(net)
    solve(net)
    _assert_res_matches(net, ref, "res_bus")
    i_expected = _current_into_b2_side(ref, b)
    assert net.res_switch.i_ka[0] == pytest.approx(abs(i_expected), rel=1e-6)
    assert net.res_switch.loading_percent[0] == pytest.approx(abs(i_expected) / 0.5 * 100, rel=1e-6)
    s_expected = np.sqrt(3) * ref.res_bus.vm_pu[b[1]] * 110.0 * np.exp(1j * np.deg2rad(ref.res_bus.va_degree[b[1]]))
    s_expected *= np.conj(i_expected)
    assert net.res_switch.p_from_mw[0] == pytest.approx(s_expected.real, rel=1e-6)
    assert net.res_switch.q_from_mvar[0] == pytest.approx(s_expected.imag, rel=1e-6)
    assert net.res_switch.p_to_mw[0] == pytest.approx(-s_expected.real, rel=1e-6)


@SOLVERS
@pytest.mark.parametrize("z_ohm, share", [((0.0, 0.0), 0.5), ((0.010, 0.030), 0.75), ((0.0, 0.020), 1.0)])
def test_switch_loop_splits_by_contact_impedance(z_ohm, share, solve):
    """Two parallel closed switches (a zero-impedance loop): the current splits by 1/z_ohm; a z_ohm
    of 0 counts as a (much smaller) placeholder, so it takes the whole current against z > 0."""
    net, b = _net_switch_feeder(n_switches=2, z_ohm=z_ohm)
    ref = _reference_pp(_net_switch_feeder()[0])  # pandapower fuses only z_ohm == 0; same flows
    solve(net)
    total = abs(_current_into_b2_side(ref, b))
    np.testing.assert_allclose(
        net.res_switch.i_ka.to_numpy(float), [share * total, (1 - share) * total], rtol=1e-4, atol=1e-4 * total
    )


@SOLVERS
def test_switch_between_two_generators(solve):
    """PV - switch - PV: fine after fusion (the +-jX switch model is singular here). The units share
    the node's Q by pandapower's rule; the switch current follows from that split."""
    net, b = _net_switch_feeder()
    create_gen(net, b[1], p_mw=5.0, vm_pu=1.01)
    create_gen(net, b[2], p_mw=5.0, vm_pu=1.01)
    ref = _reference_pp(net)
    solve(net)
    _assert_res_matches(net, ref, "res_bus", ["vm_pu", "va_degree"])
    _assert_res_matches(net, ref, "res_gen")
    assert net.res_switch.i_ka[0] == pytest.approx(abs(_current_into_b2_side(ref, b)), rel=1e-6)


def _net_open_ends() -> pandapowerNet:
    """A meshed net with an open line switch, an open trafo switch (lv side) and a trafo3w with an
    open mv terminal -- every branch stays connected at its other end(s)."""
    net = create_empty_network(sn_mva=1.0)
    hv = [create_bus(net, vn_kv=110.0) for _ in range(3)]
    mv = [create_bus(net, vn_kv=20.0) for _ in range(2)]
    lv = create_bus(net, vn_kv=10.0)
    create_ext_grid(net, hv[0])
    _line(net, hv[0], hv[1])
    _line(net, hv[1], hv[2])
    ring = create_line_from_parameters(net, hv[0], hv[2], length_km=40.0, **LINE)
    create_switch(net, hv[2], ring, et="l", closed=False)
    t_params = dict(
        sn_mva=40.0, vn_hv_kv=110.0, vn_lv_kv=20.0, vk_percent=12.0, vkr_percent=0.4, pfe_kw=20.0, i0_percent=0.05
    )
    create_transformer_from_parameters(net, hv[1], mv[0], **t_params)
    t_open = create_transformer_from_parameters(net, hv[2], mv[0], **t_params)
    create_switch(net, mv[0], t_open, et="t", closed=False)
    t3 = create_transformer3w_from_parameters(
        net,
        hv[2],
        mv[1],
        lv,
        vn_hv_kv=110.0,
        vn_mv_kv=20.0,
        vn_lv_kv=10.0,
        sn_hv_mva=40.0,
        sn_mv_mva=25.0,
        sn_lv_mva=15.0,
        vk_hv_percent=12.0,
        vk_mv_percent=10.0,
        vk_lv_percent=8.0,
        vkr_hv_percent=0.4,
        vkr_mv_percent=0.35,
        vkr_lv_percent=0.3,
        pfe_kw=25.0,
        i0_percent=0.06,
    )
    create_switch(net, mv[1], t3, et="t3", closed=False)
    _line(net, mv[0], mv[1], length_km=2.0)  # mv[1] fed from mv[0], not through the open trafo3w port
    create_load(net, mv[0], p_mw=15.0, q_mvar=5.0)
    create_load(net, mv[1], p_mw=5.0, q_mvar=2.0)
    create_load(net, lv, p_mw=3.0, q_mvar=1.0)
    return net


@SOLVERS
def test_open_branch_switches_keep_charging(solve):
    """Open line / trafo / trafo3w switches: as pandapower (auxiliary bus at the open end), the
    branch keeps its charging / magnetising and reports the open end's voltage."""
    net = _net_open_ends()
    ref = _reference_pp(net)
    solve(net)
    _assert_res_matches(net, ref, "res_bus")
    _assert_res_matches(net, ref, "res_ext_grid")
    _assert_res_matches(net, ref, "res_line")
    _assert_res_matches(net, ref, "res_trafo")
    # p3s reports the bus voltage at an open trafo3w terminal (pandapower: its auxiliary bus)
    t3_cols = [c for c in ref.res_trafo3w.columns if not c.endswith(("mv_pu", "mv_degree"))]
    _assert_res_matches(net, ref, "res_trafo3w", t3_cols)
    assert net.res_line.i_to_ka[2] == 0.0 and net.res_line.i_from_ka[2] > 0.0  # charging only
    # res_switch: open switches carry nothing
    np.testing.assert_allclose(net.res_switch.i_ka.to_numpy(float), 0.0, atol=1e-12)


@SOLVERS
def test_line_losses_include_charging(solve):
    """res_line.pl_mw / ql_mvar = what enters the line at both ends, as pandapower. A lightly loaded
    cable ring generates reactive power (ql < 0); on the middle cable both ends push Q out of it,
    where the former |abs(q_to) - abs(q_from)| reported 0.33 Mvar instead of -23 Mvar."""
    net = create_empty_network(sn_mva=1.0)
    b = [create_bus(net, vn_kv=110.0) for _ in range(3)]
    create_ext_grid(net, b[0])
    cable = dict(length_km=20.0, r_ohm_per_km=0.03, x_ohm_per_km=0.12, c_nf_per_km=300.0, max_i_ka=1.0)
    for f, t in ((0, 1), (1, 2), (2, 0)):
        create_line_from_parameters(net, b[f], b[t], **cable)
    create_load(net, b[1], p_mw=2.0, q_mvar=0.5)
    ref = _reference_pp(net)
    solve(net)
    _assert_res_matches(net, ref, "res_line")
    assert (net.res_line.ql_mvar < 0).all()


@SOLVERS
def test_closed_line_switch_current_is_the_line_end_current(solve):
    net, b = _net_feeder(3)
    create_ext_grid(net, b[0])
    line = net.line.index[1]
    create_switch(net, b[1], line, et="l", closed=True, in_ka=0.4)  # from end
    create_switch(net, b[2], line, et="l", closed=True)  # to end
    solve(net)
    assert net.res_switch.i_ka[0] == pytest.approx(net.res_line.i_from_ka[line], rel=1e-12)
    assert net.res_switch.i_ka[1] == pytest.approx(net.res_line.i_to_ka[line], rel=1e-12)
    assert net.res_switch.p_from_mw[0] == pytest.approx(net.res_line.p_from_mw[line], rel=1e-12)
    assert net.res_switch.loading_percent[0] == pytest.approx(net.res_line.i_from_ka[line] / 0.4 * 100, rel=1e-12)


@SOLVERS
def test_unsupplied_area(solve):
    """Buses behind an open switch: NaN voltages, no flows, NaN switch results -- like pandapower --
    and the rest of the grid solves normally."""
    net, b = _net_feeder(3)
    create_ext_grid(net, b[0])
    iso = [create_bus(net, vn_kv=110.0) for _ in range(3)]
    create_switch(net, b[2], iso[0], et="b", closed=False)
    create_switch(net, iso[0], iso[1], et="b", closed=True)
    _line(net, iso[1], iso[2])
    create_load(net, iso[2], p_mw=5.0, q_mvar=2.0)
    ref = _reference_pp(net)
    solve(net)
    _assert_res_matches(net, ref, "res_bus")
    _assert_res_matches(net, ref, "res_line")
    assert net.res_switch.i_ka[0] == 0.0  # open
    assert np.isnan(net.res_switch.i_ka[1])  # closed, but unsupplied


def test_calculate_returns_bus_voltages():
    """NewtonPowerflowCpp.calculate solves on nodes but returns one voltage per net.bus row."""
    net, b = _net_switch_feeder()
    iso = create_bus(net, vn_kv=110.0)
    create_switch(net, b[3], iso, et="b", closed=False)
    npf, v = _solve_p3s(net)
    assert npf._YBus.shape == (3, 3)  # b1/b2 fused, iso unsupplied
    assert len(v) == len(net.bus)
    assert v[b[1]] == v[b[2]] and np.isnan(v[iso])
    # a start voltage per bus is accepted as well
    v2 = npf.calculate(net, init="flat", voltage=v, tolerance=1e-8, voltage_band=None)
    np.testing.assert_allclose(v2, v, atol=1e-10)


# --- 5. trafo rated voltage != bus nominal voltage ------------------------------------------


def _net_trafo(vn_hv_kv: float, vn_lv_kv: float, net_sn_mva: float = 1.0, **tap) -> pandapowerNet:
    net = create_empty_network(sn_mva=net_sn_mva)
    hv = create_bus(net, vn_kv=110.0)
    lv = create_bus(net, vn_kv=20.0)
    lv2 = create_bus(net, vn_kv=20.0)
    create_ext_grid(net, hv)
    create_transformer_from_parameters(
        net,
        hv,
        lv,
        sn_mva=40.0,
        vn_hv_kv=vn_hv_kv,
        vn_lv_kv=vn_lv_kv,
        vk_percent=12.0,
        vkr_percent=0.4,
        pfe_kw=20.0,
        i0_percent=0.05,
        **tap,
    )
    create_line_from_parameters(
        net, lv, lv2, length_km=2.0, r_ohm_per_km=0.16, x_ohm_per_km=0.11, c_nf_per_km=250.0, max_i_ka=0.4
    )
    create_load(net, lv2, p_mw=15.0, q_mvar=5.0)
    return net


@pytest.mark.parametrize(
    "vn_hv_kv, vn_lv_kv",
    [
        (110.0, 20.0),  # nominal -- passes today, guards the fix
        (115.0, 20.0),  # off-nominal hv rating (like case14@cimoxide: 0.932 / 1.0)
        (110.0, 21.0),  # off-nominal lv rating (also changes the impedance base)
        (241.638 / 230 * 110, 20.0),  # case300@cimoxide ratio
    ],
)
def test_trafo_rated_voltage_differs_from_bus(vn_hv_kv, vn_lv_kv):
    net = _net_trafo(vn_hv_kv, vn_lv_kv)
    ref = copy.deepcopy(net)
    runpp(ref)
    calculate_trafo_characteristic(net, inplace=True)
    npf = NewtonPowerflow(net)
    np.testing.assert_allclose(npf._YBus.toarray(), ref._ppc["internal"]["Ybus"].toarray(), atol=1e-8)
    _assert_matches_pandapower(_net_trafo(vn_hv_kv, vn_lv_kv))


RATIO_TAP = dict(tap_changer_type="Ratio", tap_neutral=0, tap_min=-5, tap_max=5, tap_step_percent=1.25)


@pytest.mark.parametrize(
    "tap",
    [{}, dict(tap_side="hv", tap_pos=3, **RATIO_TAP), dict(tap_side="lv", tap_pos=-2, **RATIO_TAP)],
    ids=["no_tap", "hv_tap", "lv_tap"],
)
@pytest.mark.parametrize("net_sn_mva", [1.0, 100.0])
def test_trafo_rated_voltage_with_tap_and_base_power(net_sn_mva, tap):
    """The rated-voltage ratio must combine with a tap on either side, and every per-unit
    quantity (incl. the iron losses pfe_kw) must follow net.sn_mva."""
    net = _net_trafo(115.0, 21.0, net_sn_mva, **tap)
    ref = copy.deepcopy(net)
    runpp(ref)
    calculate_trafo_characteristic(net, inplace=True)
    npf = NewtonPowerflow(net)
    np.testing.assert_allclose(npf._YBus.toarray(), ref._ppc["internal"]["Ybus"].toarray(), rtol=1e-10, atol=1e-10)
    _assert_matches_pandapower(_net_trafo(115.0, 21.0, net_sn_mva, **tap))


# --- 6. calculateTrafoTapTable -----------------------------------------------------------


def _net_two_trafos(tap_changer_type: str, **tap) -> pandapowerNet:
    """Two parallel trafos, only the second one with the tap changer under test -- the
    mask length (2) then differs from the number of trafos of that type (1)."""
    net = create_empty_network(sn_mva=1.0)
    hv = create_bus(net, vn_kv=110.0)
    lv = create_bus(net, vn_kv=20.0)
    create_ext_grid(net, hv)
    params = dict(
        sn_mva=40.0, vn_hv_kv=110.0, vn_lv_kv=20.0, vk_percent=12.0, vkr_percent=0.4, pfe_kw=20.0, i0_percent=0.05
    )
    create_transformer_from_parameters(net, hv, lv, **params)
    create_transformer_from_parameters(net, hv, lv, tap_changer_type=tap_changer_type, **tap, **params)
    create_load(net, lv, p_mw=30.0, q_mvar=10.0)
    return net


def test_symmetrical_tap_changer_does_not_crash():
    net = _net_two_trafos(
        "Symmetrical",
        tap_side="hv",
        tap_neutral=0,
        tap_min=-10,
        tap_max=10,
        tap_pos=3,
        tap_step_percent=1.5,
        tap_step_degree=0.0,
    )
    calculate_trafo_characteristic(net, inplace=True)  # IndexError: Boolean index has wrong length


def test_symmetrical_tap_changer_powerflow():
    """pandapower treats "Symmetrical" like "Ratio" (complex tap): here a 1.045 ratio, no shift."""
    net = _net_two_trafos(
        "Symmetrical",
        tap_side="hv",
        tap_neutral=0,
        tap_min=-10,
        tap_max=10,
        tap_pos=3,
        tap_step_percent=1.5,
        tap_step_degree=0.0,
    )
    _assert_matches_pandapower(net)


def _net_trafo3w(
    pfe_kw: float = 25.0, i0_percent: float = 0.06, net_sn_mva: float = 1.0, vn=(110.0, 20.0, 10.0), **extra
) -> pandapowerNet:
    net = create_empty_network(sn_mva=net_sn_mva)
    hv = create_bus(net, vn_kv=110.0)
    mv = create_bus(net, vn_kv=20.0)
    lv = create_bus(net, vn_kv=10.0)
    create_ext_grid(net, hv)
    create_transformer3w_from_parameters(
        net,
        hv,
        mv,
        lv,
        vn_hv_kv=vn[0],
        vn_mv_kv=vn[1],
        vn_lv_kv=vn[2],
        sn_hv_mva=40.0,
        sn_mv_mva=25.0,
        sn_lv_mva=15.0,
        vk_hv_percent=12.0,
        vk_mv_percent=10.0,
        vk_lv_percent=8.0,
        vkr_hv_percent=0.4,
        vkr_mv_percent=0.35,
        vkr_lv_percent=0.3,
        pfe_kw=pfe_kw,
        i0_percent=i0_percent,
        **extra,
    )
    create_load(net, mv, p_mw=15.0, q_mvar=5.0)
    create_load(net, lv, p_mw=8.0, q_mvar=2.0)
    return net


def _kron_reduced_ybus(ref: pandapowerNet, net: pandapowerNet) -> np.ndarray:
    """pandapower's Ybus with its auxiliary (trafo3w star) buses eliminated."""
    y = ref._ppc["internal"]["Ybus"].toarray()
    keep = ref._pd2ppc_lookups["bus"][net.bus.index.values]
    star = np.setdiff1d(np.arange(y.shape[0]), keep)
    return y[np.ix_(keep, keep)] - y[np.ix_(keep, star)] @ np.linalg.solve(y[np.ix_(star, star)], y[np.ix_(star, keep)])


TRAFO3W_LOSSES = pytest.mark.parametrize("pfe_kw, i0_percent", [(25.0, 0.06), (0.0, 0.0)], ids=["mag", "no_mag"])


def test_trafo3w_without_characteristic_table_builds():
    """trafo3w.id_characteristic_table is NA unless a table exists (cgmes_minigrid)."""
    net = _net_trafo3w()
    calculate_trafo_characteristic(net, inplace=True)
    NewtonPowerflow(net)  # ValueError: cannot convert NA to integer


@TRAFO3W_LOSSES
def test_trafo3w_ybus_matches_pandapower(pfe_kw, i0_percent):
    """Not visible on grid-bench (the NA crash came first): the former trafo3w stamp was off by
    factors 0.5-2.5 against pandapower, and all-NaN without magnetising branch (pfe_kw =
    i0_percent = 0, the star-node elimination divided by y_mag).

    pandapower models the trafo3w with an auxiliary star bus; Kron-reduce it away to
    compare against p3s's bus-only Ybus."""
    net = _net_trafo3w(pfe_kw, i0_percent)
    ref = copy.deepcopy(net)
    runpp(ref)
    calculate_trafo_characteristic(net, inplace=True)
    npf = NewtonPowerflow(net)
    np.testing.assert_allclose(npf._YBus.toarray(), _kron_reduced_ybus(ref, net), atol=1e-8)


@TRAFO3W_LOSSES
def test_trafo3w_powerflow(pfe_kw, i0_percent):
    _assert_matches_pandapower(_net_trafo3w(pfe_kw, i0_percent))


T3_TAP = dict(tap_neutral=0, tap_min=-8, tap_max=8, tap_step_percent=1.5, tap_changer_type="Ratio")
# pandapower drops a tap_at_star_point tap whose tap_step_degree is NaN (its star-point correction
# multiplies by exp(1j*deg2rad(NaN))), so the star-point cases set tap_step_degree=0 explicitly.
T3_STAR = dict(tap_at_star_point=True, tap_step_degree=0.0, **T3_TAP)
TRAFO3W_CASES = {
    "plain": {},
    "net_sn_100": dict(net_sn_mva=100.0),
    "off_nominal": dict(vn=(115.0, 21.0, 10.5)),
    "shifts": dict(shift_mv_degree=150.0, shift_lv_degree=330.0),
    "tap_hv": dict(tap_side="hv", tap_pos=3, **T3_TAP),
    "tap_mv": dict(tap_side="mv", tap_pos=-2, **T3_TAP),
    "tap_lv": dict(tap_side="lv", tap_pos=2, **T3_TAP),
    "tap_hv_star": dict(tap_side="hv", tap_pos=3, **T3_STAR),
    "tap_mv_star": dict(tap_side="mv", tap_pos=-2, **T3_STAR),
    "tap_lv_star": dict(tap_side="lv", tap_pos=2, **T3_STAR),
    "phase_tap_mv_star": dict(tap_side="mv", tap_pos=3, **{**T3_STAR, "tap_step_degree": 30.0}),
    "ideal_lv_shifts": dict(
        tap_side="lv",
        tap_pos=-3,
        tap_neutral=0,
        tap_min=-8,
        tap_max=8,
        tap_step_degree=2.5,
        tap_changer_type="Ideal",
        shift_mv_degree=150.0,
        shift_lv_degree=330.0,
    ),
    "everything": dict(vn=(115.0, 21.0, 10.5), tap_side="mv", tap_pos=-2, shift_mv_degree=30.0, **T3_STAR),
}
LOSS_SIDES = pytest.mark.parametrize("loss_side", ["hv", "mv", "lv", "star"])


@LOSS_SIDES
@pytest.mark.parametrize("case", list(TRAFO3W_CASES))
def test_trafo3w_ybus_and_dc_match_pandapower(case, loss_side):
    """AC stamp, DC B-matrix and DC phase-shift injection against pandapower (runpp/rundcpp with
    trafo3w_losses=loss_side), for every loss side and tap placement."""
    net = _net_trafo3w(**TRAFO3W_CASES[case])
    net.trafo3w["loss_side"] = loss_side
    ref = copy.deepcopy(net)
    runpp(ref, trafo3w_losses=loss_side)
    dc = copy.deepcopy(net)
    rundcpp(dc, trafo3w_losses=loss_side, calculate_voltage_angles=True)

    calculate_trafo_characteristic(net, inplace=True)
    npf = NewtonPowerflow(net)
    np.testing.assert_allclose(npf._YBus.toarray(), _kron_reduced_ybus(ref, net), rtol=1e-10, atol=1e-10)

    # DC: B theta = P + p_shift with the slack (bus 0) at 0 deg
    pvpq = np.arange(1, len(net.bus))
    p = (npf._sBus.real + npf._p_shift)[pvpq]
    theta = np.linalg.solve(npf._Bbus.toarray()[np.ix_(pvpq, pvpq)], p)
    va_dc = (np.rad2deg(theta) - dc.res_bus.va_degree.values[pvpq] + 180.0) % 360.0 - 180.0
    np.testing.assert_allclose(va_dc, 0.0, atol=1e-8)


def test_trafo3w_loss_side_default_is_pandapower():
    """Without a loss_side column the magnetising branch sits on the hv branch (trafo3w_losses="hv")."""
    net = _net_trafo3w()
    assert "loss_side" not in net.trafo3w
    ref = copy.deepcopy(net)
    runpp(ref)  # pandapower's default trafo3w_losses="hv"
    calculate_trafo_characteristic(net, inplace=True)
    np.testing.assert_allclose(NewtonPowerflow(net)._YBus.toarray(), _kron_reduced_ybus(ref, net), atol=1e-10)


def _net_trafo3w_parallel(in_service: bool) -> pandapowerNet:
    """Two trafo3w in parallel, the second one switchable -- the grid stays connected without it."""
    net = _net_trafo3w(tap_side="mv", tap_pos=-2, shift_lv_degree=150.0, **T3_TAP)
    t = net.trafo3w.iloc[0]
    create_transformer3w_from_parameters(
        net,
        t.hv_bus,
        t.mv_bus,
        t.lv_bus,
        vn_hv_kv=110.0,
        vn_mv_kv=20.0,
        vn_lv_kv=10.0,
        sn_hv_mva=30.0,
        sn_mv_mva=20.0,
        sn_lv_mva=10.0,
        vk_hv_percent=11.0,
        vk_mv_percent=9.0,
        vk_lv_percent=7.0,
        vkr_hv_percent=0.35,
        vkr_mv_percent=0.3,
        vkr_lv_percent=0.25,
        pfe_kw=18.0,
        i0_percent=0.05,
        shift_lv_degree=150.0,
        in_service=in_service,
    )
    return net


@pytest.mark.parametrize("solve", ["cpp", "python"])
@pytest.mark.parametrize("in_service", [True, False], ids=["both_in_service", "one_out_of_service"])
def test_res_trafo3w_matches_pandapower(in_service, solve):
    net = _net_trafo3w_parallel(in_service)
    ref = copy.deepcopy(net)
    runpp(ref, init="flat", calculate_voltage_angles=True, tolerance_mva=1e-8)
    if solve == "cpp":
        npf, v = _solve_p3s(net)
        npf._parse_results(net, v)
    else:
        from p3s.NewtonPowerflow import NewtonPowerflow as NewtonPowerflowPy

        calculate_trafo_characteristic(net, inplace=True)
        NewtonPowerflowPy(net).calculate(net, init="flat", tolerance=1e-8, max_iterations=30)

    on = net.trafo3w.in_service.to_numpy(bool)
    for column in ref.res_trafo3w.columns:
        expected = ref.res_trafo3w[column].to_numpy(float)
        actual = net.res_trafo3w[column].to_numpy(float)
        np.testing.assert_allclose(actual[on], expected[on], rtol=1e-6, atol=1e-6, err_msg=column)
    # out of service: no flow, no loading, no internal voltage (pandapower's aux bus is out of service)
    off = ~on
    for column in ("p_hv_mw", "q_mv_mvar", "p_lv_mw", "pl_mw", "i_hv_ka", "loading_percent"):
        np.testing.assert_allclose(net.res_trafo3w[column].to_numpy(float)[off], 0.0, atol=1e-12, err_msg=column)
    assert np.isnan(net.res_trafo3w.vm_internal_pu.to_numpy(float)[off]).all()


def test_out_of_service_trafo3w_is_absent():
    net = _net_trafo3w_parallel(in_service=False)
    calculate_trafo_characteristic(net, inplace=True)
    model = NewtonPowerflow(net)._ybus_elements["trafo3w"]
    for name in ("_Y_11", "_Y_23", "_Y_32", "_DC_Y_11", "_DC_Y_13"):
        assert getattr(model, name)[1] == 0 and getattr(model, name)[0] != 0
    assert len(model._Y_11) == len(net.trafo3w)


def _net_tabular_phase_shifter() -> pandapowerNet:
    """A Tabular phase tap changer, as cim2pp imports it (cgmes_powerflow): the per-step
    ratio/angle live only in net.trafo_characteristic_table."""
    net = _net_two_trafos(
        "Tabular",
        tap_side="hv",
        tap_neutral=0,
        tap_min=-2,
        tap_max=2,
        tap_pos=2,
        tap_step_percent=0.0,
        tap_step_degree=0.0,
    )
    net.trafo["id_characteristic_table"] = pd.array([pd.NA, 0], dtype="Int64")
    net.trafo["tap_dependency_table"] = [False, True]
    steps = [-2, -1, 0, 1, 2]
    net["trafo_characteristic_table"] = pd.DataFrame(
        {
            "id_characteristic": [0] * 5,
            "step": steps,
            "voltage_ratio": [1.0] * 5,
            "angle_deg": [-6.0, -3.0, 0.0, 3.0, 6.0],
            "vk_percent": [12.0] * 5,
            "vkr_percent": [0.4] * 5,
            "vk_hv_percent": np.nan,
            "vkr_hv_percent": np.nan,
            "vk_mv_percent": np.nan,
            "vkr_mv_percent": np.nan,
            "vk_lv_percent": np.nan,
            "vkr_lv_percent": np.nan,
        }
    )
    return net


def test_existing_tabular_characteristic_is_kept():
    net = _net_tabular_phase_shifter()
    expected = _reference(net)
    assert not np.isclose(np.angle(expected[1], deg=True), 0.0, atol=0.5)  # the shifter does act
    _assert_matches_pandapower(net)


def test_existing_characteristic_rows_and_ids_are_untouched():
    net = _net_tabular_phase_shifter()
    user_rows = net.trafo_characteristic_table.set_index(["id_characteristic", "step"])
    calculate_trafo_characteristic(net, inplace=True)
    table = net.trafo_characteristic_table
    assert table.index.names == ["id_characteristic", "step"]
    assert net.trafo.id_characteristic_table[1] == 0
    pd.testing.assert_frame_equal(table.loc[[0], user_rows.columns], user_rows, check_dtype=False)


def test_calculate_trafo_characteristic_twice():
    """A second call (e.g. after changing a tap) replaces the generated rows instead of adding more."""
    net = _net_two_trafos("Ratio", tap_side="hv", tap_neutral=0, tap_min=-5, tap_max=5, tap_pos=1, tap_step_percent=1.5)
    calculate_trafo_characteristic(net, inplace=True)
    n_rows = len(net.trafo_characteristic_table)
    net.trafo.loc[1, "tap_pos"] = 4
    calculate_trafo_characteristic(net, inplace=True)
    assert len(net.trafo_characteristic_table) == n_rows
    _assert_matches_pandapower(net)


TAP = dict(tap_neutral=0, tap_min=-10, tap_max=10)


@pytest.mark.parametrize(
    "tap_changer_type, tap",
    [
        ("Ratio", dict(tap_side="hv", tap_pos=3, tap_step_percent=1.5, **TAP)),
        ("Ratio", dict(tap_side="lv", tap_pos=-2, tap_step_percent=1.5, **TAP)),
        ("Ratio", dict(tap_side="hv", tap_pos=3, tap_step_percent=1.5, tap_step_degree=30.0, **TAP)),
        ("Ratio", dict(tap_side="lv", tap_pos=3, tap_step_percent=1.5, tap_step_degree=30.0, **TAP)),
        ("Symmetrical", dict(tap_side="lv", tap_pos=2, tap_step_percent=2.0, tap_step_degree=60.0, **TAP)),
        ("Ideal", dict(tap_side="hv", tap_pos=3, tap_step_degree=2.5, **TAP)),
        ("Ideal", dict(tap_side="lv", tap_pos=3, tap_step_degree=2.5, **TAP)),
        ("Ideal", dict(tap_side="hv", tap_pos=-4, tap_step_percent=1.5, **TAP)),
    ],
    ids=[
        "ratio_hv",
        "ratio_lv",
        "ratio_hv_angle",
        "ratio_lv_angle",
        "symmetrical_lv_angle",
        "ideal_hv",
        "ideal_lv",
        "ideal_percent",
    ],
)
def test_tap_changer_types_match_pandapower(tap_changer_type, tap):
    """Every tap changer type converted to the characteristic table, on either side, as pandapower
    computes it (lv-side angles count negative)."""
    net = _net_two_trafos(tap_changer_type, **tap)
    ref = copy.deepcopy(net)
    runpp(ref)
    calculate_trafo_characteristic(net, inplace=True)
    npf = NewtonPowerflow(net)
    np.testing.assert_allclose(npf._YBus.toarray(), ref._ppc["internal"]["Ybus"].toarray(), rtol=1e-10, atol=1e-10)
    _assert_matches_pandapower(_net_two_trafos(tap_changer_type, **tap))
