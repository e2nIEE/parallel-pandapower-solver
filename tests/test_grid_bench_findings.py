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
from pandapower.run import runpp

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


def _net_trafo3w(pfe_kw: float = 25.0, i0_percent: float = 0.06) -> pandapowerNet:
    net = create_empty_network(sn_mva=1.0)
    hv = create_bus(net, vn_kv=110.0)
    mv = create_bus(net, vn_kv=20.0)
    lv = create_bus(net, vn_kv=10.0)
    create_ext_grid(net, hv)
    create_transformer3w_from_parameters(
        net,
        hv,
        mv,
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
        pfe_kw=pfe_kw,
        i0_percent=i0_percent,
    )
    create_load(net, mv, p_mw=15.0, q_mvar=5.0)
    create_load(net, lv, p_mw=8.0, q_mvar=2.0)
    return net


TRAFO3W_LOSSES = pytest.mark.parametrize("pfe_kw, i0_percent", [(25.0, 0.06), (0.0, 0.0)], ids=["mag", "no_mag"])


def test_trafo3w_without_characteristic_table_builds():
    """trafo3w.id_characteristic_table is NA unless a table exists (cgmes_minigrid)."""
    net = _net_trafo3w()
    calculate_trafo_characteristic(net, inplace=True)
    NewtonPowerflow(net)  # ValueError: cannot convert NA to integer


@TRAFO3W_LOSSES
def test_trafo3w_ybus_matches_pandapower(pfe_kw, i0_percent):
    """Not visible on grid-bench (the NA crash comes first): with a characteristic row
    supplied by hand, p3s's trafo3w stamp is off by factors 0.5-2.5 against pandapower,
    and all-NaN without magnetising branch (pfe_kw = i0_percent = 0).

    pandapower models the trafo3w with an auxiliary star bus; Kron-reduce it away to
    compare against p3s's bus-only Ybus."""
    net = _net_trafo3w(pfe_kw, i0_percent)
    ref = copy.deepcopy(net)
    runpp(ref)
    y = ref._ppc["internal"]["Ybus"].toarray()
    keep = ref._pd2ppc_lookups["bus"][net.bus.index.values]
    star = np.setdiff1d(np.arange(y.shape[0]), keep)
    y_red = y[np.ix_(keep, keep)] - y[np.ix_(keep, star)] @ np.linalg.solve(
        y[np.ix_(star, star)], y[np.ix_(star, keep)]
    )

    calculate_trafo_characteristic(net, inplace=True)
    npf = NewtonPowerflow(net)
    np.testing.assert_allclose(npf._YBus.toarray(), y_red, atol=1e-8)


@TRAFO3W_LOSSES
def test_trafo3w_powerflow(pfe_kw, i0_percent):
    _assert_matches_pandapower(_net_trafo3w(pfe_kw, i0_percent))


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
