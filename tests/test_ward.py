# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the pandapower ``ward`` element (p3s.models.WardModel).

A ward is the first ACTIVE element in p3s/models: it contributes to BOTH the Ybus
diagonal (its constant-impedance half, pz/qz) and the bus injection vector Sbus (its
constant-power half, ps/qs). These tests pin each half separately as well as the
combined power flow and res_ward output against pandapower.
"""

import numpy as np
from pandapower.auxiliary import pandapowerNet
from pandapower.create import (
    create_bus,
    create_empty_network,
    create_ext_grid,
    create_line_from_parameters,
    create_load,
    create_ward,
)
from pandapower.run import runpp

from p3s.models.WardModel import WardModel
from p3s.NewtonPowerflow import NewtonPowerflow

# pandapower and p3s must be compared at the SAME convergence tolerance: at the
# default 1e-5 the two solvers stop at slightly different points and agree only to ~1e-9,
# which would hide a real modelling error.
TOL = 1e-12


def net_with_wards() -> pandapowerNet:
    net = create_empty_network(sn_mva=1.0)
    net.name = "ward_test_network"
    b = [create_bus(net, vn_kv=110.0) for _ in range(4)]
    create_ext_grid(net, b[0])
    for i in range(3):
        create_line_from_parameters(
            net, b[i], b[i + 1], length_km=8.0, r_ohm_per_km=0.12, x_ohm_per_km=0.38, c_nf_per_km=9.5, max_i_ka=0.5
        )
    create_load(net, b[3], p_mw=5.0, q_mvar=2.0)
    # both halves active
    create_ward(net, b[1], ps_mw=3.0, qs_mvar=1.5, pz_mw=2.0, qz_mvar=1.0)
    # pure constant power (no shunt contribution)
    create_ward(net, b[2], ps_mw=4.0, qs_mvar=-1.0, pz_mw=0.0, qz_mvar=0.0)
    # pure constant impedance (no Sbus contribution)
    create_ward(net, b[3], ps_mw=0.0, qs_mvar=0.0, pz_mw=1.5, qz_mvar=0.8)
    return net


def test_shunt_half_matches_pandapower_ybus():
    """pz/qz must land on the Ybus diagonal with pandapower's sign convention."""
    net = net_with_wards()
    runpp(net)

    n_bus = len(net.bus)
    model = WardModel(net.ward, n_bus=n_bus, sn_mva=net.sn_mva)
    y_ward = model.create_y_matrix(n_bus=n_bus).toarray()

    # isolate the ward contribution: pandapower's Ybus minus the line stamp
    from p3s.models.TransmissionLineModel import TransmissionLineModel

    voltages = net.bus.vn_kv[net.line.from_bus].values
    y_line = (
        TransmissionLineModel(net.line, voltages, f_hz=net.f_hz, sn_mva=net.sn_mva)
        .create_y_matrix(n_bus=n_bus)
        .toarray()
    )
    ybus_pp = net._ppc["internal"]["Ybus"].toarray()

    assert np.allclose(y_ward + y_line, ybus_pp, rtol=1e-10, atol=1e-12)


def test_shunt_is_diagonal_only():
    """A ward hangs off one bus: it must not create any off-diagonal coupling."""
    net = net_with_wards()
    model = WardModel(net.ward, n_bus=len(net.bus), sn_mva=net.sn_mva)
    assert np.allclose(np.asarray(model._Y_ft), 0.0)
    assert np.allclose(np.asarray(model._Y_tf), 0.0)
    assert np.allclose(np.asarray(model._Y_tt), 0.0)
    y = model.create_y_matrix(n_bus=len(net.bus)).toarray()
    assert np.allclose(y - np.diag(np.diag(y)), 0.0)


def test_reactive_sign_convention():
    """qz_mvar enters Ybus NEGATED (pandapower stores BS = -qz_mvar)."""
    net = net_with_wards()
    model = WardModel(net.ward, n_bus=len(net.bus), sn_mva=net.sn_mva)
    # first ward: pz = 2.0, qz = 1.0, sn_mva = 1.0
    assert np.isclose(np.asarray(model._Y_ff)[0], 2.0 - 1.0j)


def test_constant_power_half_goes_to_s_bus():
    """ps/qs must appear in s_bus at the ward bus, in MVA and unscaled."""
    net = net_with_wards()
    model = WardModel(net.ward, n_bus=len(net.bus), sn_mva=net.sn_mva)
    buses = net.ward.bus.values
    assert np.isclose(model.s_bus[buses[0]], 3.0 + 1.5j)
    assert np.isclose(model.s_bus[buses[1]], 4.0 - 1.0j)
    # pure-impedance ward contributes nothing to Sbus
    assert np.isclose(model.s_bus[buses[2]], 0.0)


def test_several_wards_on_one_bus_accumulate():
    """s_bus is built with np.add.at, so wards sharing a bus must sum."""
    net = net_with_wards()
    create_ward(net, net.ward.bus.values[0], ps_mw=1.0, qs_mvar=0.5, pz_mw=0.0, qz_mvar=0.0)
    model = WardModel(net.ward, n_bus=len(net.bus), sn_mva=net.sn_mva)
    assert np.isclose(model.s_bus[net.ward.bus.values[0]], 4.0 + 2.0j)


def test_powerflow_and_res_ward_match_pandapower():
    """A full solve must reproduce pandapower's voltages and every res_ward column."""
    ref = net_with_wards()
    runpp(ref, tolerance_mva=TOL)

    net = net_with_wards()
    NewtonPowerflow(net).calculate(net, tolerance=TOL)

    assert np.allclose(net.res_bus.vm_pu.values, ref.res_bus.vm_pu.values, rtol=1e-10, atol=1e-12)
    assert np.allclose(net.res_bus.va_degree.values, ref.res_bus.va_degree.values, rtol=1e-10, atol=1e-12)
    for column in ref.res_ward.columns:
        assert np.allclose(net.res_ward[column].values, ref.res_ward[column].values, rtol=1e-10, atol=1e-12), (
            f"res_ward.{column} differs"
        )


def test_res_ward_recombines_both_halves():
    """res_ward reports ps + vm**2 * pz (and the same for q, with qz NOT negated)."""
    net = net_with_wards()
    NewtonPowerflow(net).calculate(net, tolerance=TOL)
    vm = net.res_ward.vm_pu.values
    expected_p = net.ward.ps_mw.values + vm**2 * net.ward.pz_mw.values
    expected_q = net.ward.qs_mvar.values + vm**2 * net.ward.qz_mvar.values
    assert np.allclose(net.res_ward.p_mw.values, expected_p, rtol=1e-10, atol=1e-12)
    assert np.allclose(net.res_ward.q_mvar.values, expected_q, rtol=1e-10, atol=1e-12)


def test_out_of_service_ward_is_absent_in_both_halves():
    """An out-of-service ward must drop out of Ybus AND out of Sbus."""
    net = net_with_wards()
    net.ward.at[0, "in_service"] = False

    model = WardModel(net.ward, n_bus=len(net.bus), sn_mva=net.sn_mva)
    assert np.asarray(model._Y_ff)[0] == 0  # shunt half gone
    assert model.s_bus[net.ward.bus.values[0]] == 0  # power half gone
    assert np.asarray(model._Y_ff)[2] != 0  # others untouched
    assert len(np.asarray(model._Y_ff)) == len(net.ward)

    ref = net_with_wards()
    ref.ward.at[0, "in_service"] = False
    runpp(ref, tolerance_mva=TOL)
    solved = net_with_wards()
    solved.ward.at[0, "in_service"] = False
    NewtonPowerflow(solved).calculate(solved, tolerance=TOL)
    assert np.allclose(solved.res_bus.vm_pu.values, ref.res_bus.vm_pu.values, rtol=1e-10, atol=1e-12)


def test_ward_contributes_nothing_to_dc_bmatrix():
    """A bus shunt is not part of the DC B-matrix (makeBdc builds it from 1/x only)."""
    net = net_with_wards()
    model = WardModel(net.ward, n_bus=len(net.bus), sn_mva=net.sn_mva)
    b_dc = model.create_y_dc_matrix(n_bus=len(net.bus)).toarray()
    assert np.allclose(b_dc, 0.0)


def test_ward_bus_stays_pq():
    """A ward must not change its bus type -- it is a demand, not a generator."""
    net = net_with_wards()
    solver = NewtonPowerflow(net)
    ward_buses = set(net.ward.bus.values)
    assert ward_buses <= set(solver.busses["pq"].tolist())
    assert not (ward_buses & set(solver.busses["pv"].tolist()))
    assert not (ward_buses & set(solver.busses["ref"].tolist()))
