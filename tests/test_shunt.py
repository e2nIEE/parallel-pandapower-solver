# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the pandapower ``shunt`` element (p3s.models.ShuntModel).

A shunt is a constant-impedance load hung off a single bus: it lands on the Ybus
diagonal only. Its rating ``vn_kv`` may differ from the connected bus voltage, in
which case pandapower scales the per-unit admittance by ``(vn_kv_bus / vn_kv_shunt)**2``
(``build_bus._calc_shunts_and_add_on_ppc``). These tests pin that voltage-ratio
correction, the NaN ``vn_kv`` fallback, the diagonal-only stamp, the out-of-service
mask, and the full power flow against pandapower.
"""

import numpy as np
from pandapower.auxiliary import pandapowerNet
from pandapower.create import (
    create_bus,
    create_empty_network,
    create_ext_grid,
    create_line_from_parameters,
    create_load,
    create_shunt,
)
from pandapower.run import runpp

from p3s.models.ShuntModel import ShuntModel
from p3s.NewtonPowerflow import NewtonPowerflow

# pandapower and p3s must be compared at the SAME convergence tolerance: at the
# default 1e-5 the two solvers stop at slightly different points and agree only to ~1e-9,
# which would hide a real modelling error.
TOL = 1e-12

BUS_VN_KV = 20.0


def net_with_shunts() -> pandapowerNet:
    net = create_empty_network(sn_mva=1.0)
    net.name = "shunt_test_network"
    b = [create_bus(net, vn_kv=BUS_VN_KV) for _ in range(3)]
    create_ext_grid(net, b[0])
    for i in range(2):
        create_line_from_parameters(
            net, b[i], b[i + 1], length_km=6.0, r_ohm_per_km=0.2, x_ohm_per_km=0.35, c_nf_per_km=10.0, max_i_ka=0.4
        )
    create_load(net, b[2], p_mw=3.0, q_mvar=1.0)
    # rated voltage differs from the bus voltage -> v_ratio = (20/21)**2
    create_shunt(net, b[1], q_mvar=2.0, p_mw=1.0, vn_kv=21.0)
    # vn_kv omitted -> pandapower fills it with the bus vn_kv -> v_ratio = 1
    create_shunt(net, b[2], q_mvar=-1.5, p_mw=0.5)
    # explicitly NaN -> replaced by the bus vn_kv during the power flow -> v_ratio = 1
    create_shunt(net, b[2], q_mvar=0.7, p_mw=0.3, vn_kv=np.nan)
    return net


def test_shunt_stamp_matches_pandapower_ybus():
    """The shunt stamp must land on the Ybus diagonal with pandapower's v_ratio."""
    net = net_with_shunts()
    runpp(net)

    n_bus = len(net.bus)
    y_shunt = ShuntModel(net.shunt, net.bus, sn_mva=net.sn_mva).create_y_matrix(n_bus=n_bus).toarray()

    # isolate the shunt contribution: pandapower's Ybus minus the line stamp
    from p3s.models.TransmissionLineModel import TransmissionLineModel

    voltages = net.bus.vn_kv[net.line.from_bus].values
    y_line = (
        TransmissionLineModel(net.line, voltages, f_hz=net.f_hz, sn_mva=net.sn_mva)
        .create_y_matrix(n_bus=n_bus)
        .toarray()
    )
    ybus_pp = net._ppc["internal"]["Ybus"].toarray()

    assert np.allclose(y_shunt + y_line, ybus_pp, rtol=1e-10, atol=1e-12)


def test_voltage_ratio_scales_the_stamp():
    """y = (p - jq) * step * (vn_kv_bus / vn_kv_shunt)**2 / sn_mva, with NaN -> bus voltage."""
    net = net_with_shunts()
    model = ShuntModel(net.shunt, net.bus, sn_mva=net.sn_mva)
    y = np.asarray(model._Y_ff)

    # shunt 0: p=1.0, q=2.0, rated 21 kV on a 20 kV bus
    assert np.isclose(y[0], (1.0 - 2.0j) * (BUS_VN_KV / 21.0) ** 2)
    # shunt 1: rated voltage defaults to the bus voltage -> v_ratio = 1
    assert np.isclose(y[1], 0.5 + 1.5j)
    # shunt 2: NaN rated voltage falls back to the bus voltage -> v_ratio = 1
    assert np.isclose(y[2], 0.3 - 0.7j)

    assert np.allclose(model.shunt_vn_kv, [21.0, BUS_VN_KV, BUS_VN_KV])
    # the correction actually matters: the naive stamp would be off for shunt 0
    assert not np.isclose(y[0], 1.0 - 2.0j)


def test_nan_vn_kv_falls_back_to_bus_voltage():
    """A NaN shunt vn_kv must be replaced by the connected bus vn_kv (v_ratio == 1)."""
    net = net_with_shunts()
    net.shunt["vn_kv"] = np.nan
    model = ShuntModel(net.shunt, net.bus, sn_mva=net.sn_mva)
    assert np.allclose(model.shunt_vn_kv, BUS_VN_KV)
    y = np.asarray(model._Y_ff)
    assert np.isclose(y[0], 1.0 - 2.0j)
    assert np.isclose(y[1], 0.5 + 1.5j)
    assert np.isclose(y[2], 0.3 - 0.7j)


def test_shunt_is_diagonal_only():
    """A shunt hangs off one bus: it must not create any off-diagonal coupling."""
    net = net_with_shunts()
    model = ShuntModel(net.shunt, net.bus, sn_mva=net.sn_mva)
    assert np.allclose(np.asarray(model._Y_ft), 0.0)
    assert np.allclose(np.asarray(model._Y_tf), 0.0)
    assert np.allclose(np.asarray(model._Y_tt), 0.0)
    y = model.create_y_matrix(n_bus=len(net.bus)).toarray()
    assert np.allclose(y - np.diag(np.diag(y)), 0.0)


def test_out_of_service_shunt_is_absent():
    """An out-of-service shunt must contribute nothing while keeping row alignment."""
    net = net_with_shunts()
    net.shunt.at[0, "in_service"] = False

    model = ShuntModel(net.shunt, net.bus, sn_mva=net.sn_mva)
    assert np.asarray(model._Y_ff)[0] == 0  # dropped
    assert np.asarray(model._Y_ff)[1] != 0  # others untouched
    assert len(np.asarray(model._Y_ff)) == len(net.shunt)  # rows stay aligned

    ref = net_with_shunts()
    ref.shunt.at[0, "in_service"] = False
    runpp(ref, tolerance_mva=TOL)
    solved = net_with_shunts()
    solved.shunt.at[0, "in_service"] = False
    NewtonPowerflow(solved).calculate(solved, tolerance=TOL)
    assert np.allclose(solved.res_bus.vm_pu.values, ref.res_bus.vm_pu.values, rtol=1e-10, atol=1e-12)


def test_shunt_contributes_nothing_to_dc_bmatrix():
    """A bus shunt is not part of the DC B-matrix (makeBdc builds it from 1/x only)."""
    net = net_with_shunts()
    model = ShuntModel(net.shunt, net.bus, sn_mva=net.sn_mva)
    b_dc = model.create_y_dc_matrix(n_bus=len(net.bus)).toarray()
    assert np.allclose(b_dc, 0.0)


def test_powerflow_matches_pandapower():
    """A full solve must reproduce pandapower's voltages with the voltage ratio applied."""
    ref = net_with_shunts()
    runpp(ref, tolerance_mva=TOL)

    net = net_with_shunts()
    NewtonPowerflow(net).calculate(net, tolerance=TOL)

    assert np.allclose(net.res_bus.vm_pu.values, ref.res_bus.vm_pu.values, rtol=1e-10, atol=1e-12)
    assert np.allclose(net.res_bus.va_degree.values, ref.res_bus.va_degree.values, rtol=1e-10, atol=1e-12)
