# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the pandapower ``impedance`` element (graviton.models.ImpedanceModel).

The impedance is the only branch element with an ASYMMETRIC series admittance
(z_ft != z_tf) and per-unit values referred to its own sn_mva, so these tests pin
the Ybus stamp, the DC B-matrix and the res_impedance flows against pandapower.
"""

import numpy as np
import pytest
from graviton.models.ImpedanceModel import ImpedanceModel
from graviton.NewtonPowerflow import NewtonPowerflow
from pandapower.auxiliary import pandapowerNet
from pandapower.create import (
    create_bus,
    create_empty_network,
    create_ext_grid,
    create_impedance,
    create_line_from_parameters,
    create_load,
)
from pandapower.run import rundcpp, runpp

# asymmetric impedance with terminal shunts, as used for network equivalents
ASYM = dict(
    rft_pu=0.02,
    xft_pu=0.08,
    rtf_pu=0.03,
    xtf_pu=0.11,
    gf_pu=0.004,
    bf_pu=-0.012,
    gt_pu=0.002,
    bt_pu=-0.007,
    sn_mva=25.0,
)
# symmetric impedance without shunts -- must reduce to the ordinary branch stamp
SYM = dict(rft_pu=0.05, xft_pu=0.15, rtf_pu=0.05, xtf_pu=0.15, sn_mva=10.0)


def net_with_impedances() -> pandapowerNet:
    net = create_empty_network(sn_mva=1.0)
    net.name = "impedance_test_network"
    b = [create_bus(net, vn_kv=110.0) for _ in range(4)]
    create_ext_grid(net, b[0])
    create_line_from_parameters(
        net, b[0], b[1], length_km=10.0, r_ohm_per_km=0.12, x_ohm_per_km=0.38, c_nf_per_km=9.5, max_i_ka=0.5
    )
    create_impedance(net, b[1], b[2], **ASYM)
    create_impedance(net, b[2], b[3], **SYM)
    create_load(net, b[2], p_mw=4.0, q_mvar=1.5)
    create_load(net, b[3], p_mw=8.0, q_mvar=3.0)
    return net


def test_ybus_matches_pandapower():
    """The AC stamp must reproduce pandapower's Ybus for asymmetric + symmetric impedances."""
    net = net_with_impedances()
    runpp(net)

    # isolate the impedance contribution: build the line stamp separately and add it
    from graviton.models.TransmissionLineModel import TransmissionLineModel

    n_bus = len(net.bus)
    voltages = net.bus.vn_kv[net.line.from_bus].values
    y_line = TransmissionLineModel(net.line, voltages, f_hz=net.f_hz, sn_mva=net.sn_mva).create_y_matrix(n_bus=n_bus)
    y_imp = ImpedanceModel(net.impedance, net.bus, sn_mva=net.sn_mva).create_y_matrix(n_bus=n_bus)

    ybus_pp = net._ppc["internal"]["Ybus"].toarray()
    assert np.allclose((y_line + y_imp).toarray(), ybus_pp, rtol=1e-10, atol=1e-12)


def test_dc_bbus_matches_pandapower():
    """The DC B-matrix uses the from->to reactance only (makeBdc ignores BR_X_ASYM)."""
    net = net_with_impedances()
    rundcpp(net)

    from graviton.models.TransmissionLineModel import TransmissionLineModel

    n_bus = len(net.bus)
    voltages = net.bus.vn_kv[net.line.from_bus].values
    b_line = TransmissionLineModel(net.line, voltages, f_hz=net.f_hz, sn_mva=net.sn_mva).create_y_dc_matrix(n_bus=n_bus)
    b_imp = ImpedanceModel(net.impedance, net.bus, sn_mva=net.sn_mva).create_y_dc_matrix(n_bus=n_bus)

    bbus_pp = net._ppc["internal"]["Bbus"].toarray().real
    assert np.allclose((b_line + b_imp).toarray().imag, bbus_pp, rtol=1e-10, atol=1e-12)


def test_powerflow_and_res_impedance_match_pandapower():
    """A full solve must reproduce pandapower's bus voltages and every res_impedance column."""
    ref = net_with_impedances()
    runpp(ref)

    net = net_with_impedances()
    NewtonPowerflow(net).calculate(net)

    assert np.allclose(net.res_bus.vm_pu.values, ref.res_bus.vm_pu.values, rtol=1e-8, atol=1e-10)
    assert np.allclose(net.res_bus.va_degree.values, ref.res_bus.va_degree.values, rtol=1e-8, atol=1e-10)

    for column in ref.res_impedance.columns:
        assert np.allclose(net.res_impedance[column].values, ref.res_impedance[column].values, rtol=1e-8, atol=1e-10), (
            f"res_impedance.{column} differs"
        )


def test_asymmetric_stamp_is_actually_asymmetric():
    """Guard against silently collapsing z_tf onto z_ft: Y_ft and Y_tf must differ."""
    net = net_with_impedances()
    model = ImpedanceModel(net.impedance, net.bus, sn_mva=net.sn_mva)
    # row 0 is the asymmetric element, row 1 the symmetric one
    assert not np.isclose(model._Y_ft[0], model._Y_tf[0])
    assert np.isclose(model._Y_ft[1], model._Y_tf[1])


def test_out_of_service_impedance_is_absent():
    """An out-of-service impedance must contribute nothing to Ybus but keep its row."""
    net = net_with_impedances()
    net.impedance.at[1, "in_service"] = False

    model = ImpedanceModel(net.impedance, net.bus, sn_mva=net.sn_mva)
    assert model._Y_ff[1] == 0 and model._Y_ft[1] == 0
    assert model._Y_tf[1] == 0 and model._Y_tt[1] == 0
    # the in-service element is untouched, and rows stay aligned with net.impedance
    assert model._Y_ff[0] != 0
    assert len(model._Y_ff) == len(net.impedance)


def test_out_of_service_impedance_solves():
    """De-energising an impedance must match pandapower on a net that stays connected.

    A parallel line keeps the far bus attached -- switching off the only path to a bus
    would island it and make the Jacobian singular, which is correct physics but tests
    nothing about the stamp.
    """

    def build() -> pandapowerNet:
        net = net_with_impedances()
        # parallel path around the second (switchable) impedance
        create_line_from_parameters(
            net, 2, 3, length_km=4.0, r_ohm_per_km=0.12, x_ohm_per_km=0.38, c_nf_per_km=9.5, max_i_ka=0.5
        )
        net.impedance.at[1, "in_service"] = False
        return net

    ref = build()
    runpp(ref)
    net = build()
    NewtonPowerflow(net).calculate(net)
    assert np.allclose(net.res_bus.vm_pu.values, ref.res_bus.vm_pu.values, rtol=1e-8, atol=1e-10)
    # the de-energised element reports zero flow rather than shifting other rows
    assert np.allclose(net.res_impedance.loc[1, ["p_from_mw", "q_from_mvar"]].values, 0.0)


@pytest.mark.parametrize(
    "dropped",
    [
        ["gf_pu", "bf_pu", "gt_pu", "bt_pu"],
        ["bf_pu"],
    ],
)
def test_missing_shunt_columns_default_to_zero(dropped):
    """Nets from older pandapower versions may lack the terminal shunt columns."""
    net = net_with_impedances()
    table = net.impedance.drop(columns=dropped)
    model = ImpedanceModel(table, net.bus, sn_mva=net.sn_mva)
    assert np.isfinite(np.asarray(model._Y_ff)).all()
    assert np.isfinite(np.asarray(model._Y_tt)).all()


def test_nan_shunt_values_treated_as_zero():
    """NaN in a shunt column must not poison Ybus."""
    net = net_with_impedances()
    table = net.impedance.copy()
    table["bf_pu"] = np.nan
    model = ImpedanceModel(table, net.bus, sn_mva=net.sn_mva)
    assert np.isfinite(np.asarray(model._Y_ff)).all()


def test_sn_mva_rebasing():
    """Doubling the element sn_mva halves its per-unit impedance on the network base."""
    net = net_with_impedances()
    base = ImpedanceModel(net.impedance, net.bus, sn_mva=net.sn_mva)

    table = net.impedance.copy()
    table["sn_mva"] = table["sn_mva"] * 2.0
    doubled = ImpedanceModel(table, net.bus, sn_mva=net.sn_mva)

    # y = 1/z and z scales with sn_net/sn_mva, so halving z doubles the series admittance
    assert np.allclose(doubled._Y_ft, 2.0 * np.asarray(base._Y_ft), rtol=1e-10)
