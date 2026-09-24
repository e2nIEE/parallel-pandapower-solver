# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for reactive capability curves (p3s.q_capability).

Two things are pinned here:

* the curve EVALUATOR, against pandapower for ``straightLineYValues`` (parity) and
  against the documented step semantics for ``constantYValue`` (where pandapower is
  knowingly wrong -- see the module docstring);
* the sgen CLIPPING, which applies those limits to an sgen's reactive injection.
  pandapower does not do this at all in a power flow, so there is no parity to test:
  the reference is an sgen whose q_mvar was capped by hand.
"""

import logging

import numpy as np
import pandas as pd
import pytest
from pandapower.auxiliary import pandapowerNet
from pandapower.create import (
    create_bus,
    create_empty_network,
    create_ext_grid,
    create_line_from_parameters,
    create_load,
    create_sgen,
)
from pandapower.run import runpp

from p3s.NewtonPowerflow import NewtonPowerflow
from p3s.q_capability import (
    STYLE_CONSTANT,
    STYLE_LINEAR,
    QCapabilityCurves,
    resolve_q_limits,
)

# curve: (0, +-0) -> (4, +-2) -> (10, +-6)
CURVE = pd.DataFrame(
    {
        "id_q_capability_curve": [0, 0, 0],
        "p_mw": [0.0, 4.0, 10.0],
        "q_min_mvar": [0.0, -2.0, -6.0],
        "q_max_mvar": [0.0, 2.0, 6.0],
    }
)


def net_with_sgen_curve(q_request=20.0, p_mw=5.0, style=STYLE_LINEAR, with_curve=True) -> pandapowerNet:
    net = create_empty_network(sn_mva=1.0)
    net.name = "q_capability_test_network"
    b0 = create_bus(net, vn_kv=20.0)
    b1 = create_bus(net, vn_kv=20.0)
    create_ext_grid(net, b0)
    create_line_from_parameters(
        net, b0, b1, length_km=1.0, r_ohm_per_km=0.1, x_ohm_per_km=0.3, c_nf_per_km=0.0, max_i_ka=1.0
    )
    create_load(net, b1, p_mw=2.0, q_mvar=1.0)
    kwargs = (
        dict(reactive_capability_curve=True, id_q_capability_characteristic=0, curve_style=style, controllable=True)
        if with_curve
        else {}
    )
    create_sgen(net, b1, p_mw=p_mw, q_mvar=q_request, sn_mva=10.0, **kwargs)
    if with_curve:
        net["q_capability_curve_table"] = CURVE.copy()
    return net


# ---------------------------------------------------------------- evaluator ----


@pytest.mark.parametrize(
    "p, q_max",
    [
        (0.0, 0.0),
        (2.0, 1.0),
        (4.0, 2.0),
        (7.0, 4.0),
        (10.0, 6.0),
    ],
)
def test_linear_matches_pandapower(p, q_max):
    """straightLineYValues must reproduce pandapower's Characteristic exactly."""
    from pandapower.control.util.auxiliary import create_q_capability_characteristics_object

    net = net_with_sgen_curve(p_mw=p, style=STYLE_LINEAR)

    g_min, g_max = resolve_q_limits(net, "sgen")

    create_q_capability_characteristics_object(net)
    pp_max = net.q_capability_characteristic.loc[0, "q_max_characteristic"](p)
    pp_min = net.q_capability_characteristic.loc[0, "q_min_characteristic"](p)

    assert np.isclose(g_max[0], pp_max)
    assert np.isclose(g_min[0], pp_min)
    assert np.isclose(g_max[0], q_max)


@pytest.mark.parametrize(
    "p, q_max",
    [
        (0.0, 0.0),  # exactly on the first point
        (2.0, 0.0),  # holds the p=0 value (pandapower would give 1.0)
        (4.0, 2.0),  # exactly on a breakpoint
        (7.0, 2.0),  # holds the p=4 value (pandapower would give 4.0)
        (10.0, 6.0),  # exactly on the last point
    ],
)
def test_constant_style_is_a_step(p, q_max):
    """constantYValue is a zero-order hold -- DELIBERATELY unlike pandapower.

    pandapower's Characteristic.__call__ is unconditionally np.interp, so it linearly
    interpolates both styles and curve_style never reaches the interpolation. p3s
    implements the documented step semantics, so these values differ on purpose.
    """
    net = net_with_sgen_curve(p_mw=p, style=STYLE_CONSTANT)
    _, g_max = resolve_q_limits(net, "sgen")
    assert np.isclose(g_max[0], q_max)


def test_constant_and_linear_actually_differ():
    """Guard against curve_style silently becoming a no-op here too."""
    lin = resolve_q_limits(net_with_sgen_curve(p_mw=7.0, style=STYLE_LINEAR), "sgen")[1]
    con = resolve_q_limits(net_with_sgen_curve(p_mw=7.0, style=STYLE_CONSTANT), "sgen")[1]
    assert not np.isclose(lin[0], con[0])


@pytest.mark.parametrize("p, expected", [(-3.0, 0.0), (99.0, 6.0)])
def test_out_of_range_clamps_and_warns(p, expected, caplog):
    """P outside the curve range clamps to the endpoint limits, with a warning."""
    net = net_with_sgen_curve(p_mw=p)
    with caplog.at_level(logging.WARNING, logger="p3s.q_capability"):
        _, q_max = resolve_q_limits(net, "sgen")
    assert np.isclose(q_max[0], expected)
    assert any("outside the capability curve range" in r.message for r in caplog.records)


def test_unknown_curve_id_falls_back_to_fixed_limits(caplog):
    """A dangling id_q_capability_characteristic must not crash the solve."""
    net = net_with_sgen_curve()
    net.sgen.at[0, "id_q_capability_characteristic"] = 99
    net.sgen["max_q_mvar"] = 3.5
    net.sgen["min_q_mvar"] = -3.5
    with caplog.at_level(logging.WARNING, logger="p3s.q_capability"):
        q_min, q_max = resolve_q_limits(net, "sgen")
    assert np.isclose(q_max[0], 3.5) and np.isclose(q_min[0], -3.5)
    assert any("no matching curve" in r.message for r in caplog.records)


def test_no_curve_table_uses_fixed_limits():
    """Nets without a curve table fall back to min/max_q_mvar."""
    net = net_with_sgen_curve(with_curve=False)
    net.sgen["max_q_mvar"] = 4.0
    net.sgen["min_q_mvar"] = -1.0
    q_min, q_max = resolve_q_limits(net, "sgen")
    assert np.isclose(q_max[0], 4.0) and np.isclose(q_min[0], -1.0)


def test_curve_points_may_be_unsorted():
    """Row order in q_capability_curve_table is not guaranteed; the evaluator sorts."""
    net = net_with_sgen_curve(p_mw=7.0)
    net["q_capability_curve_table"] = CURVE.iloc[::-1].reset_index(drop=True)
    _, q_max = resolve_q_limits(net, "sgen")
    assert np.isclose(q_max[0], 4.0)


def test_invalid_style_raises():
    net = net_with_sgen_curve()
    net.sgen.at[0, "curve_style"] = "cubicSpline"
    with pytest.raises(ValueError, match="unsupported curve_style"):
        resolve_q_limits(net, "sgen")


@pytest.mark.parametrize(
    "bad, match",
    [
        (
            pd.DataFrame({"id_q_capability_curve": [0], "p_mw": [1.0], "q_min_mvar": [0.0], "q_max_mvar": [1.0]}),
            "at least 2 points",
        ),
        (
            pd.DataFrame(
                {
                    "id_q_capability_curve": [0, 0],
                    "p_mw": [2.0, 2.0],
                    "q_min_mvar": [0.0, 0.0],
                    "q_max_mvar": [1.0, 1.0],
                }
            ),
            "strictly increasing",
        ),
    ],
)
def test_malformed_curve_raises(bad, match):
    net = net_with_sgen_curve()
    net["q_capability_curve_table"] = bad
    with pytest.raises(ValueError, match=match):
        QCapabilityCurves.from_net(net)


def test_missing_column_raises():
    net = net_with_sgen_curve()
    net["q_capability_curve_table"] = CURVE.drop(columns=["q_max_mvar"])
    with pytest.raises(ValueError, match="missing column"):
        QCapabilityCurves.from_net(net)


# ----------------------------------------------------------- sgen clipping ----


def test_clipping_is_off_by_default():
    """enforce_q_lims defaults to False, matching p3s's previous behaviour."""
    net = net_with_sgen_curve(q_request=20.0)
    solver = NewtonPowerflow(net)
    assert solver.enforce_q_lims is False
    # the full unclipped 20 MVAr reaches Sbus (minus the 1 MVAr load)
    assert np.isclose(solver._sBus[1].imag, 19.0)


def test_clipping_applies_curve_limit():
    """With enforcement on, the injected Q is capped at the curve value."""
    net = net_with_sgen_curve(q_request=20.0, p_mw=5.0)
    solver = NewtonPowerflow(net, enforce_q_lims=True)
    # curve q_max at p=5 is 2 + (5-4)/(10-4)*(6-2) = 2.6667; load draws 1 MVAr
    assert np.isclose(solver._sBus[1].imag, 2.0 + 4.0 / 6.0 - 1.0)


def test_clipping_applies_lower_limit():
    net = net_with_sgen_curve(q_request=-20.0, p_mw=5.0)
    solver = NewtonPowerflow(net, enforce_q_lims=True)
    assert np.isclose(solver._sBus[1].imag, -(2.0 + 4.0 / 6.0) - 1.0)


def test_clipping_changes_the_solution():
    """A violated limit must actually move the voltages, not just the input."""
    free = net_with_sgen_curve(q_request=20.0)
    NewtonPowerflow(free, enforce_q_lims=False).calculate(free)
    held = net_with_sgen_curve(q_request=20.0)
    NewtonPowerflow(held, enforce_q_lims=True).calculate(held)
    assert not np.allclose(free.res_bus.vm_pu.values, held.res_bus.vm_pu.values)


def test_clipped_equals_explicitly_capped_sgen():
    """The clipped solve must equal one where q_mvar was capped by hand."""
    held = net_with_sgen_curve(q_request=20.0)
    NewtonPowerflow(held, enforce_q_lims=True).calculate(held)
    manual = net_with_sgen_curve(q_request=2.0 + 4.0 / 6.0)
    NewtonPowerflow(manual, enforce_q_lims=False).calculate(manual)
    assert np.allclose(held.res_bus.vm_pu.values, manual.res_bus.vm_pu.values, atol=1e-12)


def test_request_within_limits_is_untouched():
    """Clipping must be a no-op when the request respects the curve."""
    a = net_with_sgen_curve(q_request=1.0)
    NewtonPowerflow(a, enforce_q_lims=True).calculate(a)
    b = net_with_sgen_curve(q_request=1.0)
    NewtonPowerflow(b, enforce_q_lims=False).calculate(b)
    assert np.allclose(a.res_bus.vm_pu.values, b.res_bus.vm_pu.values, atol=1e-12)


def test_unenforced_still_matches_pandapower():
    """With enforcement off p3s must stay bit-compatible with pandapower.

    pandapower never applies the curve to an sgen during a power flow (sgen only enters
    the ppc gen matrix under mode == "opf"), so its result is the unclipped one.
    """
    ref = net_with_sgen_curve(q_request=20.0)
    runpp(ref, tolerance_mva=1e-12)
    net = net_with_sgen_curve(q_request=20.0)
    NewtonPowerflow(net, enforce_q_lims=False).calculate(net, tolerance=1e-12)
    assert np.allclose(net.res_bus.vm_pu.values, ref.res_bus.vm_pu.values, rtol=1e-10, atol=1e-12)


def test_scaling_is_applied_after_clipping():
    """The curve limits Q at the machine, so scaling multiplies the CLIPPED value."""
    net = net_with_sgen_curve(q_request=20.0)
    net.sgen.at[0, "scaling"] = 0.5
    solver = NewtonPowerflow(net, enforce_q_lims=True)
    assert np.isclose(solver._sBus[1].imag, 0.5 * (2.0 + 4.0 / 6.0) - 1.0)


def test_sgen_without_curve_uses_fixed_limits():
    """enforce_q_lims also applies plain min/max_q_mvar, which pandapower likewise skips."""
    net = net_with_sgen_curve(q_request=20.0, with_curve=False)
    net.sgen["max_q_mvar"] = 3.0
    net.sgen["min_q_mvar"] = -3.0
    solver = NewtonPowerflow(net, enforce_q_lims=True)
    assert np.isclose(solver._sBus[1].imag, 3.0 - 1.0)


def test_nan_limit_means_unlimited():
    """A NaN on one side must not clip that side."""
    net = net_with_sgen_curve(q_request=20.0, with_curve=False)
    net.sgen["max_q_mvar"] = np.nan
    net.sgen["min_q_mvar"] = -3.0
    solver = NewtonPowerflow(net, enforce_q_lims=True)
    assert np.isclose(solver._sBus[1].imag, 19.0)
