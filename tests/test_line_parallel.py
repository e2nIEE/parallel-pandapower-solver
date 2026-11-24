# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lines with ``parallel > 1`` (p3s.models.TransmissionLineModel).

``parallel = k`` means k identical systems side by side: every admittance of the
pi model, series and shunt, is k times that of one system (pandapower:
``build_branch._calc_branch_values_from_line``). p3s 1.2.2 divided the series
admittance by k instead, so a double line had a quarter of its true series
admittance (case118 bus 9 was off by 0.032 pu). These tests pin the full power
flow against pandapower and the equivalence with k single lines.
"""

import copy

import numpy as np
import pytest
from pandapower.create import (
    create_bus,
    create_empty_network,
    create_ext_grid,
    create_gen,
    create_line_from_parameters,
    create_load,
)
from pandapower.run import runpp

from p3s.NewtonPowerflow import NewtonPowerflow

# same tolerance for pandapower and p3s, see test_shunt.py
TOL = 1e-12
LINE = dict(length_km=40.0, r_ohm_per_km=0.05, x_ohm_per_km=0.4, c_nf_per_km=12.0, max_i_ka=1.0)


def net_with_parallel_line(parallel, g_us_per_km=0.0):
    net = create_empty_network(sn_mva=100.0)
    b = [create_bus(net, vn_kv=220.0) for _ in range(3)]
    create_ext_grid(net, b[0], vm_pu=1.02)
    create_line_from_parameters(net, b[0], b[1], parallel=parallel, g_us_per_km=g_us_per_km, **LINE)
    create_line_from_parameters(net, b[1], b[2], **LINE)
    create_gen(net, b[2], p_mw=50.0, vm_pu=1.01)
    create_load(net, b[1], p_mw=250.0, q_mvar=80.0)
    return net


def net_with_single_lines(k):
    """The same grid with the parallel line drawn as k separate lines."""
    net = net_with_parallel_line(1)
    for _ in range(k - 1):
        create_line_from_parameters(net, 0, 1, **LINE)
    return net


def _solve_p3s(net, solver_cls=NewtonPowerflow):
    solved = copy.deepcopy(net)
    solver_cls(solved).calculate(solved, tolerance=TOL, max_iterations=30)
    return solved


def _assert_same_bus_results(solved, ref):
    np.testing.assert_allclose(solved.res_bus.vm_pu.values, ref.res_bus.vm_pu.values, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(solved.res_bus.va_degree.values, ref.res_bus.va_degree.values,
                               rtol=1e-10, atol=1e-10)


_G_NOT_PER_UNIT = pytest.mark.xfail(
    strict=True,
    reason="pre-existing, independent of parallel: pi_model adds G * 1e-6 (siemens) to the "
           "per-unit shunt admittance without the Z_N scaling that B gets")


@pytest.mark.parametrize("parallel", [1, 2, 3])
@pytest.mark.parametrize("g_us_per_km", [0.0, pytest.param(0.5, marks=_G_NOT_PER_UNIT)])
def test_parallel_line_power_flow_matches_pandapower(parallel, g_us_per_km):
    net = net_with_parallel_line(parallel, g_us_per_km)
    ref = copy.deepcopy(net)
    runpp(ref, tolerance_mva=TOL)
    _assert_same_bus_results(_solve_p3s(net), ref)


@pytest.mark.parametrize("k", [2, 3])
def test_parallel_line_equals_k_single_lines(k):
    _assert_same_bus_results(_solve_p3s(net_with_parallel_line(k)), _solve_p3s(net_with_single_lines(k)))


def test_parallel_line_power_flow_matches_pandapower_cpp():
    try:
        from p3s import nr_klu  # noqa: F401  (the C++ extra is optional)
    except ImportError:
        pytest.skip("p3s nr_klu (C++ extra) not installed")
    from p3s.NewtonPowerflowCpp import NewtonPowerflow as NewtonPowerflowCpp

    net = net_with_parallel_line(2)
    ref = copy.deepcopy(net)
    runpp(ref, tolerance_mva=TOL)
    # the C++ solver returns the bus voltages and does not fill net.res_bus
    v = np.asarray(NewtonPowerflowCpp(net).calculate(net, tolerance=TOL, max_iterations=30))
    np.testing.assert_allclose(np.abs(v), ref.res_bus.vm_pu.values, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.angle(v, deg=True), ref.res_bus.va_degree.values, rtol=1e-10, atol=1e-10)
