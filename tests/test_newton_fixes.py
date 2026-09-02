# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for two p3s-Newton.

1. _pre_dc_solve fell back via nan_to_num on a singular DC-init matrix (radial, line-only
   feeders), emitting a MatrixRankWarning and doing a wasted singular solve. It now detects
   singularity and falls back cleanly to a flat start (no warning).
2. NewtonPowerflow.calculate had _parse_results commented out, so net.res_bus was never
   written. It is now enabled; results must match pandapower runpp.
"""
import copy
import warnings

import numpy as np
import pytest
from scipy.sparse.linalg import MatrixRankWarning

from pandapower.run import runpp
from pandapower.create import (
    create_empty_network, create_bus, create_ext_grid,
    create_line_from_parameters, create_load,
)
from pandapower.networks.power_system_test_cases import case9, case14

from p3s.NewtonPowerflow import NewtonPowerflow
from p3s.calculateTrafoTapTable import calculateTrafoCharacteristic


def _radial_feeder(n_load):
    net = create_empty_network(sn_mva=1.0)
    b = [create_bus(net, vn_kv=20.0) for _ in range(n_load + 1)]
    create_ext_grid(net, b[0], vm_pu=1.0)
    for i in range(n_load):
        create_line_from_parameters(
            net, b[i], b[i + 1], length_km=0.5,
            r_ohm_per_km=0.3, x_ohm_per_km=0.4, c_nf_per_km=8.0, max_i_ka=1.0,
        )
    for i in range(1, n_load + 1):
        create_load(net, b[i], p_mw=0.05, q_mvar=0.02)
    return net


def test_dc_init_no_rank_warning_on_radial_feeder():
    """DC init on a singular-Bbus radial feeder must not emit MatrixRankWarning and must
    still converge to the correct solution."""
    net = _radial_feeder(20)
    ref = copy.deepcopy(net)
    runpp(ref)

    work = copy.deepcopy(net)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)  # fail the test if it warns
        NewtonPowerflow(work).calculate(work, init="dc", tolerance=1e-8, max_iterations=30)

    np.testing.assert_allclose(
        work.res_bus.vm_pu.values, ref.res_bus.vm_pu.values, rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        work.res_bus.va_degree.values, ref.res_bus.va_degree.values, rtol=0, atol=1e-6
    )


@pytest.mark.parametrize("init", ["dc", "flat"])
def test_newton_writes_res_bus_matching_runpp(init):
    """_parse_results is enabled: res_bus must be populated and match runpp."""
    net = _radial_feeder(15)
    ref = copy.deepcopy(net)
    runpp(ref)

    work = copy.deepcopy(net)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        NewtonPowerflow(work).calculate(work, init=init, tolerance=1e-8, max_iterations=30)

    assert len(work.res_bus) == len(work.bus)
    np.testing.assert_allclose(
        work.res_bus.vm_pu.values, ref.res_bus.vm_pu.values, rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        work.res_bus.va_degree.values, ref.res_bus.va_degree.values, rtol=0, atol=1e-6
    )


@pytest.mark.parametrize("fn", [case9, case14])
def test_dc_init_preserves_pv_magnitudes(fn):
    """DC init returns magnitude 1.0 for all pvpq buses; this must not clobber the known
    voltage-magnitude set-points at PV (gen) buses. Regression for a bug where dc-init
    PV nets converged with |V|=1.0 at gen buses (off by ~0.09 vs runpp)."""
    net = fn()
    calculateTrafoCharacteristic(net, inplace=True)
    ref = copy.deepcopy(net)
    runpp(ref)

    work = copy.deepcopy(net)
    nf = NewtonPowerflow(work)
    pv = nf.busses["pv"]
    assert len(pv) > 0  # this case actually has PV buses

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nf.calculate(work, init="dc", tolerance=1e-8, max_iterations=30)

    # PV-bus magnitudes must equal their set-points (and runpp), not 1.0.
    np.testing.assert_allclose(
        work.res_bus.vm_pu.values[pv], ref.res_bus.vm_pu.values[pv], rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        work.res_bus.vm_pu.values, ref.res_bus.vm_pu.values, rtol=0, atol=1e-6
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
