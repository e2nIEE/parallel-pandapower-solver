# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Correctness test for the batched GPU time-series Newton.

For each time step the batched GPU solver must converge to the same bus voltages as a
per-step CPU reference (pandapower ``runpp`` at that operating point). Uses a synthetic
load-scaling profile so the test is hermetic (no simbench/network fetch).
"""

import copy

import numpy as np
import pytest
from pandapower import runpp
from pandapower.networks import case9, case14

from p3s.calculateTrafoTapTable import calculateTrafoCharacteristic
from p3s.cuda.NewtonPowerflowCuda import NewtonPowerflowCUDA

CASE_FUNCS = {"case9": case9, "case14": case14}


def _make_profile(net, T, seed=0):
    """Synthetic timeseries: scale each load's p_mw/q_mvar by a per-step factor.

    Returns (timeseries_dict, scale[T]) where scale is the per-timestep multiplier
    applied to the base loads -- used to build the CPU reference identically.
    """
    rng = np.random.default_rng(seed)
    scale = 0.7 + 0.6 * rng.random(T)  # in [0.7, 1.3]
    p0 = net.load.p_mw.to_numpy()[:, None]  # (n_load, 1)
    q0 = net.load.q_mvar.to_numpy()[:, None]
    ts = {
        ("load", "p_mw"): p0 * scale[None, :],  # (n_load, T)
        ("load", "q_mvar"): q0 * scale[None, :],
    }
    return ts, scale


@pytest.mark.parametrize("case", list(CASE_FUNCS))
def test_timeseries_matches_per_step_runpp(case):
    net = CASE_FUNCS[case]()
    calculateTrafoCharacteristic(net, inplace=True)
    T = 12

    npf = NewtonPowerflowCUDA(net)
    ts, scale = _make_profile(net, T)

    # GPU batched solve (force a small batch so chunking is exercised)
    voltages = npf.calculate_timeseries_cuda(net, ts, batch_size=5)
    assert voltages.shape == (len(net.bus), T)

    vm_gpu = np.abs(voltages)
    va_gpu = np.degrees(np.angle(voltages))

    for t in range(T):
        ref = copy.deepcopy(net)
        ref.load.p_mw = net.load.p_mw.to_numpy() * scale[t]
        ref.load.q_mvar = net.load.q_mvar.to_numpy() * scale[t]
        runpp(ref, init="flat")

        vm_err = np.abs(vm_gpu[:, t] - ref.res_bus.vm_pu.values).max()
        va_err = np.abs(va_gpu[:, t] - ref.res_bus.va_degree.values).max()
        assert vm_err < 1e-6, f"{case} t={t}: vm err {vm_err:.2e}"
        assert va_err < 1e-4, f"{case} t={t}: va err {va_err:.2e}"


if __name__ == "__main__":
    for c in CASE_FUNCS:
        test_timeseries_matches_per_step_runpp(c)
        print(f"{c}: OK")
