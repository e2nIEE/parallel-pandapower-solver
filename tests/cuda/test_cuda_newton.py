# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Correctness tests for the GPU Newton-Raphson powerflow (cuSolverRf path).

Single operating-point ``calculate_cuda`` (batch size 1) must converge to the
same bus voltages as pandapower's ``runpp``. Needs the full p3s stack
(numba + pandapower) plus a CUDA GPU.
"""
import copy

import numpy as np
import pytest
from pandapower import runpp
from pandapower.networks import case9, case14, case118

from p3s.calculateTrafoTapTable import calculateTrafoCharacteristic
from p3s.cuda.NewtonPowerflowCuda import NewtonPowerflowCUDA

CASE_FUNCS = {"case9": case9, "case14": case14, "case118": case118}


@pytest.mark.parametrize("case", list(CASE_FUNCS))
def test_calculate_cuda_matches_runpp(case):
    net = CASE_FUNCS[case]()
    calculateTrafoCharacteristic(net, inplace=True)

    ref = copy.deepcopy(net)
    runpp(ref, init="flat")

    npf = NewtonPowerflowCUDA(net)
    voltage = npf.calculate_cuda(net)

    vm = np.abs(voltage)
    va = np.degrees(np.angle(voltage))

    vm_err = np.abs(vm - ref.res_bus.vm_pu.values).max()
    va_err = np.abs(va - ref.res_bus.va_degree.values).max()
    assert vm_err < 1e-6, f"{case}: vm err {vm_err:.2e}"
    assert va_err < 1e-4, f"{case}: va err {va_err:.2e}"


if __name__ == "__main__":
    for c in CASE_FUNCS:
        test_calculate_cuda_matches_runpp(c)
        print(f"{c}: OK")
