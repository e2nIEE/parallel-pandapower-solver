# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end test: NewtonPowerflowCpp.calculate vs pandapower runpp."""

import time

import numpy as np
import pandapower as pp
from pandapower.networks.power_system_test_cases import case9, case14, case118, case9241pegase

from p3s.calculateTrafoTapTable import calculateTrafoCharacteristic
from p3s.NewtonPowerflowCpp import NewtonPowerflow

CASES = {"case9": case9, "case14": case14, "case118": case118, "case9241": case9241pegase}

lines = []
for name, fn in CASES.items():
    net = fn()
    if "trafo" in net and len(net.trafo) > 0:
        net.trafo.shift_degree = 0.0
        calculateTrafoCharacteristic(net, inplace=True)

    pp.runpp(net)  # reference
    vm_ref = net.res_bus.vm_pu.to_numpy()
    va_ref = net.res_bus.va_degree.to_numpy()

    npf = NewtonPowerflow(net)
    # first solve (cold: includes KLU analyze)
    t0 = time.perf_counter()
    V = npf.calculate(net, init="dc", tolerance=1e-8, max_iterations=30)
    t_cold = (time.perf_counter() - t0) * 1e3

    # warm solves (reuse cached Solver)
    warm = []
    for _ in range(10):
        t0 = time.perf_counter()
        npf.calculate(net, init="dc", tolerance=1e-8, max_iterations=30)
        warm.append((time.perf_counter() - t0) * 1e3)

    vm = np.abs(V)
    va = np.angle(V, deg=True)
    # align angle reference frame (p3s may differ by global shift if slack handling differs)
    vm_err: float = np.max(np.abs(vm - vm_ref))
    va_err: float = np.max(np.abs(((va - va_ref) + 180) % 360 - 180))

    lines.append(
        f"[{name}] n={len(net.bus)} vm_err={vm_err:.2e} va_err={va_err:.2e}deg "
        f"cold={t_cold:.2f}ms warm_best={min(warm):.2f}ms"
    )

txt = "\n".join(lines) + "\n"
print(txt)
