# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify the C++ solver is internally correct on p3s's OWN Ybus/Sbus:
the converged voltage must drive p3s's own power-balance mismatch to ~0.
This isolates solver correctness from any p3s-vs-pandapower model gap.
"""

import numpy as np
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
    npf = NewtonPowerflow(net)
    V = npf.calculate(net, init="dc", tolerance=1e-10, max_iterations=30)

    # p3s's own model: Ybus, Sbus, pv/pq
    Y = npf._YBus
    Sbus = npf._sBus
    pv = npf.busses["pv"]
    pq = npf.busses["pq"]
    pvpq = np.r_[pv, pq]
    mis = V * np.conj(Y @ V) - Sbus
    F = np.r_[mis[pvpq].real, mis[pq].imag]
    lines.append(f"[{name}] n={Y.shape[0]} self-mismatch |F|inf = {np.max(np.abs(F)):.2e}")

txt = "\n".join(lines) + "\n"
print(txt)
