# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end throughput benchmark:
batched GPU time-series Newton vs a CPU loop, on case9241pegase.

Success criterion: total wall-clock for the GPU batched solver
over T operating points must beat looping the CPU Newton T times, at large T.
Run as a script to print the table.
"""

import copy
import sys
import time

import numpy as np
import pytest
from pandapower.networks import case9241pegase

from p3s.calculateTrafoTapTable import calculate_trafo_characteristic
from p3s.NewtonPowerflow import NewtonPowerflow

pytest.importorskip("pycuda", reason="pycuda not installed")

from p3s.cuda.NewtonPowerflowCuda import NewtonPowerflowCUDA  # noqa: E402


def _make_profile(net, T, seed=0):
    """Per-load jitter around the (convergent) base operating point.

    A global rescaling of all loads pushes pegase 9241 out of convergence (slack can't
    absorb the imbalance); small independent per-load perturbations keep every step in
    the convergent regime, which is where a throughput comparison is meaningful. Returns
    a (n_load, T) per-load scale matrix.
    """
    rng = np.random.default_rng(seed)
    n_load = len(net.load)
    scale = 1.0 + 0.05 * rng.standard_normal((n_load, T))  # ~N(1, 0.05) per load/step
    p0 = net.load.p_mw.to_numpy()[:, None]
    q0 = net.load.q_mvar.to_numpy()[:, None]
    return {("load", "p_mw"): p0 * scale, ("load", "q_mvar"): q0 * scale}, scale


def _cpu_loop(net, scale):
    """Reference: solve each operating point with the CPU Newton. ``scale`` is
    (n_load, T): per-load multiplier for each time step."""
    base_p = net.load.p_mw.to_numpy()
    base_q = net.load.q_mvar.to_numpy()
    T = scale.shape[1]
    out = np.empty((len(net.bus), T), dtype=np.complex128)
    for t in range(T):
        n2 = copy.deepcopy(net)
        n2.load.p_mw = base_p * scale[:, t]
        n2.load.q_mvar = base_q * scale[:, t]
        npf = NewtonPowerflow(n2)
        # calculate() parses results into the net; read voltages back from res_bus.
        npf.calculate(n2)
        out[:, t] = n2.res_bus.vm_pu.values * np.exp(1j * np.radians(n2.res_bus.va_degree.values))
    return out


def benchmark(T_list=(64, 256, 1024)):
    net = case9241pegase()
    calculate_trafo_characteristic(net, inplace=True)
    print(f"case9241pegase: {len(net.bus)} buses, {len(net.load)} loads")

    for T in T_list:
        ts, scale = _make_profile(net, T)

        # --- GPU batched (end-to-end incl. all transfers/assembly) ---
        npf = NewtonPowerflowCUDA(net)
        t0 = time.perf_counter()
        v_gpu = npf.calculate_timeseries_cuda(net, ts, tolerance=1e-6)
        gpu_total = time.perf_counter() - t0

        # --- CPU loop ---
        t0 = time.perf_counter()
        v_cpu = _cpu_loop(net, scale)
        cpu_total = time.perf_counter() - t0

        # correctness spot-check on a few steps
        idx = np.linspace(0, T - 1, min(T, 5)).astype(int)
        vm_err = np.abs(np.abs(v_gpu[:, idx]) - np.abs(v_cpu[:, idx])).max()

        speedup = cpu_total / gpu_total
        print(
            f"T={T:5d} | GPU {gpu_total:8.3f}s ({gpu_total / T * 1e3:6.2f} ms/op) | "
            f"CPU {cpu_total:8.3f}s ({cpu_total / T * 1e3:6.2f} ms/op) | "
            f"speedup {speedup:5.2f}x | vm_err {vm_err:.1e}"
        )


def test_benchmark_smoke():
    """Lightweight CI guard: just run a tiny T and assert correctness (not speed)."""
    net = case9241pegase()
    calculate_trafo_characteristic(net, inplace=True)
    ts, scale = _make_profile(net, 4)
    npf = NewtonPowerflowCUDA(net)
    v_gpu = npf.calculate_timeseries_cuda(net, ts, tolerance=1e-6)
    v_cpu = _cpu_loop(net, scale)
    vm_err = np.abs(np.abs(v_gpu) - np.abs(v_cpu)).max()
    assert vm_err < 1e-5, f"vm err {vm_err:.2e}"


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:]] or [64, 256, 1024]
    benchmark(args)
