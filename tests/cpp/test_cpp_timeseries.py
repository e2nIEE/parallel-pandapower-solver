# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Correctness test for the batched C++/KLU time-series Newton.

Each time step of the batched C++ solver must converge to the same bus voltages as a
per-step pandapower ``runpp`` reference. Uses a synthetic per-load scaling profile so
the test is hermetic. Mirrors ``tests/cuda/test_cuda_timeseries.py``.
"""

import copy

import numpy as np
import pytest
from pandapower import runpp
from pandapower.networks import case9, case14, case118

from p3s.calculateTrafoTapTable import calculate_trafo_characteristic
from p3s.NewtonPowerflowCpp import NewtonPowerflow as NewtonPowerflowCpp
from p3s.timeseries import dc_initial_voltage

try:
    from p3s import nr_klu  # type: ignore[attr-defined]
except ImportError:
    from p3s.cpp import nr_klu  # type: ignore[attr-defined]


CASE_FUNCS = {"case9": case9, "case14": case14, "case118": case118}


def _make_profile(net, T, seed=0):
    """Synthetic per-load p_mw/q_mvar scaling profile. Returns (timeseries, scale)."""
    rng = np.random.default_rng(seed)
    n_load = len(net.load)
    scale = 1.0 + 0.05 * rng.standard_normal((n_load, T))
    p0 = net.load.p_mw.to_numpy()[:, None]
    q0 = net.load.q_mvar.to_numpy()[:, None]
    ts = {("load", "p_mw"): p0 * scale, ("load", "q_mvar"): q0 * scale}
    return ts, scale


@pytest.mark.parametrize("case", list(CASE_FUNCS))
@pytest.mark.parametrize("n_threads", [1, 0])
def test_cpp_timeseries_matches_runpp(case, n_threads):
    net = CASE_FUNCS[case]()
    if "trafo" in net and len(net.trafo) > 0:
        net.trafo.shift_degree = 0.0
        calculate_trafo_characteristic(net, inplace=True)
    T = 12

    npf = NewtonPowerflowCpp(net)
    ts, scale = _make_profile(net, T)
    voltages = npf.calculate_timeseries_cpp(net, ts, n_threads=n_threads)
    assert voltages.shape == (len(net.bus), T)

    vm_gpu = np.abs(voltages)
    va_gpu = np.degrees(np.angle(voltages))

    base_p = net.load.p_mw.to_numpy()
    base_q = net.load.q_mvar.to_numpy()
    for t in range(T):
        ref = copy.deepcopy(net)
        ref.load.p_mw = base_p * scale[:, t]
        ref.load.q_mvar = base_q * scale[:, t]
        runpp(ref, init="flat")
        vm_err = np.abs(vm_gpu[:, t] - ref.res_bus.vm_pu.values).max()
        va_err = np.abs(va_gpu[:, t] - ref.res_bus.va_degree.values).max()
        assert vm_err < 1e-6, f"{case} t={t} nt={n_threads}: vm err {vm_err:.2e}"
        assert va_err < 1e-4, f"{case} t={t} nt={n_threads}: va err {va_err:.2e}"


@pytest.mark.parametrize("n_threads", [1, 0])
def test_batch_column_reuse_matches_fresh_solves(n_threads):
    """Batch columns share a pooled KLU numeric (pivot order carried between columns).

    A diverging column in the middle of the batch must not poison later columns, and
    every column -- converged or not -- must match a fresh single-column solve.
    """
    net = case118()
    net.trafo.shift_degree = 0.0
    calculate_trafo_characteristic(net, inplace=True)
    npf = NewtonPowerflowCpp(net)

    Y = npf._YBus.tocsr()
    args = (
        np.ascontiguousarray(Y.indptr, dtype=np.int32),
        np.ascontiguousarray(Y.indices, dtype=np.int32),
        np.ascontiguousarray(Y.data, dtype=np.complex128),
        np.ascontiguousarray(npf.busses["pv"], dtype=np.int32),
        np.ascontiguousarray(npf.busses["pq"], dtype=np.int32),
    )
    V0 = dc_initial_voltage(npf)

    scales = [1.0, 1.02, 5.0, 0.98, 1.0, 3.0, 0.9]  # 5.0 does not converge
    Sbus = np.stack([npf._sBus * k for k in scales], axis=1)

    s = nr_klu.Solver(*args)
    rb = s.solve_batch(Sbus, V0, n_threads=n_threads)
    assert not rb["converged"][2] and rb["converged"].sum() == len(scales) - 1

    for t in range(len(scales)):
        fresh = nr_klu.Solver(*args)
        r = fresh.solve(Sbus[:, t], V0)
        assert rb["converged"][t] == r["converged"], f"t={t}"
        assert rb["iterations"][t] == r["iterations"], f"t={t}"
        if r["converged"]:
            assert np.abs(rb["V"][:, t] - r["V"]).max() < 1e-12, f"t={t}"
