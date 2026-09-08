# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""N-1 contingency analysis on the large case9241pegase network (no re-slacking).

This is the scale test for the CPU contingency solver: 9241 buses, ~16k branches. It
validates a handful of representative single-branch contingencies (lines + a transformer)
against the pandapower full-topology ground truth.

A full N-1 sweep over all ~16k branches via the pandapower oracle would be far too slow
for CI, so we test a representative subset; the small-net fixtures cover the
islanding / parallel-branch / re-slacking edge cases exhaustively. Full tests were performed
during development, and showed the same results.
"""

import copy
import time

import numpy as np
import pytest
from pandapower.networks import case9241pegase

from p3s.calculateTrafoTapTable import calculateTrafoCharacteristic
from p3s.contingency.ground_truth import enumerate_contingencies, solve_contingency
from p3s.contingency.solver_cpp import solve_contingencies_cpp
from p3s.NewtonPowerflowCpp import NewtonPowerflow as NewtonPowerflowCpp

pp = pytest.importorskip("pandapower")
nr_klu = pytest.importorskip("p3s.cpp.nr_klu", reason="compiled nr_klu not built")
if not hasattr(nr_klu.Solver, "solve_batch_contingency"):
    pytest.skip("nr_klu lacks solve_batch_contingency (rebuild ./p3s/cpp)", allow_module_level=True)

# Representative contingencies: the first few lines (each its own group) + one transformer.
N_LINE_CONTINGENCIES = 4


@pytest.fixture(scope="module")
def pegase_net():
    net = case9241pegase()
    calculateTrafoCharacteristic(net, inplace=True)
    net.line["outage_group"] = None
    net.trafo["outage_group"] = None
    for i, lid in enumerate(list(net.line.index[:N_LINE_CONTINGENCIES])):
        net.line.loc[lid, "outage_group"] = f"L{i}"
    net.trafo.loc[net.trafo.index[0], "outage_group"] = "T0"
    return net


@pytest.fixture(scope="module")
def base_accuracy(pegase_net):
    net = copy.deepcopy(pegase_net)
    v = NewtonPowerflowCpp(net).calculate(net, init="flat", max_iterations=50)
    ref = copy.deepcopy(net)
    pp.runpp(ref, init="flat")
    vm_err = float(np.abs(np.abs(v) - ref.res_bus.vm_pu.values).max())
    va_err = float(np.abs(np.degrees(np.angle(v)) - ref.res_bus.va_degree.values).max())
    return vm_err, va_err


def test_pegase_n_minus_1_matches_pandapower(pegase_net, base_accuracy):
    base_vm_err, base_va_err = base_accuracy
    # allow the contingency solve to be no worse than the base-case modeling error,
    # plus a small absolute margin for the topology change.
    vm_tol = base_vm_err + 1e-5
    va_tol = base_va_err + 1e-3

    res = solve_contingencies_cpp(pegase_net, reslack_islands=False, n_threads=0, init="flat")
    assert res.groups == enumerate_contingencies(pegase_net)
    assert len(res.groups) == N_LINE_CONTINGENCIES + 1  # lines + one trafo

    for c, g in enumerate(res.groups):
        gt = solve_contingency(pegase_net, g, reslack_islands=False)
        assert gt.converged, f"{g}: pandapower reference did not converge"
        assert bool(res.converged[c]), f"{g}: p3s contingency did not converge"

        served = res.served[:, c]
        assert np.array_equal(served, gt.served), f"{g}: served mask mismatch"
        assert np.isnan(res.V[~served, c]).all(), f"{g}: unserved buses must be NaN"

        vm_err = float(np.nanmax(np.abs(res.vm[served, c] - gt.vm[served])))
        va_err = float(np.nanmax(np.abs(res.va[served, c] - gt.va[served])))
        assert vm_err < vm_tol, f"{g}: vm err {vm_err:.2e} > {vm_tol:.2e}"
        assert va_err < va_tol, f"{g}: va err {va_err:.2e} > {va_tol:.2e}"


def test_pegase_shares_one_symbolic_factorization(pegase_net):
    """All pegase contingencies solve in one batch (shared pattern) and converge."""
    res = solve_contingencies_cpp(pegase_net, reslack_islands=False, n_threads=0, init="flat")
    assert res.converged.all()
    assert res.V.shape == (len(pegase_net.bus), len(res.groups))
    # every bus served (these contingencies do not island the meshed transmission net)
    assert res.served.all()


# Number of single-line contingencies to time. Large enough that the shared-symbolic
# amortization shows (batch overhead is paid once), small enough that the pandapower
# reference loop -- which does a full runpp per contingency -- finishes in reasonable CI
# time (each pegase runpp is ~hundreds of ms).
N_BENCH_CONTINGENCIES = 50


@pytest.fixture(scope="module")
def pegase_bench_net():
    net = case9241pegase()
    calculateTrafoCharacteristic(net, inplace=True)
    net.line["outage_group"] = None
    net.trafo["outage_group"] = None
    for i, lid in enumerate(list(net.line.index[:N_BENCH_CONTINGENCIES])):
        net.line.loc[lid, "outage_group"] = f"L{i}"
    return net


def test_pegase_speedup_vs_pandapower(pegase_bench_net):
    """Time the p3s N-1 batch against a pandapower per-contingency loop and report
    the speedup. p3s shares one symbolic factorization across the whole batch; the
    pandapower reference re-solves the full net per contingency.

    This is a reporting test: it prints the timings/speedup (run with ``-s``) and only
    asserts the batch is meaningfully faster, so it does not flake on timing noise.
    """
    net = pegase_bench_net
    groups = enumerate_contingencies(net)
    n_cont = len(groups)
    assert n_cont == N_BENCH_CONTINGENCIES

    # --- warm-up (build solver / pattern, JIT numba, fill caches) so the timed run
    # measures steady-state work, not one-off setup ---
    _ = solve_contingencies_cpp(net, reslack_islands=False, n_threads=0, init="flat")

    # --- p3s batch (all cores) ---
    t0 = time.perf_counter()
    res = solve_contingencies_cpp(net, reslack_islands=False, n_threads=0, init="flat")
    t_batch = time.perf_counter() - t0
    assert res.converged.all(), "all benchmark contingencies must converge"

    # --- pandapower per-contingency loop (the reference an N-1 study would otherwise run) ---
    t0 = time.perf_counter()
    for g in groups:
        gt = solve_contingency(net, g, reslack_islands=False)
        assert gt.converged
    t_pp = time.perf_counter() - t0

    speedup = t_pp / t_batch
    n_bus = len(net.bus)
    print(
        f"\n[pegase N-1 speedup] {n_bus} buses, {n_cont} single-line contingencies\n"
        f"  p3s CPU batch (all cores): {t_batch * 1e3:8.1f} ms "
        f"({t_batch / n_cont * 1e3:.3f} ms/contingency)\n"
        f"  pandapower per-contingency loop:{t_pp * 1e3:8.1f} ms "
        f"({t_pp / n_cont * 1e3:.3f} ms/contingency)\n"
        f"  speedup: {speedup:.1f}x"
    )

    # Sanity floor only -- the real number is in the printout; keep the assert loose so
    # the test does not flake on a busy CI host.
    assert speedup > 5.0, f"expected a large speedup, got {speedup:.1f}x"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
