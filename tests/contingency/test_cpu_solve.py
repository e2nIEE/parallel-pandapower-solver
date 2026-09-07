# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests: CPU N-1 contingency solver (C++/KLU ``solve_batch_contingency``).

Every contingency solved by the batched C++ path must match the pandapower
full-topology ground truth (Phase 0): served buses to ~1e-6 in voltage magnitude and
~1e-4 in angle, the served/unserved mask exactly, NaN at unserved buses, and a per-case
``converged`` flag. Covers branch + transformer outages, parallel branches, islanding,
multi-branch groups, and the optional island re-slacking.

Requires the compiled ``nr_klu`` with ``solve_batch_contingency`` (build the C++
extension: ``pip install ./p3s/cpp``).
"""

import numpy as np
import pytest

from p3s.contingency.fixtures import FIXTURES
from p3s.contingency.ground_truth import ground_truth
from p3s.contingency.solver_cpp import solve_contingencies_cpp

# Skip the whole module cleanly if the compiled solver (or the new method) is absent.
nr_klu = pytest.importorskip("p3s.cpp.nr_klu", reason="compiled nr_klu not built")
if not hasattr(nr_klu.Solver, "solve_batch_contingency"):
    pytest.skip("nr_klu lacks solve_batch_contingency (rebuild ./p3s/cpp)", allow_module_level=True)


def _assert_matches_ground_truth(res, gt, reslack):
    for c, g in enumerate(res.groups):
        gtc = gt[g]
        served = res.served[:, c]
        # served/unserved mask must match pandapower exactly
        assert np.array_equal(served, gtc.served), f"{g}: served mask mismatch"
        # unserved buses are NaN
        assert np.isnan(res.V[~served, c]).all(), f"{g}: unserved buses must be NaN"
        # case converges
        assert bool(res.converged[c]), f"{g}: did not converge"
        # served-bus voltages match
        if served.any():
            vm_err = np.nanmax(np.abs(res.vm[served, c] - gtc.vm[served]))
            va_err = np.nanmax(np.abs(res.va[served, c] - gtc.va[served]))
            assert vm_err < 1e-6, f"{g}: vm err {vm_err:.2e}"
            assert va_err < 1e-4, f"{g}: va err {va_err:.2e}"


@pytest.mark.parametrize("name", list(FIXTURES))
@pytest.mark.parametrize("n_threads", [1, 0])
def test_cpu_contingency_matches_ground_truth(name, n_threads):
    net = FIXTURES[name]()
    res = solve_contingencies_cpp(net, reslack_islands=False, n_threads=n_threads)
    gt = ground_truth(net, reslack_islands=False)
    assert res.groups == list(gt.keys())
    _assert_matches_ground_truth(res, gt, reslack=False)


def test_cpu_contingency_reslacking_matches_ground_truth():
    net = FIXTURES["generator_island"]()
    res = solve_contingencies_cpp(net, reslack_islands=True, n_threads=1)
    gt = ground_truth(net, reslack_islands=True)
    _assert_matches_ground_truth(res, gt, reslack=True)
    # with re-slacking the whole island is served
    assert res.served[:, 0].all()


def test_result_table_shapes_and_dtypes():
    net = FIXTURES["radial_spur"]()
    res = solve_contingencies_cpp(net, n_threads=1)
    n = len(net.bus)
    L = len(res.groups)
    assert res.V.shape == (n, L)
    assert res.served.shape == (n, L)
    assert res.converged.shape == (L,)
    assert res.iterations.shape == (L,)
    assert res.V.dtype == np.complex128
    assert res.served.dtype == bool
    # vm/va derived views are NaN exactly where unserved
    assert np.array_equal(np.isnan(res.vm), ~res.served)


def test_threads_give_identical_results():
    """Serial and parallel batches must produce identical results (cases independent)."""
    net = FIXTURES["multi_branch_group"]()
    r1 = solve_contingencies_cpp(net, n_threads=1)
    r0 = solve_contingencies_cpp(net, n_threads=0)
    # compare served buses (NaN positions equal, values equal)
    assert np.array_equal(np.isnan(r1.V), np.isnan(r0.V))
    m = ~np.isnan(r1.V)
    assert np.allclose(r1.V[m], r0.V[m], atol=1e-10)
    assert np.array_equal(r1.converged, r0.converged)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
