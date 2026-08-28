# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests: GPU N-1 contingency solver (fully-resident polar cuSolverRf path).

Mirrors ``test_cpu_solve.py`` but for ``solve_contingencies_cuda``: every
contingency solved on the GPU must match the pandapower full-topology ground truth
(served buses to ~1e-6 vm / ~1e-4 va, exact served/unserved mask, NaN at unserved) AND
the CPU C++/KLU batch path to solver tolerance.

Skipped cleanly when pycuda / a CUDA GPU / the compiled nr_klu are unavailable.
"""
import numpy as np
import pytest

pytest.importorskip("pycuda", reason="pycuda not installed")
nr_klu = pytest.importorskip("p3s.cpp.nr_klu", reason="compiled nr_klu not built")
if not hasattr(nr_klu.Solver, "solve_batch_contingency"):
    pytest.skip("nr_klu lacks solve_batch_contingency (rebuild ./p3s/cpp)",
                allow_module_level=True)

from p3s.contingency.fixtures import FIXTURES
from p3s.contingency.ground_truth import ground_truth
from p3s.contingency.solver_cpp import solve_contingencies_cpp

try:
    import pycuda.driver as _drv
    _drv.init()
    if _drv.Device.count() == 0:
        pytest.skip("no CUDA device", allow_module_level=True)
    from p3s.contingency.solver_cuda import solve_contingencies_cuda
except Exception as e:  # pragma: no cover - environment dependent
    pytest.skip(f"CUDA unavailable: {e}", allow_module_level=True)


def _detect_backends():
    """Return the linear-solve backends usable in THIS environment.

    Backends vary by CUDA install: cusolverRf ("rf") segfaults on CUDA 12.4; cuDSS
    ("cudss") needs libcudss. We probe each on a tiny fixture and keep only those that
    solve correctly, so the parametrized tests run on whatever the machine supports.
    """
    from p3s.contingency.solver_cuda import solve_contingencies_cuda as _solve
    net = FIXTURES["parallel_branch"]()
    ok = []
    for b in ("rf", "qr", "cudss"):
        try:
            r = _solve(net, backend=b)
            if bool(r.converged.all()):
                ok.append(b)
        except Exception:
            pass
    return ok


_AVAILABLE_BACKENDS = _detect_backends()
if not _AVAILABLE_BACKENDS:
    pytest.skip("no working GPU linear-solve backend", allow_module_level=True)


def _assert_matches_ground_truth(res, gt):
    for c, g in enumerate(res.groups):
        gtc = gt[g]
        served = res.served[:, c]
        assert np.array_equal(served, gtc.served), f"{g}: served mask mismatch"
        assert np.isnan(res.V[~served, c]).all(), f"{g}: unserved buses must be NaN"
        assert bool(res.converged[c]), f"{g}: did not converge"
        if served.any():
            vm_err = np.nanmax(np.abs(res.vm[served, c] - gtc.vm[served]))
            va_err = np.nanmax(np.abs(res.va[served, c] - gtc.va[served]))
            assert vm_err < 1e-6, f"{g}: vm err {vm_err:.2e}"
            assert va_err < 1e-4, f"{g}: va err {va_err:.2e}"


@pytest.mark.parametrize("backend", _AVAILABLE_BACKENDS)
@pytest.mark.parametrize("name", list(FIXTURES))
def test_gpu_contingency_matches_ground_truth(name, backend):
    net = FIXTURES[name]()
    res = solve_contingencies_cuda(net, reslack_islands=False, backend=backend)
    gt = ground_truth(net, reslack_islands=False)
    assert res.groups == list(gt.keys())
    _assert_matches_ground_truth(res, gt)


@pytest.mark.parametrize("backend", _AVAILABLE_BACKENDS)
def test_gpu_contingency_reslacking_matches_ground_truth(backend):
    net = FIXTURES["generator_island"]()
    res = solve_contingencies_cuda(net, reslack_islands=True, backend=backend)
    gt = ground_truth(net, reslack_islands=True)
    _assert_matches_ground_truth(res, gt)
    assert res.served[:, 0].all()


@pytest.mark.parametrize("backend", _AVAILABLE_BACKENDS)
@pytest.mark.parametrize("name", list(FIXTURES))
def test_gpu_matches_cpu(name, backend):
    """GPU and CPU C++ batch must agree on served mask, NaN positions, and served-bus V."""
    net = FIXTURES[name]()
    reslack = (name == "generator_island")
    rc = solve_contingencies_cpp(net, reslack_islands=reslack, n_threads=1)
    rg = solve_contingencies_cuda(net, reslack_islands=reslack, backend=backend)
    assert np.array_equal(rc.served, rg.served)
    assert np.array_equal(np.isnan(rc.V), np.isnan(rg.V))
    m = ~np.isnan(rc.V)
    assert np.allclose(rc.V[m], rg.V[m], atol=1e-8)
    assert np.array_equal(rc.converged, rg.converged)
