# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gold test for the cuDSS batched solver backend (CudssBatch).

Validates :class:`p3s.cuda.cudss_batch.CudssBatch` against
``scipy.sparse.linalg.spsolve`` on real power-flow Jacobians (built from pandapower nets
via ``nr_klu.debug_J`` so no pre-saved .npy fixtures are needed). A batch of B systems that
share one CSR pattern but have perturbed values + distinct RHS must each match scipy.

Skipped cleanly when pycuda / a CUDA GPU / libcudss / the compiled nr_klu are unavailable
-- so the suite stays green on machines without cuDSS. Run on the cluster (where cuDSS is
installed) to exercise it.
"""

import numpy as np
import pytest
from pandapower.networks import case14, case118
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from p3s.calculateTrafoTapTable import calculate_trafo_characteristic
from p3s.NewtonPowerflow import NewtonPowerflow

pytest.importorskip("pycuda", reason="pycuda not installed")
nr_klu = pytest.importorskip("p3s.cpp.nr_klu", reason="compiled nr_klu not built")

import pycuda.driver as cuda  # noqa: E402

# GPU + libcudss must both be present, else skip the module.
try:
    import pycuda.driver as _drv

    _drv.init()
    if _drv.Device.count() == 0:
        pytest.skip("no CUDA device", allow_module_level=True)
    # Use the shared PRIMARY-context setup (autoprimaryctx), NOT pycuda.autoinit -- the
    # solver/kernels use the primary context, and mixing a separate autoinit context in the
    # same process invalidates the cached kernel module (cuFuncSetBlockShape: invalid
    # resource handle) when both are exercised across a test session.
    from p3s.cuda import _ctx  # noqa: F401
    from p3s.cuda.cudss_batch import CudssBatch
except OSError as e:
    pytest.skip(f"libcudss not available: {e}", allow_module_level=True)
except Exception as e:  # pragma: no cover
    pytest.skip(f"cuDSS/CUDA unavailable: {e}", allow_module_level=True)

CASE_FUNCS = {"case14": case14, "case118": case118}


def _jacobian(net):
    """Build one power-flow Jacobian (CSR values Jx, Jp, Jj, dim m) at the flat start via
    the validated C++ ``nr_klu.debug_J``."""
    npf = NewtonPowerflow(net)
    yb = npf._YBus.tocsr()
    yb.sort_indices()
    Yp = yb.indptr.astype(np.int32)
    Yj = yb.indices.astype(np.int32)
    Yx = yb.data.astype(np.complex128)
    pv = np.asarray(npf.busses["pv"], np.int32)
    pq = np.asarray(npf.busses["pq"], np.int32)
    V0 = npf._initial_voltage.astype(np.complex128)
    Sbus = npf._sBus.astype(np.complex128)
    d = nr_klu.debug_J(Yp, Yj, Yx, Sbus, V0, pv, pq)
    # debug_J returns CSC; convert to CSR for cuDSS (base-0, sorted columns).
    m = int(d["m"])
    Jcsc = csr_matrix((d["Jx"], d["Ji"], d["Jp"]), shape=(m, m)).tocsc().tocsr()
    Jcsc.sort_indices()
    return Jcsc


@pytest.mark.parametrize("case", list(CASE_FUNCS))
def test_cudss_batch_matches_scipy(case):
    net = CASE_FUNCS[case]()
    calculate_trafo_characteristic(net, inplace=True)
    J = _jacobian(net)
    Jp, Jj, Jx = J.indptr.astype(np.int32), J.indices.astype(np.int32), J.data.astype(np.float64)
    n = J.shape[0]
    nnz = len(Jj)

    B = 8
    rng = np.random.default_rng(0)
    Jx_batch = np.empty((B, nnz), dtype=np.float64)
    rhs_batch = np.empty((B, n), dtype=np.float64)
    x_ref = np.empty((B, n), dtype=np.float64)
    for i in range(B):
        scale = 1.0 + 0.05 * (i - B / 2) / B  # mild perturbation, pattern fixed
        Jx_i = Jx * scale
        b_i = rng.standard_normal(n)
        Jx_batch[i] = Jx_i
        rhs_batch[i] = b_i
        x_ref[i] = spsolve(csr_matrix((Jx_i, Jj, Jp), shape=(n, n)), b_i)

    solver = CudssBatch(Jp, Jj, batch_size=B)
    solver.symbolic_setup(Jx)  # analysis on the base pattern
    # upload per-system values + rhs, then refactor + solve on device
    cuda.memcpy_htod(solver.d_A_batch, np.ascontiguousarray(Jx_batch).reshape(-1))
    cuda.memcpy_htod(solver.d_rhs_batch, np.ascontiguousarray(rhs_batch).reshape(-1))
    solver.reset_refactor_device()
    solver.batch_solve_device()
    cuda.Context.synchronize()
    x_gpu = np.empty(B * n, dtype=np.float64)
    cuda.memcpy_dtoh(x_gpu, solver.d_X_batch)
    x_gpu = x_gpu.reshape(B, n)
    solver.free()

    for i in range(B):
        err = np.linalg.norm(x_gpu[i] - x_ref[i], np.inf)
        denom = max(1.0, np.linalg.norm(x_ref[i], np.inf))
        assert err / denom < 1e-6, f"{case} system {i}: rel err {err / denom:.3e}"


@pytest.mark.parametrize("T", [1024, 4096])
def test_polar_solver_large_batch_no_nan(T):
    """Large time-series batches must converge for EVERY step (no NaN corruption).

    cuDSS 0.8.0's uniform-batch SOLVE non-deterministically corrupts a few systems once
    the batch exceeds ~512 -- the whole reason PolarNewtonSolverCUDA caps the per-batch
    size (``_cudss_safe_chunk``) and retries any residual failures. This regression drives
    the full solver (not just one CudssBatch solve) on a batch big enough to trip the bug
    and asserts every column converges and matches the CPU/KLU reference. The B=8 gold test
    above cannot catch it. Runs twice implicitly via parametrize; the corruption is random,
    so a clean pass at T=4096 is a strong signal.
    """
    from p3s.cuda.NewtonPowerflowCuda import NewtonPowerflowCUDA
    from p3s.NewtonPowerflowCpp import NewtonPowerflow as NewtonPowerflowCpp
    from p3s.timeseries import build_sbus_matrix, dc_initial_voltage

    net = case14()
    net.trafo.shift_degree = 0.0
    calculate_trafo_characteristic(net, inplace=True)

    rng = np.random.default_rng(0)
    n_load = len(net.load)
    scale = 1.0 + 0.05 * rng.standard_normal((n_load, T))
    ts = {
        ("load", "p_mw"): net.load.p_mw.to_numpy()[:, None] * scale,
        ("load", "q_mvar"): net.load.q_mvar.to_numpy()[:, None] * scale,
    }

    ref = NewtonPowerflowCpp(net).calculate_timeseries_cpp(net, ts, tolerance=1e-8, max_iterations=100)
    gpu = NewtonPowerflowCUDA(net)
    sb = build_sbus_matrix(gpu, net, ts)
    v0 = dc_initial_voltage(gpu)
    r = gpu._get_cudss_solver().solve_batch(np.ascontiguousarray(sb), np.ascontiguousarray(v0), max_iter=30, tol=1e-8)

    assert not np.isnan(r["V"]).any(), f"T={T}: NaN in solved voltages"
    n_bad = int((~r["converged"]).sum())
    assert n_bad == 0, f"T={T}: {n_bad} of {T} steps did not converge"
    err = np.abs(np.abs(r["V"]) - np.abs(ref)).max()
    assert err < 1e-6, f"T={T}: max |dVm| vs CPU = {err:.2e}"


def test_polar_solver_recovers_injected_corruption(monkeypatch):
    """The retry loop must recover columns that cuDSS corrupts, in safe-sized sub-batches.

    On CI hardware the real UBATCH corruption may not trigger, so this injects it: every
    ``_run_chunk`` result gets a random ~2% of its columns NaN-ed (first pass) / ~0.5%
    (later passes), mimicking the non-deterministic cuDSS defect. The bounded, chunked
    retry loop in ``_run_newton_batch`` must still return every column converged and
    correct. This guards the fix for the "24 of 1024 did not converge" cluster regression,
    where an UNCAPPED single retry re-corrupted the (large) failed set and left residuals.
    """
    from p3s.cuda import nr_polar_solver as mod
    from p3s.cuda.NewtonPowerflowCuda import NewtonPowerflowCUDA
    from p3s.NewtonPowerflowCpp import NewtonPowerflow as NewtonPowerflowCpp
    from p3s.timeseries import build_sbus_matrix, dc_initial_voltage

    net = case14()
    net.trafo.shift_degree = 0.0
    calculate_trafo_characteristic(net, inplace=True)
    T = 1024
    rng = np.random.default_rng(0)
    n_load = len(net.load)
    scale = 1.0 + 0.05 * rng.standard_normal((n_load, T))
    ts = {
        ("load", "p_mw"): net.load.p_mw.to_numpy()[:, None] * scale,
        ("load", "q_mvar"): net.load.q_mvar.to_numpy()[:, None] * scale,
    }
    ref = NewtonPowerflowCpp(net).calculate_timeseries_cpp(net, ts, tolerance=1e-8, max_iterations=100)

    orig = mod.PolarNewtonSolverCUDA._run_chunk
    prng = np.random.default_rng(1)
    state = {"pass": 0}

    def corrupting(self, V0, *a, **k):
        r = orig(self, V0, *a, **k)
        B = r["V"].shape[1]
        p = 0.02 if state["pass"] == 0 else 0.005
        mask = prng.random(B) < p
        if mask.any():
            r["V"][:, mask] = np.nan
            r["converged"][mask] = False
        state["pass"] += 1
        return r

    monkeypatch.setattr(mod.PolarNewtonSolverCUDA, "_run_chunk", corrupting)

    gpu = NewtonPowerflowCUDA(net)
    sb = build_sbus_matrix(gpu, net, ts)
    v0 = dc_initial_voltage(gpu)
    r = gpu._get_cudss_solver().solve_batch(np.ascontiguousarray(sb), np.ascontiguousarray(v0), max_iter=30, tol=1e-8)

    assert not np.isnan(r["V"]).any(), "residual NaN after retry"
    assert int((~r["converged"]).sum()) == 0, "columns still unconverged after retry"
    assert np.abs(np.abs(r["V"]) - np.abs(ref)).max() < 1e-6
