# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the batched static-pivot LU ("lean" backend) of the GPU polar Newton solver.

Two layers:

* the host-side plan (``lean_lu_plan.py``) and Jacobian topology -- pure numpy, run
  everywhere: the plan, executed serially, must solve what scipy solves;
* the GPU solver on that plan -- skipped without pycuda / a CUDA device / the compiled
  ``nr_klu``: it must reproduce the C++ solver (same voltages, same iteration counts, same
  line-search decisions), with the CPU safety net switched off unless a test is about it.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from pandapower.networks import case14, case118
from scipy.sparse.linalg import spsolve

from p3s.calculateTrafoTapTable import calculate_trafo_characteristic
from p3s.cuda.lean_lu_plan import build_lean_lu_plan, reference_factor_solve
from p3s.cuda.polar_topology import build_polar_topology, jacobian_values_host
from p3s.NewtonPowerflow import NewtonPowerflow

CASES = {"case14": case14, "case118": case118}


def _problem(case):
    """(Yp, Yj, Yx, pv, pq, Sbus, V0) of a pandapower test case."""
    net = CASES[case]()
    calculate_trafo_characteristic(net, inplace=True)
    npf = NewtonPowerflow(net)
    yb = npf._YBus.tocsr()
    yb.sort_indices()
    return (
        yb.indptr.astype(np.int32),
        yb.indices.astype(np.int32),
        yb.data.astype(np.complex128),
        np.asarray(npf.busses["pv"], np.int32),
        np.asarray(npf.busses["pq"], np.int32),
        npf._sBus.astype(np.complex128),
        npf._initial_voltage.astype(np.complex128),
    )


def _scaled_injections(Sbus, T, spread, seed=0):
    """(n, T) injections: every bus's injection scaled by 1 +- spread per column."""
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(Sbus[:, None] * (1.0 + spread * rng.standard_normal((Sbus.shape[0], T))))


# ------------------------------------------------------------------ host side
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("slots", [1, 4, 32])
def test_plan_solves_power_flow_jacobian(case, slots):
    Yp, Yj, Yx, pv, pq, _, V0 = _problem(case)
    t = build_polar_topology(Yp, Yj, Yx, pv, pq)
    rng = np.random.default_rng(1)
    Vm = np.abs(V0) * (1 + 0.02 * rng.standard_normal(t.n))
    Va = np.angle(V0) + 0.05 * rng.standard_normal(t.n)
    Jx = jacobian_values_host(t, Vm, Va)
    b = rng.standard_normal(t.m)

    plan = build_lean_lu_plan(t.Jp, t.Jj, slots=slots)
    x = reference_factor_solve(plan, Jx, b)
    x_ref = spsolve(sp.csr_matrix((Jx, t.Jj, t.Jp), shape=(t.m, t.m)).tocsc(), b)
    assert np.abs(x - x_ref).max() <= 1e-9 * max(1.0, np.abs(x_ref).max())
    # every task is scheduled exactly once, and every pair belongs to a task
    assert len(np.unique(plan.tdst & 0x7FFFFFFF)) == len(plan.tdst)
    assert plan.tpp[-1] == plan.n_pairs == len(plan.pa)


def test_plan_handles_unsymmetric_pattern():
    """The plan symmetrizes the pattern itself; entries missing on one side are zeros."""
    rng = np.random.default_rng(2)
    m = 80
    A = sp.random(m, m, density=0.05, random_state=3, format="csr")
    A = (A + sp.diags(4.0 + rng.random(m))).tocsr()  # diagonally dominant: no pivoting needed
    A.sort_indices()
    assert (A != A.T).nnz > 0, "pattern is meant to be unsymmetric"
    b = rng.standard_normal(m)
    plan = build_lean_lu_plan(A.indptr, A.indices, slots=4)
    x = reference_factor_solve(plan, A.data, b)
    x_ref = np.linalg.solve(A.toarray(), b)
    assert np.abs(x - x_ref).max() <= 1e-10 * max(1.0, np.abs(x_ref).max())


@pytest.mark.parametrize("case", list(CASES))
def test_host_jacobian_matches_cpp(case):
    nr_klu = pytest.importorskip("p3s.cpp.nr_klu", reason="compiled nr_klu not built")
    Yp, Yj, Yx, pv, pq, Sbus, V0 = _problem(case)
    t = build_polar_topology(Yp, Yj, Yx, pv, pq)
    d = nr_klu.debug_J(Yp, Yj, Yx, Sbus, V0, pv, pq)
    J_cpp = sp.csc_matrix((d["Jx"], d["Ji"], d["Jp"]), shape=(d["m"], d["m"])).toarray()
    J_host = sp.csr_matrix((jacobian_values_host(t, np.abs(V0), np.angle(V0)), t.Jj, t.Jp), shape=(t.m, t.m)).toarray()
    assert np.abs(J_cpp - J_host).max() < 1e-10
    # posJ is the inverse of (src_block, src_k)
    assert np.array_equal(t.posJ[4 * t.src_k + t.src_block], np.arange(t.nnzJ))


# ------------------------------------------------------------------ GPU side
def _gpu():
    """(PolarNewtonSolverCUDA, nr_klu) or skip."""
    pytest.importorskip("pycuda", reason="pycuda not installed")
    nr_klu = pytest.importorskip("p3s.cpp.nr_klu", reason="compiled nr_klu not built")
    try:
        import pycuda.driver as drv

        drv.init()
        if drv.Device.count() == 0:
            pytest.skip("no CUDA device")
        from p3s.cuda.nr_polar_solver import PolarNewtonSolverCUDA
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"CUDA unavailable: {e}")
    return PolarNewtonSolverCUDA, nr_klu


def _gpu_only(solver):
    """Switch the CPU safety net off: what the test sees is what the GPU computed."""
    solver.fallback = "none"
    return solver


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("max_chunk", [None, 32])
def test_timeseries_matches_cpp(case, max_chunk):
    Solver, nr_klu = _gpu()
    Yp, Yj, Yx, pv, pq, Sbus, V0 = _problem(case)
    T = 70  # with max_chunk=32: two full chunks and an odd-sized last one
    S = _scaled_injections(Sbus, T, 0.05)
    ref = nr_klu.Solver(Yp, Yj, Yx, pv, pq).solve_batch(S, V0, max_iter=30, tol=1e-8)

    gpu = _gpu_only(Solver(Yp, Yj, Yx, pv, pq))
    assert gpu.backend == "lean"
    gpu.max_chunk = max_chunk
    r = gpu.solve_batch(S, V0, max_iter=30, tol=1e-8)
    assert r["converged"].all() and ref["converged"].all()
    assert np.array_equal(r["iterations"], ref["iterations"])
    assert np.abs(r["V"] - ref["V"]).max() < 1e-10
    assert gpu.last_stats["chunks"] == (1 if max_chunk is None else 3)
    assert gpu.last_stats["failed_on_gpu"] == 0 and gpu.last_stats["handed_off"] == 0

    # per-column start voltages take the same path as a shared one
    r2 = gpu.solve_batch(S, np.ascontiguousarray(np.repeat(V0[:, None], T, axis=1)), max_iter=30, tol=1e-8)
    assert np.array_equal(r2["V"], r["V"])


@pytest.mark.parametrize("line_search", [True, False])
def test_line_search_matches_cpp_from_bad_starts(line_search):
    """Start voltages with increasingly scrambled angles make full Newton steps overshoot.
    The damped iteration must take nr_klu's decisions: the same columns converge, in the
    same number of iterations -- and damping must matter for the outcome of this sweep."""
    Solver, nr_klu = _gpu()
    Yp, Yj, Yx, pv, pq, Sbus, V0 = _problem("case118")
    T = 48
    rng = np.random.default_rng(5)
    spread = np.linspace(0.0, 0.3, T)
    V = np.ascontiguousarray(V0[:, None] * np.exp(1j * spread[None, :] * rng.standard_normal((len(V0), T))))
    S = np.ascontiguousarray(np.repeat(Sbus[:, None], T, axis=1))
    cpu = nr_klu.Solver(Yp, Yj, Yx, pv, pq)
    ref = cpu.solve_batch(S, V, max_iter=30, tol=1e-8, line_search=line_search)
    other = cpu.solve_batch(S, V, max_iter=30, tol=1e-8, line_search=not line_search)
    assert not np.array_equal(ref["iterations"], other["iterations"]), "sweep does not exercise the line search"

    gpu = _gpu_only(Solver(Yp, Yj, Yx, pv, pq))
    gpu.line_search = line_search
    r = gpu.solve_batch(S, V, max_iter=30, tol=1e-8)
    ok = ref["converged"]
    assert 0 < ok.sum() < T, "the sweep must contain solvable and unsolvable columns"
    assert np.array_equal(r["converged"], ok)
    assert np.array_equal(r["iterations"][ok], ref["iterations"][ok])
    assert np.abs(r["V"][:, ok] - ref["V"][:, ok]).max() < 1e-8


@pytest.mark.parametrize("spread", [0.05, None])
def test_mixed_precision_reaches_the_same_solution(spread, monkeypatch):
    """Single-precision factors for the early iterations must not change what the solve
    converges to -- neither on an easy batch nor on a load sweep up to the loadability
    limit -- nor may a rejected single factorization (it is redone in double at once)."""
    Solver, nr_klu = _gpu()
    Yp, Yj, Yx, pv, pq, Sbus, V0 = _problem("case118")
    T = 64
    if spread is None:
        S = np.ascontiguousarray(Sbus[:, None] * np.linspace(1.0, 4.5, T)[None, :])
    else:
        S = _scaled_injections(Sbus, T, spread)
    ref = nr_klu.Solver(Yp, Yj, Yx, pv, pq).solve_batch(S, V0, max_iter=30, tol=1e-8)
    ok = ref["converged"]

    gpu = _gpu_only(Solver(Yp, Yj, Yx, pv, pq))
    double = gpu.solve_batch(S, V0, max_iter=30, tol=1e-8)
    assert gpu.last_stats["single_factorizations"] == 0

    gpu.precision = "mixed"
    mixed = gpu.solve_batch(S, V0, max_iter=30, tol=1e-8)
    assert gpu.last_stats["single_factorizations"] > 0 and gpu.last_stats["double_factorizations"] > 0
    assert np.array_equal(mixed["converged"], ok)
    assert np.abs(mixed["V"][:, ok] - ref["V"][:, ok]).max() < 1e-8
    assert (mixed["iterations"][ok] <= ref["iterations"][ok] + 1).all()

    # every single-precision factorization rejected -> the double kernel redoes it within
    # the same iteration: exactly the all-double run
    gpu._rcond_ratio_single = 1e30
    rejected = gpu.solve_batch(S, V0, max_iter=30, tol=1e-8)
    assert np.array_equal(rejected["converged"], double["converged"])
    assert np.array_equal(rejected["iterations"], double["iterations"])
    assert np.array_equal(rejected["V"][:, ok], double["V"][:, ok])


def test_non_finite_input_fails_only_its_own_column():
    Solver, _ = _gpu()
    Yp, Yj, Yx, pv, pq, Sbus, V0 = _problem("case14")
    S = _scaled_injections(Sbus, 40, 0.05)
    S[3, 7] = np.nan
    r = _gpu_only(Solver(Yp, Yj, Yx, pv, pq)).solve_batch(S, V0)
    expect = np.ones(40, dtype=bool)
    expect[7] = False
    assert np.array_equal(r["converged"], expect)
    assert np.isfinite(r["V"][:, expect]).all()


def test_rejected_factorization_is_rescued_on_cpu(monkeypatch):
    """A pivot ratio below the acceptance bound marks the system FAILED on the GPU; with
    the fallback on, nr_klu re-solves it -- with it off, it is reported as not converged."""
    Solver, nr_klu = _gpu()
    from p3s.cuda import nr_polar_solver as mod

    Yp, Yj, Yx, pv, pq, Sbus, V0 = _problem("case118")
    S = _scaled_injections(Sbus, 48, 0.05)
    ref = nr_klu.Solver(Yp, Yj, Yx, pv, pq).solve_batch(S, V0)
    gpu = Solver(Yp, Yj, Yx, pv, pq)
    monkeypatch.setattr(mod, "_RCOND_RATIO", 1e30)  # no factorization can pass

    gpu.fallback = "none"
    r = gpu.solve_batch(S, V0)
    assert not r["converged"].any()

    gpu.fallback = "auto"
    r = gpu.solve_batch(S, V0)
    assert r["converged"].all()
    assert np.abs(r["V"] - ref["V"]).max() < 1e-9
    assert gpu.last_stats["failed_on_gpu"] == 48 and gpu.last_stats["fallback"] == "nr_klu"


def test_stragglers_are_handed_to_the_cpu():
    """One slow column in a batch of fast ones is finished by nr_klu from the GPU's iterate."""
    Solver, nr_klu = _gpu()
    Yp, Yj, Yx, pv, pq, Sbus, V0 = _problem("case118")
    T = 64
    S = _scaled_injections(Sbus, T, 0.02)
    S[:, 5] = Sbus * 3.1  # close to the loadability limit: needs many more iterations
    ref = nr_klu.Solver(Yp, Yj, Yx, pv, pq).solve_batch(S, V0, max_iter=40, tol=1e-8)
    assert ref["converged"].all() and ref["iterations"][5] > ref["iterations"][0] + 1

    gpu = Solver(Yp, Yj, Yx, pv, pq)
    r = gpu.solve_batch(S, V0, max_iter=40, tol=1e-8)
    assert gpu.last_stats["handed_off"] == 1 and gpu.last_stats["failed_on_gpu"] == 0
    assert r["converged"].all()
    assert np.abs(r["V"] - ref["V"]).max() < 1e-8

    gpu.cpu_handoff = 0  # everything on the GPU: same answer
    r_gpu = gpu.solve_batch(S, V0, max_iter=40, tol=1e-8)
    assert gpu.last_stats["handed_off"] == 0
    assert np.array_equal(r_gpu["iterations"], ref["iterations"])
    assert np.abs(r_gpu["V"] - r["V"]).max() < 1e-8


def test_update_Y_equals_rebuilding_the_solver():
    Solver, _ = _gpu()
    Yp, Yj, Yx, pv, pq, Sbus, V0 = _problem("case14")
    S = _scaled_injections(Sbus, 16, 0.05)
    Yx2 = Yx * (1.0 + 0.03 * np.random.default_rng(4).standard_normal(Yx.shape[0]))
    gpu = _gpu_only(Solver(Yp, Yj, Yx, pv, pq))
    before = gpu.solve_batch(S, V0)["V"]
    gpu.update_Y(Yx2)
    after = gpu.solve_batch(S, V0)
    fresh = _gpu_only(Solver(Yp, Yj, Yx2, pv, pq)).solve_batch(S, V0)
    assert after["converged"].all()
    assert np.array_equal(after["V"], fresh["V"])
    assert np.abs(after["V"] - before).max() > 1e-6
    with pytest.raises(ValueError):
        gpu.update_Y(Yx2[:-1])


def test_contingency_stamps_equal_dense_ybus():
    """The device-side expansion (base Ybus - sparse stamps) must give exactly what the
    dense per-case Ybus gives, and both must match nr_klu."""
    Solver, nr_klu = _gpu()
    from p3s.contingency.case_generator import ContingencyCaseGenerator
    from p3s.contingency.fixtures import FIXTURES

    net = FIXTURES["parallel_branch"]()
    gen = ContingencyCaseGenerator(net, reslack_islands=False)
    batch = gen.build()
    n, L = batch.n_bus, len(batch.cases)
    v_start = gen._npf._initial_voltage.copy()
    Sbus = np.ascontiguousarray(gen._npf._sBus, dtype=np.complex128)
    V0 = np.empty((n, L), dtype=np.complex128)
    pin = np.zeros((n, L), dtype=np.uint8)
    for c, case in enumerate(batch.cases):
        v0 = v_start.copy()
        for bus, _kind in case.pinned_refs:
            v0[bus] = np.abs(v0[bus]) + 0.0j
            pin[bus, c] = 1
        pin[~case.served, c] = 1
        v0[~case.served] = 1.0 + 0.0j
        V0[:, c] = v0
    Yx = np.ascontiguousarray(batch.Yx_matrix)
    pv = np.ascontiguousarray(batch.pv, dtype=np.int32)
    pq = np.ascontiguousarray(batch.pq, dtype=np.int32)

    gpu = _gpu_only(Solver(batch.Yp, batch.Yj, batch.Yx_base, pv, pq))
    dense = gpu.solve_batch_contingency_cx(Yx, Sbus, V0, pin)
    stamp = gpu.solve_batch_contingency_stamp(batch._stamp, Sbus, V0, pin)
    polar = gpu.solve_batch_contingency(np.abs(Yx), np.angle(Yx), Sbus, V0, pin)
    ref = nr_klu.Solver(batch.Yp, batch.Yj, batch.Yx_base, pv, pq).solve_batch_contingency(
        np.abs(Yx), np.angle(Yx), Sbus, V0, pin
    )
    assert np.array_equal(dense["converged"], ref["converged"]) and dense["converged"].any()
    assert np.array_equal(stamp["V"], dense["V"])
    assert np.array_equal(stamp["iterations"], ref["iterations"])
    ok = ref["converged"]
    assert np.abs(dense["V"][:, ok] - ref["V"][:, ok]).max() < 1e-10
    assert np.abs(polar["V"][:, ok] - ref["V"][:, ok]).max() < 1e-10
