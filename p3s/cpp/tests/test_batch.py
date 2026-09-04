# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate Solver.solve_batch against scipy and check thread-invariance.

Loads the saved Jacobian/Ybus fixtures (`case*.npz`), builds a batch of operating
points by perturbing Sbus per column, and asserts:
  1. each column's converged voltage matches a per-column scipy Newton solve, and
  2. results are identical for n_threads=1 vs n_threads=N (columns are independent).

Run standalone:  python p3s/cpp/test_batch.py
"""

import os
import sys

import numpy as np
import scipy.sparse as sp

try:
    from p3s import nr_klu  # type: ignore[attr-defined]
except ImportError:
    from p3s.cpp import nr_klu  # type: ignore[attr-defined]

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

CASES = ["case9", "case14", "case118", "case9241"]


def _scipy_newton(Yp, Yj, Yx, Sbus, V0, pv, pq, tol=1e-8, max_it=30):
    """Reference polar Newton (mirrors make_reference.run_nr)."""
    n = len(Yp) - 1
    Ybus = sp.csr_matrix((Yx, Yj, Yp), shape=(n, n))
    pvpq = np.r_[pv, pq]
    npv, npq = len(pv), len(pq)
    V = V0.copy()

    def mismatch(V):
        mis = V * np.conj(Ybus @ V) - Sbus
        return np.r_[mis[pvpq].real, mis[pq].imag]

    def dSbus_dV(V):
        Ibus = Ybus @ V
        diagV = sp.diags(V)
        diagI = sp.diags(Ibus)
        diagVn = sp.diags(V / np.abs(V))
        dS_dVm = diagV @ np.conj(Ybus @ diagVn) + np.conj(diagI) @ diagVn
        dS_dVa = 1j * diagV @ np.conj(diagI - Ybus @ diagV)
        return dS_dVm, dS_dVa

    F = mismatch(V)
    it = 0
    while np.linalg.norm(F, np.inf) >= tol and it < max_it:
        it += 1
        dS_dVm, dS_dVa = dSbus_dV(V)
        J11 = dS_dVa[np.ix_(pvpq, pvpq)].real
        J12 = dS_dVm[np.ix_(pvpq, pq)].real
        J21 = dS_dVa[np.ix_(pq, pvpq)].imag
        J22 = dS_dVm[np.ix_(pq, pq)].imag
        J = sp.vstack([sp.hstack([J11, J12]), sp.hstack([J21, J22])], format="csc")
        dx = sp.linalg.spsolve(J, -F)
        Va = np.angle(V)
        Vm = np.abs(V)
        Va[pvpq] += dx[: npv + npq]
        Vm[pq] += dx[npv + npq :]
        V = Vm * np.exp(1j * Va)
        F = mismatch(V)
    return V


def _check_case(case, T=16, seed=0):
    d = np.load(os.path.join(_HERE, f"{case}.npz"))
    Yp, Yj, Yx = d["Yp"], d["Yj"], d["Yx"]
    Sbus, V0, pv, pq = d["Sbus"], d["V0"], d["pv"], d["pq"]
    n = len(Yp) - 1

    rng = np.random.default_rng(seed)
    # per-column Sbus perturbation (mild, keeps every column convergent)
    Sb_mat = np.empty((n, T), dtype=np.complex128)
    V_ref = np.empty((n, T), dtype=np.complex128)
    for t in range(T):
        scale = 1.0 + 0.05 * rng.standard_normal(n)
        Sb_t = Sbus * scale
        Sb_mat[:, t] = Sb_t
        V_ref[:, t] = _scipy_newton(Yp, Yj, Yx, Sb_t, V0, pv, pq)

    s = nr_klu.Solver(Yp, Yj, Yx, pv, pq)

    r1 = s.solve_batch(Sb_mat, V0, tol=1e-8, n_threads=1)
    rN = s.solve_batch(Sb_mat, V0, tol=1e-8, n_threads=0)  # all cores

    # 1) correctness vs scipy (per column)
    err = np.max(np.abs(r1["V"] - V_ref))
    assert err < 1e-7, f"{case}: batch vs scipy max err {err:.2e}"
    assert bool(np.all(r1["converged"])), f"{case}: not all columns converged"

    # 2) thread invariance (identical, columns independent)
    dthread = np.max(np.abs(r1["V"] - rN["V"]))
    assert dthread == 0.0, f"{case}: n_threads=1 vs N differ by {dthread:.2e}"
    assert np.array_equal(r1["iterations"], rN["iterations"])

    print(
        f"[{case}] T={T} n={n}: batch-vs-scipy={err:.2e}  thread-invariance={dthread:.0e}  "
        f"iters[min..max]={r1['iterations'].min()}..{r1['iterations'].max()}"
    )


if __name__ == "__main__":
    cases = sys.argv[1:] or CASES
    for c in cases:
        _check_case(c)
    print("BATCH VALIDATION: PASS")
