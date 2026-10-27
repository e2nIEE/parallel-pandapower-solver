# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for an installed p3s-cpp wheel (cibuildwheel ``test-command``).

Needs only numpy and the installed ``nr_klu`` extension -- not the p3s sources, pandapower
or test fixtures -- so it checks exactly what the wheel ships: that the module imports
(i.e. the bundled KLU / OpenMP libraries are found), and that single and batch solves
converge on a small hand-built grid.

Set P3S_EXPECT_OPENMP=1 to also fail when the extension was built without OpenMP (which
CMake otherwise skips silently, leaving the batch solver single-threaded).

Run:  python p3s/cpp/ci/smoke_test.py   (from a directory that does not shadow p3s)
"""

import os
import sys

import numpy as np

from p3s import nr_klu  # type: ignore[attr-defined]


def _ybus() -> np.ndarray:
    """4-bus meshed grid: lines 0-1, 0-2, 1-2, 2-3 plus a shunt at bus 3."""
    n = 4
    Y: np.ndarray = np.zeros((n, n), dtype=complex)
    for i, j, z, b in [
        (0, 1, 0.01 + 0.08j, 0.02),
        (0, 2, 0.02 + 0.10j, 0.03),
        (1, 2, 0.015 + 0.09j, 0.02),
        (2, 3, 0.01 + 0.05j, 0.01),
    ]:
        y = 1 / z
        Y[i, i] += y + 0.5j * b
        Y[j, j] += y + 0.5j * b
        Y[i, j] -= y
        Y[j, i] -= y
    Y[3, 3] += 0.05j
    return Y


def _csr(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Yp: list[int] = [0]
    Yj: list[int] = []
    Yx: list[complex] = []
    for row in Y:
        nz = np.flatnonzero(row)
        Yj.extend(nz.tolist())
        Yx.extend(row[nz].tolist())
        Yp.append(len(Yj))
    return np.array(Yp, dtype=np.int32), np.array(Yj, dtype=np.int32), np.array(Yx, dtype=complex)


def _mismatch(Y: np.ndarray, V: np.ndarray, Sbus: np.ndarray, pv: np.ndarray, pq: np.ndarray) -> float:
    mis = V * np.conj(Y @ V) - Sbus
    return float(np.max(np.abs(np.r_[mis[np.r_[pv, pq]].real, mis[pq].imag])))


def main() -> None:
    print(f"nr_klu: {nr_klu.__file__}")
    print(f"nr_klu.openmp = {getattr(nr_klu, 'openmp', 'n/a')}")
    if os.environ.get("P3S_EXPECT_OPENMP") == "1" and not getattr(nr_klu, "openmp", False):
        sys.exit("nr_klu was built without OpenMP, but P3S_EXPECT_OPENMP=1")

    Y = _ybus()
    Yp, Yj, Yx = _csr(Y)
    pv = np.array([1], dtype=np.int32)  # bus 0 is the slack
    pq = np.array([2, 3], dtype=np.int32)
    Sbus = np.array([0.0, 0.4 + 0.0j, -0.6 - 0.2j, -0.3 - 0.1j])
    V0 = np.array([1.02, 1.01, 1.0, 1.0], dtype=complex)

    r = nr_klu.solve_single(Yp, Yj, Yx, Sbus, V0, pv, pq, max_iter=30, tol=1e-10)
    assert r["converged"], r
    err = _mismatch(Y, r["V"], Sbus, pv, pq)
    print(f"solve_single: {r['iterations']} iterations, |F|inf = {err:.2e}")
    assert err < 1e-9, err

    solver = nr_klu.Solver(Yp, Yj, Yx, pv, pq)
    rs = solver.solve(Sbus, V0, max_iter=30, tol=1e-10)
    assert rs["converged"]
    np.testing.assert_allclose(rs["V"], r["V"], atol=1e-10)

    # Batch: scale the loads per column; serial and threaded results must be identical.
    scale = np.linspace(0.5, 1.3, 16)
    S_batch = np.outer(Sbus, scale)
    S_batch[1, :] = Sbus[1]  # keep the PV generator fixed
    rb1 = solver.solve_batch(S_batch, V0, max_iter=30, tol=1e-10, n_threads=1)
    rb4 = solver.solve_batch(S_batch, V0, max_iter=30, tol=1e-10, n_threads=4)
    assert np.all(rb1["converged"]) and np.all(rb4["converged"])
    np.testing.assert_array_equal(rb1["V"], rb4["V"])
    worst = max(_mismatch(Y, rb4["V"][:, k], S_batch[:, k], pv, pq) for k in range(len(scale)))
    print(f"solve_batch: {len(scale)} cases, worst |F|inf = {worst:.2e}")
    assert worst < 1e-9, worst

    print("p3s-cpp smoke test passed")


if __name__ == "__main__":
    main()
