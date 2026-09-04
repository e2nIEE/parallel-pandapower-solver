# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test for the batched cuSolverRf solver.

Validates :class:`CusolverRfBatch` against ``scipy.sparse.linalg.spsolve`` on the
saved Jacobian fixtures (``case*_Jx.npy`` / ``_Jp`` / ``_Jj`` / ``_rhs`` in the repo
root). Needs only numpy/scipy/pycuda + a CUDA GPU -- no pandapower.
"""

import numpy as np
import pytest
from pandapower import runpp
from pandapower.networks import case9, case14, case118, case9241pegase
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from p3s.cuda.cusolver_rf_batch import CusolverRfBatch

CASES = {"case9": case9(), "case14": case14(), "case118": case118(), "case9241": case9241pegase()}


def _load(case):
    """Build the polar NR Jacobian by running pandapower runpp and reading
    net["_ppc"]["internal"]["J"] (the converged-point Jacobian)."""
    net = CASES[case]
    runpp(net)
    J = net["_ppc"]["internal"]["J"].tocsr()
    Jx = J.data.astype(np.float64)
    Jp = J.indptr.astype(np.int32)
    Jj = J.indices.astype(np.int32)
    n = J.shape[0]
    return Jx, Jp, Jj, n


@pytest.mark.parametrize("case", CASES)
def test_batch_solve_matches_scipy(case):
    Jx, Jp, Jj, n = _load(case)

    B = 8
    rng = np.random.default_rng(0)
    # batch of B systems: same pattern, perturbed values + distinct RHS
    Jx_batch = np.empty((B, len(Jx)), dtype=np.float64)
    rhs_batch = np.empty((B, n), dtype=np.float64)
    x_ref = np.empty((B, n), dtype=np.float64)
    for i in range(B):
        scale = 1.0 + 0.05 * (i - B / 2) / B  # mild perturbation, keeps pattern
        Jx_i = Jx * scale
        b_i = rng.standard_normal(n)
        Jx_batch[i] = Jx_i
        rhs_batch[i] = b_i
        x_ref[i] = spsolve(csr_matrix((Jx_i, Jj, Jp), shape=(n, n)), b_i)

    solver = CusolverRfBatch(Jp, Jj, batch_size=B)
    solver.symbolic_setup(Jx)  # symbolic from the base (unperturbed) matrix
    x_gpu = solver.solve(Jx_batch, rhs_batch)

    for i in range(B):
        err = np.linalg.norm(x_gpu[i] - x_ref[i], np.inf)
        denom = max(1.0, np.linalg.norm(x_ref[i], np.inf))
        assert err / denom < 1e-6, f"{case} system {i}: rel err {err / denom:.3e}"


if __name__ == "__main__":
    import sys

    cases = sys.argv[1:] or CASES
    for c in cases:
        test_batch_solve_matches_scipy(c)
        print(f"{c}: OK")
