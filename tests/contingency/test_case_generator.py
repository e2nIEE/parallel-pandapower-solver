# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the backend-agnostic N-1 case generator.

These validate the inputs the batch solvers will consume, with NO power
flow solve and NO compiled backend (the case generator is pure NumPy/SciPy):

  1. Stamp subtraction is EXACT: the per-contingency Ybus values (shared pattern, group
     stamps removed) equal the Ybus p3s builds when the group's branches are
     physically dropped -- for asymmetric trafos, parallel branches, and multi-branch
     groups alike.
  2. The shared pattern (Yp/Yj) is identical across all cases (the property the batch
     solvers rely on to amortize the symbolic factorization).
  3. The served mask + reference resolution match the Phase 0 pandapower ground truth,
     both with re-slacking off and on.
"""

import copy

import numpy as np
import pytest
import scipy.sparse as sp

from p3s.contingency.case_generator import (
    REF_RESLACK_GEN,
    ContingencyCaseGenerator,
    generate_cases,
)
from p3s.contingency.fixtures import FIXTURES
from p3s.contingency.ground_truth import ground_truth
from p3s.NewtonPowerflow import NewtonPowerflow


def _rebuilt_ybus_without_group(net, group):
    """Ground-truth Ybus with the group's branches physically removed from the tables
    (p3s's make_ybus does not filter in_service, so we drop the rows)."""
    ref_net = copy.deepcopy(net)
    for tbl in ("line", "trafo"):
        if tbl in ref_net and len(ref_net[tbl]) and "outage_group" in ref_net[tbl].columns:
            drop = ref_net[tbl].index[ref_net[tbl]["outage_group"] == group]
            ref_net[tbl].drop(index=drop, inplace=True)
    Y = NewtonPowerflow(ref_net)._YBus.tocsr()
    Y.sort_indices()
    return Y


@pytest.mark.parametrize("name", list(FIXTURES))
def test_stamp_subtraction_matches_rebuilt_ybus(name):
    net = FIXTURES[name]()
    batch = generate_cases(net, reslack_islands=False)
    n = batch.n_bus
    for case in batch.cases:
        case_M = sp.csr_matrix((case.Yx, batch.Yj, batch.Yp), shape=(n, n))
        ref_Y = _rebuilt_ybus_without_group(net, case.group)
        max_err = np.abs((case_M - ref_Y).toarray()).max()
        assert max_err < 1e-9, f"{name}/{case.group}: Ybus stamp error {max_err:.2e}"


@pytest.mark.parametrize("name", list(FIXTURES))
def test_shared_pattern_is_constant(name):
    """Every case shares the same Yp/Yj as the base."""
    net = FIXTURES[name]()
    batch = generate_cases(net, reslack_islands=False)
    # Yp/Yj are the base pattern; cases only carry values (Yx). Confirm shapes line up
    # and the base pattern round-trips to a valid CSR for every case's values.
    for case in batch.cases:
        assert case.Yx.shape == batch.Yx_base.shape
        M = sp.csr_matrix((case.Yx, batch.Yj, batch.Yp), shape=(batch.n_bus,) * 2)
        assert M.nnz == len(batch.Yj)


def test_parallel_branch_keeps_other_line_contribution():
    """Removing ONE of two parallel lines must leave the other's stamp in the
    off-diagonal (proves stamp subtraction, not entry zeroing)."""
    net = FIXTURES["parallel_branch"]()
    batch = generate_cases(net, reslack_islands=False)
    case = batch.cases[0]  # "one_of_two"
    M = sp.csr_matrix((case.Yx, batch.Yj, batch.Yp), shape=(batch.n_bus,) * 2)
    # buses 0 and 1 are the parallel pair; off-diagonal must be non-zero (one line left)
    assert abs(M[0, 1]) > 1e-6
    assert abs(M[1, 0]) > 1e-6
    # and it must equal the single-line ground truth
    ref = _rebuilt_ybus_without_group(net, "one_of_two")
    assert abs(M[0, 1] - ref[0, 1]) < 1e-9


def test_tapped_trafo_stamp_is_asymmetric():
    """The trafo stamp must be asymmetric (Y[f,t] != Y[t,f]); subtracting it correctly
    is what the rebuilt-Ybus match in the parametrized test already proves. Here we
    assert the asymmetry is actually present so the test is meaningful."""
    net = FIXTURES["tapped_trafo"]()
    gen = ContingencyCaseGenerator(net, reslack_islands=False)
    base = sp.csr_matrix((gen.Yx_base, gen.Yj, gen.Yp), shape=(gen.n_bus,) * 2)
    # hv bus 0, lv bus 1 carry the trafo; off-nominal tap+shift -> asymmetric
    assert abs(base[0, 1] - base[1, 0]) > 1e-6


@pytest.mark.parametrize("name", list(FIXTURES))
def test_served_mask_matches_ground_truth_reslack_off(name):
    net = FIXTURES[name]()
    batch = generate_cases(net, reslack_islands=False)
    gt = ground_truth(net, reslack_islands=False)
    for case in batch.cases:
        assert np.array_equal(case.served, gt[case.group].served), (
            f"{name}/{case.group}: served mask mismatch vs ground truth"
        )


def test_generator_island_reslacking_matches_ground_truth():
    net = FIXTURES["generator_island"]()

    off = generate_cases(net, reslack_islands=False).cases[0]
    gt_off = ground_truth(net, reslack_islands=False)["tie"]
    assert np.array_equal(off.served, gt_off.served)
    assert off.pinned_refs == []  # nothing promoted

    on = generate_cases(net, reslack_islands=True).cases[0]
    gt_on = ground_truth(net, reslack_islands=True)["tie"]
    assert np.array_equal(on.served, gt_on.served)
    assert on.served.all()
    # the lowest-Z generator (only gen, at bus g1=index 3) is the pinned island ref
    assert on.pinned_refs == [(3, REF_RESLACK_GEN)]
    # cluster A buses reference the original slack (bus 0); cluster B references bus 3
    assert on.ref_bus.tolist() == [0, 0, 3, 3]


def test_lowest_z_generator_is_chosen_among_several():
    """When an island has multiple generators, the lowest-Z one is the reference."""
    net = FIXTURES["generator_island"]()
    # add a second, weaker (higher Z) generator in cluster B (bus g0 = index 2)
    import pandapower as pp

    g2 = pp.create_gen(net, 2, p_mw=2.0, vm_pu=1.0, sn_mva=10.0)
    net.gen.loc[g2, "xdss_pu"] = 0.40  # higher Z than the existing 0.18 gen
    net.gen.loc[g2, "rdss_ohm"] = 0.01
    net.gen.loc[g2, "vn_kv"] = net.bus.vn_kv[2]

    on = generate_cases(net, reslack_islands=True).cases[0]
    # the stiffer gen (xdss 0.18 at bus 3) must win, not the new one at bus 2
    assert on.pinned_refs == [(3, REF_RESLACK_GEN)]


def test_Yx_matrix_columns_match_cases():
    net = FIXTURES["radial_spur"]()
    batch = generate_cases(net, reslack_islands=False)
    M = batch.Yx_matrix
    assert M.shape == (len(batch.Yj), len(batch.cases))
    for c, case in enumerate(batch.cases):
        assert np.array_equal(M[:, c], case.Yx)


@pytest.mark.parametrize("name", list(FIXTURES))
@pytest.mark.parametrize("reslack", [False, True])
def test_vectorized_build_matches_per_case(name, reslack):
    """The fast vectorized ``build()`` must produce cases bit-identical to the
    clarity-first per-case ``build_case()`` same Yx, served mask, and ref_bus. This
    guards the build() vectorization (stamp matrix, bridge-skip connectivity, vectorized
    reference resolution) against silent divergence from the reference path."""
    gen = ContingencyCaseGenerator(FIXTURES[name](), reslack_islands=reslack)
    batch = gen.build()
    for case in batch.cases:
        ref = gen.build_case(case.group)
        assert np.allclose(case.Yx, ref.Yx, atol=1e-12), f"{name}/{case.group}: Yx"
        assert np.array_equal(case.served, ref.served), f"{name}/{case.group}: served"
        assert np.array_equal(case.ref_bus, ref.ref_bus), f"{name}/{case.group}: ref_bus"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
