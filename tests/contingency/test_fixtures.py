# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for N-1 contingency analysis: fixtures + pandapower ground truth.

These lock in the *reference* behavior that the p3s batch solvers must reproduce.
They assert two things:

  1. Each fixture's ``outage_group`` columns enumerate the intended contingencies.
  2. The pandapower oracle classifies served/unserved buses and convergence exactly
     as the scoping design predicts (incl. the optional island re-slacking path).

No p3s solver is exercised yet -- only pandapower is the oracle here.
"""

import numpy as np
import pytest

from p3s.contingency.fixtures import FIXTURES
from p3s.contingency.ground_truth import (
    enumerate_contingencies,
    generator_z_pu,
    ground_truth,
)

# Expected contingency groups per fixture (the enumeration contract).
EXPECTED_GROUPS = {
    "radial_spur": ["core", "spur"],
    "parallel_branch": ["one_of_two"],
    "parallel_branch_both": ["both"],
    "tapped_trafo": ["trafo"],
    "generator_island": ["tie"],
    "multi_branch_group": ["string"],
}

# Expected served-bus count per (fixture, group) with re-slacking OFF.
EXPECTED_SERVED_OFF = {
    ("radial_spur", "core"): 5,
    ("radial_spur", "spur"): 3,  # b3, b4 islanded
    ("parallel_branch", "one_of_two"): 3,
    ("parallel_branch_both", "both"): 3,
    ("tapped_trafo", "trafo"): 3,
    ("generator_island", "tie"): 2,  # cluster B (g0,g1) unserved
    ("multi_branch_group", "string"): 4,  # b4, b5 islanded
}


@pytest.mark.parametrize("name", list(FIXTURES))
def test_fixture_builds_and_base_case_converges(name):
    import pandapower as pp

    net = FIXTURES[name]()
    # the intact net (no outage) must solve -- a sanity check on the fixture itself
    pp.runpp(net, init="flat")
    assert bool(net["converged"])
    assert np.isfinite(net.res_bus.vm_pu).all()


@pytest.mark.parametrize("name", list(FIXTURES))
def test_enumerate_contingencies(name):
    net = FIXTURES[name]()
    assert enumerate_contingencies(net) == EXPECTED_GROUPS[name]


@pytest.mark.parametrize("name", list(FIXTURES))
def test_ground_truth_served_and_converged_reslack_off(name):
    net = FIXTURES[name]()
    gt = ground_truth(net, reslack_islands=False)
    assert set(gt) == set(EXPECTED_GROUPS[name])
    for group, res in gt.items():
        assert res.converged, f"{name}/{group} should converge (served component)"
        n_served = int(res.served.sum())
        assert n_served == EXPECTED_SERVED_OFF[(name, group)], (
            f"{name}/{group}: served {n_served}, expected {EXPECTED_SERVED_OFF[(name, group)]}"
        )
        # served <=> vm is finite; unserved <=> NaN
        assert np.array_equal(res.served, np.isfinite(res.vm))
        assert np.isnan(res.vm[~res.served]).all()


def test_radial_spur_islands_exactly_the_spur_buses():
    net = FIXTURES["radial_spur"]()
    gt = ground_truth(net, reslack_islands=False)
    spur = gt["spur"]
    # buses 0,1,2 served; 3,4 (the spur chain) unserved
    assert spur.served.tolist() == [True, True, True, False, False]


def test_generator_island_reslacking_restores_island():
    """The optional re-slacking path: with it ON, the generatorless-slack island is
    served by promoting the lowest-Z generator; with it OFF those buses are NaN."""
    net = FIXTURES["generator_island"]()
    off = ground_truth(net, reslack_islands=False)["tie"]
    on = ground_truth(net, reslack_islands=True)["tie"]

    assert off.served.tolist() == [True, True, False, False]
    assert on.served.all(), "re-slacking should serve the whole island"
    assert on.converged
    # the promoted generator's bus keeps its vm setpoint (gen vm_pu = 1.01 at bus g1)
    assert on.vm[3] == pytest.approx(1.01, abs=1e-6)


def test_generator_z_pu_computed_from_sc_columns():
    net = FIXTURES["generator_island"]()
    z = generator_z_pu(net)
    # single gen, xdss_pu=0.18, rdss negligible -> Z ~ 0.18
    assert len(z) == 1
    assert z.iloc[0] == pytest.approx(0.18, abs=1e-3)


def test_ungrouped_branches_are_never_taken_out():
    """A branch with a null outage_group must not appear in any contingency."""
    net = FIXTURES["radial_spur"]()
    # most lines are ungrouped; only 'core' and 'spur' are contingencies
    grouped = net.line["outage_group"].dropna().unique().tolist()
    assert set(grouped) == {"core", "spur"}
    assert set(enumerate_contingencies(net)) == {"core", "spur"}
