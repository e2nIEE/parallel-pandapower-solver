# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 0 fixtures for N-1 contingency analysis.

Each fixture is a pandapower net carrying an ``outage_group`` column on ``net.line``
and ``net.trafo``. A *contingency* is one distinct (non-null) ``outage_group`` value:
all branches sharing that value are taken out of service together. Branches with a
null ``outage_group`` stay in service and are never tested (confirmed scope).

The fixtures are chosen to exercise every scoping scenario:

  * ``radial_spur``      -- a single line/trafo whose outage islands an end bus
                            (served=True elsewhere, NaN at the stranded bus).
  * ``parallel_branch``  -- two lines on the same bus pair grouped together vs.
                            singly, to verify per-branch *stamp* subtraction
                            (not entry-zeroing) and the parallel-sum case.
  * ``tapped_trafo``     -- an off-nominal-tap / phase-shift transformer outage,
                            where Y[f,t] != conj(Y[t,f]).
  * ``generator_island`` -- an outage that islands a component containing a
                            generator with short-circuit data, for the optional
                            re-slacking path (gen SC columns populated).
  * ``multi_branch_group`` -- one outage_group spanning several branches (a whole
                            "string" of the grid) removed at once.

The standard pandapower ``in_service`` flag is the mechanism for taking a branch
out, so the pandapower ground-truth oracle (``ground_truth.py``) needs no special
handling -- it just flips ``in_service`` and runs ``runpp``.
"""
from __future__ import annotations

import numpy as np
import pandapower as pp

from p3s.calculateTrafoTapTable import calculateTrafoCharacteristic


# Generator short-circuit columns required for the optional island re-slacking
# (lowest-Z generator becomes the island reference). Populated only where a fixture
# needs the re-slacking path; the closed-form Z uses xdss_pu / rdss_ohm / sn_mva.
_GEN_SC_COLS = ("xdss_pu", "rdss_ohm", "vn_kv")


def _add_outage_group_columns(net):
    """Ensure ``net.line`` and ``net.trafo`` have an ``outage_group`` column (object
    dtype, default None). Idempotent."""
    for tbl in ("line", "trafo"):
        if tbl in net and "outage_group" not in net[tbl].columns:
            net[tbl]["outage_group"] = None
    return net


def _set_gen_sc(net, gen_idx, xdss_pu, rdss_ohm=0.0, vn_kv=None):
    """Populate the short-circuit columns on a generator row (for re-slacking)."""
    for col in _GEN_SC_COLS:
        if col not in net.gen.columns:
            net.gen[col] = np.nan
    net.gen.loc[gen_idx, "xdss_pu"] = xdss_pu
    net.gen.loc[gen_idx, "rdss_ohm"] = rdss_ohm
    if vn_kv is None:
        vn_kv = net.bus.vn_kv[net.gen.bus[gen_idx]]
    net.gen.loc[gen_idx, "vn_kv"] = vn_kv


def radial_spur():
    """Meshed core + a single radial line feeding a spur bus.

    The spur line is its own outage_group ``"spur"``. Taking it out islands the spur
    bus: pandapower returns NaN there (served=False) while the case still converges
    (served core). A second group ``"core"`` removes one meshed line that does NOT
    island anything (pure values-only contingency).
    """
    net = pp.create_empty_network(sn_mva=1.0)
    b = [pp.create_bus(net, vn_kv=110.0, name=f"b{i}") for i in range(5)]
    pp.create_ext_grid(net, b[0], vm_pu=1.02, va_degree=0.0)

    # meshed core: 0-1, 1-2, 0-2
    lp = dict(length_km=1.0, r_ohm_per_km=0.08, x_ohm_per_km=0.32,
              c_nf_per_km=12.0, max_i_ka=1.0)
    l01 = pp.create_line_from_parameters(net, b[0], b[1], **lp)
    pp.create_line_from_parameters(net, b[1], b[2], **lp)
    pp.create_line_from_parameters(net, b[0], b[2], **lp)
    # radial spur: 2-3 (single feed), and a deeper spur 3-4
    l_spur = pp.create_line_from_parameters(net, b[2], b[3], **lp)
    pp.create_line_from_parameters(net, b[3], b[4], **lp)

    pp.create_load(net, b[1], p_mw=8.0, q_mvar=2.0)
    pp.create_load(net, b[3], p_mw=5.0, q_mvar=1.0)
    pp.create_load(net, b[4], p_mw=3.0, q_mvar=0.5)

    _add_outage_group_columns(net)
    net.line.loc[l_spur, "outage_group"] = "spur"   # outage islands b3 AND b4
    net.line.loc[l01, "outage_group"] = "core"      # outage islands nothing
    return net


def parallel_branch():
    """Two parallel lines on the same bus pair; remove only ONE of them.

    Group ``"one_of_two"`` removes a single parallel line -> the pair's Ybus
    off-diagonal must KEEP the other line's contribution (the parallel-sum case that
    proves we subtract a per-branch *stamp*, not zero the matrix entry).
    """
    net = pp.create_empty_network(sn_mva=1.0)
    b = [pp.create_bus(net, vn_kv=110.0, name=f"b{i}") for i in range(3)]
    pp.create_ext_grid(net, b[0], vm_pu=1.0, va_degree=0.0)

    lp = dict(length_km=1.0, r_ohm_per_km=0.1, x_ohm_per_km=0.3,
              c_nf_per_km=10.0, max_i_ka=1.0)
    la = pp.create_line_from_parameters(net, b[0], b[1], **lp)
    pp.create_line_from_parameters(net, b[0], b[1], **lp)   # parallel to la
    pp.create_line_from_parameters(net, b[1], b[2], **lp)
    pp.create_line_from_parameters(net, b[0], b[2], **lp)   # alt path keeps b1 fed

    pp.create_load(net, b[1], p_mw=10.0, q_mvar=3.0)
    pp.create_load(net, b[2], p_mw=6.0, q_mvar=1.5)

    _add_outage_group_columns(net)
    net.line.loc[la, "outage_group"] = "one_of_two"
    return net


def parallel_branch_both():
    """Both parallel lines share one group ``"both"`` -> the whole parallel edge is
    removed in a single contingency (the edge is fully gone, but b1 stays fed via the
    0-2-1 path so it does not island)."""
    net = pp.create_empty_network(sn_mva=1.0)
    b = [pp.create_bus(net, vn_kv=110.0, name=f"b{i}") for i in range(3)]
    pp.create_ext_grid(net, b[0], vm_pu=1.0, va_degree=0.0)

    lp = dict(length_km=1.0, r_ohm_per_km=0.1, x_ohm_per_km=0.3,
              c_nf_per_km=10.0, max_i_ka=1.0)
    la = pp.create_line_from_parameters(net, b[0], b[1], **lp)
    lb = pp.create_line_from_parameters(net, b[0], b[1], **lp)
    pp.create_line_from_parameters(net, b[1], b[2], **lp)
    pp.create_line_from_parameters(net, b[0], b[2], **lp)

    pp.create_load(net, b[1], p_mw=10.0, q_mvar=3.0)
    pp.create_load(net, b[2], p_mw=6.0, q_mvar=1.5)

    _add_outage_group_columns(net)
    net.line.loc[[la, lb], "outage_group"] = "both"
    return net


def tapped_trafo():
    """Two-winding transformer with off-nominal tap + phase shift, as a contingency.

    The trafo's outage_group ``"trafo"`` exercises an asymmetric branch stamp
    (Y[f,t] != conj(Y[t,f])). A parallel line keeps the LV bus fed so the outage is
    a values-only change rather than an islanding one.
    """
    net = pp.create_empty_network(sn_mva=1.0)
    bhv = pp.create_bus(net, vn_kv=110.0, name="hv")
    blv = pp.create_bus(net, vn_kv=20.0, name="lv")
    blv2 = pp.create_bus(net, vn_kv=20.0, name="lv2")
    pp.create_ext_grid(net, bhv, vm_pu=1.03, va_degree=0.0)

    t = pp.create_transformer_from_parameters(
        net, hv_bus=bhv, lv_bus=blv, sn_mva=40.0, vn_hv_kv=110.0, vn_lv_kv=20.0,
        vk_percent=12.0, vkr_percent=0.5, pfe_kw=30.0, i0_percent=0.1,
        shift_degree=30.0, tap_side="hv", tap_neutral=0, tap_min=-9, tap_max=9,
        tap_step_percent=1.5, tap_step_degree=0.0, tap_pos=3,
    )
    # second trafo keeps lv fed when the first is out (avoid islanding here)
    pp.create_transformer_from_parameters(
        net, hv_bus=bhv, lv_bus=blv, sn_mva=40.0, vn_hv_kv=110.0, vn_lv_kv=20.0,
        vk_percent=12.0, vkr_percent=0.5, pfe_kw=30.0, i0_percent=0.1,
        shift_degree=30.0, tap_side="hv", tap_neutral=0, tap_min=-9, tap_max=9,
        tap_step_percent=1.5, tap_step_degree=0.0, tap_pos=0,
    )
    pp.create_line_from_parameters(net, blv, blv2, length_km=1.0, r_ohm_per_km=0.1,
                                   x_ohm_per_km=0.3, c_nf_per_km=10.0, max_i_ka=1.0)
    pp.create_load(net, blv, p_mw=12.0, q_mvar=4.0)
    pp.create_load(net, blv2, p_mw=8.0, q_mvar=2.0)

    # p3s's trafo model reads tap data from net.trafo_characteristic_table /
    # net.trafo.id_characteristic_table -- populate them so the fixture is ready to use.
    calculateTrafoCharacteristic(net, inplace=True)

    _add_outage_group_columns(net)
    net.trafo.loc[t, "outage_group"] = "trafo"
    return net


def generator_island():
    """An outage that islands a component containing a generator (re-slacking path).

    The grid is two clusters joined by a single tie line ``tie`` (its own
    outage_group). Cluster B holds a generator with populated short-circuit columns
    but NO ext_grid. Taking the tie out:

      * with re-slacking OFF -> cluster B has no original slack -> served=False (NaN).
      * with re-slacking ON  -> cluster B's generator (lowest Z) becomes the island
                                slack -> served=True, voltages defined.

    The pandapower ground truth for the re-slacking-ON expectation is produced by
    temporarily converting the island generator to an ext_grid (see ground_truth.py).
    """
    net = pp.create_empty_network(sn_mva=1.0)
    # cluster A (slack side)
    a0 = pp.create_bus(net, vn_kv=110.0, name="a0")
    a1 = pp.create_bus(net, vn_kv=110.0, name="a1")
    # cluster B (generator side)
    g0 = pp.create_bus(net, vn_kv=110.0, name="g0")
    g1 = pp.create_bus(net, vn_kv=110.0, name="g1")
    pp.create_ext_grid(net, a0, vm_pu=1.02, va_degree=0.0)

    lp = dict(length_km=1.0, r_ohm_per_km=0.08, x_ohm_per_km=0.32,
              c_nf_per_km=12.0, max_i_ka=1.0)
    pp.create_line_from_parameters(net, a0, a1, **lp)
    tie = pp.create_line_from_parameters(net, a1, g0, **lp)   # the only A<->B link
    pp.create_line_from_parameters(net, g0, g1, **lp)

    pp.create_load(net, a1, p_mw=6.0, q_mvar=1.5)
    pp.create_load(net, g0, p_mw=4.0, q_mvar=1.0)
    pp.create_load(net, g1, p_mw=3.0, q_mvar=0.8)
    gen = pp.create_gen(net, g1, p_mw=7.0, vm_pu=1.01, sn_mva=20.0)
    _set_gen_sc(net, gen, xdss_pu=0.18, rdss_ohm=0.01)

    _add_outage_group_columns(net)
    net.line.loc[tie, "outage_group"] = "tie"
    return net


def multi_branch_group():
    """One outage_group spanning several branches (a whole "string" removed at once).

    A ring 0-1-2-3-0 with a spur chain off bus 3. Group ``"string"`` removes the two
    branches forming the spur chain together, islanding its two end buses in a single
    contingency -- the group-outage-causes-islanding case that is the *normal* path,
    not an edge case.
    """
    net = pp.create_empty_network(sn_mva=1.0)
    b = [pp.create_bus(net, vn_kv=110.0, name=f"b{i}") for i in range(6)]
    pp.create_ext_grid(net, b[0], vm_pu=1.0, va_degree=0.0)

    lp = dict(length_km=1.0, r_ohm_per_km=0.08, x_ohm_per_km=0.32,
              c_nf_per_km=12.0, max_i_ka=1.0)
    pp.create_line_from_parameters(net, b[0], b[1], **lp)
    pp.create_line_from_parameters(net, b[1], b[2], **lp)
    pp.create_line_from_parameters(net, b[2], b[3], **lp)
    pp.create_line_from_parameters(net, b[3], b[0], **lp)   # closes the ring
    s0 = pp.create_line_from_parameters(net, b[3], b[4], **lp)  # spur chain start
    s1 = pp.create_line_from_parameters(net, b[4], b[5], **lp)  # spur chain end

    pp.create_load(net, b[2], p_mw=7.0, q_mvar=2.0)
    pp.create_load(net, b[4], p_mw=4.0, q_mvar=1.0)
    pp.create_load(net, b[5], p_mw=3.0, q_mvar=0.6)

    _add_outage_group_columns(net)
    net.line.loc[[s0, s1], "outage_group"] = "string"   # removes both -> islands b4,b5
    return net


# Registry of all Phase 0 fixtures, keyed by name.
FIXTURES = {
    "radial_spur": radial_spur,
    "parallel_branch": parallel_branch,
    "parallel_branch_both": parallel_branch_both,
    "tapped_trafo": tapped_trafo,
    "generator_island": generator_island,
    "multi_branch_group": multi_branch_group,
}
