# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reactive-power capability curves for ``gen`` / ``sgen`` elements.

A capability curve makes a machine's reactive limits depend on its active power:
instead of a fixed ``[min_q_mvar, max_q_mvar]`` pair, the limits are read off a
piecewise curve at the element's current ``p_mw``.

This module is a pure, vectorized evaluator. It reads pandapower's RAW curve table
(``net.q_capability_curve_table``) and never touches ``net``: no characteristic objects
are created, nothing is cached on the net, and no element table is mutated.

Why not reuse pandapower's ``net.q_capability_characteristic``
--------------------------------------------------------------
pandapower stores one pickled ``Characteristic`` callable per curve per limit in a
DataFrame cell, built by ``create_q_capability_characteristics_object`` (which lives in
the *control* package) and evaluated with ``np.vectorize(lambda f, p: f(p))`` -- a Python
loop over objects. The curve is a pure function of ``(p, points)``, so this indirection
buys nothing at solve time; here the points are packed into flat arrays instead.

Curve styles
------------
``straightLineYValues``
    Linear interpolation between neighboring points. Matches pandapower.
``constantYValue``
    Zero-order hold: the limit keeps the value of the point at or below ``p`` until the
    next breakpoint is reached.

    NOTE -- this DELIBERATELY DIFFERS from pandapower, which applies linear interpolation
    to both styles: its ``Characteristic.__call__`` is unconditionally ``numpy.interp``,
    so ``curve_style`` is stored and validated but never reaches the interpolation. A
    "constantYValue" curve therefore behaves identically to a straight-line one there.
    p3s implements the documented (and CIM ReactiveCapabilityCurve) semantics
    instead, so results for such curves differ from pandapower BY DESIGN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# pandapower's two documented curve styles.
STYLE_LINEAR = "straightLineYValues"
STYLE_CONSTANT = "constantYValue"
VALID_STYLES = (STYLE_LINEAR, STYLE_CONSTANT)

CURVE_TABLE = "q_capability_curve_table"
CURVE_ID_COL = "id_q_capability_curve"
ELEMENT_ID_COL = "id_q_capability_characteristic"


@dataclass(frozen=True)
class QCapabilityCurves:
    """Packed, ragged set of capability curves.

    All curves are concatenated into flat ``xs``/``ys_min``/``ys_max`` arrays; curve ``k``
    occupies the slice ``[starts[k], starts[k] + counts[k])``. ``curve_pos`` maps a
    pandapower ``id_q_capability_curve`` onto its position ``k``.
    """

    curve_pos: dict  # id_q_capability_curve -> position in starts/counts
    starts: np.ndarray  # (n_curves,) int
    counts: np.ndarray  # (n_curves,) int
    xs: np.ndarray  # (n_points,) p_mw, ascending within each curve
    ys_min: np.ndarray  # (n_points,) q_min_mvar
    ys_max: np.ndarray  # (n_points,) q_max_mvar

    @classmethod
    def from_net(cls, net) -> QCapabilityCurves | None:
        """Build from ``net.q_capability_curve_table``, or None if there is none.

        The table is optional in pandapower (it is not part of the default net), so a net
        without it simply has no curves and every caller falls back to the fixed
        ``min_q_mvar`` / ``max_q_mvar`` columns.
        """
        if CURVE_TABLE not in net.keys():
            return None
        table = net[CURVE_TABLE]
        if table is None or len(table) == 0:
            return None

        required = {CURVE_ID_COL, "p_mw", "q_min_mvar", "q_max_mvar"}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{CURVE_TABLE} is missing column(s): {sorted(missing)}")

        starts, counts, xs, ys_min, ys_max, curve_pos = [], [], [], [], [], {}  # type: ignore[var-annotated]
        offset = 0
        for curve_id, group in table.groupby(CURVE_ID_COL, sort=True):
            p = np.asarray(group["p_mw"].values, dtype=float)
            q_lo = np.asarray(group["q_min_mvar"].values, dtype=float)
            q_hi = np.asarray(group["q_max_mvar"].values, dtype=float)

            if not np.isfinite(p).all():
                raise ValueError(f"curve {curve_id}: p_mw contains NaN/inf")
            # searchsorted needs ascending x; pandapower does not guarantee row order.
            order = np.argsort(p, kind="stable")
            p, q_lo, q_hi = p[order], q_lo[order], q_hi[order]

            if len(p) < 2:
                raise ValueError(f"curve {curve_id}: needs at least 2 points, got {len(p)}")
            if np.any(np.diff(p) <= 0):
                raise ValueError(f"curve {curve_id}: p_mw values must be strictly increasing")

            curve_pos[curve_id] = len(starts)
            starts.append(offset)
            counts.append(len(p))
            xs.append(p)
            ys_min.append(q_lo)
            ys_max.append(q_hi)
            offset += len(p)

        return cls(
            curve_pos=curve_pos,
            starts=np.asarray(starts, dtype=np.intp),
            counts=np.asarray(counts, dtype=np.intp),
            xs=np.concatenate(xs),
            ys_min=np.concatenate(ys_min),
            ys_max=np.concatenate(ys_max),
        )

    def evaluate(self, curve_ids, p_mw, styles):
        """Evaluate ``(q_min, q_max)`` for a batch of elements.

        Parameters
        ----------
        curve_ids : array-like
            ``id_q_capability_characteristic`` per element; NaN / unknown ids yield NaN
            limits so the caller can fall back to the fixed columns.
        p_mw : array-like
            Active power per element, at which the curve is read.
        styles : array-like of str
            ``curve_style`` per element. Anything other than the two documented styles
            raises -- silently guessing an interpolation would hide bad data.

        Returns
        -------
        (q_min, q_max) : two float arrays, NaN where no curve applies.
        """
        curve_ids = np.asarray(curve_ids, dtype=object)
        p_mw = np.asarray(p_mw, dtype=float)
        styles = np.asarray(styles, dtype=object)
        n = len(curve_ids)

        q_min = np.full(n, np.nan)
        q_max = np.full(n, np.nan)
        if n == 0:
            return q_min, q_max

        # Resolve each element's curve to a packed position; -1 = no curve.
        pos = np.full(n, -1, dtype=np.intp)
        for i, cid in enumerate(curve_ids):
            if cid is None or (isinstance(cid, float) and np.isnan(cid)):
                continue
            try:
                key = int(cid)
            except (TypeError, ValueError):
                continue
            if key in self.curve_pos:
                pos[i] = self.curve_pos[key]
            else:
                logger.warning(
                    "id_q_capability_characteristic %s has no matching curve in %s; "
                    "falling back to the fixed reactive limits",
                    key,
                    CURVE_TABLE,
                )

        active = np.flatnonzero(pos >= 0)
        if len(active) == 0:
            return q_min, q_max

        bad = {s for s in styles[active] if s not in VALID_STYLES}
        if bad:
            raise ValueError(f"unsupported curve_style {sorted(map(str, bad))}; expected one of {list(VALID_STYLES)}")

        # Group the active elements by curve style so each style is applied with one
        # vectorized call per curve rather than per element.
        for k in np.unique(pos[active]):
            lo, cnt = int(self.starts[k]), int(self.counts[k])
            xs = self.xs[lo : lo + cnt]
            ys_lo = self.ys_min[lo : lo + cnt]
            ys_hi = self.ys_max[lo : lo + cnt]

            members = active[pos[active] == k]
            p_sel = p_mw[members]

            # Clamp outside the curve's P range: the curve describes the machine's whole
            # envelope, so the endpoint limits are the defensible reading (and this is what
            # np.interp/pandapower do). Warn, because it usually means a malformed curve or
            # a dispatch beyond the machine's rating.
            out_of_range = (p_sel < xs[0]) | (p_sel > xs[-1])
            if out_of_range.any():
                logger.warning(
                    "%d element(s) have p_mw outside the capability curve range [%.6g, %.6g]; "
                    "clamping to the endpoint reactive limits",
                    int(out_of_range.sum()),
                    xs[0],
                    xs[-1],
                )

            is_linear = np.array([styles[i] == STYLE_LINEAR for i in members], dtype=bool)

            if is_linear.any():
                sel = members[is_linear]
                # np.interp clamps to the endpoints outside [xs[0], xs[-1]].
                q_min[sel] = np.interp(p_mw[sel], xs, ys_lo)
                q_max[sel] = np.interp(p_mw[sel], xs, ys_hi)

            if (~is_linear).any():
                sel = members[~is_linear]
                # Zero-order hold: take the point at or below p (and the first point for p
                # below the curve). side="right" gives the index AFTER the match, so -1.
                j = np.searchsorted(xs, p_mw[sel], side="right") - 1
                j = np.clip(j, 0, cnt - 1)
                q_min[sel] = ys_lo[j]
                q_max[sel] = ys_hi[j]

        return q_min, q_max


def resolve_q_limits(net, element: str):
    """Reactive limits for every row of ``net[element]``, curve-aware.

    Returns ``(q_min, q_max)`` in MVAr, row-aligned with ``net[element]``, or None if the
    element table is absent or empty. Elements with a capability curve get their limits
    from the curve at the element's ``p_mw``; every other element falls back to the fixed
    ``min_q_mvar`` / ``max_q_mvar`` columns (NaN where those are absent, meaning
    "unlimited").

    ``p_mw`` is read WITHOUT scaling, matching pandapower's
    ``_calculate_qmin_qmax_from_q_capability_characteristics``, which evaluates the curve
    at the raw ``tab["p_mw"]``.
    """
    if element not in net.keys() or len(net[element]) == 0:
        return None

    table = net[element]
    n = len(table)

    q_min = np.full(n, np.nan)
    q_max = np.full(n, np.nan)
    if "min_q_mvar" in table.columns:
        q_min = np.asarray(table["min_q_mvar"].values, dtype=float)
    if "max_q_mvar" in table.columns:
        q_max = np.asarray(table["max_q_mvar"].values, dtype=float)

    curves = QCapabilityCurves.from_net(net)
    if curves is None or ELEMENT_ID_COL not in table.columns:
        return q_min, q_max

    # Only elements explicitly flagged for curve use are evaluated. pandapower derives
    # this flag from the id + style columns; respect an existing column but fall back to
    # "has an id" so a net that never ran the control-module helper still works.
    if "reactive_capability_curve" in table.columns:
        use_curve = table["reactive_capability_curve"].fillna(False).to_numpy(dtype=bool)
    else:
        use_curve = table[ELEMENT_ID_COL].notna().to_numpy(dtype=bool)
    if not use_curve.any():
        return q_min, q_max

    if "curve_style" in table.columns:
        styles = table["curve_style"]
    else:
        styles = pd.Series([STYLE_LINEAR] * n, index=table.index)
    styles = styles.where(styles.notna(), STYLE_LINEAR).to_numpy(dtype=object)

    ids = table[ELEMENT_ID_COL].to_numpy(dtype=object)
    ids = np.where(use_curve, ids, None)

    curve_min, curve_max = curves.evaluate(ids, table["p_mw"].to_numpy(dtype=float), styles)

    # Curve values win where they resolved; the fixed columns remain the fallback.
    q_min = np.where(np.isnan(curve_min), q_min, curve_min)
    q_max = np.where(np.isnan(curve_max), q_max, curve_max)
    return q_min, q_max
