# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared time-series helpers for the batched Newton-Raphson solvers.

Both the GPU (`p3s.cuda.NewtonPowerflowCuda`) and C++/KLU
(`p3s.NewtonPowerflowCpp`) time-series drivers need the same two operations:

  * turn a pandapower time-series dict into a per-bus, per-step complex injection
    matrix ``Sbus`` of shape ``(n_bus, T)``, and
  * produce a DC-power-flow-initialised start voltage (constant across time steps).

These were originally private methods on the CUDA solver; they are lifted here so both
drivers share one validated implementation. The functions are duck-typed on a
``NewtonPowerflow``-like object (any object exposing ``_sBus``, ``_Bbus``, ``_p_shift``,
``busses``, ``_initial_voltage`` and ``_pre_dc_solve``); both solver classes qualify.
"""

import logging
import sys

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# elements whose time-series we support, with the load/generation sign convention.
_TS_ELEMENTS = ("load", "sgen", "gen")


def build_sbus_matrix(pf, net, timeseries) -> NDArray:
    """Per-bus, per-timestep complex injection matrix ``Sbus`` of shape ``(n_bus, T)``.

    Starts from the static base injection ``pf._sBus`` (which already contains
    generators, ext_grid set-points and the *base* loads) and applies, per
    element/variable present in ``timeseries``, the **delta** between the time-series
    value and that element's base value. Elements absent from the time-series (and the
    generator active power / PV set-points) are preserved.

    Sign/scaling mirror ``_setup_pf``: loads inject ``-P/sn_mva``, sgen/gen inject
    ``+P/sn_mva`` (``_sBus`` carries an overall ``-1`` factor, so loads end up negative
    and generation positive). The time-series carries ``p_mw`` always and ``q_mvar``
    sometimes; generator voltage set-points are fixed across steps and never appear here.
    """
    net_elements = [x for x in _TS_ELEMENTS if x in net and len(net[x]) > 0]

    T = sys.maxsize
    for element in net_elements:
        for variable in ("p_mw", "q_mvar"):
            if (element, variable) in timeseries:
                T = min(T, timeseries[(element, variable)].shape[1])
    if T == sys.maxsize:
        raise ValueError("no time-series data found for load/sgen/gen p_mw/q_mvar")
    logger.info("Found %d time_steps for calculation.", T)

    sn = net.sn_mva
    sbus_matrix = np.repeat(pf._sBus[:, None], T, axis=1)  # (n_bus, T) complex

    for element in net_elements:
        sign = 1.0 if element == "load" else -1.0
        lookup = net[element]["_lookup"].to_numpy().astype(int)
        base_p = net[element].p_mw.to_numpy()
        base_q = net[element].q_mvar.to_numpy() if "q_mvar" in net[element] else np.zeros_like(base_p)

        if (element, "p_mw") in timeseries:
            dP = timeseries[(element, "p_mw")] - base_p[:, None]  # (n_el, T)
            np.add.at(sbus_matrix, lookup, (-sign * dP / sn).astype(np.complex128))
        if (element, "q_mvar") in timeseries:
            dQ = timeseries[(element, "q_mvar")] - base_q[:, None]
            np.add.at(sbus_matrix, lookup, (-sign * (1j * dQ) / sn).astype(np.complex128))

    return sbus_matrix


# If the DC solve yields angles beyond this magnitude (deg), treat it as unreliable and
# fall back to a flat start. Real operating points sit well under this (case118 ~35 deg,
# a phase-shifter fixture ~29 deg); p3s's simplified DC model goes non-physical
# (~+-180 deg) on very large meshed nets like case9241pegase, where the resulting start
# falls outside Newton's basin and even fails the first Jacobian factorization. A flat
# start converges there in ~6 iterations. See `p3s-dc-init-bug`.
_DC_ANGLE_SANITY_DEG = 120.0


def dc_initial_voltage(pf) -> NDArray:
    """DC-power-flow-initialised start voltage (one vector, reused for all steps).

    Mirrors the CPU ``calculate(init='dc')`` path. The DC solve seeds bus angles so
    Newton converges on nets that a flat start cannot handle (notably phase-shifting
    transformers). PV magnitudes are restored after the DC solve (which only yields
    angles), keeping the DC-estimated angle.

    Robustness: p3s's DC model is inaccurate on very large meshed nets (e.g.
    case9241pegase), producing non-physical near-+-180 deg angles whose start breaks
    Newton. When the DC angles exceed ``_DC_ANGLE_SANITY_DEG`` we fall back to a flat
    start (the setpoint-carrying ``_initial_voltage``), which converges there.
    """
    flat = pf._initial_voltage.copy()
    voltage = flat.copy()
    pvpq = np.r_[pf.busses["pv"], pf.busses["pq"]]
    pv = pf.busses["pv"]
    pv_vm = np.abs(voltage[pv]) if len(pv) > 0 else None
    # DC init solves B*theta = P_inj with P_inj = real bus power injection
    # (pf._sBus.real) + transformer phase-shift injection (pf._p_shift). pf._Bbus is
    # already the real susceptance matrix, so it is passed directly (NOT pf._Bbus.imag,
    # which double-.imag'd to zeros -> singular -> flat start). See NewtonPowerflowCpp
    # for the full bug note.
    voltage[pvpq] = pf._pre_dc_solve(
        yBus=pf._Bbus,
        voltage=voltage,
        Pinj=pf._sBus.real + pf._p_shift,
        ref=pf.busses["ref"],
        pvpq=pvpq,
    )
    if len(pv) > 0:
        voltage[pv] = pv_vm * np.exp(1j * np.angle(voltage[pv]))

    # Sanity fallback: reject a non-physical DC start (p3s's DC model at scale).
    max_angle_deg = np.abs(np.degrees(np.angle(voltage))).max()
    if not np.isfinite(max_angle_deg): # or max_angle_deg > _DC_ANGLE_SANITY_DEG:
        return flat
    return voltage
