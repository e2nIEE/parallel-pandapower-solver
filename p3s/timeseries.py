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
import os
import sys

import numba
import numpy as np
from numba import njit, prange
from numba.typed import List
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
    n_bus = pf._sBus.shape[0]
    # Every (element, variable) series contributes coef * (value - base) to the bus the
    # element is connected to (loads -1/sn, sgen/gen +1/sn; q to the imaginary part).
    # Collect all contributions as one bus-major list and let a parallel kernel write each
    # output row once: the (n_bus, T) result is the only large array this touches.
    src: List = []
    sources, bus, row, coef, base, imag = [], [], [], [], [], []
    for element in net_elements:
        lookup = net[element]["_lookup"].to_numpy().astype(np.int64)  # node, -1 = unsupplied bus
        el = np.nonzero(lookup >= 0)[0]
        sign = -1.0 if element == "load" else 1.0
        for variable, is_imag in (("p_mw", False), ("q_mvar", True)):
            if (element, variable) not in timeseries:
                continue
            if variable in net[element]:
                base_v = net[element][variable].to_numpy().astype(np.float64)
            else:
                base_v = np.zeros(lookup.size)
            sources.append(np.ascontiguousarray(np.asarray(timeseries[(element, variable)])[:, :T], dtype=np.float64))
            bus.append(lookup[el])
            src.append(np.full(el.size, len(sources) - 1, dtype=np.int64))
            row.append(el)
            coef.append(np.full(el.size, sign / sn))
            base.append(base_v[el])
            imag.append(np.full(el.size, is_imag))

    bus_a = np.concatenate(bus)
    order = np.argsort(bus_a, kind="stable")  # bus-major, element order kept within a bus
    ptr = np.zeros(n_bus + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(bus_a, minlength=n_bus))
    sbus_matrix = np.empty((n_bus, T), dtype=np.complex128)
    srcs = List()
    for x in sources:
        srcs.append(x)
    _set_sbus_threads()
    _fill_sbus(
        sbus_matrix,
        np.ascontiguousarray(pf._sBus, dtype=np.complex128),
        ptr,
        np.concatenate(src)[order],
        np.concatenate(row)[order],
        np.concatenate(coef)[order],
        np.concatenate(base)[order],
        np.concatenate(imag)[order],
        srcs,
    )
    return sbus_matrix


@njit(parallel=True, cache=True)
def _fill_sbus(out, s0, ptr, src, row, coef, base, imag, sources):  # pragma: no cover - numba
    """``out[i, t] = s0[i] + sum_k coef[k] * (sources[src[k]][row[k], t] - base[k])`` (times j
    for imaginary contributions) over the contributions ``k`` of bus ``i``.

    One parallel task per bus: the row is initialized and then updated by each of its
    contributions while it is cache-resident, and distinct buses never write the same row.
    """
    n, T = out.shape
    for i in prange(n):
        v = s0[i]
        for t in range(T):
            out[i, t] = v
        for k in range(ptr[i], ptr[i + 1]):
            x = sources[src[k]][row[k]]
            c, b = coef[k], base[k]
            if imag[k]:
                for t in range(T):
                    out[i, t] += 1j * (c * (x[t] - b))
            else:
                for t in range(T):
                    out[i, t] += c * (x[t] - b)


def usable_cpus() -> int:
    """CPUs this process may run on.

    Respects a SLURM / taskset CPU mask where the OS reports it (Linux); ``os.cpu_count()``
    alone counts the whole node and would oversubscribe the allocated cores.
    """
    affinity = getattr(os, "sched_getaffinity", None)  # not available on Windows / macOS
    if affinity is not None:
        return len(affinity(0))
    return os.cpu_count() or 1


def _set_sbus_threads():
    """Cap numba's thread count (sized from ``os.cpu_count()``) at the usable CPUs."""
    numba.set_num_threads(max(1, min(usable_cpus(), numba.config.NUMBA_NUM_THREADS)))


def mean_setpoint_vm(net, default: float = 1.0) -> float:
    """Mean of the in-service generator and ext_grid voltage set-points.

    This is the magnitude pandapower seeds PQ buses with under ``init="auto"``
    (see ``pandapower/auxiliary.py``, the ``init_vm_pu`` branch). Seeding PQ buses at a
    flat 1.0 pu instead, starts the iteration below the network's actual operating level on
    grids whose generators sit above 1.0, and Newton spends its first steps just lifting
    the whole voltage profile.

    Returns ``default`` for a net with no in-service generators or ext_grids (nothing to
    average), which keeps the flat 1.0 pu behavior for such nets.
    """
    parts = []
    for table, col in (("ext_grid", "vm_pu"), ("gen", "vm_pu")):
        if table in net and len(net[table]) > 0:
            df = net[table]
            vm = df[col].to_numpy(dtype=float)
            if "in_service" in df:
                vm = vm[df["in_service"].to_numpy(dtype=bool)]
            parts.append(vm)
    if not parts:
        return default
    vm_all = np.concatenate(parts)
    # Drop non-finite set-points rather than poisoning the mean with NaN.
    vm_all = vm_all[np.isfinite(vm_all)]
    if vm_all.size == 0:
        return default
    return float(vm_all.mean())


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
    pq = pf.busses["pq"]
    pv_vm = np.abs(voltage[pv]) if len(pv) > 0 else None
    pq_vm = np.abs(voltage[pq]) if len(pq) > 0 else None
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
    # ...and the same for PQ: the DC solve yields ANGLES only and returns unit-magnitude
    # phasors, which would silently reset the PQ seed magnitude (the mean set-point, see
    # mean_setpoint_vm) back to 1.0 pu. Restore it, keeping the DC-estimated angle.
    if len(pq) > 0:
        voltage[pq] = pq_vm * np.exp(1j * np.angle(voltage[pq]))

    # Sanity fallback: reject a non-physical DC start (p3s's DC model at scale).
    max_angle_deg = np.abs(np.degrees(np.angle(voltage))).max()
    if not np.isfinite(max_angle_deg):  # or max_angle_deg > _DC_ANGLE_SANITY_DEG:
        return flat
    return voltage
