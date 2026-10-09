# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Three-winding transformer as a three-port.

Equivalent circuit (positive sequence, per unit of net.sn_mva on the transformer's rated voltages):
three winding branches z_hv, z_mv, z_lv meet at an internal star node. The star node is a property
of the equivalent circuit (delta -> star of the three pair short-circuit impedances), not the
neutral of a Y winding; the vector group only enters as the phase shifts shift_mv/lv_degree.

  1. pair impedances z_hm, z_ml, z_hl from vk/vkr_hv/mv/lv_percent, each per unit of the smaller
     rating of its two windings (vk_hv: hv-mv, vk_mv: mv-lv, vk_lv: hv-lv)
  2. star branches   z_hv = (z_hm + z_hl - z_ml) / 2, z_mv = (z_hm + z_ml - z_hl) / 2,
                     z_lv = (z_hl + z_ml - z_hm) / 2
  3. every branch k is a pi: series y_k, shunt y_a,k at its terminal end, shunt y_b,k at its star
     end (only the loss-side branch has shunts), plus a shunt y_0 at the star node (loss_side "star")
  4. ideal transformers: V_bus,k = t_k * V'_k at the terminal and V_b,k = s_k * V_star at the star
     end of branch k, t_k = vn_k,rated / vn_k,bus * exp(-j*shift_k) * N_k (tap at the terminal),
     s_k = N_k (tap_at_star_point), N_k = voltage_ratio * exp(j*angle_deg) of the tap side's row in
     net.trafo_characteristic_table (1 on the other windings)
  5. star node eliminated (rank-1 update):
         Y'  = diag(y + y_a) - (y*s) (y*conj(s))^T / S,   S = y_0 + sum_k |s_k|^2 (y_k + y_b,k)
         Y_ij = Y'_ij / (conj(t_i) * t_j)

This reproduces pandapower's trafo3w (three equivalent 2W trafos around an auxiliary star bus),
including its magnetising position: the T-model on the loss-side branch (loss_side "hv"/"mv"/"lv",
default "hv" as pandapower's trafo3w_losses) or a shunt at the star node ("star").
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame

from p3s.models.ThreePort import ThreePort

SIDES = ("hv", "mv", "lv")
LOSS_SIDES = ("hv", "mv", "lv", "star")


def _pair_impedance(vk_percent, vkr_percent, sn_a, sn_b, sn_mva) -> NDArray:
    """Short-circuit impedance between two windings, per unit of sn_mva (rated voltages)."""
    base = sn_mva / np.minimum(sn_a, sn_b)
    r = vkr_percent / 100.0 * base
    x = np.sqrt(np.square(vk_percent / 100.0 * base) - np.square(r))
    return r + 1j * x


class ThreeWindingTransformerModel(ThreePort):
    def __init__(
        self,
        trafo3w_table: DataFrame,
        bus_table: DataFrame,
        tap_table: DataFrame,
        trafo3w_model: str = "t",
        sn_mva: float = 1.0,
        loss_side: str = "hv",  # pandapower's default for trafo3w_losses
    ):
        """
        Args:
            trafo3w_table: net.trafo3w
            bus_table: net.bus
            tap_table: net.trafo_characteristic_table (MultiIndex id_characteristic / step)
            trafo3w_model: only "t" (the pandapower model)
            sn_mva: net.sn_mva
            loss_side: where the magnetising branch sits for rows without a ``loss_side`` value:
                "hv", "mv", "lv" (middle of that winding's branch) or "star" (star node)
        """
        super().__init__()
        if trafo3w_model != "t":
            raise UserWarning(f"Trafo3w Model: {trafo3w_model}, not supported. Only t is available.")
        if loss_side not in LOSS_SIDES:
            raise ValueError(f"loss_side must be one of {LOSS_SIDES}, got {loss_side!r}")

        t3 = trafo3w_table
        n = len(t3)
        self._hv_bus = t3["hv_bus"].to_numpy()
        self._mv_bus = t3["mv_bus"].to_numpy()
        self._lv_bus = t3["lv_bus"].to_numpy()

        # per element and side (columns hv, mv, lv)
        self.sn = np.stack([t3[f"sn_{s}_mva"].to_numpy(dtype=float) for s in SIDES], axis=1)
        self.vn_rated = np.stack([t3[f"vn_{s}_kv"].to_numpy(dtype=float) for s in SIDES], axis=1)
        self.vn_bus = np.stack([bus_table.loc[t3[f"{s}_bus"], "vn_kv"].to_numpy(dtype=float) for s in SIDES], axis=1)

        # -- tap state from the characteristic table -----------------------------------------
        tap_pos: NDArray = t3["tap_pos"].fillna(0.0).to_numpy(dtype=float).astype(int)
        ids = t3["id_characteristic_table"].to_numpy(dtype=int)
        tap_row = tap_table.index.get_indexer(pd.MultiIndex.from_arrays([ids, tap_pos]))
        if (tap_row < 0).any():
            missing = list(zip(ids[tap_row < 0], tap_pos[tap_row < 0], strict=True))
            raise KeyError(
                f"trafo_characteristic_table has no row for trafo3w (id_characteristic, step) {missing}; "
                "run calculate_trafo_characteristic(net, inplace=True)"
            )

        def _row(column: str) -> NDArray:
            return tap_table[column].to_numpy(dtype=float)[tap_row]

        # -- 1./2. pair impedances and star branches ---------------------------------------
        z_hm = _pair_impedance(_row("vk_hv_percent"), _row("vkr_hv_percent"), self.sn[:, 0], self.sn[:, 1], sn_mva)
        z_ml = _pair_impedance(_row("vk_mv_percent"), _row("vkr_mv_percent"), self.sn[:, 1], self.sn[:, 2], sn_mva)
        z_hl = _pair_impedance(_row("vk_lv_percent"), _row("vkr_lv_percent"), self.sn[:, 0], self.sn[:, 2], sn_mva)
        z = 0.5 * np.stack([z_hm + z_hl - z_ml, z_hm + z_ml - z_hl, z_hl + z_ml - z_hm], axis=1)
        if np.any(z == 0):
            raise UserWarning("Equivalent star branch of a trafo3w with zero impedance!")

        # -- 3. magnetising branch ------------------------------------------------------------
        if "loss_side" in t3:
            sides = t3["loss_side"].fillna(loss_side).astype(str).str.lower().to_numpy()
        else:
            sides = np.full(n, loss_side)
        if not np.isin(sides, LOSS_SIDES).all():
            raise ValueError(f"trafo3w.loss_side must be one of {LOSS_SIDES}")

        # i0_percent refers to the rating of the loss side (sn_hv for the star node)
        sn_loss = np.where(sides == "mv", self.sn[:, 1], np.where(sides == "lv", self.sn[:, 2], self.sn[:, 0]))
        pfe_mw = t3["pfe_kw"].to_numpy(dtype=float) / 1000.0
        i0_mva = t3["i0_percent"].to_numpy(dtype=float) / 100.0 * sn_loss
        y_mag = (pfe_mw - 1j * np.sqrt(np.maximum(np.square(i0_mva) - np.square(pfe_mw), 0.0))) / sn_mva

        y_a: NDArray = np.zeros((n, 3), dtype=complex)  # shunt at the terminal end of a branch
        y_b: NDArray = np.zeros((n, 3), dtype=complex)  # shunt at the star end of a branch
        y_0 = np.where(sides == "star", y_mag, 0.0)  # shunt at the star node

        for k, side in enumerate(SIDES):
            # T-model (z/2 - y_mag - z/2) of the loss-side branch as an equivalent pi (wye -> delta)
            mask = (sides == side) & (y_mag != 0)
            if not mask.any():
                continue
            half = z[mask, k] / 2.0
            z_mag = 1.0 / y_mag[mask]
            z_sum = half * half + 2.0 * half * z_mag
            z[mask, k] = z_sum / z_mag
            y_a[mask, k] = half / z_sum
            y_b[mask, k] = half / z_sum
        y = 1.0 / z

        # -- 4. ideal transformers -------------------------------------------------------------
        shift_mv = t3["shift_mv_degree"].fillna(0.0).to_numpy(dtype=float)
        shift_lv = t3["shift_lv_degree"].fillna(0.0).to_numpy(dtype=float)
        shift = np.deg2rad(np.stack([np.zeros(n), shift_mv, shift_lv], axis=1))
        t = self.vn_rated / self.vn_bus * np.exp(-1j * shift)
        s: NDArray = np.ones((n, 3), dtype=complex)
        ratio = np.nan_to_num(_row("voltage_ratio"), nan=1.0) * np.exp(
            1j * np.deg2rad(np.nan_to_num(_row("angle_deg")))
        )
        tap_side = t3["tap_side"].to_numpy(dtype=object) if "tap_side" in t3 else np.full(n, None)
        if "tap_at_star_point" in t3:
            at_star = t3["tap_at_star_point"].fillna(False).to_numpy(dtype=bool)
        else:
            at_star = np.zeros(n, dtype=bool)
        for k, side in enumerate(SIDES):
            on_side = tap_side == side
            t[:, k] = np.where(on_side & ~at_star, t[:, k] * ratio, t[:, k])
            s[:, k] = np.where(on_side & at_star, ratio, s[:, k])

        # -- 5. star node eliminated, referred to the buses ---------------------------------------
        u = y * s
        w = y * np.conj(s)
        star_sum = y_0 + np.sum(np.square(np.abs(s)) * (y + y_b), axis=1)
        eye = np.eye(3)
        y_int = np.einsum("ei,ij->eij", y + y_a, eye) - np.einsum("ei,ej->eij", u, w) / star_sum[:, None, None]
        y_bus = y_int / (np.conj(t)[:, :, None] * t[:, None, :])
        for i in range(3):
            for j in range(3):
                setattr(self, f"_Y_{i + 1}{j + 1}", y_bus[:, i, j])

        # kept for the star voltage in the results
        self._t, self._w, self._star_sum = t, w, star_sum

        # -- DC: pandapower's makeBdc on the three equivalent 2W branches, star eliminated -------
        # Each branch k carries b_k = 1 / (x_k,bus * |tap_k|) and the phase shift delta_k
        # (terminal -> star). With pandapower's 2W tap placement and its auxiliary star bus (per
        # unit of the hv BUS voltage) that is b_k = |s_k| / (x_k * |t_k| * c), c = vn_hv,rated /
        # vn_hv,bus: a tap at the star point sits on the other side of pandapower's 2W branch, so
        # it enters the DC susceptance with |N| instead of 1/|N|.
        x = np.imag(1.0 / y)
        c = self.vn_rated[:, [0]] / self.vn_bus[:, [0]]
        b = np.abs(s) / (x * np.abs(t) * c)
        delta = np.angle(t) + np.angle(s)
        b_sum = b.sum(axis=1)
        b_red = np.einsum("ei,ij->eij", b, eye) - np.einsum("ei,ej->eij", b, b) / b_sum[:, None, None]
        for i in range(3):
            for j in range(3):
                setattr(self, f"_DC_Y_{i + 1}{j + 1}", 1j * b_red[:, i, j])

        self.in_service = self._apply_in_service(t3)

        # kept to rebuild the star voltage and the DC injection when a winding is switched off
        self._y, self._y_a, self._y_b, self._s, self._y_0 = y, y_a, y_b, s, y_0
        self._dc_b, self._dc_delta = b, delta
        # Iron losses at the star node (loss_side "star") are a DC load at pandapower's auxiliary
        # bus (a shunt conductance counts as load at 1 pu there).
        self._p_star = -np.real(y_0) / c[:, 0] ** 2
        self.p_shift: NDArray = np.zeros(len(bus_table))
        self._update_dc_injection(np.ones((n, 3), dtype=bool))

    def _update_dc_injection(self, connected: NDArray) -> None:
        """DC phase-shift injection (TwoWindingTransformerModel convention: RHS = Sbus.real +
        p_shift) of the connected windings, star node eliminated; the star-node load spreads over
        the terminals in proportion to b_k."""
        b = np.where(connected, self._dc_b, 0.0)
        b_sum = b.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(b_sum[:, None] != 0, b / b_sum[:, None], 0.0)
        p_shift_inj = b * self._dc_delta - share * np.sum(b * self._dc_delta, axis=1)[:, None]
        p_shift_inj += share * self._p_star[:, None]
        p_shift_inj[~self.in_service] = 0.0
        self.p_shift[:] = 0.0
        for k, buses in enumerate(self._buses()):
            np.add.at(self.p_shift, buses, p_shift_inj[:, k])

    def _after_open_ends(self, open_ends: NDArray) -> None:
        """A winding disconnected by an open switch: no DC flow through it, and for the star voltage
        it is just a shunt hanging off the star node (its branch ends in its terminal shunt)."""
        connected = ~open_ends
        y, y_a, y_b, s = self._y, self._y_a, self._y_b, self._s
        with np.errstate(divide="ignore", invalid="ignore"):
            dangling = np.where(y + y_a != 0, y * y_a / (y + y_a), 0.0)
        abs_s2 = np.square(np.abs(s))
        self._w = np.where(connected, self._w, 0.0)
        self._star_sum = self._y_0 + np.sum(abs_s2 * (y_b + np.where(connected, y, dangling)), axis=1)
        self._update_dc_injection(connected)

    def results(self, voltage: NDArray, sn_mva: float) -> dict[str, NDArray]:
        """res_trafo3w columns for the solved bus voltages (NaN on unsupplied buses), as pandapower
        defines them."""
        reported = voltage[np.stack(self._buses(), axis=1)]
        voltage = np.nan_to_num(voltage)
        current = self.port_currents(voltage)  # (n, 3), into the transformer
        buses = np.stack(self._buses(), axis=1)
        v_bus = voltage[buses]
        vm = np.abs(reported)
        s_mva = v_bus * np.conj(current) * sn_mva
        i_ka = np.abs(s_mva) / (np.sqrt(3) * np.where(np.isnan(vm), 1.0, vm) * self.vn_bus)

        # star voltage, per unit of the hv bus voltage (pandapower's auxiliary bus)
        v_star = np.sum(self._w * v_bus / self._t, axis=1) / self._star_sum * (self.vn_rated[:, 0] / self.vn_bus[:, 0])
        energised = self.in_service & ~np.isnan(reported).all(axis=1)

        res = {}
        for k, side in enumerate(SIDES):
            res[f"p_{side}_mw"] = s_mva[:, k].real
            res[f"q_{side}_mvar"] = s_mva[:, k].imag
        res["pl_mw"] = s_mva.real.sum(axis=1)
        res["ql_mvar"] = s_mva.imag.sum(axis=1)
        for k, side in enumerate(SIDES):
            res[f"i_{side}_ka"] = i_ka[:, k]
        for k, side in enumerate(SIDES):
            res[f"vm_{side}_pu"] = vm[:, k]
            res[f"va_{side}_degree"] = np.angle(reported[:, k], deg=True)
        res["va_internal_degree"] = np.where(energised, np.angle(v_star, deg=True), np.nan)
        res["vm_internal_pu"] = np.where(energised, np.abs(v_star), np.nan)
        loading = np.max(i_ka * self.vn_rated * np.sqrt(3) / self.sn * 100.0, axis=1)  # trafo_loading="current"
        res["loading_percent"] = np.where(energised, loading, 0.0)
        return res
