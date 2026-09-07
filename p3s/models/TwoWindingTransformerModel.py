# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pandas as pd
from pandas import DataFrame

from p3s.models.TwoPort import TwoPort


class TwoWindingTransformerModel(TwoPort):
    def __init__(
        self,
        trafo_table: DataFrame,
        bus_table: DataFrame,
        tap_table: DataFrame,
        trafo_model: str = "t",
        sn_mva: float = 1.0,
    ):

        super().__init__()
        self._from_bus = trafo_table["hv_bus"].values
        self._to_bus = trafo_table["lv_bus"].values
        n_bus = len(bus_table)
        self.voltages_from = bus_table.loc[self._from_bus, "vn_kv"]
        self.voltages_to = bus_table.loc[self._to_bus, "vn_kv"]

        # Fetch voltage levels
        # hv_kv = bus_table.loc[self._from_bus, 'vn_kv'].values
        # lv_kv = bus_table.loc[self._to_bus, 'vn_kv'].values

        # Input Values
        tap_pos = trafo_table["tap_pos"].fillna(0.0).astype(int).values
        id_characteristic_table = trafo_table["id_characteristic_table"].astype(int).values

        # Resolve the (id_characteristic, step) -> row position once with a single
        # vectorized MultiIndex lookup.
        tap_row = tap_table.index.get_indexer(
            pd.MultiIndex.from_tuples(list(zip(id_characteristic_table, tap_pos, strict=False)))
        )

        # vk_percent = trafo_table["vk_percent"].values
        vk_percent = tap_table["vk_percent"].to_numpy()[tap_row]
        # vkr_percent = trafo_table["vkr_percent"].values
        vkr_percent = tap_table["vkr_percent"].to_numpy()[tap_row]

        trafo_sn_mva = trafo_table["sn_mva"].values
        i0_percent = trafo_table["i0_percent"].values
        pfe_mw = trafo_table["pfe_kw"].values / 1000.0
        # lv_voltage = trafo_table["vn_lv_kv"].values
        # hv_voltage = trafo_table["vn_hv_kv"].values

        # tap changer
        angle_deg = tap_table["angle_deg"].to_numpy()[tap_row]
        voltage_ratio = tap_table["voltage_ratio"].to_numpy()[tap_row]
        shift_degree = trafo_table["shift_degree"].values
        theta = np.deg2rad(angle_deg + shift_degree)

        N = voltage_ratio * np.exp(1j * theta)
        a = np.ones_like(N)
        b = np.ones_like(N)

        if "tap_side" in trafo_table.columns:
            tap_side = trafo_table["tap_side"].values
            a = np.where(tap_side == "hv", N, 1.0)
            b = np.where(tap_side == "lv", N, 1.0)
            # A pure phase shifter (voltage_ratio == 1, shift only) has tap_side NaN/None,
            # so neither branch above fires and the exp(1j*theta) shift would be dropped
            # from the AC stamp entirely (previously it only survived via the DC p_shift
            # term). pandapower always carries the full complex ratio N = ratio*exp(1j*shift)
            # on the from/hv side, so route the shift onto a wherever no magnitude tap side
            # claimed it.
            no_side = ~np.isin(tap_side, ("hv", "lv"))
            a = np.where(no_side, N, a)

        if "tap_side2" in trafo_table.columns:
            tap_side2 = trafo_table["tap_side2"].values
            a = np.where(tap_side2 == "hv", N, a)
            b = np.where(tap_side2 == "lv", N, b)

        # ratio
        # a_ratio = hv_voltage / lv_voltage
        # v_base  = lv_voltage ** 2

        # impedance values
        sn_mva_scaled = sn_mva / trafo_sn_mva
        # sn_mva_scaled2 = sn_mva * sn_mva_scaled

        # z_n = lv_kv ** 2 / sn_mva
        # z_ref = lv_voltage**2 * sn_mva_scaled

        z_sc = vk_percent / 100.0 * sn_mva_scaled
        r_sc = vkr_percent / 100.0 * sn_mva_scaled
        # Preserve the sign of the short-circuit impedance: vk_percent (hence z_sc) can be
        # NEGATIVE for some equivalent transformers (e.g. RTE/Polish grids), which encodes a
        # capacitive/negative reactance. A bare sqrt drops that sign and yields +x, which
        # conjugates the whole trafo admittance and breaks convergence. Match pandapower's
        # _calc_r_x_from_dataframe: x_sc = sign(z_sc) * sqrt(z_sc**2 - r_sc**2).
        x_sc = np.sign(z_sc) * np.sqrt(z_sc**2 - r_sc**2)
        z_k = r_sc + 1j * x_sc  # * sn_mva_scaled2 # * z_n / z_ref

        # magnetising admittance
        i_0 = i0_percent / 100.0 * trafo_sn_mva
        # iron losses are the real part of the admittance
        g_m = pfe_mw / sn_mva

        # the voltage change and the iron losses together form the imaginary part
        # b_m = -np.sqrt(i_0 ** 2 - g_m ** 2) / sn_mva
        # b_m = i_0 * sn_mva / v_base
        # b_m = np.sqrt(np.square(i_0) - np.square(pfe_mw)) / sn_mva
        # b_m[np.isnan(b_m)] = 0

        # when i_0 is not set / or zero, we can just use zero as a value, since the sqrt would be nan
        b_m_squared = np.square(i_0) - np.square(pfe_mw)
        b_m = np.where(b_m_squared < 0, 0, np.sqrt(b_m_squared) / sn_mva)
        y_ = g_m * sn_mva - 1j * b_m  # / sn_mva_scaled # * z_ref / z_n

        if trafo_model == "pi":
            # optimised formula, z1 = y_, z2 = z_, z3 = y_
            z_ = 1 / z_k
            self._Y_ff = z_ + y_ / 2.0  # = 1/z1 + 1/z2
            self._Y_tf = -z_  # = - 1/z2
            self._Y_ft = -z_  # = - 1/z2
            self._Y_tt = z_ + y_ / 2.0  # = 1/z2 + 1/z3

        elif trafo_model == "t":
            # pandapower's "t" model does NOT invert a literal T-network. It converts the
            # T (series leg z_k split 50/50 around the magnetising shunt y_) into an
            # equivalent pi via a Wye->Delta transform, then stamps a STANDARD pi branch
            # (pandapower.build_branch._wye_delta + pypower.makeYbus.branch_vectors).
            #
            # The critical property this restores: the tap ratio scales ONLY the from/to
            # diagonals and the off-diagonals -- it must NEVER scale the series admittance.
            # The previous implementation folded the magnetising branch into K and let the
            # tap leak into the series term, so every OFF-nominal-tap trafo got a series
            # impedance wrong by the tap ratio (correct only when tap == 1). That produced a
            # wrong Ybus on Polish/RTE grids (case3120sp/case1888rte/case6515rte) -> the
            # Newton solved the wrong system and diverged.
            #
            # y_ already carries pandapower's sign convention (y_ = g + 1j*b with b <= 0),
            # so it can be used directly as the magnetising admittance zc = 1/y_.
            has_mag = y_ != 0

            # Wye (T) -> Delta (pi). za_star = hv half-leg, zb_star = lv half-leg,
            # zc_star = magnetising branch. r_ratio = x_ratio = 0.5 (pandapower default
            # leakage_*_ratio_hv) => the series leg is split evenly hv/lv.
            za_star = z_k / 2.0
            zb_star = z_k / 2.0
            # Guard the magnetising branch: where y_ == 0 there is no shunt leg, so use a
            # finite placeholder (1) to keep the arithmetic clean -- those entries are
            # discarded by the np.where(has_mag, ...) selections below.
            y_safe = np.where(has_mag, y_, 1.0)
            zc_star = 1.0 / y_safe

            zSum = za_star * zb_star + za_star * zc_star + zb_star * zc_star
            # Delta branch impedances: ab = series (hv<->lv), ac = hv shunt, bc = lv shunt.
            z_series = np.where(has_mag, zSum / zc_star, z_k)  # -> series admittance
            y_from = np.where(has_mag, zb_star / zSum, 0.0)  # 1/zac_triangle (hv shunt)
            y_to = np.where(has_mag, za_star / zSum, 0.0)  # 1/zbc_triangle (lv shunt)

            Ys = 1.0 / z_series

            # Standard pi stamp with the tap on whichever side carries it (a on hv, b on lv;
            # both == 1 for an untapped side). Mirrors makeYbus:
            #   Yff = (Ysf + Bcf/2) / (tap*conj(tap)); Yft = -Ysf/conj(tap); etc.
            abs2_a = a * np.conj(a)
            abs2_b = b * np.conj(b)

            self._Y_ff = (Ys + y_from) / abs2_a
            self._Y_ft = -Ys / (np.conj(a) * np.conj(b))
            self._Y_tf = -Ys / (a * b)
            self._Y_tt = (Ys + y_to) / abs2_b

        else:
            raise UserWarning(f"Trafo Model: {trafo_model}, not supported. Only pi and t are available.")

        # for DC powerflow.
        # The DC B' matrix uses a DIFFERENT (simpler, symmetric) tap convention than the AC
        # stamp -- it must match pandapower's makeBdc.calc_b_from_branch, which is what
        # rundcpp uses and what the DC seed is validated against:
        #     b = (1/x) / tap                      # signed 1/x, divided by tap MAGNITUDE
        #     Bff = Btt = +b ,  Bft = Btf = -b      # symmetric Laplacian stamp
        # i.e. all four entries share the SAME 1/tap scaling (to the first power) and both
        # diagonals are equal. The earlier version copied the AC ideal-transformer stamp
        # (Yff ~ 1/|a|^2 but Ytt ~ 1, off-diagonals ~ 1/(a*b)) -- an inconsistent, asymmetric
        # tap scaling that corrupts the DC B-matrix for tapped trafos (sorted-diagonal error
        # up to ~90 vs pandapower on RTE grids, biggest where taps are largest), giving a
        # poor DC seed that lands the AC Newton in a collapse basin. The phase shift is NOT
        # in the DC B-matrix; it enters via the separate p_shift injection below (matching
        # makeBdc's Pbusinj). tap magnitude = |a|*|b| (one side is 1, the other the ratio).
        # Sign convention matches TransmissionLineModel (diag -one_over_x, off +one_over_x)
        # so line and trafo diagonals add rather than cancel.
        one_over_x = 1 / (1j * x_sc)
        tap_mag = np.abs(a) * np.abs(b)
        b_dc = one_over_x / tap_mag
        self._DC_Yff = -b_dc
        self._DC_Ytt = -b_dc
        self._DC_Ytf = b_dc
        self._DC_Yft = b_dc

        # Zero out de-energised transformers (see TwoPort._apply_in_service). Must run
        # before p_shift is accumulated below, so an out-of-service phase shifter also
        # stops injecting DC shift power.
        in_service = self._apply_in_service(trafo_table)

        # DC phase-shift power injection. This must match pandapower's makeBdc, which is
        # what the DC seed is validated against and what determines the DC-init ANGLES
        # (a wrong phase-shift injection skews the DC angles at phase-shifter buses by up to
        # ~20 deg -> the AC Newton starts in a voltage-collapse basin and diverges).
        #
        # pandapower: Pfinj = b * (-theta_rad),  b = (1/x)/tap ;  then Pbusinj = Cft.T*Pfinj
        # puts +Pfinj at the from bus and -Pfinj at the to bus, and the DC RHS is
        # P = Sbus.real - Pbusinj. p3s assembles the RHS as Sbus.real + p_shift, so
        #     p_shift = -Pbusinj  ->  +b*theta_rad at from,  -b*theta_rad at to.
        # The previous form (-theta/x_sc at from) had BOTH the wrong SIGN and was missing the
        # /tap factor (b = 1/(x*tap), not 1/x). theta already includes the tap angle + the
        # fixed vector-group shift_degree. tap magnitude = |a|*|b| (one side is 1).
        p_shift_inj = np.where(in_service, theta / (x_sc * np.abs(a) * np.abs(b)), 0.0)
        self.p_shift = np.zeros(n_bus)
        np.add.at(self.p_shift, self._from_bus, p_shift_inj)
        np.add.at(self.p_shift, self._to_bus, -p_shift_inj)
