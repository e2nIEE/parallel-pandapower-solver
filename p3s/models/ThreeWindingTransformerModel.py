# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from p3s.models.ThreePort import ThreePort


class ThreeWindingTransformerModel(ThreePort):
    def __init__(
        self,
        trafo3w_table: DataFrame,
        bus_table: DataFrame,
        tap_table: DataFrame,
        trafo3w_model: str = "t",
        sn_mva: float = 1.0,
    ):

        super().__init__()
        self._hv_bus = trafo3w_table["hv_bus"].values
        self._mv_bus = trafo3w_table["mv_bus"].values
        self._lv_bus = trafo3w_table["lv_bus"].values

        self.voltages_hv = bus_table.loc[self._hv_bus, "vn_kv"]
        self.voltages_mv = bus_table.loc[self._mv_bus, "vn_kv"]
        self.voltages_lv = bus_table.loc[self._lv_bus, "vn_kv"]

        # Input Values
        tap_pos = trafo3w_table["tap_pos"].fillna(0.0).astype(int).values
        id_characteristic_table = trafo3w_table["id_characteristic_table"].astype(int).values

        i0_percent = trafo3w_table["i0_percent"].values
        pfe_mw = trafo3w_table["pfe_kw"].values / 1000.0

        n = len(id_characteristic_table)

        angle_deg = np.array(
            [tap_table.loc[(id_characteristic_table[i], tap_pos[i]), "angle_deg"] for i in range(n)], dtype=float
        )
        voltage_ratio = np.array(
            [tap_table.loc[(id_characteristic_table[i], tap_pos[i]), "voltage_ratio"] for i in range(n)], dtype=float
        )

        vk_hv = np.array(
            [tap_table.loc[(id_characteristic_table[i], tap_pos[i]), "vk_hv_percent"] for i in range(n)], dtype=float
        )
        vkr_hv = np.array(
            [tap_table.loc[(id_characteristic_table[i], tap_pos[i]), "vkr_hv_percent"] for i in range(n)], dtype=float
        )

        vk_mv = np.array(
            [tap_table.loc[(id_characteristic_table[i], tap_pos[i]), "vk_mv_percent"] for i in range(n)], dtype=float
        )
        vkr_mv = np.array(
            [tap_table.loc[(id_characteristic_table[i], tap_pos[i]), "vkr_mv_percent"] for i in range(n)], dtype=float
        )

        vk_lv = np.array(
            [tap_table.loc[(id_characteristic_table[i], tap_pos[i]), "vk_lv_percent"] for i in range(n)], dtype=float
        )
        vkr_lv = np.array(
            [tap_table.loc[(id_characteristic_table[i], tap_pos[i]), "vkr_lv_percent"] for i in range(n)], dtype=float
        )

        shift_mv: NDArray = trafo3w_table["shift_mv_degree"].fillna(0.0).values
        shift_lv: NDArray = trafo3w_table["shift_lv_degree"].fillna(0.0).values
        print(angle_deg.shape)
        if "tap_side" in trafo3w_table.columns:
            tap_side = trafo3w_table["tap_side"].fillna("hv").values
        else:
            tap_side = np.array(["hv"] * len(trafo3w_table), dtype=object)

        theta_hv: NDArray = np.zeros_like(angle_deg, dtype=float)  # hv-Referenz
        theta_mv: NDArray = shift_mv.astype(float)  # Grad
        theta_lv: NDArray = shift_lv.astype(float)  # Grad

        mag_hv: NDArray = np.ones_like(voltage_ratio, dtype=float)
        mag_mv: NDArray = np.ones_like(voltage_ratio, dtype=float)
        mag_lv: NDArray = np.ones_like(voltage_ratio, dtype=float)

        theta_hv = np.where(tap_side == "hv", theta_hv + angle_deg, theta_hv)
        mag_hv = np.where(tap_side == "hv", voltage_ratio, mag_hv)

        theta_mv = np.where(tap_side == "mv", theta_mv + angle_deg, theta_mv)
        mag_mv = np.where(tap_side == "mv", voltage_ratio, mag_mv)

        theta_lv = np.where(tap_side == "lv", theta_lv + angle_deg, theta_lv)
        mag_lv = np.where(tap_side == "lv", voltage_ratio, mag_lv)

        # Komplexe Übersetzungsfaktoren je Wicklung
        a = mag_hv * np.exp(1j * np.deg2rad(theta_hv))  # HV-Zweig
        b = mag_mv * np.exp(1j * np.deg2rad(theta_mv))  # MV-Zweig
        c = mag_lv * np.exp(1j * np.deg2rad(theta_lv))  # LV-Zweig
        abs2_a = a * np.conj(a)
        abs2_b = b * np.conj(b)
        abs2_c = c * np.conj(c)

        sn_hv = trafo3w_table["sn_hv_mva"].values
        sn_mv = trafo3w_table["sn_mv_mva"].values
        sn_lv = trafo3w_table["sn_lv_mva"].values

        z_hv = vk_hv / 100 * (sn_mva / sn_hv)
        r_hv = vkr_hv / 100 * (sn_mva / sn_hv)
        x_hv = np.sqrt(z_hv**2 - r_hv**2)
        z12 = r_hv + 1j * x_hv  # hv-mv

        z_mv = vk_mv / 100 * (sn_mva / sn_mv)
        r_mv = vkr_mv / 100 * (sn_mva / sn_mv)
        x_mv = np.sqrt(z_mv**2 - r_mv**2)
        z23 = r_mv + 1j * x_mv  # mv-lv

        z_lv = vk_lv / 100 * (sn_mva / sn_lv)
        r_lv = vkr_lv / 100 * (sn_mva / sn_lv)
        x_lv = np.sqrt(z_lv**2 - r_lv**2)
        z13 = r_lv + 1j * x_lv  # hv-lv

        # magnetising admittance
        i_0 = i0_percent / 100.0 * sn_mva
        # iron losses are the real part of the admittance
        g_m = pfe_mw / sn_mva

        # when i_0 is not set / or zero, we can just use zero as a value, since the sqrt would be nan
        b_m_squared = np.square(i_0) - np.square(pfe_mw)
        b_m = np.where(b_m_squared < 0, 0, np.sqrt(b_m_squared) / sn_mva)
        y_ = g_m - 1j * b_m

        z1 = 0.5 * (z12 + z13 - z23)
        z2 = 0.5 * (z12 + z23 - z13)
        z3 = 0.5 * (z13 + z23 - z12)
        zm = 1.0 / y_

        if trafo3w_model == "t":
            # if np.any(mask_y_): # case transformer with no losses
            # TODO: no losses check if needed

            K = z1 * z2 * z3 + zm * (z1 * z2 + z2 * z3 + z1 * z3)
            self._Y_11 = (z2 * z3 + zm * (z2 + z3)) * abs2_a / K
            self._Y_22 = (z1 * z3 + zm * (z1 + z3)) * abs2_b / K
            self._Y_33 = (z1 * z2 + zm * (z1 + z2)) * abs2_c / K

            self._Y_12 = -(z3 * zm) * (np.conj(a) * b) / K
            self._Y_21 = -(z3 * zm) * (np.conj(b) * a) / K

            self._Y_13 = -(z2 * zm) * (np.conj(a) * c) / K
            self._Y_31 = -(z2 * zm) * (np.conj(c) * a) / K

            self._Y_23 = -(z1 * zm) * (np.conj(b) * c) / K
            self._Y_32 = -(z1 * zm) * (np.conj(c) * b) / K

        else:
            raise UserWarning(f"Trafo Model: {trafo3w_model}, not supported. Only pi and t are available.")

        # for DC powerflow
        X1 = np.imag(z1)
        X2 = np.imag(z2)
        X3 = np.imag(z3)

        z1_dc = 1j * X1
        z2_dc = 1j * X2
        z3_dc = 1j * X3

        K_dc = z1_dc * z2_dc + z1_dc * z3_dc + z2_dc * z3_dc

        Y_hh_dc0 = (z2_dc + z3_dc) / K_dc
        Y_mm_dc0 = (z1_dc + z3_dc) / K_dc
        Y_ll_dc0 = (z1_dc + z2_dc) / K_dc

        Y_hm_dc0 = -z3_dc / K_dc
        Y_mh_dc0 = -z3_dc / K_dc

        Y_hl_dc0 = -z2_dc / K_dc
        Y_lh_dc0 = -z2_dc / K_dc

        Y_ml_dc0 = -z1_dc / K_dc
        Y_lm_dc0 = -z1_dc / K_dc

        a_dc = np.abs(a)
        b_dc = np.abs(b)
        c_dc = np.abs(c)

        self._DC_Y_hh = Y_hh_dc0 / (a_dc**2)
        self._DC_Y_mm = Y_mm_dc0 / (b_dc**2)
        self._DC_Y_ll = Y_ll_dc0 / (c_dc**2)

        self._DC_Y_hm = Y_hm_dc0 / (a_dc * b_dc)
        self._DC_Y_mh = Y_mh_dc0 / (a_dc * b_dc)

        self._DC_Y_hl = Y_hl_dc0 / (a_dc * c_dc)
        self._DC_Y_lh = Y_lh_dc0 / (a_dc * c_dc)

        self._DC_Y_ml = Y_ml_dc0 / (b_dc * c_dc)
        self._DC_Y_lm = Y_lm_dc0 / (b_dc * c_dc)
