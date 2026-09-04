# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

from pandas import DataFrame

from p3s.models.TwoPort import TwoPort


def dc_pi_model(R, G, length, vn_kv, parallel, sn_mva=1.0):

    # 1) Calculate scaling, to scale everything in p.u.
    R_N = (vn_kv**2) / sn_mva

    # 2) calculate series impedance
    Z_sr = 1 / (R / R_N) * length / parallel

    # 3) calculate shunt impedance
    Y_sr = 0.5 * G * 1e-6 * R_N * length / parallel

    # 4) Calculate transfer parameters
    Y_ff = Z_sr + Y_sr
    Y_tt = Z_sr + Y_sr
    Y_ft = -Z_sr
    Y_tf = -Z_sr

    return Y_ff, Y_ft, Y_tf, Y_tt


class DCLineModel(TwoPort):
    def __init__(self, line_dc_table: DataFrame, sn_mva: float = 1.0):
        super().__init__()
        self._from_bus_dc = line_dc_table["from_bus_dc"].values
        self._to_bus_dc = line_dc_table["to_bus_dc"].values
        self.voltages = line_dc_table["vn_kv"].values

        length = line_dc_table["length_km"].values
        R = line_dc_table["r_ohm_per_km"].values
        G = line_dc_table["g_us_per_km"].values

        parallel = line_dc_table["parallel"].values

        self._Y_ff, self._Y_ft, self._Y_tf, self._Y_tt = dc_pi_model(
            R=R,
            G=G,
            length=length,
            vn_kv=self.voltages,
            sn_mva=sn_mva,
            parallel=parallel,
        )
