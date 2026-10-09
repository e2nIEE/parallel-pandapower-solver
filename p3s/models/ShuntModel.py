# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pandas as pd

from p3s.models.TwoPort import TwoPort


class ShuntModel(TwoPort):
    def __init__(self, shunt_table: pd.DataFrame, bus_table: pd.DataFrame, sn_mva: float = 1.0):

        super().__init__()
        self._from_bus = shunt_table["bus"]
        self._to_bus = shunt_table["bus"]

        # Base voltages of both terminals, kept for the res_impedance currents
        # (i_ka = |S| / (vn_kv * vm_pu * sqrt(3))). Unlike a line, an impedance may span a
        # voltage step.
        self.voltages_bus = bus_table.loc[self._from_bus, "vn_kv"].values

        q_mvar = shunt_table["q_mvar"]
        p_mw = shunt_table["p_mw"]
        step = shunt_table["step"]
        in_service = shunt_table["in_service"]

        # voltage ratio correction for shunts
        self.shunt_vn_kv = shunt_table["vn_kv"].to_numpy(dtype=float)
        self.shunt_vn_kv = np.where(np.isnan(self.shunt_vn_kv), self.voltages_bus, self.shunt_vn_kv)

        # TODO: shunt stepping behaviour still missing
        # max_step = shunt_table["max_step"]
        # step_dependency_table = shunt_table["step_dependency_table"]
        # id_characteristic_table = shunt_table["id_characteristic_table"]

        s_shunt = (p_mw - q_mvar * 1j) * step
        v_ratio = (self.voltages_bus / self.shunt_vn_kv) ** 2
        y_shunt = s_shunt * v_ratio / sn_mva
        zeros = np.zeros_like(y_shunt)

        self._Y_ff = np.where(in_service, y_shunt, zeros)
        self._Y_tf = np.zeros_like(y_shunt)
        self._Y_ft = np.zeros_like(y_shunt)
        self._Y_tt = np.zeros_like(y_shunt)

        # A shunt is a pure admittance with no series reactance, so it contributes
        # nothing to the DC B-matrix (pandapower's makeBdc builds B from branch 1/x only).
        # Kept explicit so create_y_dc_matrix() stays safe to call on this element.
        self._DC_Yff = np.zeros_like(y_shunt)
        self._DC_Yft = np.zeros_like(y_shunt)
        self._DC_Ytf = np.zeros_like(y_shunt)
        self._DC_Ytt = np.zeros_like(y_shunt)
