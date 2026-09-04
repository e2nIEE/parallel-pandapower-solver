# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pandas as pd

from p3s.models.TwoPort import TwoPort


class ShuntModel(TwoPort):
    def __init__(self, shunt_table: pd.DataFrame, sn_mva: float = 1.0):

        super().__init__()
        self._from_bus = shunt_table["bus"]
        self._to_bus = shunt_table["bus"]

        q_mvar = shunt_table["q_mvar"]
        p_mw = shunt_table["p_mw"]
        step = shunt_table["step"]
        in_service = shunt_table["in_service"]

        # TODO: implement stepping behaviour for shunts
        # vn_kv = shunt_table["vn_kv"]
        # max_step = shunt_table["max_step"]
        # step_dependency_table = shunt_table["step_dependency_table"]
        # id_characteristic_table = shunt_table["id_characteristic_table"]

        s_shunt = (p_mw - q_mvar * 1j) * step
        y_shunt = s_shunt / sn_mva
        zeros = np.zeros_like(y_shunt)

        self._Y_ff = np.where(in_service, y_shunt, zeros)
        self._Y_tf = np.zeros_like(y_shunt)
        self._Y_ft = np.zeros_like(y_shunt)
        self._Y_tt = np.zeros_like(y_shunt)
