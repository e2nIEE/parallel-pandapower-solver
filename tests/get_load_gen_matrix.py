# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy.typing as npt
from pandapower.auxiliary import pandapowerNet
from pandapower.control.controller.const_control import ConstControl


def extract_timeseries(
    net: pandapowerNet,
    elements=None,
    transpose=True,
) -> dict[tuple[str, str], npt.NDArray]:
    """Function to extract timeseries as several matrices for faster calculation."""

    if elements is None:
        elements = {
            "load": ("p_mw", "q_mvar"),
            "sgen": ("p_mw", "q_mvar"),
            "gen": ("p_mw", "q_mvar"),
        }

    # build a map of controller
    cntrl_element_map: dict[tuple[str, str], ConstControl] = {}

    for _, cntrl_ser in net.controller.iterrows():
        controller = cntrl_ser.iloc[0]
        if isinstance(controller, ConstControl):
            cntrl_element_map[(controller.element, controller.variable)] = controller

    # build matrix from DFData data_sources
    ts_data: dict[tuple[str, str], npt.NDArray] = {}

    for element, variables in elements.items():
        for variable in variables:
            key = (element, variable)

            if key in cntrl_element_map:
                df = cntrl_element_map[key].data_source.df
                if transpose:
                    ts_data[key] = df.to_numpy().transpose()
                else:
                    ts_data[key] = df.to_numpy()

    return ts_data
