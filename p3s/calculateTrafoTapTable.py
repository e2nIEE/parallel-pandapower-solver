# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pandas as pd

from pandapower.auxiliary import pandapowerNet


def calculateTrafoCharacteristic(net: pandapowerNet, inplace: bool = False):

    tap_min = net.trafo['tap_min']
    tap_max = net.trafo['tap_max']
    tap_pos = net.trafo['tap_pos']
    tap_neutral = net.trafo['tap_neutral']
    id_characteristic = range(len(net.trafo))
    tap_step_percent = net.trafo['tap_step_percent']
    vk_percent = net.trafo['vk_percent']
    vkr_percent = net.trafo['vkr_percent']
    # NOTE: this local is the per-step TAP phase shift used by the _ideal/_ratio tap
    # builders below (their tap = ...exp(1j*deg2rad(tap_step_degree...))). It is the
    # trafo's tap_step_degree, NOT shift_degree: the fixed vector-group shift_degree is
    # added separately in TwoWindingTransformerModel (theta = angle_deg + shift_degree).
    # Using shift_degree here previously double-counted it (angle_deg carried shift_degree
    # and the model added shift_degree again -> 2x the intended phase shift).
    # tap_step_degree is NaN for non-phase-shifting taps (pandapower convention) -> 0.
    tap_step_degree = net.trafo['tap_step_degree'].fillna(0.0)

    # Fetch voltage levels
    hv_kv = net.bus.loc[net.trafo.hv_bus, 'vn_kv']
    lv_kv = net.bus.loc[net.trafo.lv_bus, 'vn_kv']

    lv_voltage = net.trafo["vn_lv_kv"].values
    hv_voltage = net.trafo["vn_hv_kv"].values

    # and calculate a correction if the trafo has a different voltage level than the attached buses
    a_ratio = hv_voltage / lv_voltage * hv_kv / lv_kv

    # helper functions for every type -----------------------------------

    n_taps = np.empty(len(net.trafo), dtype=float)
    tap = np.empty(len(net.trafo), dtype=complex)
    ratios = np.ones(shape=len(net.trafo))

    def _ideal(idx):
        n_taps[idx] = 2 * np.rad2deg((tap_pos[idx] - tap_neutral[idx]) * tap_step_percent[idx] / 200.)
        tap[idx] = np.exp(1j * np.deg2rad(tap_step_degree[idx] + n_taps[idx]))

    def _standard(idx):
        tap[idx] = 1. + 0j

    def _ratio(idx):
        # voltages[idx] *= (tap_pos[idx] - tap_neutral[idx]) * tap_st_per[idx] / 100.
        ratios[idx] = 1 + (tap_pos[idx] - tap_neutral[idx]) * tap_step_percent[idx] / 100.
        tap[idx] = ratios[idx] * np.exp(1j * np.deg2rad(tap_step_degree[idx]))

    def _symmetrical(idx):
        n_taps = -2 * np.rad2deg((tap_pos[idx] - tap_neutral[idx]) * tap_step_percent[idx] / 200.)
        tap[idx] = np.exp(1j * np.deg2rad(-tap_step_degree[idx] + n_taps[idx]))

    def _tabular(idx):
        n_taps[idx] = 0.
        tap[idx] = 0j

    dispatch = {
        "Ideal":        _ideal,
        "Ratio":        _ratio,
        "Symmetrical":  _symmetrical,
        "Tabular":      _tabular,
        None:           _standard,
    }

    tap_changer_type = net.trafo["tap_changer_type"].values

    for type, func in dispatch.items():
        mask = tap_changer_type == type
        if np.any(mask):
            func(mask)

    # Tap-induced phase shift for the current step: (tap_pos - tap_neutral) *
    # tap_step_degree. This is what goes into the characteristic table's angle_deg; the
    # model adds the fixed shift_degree on top. (Previously angle_deg held shift_degree,
    # double-counting it.)
    tap_angle_deg = (tap_pos.fillna(0.) - tap_neutral.fillna(0.)) * tap_step_degree

    df = pd.DataFrame(
        {'id_characteristic': range(len(net.trafo)),
         'step': tap_pos.fillna(0.),
         'voltage_ratio': ratios,
         'angle_deg': tap_angle_deg,
         'vk_percent': vk_percent,
         'vkr_percent': vkr_percent,
         'vk_hv_percent': np.nan,
         'vkr_hv_percent': np.nan,
         'vk_mv_percent': np.nan,
         'vkr_mv_percent': np.nan,
         'vk_lv_percent': np.nan,
         'vkr_lv_percent': np.nan
         }
    ).set_index(['id_characteristic', 'step']).sort_index()

    net.trafo["id_characteristic_table"] = range(len(net.trafo))

    if inplace:
        net["trafo_characteristic_table"] = df
        return None
    else:
        return df


if __name__ == '__main__':
    import pandapower.networks as nw
    net = nw.case14()

    calculateTrafoCharacteristic(net, inplace=True)
    pass
