# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Main module."""

import copy
import cProfile
import pstats
from time import time

import numpy as np
import pandas as pd
import pytest
from pandapower.auxiliary import pandapowerNet
from pandapower.create import (
    create_bus,
    create_empty_network,
    create_ext_grid,
    create_line_from_parameters,
    create_load,
    create_transformer,
)
from pandapower.networks.power_system_test_cases import case5, case9, case14, case118, case9241pegase
from pandapower.run import runpp
from pandapower.toolbox.data_modification import create_continuous_bus_index

from p3s.calculateTrafoTapTable import calculateTrafoCharacteristic
from p3s.NewtonPowerflow import NewtonPowerflow


def net_with_trafo_characteristic() -> pandapowerNet:
    net = create_empty_network()
    net.name = "trafo_test_network"
    vn_kv = 20
    b1 = create_bus(net, vn_kv=vn_kv)
    b2 = create_bus(net, vn_kv=vn_kv)
    create_ext_grid(net, b1, vm_pu=1.01)
    create_line_from_parameters(net, b1, b2, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12, c_nf_per_km=300, max_i_ka=0.2)
    cb = create_bus(net, vn_kv=0.4)
    create_load(net, cb, 0.2, 0.05)
    create_transformer(
        net,
        hv_bus=b2,
        lv_bus=cb,
        std_type="0.25 MVA 20/0.4 kV",
        tap_pos=-2,
        shift_degree=150,
        id_characteristic_table=0,
        tap_dependency_table=True,
    )

    # add trafo_characteristic_table, it is only a 2W trafo
    df = (
        pd.DataFrame(
            {
                "id_characteristic": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                "step": [-2, -1, 0, 1, 2, -2, -1, 0, 1, 2],
                "voltage_ratio": [0.95, 0.975, 1, 1.025, 1.05, 0.95, 0.975, 1, 1.025, 1.05],
                "angle_deg": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "vk_percent": [5, 5.2, 6, 6.8, 7, 5, 5.2, 6, 6.8, 7],
                "vkr_percent": [1.3, 1.4, 1.44, 1.5, 1.6, 1.3, 1.4, 1.44, 1.5, 1.6],
                "vk_hv_percent": np.nan,
                "vkr_hv_percent": np.nan,
                "vk_mv_percent": np.nan,
                "vkr_mv_percent": np.nan,
                "vk_lv_percent": np.nan,
                "vkr_lv_percent": np.nan,
            }
        )
        .set_index(["id_characteristic", "step"])
        .sort_index()
    )

    net["trafo_characteristic_table"] = df

    create_continuous_bus_index(net)

    return net


@pytest.fixture
def networks():
    nets = [net_with_trafo_characteristic(), case5(), case9(), case14(), case118(), case9241pegase()]
    return nets


def test_trafo():
    net = case9241pegase()
    # net.trafo.shift_degree = 0.
    net2 = copy.deepcopy(net)

    runpp(net2, trafo_model="t", init="flat", lightsim2grid=False, numba=False, calculate_voltage_angles=True)
    calculateTrafoCharacteristic(net, inplace=True)

    npf = NewtonPowerflow(net)
    npf.calculate(net, init="flat")

    profiler = cProfile.Profile()
    profiler.enable()
    npf.calculate(net, init="flat")
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("time").print_stats(10)
    stats.dump_stats("performance.prof")


def test_p3s(networks: pandapowerNet):
    N = 1

    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.max_columns", 1000)
    pd.set_option("display.width", 1000)

    # initialize all the jit parts, for both pp and p3s
    runpp(networks[0], init="flat")

    calculateTrafoCharacteristic(networks[0], inplace=True)
    npf = NewtonPowerflow(networks[0])
    npf.calculate(networks[0], init="flat")

    for net in networks:
        print(f"{'-' * 80}")
        print(net)
        net2 = copy.deepcopy(net)

        pp_time = time()
        for _ in range(N):
            runpp(net2, trafo_model="t", init="flat", lightsim2grid=False, numba=False)
        pp_time = (time() - pp_time) / N

        if len(net["trafo"]) > 0 and "trafo_characteristic_table" not in net:
            calculateTrafoCharacteristic(net, inplace=True)

        # voltage = (net2.res_bus.vm_pu * np.exp(1j * np.deg2rad(net2.res_bus.va_degree))).values

        net.name = f"case{len(net.bus)}"

        cpu_time = time()
        npf = NewtonPowerflow(net)
        for _ in range(N):
            npf.calculate(net, init="flat")  # , voltage=voltage)
        cpu_time = (time() - cpu_time) / N

        print(
            f"pp, {net2['_ppc']['iterations']} time: {round(pp_time, 3)}s "
            f"cpu, {net['_ppc']['iterations']} time: {round(cpu_time, 3)}s"
        )
