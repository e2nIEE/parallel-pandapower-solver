# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Main module."""

import copy
import os
from time import time

import pandas as pd
from pandapower.auxiliary import pandapowerNet
from pandapower.create import (
    create_bus,
    create_empty_network,
    create_ext_grid,
    create_line_from_parameters,
    create_load,
    create_transformer_from_parameters,
)
from pandapower.networks.power_system_test_cases import (
    case5,
    case9,
    case14,
    case118,
    case1354pegase,
    case2848rte,
    case6495rte,
    case9241pegase,
)
from pandapower.run import runpp
from pandapower.toolbox.data_modification import create_continuous_bus_index

from p3s.calculateTrafoTapTable import calculate_trafo_characteristic
from p3s.NewtonPowerflow import NewtonPowerflow

_HAS_CUDA = False
if os.environ.get("P3S_BENCH_CUDA") == "1":
    try:
        from p3s.cuda.NewtonPowerflowCuda import NewtonPowerflowCUDA

        _HAS_CUDA = True
    except (ImportError, FileNotFoundError):
        _HAS_CUDA = False


def net_with_trafo_characteristic() -> pandapowerNet:
    net = create_empty_network()
    vn_kv = 20
    b1 = create_bus(net, vn_kv=vn_kv)
    b2 = create_bus(net, vn_kv=vn_kv)
    create_ext_grid(net, b1, vm_pu=1.01)
    create_line_from_parameters(net, b1, b2, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12, c_nf_per_km=300, max_i_ka=0.2)
    cb = create_bus(net, vn_kv=0.4)
    create_load(net, cb, 0.2, 0.05)

    create_transformer_from_parameters(
        net,
        hv_bus=b2,
        lv_bus=cb,
        sn_mva=9900.0,
        vn_hv_kv=20,
        vn_lv_kv=0.4,
        vk_percent=2070.288,
        vkr_percent=0.0,
        pfe_kw=0.0,
        i0_percent=0.0,
        shift_degree=0.0,
        tap_side="hv",
        tap_neutral=0.0,
        tap_step_percent=2.2,
        tap_pos=-1.0,
        in_service=True,
        max_loading_percent=100.0,
        tap_changer_type="Ratio",
    )

    return net


if __name__ == "__main__":
    N = 100

    net = net_with_trafo_characteristic()
    create_continuous_bus_index(net)
    calculate_trafo_characteristic(net, inplace=True)

    npf_cuda = NewtonPowerflowCUDA(net)
    npf_cuda.calculate_cuda(net, init="flat")

    npf = NewtonPowerflow(net)
    npf.calculate(net, init="flat")

    nets = {
        "case5": case5(),
        "case9": case9(),
        "case14": case14(),
        "case118": case118(),
        "case1354pegase": case1354pegase(),
        "case2848rte": case2848rte(),
        "case6495rte": case6495rte(),
        "case9241pegase": case9241pegase(),
    }

    results = pd.DataFrame(index=list(nets.keys()), columns=["pp", "p3s", "gpu"])

    for name, net in nets.items():
        print(f"{'-' * 80}")
        print(name)
        net.trafo.shift_degree = 0.0
        net2 = copy.deepcopy(net)

        pp_start_time = time()
        for _ in range(N):
            runpp(net2, trafo_model="t", init="flat", tolerance_mva=1e-5)
        pp_end_time = time()
        pp_iter = net2["_ppc"]["iterations"]

        if len(net["trafo"]) > 0 and "trafo_characteristic_table" not in net:
            calculate_trafo_characteristic(net, inplace=True)

        if _HAS_CUDA:
            cu_start_time = time()
            npf_cuda = NewtonPowerflowCUDA(net)
            cuda_iter = -1
            for _ in range(N):
                _, cuda_iter = npf_cuda.calculate_cuda(net, init="flat", max_iterations=100, tolerance=1e-5)
            cu_end_time = time()
        else:
            cu_start_time = time()
            cu_end_time = cu_start_time
            cuda_iter = 0

        p3s_start_time = time()
        npf = NewtonPowerflow(net)
        for _ in range(N):
            npf.calculate(net, init="flat", max_iterations=100, tolerance=1e-5)
        p3s_end_time = time()
        p3s_iter = net["_ppc"]["iterations"]

        print(
            f"pp time[ms]: {(pp_end_time - pp_start_time) / N * 1000.0}, iterations: {pp_iter}, "
            f"gpu time[ms]: {(cu_end_time - cu_start_time) / N * 1000.0}, iterations: {cuda_iter}, "
            f"p3s time[ms]: {(p3s_end_time - p3s_start_time) / N * 1000.0}, iterations: {p3s_iter},"
        )
        results.loc[name, "pp"] = (pp_end_time - pp_start_time) / N * 1000.0
        results.loc[name, "gpu"] = (cu_end_time - cu_start_time) / N * 1000.0
        results.loc[name, "p3s"] = (p3s_end_time - p3s_start_time) / N * 1000.0

    # optional saving results for analysis
    # results.to_excel('results.xlsx')
