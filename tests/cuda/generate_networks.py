# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause
"""Module for loading and processing power system test cases.

This module provides functionalities to manage power system test cases, including
the addition of profiles and controllers using Simbench. It supports loading networks
from JSON files or directly from test case definitions. If Simbench profiles are
available, they are applied to the networks.

It is needed for timeseries test, to prepare networks with timeseries data.

"""

import os

import pandapower.networks.power_system_test_cases as pstc
from pandapower.file_io import from_json, to_json

from tests.util import extract_timeseries

try:
    import simbench as sb

    simbench_available = True
except ImportError:
    sb = None
    simbench_available = False

_filepath = os.path.dirname(os.path.abspath(__file__))


def _json_available(name: str) -> bool:
    if name in os.listdir(_filepath):
        return True
    return False


def _add_controllers_from_simbench_profiles(net):
    if not simbench_available:
        raise RuntimeError("simbench not available, failed to load profiles")

    profiles = sb.get_all_simbench_profiles(0)
    net["profiles"] = profiles
    absolute_values = sb.get_absolute_values(net, True)
    sb.apply_const_controllers(net, absolute_values)


def case_with_profiles(name: str):
    if _json_available(f"{name}.json"):
        return from_json(os.path.join(_filepath, f"{name}.json"))

    net_func = getattr(pstc, name)

    net = net_func()
    # Add controllers from Simbench
    _add_controllers_from_simbench_profiles(net)
    to_json(net, os.path.join(_filepath, f"{name}.json"))
    return net


if __name__ == "__main__":
    for net_name in ["case5", "case9", "case14", "case118", "case9241pegase"]:
        net = case_with_profiles(net_name)
        extract_timeseries(net)
