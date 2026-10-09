# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandapower import LoadflowNotConverged, pandapowerNet
from scipy.sparse import coo_matrix
from scipy.sparse import coo_matrix as sparse
from scipy.sparse.linalg import spsolve

from p3s.models.ImpedanceModel import ImpedanceModel
from p3s.models.ShuntModel import ShuntModel
from p3s.models.ThreeWindingTransformerModel import ThreeWindingTransformerModel
from p3s.models.TransmissionLineModel import TransmissionLineModel
from p3s.models.TwoWindingTransformerModel import TwoWindingTransformerModel
from p3s.models.WardModel import WardModel
from p3s.PowerflowObject import PowerflowObject
from p3s.PQPVPowerflow import PQPVPowerflow
from p3s.timeseries import build_sbus_matrix, dc_initial_voltage, mean_setpoint_vm
from p3s.topology import Topology, build_topology, switch_flows, write_switch_results, zero_unsupplied_flows
from p3s.voltage_sources import bus_types, voltage_sources, write_unit_results

# Fast C++ Newton-Raphson solver (polar formulation, KLU linear solve). Installed into
# the p3s package by `pip install parallel-pandapower-solver[cpp]` (CMake / scikit-build-core; see
# p3s/cpp/). For an in-place dev build there is also p3s/cpp/build.sh.
try:
    from p3s import nr_klu  # type: ignore[attr-defined]
except ImportError:
    from p3s.cpp import nr_klu  # type: ignore[attr-defined]


class NewtonPowerflow:
    def __init__(self, net):
        self.pf_objects: dict[str, PowerflowObject] = {}
        self._sBus: NDArray = None
        self._YBus: sparse = None
        self._Bbus: sparse = None
        self._p_shift = None
        self._lookup = None
        self._reverse_lookup = None
        self._initial_voltage = None
        # keeping a reference on all classes which create the ybus.
        self._ybus_elements = {}
        self.busses: dict[str, NDArray] = None
        # in-service ext_grids + gens as one table (p3s.voltage_sources)
        self._units: pd.DataFrame = None
        # Switches / connectivity (p3s.topology): the power flow runs on nodes (buses fused by
        # closed bus-bus switches, unsupplied buses left out). _YBus/_Bbus/_sBus/_p_shift/busses
        # are node-level; _YBus_bus and _sBus_bus keep the bus-level versions for the results.
        self._topology: Topology = None
        self._YBus_bus: sparse = None
        self._sBus_bus: NDArray = None
        self._unit_bus_pos: NDArray = None
        # Cached C++ solver (KLU symbolic analyze is done once per topology and
        # reused across calculate() calls -> only the cheap numeric refactor runs
        # on subsequent solves). Invalidated when the Ybus structure changes.
        self._cpp_solver = None
        self._cpp_solver_nnz = None
        self._setup_pf(net)

    def make_ybus(self, net: pandapowerNet) -> tuple[sparse, sparse]:
        n_bus = len(net.bus)
        Ybus_dat = []
        Ybus_row = []
        Ybus_col = []
        Bbus_dat = []
        Bbus_row = []
        Bbus_col = []

        if "line" in net and len(net.line) > 0:
            voltages = net.bus.vn_kv[net.line.from_bus].values
            lines = TransmissionLineModel(net.line, voltages, f_hz=net.f_hz, sn_mva=net.sn_mva)
            lines.apply_open_ends(self._topology.open_ends["line"])
            self._ybus_elements["line"] = lines
            Ybus_lines = lines.create_y_matrix(n_bus=n_bus)
            Ybus_dat.extend(Ybus_lines.data)
            Ybus_row.extend(Ybus_lines.row)
            Ybus_col.extend(Ybus_lines.col)

            Bbus_lines = lines.create_y_dc_matrix(n_bus=n_bus)
            Bbus_dat.extend(Bbus_lines.data)
            Bbus_row.extend(Bbus_lines.row)
            Bbus_col.extend(Bbus_lines.col)

        if "trafo" in net and len(net.trafo) > 0:
            trafos = TwoWindingTransformerModel(
                net.trafo, net.bus, sn_mva=net.sn_mva, tap_table=net.trafo_characteristic_table
            )
            trafos.apply_open_ends(self._topology.open_ends["trafo"])
            self._ybus_elements["trafo"] = trafos
            Ybus_trafos = trafos.create_y_matrix(n_bus=n_bus)
            Ybus_dat.extend(Ybus_trafos.data)
            Ybus_row.extend(Ybus_trafos.row)
            Ybus_col.extend(Ybus_trafos.col)

            Bbus_trafos = trafos.create_y_dc_matrix(n_bus=n_bus)
            Bbus_dat.extend(Bbus_trafos.data)
            Bbus_row.extend(Bbus_trafos.row)
            Bbus_col.extend(Bbus_trafos.col)

            self._p_shift += trafos.p_shift

        if "trafo3w" in net and len(net.trafo3w) > 0:
            trafo3ws = ThreeWindingTransformerModel(
                net.trafo3w,
                bus_table=net.bus,
                tap_table=net.trafo_characteristic_table,
                sn_mva=net.sn_mva,
            )
            trafo3ws.apply_open_ends(self._topology.open_ends["trafo3w"])
            self._ybus_elements["trafo3w"] = trafo3ws
            Ybus_trafos3w = trafo3ws.create_y_matrix(n_bus=n_bus)
            Ybus_dat.extend(Ybus_trafos3w.data)
            Ybus_row.extend(Ybus_trafos3w.row)
            Ybus_col.extend(Ybus_trafos3w.col)

            Bbus_trafos3w = trafo3ws.create_y_dc_matrix(n_bus=n_bus)
            Bbus_dat.extend(Bbus_trafos3w.data)
            Bbus_row.extend(Bbus_trafos3w.row)
            Bbus_col.extend(Bbus_trafos3w.col)

            self._p_shift += trafo3ws.p_shift

        # A ward is an ACTIVE element: only its constant-impedance half (pz/qz) can be
        # stamped here. The constant-power half (ps/qs) is picked up from
        # ``wards.s_bus`` in _setup_pf, which runs after make_ybus.
        if "ward" in net and len(net.ward) > 0:
            wards = WardModel(net.ward, n_bus=n_bus, sn_mva=net.sn_mva)
            self._ybus_elements["ward"] = wards
            Ybus_wards = wards.create_y_matrix(n_bus=n_bus)
            Ybus_dat.extend(Ybus_wards.data)
            Ybus_row.extend(Ybus_wards.row)
            Ybus_col.extend(Ybus_wards.col)

        if "shunt" in net and len(net.shunt) > 0:
            shunts = ShuntModel(net.shunt, net.bus, sn_mva=net.sn_mva)
            self._ybus_elements["shunt"] = shunts
            Ybus_shunts = shunts.create_y_matrix(n_bus=n_bus)
            Ybus_dat.extend(Ybus_shunts.data)
            Ybus_row.extend(Ybus_shunts.row)
            Ybus_col.extend(Ybus_shunts.col)

        if "impedance" in net and len(net.impedance) > 0:
            impedances = ImpedanceModel(net.impedance, net.bus, sn_mva=net.sn_mva)
            self._ybus_elements["impedance"] = impedances
            Ybus_impedances = impedances.create_y_matrix(n_bus=n_bus)
            Ybus_dat.extend(Ybus_impedances.data)
            Ybus_row.extend(Ybus_impedances.row)
            Ybus_col.extend(Ybus_impedances.col)

            Bbus_impedances = impedances.create_y_dc_matrix(n_bus=n_bus)
            Bbus_dat.extend(Bbus_impedances.data)
            Bbus_row.extend(Bbus_impedances.row)
            Bbus_col.extend(Bbus_impedances.col)

        Ybus = coo_matrix((Ybus_dat, (Ybus_row, Ybus_col)), shape=(n_bus, n_bus)).tocsr()
        Bbus = coo_matrix((Bbus_dat, (Bbus_row, Bbus_col)), shape=(n_bus, n_bus)).tocsr().imag
        return Ybus, Bbus

    def _setup_pf(self, net: pandapowerNet):
        # Switches and connectivity first: the open branch ends shape the element stamps, and
        # the node mapping decides where everything goes (p3s.topology).
        topo = self._topology = build_topology(net)
        self._p_shift = np.zeros(len(net.bus))

        if self._YBus is None:
            self._YBus_bus, bbus_bus = self.make_ybus(net)
            self._YBus = topo.to_nodes(self._YBus_bus)
            self._Bbus = topo.to_nodes(bbus_bus)
        self._p_shift = topo.vector_to_nodes(self._p_shift)

        # Lookup tables: pandapower bus index -> node (Ybus row, -1 = unsupplied) and node -> the
        # pandapower index of its first bus. One-to-one unless buses are fused or unsupplied.
        self._lookup = topo.lookup
        self._reverse_lookup = topo.representative_bus

        # create an initial voltage vector, for flat start (1pu, 0°) and overwrite it with setpoints of gen's
        self._initial_voltage = np.ones(shape=topo.n_node, dtype=np.complex128)

        # Scheduled demand per bus (net.bus row; loads positive), aggregated onto the nodes below.
        # The bus-level version is kept for the results (bus-bus switch currents).
        s_demand: NDArray = np.zeros(len(net.bus), dtype=complex)

        def _add(element: str, s_mva: NDArray) -> None:
            net[element]["_lookup"] = self._lookup[net[element].bus].to_numpy(dtype=int)  # timeseries
            np.add.at(s_demand, net.bus.index.get_indexer(net[element].bus), s_mva)

        if "load" in net and len(net["load"]) > 0:
            _add("load", ((net.load.p_mw + net.load.q_mvar * 1j) * net.load.scaling * net.load.in_service).to_numpy())

        if "sgen" in net and len(net["sgen"]) > 0:
            s_sgen = (net.sgen.p_mw + net.sgen.q_mvar * 1j) * net.sgen.scaling * net.sgen.in_service
            _add("sgen", -s_sgen.to_numpy())

        if "gen" in net and len(net["gen"]) > 0:
            # used by timeseries.build_sbus_matrix
            net.gen["_lookup"] = self._lookup[net.gen.bus].to_numpy(dtype=int)

        # ext_grids and gens as one table of in-service voltage-controlling units (see
        # p3s.voltage_sources): an ext_grid is a slack unit without scheduled P, a gen with
        # slack=True is a slack unit too. They define the bus types and start voltages.
        self._units = voltage_sources(net, self._lookup)
        self._unit_bus_pos = np.zeros(len(self._units), dtype=int)
        for et in ("ext_grid", "gen"):
            sel = (self._units.et == et).to_numpy()
            if sel.any():
                self._unit_bus_pos[sel] = net.bus.index.get_indexer(net[et].bus.loc[self._units.idx[sel]])
        np.add.at(s_demand, self._unit_bus_pos, -self._units.p_mw.to_numpy(dtype=float))
        ref, pv, pq = bus_types(self._units, topo.n_node, self._initial_voltage)

        # Seed the remaining PQ buss at the mean generator / ext_grid set point rather than a flat 1.0 pu.
        # This is what pandapower's init = "auto" does, and it saves Newton iterations.
        if len(pq) > 0:
            self._initial_voltage[pq] = mean_setpoint_vm(net)

        # Constant-power half of the ward equivalents. The shunt half is already in Ybus
        # (see make_ybus); this adds ps/qs as an ordinary PQ demand. A ward has no
        # scaling column -- pandapower hardcodes scaling = 1.0 for ward/xward -- and
        # out-of-service wards were zeroed when the model was built.
        if "ward" in self._ybus_elements:
            s_demand += self._ybus_elements["ward"].s_bus

        self._sBus_bus = -1.0 * s_demand / net.sn_mva
        self._sBus = topo.vector_to_nodes(self._sBus_bus)
        self.pf_objects["PVPQ"] = PQPVPowerflow(YBus=self._YBus, pv=pv, pq=pq, ref=ref)
        self.busses = {"ref": ref, "pv": pv, "pq": pq}

    def _calc_mismatch(self, Sbus, voltage) -> NDArray:
        mismatch_partials = []
        for pf_object in self.pf_objects.values():
            mismatch_partials.append(pf_object.evaluate_Fx(Sbus, voltage))

        mismatch_F: NDArray = np.r_[*mismatch_partials]
        return mismatch_F

    def _calc_results_dx(self, dx: NDArray, voltage):
        results_partials = []
        for pf_object in self.pf_objects.values():
            results_partials.append(pf_object.evaluate_Results(dx, voltage))

        results_dx: NDArray = np.r_[*results_partials]
        return results_dx

    def _parse_results(self, net, voltage):
        def _ensure_index(df: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
            if len(df) != len(index) or not df.index.equals(index):
                return df.reindex(index)
            return df

        topo = self._topology
        if len(voltage) != topo.n_node:
            voltage = topo.node_voltage(voltage)  # given per net.bus row (as calculate returns it)

        # -- gen / ext_grid results (on nodes) --
        # The residual (computed minus scheduled injection) is what the voltage-controlling
        # units supplied on top of their schedule; p3s.voltage_sources shares it out.
        s_node = np.conj(self._YBus * voltage) * voltage * net.sn_mva
        p_unit, q_unit = write_unit_results(
            net, self._units, s_node - self._sBus * net.sn_mva, np.abs(voltage), np.angle(voltage, deg=True)
        )

        # -- from nodes to buses --
        # v_bus is NaN on unsupplied buses (reported as such); flows use 0 there.
        v_bus = topo.bus_voltage(voltage)
        voltage = np.nan_to_num(v_bus)
        vm = np.abs(v_bus)
        va = np.angle(v_bus, deg=True)
        vm_calc = np.where(np.isnan(vm), 1.0, vm)  # for I = S / V where S = 0 on unsupplied buses

        # Element injection per bus (per unit, injection positive): the schedule plus what the
        # units supplied on top of it, at their own bus. Whatever a bus injects but does not send
        # into its branches and shunts flows through its closed bus-bus switches (KCL per group of
        # fused buses, p3s.topology.switch_flows).
        s_elements = self._sBus_bus.copy()
        unit_extra = (p_unit - self._units.p_mw.to_numpy(dtype=float) + 1j * q_unit) / net.sn_mva
        np.add.at(s_elements, self._unit_bus_pos, unit_extra)
        i_network, i_switches, i_sw = switch_flows(topo, self._YBus_bus, voltage, s_elements)
        sBus = voltage * np.conj(i_network + i_switches) * net.sn_mva

        # -- set bus results --
        net.res_bus = _ensure_index(net.res_bus, net.bus.index)
        res_bus = net.res_bus
        res_bus["vm_pu"].to_numpy(copy=False)[:] = vm
        res_bus["va_degree"].to_numpy(copy=False)[:] = va
        res_bus["p_mw"].to_numpy(copy=False)[:] = -1 * sBus.real
        res_bus["q_mvar"].to_numpy(copy=False)[:] = -1 * sBus.imag

        # -- calculate line results --
        if "line" in self._ybus_elements and "line" in net and len(net.line):
            net.res_line = _ensure_index(net.res_line, net.line.index)
            lines = self._ybus_elements["line"]
            vm_from_pu = vm_calc[lines._from_bus]
            line_power_from = np.conj(lines.yf_matrix * voltage) * voltage[lines._from_bus] * net.sn_mva
            line_currents_from = np.abs(line_power_from / (lines.voltages * vm_from_pu * np.sqrt(3)))

            vm_to_pu = vm_calc[lines._to_bus]
            line_power_to = np.conj(lines.yt_matrix * voltage) * voltage[lines._to_bus] * net.sn_mva
            line_currents_to = np.abs(line_power_to / (lines.voltages * vm_to_pu * np.sqrt(3)))

            # reported end voltages: at an end opened by a switch the voltage of the open terminal
            v_line_from, v_line_to = lines.end_voltages(v_bus)

            res_line = net.res_line
            res_line["p_from_mw"].to_numpy(copy=False)[:] = line_power_from.real
            res_line["q_from_mvar"].to_numpy(copy=False)[:] = line_power_from.imag
            res_line["i_from_ka"].to_numpy(copy=False)[:] = line_currents_from
            res_line["vm_from_pu"].to_numpy(copy=False)[:] = np.abs(v_line_from)
            res_line["va_from_degree"].to_numpy(copy=False)[:] = np.angle(v_line_from, deg=True)

            res_line["p_to_mw"].to_numpy(copy=False)[:] = line_power_to.real
            res_line["q_to_mvar"].to_numpy(copy=False)[:] = line_power_to.imag
            res_line["i_to_ka"].to_numpy(copy=False)[:] = line_currents_to
            res_line["vm_to_pu"].to_numpy(copy=False)[:] = np.abs(v_line_to)
            res_line["va_to_degree"].to_numpy(copy=False)[:] = np.angle(v_line_to, deg=True)

            line_currents_max = np.maximum(line_currents_from, line_currents_to)
            res_line["i_ka"].to_numpy(copy=False)[:] = line_currents_max
            # Losses = what enters the line at both ends (pandapower: pl = p_from + p_to). ql is
            # negative where the line's charging outweighs its series losses (it generates Q).
            res_line["pl_mw"].to_numpy(copy=False)[:] = line_power_from.real + line_power_to.real
            res_line["ql_mvar"].to_numpy(copy=False)[:] = line_power_from.imag + line_power_to.imag

            max_i_ka = net.line["max_i_ka"].to_numpy(copy=False)
            res_line["loading_percent"].to_numpy(copy=False)[:] = line_currents_max / max_i_ka * 100.0

        if "trafo" in self._ybus_elements and "trafo" in net and len(net.trafo):
            net.res_trafo = _ensure_index(net.res_trafo, net.trafo.index)
            trafos = self._ybus_elements["trafo"]
            vm_hv_pu = pd.Series(vm_calc, index=net.bus.index)[net.trafo.hv_bus]
            trafo_power_from = np.conj(trafos.yf_matrix * voltage) * voltage[trafos._from_bus] * net.sn_mva
            trafo_currents_from = np.abs(trafo_power_from / (trafos.voltages_from * vm_hv_pu * np.sqrt(3)))

            vm_lv_pu = pd.Series(vm_calc, index=net.bus.index)[net.trafo.lv_bus]
            trafo_power_to = np.conj(trafos.yt_matrix * voltage) * voltage[trafos._to_bus] * net.sn_mva
            trafo_currents_to = np.abs(trafo_power_to / (trafos.voltages_to * vm_lv_pu * np.sqrt(3)))

            # reported end voltages: at an end opened by a switch the voltage of the open terminal
            v_trafo_hv, v_trafo_lv = trafos.end_voltages(v_bus)

            res_trafo = net.res_trafo
            res_trafo["p_hv_mw"].to_numpy(copy=False)[:] = trafo_power_from.real
            res_trafo["q_hv_mvar"].to_numpy(copy=False)[:] = trafo_power_from.imag
            res_trafo["i_hv_ka"].to_numpy(copy=False)[:] = trafo_currents_from
            res_trafo["vm_hv_pu"].to_numpy(copy=False)[:] = np.abs(v_trafo_hv)
            res_trafo["va_hv_degree"].to_numpy(copy=False)[:] = np.angle(v_trafo_hv, deg=True)

            res_trafo["p_lv_mw"].to_numpy(copy=False)[:] = trafo_power_to.real
            res_trafo["q_lv_mvar"].to_numpy(copy=False)[:] = trafo_power_to.imag
            res_trafo["i_lv_ka"].to_numpy(copy=False)[:] = trafo_currents_to
            res_trafo["vm_lv_pu"].to_numpy(copy=False)[:] = np.abs(v_trafo_lv)
            res_trafo["va_lv_degree"].to_numpy(copy=False)[:] = np.angle(v_trafo_lv, deg=True)

            res_trafo["pl_mw"].to_numpy(copy=False)[:] = trafo_power_from.real + trafo_power_to.real
            res_trafo["ql_mvar"].to_numpy(copy=False)[:] = trafo_power_from.imag + trafo_power_to.imag

            loading_percent = np.maximum(
                trafo_currents_from.values * trafos.voltages_from.values * np.sqrt(3),
                trafo_currents_to.values * trafos.voltages_to.values * np.sqrt(3),
            )
            res_trafo["loading_percent"].to_numpy(copy=False)[:] = loading_percent / net.trafo.sn_mva * 100

        # -- calculate trafo3w results --
        if "trafo3w" in self._ybus_elements and "trafo3w" in net and len(net.trafo3w):
            net.res_trafo3w = _ensure_index(net.res_trafo3w, net.trafo3w.index)
            for column, values in self._ybus_elements["trafo3w"].results(v_bus, net.sn_mva).items():
                net.res_trafo3w[column] = values

        # -- calculate impedance results --
        # res_impedance has no vm_*/va_*/loading_percent columns (an impedance carries no
        # rating), so this is the short form of the line block. Losses are the SUM of both
        # terminal flows -- pandapower's _get_impedance_results uses pl = p_from + p_to,
        # the same convention as res_line and res_trafo.
        if "impedance" in self._ybus_elements and "impedance" in net and len(net.impedance):
            net.res_impedance = _ensure_index(net.res_impedance, net.impedance.index)
            impedances = self._ybus_elements["impedance"]

            imp_power_from = np.conj(impedances.yf_matrix * voltage) * voltage[impedances._from_bus] * net.sn_mva
            imp_power_to = np.conj(impedances.yt_matrix * voltage) * voltage[impedances._to_bus] * net.sn_mva

            # Each terminal is referred to its OWN base voltage: an impedance may span a
            # voltage step, unlike a line.
            imp_currents_from = np.abs(
                imp_power_from / (impedances.voltages_from * vm_calc[impedances._from_bus] * np.sqrt(3))
            )
            imp_currents_to = np.abs(imp_power_to / (impedances.voltages_to * vm_calc[impedances._to_bus] * np.sqrt(3)))

            res_impedance = net.res_impedance
            res_impedance["p_from_mw"].to_numpy(copy=False)[:] = imp_power_from.real
            res_impedance["q_from_mvar"].to_numpy(copy=False)[:] = imp_power_from.imag
            res_impedance["i_from_ka"].to_numpy(copy=False)[:] = imp_currents_from

            res_impedance["p_to_mw"].to_numpy(copy=False)[:] = imp_power_to.real
            res_impedance["q_to_mvar"].to_numpy(copy=False)[:] = imp_power_to.imag
            res_impedance["i_to_ka"].to_numpy(copy=False)[:] = imp_currents_to

            res_impedance["pl_mw"].to_numpy(copy=False)[:] = imp_power_from.real + imp_power_to.real
            res_impedance["ql_mvar"].to_numpy(copy=False)[:] = imp_power_from.imag + imp_power_to.imag
        # -- calculate ward results --
        # res_ward reports the TWO halves recombined, as pandapower does:
        #     p_mw = ps_mw + vm**2 * pz_mw ,  q_mvar = qs_mvar + vm**2 * qz_mvar
        # (_get_pq_results writes the constant-power part, then results_bus adds the
        # voltage-dependent impedance part on top). Note q uses +qz_mvar here: the sign
        # flip lives only in the Ybus stamp (BS = -qz_mvar), not in the reported demand.
        if "ward" in self._ybus_elements and "ward" in net and len(net.ward):
            net.res_ward = _ensure_index(net.res_ward, net.ward.index)
            wards = self._ybus_elements["ward"]
            vm_ward = vm[wards._from_bus]

            res_ward = net.res_ward
            res_ward["vm_pu"].to_numpy(copy=False)[:] = vm_ward
            res_ward["p_mw"].to_numpy(copy=False)[:] = wards._ps_mw + vm_ward**2 * wards._pz_mw
            res_ward["q_mvar"].to_numpy(copy=False)[:] = wards._qs_mvar + vm_ward**2 * wards._qz_mvar

        # -- unsupplied elements and switches --
        zero_unsupplied_flows(net, topo)
        if "switch" in net and len(net.switch):
            write_switch_results(net, topo, i_sw, voltage, net.sn_mva)

    def _pre_dc_solve(self, yBus: sparse, voltage: NDArray, Pinj: NDArray, ref, pvpq):
        # "DC" Lastfluss zur initialisierung
        # 1) Ykk ~ Ybus.imag
        # 2) S_inj berechnen
        # 3) Slack auf 1.0 setzen
        # 3.1) Slack spalte und zeile auf null setzen außer slackbus, der ist gleich 1
        # 4) Ykk\(S_inj') berechnen (wahrscheinlich ein spsolve)

        # Reduced DC system: B[pvpq,pvpq] * theta = Pinj[pvpq] - B[pvpq,ref] * theta_ref.
        # The reference-bus coupling must use the reference ANGLE (theta_ref), not
        # voltage.imag[ref] -- the old form used the imaginary part, which equals
        # vm*sin(va) and only approximates the angle for small va. With a nonzero slack
        # angle set-point it injects a wrong coupling term, skewing all DC angles.
        Bbus = yBus[pvpq.T, :][:, pvpq]
        theta_ref = np.angle(voltage[ref])
        ref_matrix = np.transpose(Pinj[pvpq] - yBus[pvpq.T, :][:, ref] * theta_ref)
        Va = np.real(spsolve(Bbus, ref_matrix))
        np.nan_to_num(Va, copy=False)
        return 1.0 * np.exp(1j * Va)

    def calculate(self, net: pandapowerNet, tolerance: float = 1e-5, max_iterations: int = 30, **kwargs):
        """
        This implementation uses a list approach. Therefore every "thing" in the network needs to be implemented as a
        function which returns a jacobi. This is for example relevant for FACTS devices, temperature dependent powerflow
        or distributed slack (since these elements actually change the jacobian).
        Loads, sgen, gen, ... are already calculated with the standard jacobian

        Args:
            tolerance:
            max_iterations:
            **kwargs: are passed directly to scipy.spsolve. ``voltage`` (start voltage) may be
                given per net.bus row or per node (see p3s.topology).

        Returns:
            The bus voltages per net.bus row (NaN on unsupplied buses; buses fused by closed
            bus-bus switches share their node's voltage).
        """
        initialize_pf = kwargs.pop("init", "dc")
        # Damped Newton (backtracking line search) in the C++ solver. Default on: it is
        # free on well-behaved grids (the full step is accepted, its residual reused as the
        # next convergence check) and rescues stiff grids where undamped Newton overshoots.
        # Pass line_search=False to restore the exact undamped hot path (benchmarking / A-B).
        line_search = kwargs.pop("line_search", True)

        # Flat-start fallback: when a DC-seeded solve fails to converge, retry once from a
        # flat start (default on for the dc path). The DC seed is actively harmful on some
        # stiff grids -- it can drag weak buses into a voltage-collapse pocket the Newton
        # then cannot escape -- while a flat 1.0pu/0deg start converges cleanly (e.g.
        # case145). See docs/damped_newton_plan.md. Pass flat_fallback=False to disable.
        flat_fallback = kwargs.pop("flat_fallback", True)

        # Voltage-band feasibility check: a Newton solve can CONVERGE (residual -> 0) to a
        # spurious, non-physical low-voltage branch -- e.g. a voltage-collapse solution with
        # buses at ~0.02 pu (case2848rte converges to such a branch with vm_err ~ 1.0). The
        # residual alone cannot tell it apart from the real operating point; the tell is that
        # voltages fall outside a physical band. So we ACCEPT a converged solve only when all
        # bus magnitudes lie in [vmin_pu, vmax_pu]; otherwise treat it like a non-convergence
        # and try the next seed. Mirrors the SAM feasibility-boundary logic. Set
        # voltage_band=None to disable the check (accept any converged solve).
        voltage_band = kwargs.pop("voltage_band", (0.5, 1.5))

        # initialize the voltage vector, either with a dcpf or simply with 1.0pu 0° aka flat
        # start. IMPORTANT: work on a COPY -- the DC block writes voltage[pvpq] in place, and
        # self._initial_voltage must stay a pristine flat start for the fallback (and reuse).
        voltage = np.array(kwargs.pop("voltage", self._initial_voltage), dtype=np.complex128)
        if len(voltage) != self._topology.n_node:
            voltage = self._topology.node_voltage(voltage)  # given per net.bus row
        flat_start = np.array(self._initial_voltage, dtype=np.complex128)

        pv: NDArray = self.busses["pv"]
        pq: NDArray = self.busses["pq"]
        pvpq = np.r_[pv, pq]

        if initialize_pf == "dc":
            # Capture PV magnitude set-points before the DC solve overwrites them:
            # _pre_dc_solve only provides angles and returns magnitude 1.0 for every
            # pvpq bus, which would clobber the known voltage-magnitude set-points at
            # PV buses. Since a PV bus holds |V| fixed during the solve, an unrestored
            # 1.0 here is never corrected and propagates a large error to neighbours.
            pv_vm = np.abs(voltage[pv]) if len(pv) > 0 else None
            # Same for PQ: the seed magnitude there is the mean generator set point
            # (see mean_setpoint_vm), not 1.0 pu, and _pre_dc_solve would reset it.
            pq_vm = np.abs(voltage[pq]) if len(pq) > 0 else None
            # DC init solves B*theta = P_inj. P_inj is the real bus power injection
            # (self._sBus.real) PLUS the transformer phase-shift injection
            # (self._p_shift). Two prior bugs are fixed here: (1) self._Bbus is ALREADY
            # the real susceptance matrix (make_ybus took .imag), so passing it directly
            # -- not self._Bbus.imag, which double-.imag'd to all zeros -> singular DC
            # -> flat start; (2) the RHS previously used only _p_shift, which is zero for
            # nets without phase shifters, so DC angles collapsed to 0 (a flat start) and
            # the real loads were ignored.
            voltage[pvpq] = self._pre_dc_solve(
                yBus=self._Bbus,
                voltage=voltage,
                Pinj=self._sBus.real + self._p_shift,
                ref=self.busses["ref"],
                pvpq=pvpq,
            )
            # Restore PV and PQ magnitudes (keep the DC-estimated angle).
            if len(pv) > 0:
                voltage[pv] = pv_vm * np.exp(1j * np.angle(voltage[pv]))
            if len(pq) > 0:
                voltage[pq] = pq_vm * np.exp(1j * np.angle(voltage[pq]))

        # CSR arrays of Ybus, passed zero-copy into the C++ solver (no .tolist()).
        # scipy guarantees C-contiguous indptr/indices/data, and the dtype casts
        # below are no-ops when the arrays already have the target dtype.
        Yp = np.ascontiguousarray(self._YBus.indptr, dtype=np.int32)
        Yj = np.ascontiguousarray(self._YBus.indices, dtype=np.int32)
        Yx = np.ascontiguousarray(self._YBus.data, dtype=np.complex128)
        pv_i = np.ascontiguousarray(pv, dtype=np.int32)
        pq_i = np.ascontiguousarray(pq, dtype=np.int32)
        Sbus = np.ascontiguousarray(self._sBus, dtype=np.complex128)

        # Build (or reuse) the cached solver. The KLU symbolic analyze depends
        # only on the topology (Yp/Yj/pv/pq), so it is done once and reused while
        # the Ybus structure is unchanged.
        nnz = Yj.shape[0]
        if self._cpp_solver is None or self._cpp_solver_nnz != nnz:
            self._cpp_solver = nr_klu.Solver(Yp, Yj, Yx, pv_i, pq_i)
            self._cpp_solver_nnz = nnz
        else:
            # topology unchanged, but Ybus values may have changed (e.g. taps);
            # refresh them cheaply without redoing the KLU symbolic analyze.
            self._cpp_solver.update_Y(Yx)

        # Try each start voltage in turn, accepting only a CONVERGED and PHYSICALLY VALID
        # (in-band) solution. Seeds: the primary start (DC-seeded or user-supplied), then a
        # flat start as a fallback (a DC seed can land in a collapse basin OR converge to a
        # spurious low-voltage branch; flat often lands in the real basin). The flat retry is
        # only meaningful when the primary seed was different from flat, i.e. the dc path.
        seeds = [np.ascontiguousarray(voltage, dtype=np.complex128)]
        if flat_fallback and initialize_pf == "dc":
            seeds.append(np.ascontiguousarray(flat_start, dtype=np.complex128))

        def _in_band(v: NDArray) -> bool:
            if voltage_band is None:
                return True
            vm = np.abs(v)
            return bool(np.all(vm >= voltage_band[0]) and np.all(vm <= voltage_band[1]))

        result = None
        for V0_seed in seeds:
            result = self._cpp_solver.solve(
                Sbus, V0_seed, max_iter=max_iterations, tol=tolerance, line_search=line_search
            )
            if result["converged"] and _in_band(result["V"]):
                return self._topology.bus_voltage(result["V"])  # converged AND physical -> accept

        # No seed produced a converged, in-band solution. Distinguish the two failure modes
        # for a useful message: a converged-but-out-of-band result is a spurious/collapsed
        # branch, not a non-convergence.
        if result is not None and result["converged"]:
            vm = np.abs(result["V"])
            raise LoadflowNotConverged(
                f"Loadflow converged to a non-physical solution: bus voltages in "
                f"[{vm.min():.3f}, {vm.max():.3f}] pu fall outside the accepted band "
                f"{voltage_band}. Likely a voltage-collapse / spurious branch."
            )
        raise LoadflowNotConverged(
            f"Loadflow did not converge in {max_iterations} iterations "
            f"(reached {result['iterations'] if result is not None else 0})."
        )

    def calculate_timeseries_cpp(
        self,
        net: pandapowerNet,
        timeseries: dict,
        tolerance: float = 1e-8,
        max_iterations: int = 30,
        n_threads: int = 0,
    ):
        """Batched (time-series) Newton-Raphson via the C++/KLU ``solve_batch``.

        Every time step is an independent Newton solve on the *same* grid topology, so
        the KLU symbolic analyze is done once and shared; the C++ batch then solves all
        steps (optionally across CPU cores via OpenMP).

        Mirrors ``NewtonPowerflowCuda.calculate_timeseries_cuda``: the same Sbus delta
        matrix + DC-init are reused (``p3s.timeseries``), so results match the GPU
        path and pandapower per step.

        Parameters
        ----------
        timeseries : dict[(element, var) -> (n_element, T) array]
            Per-element time-series of ``p_mw`` (always) and ``q_mvar`` (optional);
            see ``p3s.simbench.get_load_gen_matrix.extract_timeseries``.
        n_threads : int, default 0
            OpenMP worker count for the batch (0 = all cores, 1 = serial).

        Returns
        -------
        voltages : complex128 (n_bus, T) -- one converged voltage vector per time step, per
            net.bus row (NaN on unsupplied buses).
        """
        sbus_matrix = build_sbus_matrix(self, net, timeseries)  # (n_node, T)
        n_bus, T = sbus_matrix.shape

        if n_bus == 0:
            raise ValueError("Network has no buses")
        if T == 0:
            raise ValueError("Time series has zero time steps")
        if np.any(np.isnan(sbus_matrix)) or np.any(np.isinf(sbus_matrix)):
            raise ValueError("Sbus matrix contains NaN or inf values")

        v_init = dc_initial_voltage(self)

        if np.any(np.isnan(v_init)) or np.any(np.isinf(v_init)):
            raise ValueError("DC initial voltage contains NaN or inf values")

        Yp = np.ascontiguousarray(self._YBus.indptr, dtype=np.int32)
        Yj = np.ascontiguousarray(self._YBus.indices, dtype=np.int32)
        Yx = np.ascontiguousarray(self._YBus.data, dtype=np.complex128)
        pv_i = np.ascontiguousarray(self.busses["pv"], dtype=np.int32)
        pq_i = np.ascontiguousarray(self.busses["pq"], dtype=np.int32)

        if not Yp.flags.c_contiguous or Yp.dtype != np.int32:
            Yp = np.ascontiguousarray(Yp, dtype=np.int32)
        if not Yj.flags.c_contiguous or Yj.dtype != np.int32:
            Yj = np.ascontiguousarray(Yj, dtype=np.int32)
        if not Yx.flags.c_contiguous or Yx.dtype != np.complex128:
            Yx = np.ascontiguousarray(Yx, dtype=np.complex128)

        nnz = Yj.shape[0]
        if self._cpp_solver is None or self._cpp_solver_nnz != nnz:
            self._cpp_solver = nr_klu.Solver(Yp, Yj, Yx, pv_i, pq_i)
            self._cpp_solver_nnz = nnz
        else:
            self._cpp_solver.update_Y(Yx)

        sbus_contiguous = np.ascontiguousarray(sbus_matrix, dtype=np.complex128)
        v_init_contiguous = np.ascontiguousarray(v_init, dtype=np.complex128)

        if sbus_contiguous.shape[0] != n_bus or sbus_contiguous.shape[1] != T:
            raise ValueError(f"Sbus shape mismatch: expected ({n_bus}, {T}), got {sbus_contiguous.shape}")
        if v_init_contiguous.shape[0] != n_bus:
            raise ValueError(f"V0 shape mismatch: expected ({n_bus},), got {v_init_contiguous.shape}")

        result = self._cpp_solver.solve_batch(
            sbus_contiguous,
            v_init_contiguous,
            max_iter=max_iterations,
            tol=tolerance,
            n_threads=n_threads,
        )

        if np.any(np.isnan(result["V"])) or np.any(np.isinf(result["V"])):
            raise ValueError("C++ batch solver returned NaN or inf voltages")

        if not bool(np.all(result["converged"])):
            n_bad = int((~result["converged"]).sum())
            raise LoadflowNotConverged(
                f"C++ batch did not converge for {n_bad} of {T} time steps in {max_iterations} iterations."
            )
        return self._topology.bus_voltage(result["V"])
