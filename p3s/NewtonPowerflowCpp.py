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

from p3s.models.ShuntModel import ShuntModel
from p3s.models.ThreeWindingTransformerModel import ThreeWindingTransformerModel
from p3s.models.TransmissionLineModel import TransmissionLineModel
from p3s.models.TwoWindingTransformerModel import TwoWindingTransformerModel
from p3s.models.WardModel import WardModel
from p3s.PowerflowObject import PowerflowObject
from p3s.PQPVPowerflow import PQPVPowerflow
from p3s.timeseries import build_sbus_matrix, dc_initial_voltage, mean_setpoint_vm

# Fast C++ Newton-Raphson solver (polar formulation, KLU linear solve). Installed into
# the p3s package by `pip install p3s[cpp]` (CMake / scikit-build-core; see
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
            self._ybus_elements["trafo3w"] = trafo3ws
            Ybus_trafos3w = trafo3ws.create_y_matrix(n_bus=n_bus)
            Ybus_dat.extend(Ybus_trafos3w.data)
            Ybus_row.extend(Ybus_trafos3w.row)
            Ybus_col.extend(Ybus_trafos3w.col)

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
            shunts = ShuntModel(net.shunt, sn_mva=net.sn_mva)
            self._ybus_elements["shunt"] = shunts
            Ybus_shunts = shunts.create_y_matrix(n_bus=n_bus)
            Ybus_dat.extend(Ybus_shunts.data)
            Ybus_row.extend(Ybus_shunts.row)
            Ybus_col.extend(Ybus_shunts.col)

        Ybus = coo_matrix((Ybus_dat, (Ybus_row, Ybus_col)), shape=(n_bus, n_bus)).tocsr()
        Bbus = coo_matrix((Bbus_dat, (Bbus_row, Bbus_col)), shape=(n_bus, n_bus)).tocsr().imag
        return Ybus, Bbus

    def _setup_pf(self, net: pandapowerNet):
        sBus: pd.Series = pd.Series(data=np.zeros(shape=len(net.bus)), index=range(len(net.bus)), dtype=complex)
        self._p_shift = np.zeros(len(net.bus))

        if self._YBus is None:
            self._YBus, self._Bbus = self.make_ybus(net)

        # Lookup tables, since the Ybus is 0 indexed. Lookup is from bus to ybus index.
        # Reverse_lookup is the other way around.
        self._reverse_lookup = pd.Series(net.bus.index)
        self._lookup = pd.Series(index=self._reverse_lookup.values, data=self._reverse_lookup.index.values)

        # create an initial voltage vector, for flat start (1pu, 0°) and overwrite it with setpoints of gen's
        self._initial_voltage = np.ones(shape=len(net.bus), dtype=np.complex128)

        # all buses are pq buses by default
        pq = self._lookup.values
        pv = np.ndarray(shape=0, dtype=int)
        ref = np.ndarray(shape=0, dtype=int)

        if "load" in net and len(net["load"]) > 0:
            net.load["_lookup"] = self._lookup[net.load.bus].values.astype(int)
            res_mw = (net.load.p_mw + net.load.q_mvar * 1j) * net.load.scaling * net.load.in_service

            _load = res_mw.groupby(net.load["_lookup"]).sum()
            sBus = sBus.add(_load, fill_value=0)

        if "sgen" in net and len(net["sgen"]) > 0:
            net.sgen["_lookup"] = self._lookup[net.sgen.bus].values.astype(int)
            res_mw = (net.sgen.p_mw + net.sgen.q_mvar * 1j) * net.sgen.scaling * net.sgen.in_service

            _sgen = -1.0 * res_mw.groupby(net.sgen["_lookup"]).sum()
            sBus = sBus.add(_sgen, fill_value=0)

        if "gen" in net and len(net["gen"]) > 0:
            net.gen["_lookup"] = self._lookup[net.gen.bus].values.astype(int)

            res_mw = net.gen.p_mw * net.gen.scaling * net.gen.in_service
            _gen = -1.0 * res_mw.groupby(net.gen["_lookup"]).sum()
            sBus = sBus.add(_gen, fill_value=0)

            # one PV bus per distinct gen bus (multiple gens can share a bus)
            vm_by_bus = net.gen.groupby("_lookup").vm_pu.first()  # or .mean() / .max()
            pv = vm_by_bus.index.values
            self._initial_voltage[pv] = vm_by_bus.values

            # If a bus has a generator, it will be changed to a pv bus.
            pq = np.setdiff1d(pq, pv)

        if "ext_grid" in net and len(net["ext_grid"]) > 0:
            net.ext_grid["_lookup"] = self._lookup[net.ext_grid.bus].values.astype(int)
            ref = net.ext_grid["_lookup"].values

            # remove pv busses from pq, since pv busses are handled differently
            pq = np.setdiff1d(pq, ref)

            # and add the vm and va set point to the initial voltage vector
            self._initial_voltage[ref] = net.ext_grid.vm_pu * np.exp(np.deg2rad(net.ext_grid.va_degree) * 1j)

        # Seed the remaining PQ buss at the mean generator / ext_grid set point rather than a flat 1.0 pu.
        # This is what pandapower's init = "auto" does, and it saves Newton iterations.
        if len(pq) > 0:
            self._initial_voltage[pq] = mean_setpoint_vm(net)

        # Constant-power half of the ward equivalents. The shunt half is already in Ybus
        # (see make_ybus); this adds ps/qs as an ordinary PQ demand. A ward has no
        # scaling column -- pandapower hardcodes scaling = 1.0 for ward/xward -- and
        # out-of-service wards were zeroed when the model was built.
        if "ward" in self._ybus_elements:
            sBus = sBus.add(pd.Series(self._ybus_elements["ward"].s_bus), fill_value=0)

        self._sBus = -1.0 * sBus.values / net.sn_mva  # type: ignore[operator]
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

        sBus = np.conj(self._YBus * voltage) * voltage * net.sn_mva

        # -- calculate bus results --
        vm = np.abs(voltage)
        va = np.angle(voltage, deg=True)

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
            vm_from_pu = vm[lines._from_bus]
            line_power_from = np.conj(lines.yf_matrix * voltage) * voltage[lines._from_bus] * net.sn_mva
            line_currents_from = np.abs(line_power_from / (lines.voltages * vm_from_pu * np.sqrt(3)))

            vm_to_pu = vm[lines._to_bus]
            line_power_to = np.conj(lines.yt_matrix * voltage) * voltage[lines._to_bus] * net.sn_mva
            line_currents_to = np.abs(line_power_to / (lines.voltages * vm_to_pu * np.sqrt(3)))

            res_line = net.res_line
            res_line["p_from_mw"].to_numpy(copy=False)[:] = line_power_from.real
            res_line["q_from_mvar"].to_numpy(copy=False)[:] = line_power_from.imag
            res_line["i_from_ka"].to_numpy(copy=False)[:] = line_currents_from
            res_line["vm_from_pu"].to_numpy(copy=False)[:] = vm_from_pu
            res_line["va_from_degree"].to_numpy(copy=False)[:] = va[lines._from_bus]

            res_line["p_to_mw"].to_numpy(copy=False)[:] = line_power_to.real
            res_line["q_to_mvar"].to_numpy(copy=False)[:] = line_power_to.imag
            res_line["i_to_ka"].to_numpy(copy=False)[:] = line_currents_to
            res_line["vm_to_pu"].to_numpy(copy=False)[:] = vm_to_pu
            res_line["va_to_degree"].to_numpy(copy=False)[:] = va[lines._to_bus]

            line_currents_max = np.maximum(line_currents_from, line_currents_to)
            res_line["i_ka"].to_numpy(copy=False)[:] = line_currents_max
            res_line["pl_mw"].to_numpy(copy=False)[:] = np.abs(
                np.abs(line_power_to.real) - np.abs(line_power_from.real)
            )
            res_line["ql_mvar"].to_numpy(copy=False)[:] = np.abs(
                np.abs(line_power_to.imag) - np.abs(line_power_from.imag)
            )

            max_i_ka = net.line["max_i_ka"].to_numpy(copy=False)
            res_line["loading_percent"].to_numpy(copy=False)[:] = line_currents_max / max_i_ka * 100.0

        if "trafo" in self._ybus_elements and "trafo" in net and len(net.trafo):
            net.res_trafo = _ensure_index(net.res_trafo, net.trafo.index)
            trafos = self._ybus_elements["trafo"]
            vm_hv_pu = res_bus["vm_pu"][net.trafo.hv_bus]
            trafo_power_from = np.conj(trafos.yf_matrix * voltage) * voltage[trafos._from_bus] * net.sn_mva
            trafo_currents_from = np.abs(trafo_power_from / (trafos.voltages_from * vm_hv_pu * np.sqrt(3)))

            vm_lv_pu = res_bus["vm_pu"][net.trafo.lv_bus]
            trafo_power_to = np.conj(trafos.yt_matrix * voltage) * voltage[trafos._to_bus] * net.sn_mva
            trafo_currents_to = np.abs(trafo_power_to / (trafos.voltages_to * vm_lv_pu * np.sqrt(3)))

            res_trafo = net.res_trafo
            res_trafo["p_hv_mw"].to_numpy(copy=False)[:] = trafo_power_from.real
            res_trafo["q_hv_mvar"].to_numpy(copy=False)[:] = trafo_power_from.imag
            res_trafo["i_hv_ka"].to_numpy(copy=False)[:] = trafo_currents_from
            res_trafo["vm_hv_pu"].to_numpy(copy=False)[:] = vm_hv_pu
            res_trafo["va_hv_degree"].to_numpy(copy=False)[:] = res_bus["va_degree"][net.trafo.hv_bus]

            res_trafo["p_lv_mw"].to_numpy(copy=False)[:] = trafo_power_to.real
            res_trafo["q_lv_mvar"].to_numpy(copy=False)[:] = trafo_power_to.imag
            res_trafo["i_lv_ka"].to_numpy(copy=False)[:] = trafo_currents_to
            res_trafo["vm_lv_pu"].to_numpy(copy=False)[:] = vm_lv_pu
            res_trafo["va_lv_degree"].to_numpy(copy=False)[:] = res_bus["va_degree"][net.trafo.lv_bus]

            res_trafo["pl_mw"].to_numpy(copy=False)[:] = trafo_power_from.real + trafo_power_to.real
            res_trafo["ql_mvar"].to_numpy(copy=False)[:] = trafo_power_from.imag + trafo_power_to.imag

            loading_percent = np.maximum(
                trafo_currents_from.values * trafos.voltages_from.values * np.sqrt(3),
                trafo_currents_to.values * trafos.voltages_to.values * np.sqrt(3),
            )
            res_trafo["loading_percent"].to_numpy(copy=False)[:] = loading_percent / net.trafo.sn_mva * 100

        # -- calculate impedance results --
        # res_impedance has no vm_*/va_*/loading_percent columns (an impedance carries no
        # rating), so this is the short form of the line block. Losses are the SUM of both
        # terminal flows -- pandapower's _get_impedance_results uses pl = p_from + p_to,
        # the res_trafo convention, not res_line's absolute difference.
        if "impedance" in self._ybus_elements and "impedance" in net and len(net.impedance):
            net.res_impedance = _ensure_index(net.res_impedance, net.impedance.index)
            impedances = self._ybus_elements["impedance"]

            imp_power_from = np.conj(impedances.yf_matrix * voltage) * voltage[impedances._from_bus] * net.sn_mva
            imp_power_to = np.conj(impedances.yt_matrix * voltage) * voltage[impedances._to_bus] * net.sn_mva

            # Each terminal is referred to its OWN base voltage: an impedance may span a
            # voltage step, unlike a line.
            imp_currents_from = np.abs(
                imp_power_from / (impedances.voltages_from * vm[impedances._from_bus] * np.sqrt(3))
            )
            imp_currents_to = np.abs(imp_power_to / (impedances.voltages_to * vm[impedances._to_bus] * np.sqrt(3)))

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

        # -- calculate gen results --
        if "gen" in self._ybus_elements and "gen" in net and len(net.gen):
            net.res_gen = _ensure_index(net.res_gen, net.gen.index)
            gen_bus = net.res_bus.loc[self.pf_objects["PVPQ"]._pv]
            gen_lookup = net.gen["_lookup"].to_numpy().astype(int)
            res_gen = net.res_gen

            # vm/va are per-bus quantities -> broadcast to every gen on that bus
            res_gen["vm_pu"].to_numpy(copy=False)[:] = gen_bus.vm_pu.to_numpy(copy=False)
            res_gen["va_degree"].to_numpy(copy=False)[:] = gen_bus.va_degree.to_numpy(copy=False)

            # direct copy
            res_gen["p_mw"].to_numpy(copy=False)[:] = net.gen.p_mw

            # total Q injected at each PV bus (per-bus), then split across gens on that bus
            q_bus = -1 * net.res_bus["q_mvar"].to_numpy() - self._sBus.imag * net.sn_mva  # per-bus (n_bus,)
            # equal split: divide each bus's Q by the number of gens on it
            gens_per_bus = np.bincount(gen_lookup, minlength=len(net.bus))
            res_gen["q_mvar"].to_numpy(copy=False)[:] = q_bus[gen_lookup] / gens_per_bus[gen_lookup]

        # -- calculate ext_grid results --
        net.res_ext_grid = _ensure_index(net.res_ext_grid, net.ext_grid.index)
        ext_grid_power = (sBus - self._sBus * net.sn_mva)[net.ext_grid.bus]
        res_ext = net.res_ext_grid
        res_ext["p_mw"].to_numpy(copy=False)[:] = ext_grid_power.real
        res_ext["q_mvar"].to_numpy(copy=False)[:] = ext_grid_power.imag

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
            **kwargs: are passed directly to scipy.spsolve

        Returns:

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
                return result["V"]  # converged AND physical -> accept

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
        voltages : complex128 (n_bus, T) -- one converged voltage vector per time step.
        """
        sbus_matrix = build_sbus_matrix(self, net, timeseries)  # (n_bus, T)
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
        return result["V"]
