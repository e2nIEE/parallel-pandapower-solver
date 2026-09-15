# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandapower import LoadflowNotConverged, pandapowerNet
from scipy.sparse import csr_matrix as sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from p3s.models.ShuntModel import ShuntModel
from p3s.models.ThreePort import ThreePort
from p3s.models.ThreeWindingTransformerModel import ThreeWindingTransformerModel
from p3s.models.TransmissionLineModel import TransmissionLineModel
from p3s.models.TwoPort import TwoPort
from p3s.models.TwoWindingTransformerModel import TwoWindingTransformerModel
from p3s.PowerflowObject import PowerflowObject
from p3s.PQPVPowerflow import PQPVPowerflow


class NewtonPowerflow:
    def __init__(self, net: pandapowerNet):
        self.pf_objects: dict[str, PowerflowObject] = {}
        self._sBus: NDArray = None
        self._YBus: sparse = None
        self._Bbus: sparse = None
        self._p_shift: NDArray = None
        self._lookup: pd.Series | None = None
        self._reverse_lookup: pd.Series | None = None
        self._initial_voltage: NDArray = None
        # keeping a reference on all classes which create the ybus.
        self._ybus_elements: dict[str, TwoPort | ThreePort] = {}
        self.busses: dict = {}
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
            trafos = ThreeWindingTransformerModel(
                net.trafo3w, net.bus, sn_mva=net.sn_mva, tap_table=net.trafo_characteristic_table
            )
            self._ybus_elements["trafo3w"] = trafos
            Ybus_trafos3w = trafos.create_y_matrix(n_bus=n_bus)
            Ybus_dat.extend(Ybus_trafos3w.data)
            Ybus_row.extend(Ybus_trafos3w.row)
            Ybus_col.extend(Ybus_trafos3w.col)

        if "shunt" in net and len(net.shunt) > 0:
            shunts = ShuntModel(net.shunt, sn_mva=net.sn_mva)
            self._ybus_elements["shunt"] = shunts
            Ybus_shunts = shunts.create_y_matrix(n_bus=n_bus)
            Ybus_dat.extend(Ybus_shunts.data)
            Ybus_row.extend(Ybus_shunts.row)
            Ybus_col.extend(Ybus_shunts.col)

        Ybus = sparse((Ybus_dat, (Ybus_row, Ybus_col)), shape=(n_bus, n_bus))
        Bbus = sparse((Bbus_dat, (Bbus_row, Bbus_col)), shape=(n_bus, n_bus)).imag
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
        # self._sBus = (sBus - self._sBus.imag * 1j) * net.sn_mva

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

            # loading_percent = np.maximum(trafo_currents_from.values * res_trafo['vm_hv_pu'].values,
            #                             trafo_currents_to.values * res_trafo['vm_lv_pu'].values)

            loading_percent = np.maximum(
                trafo_currents_from.values * trafos.voltages_from.values * np.sqrt(3),
                trafo_currents_to.values * trafos.voltages_to.values * np.sqrt(3),
            )
            res_trafo["loading_percent"].to_numpy(copy=False)[:] = loading_percent / net.trafo.sn_mva * 100

        # -- calculate gen results --
        net.res_gen = _ensure_index(net.res_gen, net.gen.index)
        # Only nets with generators do the gen-result maths: "_lookup" is added to net.gen
        # by _setup_pf solely when len(net.gen) > 0, so on a PQ-only net (SAM's usual case)
        # net.gen exists but has no "_lookup" column. Guard the whole block (the ext_grid
        # results below must still run).
        if len(net.gen) > 0:
            gen_lookup = net.gen["_lookup"].to_numpy().astype(int)
            res_gen = net.res_gen

            # vm/va are per-bus quantities -> broadcast to every gen on that bus. Index the
            # per-bus results by each gen's bus (gen_lookup) rather than by the distinct
            # PV-bus list (_pv): when several gens share a bus, res_gen has one row per gen
            # while _pv has one entry per bus, so a per-bus assignment would shape-mismatch.
            bus_vm = net.res_bus["vm_pu"].to_numpy(copy=False)
            bus_va = net.res_bus["va_degree"].to_numpy(copy=False)
            res_gen["vm_pu"].to_numpy(copy=False)[:] = bus_vm[gen_lookup]
            res_gen["va_degree"].to_numpy(copy=False)[:] = bus_va[gen_lookup]

            # direct copy
            res_gen["p_mw"].to_numpy(copy=False)[:] = net.gen.p_mw

            # Total Q injected at each bus (per-bus), then distributed across the gens on
            # that bus. q_bus[b] is the whole bus reactive injection; q_bus[gen_lookup]
            # broadcasts that bus total onto every gen at the bus (== Qg_tot in pfsoln).
            q_bus = -1 * net.res_bus["q_mvar"].to_numpy() - self._sBus.imag * net.sn_mva  # per-bus (n_bus,)
            q_tot = q_bus[gen_lookup]  # per-gen: total Q of that gen's bus

            # Distribute the bus Q across its gens in proportion to each gen's reactive
            # range, matching pandapower/PYPOWER pfsoln._update_q:
            #   Qg[i] = Qmin[i] + (Qg_tot - Qmin_tot)/(Qmax_tot - Qmin_tot + EPS)*(Qmax-Qmin)
            # where *_tot are the per-bus sums over the bus's gens. A bare equal split
            # (Qtot/n) is wrong whenever the gens have unequal Q ranges (e.g.
            # GBreducednetwork). For buses with zero total range the proportional term
            # collapses to ~0, so fall back to the equal split there (matches PYPOWER).
            q_min = net.gen.get("min_q_mvar")
            q_max = net.gen.get("max_q_mvar")
            if q_min is not None and q_max is not None:
                q_min = q_min.to_numpy(dtype=float)
                q_max = q_max.to_numpy(dtype=float)
                qmin_tot = np.bincount(gen_lookup, weights=q_min, minlength=len(net.bus))[gen_lookup]
                qmax_tot = np.bincount(gen_lookup, weights=q_max, minlength=len(net.bus))[gen_lookup]
                eps = np.finfo(float).eps
                q_gen = q_min + (q_tot - qmin_tot) / (qmax_tot - qmin_tot + eps) * (q_max - q_min)
                # zero-range buses: fall back to an equal split (proportional term ~0 there)
                zero_range = np.isclose(qmax_tot, qmin_tot)
                if zero_range.any():
                    gens_per_bus = np.bincount(gen_lookup, minlength=len(net.bus))[gen_lookup]
                    q_gen = np.where(zero_range, q_tot / gens_per_bus, q_gen)
            else:
                # no reactive-limit columns -> equal split is the only defensible convention
                gens_per_bus = np.bincount(gen_lookup, minlength=len(net.bus))[gen_lookup]
                q_gen = q_tot / gens_per_bus
            res_gen["q_mvar"].to_numpy(copy=False)[:] = q_gen

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

        Bbus = yBus[pvpq.T, :][:, pvpq]
        ref_matrix = np.transpose(Pinj[pvpq] - yBus[pvpq.T, :][:, ref] * voltage.imag[ref])

        # The DC init matrix can be singular (e.g. purely radial, line-only feeders whose
        # susceptance-derived Bbus is rank-deficient). In that case spsolve emits a
        # MatrixRankWarning and returns NaNs; fall back to a flat start (Va = 0) instead
        # of masking NaNs after the fact, which also avoids a wasted singular solve.
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            try:
                Va = np.real(spsolve(Bbus, ref_matrix))
            except (MatrixRankWarning, RuntimeError):
                Va = np.zeros(len(pvpq))
        if not np.all(np.isfinite(Va)):
            Va = np.zeros(len(pvpq))
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

        # initialize the voltage vector, either with a dcpf or simply with 1.0pu 0° aka flat start
        voltage = kwargs.pop("voltage", self._initial_voltage)

        if initialize_pf == "dc":
            pvpq = np.r_[self.busses["pv"], self.busses["pq"]]
            pv = self.busses["pv"]
            # Capture PV magnitude set-points before the DC solve overwrites them
            # (voltage may alias self._initial_voltage, which _pre_dc_solve writes into).
            pv_vm = np.abs(voltage[pv]) if len(pv) > 0 else None
            # DC init: B*theta = P_inj (real bus injection + phase-shift injection).
            # _Bbus is already real susceptance (pass directly, not .imag -> would
            # double-.imag to zeros); RHS must include _sBus.real, not just _p_shift.
            voltage[pvpq] = self._pre_dc_solve(
                yBus=self._Bbus,
                voltage=voltage,
                Pinj=self._sBus.real + self._p_shift,
                ref=self.busses["ref"],
                pvpq=pvpq,
            )
            # The DC init only provides angles; it returns magnitude 1.0 for all pvpq
            # buses, which would clobber the known voltage-magnitude set-points at PV
            # buses. Restore PV magnitudes (keep the DC-estimated angle).
            if len(pv) > 0:
                voltage[pv] = pv_vm * np.exp(1j * np.angle(voltage[pv]))

        i = 0
        converged = False

        # calculate the first batch of mismatches based on the first assumptions in the network
        mismatch = self._calc_mismatch(self._sBus, voltage)

        # the actual newton raphson loop
        while not converged:
            i += 1
            if i > max_iterations:
                raise LoadflowNotConverged(f"Loadflow did not converge in {max_iterations} iterations.")

            # TODO: When FACT devices are being implemented, the jacobi needs to be expanded
            # J_partial = []
            # offset = 0
            # for pf_object in self.pf_objects.values():
            #     J = pf_object.create_J(voltage)
            #     pf_object.offset = offset
            #     offset += J.shape[1]
            #     J_partial.append(J)
            #
            # if len(J_partial) > 1:
            #     J = block_diag(J_partial)
            # else:
            #     J = J_partial[0]
            # N = J.shape[0]

            Jx, Jp, Jj = self.pf_objects["PVPQ"].create_J(voltage)

            J = sparse((Jx, Jj, Jp))
            dx = -1 * spsolve(J, mismatch, **kwargs)

            voltage = self._calc_results_dx(dx, voltage)

            mismatch = self._calc_mismatch(self._sBus, voltage)

            converged = np.linalg.norm(mismatch, np.inf) < tolerance

        # -- eval --
        net["_ppc"] = {"iterations": i}
        self._parse_results(net, voltage)
