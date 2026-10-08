# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix as sparse


class TwoPort:
    def __init__(self):
        self._both_open: NDArray | None = None
        self._open_from_gain: NDArray | None = None
        self._open_to_gain: NDArray | None = None
        self._p_shift_inj: NDArray | None = None
        self.p_shift: NDArray | None = None
        self._from_bus = []
        self._to_bus = []
        self._Y_ff = []
        self._Y_ft = []
        self._Y_tf = []
        self._Y_tt = []
        self._DC_Yff = []
        self._DC_Yft = []
        self._DC_Ytf = []
        self._DC_Ytt = []

        self.y_matrix: sparse | None = None
        self.yf_matrix: sparse | None = None
        self.yt_matrix: sparse | None = None

        self.y_dc_matrix: sparse | None = None
        self._n_bus: int | None = None

    def _apply_in_service(self, element_table) -> NDArray:
        """Zero this element's branch stamps wherever ``in_service`` is False.

        An out-of-service branch is electrically absent: it must contribute nothing to
        Ybus, to the DC B-matrix, or to the SAM blocks. The stamps are *masked in place*
        rather than the rows dropped, so ``_from_bus``/``_to_bus`` and the per-branch
        arrays stay row-aligned with the element table -- ``yf_matrix``/``yt_matrix``
        index by element row to fill ``res_line``/``res_trafo``, and a de-energised branch
        should report zero flow rather than shift every subsequent row's results.

        Returns the boolean in-service mask so callers can apply it to any additional
        per-branch quantity they build (e.g. the trafo phase-shift injection).
        """
        if "in_service" not in element_table:
            return np.ones(len(np.asarray(self._from_bus)), dtype=bool)

        in_service = np.asarray(element_table["in_service"].values, dtype=bool)
        if in_service.all():
            return in_service

        for attr in ("_Y_ff", "_Y_ft", "_Y_tf", "_Y_tt", "_DC_Yff", "_DC_Yft", "_DC_Ytf", "_DC_Ytt"):
            values = getattr(self, attr, None)
            if values is None or len(np.shape(values)) == 0:
                continue
            values = np.asarray(values)
            if values.shape[:1] != in_service.shape:
                continue
            # np.where (not in-place) because pi_model/t_model may hand back broadcast or
            # shared arrays -- e.g. TransmissionLineModel aliases _DC_Yff to _DC_Ytt.
            setattr(self, attr, np.where(in_service, values, 0.0))

        return in_service

    def apply_open_ends(self, open_ends: NDArray) -> None:
        """Disconnect branch ends whose switch is open; ``open_ends`` is (n_branch, 2) bool (from, to).

        As in pandapower, which hangs the open end on an auxiliary bus, the branch stays energised
        from its other end: the open terminal (no injection) is Kron-eliminated from the stamp,
            Y_ff' = Y_ff - Y_ft * Y_tf / Y_tt     (to end open; from end analogous)
        so line charging / magnetising remain while no current passes the open end. Both ends
        open leaves nothing. The open-end voltage is kept for the results (``end_voltages``).
        """
        open_ends = np.asarray(open_ends, dtype=bool)
        if not open_ends.any():
            return
        n = len(np.asarray(self._from_bus))
        open_f, open_t = open_ends[:, 0], open_ends[:, 1]

        def _stamps(names):
            return [np.broadcast_to(np.asarray(getattr(self, a), dtype=complex), (n,)).copy() for a in names]

        def _eliminate(ff, ft, tf, tt):
            with np.errstate(divide="ignore", invalid="ignore"):
                ff_red = np.where(tt != 0, ff - ft * tf / tt, ff)
                tt_red = np.where(ff != 0, tt - tf * ft / ff, tt)
            ff_new = np.where(open_f, 0.0, np.where(open_t, ff_red, ff))
            tt_new = np.where(open_t, 0.0, np.where(open_f, tt_red, tt))
            either = open_f | open_t
            return ff_new, np.where(either, 0.0, ft), np.where(either, 0.0, tf), tt_new

        ac = _stamps(("_Y_ff", "_Y_ft", "_Y_tf", "_Y_tt"))
        # open-end voltage from the closed end: V_t = -Y_tf / Y_tt * V_f, V_f = -Y_ft / Y_ff * V_t
        with np.errstate(divide="ignore", invalid="ignore"):
            self._open_to_gain = np.where(open_t & ~open_f, -ac[2] / ac[3], np.nan)
            self._open_from_gain = np.where(open_f & ~open_t, -ac[1] / ac[0], np.nan)
        self._both_open = open_f & open_t
        self._Y_ff, self._Y_ft, self._Y_tf, self._Y_tt = _eliminate(*ac)
        dc_names = ("_DC_Yff", "_DC_Yft", "_DC_Ytf", "_DC_Ytt")
        if all(len(np.shape(getattr(self, a, []))) for a in dc_names):
            self._DC_Yff, self._DC_Yft, self._DC_Ytf, self._DC_Ytt = _eliminate(*_stamps(dc_names))

        # a disconnected end carries no flow, so no DC phase-shift injection either
        p_shift_inj = getattr(self, "_p_shift_inj", None)
        if p_shift_inj is not None:
            self._p_shift_inj = np.where(open_f | open_t, 0.0, p_shift_inj)
            self.p_shift = np.zeros_like(self.p_shift)
            np.add.at(self.p_shift, np.asarray(self._from_bus, dtype=np.intp), self._p_shift_inj)
            np.add.at(self.p_shift, np.asarray(self._to_bus, dtype=np.intp), -self._p_shift_inj)
        self.y_matrix = self.y_dc_matrix = None

    def end_voltages(self, voltage: NDArray) -> tuple[NDArray, NDArray]:
        """Voltage at the from / to end of every branch; at an open end the voltage of the
        disconnected terminal (pandapower's auxiliary bus), NaN when both ends are open."""
        fb = np.asarray(self._from_bus, dtype=np.intp)
        tb = np.asarray(self._to_bus, dtype=np.intp)
        v_f, v_t = voltage[fb].astype(complex), voltage[tb].astype(complex)
        gain_t = getattr(self, "_open_to_gain", None)
        if gain_t is not None:
            gain_f = self._open_from_gain
            v_t_open = gain_t * v_f
            v_f_open = gain_f * v_t
            v_t = np.where(np.isnan(gain_t), v_t, v_t_open)
            v_f = np.where(np.isnan(gain_f), v_f, v_f_open)
            v_f = np.where(self._both_open, np.nan, v_f)
            v_t = np.where(self._both_open, np.nan, v_t)
        return v_f, v_t

    def create_y_matrix(self, n_bus: int) -> sparse:
        if self.y_matrix and n_bus == self._n_bus:
            return self.y_matrix

        self._n_bus = n_bus
        # Vectorized COO assembly: the per-branch stamp [[yff, yft], [ytf, ytt]] is laid
        # out with array concatenation rather than a Python loop over zip(...) (which
        # dominated this method, ~29 ms -> 0.3 ms for the pegase-9241 line element).
        fb = np.asarray(self._from_bus, dtype=np.intp)
        tb = np.asarray(self._to_bus, dtype=np.intp)
        yff = np.asarray(self._Y_ff, dtype=complex)
        yft = np.asarray(self._Y_ft, dtype=complex)
        ytf = np.asarray(self._Y_tf, dtype=complex)
        ytt = np.asarray(self._Y_tt, dtype=complex)

        rows = np.concatenate([fb, fb, tb, tb])
        cols = np.concatenate([fb, tb, fb, tb])
        data = np.concatenate([yff, yft, ytf, ytt])
        self.y_matrix = sparse((data, (rows, cols)), shape=(n_bus, n_bus), dtype=complex)

        # calculation of forward matrix. Needed to later calculate the currents over the lines.
        # Normally there would be two matrices Y_f and Y_t, but Y_t = -1 * Y_f.
        # The matrix itself has the shape (nr_of_lines, nr_of_buses).
        # Since it gets multiplied in the end with the bus voltages and returns the line currents.
        n_lines = len(fb)
        # line indices repeated for the two (forward / to) entries per branch
        i = np.concatenate([np.arange(n_lines), np.arange(n_lines)])
        # forward matrix uses (yff at fb, yft at tb); to matrix uses (ytt at tb, ytf at fb)
        self.yf_matrix = sparse(
            (np.concatenate([yff, yft]), (i, np.concatenate([fb, tb]))), shape=(n_lines, n_bus), dtype=complex
        )
        self.yt_matrix = sparse(
            (np.concatenate([ytt, ytf]), (i, np.concatenate([tb, fb]))), shape=(n_lines, n_bus), dtype=complex
        )
        return self.y_matrix

    def create_y_dc_matrix(self, n_bus: int) -> sparse:
        if self.y_dc_matrix and n_bus == self._n_bus:
            return self.y_dc_matrix

        self._n_bus = n_bus
        # Vectorized COO assembly (see create_y_matrix); replaces the per-branch zip loop.
        fb = np.asarray(self._from_bus, dtype=np.intp)
        tb = np.asarray(self._to_bus, dtype=np.intp)
        rows = np.concatenate([fb, fb, tb, tb])
        cols = np.concatenate([fb, tb, fb, tb])
        data = np.concatenate(
            [
                np.asarray(self._DC_Yff, dtype=complex),
                np.asarray(self._DC_Yft, dtype=complex),
                np.asarray(self._DC_Ytf, dtype=complex),
                np.asarray(self._DC_Ytt, dtype=complex),
            ]
        )

        self.y_dc_matrix = sparse((data, (rows, cols)), shape=(n_bus, n_bus), dtype=complex)
        return self.y_dc_matrix
