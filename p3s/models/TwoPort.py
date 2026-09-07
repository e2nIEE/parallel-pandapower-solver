# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix as sparse


class TwoPort:
    def __init__(self):
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
        # TODO: decide if a branch directional dc powerflow is needed.
        self.yf_dc_matrix: sparse | None = None
        self.yt_dc_matrix: sparse | None = None

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
