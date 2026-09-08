# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause


from scipy.sparse import coo_matrix as sparse


class ThreePort:
    def __init__(self):
        self._hv_bus = []
        self._mv_bus = []
        self._lv_bus = []
        self._Y_11 = []
        self._Y_12 = []
        self._Y_13 = []
        self._Y_21 = []
        self._Y_22 = []
        self._Y_23 = []
        self._Y_31 = []
        self._Y_32 = []
        self._Y_33 = []
        self.y_matrix: sparse | None = None
        self._n_bus: int | None = None

    def create_y_matrix(self, n_bus: int = -1) -> sparse:
        # if self.y_matrix and n_bus == self._n_bus:
        #    return self.y_matrix

        # if nbus == -1:
        #    nbus = max(max(self.hv_bus), max(self.mv_bus), max(self.lv_bus)) + 1

        self._n_bus = n_bus
        rows, cols, data = [], [], []
        for i, j, k, y11, y12, y13, y21, y22, y23, y31, y32, y33 in zip(
            self._hv_bus,
            self._mv_bus,
            self._lv_bus,
            self._Y_11,
            self._Y_12,
            self._Y_13,
            self._Y_21,
            self._Y_22,
            self._Y_23,
            self._Y_31,
            self._Y_32,
            self._Y_33,
            strict=True,
        ):
            rows += [i, i, i, j, j, j, k, k, k]

            cols += [i, j, k, i, j, k, i, j, k]

            data += [y11, y12, y13, y21, y22, y23, y31, y32, y33]

        self.y_matrix = sparse((data, (rows, cols)), shape=(n_bus, n_bus), dtype=complex)
        print("DEBUG ThreePort Y-matrix nnz:", self.y_matrix.nnz)
        return self.y_matrix
