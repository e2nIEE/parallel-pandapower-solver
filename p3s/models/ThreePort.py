# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix as sparse

_PORTS = ("1", "2", "3")
_STAMPS = tuple(f"_Y_{i}{j}" for i in _PORTS for j in _PORTS)
_DC_STAMPS = tuple(f"_DC_Y_{i}{j}" for i in _PORTS for j in _PORTS)


class ThreePort:
    """A three-terminal element (hv, mv, lv bus) given by its 3x3 bus admittance block per
    element: ``_Y_ij`` is the current into terminal i per volt at terminal j (i, j = 1 hv,
    2 mv, 3 lv), ``_DC_Y_ij`` the same for the DC B-matrix (stored as -1/(j*x), like TwoPort)."""

    def __init__(self):
        self._hv_bus = []
        self._mv_bus = []
        self._lv_bus = []
        for name in _STAMPS + _DC_STAMPS:
            setattr(self, name, [])
        # DC phase-shift injection per bus (added to the DC right-hand side, see TwoPort)
        self.p_shift: NDArray | None = None
        self.y_matrix: sparse | None = None
        self.y_dc_matrix: sparse | None = None
        self._n_bus: int | None = None

    def _apply_in_service(self, element_table) -> NDArray:
        """Zero the AC and DC stamps of out-of-service elements, keeping rows aligned with the
        element table (see TwoPort._apply_in_service)."""
        n = len(np.asarray(self._hv_bus))
        if "in_service" not in element_table:
            return np.ones(n, dtype=bool)
        in_service = np.asarray(element_table["in_service"].values, dtype=bool)
        if not in_service.all():
            for name in _STAMPS + _DC_STAMPS:
                setattr(self, name, np.where(in_service, np.asarray(getattr(self, name)), 0.0))
        return in_service

    def _buses(self) -> tuple[NDArray, NDArray, NDArray]:
        return (
            np.asarray(self._hv_bus, dtype=np.intp),
            np.asarray(self._mv_bus, dtype=np.intp),
            np.asarray(self._lv_bus, dtype=np.intp),
        )

    def _assemble(self, names: tuple[str, ...], n_bus: int) -> sparse:
        buses = self._buses()
        rows = np.concatenate([buses[i] for i in range(3) for _ in range(3)])
        cols = np.concatenate([buses[j] for _ in range(3) for j in range(3)])
        data = np.concatenate([np.asarray(getattr(self, name), dtype=complex) for name in names])
        return sparse((data, (rows, cols)), shape=(n_bus, n_bus), dtype=complex)

    def create_y_matrix(self, n_bus: int = -1) -> sparse:
        self._n_bus = n_bus
        self.y_matrix = self._assemble(_STAMPS, n_bus)
        return self.y_matrix

    def create_y_dc_matrix(self, n_bus: int = -1) -> sparse:
        self._n_bus = n_bus
        self.y_dc_matrix = self._assemble(_DC_STAMPS, n_bus)
        return self.y_dc_matrix

    def port_currents(self, voltage: NDArray) -> NDArray:
        """Currents into the three terminals per element, shape (n_element, 3), per unit."""
        v = np.stack([voltage[b] for b in self._buses()], axis=1)
        y = np.stack([np.asarray(getattr(self, name), dtype=complex) for name in _STAMPS], axis=1).reshape(-1, 3, 3)
        return np.einsum("eij,ej->ei", y, v)
