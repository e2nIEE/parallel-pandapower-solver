# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from numpy.typing import NDArray
from numba import jit
from scipy.sparse import coo_matrix as sparse
from scipy.sparse import csr_matrix

from p3s.PowerflowObject import PowerflowObject


@jit(nopython=True, cache=False)
def _count_row_nnz(row_list, Yp, Yj, pvpq_pos, pq_pos):
    """Count contributions per output row (same as CUDA count kernel)."""
    counts = np.zeros(len(row_list), dtype=np.int32)
    for r, row_bus in enumerate(row_list):
        start = Yp[row_bus]
        end = Yp[row_bus + 1]
        c = 0
        for k in range(start, end):
            col_bus = Yj[k]
            if pvpq_pos[col_bus] != -1:
                c += 1
            if pq_pos[col_bus] != -1:
                c += 1
        counts[r] = c
    return counts


@jit(nopython=True, cache=False)
def get_ybus_diag_ix(Ybus_indices, Ybus_indptr, N_YBUS_SHAPE):
    diag_data_ix = np.zeros(N_YBUS_SHAPE, dtype=np.int32)
    for col in range(N_YBUS_SHAPE):
        for data_ix, row_ix in zip(
            range(Ybus_indptr[col], Ybus_indptr[col + 1]),
            Ybus_indices[Ybus_indptr[col] : Ybus_indptr[col + 1]],
            strict=False,
        ):
            if row_ix == col:
                diag_data_ix[row_ix] = data_ix
    return diag_data_ix


@jit(nopython=True, cache=False)
def _dSbus_dV_numba_faster_sparse(
    Yx: NDArray, Yp: NDArray, Yj: NDArray, Yd: NDArray, voltage: NDArray
):  # pragma: no cover
    Vnorm = np.abs(voltage)

    dS_dVm: NDArray = np.zeros(len(Yx), dtype=np.complex128)
    dS_dVa: NDArray = np.zeros(len(Yx), dtype=np.complex128)

    for row in range(len(Yp) - 1):
        diag_ix = Yd[row]
        for data_ix in range(Yp[row], Yp[row + 1]):
            col = Yj[data_ix]
            val = Yx[data_ix]

            tmp = voltage[row] * np.conj(val * voltage[col])
            tmp_j = tmp * 1j

            dS_dVm[data_ix] += tmp / Vnorm[col]
            dS_dVm[diag_ix] += tmp / Vnorm[row]

            dS_dVa[data_ix] += -tmp_j
            dS_dVa[diag_ix] += tmp_j

    return dS_dVm, dS_dVa


@jit(nopython=True, cache=False)
def _create_J_numba(Yp, Yj, Yx, Yd, _pvpq, _pq, pvpq_pos, pq_pos, voltage):  # pragma: no cover
    """Construct Jacobian J by first computing dS derivatives, then assembling J.

    This version computes dS_dVm and dS_dVa entries aligned with Ybus CSR
    using the fast sparse routine, then selects the appropriate blocks for J
    via O(1) index lookups.
    """

    # Compute partial derivatives in CSR-data form aligned with Ybus structure
    # dVm_x, dVa_x = _dSbus_dV_numba_sparse(Yx, Yp, Yj, voltage)
    dVm_x, dVa_x = _dSbus_dV_numba_faster_sparse(Yx, Yp, Yj, Yd, voltage)

    # Dimensions
    lpvpq = len(_pvpq)
    lpq = len(_pq)
    n_rows = lpvpq + lpq

    # Reserve and allocate CSR arrays for J
    reserve = max(1, len(dVm_x) * 4)
    Jx = np.empty(reserve, dtype=np.float64)
    Jj = np.empty(reserve, dtype=np.int32)
    Jp = np.zeros(n_rows + 1, dtype=np.int32)

    nnz = 0

    # Top block rows: pvpq (J11 real from dVa, J12 real from dVm)
    for r in range(lpvpq):
        row_bus = _pvpq[r]
        nnz_start = nnz
        for k in range(Yp[row_bus], Yp[row_bus + 1]):
            col_bus = Yj[k]
            c_pvpq = pvpq_pos[col_bus]
            if c_pvpq != -1:
                if nnz >= Jx.size:
                    new_cap = int(Jx.size * 2) + 1
                    Jx = np.resize(Jx, new_cap)
                    Jj = np.resize(Jj, new_cap)
                Jx[nnz] = dVa_x[k].real
                Jj[nnz] = c_pvpq
                nnz += 1
            c_pq = pq_pos[col_bus]
            if c_pq != -1:
                if nnz >= Jx.size:
                    new_cap = int(Jx.size * 2) + 1
                    Jx = np.resize(Jx, new_cap)
                    Jj = np.resize(Jj, new_cap)
                Jx[nnz] = dVm_x[k].real
                Jj[nnz] = lpvpq + c_pq
                nnz += 1
        Jp[r + 1] = Jp[r] + (nnz - nnz_start)

    # Bottom block rows: pq (J21 imag from dVa, J22 imag from dVm)
    for r in range(lpq):
        row_bus = _pq[r]
        nnz_start = nnz
        for k in range(Yp[row_bus], Yp[row_bus + 1]):
            col_bus = Yj[k]
            c_pvpq = pvpq_pos[col_bus]
            if c_pvpq != -1:
                if nnz >= Jx.size:
                    new_cap = int(Jx.size * 2) + 1
                    Jx = np.resize(Jx, new_cap)
                    Jj = np.resize(Jj, new_cap)
                Jx[nnz] = dVa_x[k].imag
                Jj[nnz] = c_pvpq
                nnz += 1
            c_pq = pq_pos[col_bus]
            if c_pq != -1:
                if nnz >= Jx.size:
                    new_cap = int(Jx.size * 2) + 1
                    Jx = np.resize(Jx, new_cap)
                    Jj = np.resize(Jj, new_cap)
                Jx[nnz] = dVm_x[k].imag
                Jj[nnz] = lpvpq + c_pq
                nnz += 1
        Jp[lpvpq + r + 1] = Jp[lpvpq + r] + (nnz - nnz_start)

    # Trim to actual nnz
    Jx = Jx[:nnz]
    Jj = Jj[:nnz]

    return Jx, Jp, Jj


class PQPVPowerflow(PowerflowObject):
    def __init__(self, YBus: sparse, pq: NDArray, pv: NDArray, ref: NDArray):
        super().__init__()
        self._YBus: sparse = YBus
        self._pv: NDArray = pv
        self._pq: NDArray = pq
        self._ref: NDArray = ref
        self._pvpq: NDArray = np.r_[pv, pq]
        self._cols_pvpq: NDArray = self._pvpq
        self._rows_pvpq: NDArray = self._pvpq.T
        self._n_bus: int = YBus.shape[0]
        self._Yd = get_ybus_diag_ix(self._YBus.indices, self._YBus.indptr, self._n_bus)

        # Lookups from bus index -> position in pvpq/pq blocks
        n_bus = self._n_bus
        self.pvpq_pos = -np.ones(n_bus, dtype=int)
        for pos, bus in enumerate(self._pvpq):
            self.pvpq_pos[bus] = pos
        self.pq_pos = -np.ones(n_bus, dtype=int)
        for pos, bus in enumerate(self._pq):
            self.pq_pos[bus] = pos

    def _dSbus_dV(self, V: NDArray):
        Ibus = self._YBus * V
        ib = range(len(V))
        diagV = sparse((V, (ib, ib)))
        diagIbus = sparse((Ibus, (ib, ib)))
        diagVnorm = sparse((V / abs(V), (ib, ib)))
        dS_dVm = diagV * np.conj(self._YBus * diagVnorm) + np.conj(diagIbus) * diagVnorm
        dS_dVa = 1j * diagV * np.conj(diagIbus - self._YBus * diagV)
        return dS_dVm, dS_dVa

    def create_J(self, voltage):
        ybus: csr_matrix = self._YBus.tocsr()
        Yp = ybus.indptr
        Yj = ybus.indices
        Yx = ybus.data

        Jx, Jp, Jj = _create_J_numba(Yp, Yj, Yx, self._Yd, self._pvpq, self._pq, self.pvpq_pos, self.pq_pos, voltage)
        return Jx, Jp, Jj

    def evaluate_Results(self, dx, voltage):
        npq = len(self._pq)
        npv = len(self._pv)
        Va = np.angle(voltage)
        Vm = np.abs(voltage)
        Va[self._pvpq] += dx[self._offset : self._offset + npv + npq]
        Vm[self._pq] += dx[self._offset + npv + npq : self._offset + npv + 2 * npq]
        return Vm * np.exp(1j * Va)

    def evaluate_Fx(self, Sbus: NDArray, V: NDArray):
        mis = V * np.conj(self._YBus * V) - Sbus
        F = np.r_[mis[self._pv].real, mis[self._pq].real, mis[self._pq].imag]
        return F
