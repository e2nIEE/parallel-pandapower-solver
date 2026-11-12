# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Host-side polar Newton topology for the fully-resident GPU solver.

Mirrors the constant, topology-only setup of ``p3s/cpp/nr_klu.cpp``
(``build_topology`` + ``build_J_pattern``): everything that depends only on the Ybus
sparsity pattern and the pv/pq classification, computed once and uploaded to the device.

The Jacobian pattern is emitted in **CSR** with the columns of every row sorted (KLU on
the CPU consumes CSC; cuDSS and the lean LU plan take CSR). Each Jacobian nonzero is one
of the four polar derivative blocks of one Ybus nonzero, recorded as ``(src_block,
src_k)`` -- the ``blocks[...][...]`` gather of ``nr_klu.cpp::eval_F_and_J``. The GPU
kernel (``nr_polar_kernels.cu::eval_FJ``) uses the inverse map ``posJ``: it walks a Ybus
row once and writes the up to four Jacobian entries of every nonzero straight to their
positions.

Block ids match nr_klu: 0=dP/dVa, 1=dP/dVm, 2=dQ/dVa, 3=dQ/dVm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# derivative block ids (must match nr_polar_kernels.cu and nr_klu.cpp)
BLK_dP_dVa = 0
BLK_dP_dVm = 1
BLK_dQ_dVa = 2
BLK_dQ_dVm = 3


@dataclass
class PolarTopology:
    """Constant, topology-only arrays uploaded once to the device.

    Sizes: n = n_bus, nnzY = Ybus nnz, m = npvpq + npq (Jacobian dim),
    nnzJ = Jacobian nnz.
    """

    n: int
    m: int
    npvpq: int
    npq: int

    # Ybus CSR pattern + values, constant across iterations.
    Yp: NDArray  # (n+1,) int32
    Yj: NDArray  # (nnzY,) int32
    Yx: NDArray  # (nnzY,) complex128  G + jB
    Ym: NDArray  # (nnzY,) float64  |Y|
    Ya: NDArray  # (nnzY,) float64  angle(Y)
    Ydiag: NDArray  # (n,) int32  CSR data index of each row's diagonal (-1 if absent)

    # bus classification / position maps
    pvpq: NDArray  # (npvpq,) int32  bus indices (pv then pq)
    pq: NDArray  # (npq,) int32
    pvpq_pos: NDArray  # (n,) int32  bus -> row in pvpq block, else -1
    pq_pos: NDArray  # (n,) int32  bus -> row in pq block, else -1

    # Jacobian CSR pattern (columns sorted within each row).
    Jp: NDArray  # (m+1,) int32
    Jj: NDArray  # (nnzJ,) int32
    src_k: NDArray  # (nnzJ,) int32  source Ybus CSR data index
    src_block: NDArray  # (nnzJ,) int32  which derivative block (0..3)
    # (4*nnzY,) int32: Jacobian nonzero holding block b of Ybus nonzero k at [4k+b], -1 = none
    posJ: NDArray

    @property
    def nnzY(self) -> int:
        return int(self.Yj.shape[0])

    @property
    def nnzJ(self) -> int:
        return int(self.Jj.shape[0])

    @property
    def row_p(self) -> NDArray:
        """(n,) Jacobian row / variable index of each bus's P equation / angle, -1 = none."""
        return self.pvpq_pos

    @property
    def row_q(self) -> NDArray:
        """(n,) Jacobian row / variable index of each bus's Q equation / magnitude, -1 = none."""
        return np.where(self.pq_pos >= 0, self.npvpq + self.pq_pos, -1).astype(np.int32)


def _compute_diag_ix(Yp: NDArray, Yj: NDArray, n: int) -> NDArray:
    """CSR data index of the diagonal entry of each row (-1 if absent)."""
    row: NDArray = np.repeat(np.arange(n, dtype=np.int64), np.diff(Yp))
    k = np.nonzero(Yj == row)[0]
    Ydiag: NDArray = np.full(n, -1, dtype=np.int32)
    # assigned in reverse, so that for a (malformed) duplicated diagonal the FIRST entry wins
    Ydiag[row[k[::-1]]] = k[::-1]
    return Ydiag


def build_polar_topology(Yp, Yj, Yx, pv, pq) -> PolarTopology:
    """Build the constant polar topology from a CSR Ybus and pv/pq bus index sets.

    Parameters mirror ``nr_klu.Solver``/``solve_single``:
      Yp,Yj : CSR structure of Ybus (int)
      Yx    : complex CSR data
      pv,pq : bus index sets

    The Ybus pattern MUST have sorted column indices within each row (p3s's
    contingency generator already sorts; ``sp.csr_matrix.sort_indices`` otherwise).
    """
    Yp = np.ascontiguousarray(Yp, dtype=np.int32)
    Yj = np.ascontiguousarray(Yj, dtype=np.int32)
    Yx = np.ascontiguousarray(Yx, dtype=np.complex128)
    n = int(Yp.shape[0] - 1)
    nnzY = int(Yj.shape[0])

    Ym = np.ascontiguousarray(np.abs(Yx), dtype=np.float64)
    Ya = np.ascontiguousarray(np.angle(Yx), dtype=np.float64)
    Ydiag = _compute_diag_ix(Yp, Yj, n)

    pv = np.asarray(pv, dtype=np.int32).ravel()
    pq = np.asarray(pq, dtype=np.int32).ravel()
    pvpq = np.concatenate([pv, pq]).astype(np.int32)
    npvpq = int(pvpq.shape[0])
    npq = int(pq.shape[0])
    m = npvpq + npq

    pvpq_pos: NDArray = np.full(n, -1, dtype=np.int32)
    pvpq_pos[pvpq] = np.arange(npvpq, dtype=np.int32)
    pq_pos: NDArray = np.full(n, -1, dtype=np.int32)
    pq_pos[pq] = np.arange(npq, dtype=np.int32)

    # -- Jacobian CSR pattern + source map --
    # Row r < npvpq  : P-mismatch at bus pvpq[r]
    # Row r >= npvpq : Q-mismatch at bus pq[r-npvpq]
    # A Ybus nonzero k = (bus_row, bus_col) contributes one entry per (equation, variable)
    # pair that exists:  P row iff bus_row is pvpq, Q row iff bus_row is pq;
    #                    dVa column iff bus_col is pvpq, dVm column iff bus_col is pq.
    k_all: NDArray = np.arange(nnzY, dtype=np.int64)
    bus_row: NDArray = np.repeat(np.arange(n, dtype=np.int64), np.diff(Yp))
    bus_col = Yj.astype(np.int64)
    row_p = pvpq_pos[bus_row].astype(np.int64)
    row_q = np.where(pq_pos[bus_row] >= 0, npvpq + pq_pos[bus_row], -1).astype(np.int64)
    col_va = pvpq_pos[bus_col].astype(np.int64)
    col_vm = np.where(pq_pos[bus_col] >= 0, npvpq + pq_pos[bus_col], -1).astype(np.int64)

    rows: list[NDArray] = []
    cols: list[NDArray] = []
    ks: list[NDArray] = []
    blks: list[NDArray] = []
    for blk, r, c in (
        (BLK_dP_dVa, row_p, col_va),
        (BLK_dP_dVm, row_p, col_vm),
        (BLK_dQ_dVa, row_q, col_va),
        (BLK_dQ_dVm, row_q, col_vm),
    ):
        keep = (r >= 0) & (c >= 0)
        rows.append(r[keep])
        cols.append(c[keep])
        ks.append(k_all[keep])
        blks.append(np.full(int(keep.sum()), blk, dtype=np.int64))
    rows_a = np.concatenate(rows)
    cols_a = np.concatenate(cols)
    order = np.lexsort((cols_a, rows_a))  # by row, columns ascending within a row
    Jj = cols_a[order].astype(np.int32)
    src_k = np.concatenate(ks)[order].astype(np.int32)
    src_block = np.concatenate(blks)[order].astype(np.int32)
    Jp: NDArray = np.zeros(m + 1, dtype=np.int32)
    Jp[1:] = np.cumsum(np.bincount(rows_a, minlength=m))

    posJ: NDArray = np.full(4 * nnzY, -1, dtype=np.int32)
    posJ[4 * src_k.astype(np.int64) + src_block] = np.arange(Jj.shape[0], dtype=np.int32)

    return PolarTopology(
        n=n,
        m=m,
        npvpq=npvpq,
        npq=npq,
        Yp=Yp,
        Yj=Yj,
        Yx=Yx,
        Ym=Ym,
        Ya=Ya,
        Ydiag=Ydiag,
        pvpq=pvpq,
        pq=pq,
        pvpq_pos=pvpq_pos,
        pq_pos=pq_pos,
        Jp=Jp,
        Jj=Jj,
        src_k=src_k,
        src_block=src_block,
        posJ=posJ,
    )


def jacobian_values_host(t: PolarTopology, Vm: NDArray, Va: NDArray, Yx: NDArray | None = None) -> NDArray:
    """Jacobian values at ``(Vm, Va)`` in the CSR order of ``t.Jp``/``t.Jj`` (numpy).

    Host replica of the GPU assembly (and of ``nr_klu.cpp::eval_F_and_J``). Used for the
    one-off symbolic setup of the library backends and by the tests; ``Yx`` overrides the
    topology's Ybus values (same pattern).
    """
    Ym = t.Ym if Yx is None else np.abs(Yx)
    Ya = t.Ya if Yx is None else np.angle(Yx)
    n = t.n
    row: NDArray = np.repeat(np.arange(n), np.diff(t.Yp))
    col = t.Yj
    delta = Va[row] - Ya - Va[col]
    s = np.sin(delta)
    c = np.cos(delta)
    VmYm = Vm[row] * Ym
    vij = VmYm * Vm[col]
    off = row != col
    blocks = np.zeros((4, t.nnzY))
    blocks[BLK_dP_dVa] = np.where(off, vij * s, 0.0)
    blocks[BLK_dQ_dVa] = np.where(off, -vij * c, 0.0)
    blocks[BLK_dP_dVm] = np.where(off, VmYm * c, 0.0)
    blocks[BLK_dQ_dVm] = np.where(off, VmYm * s, 0.0)
    has_diag = t.Ydiag >= 0
    d = t.Ydiag[has_diag]
    blocks[BLK_dP_dVa][d] = -np.bincount(row, weights=blocks[BLK_dP_dVa], minlength=n)[has_diag]
    blocks[BLK_dQ_dVa][d] = -np.bincount(row, weights=blocks[BLK_dQ_dVa], minlength=n)[has_diag]
    wc = np.where(off, Ym * Vm[col] * c, 2.0 * Vm[row] * Ym * c)
    ws = np.where(off, Ym * Vm[col] * s, 2.0 * Vm[row] * Ym * s)
    blocks[BLK_dP_dVm][d] = np.bincount(row, weights=wc, minlength=n)[has_diag]
    blocks[BLK_dQ_dVm][d] = np.bincount(row, weights=ws, minlength=n)[has_diag]
    return np.ascontiguousarray(blocks[t.src_block, t.src_k])
