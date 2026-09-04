# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Host-side polar Newton topology for the fully-resident GPU solver.

Mirrors the constant, topology-only setup of ``p3s/cpp/nr_klu.cpp``
(``build_topology`` + ``build_J_pattern``): everything that depends only on the Ybus
sparsity pattern and the pv/pq classification, computed once and uploaded to the device.

The GPU kernels (``nr_polar_kernels.cu``) evaluate the four polar derivative blocks per
Ybus nonzero and then *gather* them into the Jacobian value array ``Jx`` via a
per-Jacobian-nonzero source map ``(src_block, src_k)`` -- exactly the ``blocks[...][...]``
gather in ``nr_klu.cpp::eval_F_and_J``.

Two differences from the C++ topology, both to serve cuSolverRf:
  * The Jacobian is emitted in **CSR** (row-major) here -- cuSolverRf's host LU consumes
    row-sorted CSR (see ``cusolver_rf_batch.py``), whereas KLU consumes CSC. The polar
    derivative math is identical; only the assembly order differs.
  * The per-row **column sort** cuSolverRf needs is folded into the gather map once, so
    the kernel writes ``Jx`` already column-sorted and no per-iteration ``Jx[perm]``
    gather is required.

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

    # Ybus CSR pattern + polar values (magnitude/angle), constant across iterations.
    Yp: NDArray  # (n+1,) int32
    Yj: NDArray  # (nnzY,) int32
    Ym: NDArray  # (nnzY,) float64  |Y|
    Ya: NDArray  # (nnzY,) float64  angle(Y)
    Ydiag: NDArray  # (n,) int32  CSR data index of each row's diagonal

    # bus classification / position maps
    pvpq: NDArray  # (npvpq,) int32  bus indices (pv then pq)
    pq: NDArray  # (npq,) int32
    pvpq_pos: NDArray  # (n,) int32  bus -> row in pvpq block, else -1
    pq_pos: NDArray  # (n,) int32  bus -> row in pq block, else -1

    # Jacobian CSR pattern (already column-sorted per row for cuSolverRf).
    Jp: NDArray  # (m+1,) int32
    Jj: NDArray  # (nnzJ,) int32  column indices (sorted within each row)
    src_k: NDArray  # (nnzJ,) int32  source Ybus CSR data index
    src_block: NDArray  # (nnzJ,) int32  which derivative block (0..3)

    @property
    def nnzY(self) -> int:
        return int(self.Yj.shape[0])

    @property
    def nnzJ(self) -> int:
        return int(self.Jj.shape[0])


def _compute_diag_ix(Yp: NDArray, Yj: NDArray, n: int) -> NDArray:
    """CSR data index of the diagonal entry of each row (-1 if absent)."""
    Ydiag: NDArray = np.full(n, -1, dtype=np.int32)
    for r in range(n):
        for k in range(Yp[r], Yp[r + 1]):
            if Yj[k] == r:
                Ydiag[r] = k
                break
    return Ydiag


def build_polar_topology(Yp, Yj, Yx, pv, pq) -> PolarTopology:
    """Build the constant polar topology from a CSR Ybus and pv/pq bus index sets.

    Parameters mirror ``nr_klu.Solver``/``solve_single``:
      Yp,Yj : CSR structure of Ybus (int)
      Yx    : complex CSR data (|Y|/angle are precomputed here, once)
      pv,pq : bus index sets

    The Ybus pattern MUST have sorted column indices within each row (p3s's
    contingency generator already sorts; ``sp.csr_matrix.sort_indices`` otherwise).
    """
    Yp = np.ascontiguousarray(Yp, dtype=np.int32)
    Yj = np.ascontiguousarray(Yj, dtype=np.int32)
    Yx = np.ascontiguousarray(Yx, dtype=np.complex128)
    n = int(Yp.shape[0] - 1)

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

    # -- build the Jacobian CSR pattern + source map, row by row --
    # Row r < npvpq  : P-mismatch at bus pvpq[r]
    # Row r >= npvpq : Q-mismatch at bus pq[r-npvpq]
    # For a J row owned by bus_row, iterate the Ybus row of bus_row; a column bus_col
    # contributes a J entry iff it is a Newton variable:
    #   * dVa column  if pvpq_pos[bus_col] != -1   (block dP/dVa or dQ/dVa)
    #   * dVm column  if pq_pos[bus_col]  != -1     (block dP/dVm or dQ/dVm)
    # We emit both, then sort the row's entries by column index (cuSolverRf needs it).
    Jp: NDArray = np.zeros(m + 1, dtype=np.int32)
    Jj_list: list[int] = []
    src_k_list: list[int] = []
    src_blk_list: list[int] = []

    def emit_row(bus_row: int, is_p_row: bool):
        cols, sk, sb = [], [], []
        for k in range(Yp[bus_row], Yp[bus_row + 1]):
            bus_col = int(Yj[k])
            cpvpq = int(pvpq_pos[bus_col])
            if cpvpq != -1:  # dVa column
                cols.append(cpvpq)
                sk.append(k)
                sb.append(BLK_dP_dVa if is_p_row else BLK_dQ_dVa)
            cpq = int(pq_pos[bus_col])
            if cpq != -1:  # dVm column (offset by npvpq)
                cols.append(npvpq + cpq)
                sk.append(k)
                sb.append(BLK_dP_dVm if is_p_row else BLK_dQ_dVm)
        # sort this row's entries by column index (stable)
        order = np.argsort(np.asarray(cols, dtype=np.int64), kind="stable")
        for o in order:
            Jj_list.append(cols[o])
            src_k_list.append(sk[o])
            src_blk_list.append(sb[o])
        return len(cols)

    row = 0
    for r in range(npvpq):
        cnt = emit_row(int(pvpq[r]), is_p_row=True)
        Jp[row + 1] = Jp[row] + cnt
        row += 1
    for r in range(npq):
        cnt = emit_row(int(pq[r]), is_p_row=False)
        Jp[row + 1] = Jp[row] + cnt
        row += 1

    Jj = np.asarray(Jj_list, dtype=np.int32)
    src_k = np.asarray(src_k_list, dtype=np.int32)
    src_block = np.asarray(src_blk_list, dtype=np.int32)

    return PolarTopology(
        n=n,
        m=m,
        npvpq=npvpq,
        npq=npq,
        Yp=Yp,
        Yj=Yj,
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
    )
