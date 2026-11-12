# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Host-side plan for the batched static-pivot sparse LU on the GPU ("lean" backend).

This is the GPU counterpart of ``p3s/cpp/lean_lu.hpp``: the pivot order and the L/U
pattern are frozen ONCE per grid topology, and every Newton iteration only redoes the
numeric work on that fixed pattern. Where the C++ kernel runs the column algorithm
serially per system (one system per CPU thread), the GPU kernel needs parallelism *inside*
a factorization as well as across the batch, so the same factorization is organised
differently here:

* **Symmetric static pivoting.** The power-flow Jacobian has a structurally symmetric
  pattern with a full diagonal, so one fill-reducing permutation is applied to rows and
  columns alike and the pivots stay on the diagonal (the order comes from SuperLU's
  minimum-degree ordering on ``A + A^T``). No pivot search at run time. A per-system pivot
  ratio check in the kernel rejects systems this order does not suit.

* **Crout form, one task per factor entry.** Entry ``(i, k)`` of the factors is
  ``a(i,k) - sum_t L(i,t) * U(t,k)`` over the pivots ``t`` that precede both. Written
  like that, every entry is one independent dot product over precomputed operand pairs
  ("pairs"), and all entries whose pivot sits at the same height of the elimination tree
  can be computed concurrently. Pegase 9241: 1.3 M pairs in 254 levels, where a
  column-oriented schedule would leave three quarters of the work in a serial chain.

* **Fixed slots per level.** Each GPU thread block owns 32 systems and ``slots`` warps.
  Inside a level the tasks are dealt to the slots (heaviest first, snake order), so one
  kernel launch runs the whole factorization: every warp walks its own task list and the
  block meets at a barrier between levels. The triangular solves use the same scheme with
  one task per row.

Values are stored batch-interleaved (``val[position * stride + system]``): the 32 systems
of a warp execute the same task on adjacent memory, so every load and store is coalesced
and the index arrays below are read once per warp, not once per system.

Layout of the factor values, per pivot ``t`` with below-diagonal structure ``S_t``::

    base[t]                     1 / U(t,t)
    base[t] + 1 + r             U(t, S_t[r])
    base[t] + 1 + |S_t| + r     L(S_t[r], t)      (unit lower: scaled by 1 / U(t,t))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numba import njit
from numpy.typing import NDArray

# Task codes (``tcode``); values >= 0 are the diagonal position of an L entry's pivot.
TASK_U = -1
TASK_DIAG = -2
# U entry without pairs: its factor value IS the matrix value. Nothing to do when the
# factorization runs in place; copied when the factors live in their own (single) array.
TASK_COPY = -3
# Set on ``tdst`` when the entry has no counterpart in A (pure fill): it starts from 0
# instead of from the value the Jacobian assembly scattered into the factor array.
FILL_BIT = np.int32(-(2**31))


@dataclass
class LeanLUPlan:
    """Device-ready index arrays (all int32) of one frozen factorization."""

    m: int  # matrix dimension
    nnz_lu: int  # number of stored factor values per system
    n_pairs: int  # multiply-subtract operations of one factorization
    slots: int  # warps per thread block (tasks of a level are dealt to this many lists)
    perm: NDArray  # (m,)  perm[k] = original row/column eliminated at step k
    pinv: NDArray  # (m,)  inverse of perm
    apos: NDArray  # (nnzA,) factor position of each entry of A (CSR order of the pattern)
    base: NDArray  # (m+1,) first factor position of each pivot (the diagonal)

    # factorization: tasks grouped by (level, slot)
    n_levels: int
    lptr: NDArray  # (n_levels * slots + 1,) task range of each (level, slot)
    tdst: NDArray  # (n_task = nnz_lu,) factor position written (FILL_BIT set: starts from 0)
    tcode: NDArray  # (n_task,) TASK_U / TASK_DIAG / TASK_COPY / diagonal position of the pivot (L)
    tpp: NDArray  # (n_task + 1,) pair range of each task
    pa: NDArray  # (n_pairs,) position of L(i, t)
    pb: NDArray  # (n_pairs,) position of U(t, k)

    # forward solve L y = b: one task per row, same levels as the factorization
    flptr: NDArray  # (n_levels * slots + 1,)
    frow: NDArray  # (m,) row of each task
    fptr: NDArray  # (m + 1,) entry range of each row
    fcol: NDArray  # (nnzL,) column j of L(i, j)
    fpos: NDArray  # (nnzL,) factor position of L(i, j)

    # backward solve U x = y: one task per row, levels by depth in the elimination tree;
    # row i of U is the contiguous run base[i] + 1 .. with columns Li[Lp[i]:Lp[i+1]]
    n_blevels: int
    blptr: NDArray  # (n_blevels * slots + 1,)
    brow: NDArray  # (m,)
    Lp: NDArray  # (m + 1,)
    Li: NDArray  # (nnzL,)


@njit(cache=True)
def _symbolic(n, Ap, Ai, perm):  # pragma: no cover - compiled
    """Symbolic Cholesky of a symmetric pattern (CSC ``Ap``/``Ai``) under ``perm``.

    Returns the strictly-lower pattern of L (CSC, rows ascending), the elimination tree,
    each pivot's height in it (factorization / forward-solve level) and its depth
    (backward-solve level).
    """
    pinv = np.empty(n, np.int64)
    for k in range(n):
        pinv[perm[k]] = k
    parent = np.full(n, -1, np.int64)
    head = np.full(n, -1, np.int64)  # children of each node, as a linked list
    nxt = np.full(n, -1, np.int64)
    mark = np.full(n, -1, np.int64)
    Lp = np.zeros(n + 1, np.int64)
    cap = 4 * Ap[n] + 16
    Li = np.empty(cap, np.int64)
    nz = 0
    for k in range(n):
        start = nz
        mark[k] = k
        j = perm[k]
        # column k of the permuted matrix ...
        for p in range(Ap[j], Ap[j + 1]):
            i = pinv[Ai[p]]
            if i > k and mark[i] != k:
                mark[i] = k
                if nz >= cap:
                    cap *= 2
                    grown = np.empty(cap, np.int64)
                    grown[:nz] = Li[:nz]
                    Li = grown
                Li[nz] = i
                nz += 1
        # ... merged with the structure of every child in the elimination tree
        c = head[k]
        while c != -1:
            for p in range(Lp[c], Lp[c + 1]):
                i = Li[p]
                if i > k and mark[i] != k:
                    mark[i] = k
                    if nz >= cap:
                        cap *= 2
                        grown = np.empty(cap, np.int64)
                        grown[:nz] = Li[:nz]
                        Li = grown
                    Li[nz] = i
                    nz += 1
            c = nxt[c]
        Li[start:nz] = np.sort(Li[start:nz])
        Lp[k + 1] = nz
        if nz > start:
            par = Li[start]
            parent[k] = par
            nxt[k] = head[par]
            head[par] = k
    height = np.zeros(n, np.int64)
    for k in range(n):
        par = parent[k]
        if par >= 0 and height[par] < height[k] + 1:
            height[par] = height[k] + 1
    depth = np.zeros(n, np.int64)
    for k in range(n - 1, -1, -1):
        if parent[k] >= 0:
            depth[k] = depth[parent[k]] + 1
    return Lp, Li[:nz].copy(), height, depth


def _deal(level: NDArray, weight: NDArray, n_levels: int, slots: int):
    """Deal tasks to ``slots`` lists per level, heaviest first in snake order.

    Returns ``(order, ptr)``: ``order`` lists the tasks grouped by (level, slot) and
    ``ptr[level * slots + s]`` is where the list of slot ``s`` starts.
    """
    o = np.lexsort((-weight, level))
    lv = level[o]
    first = np.searchsorted(lv, np.arange(n_levels))
    rank = np.arange(len(o)) - first[lv]
    rnd, pos = np.divmod(rank, slots)
    slot = np.where(rnd % 2 == 0, pos, slots - 1 - pos)
    key = lv * slots + slot
    ptr = np.zeros(n_levels * slots + 1, np.int64)
    ptr[1:] = np.cumsum(np.bincount(key, minlength=n_levels * slots))
    return o[np.argsort(key, kind="stable")], ptr


def fill_reducing_order(m: int, Ap: NDArray, Ai: NDArray) -> NDArray:
    """Minimum-degree order of a symmetric pattern (CSC), ``perm[k]`` = k-th pivot.

    SuperLU's ``MMD_AT_PLUS_A`` is the one fill-reducing symmetric ordering scipy ships.
    The ordering only looks at the pattern; the values are made strictly diagonally
    dominant so the throw-away numeric factorization SuperLU also runs cannot fail.
    """
    colcount = np.diff(Ap)
    col = np.repeat(np.arange(m), colcount)
    vals = np.where(Ai == col, 2.0 * colcount[col] + 1.0, 1.0)
    A = sp.csc_matrix((vals, Ai, Ap), shape=(m, m))
    lu = spla.splu(A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options={"SymmetricMode": True})
    return np.argsort(lu.perm_c).astype(np.int64)


def build_lean_lu_plan(Ap, Aj, slots: int = 32, perm: NDArray | None = None) -> LeanLUPlan:
    """Freeze pivot order, factor pattern and task schedule for the CSR pattern ``Ap``/``Aj``.

    The pattern is symmetrized internally (an entry missing on one side is carried as an
    explicit zero), and a missing diagonal entry is added, so any square pattern works;
    for the power-flow Jacobian neither applies. ``perm`` overrides the built-in ordering.
    """
    Ap = np.asarray(Ap, dtype=np.int64)
    Aj = np.asarray(Aj, dtype=np.int64)
    m = len(Ap) - 1
    nnzA = len(Aj)
    Arow = np.repeat(np.arange(m), np.diff(Ap))

    S = sp.csr_matrix((np.ones(nnzA), Aj, Ap), shape=(m, m))
    S = (S + S.T + sp.identity(m, format="csr")).tocsc()
    S.sort_indices()
    Sp_ = S.indptr.astype(np.int64)
    Si = S.indices.astype(np.int64)
    if perm is None:
        perm = fill_reducing_order(m, Sp_, Si)
    perm = np.ascontiguousarray(perm, dtype=np.int64)
    pinv = np.empty(m, np.int64)
    pinv[perm] = np.arange(m)

    Lp, Li, height, depth = _symbolic(m, Sp_, Si, perm)
    cc = np.diff(Lp)  # |S_t|
    nL = len(Li)
    Lcol = np.repeat(np.arange(m), cc)
    rank = np.arange(nL) - Lp[Lcol]
    base = np.zeros(m + 1, np.int64)
    base[1:] = np.cumsum(2 * cc + 1)
    nnz_lu = int(base[-1])
    if nnz_lu >= 2**31 - 1:
        raise ValueError("factor pattern too large for int32 positions")
    posU = base[Lcol] + 1 + rank  # U(t, s) for the L entry (s, t)
    posL = posU + cc[Lcol]  # L(s, t)
    Lkey = Lcol * m + Li  # ascending: locates the L entry (row, col) by binary search

    def position(i, j):
        """Factor position of entry (i, j) of the permuted matrix (vectorized)."""
        lo = np.minimum(i, j)
        e = np.searchsorted(Lkey, lo * m + np.maximum(i, j))
        e = np.where(i == j, 0, e)
        return np.where(i == j, base[lo], np.where(i > j, posL[e], posU[e]))

    apos = position(pinv[Arow], pinv[Aj])

    # ---- pairs: pivot t contributes L(a, t) * U(t, b) to entry (a, b) for all a, b in S_t
    reps = cc[Lcol]
    ea = np.repeat(np.arange(nL), reps)
    eb = Lp[Lcol[ea]] + (np.arange(len(ea)) - np.repeat(np.cumsum(reps) - reps, reps))
    pa = posL[ea]
    pb = posU[eb]
    ptgt = position(Li[ea], Li[eb])
    n_pairs = len(pa)

    # ---- tasks: one per factor entry
    piv = np.empty(nnz_lu, np.int64)
    code = np.full(nnz_lu, TASK_U, np.int64)
    piv[base[:-1]] = np.arange(m)
    code[base[:-1]] = TASK_DIAG
    piv[posU] = Lcol
    piv[posL] = Lcol
    code[posL] = base[Lcol]
    npairs = np.bincount(ptgt, minlength=nnz_lu)
    in_A = np.zeros(nnz_lu, bool)
    in_A[apos] = True
    code[(code == TASK_U) & (npairs == 0)] = TASK_COPY
    tids = np.arange(nnz_lu)
    n_levels = int(height.max()) + 1 if m else 0
    order, lptr = _deal(height[piv[tids]], npairs[tids], n_levels, slots)
    tdst = tids[order]
    task_of = np.full(nnz_lu, -1, np.int64)
    task_of[tdst] = np.arange(len(tdst))
    po = np.argsort(task_of[ptgt], kind="stable")
    tpp = np.zeros(len(tdst) + 1, np.int64)
    tpp[1:] = np.cumsum(npairs[tdst])
    tdst_enc = tdst.astype(np.int32)
    tdst_enc[~in_A[tdst]] |= FILL_BIT

    # ---- triangular solves
    ro = np.argsort(Li, kind="stable")  # L entries grouped by row
    fcount = np.bincount(Li, minlength=m)
    fptr = np.zeros(m + 1, np.int64)
    fptr[1:] = np.cumsum(fcount)
    frow, flptr = _deal(height, fcount, n_levels, slots)
    n_blevels = int(depth.max()) + 1 if m else 0
    brow, blptr = _deal(depth, cc, n_blevels, slots)

    i32 = lambda a: np.ascontiguousarray(a, dtype=np.int32)  # noqa: E731
    return LeanLUPlan(
        m=m,
        nnz_lu=nnz_lu,
        n_pairs=n_pairs,
        slots=slots,
        perm=i32(perm),
        pinv=i32(pinv),
        apos=i32(apos),
        base=i32(base),
        n_levels=n_levels,
        lptr=i32(lptr),
        tdst=tdst_enc,
        tcode=i32(code[tdst]),
        tpp=i32(tpp),
        pa=i32(pa[po]),
        pb=i32(pb[po]),
        flptr=i32(flptr),
        frow=i32(frow),
        fptr=i32(fptr),
        fcol=i32(Lcol[ro]),
        fpos=i32(posL[ro]),
        n_blevels=n_blevels,
        blptr=i32(blptr),
        brow=i32(brow),
        Lp=i32(Lp),
        Li=i32(Li),
    )


def reference_factor_solve(plan: LeanLUPlan, Ax: NDArray, b: NDArray) -> NDArray:
    """Serial numpy execution of the plan for ONE system -- the kernel's reference.

    ``Ax`` are the values of A in the CSR order the plan was built from, ``b`` the
    right-hand side. Slow (a Python loop over the tasks); for tests and debugging only.
    """
    val = np.zeros(plan.nnz_lu)
    val[plan.apos] = Ax
    W = plan.slots
    for lv in range(plan.n_levels):
        t0, t1 = plan.lptr[lv * W], plan.lptr[(lv + 1) * W]
        for t in range(t0, t1):
            if plan.tcode[t] == TASK_COPY:
                continue
            dst = int(plan.tdst[t]) & 0x7FFFFFFF
            acc = 0.0 if plan.tdst[t] < 0 else val[dst]
            s = slice(plan.tpp[t], plan.tpp[t + 1])
            acc -= np.dot(val[plan.pa[s]], val[plan.pb[s]])
            val[dst] = 1.0 / acc if plan.tcode[t] == TASK_DIAG else acc
        for t in range(t0, t1):
            if plan.tcode[t] >= 0:
                val[int(plan.tdst[t]) & 0x7FFFFFFF] *= val[plan.tcode[t]]
    y = np.zeros(plan.m)
    for lv in range(plan.n_levels):
        for t in range(plan.flptr[lv * W], plan.flptr[(lv + 1) * W]):
            i = plan.frow[t]
            s = slice(plan.fptr[i], plan.fptr[i + 1])
            y[i] = b[plan.perm[i]] - np.dot(val[plan.fpos[s]], y[plan.fcol[s]])
    for lv in range(plan.n_blevels):
        for t in range(plan.blptr[lv * W], plan.blptr[(lv + 1) * W]):
            i = plan.brow[t]
            lo, hi = plan.Lp[i], plan.Lp[i + 1]
            u = val[plan.base[i] + 1 : plan.base[i] + 1 + (hi - lo)]
            y[i] = (y[i] - np.dot(u, y[plan.Li[lo:hi]])) * val[plan.base[i]]
    x = np.empty(plan.m)
    x[plan.perm] = y
    return x
