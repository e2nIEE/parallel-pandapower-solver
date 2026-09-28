// SPDX-FileCopyrightText: 2026 Fraunhofer IEE
//
// SPDX-License-Identifier: BSD-3-Clause

// lean_lu.hpp -- static-pivot sparse LU refactorization on KLU's pivot order.
//
// klu_refactor re-runs the Gilbert-Peierls column algorithm on a fixed pivot order, but
// pays for KLU's generality on every call: packed "Unit" storage that interleaves row
// indices and values, per-call permutation and row scaling of A, and the BTF block loop.
// On graviton's pegase Jacobian (m=17036, 2.4 Mflop, flops/nnz(L+U-I)=11) that overhead
// is ~1/3 of the refactor: this kernel -- the same algorithm on flat CSC storage with a
// precomputed A->LU scatter and dependency lists -- measured 1.35-1.5x faster than
// klu_refactor, reproducing its factors to ~4e-16.
//
// (The DATE'16 NICSLU "map" variant, which also removes the dense work vector, measured
// within +-5% of this kernel here: the work vector is 17k doubles and lives in L2.)
//
// KLU still does everything that needs pivoting: klu_analyze (ordering), one klu_factor
// whose pivot order and L/U pattern this plan freezes, and the fallback whenever a
// refactor on the frozen order is rejected. The plan is read-only once built, so one
// plan can be shared by all threads; each thread owns only its LU values and scratch.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

extern "C" {
#include <klu.h>
}

struct LeanLU {
    bool valid = false;
    int n = 0;
    std::vector<int> P, Q;       // LU = A(P, Q): P[k] = row of A at pivot k, Q[k] = column
    // Values of column k live in [cp[k], cp[k+1]): U rows ascending, then the diagonal at
    // dpos[k], then L rows in KLU's order (L is unit-diagonal; its diagonal is not stored).
    std::vector<int> cp, dpos, row;
    // A -> LU scatter: column k gathers A data entries aSrc[aK[k]..aK[k+1]) into rows aRow.
    std::vector<int> aK, aSrc, aRow;
    // Dependencies of column k (U(j,k) != 0, j < k, ascending j): the U(j,k) slot and the
    // value range of L(:,j). Ascending j is a valid topological order for the column algorithm.
    std::vector<int> dK, dU, dLb, dLe;
    // min|pivot| / max|pivot| of the (unscaled) factorization the plan was built from, in
    // the same units lean_refactor returns.
    double rcond0 = 0.0;

    size_t nnz() const { return row.size(); }
};

// Freeze the pivot order and L/U pattern of a KLU factorization of the CSC matrix (Ap, Ai).
// Requires a single-block factorization (btf off, or an irreducible matrix). Returns false
// (plan left invalid) if the factorization cannot be expressed as one flat LU.
inline bool lean_build(LeanLU& pl, int n, const int* Ap, const int* Ai,
                       klu_symbolic* S, klu_numeric* N, klu_common* c) {
    pl = LeanLU();
    if (!S || !N || S->nblocks != 1 || N->nzoff != 0) return false;
    // klu_extract only fills L / U when the value arrays are passed too.
    std::vector<int> Lp(n + 1), Li(N->lnz), Up(n + 1), Ui(N->unz);
    std::vector<double> Lx(N->lnz), Ux(N->unz), Rs(n);
    pl.P.resize(n); pl.Q.resize(n);
    if (!klu_extract(N, S, Lp.data(), Li.data(), Lx.data(), Up.data(), Ui.data(), Ux.data(),
                     nullptr, nullptr, nullptr, pl.P.data(), pl.Q.data(), Rs.data(), nullptr, c))
        return false;
    pl.n = n;

    // Reference pivot ratio, in the units lean_refactor reports. KLU factors the row-scaled
    // P (R\A) Q = L U; the unscaled factorization on the same order has U' = D U with D the
    // row scales in pivot order -- and klu_factor already stores Rs permuted to pivot
    // order (klu_factor.c), so each unscaled pivot is Rs[k] * U(k,k).
    double pmin = INFINITY, pmax = 0.0;
    for (int k = 0; k < n; ++k)
        for (int p = Up[k]; p < Up[k + 1]; ++p)
            if (Ui[p] == k) {
                const double a = std::fabs(Rs[k] * Ux[p]);
                pmin = std::min(pmin, a);
                pmax = std::max(pmax, a);
            }
    if (!(pmax > 0.0) || !std::isfinite(pmax) || !(pmin > 0.0)) return false;
    pl.rcond0 = pmin / pmax;

    std::vector<int> Pinv(n);
    for (int k = 0; k < n; ++k) Pinv[pl.P[k]] = k;

    // Flat column layout: U part (ascending), diagonal, L part (ascending).
    pl.cp.assign(n + 1, 0);
    pl.dpos.assign(n, -1);
    std::vector<int> nu(n, 0), nl(n, 0);
    for (int k = 0; k < n; ++k) {
        for (int p = Up[k]; p < Up[k + 1]; ++p) if (Ui[p] != k) ++nu[k];
        for (int p = Lp[k]; p < Lp[k + 1]; ++p) if (Li[p] != k) ++nl[k];
        pl.cp[k + 1] = pl.cp[k] + nu[k] + 1 + nl[k];
    }
    pl.row.resize(pl.cp[n]);
    std::vector<int> Lbeg(n), Lend(n);
    int n_udep = 0;
    for (int k = 0; k < n; ++k) {
        int q = pl.cp[k];
        for (int p = Up[k]; p < Up[k + 1]; ++p) if (Ui[p] != k) pl.row[q++] = Ui[p];
        // U rows must be ascending (it is the elimination order); L rows may stay unsorted.
        std::sort(pl.row.begin() + pl.cp[k], pl.row.begin() + q);
        n_udep += nu[k];
        pl.dpos[k] = q;
        pl.row[q++] = k;
        Lbeg[k] = q;
        for (int p = Lp[k]; p < Lp[k + 1]; ++p) if (Li[p] != k) pl.row[q++] = Li[p];
        Lend[k] = q;
    }

    // Scatter map for A and the per-column dependency lists.
    std::vector<char> inpat(n, 0);
    pl.aK.assign(1, 0);
    pl.dK.assign(1, 0);
    pl.aK.reserve(n + 1); pl.dK.reserve(n + 1);
    pl.aSrc.reserve(Ap[n]); pl.aRow.reserve(Ap[n]);
    pl.dU.reserve(n_udep); pl.dLb.reserve(n_udep); pl.dLe.reserve(n_udep);
    for (int k = 0; k < n; ++k) {
        for (int q = pl.cp[k]; q < pl.cp[k + 1]; ++q) inpat[pl.row[q]] = 1;
        const int ac = pl.Q[k];
        for (int p = Ap[ac]; p < Ap[ac + 1]; ++p) {
            const int r = Pinv[Ai[p]];
            if (!inpat[r]) return false;              // A entry outside the LU pattern
            pl.aSrc.push_back(p);
            pl.aRow.push_back(r);
        }
        pl.aK.push_back((int)pl.aSrc.size());
        for (int q = pl.cp[k]; q < pl.dpos[k]; ++q) {
            const int j = pl.row[q];
            pl.dU.push_back(q);
            pl.dLb.push_back(Lbeg[j]);
            pl.dLe.push_back(Lend[j]);
        }
        pl.dK.push_back((int)pl.dU.size());
        for (int q = pl.cp[k]; q < pl.cp[k + 1]; ++q) inpat[pl.row[q]] = 0;
    }
    pl.valid = true;
    return true;
}

// Numeric refactorization of A (values Ax, same pattern the plan was built from) into LU
// (size pl.nnz()). x is a dense scratch vector of size n that must be all-zero on entry and
// is all-zero again on return. Returns min|pivot| / max|pivot|, or 0 if any pivot is zero
// or non-finite -- the caller compares it against rcond0 to reject an unsuitable order.
inline double lean_refactor(const LeanLU& pl, const double* Ax, double* LU, double* x) {
    const int* row = pl.row.data();
    double pmin = INFINITY, pmax = 0.0;
    bool finite = true;
    for (int k = 0; k < pl.n; ++k) {
        for (int e = pl.aK[k]; e < pl.aK[k + 1]; ++e) x[pl.aRow[e]] += Ax[pl.aSrc[e]];
        for (int d = pl.dK[k]; d < pl.dK[k + 1]; ++d) {
            const double u = x[row[pl.dU[d]]];
            if (u == 0.0) continue;
            for (int s = pl.dLb[d]; s < pl.dLe[d]; ++s) x[row[s]] -= u * LU[s];
        }
        const int dp = pl.dpos[k];
        for (int q = pl.cp[k]; q < dp; ++q) { const int r = row[q]; LU[q] = x[r]; x[r] = 0.0; }
        const double piv = x[k];
        x[k] = 0.0;
        LU[dp] = piv;
        const double a = std::fabs(piv);
        if (!(a > 0.0) || !std::isfinite(a)) finite = false;
        pmin = std::min(pmin, a);
        pmax = std::max(pmax, a);
        const double inv = 1.0 / piv;
        for (int q = dp + 1; q < pl.cp[k + 1]; ++q) { const int r = row[q]; LU[q] = x[r] * inv; x[r] = 0.0; }
    }
    if (!finite || pmax == 0.0) return 0.0;
    return pmin / pmax;
}

// Solve A x = b in place (b overwritten by x) using the LU from lean_refactor. y is scratch
// of size n.
inline void lean_solve(const LeanLU& pl, const double* LU, double* b, double* y) {
    const int n = pl.n;
    const int* row = pl.row.data();
    for (int k = 0; k < n; ++k) y[k] = b[pl.P[k]];
    for (int k = 0; k < n; ++k) {                       // L y = P b (unit diagonal)
        const double yk = y[k];
        if (yk == 0.0) continue;
        for (int q = pl.dpos[k] + 1; q < pl.cp[k + 1]; ++q) y[row[q]] -= LU[q] * yk;
    }
    for (int k = n - 1; k >= 0; --k) {                  // U z = y
        const double zk = y[k] / LU[pl.dpos[k]];
        y[k] = zk;
        if (zk == 0.0) continue;
        for (int q = pl.cp[k]; q < pl.dpos[k]; ++q) y[row[q]] -= LU[q] * zk;
    }
    for (int k = 0; k < n; ++k) b[pl.Q[k]] = y[k];      // x = Q z
}
