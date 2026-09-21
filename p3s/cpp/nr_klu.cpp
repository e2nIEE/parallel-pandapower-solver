// SPDX-FileCopyrightText: 2026 Fraunhofer IEE
//
// SPDX-License-Identifier: BSD-3-Clause

// Fast single-grid Newton-Raphson power flow (polar formulation) using KLU.
//
// Design:
//   * Polar formulation: Ybus stored as magnitude (Ym) + angle (Ya), constant
//     across iterations. Per iteration we only recompute one sincos(delta) per
//     nonzero and assemble the four Jacobian blocks via real multiply-adds.
//   * The Jacobian sparsity pattern is constant across iterations, so we build
//     the CSC structure ONCE, call klu_analyze ONCE, then klu_factor on the
//     first iteration and klu_refactor on every subsequent one (reuses the
//     pivot ordering -> much cheaper than a fresh factorization).
//
// The Jacobian layout (variables x = [dVa(pvpq); dVm(pq)]):
//     [ dP/dVa(pvpq,pvpq)   dP/dVm(pvpq,pq) ]
//     [ dQ/dVa(pq,  pvpq)   dQ/dVm(pq,  pq) ]
// rows: P-mismatch for pvpq, then Q-mismatch for pq.
//
// We construct J directly from Ybus (CSR) without forming dS_dV explicitly,
// using the standard polar derivative formulas.

#include <vector>
#include <complex>
#include <cmath>
#include <limits>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

extern "C" {
#include <klu.h>
}

namespace py = pybind11;
using cd = std::complex<double>;

// Portable sin+cos. POSIX/glibc expose ::sincos (one call, used on Linux); MSVC and
// other libcs do not, so fall back to separate std::sin/std::cos there (the compiler
// still fuses them well under fast-math).
static inline void nr_sincos(double x, double* s, double* c) {
#if defined(__GNUC__) && !defined(__clang__)
    // glibc/g++ (Linux): single fused sincos.
    ::sincos(x, s, c);
#else
    // MSVC / clang / other libc: no sincos -- separate calls (fused under fast-math).
    *s = std::sin(x);
    *c = std::cos(x);
#endif
}

// ----------------------------------------------------------------------------
// Topology: everything that depends ONLY on the grid topology (Ybus pattern +
// bus classification) and is computed ONCE. It is *read-only* during a solve, so
// a single Topology can be SHARED across threads that each solve a different
// operating point. The KLU symbolic factorization (ordering) is topology-only and
// thread-safe to share, so it lives here too.
//
// Ym/Ya hold |Y| and angle(Y). They are technically mutable (update_Y refreshes
// them when transformer taps change) but that happens single-threaded, before any
// batch, so they are safe to treat as shared-const during a (possibly threaded)
// batch solve.
// ----------------------------------------------------------------------------
struct Topology {
    int n = 0;            // number of buses
    int m = 0;            // Jacobian dimension = npvpq + npq

    // Ybus in CSR, split into magnitude / angle.
    std::vector<int> Yp, Yj;
    std::vector<double> Ym, Ya;     // |Y|, angle(Y)
    std::vector<int> Ydiag;         // index in CSR data of the diagonal of each row

    // bus-type index sets
    std::vector<int> pv, pq, pvpq;  // bus indices
    std::vector<int> pvpq_pos;      // bus -> position within pvpq block, else -1
    std::vector<int> pq_pos;        // bus -> position within pq block, else -1

    // Jacobian CSC structure (constant pattern).
    std::vector<int> Jp;            // size m+1 (column pointers)
    std::vector<int> Ji;            // row indices

    // For each Jacobian nonzero, where its value comes from:
    //   src_k     : index into Ybus CSR data of the (busrow,buscol) entry
    //   src_block : which of the 4 derivative blocks (0=dP/dVa,1=dP/dVm,2=dQ/dVa,3=dQ/dVm)
    std::vector<int> src_k;
    std::vector<char> src_block;

    // KLU symbolic factorization (ordering only) -- topology-dependent, shareable.
    // We keep a dedicated klu_common for symbolic ownership/free; per-solve numeric
    // work uses a SolveState's own klu_common (see below).
    klu_common SymCommon;
    klu_symbolic* Symbolic = nullptr;

    ~Topology() {
        if (Symbolic) klu_free_symbolic(&Symbolic, &SymCommon);
    }
};

// ----------------------------------------------------------------------------
// SolveState: the *mutable* per-solve state. Each worker thread owns one, so the
// numeric factorization and scratch buffers never alias across threads while a
// single shared Topology is read concurrently.
//
//   Jx               : Jacobian values (gathered each iteration from the topology)
//   Common/Numeric   : KLU numeric factorization (per-thread; NOT shareable)
//   Vm/Va            : working voltage (magnitude/angle)
//   Pspec/Qspec      : specified injections for this operating point
//   F/rhs            : mismatch and the linear-solve right-hand side
// ----------------------------------------------------------------------------
struct SolveState {
    std::vector<double> Jx;          // J values (size = topology nnz(J))
    std::vector<double> Vm, Va;      // working voltage state
    std::vector<double> Vm_ls, Va_ls;// line-search trial voltage scratch (size topo.n)
    std::vector<double> Pspec, Qspec;
    std::vector<double> F, rhs;

    // Optional per-solve Ybus values (magnitude/angle). When non-empty they OVERRIDE
    // the shared topo.Ym/topo.Ya for this solve -- used by the N-1 contingency batch,
    // where every case is the base Ybus with one outage group's branch stamps removed
    // (a values-only change on the shared sparsity pattern). Empty => use topo's values
    // (the time-series path). Same length/layout as topo.Ym/Ya (nnz of Ybus CSR).
    std::vector<double> Ym, Ya;

    // Optional per-bus pin mask (size topo.n, 1 = pinned). A pinned bus is frozen at its
    // start voltage: its Jacobian row(s) become an identity row and its mismatch is
    // zeroed, so the Newton step for that bus's variable(s) is exactly 0 while the
    // sparsity pattern (and the shared symbolic factorization) is unchanged. Used to
    // pin island reference buses (re-slack generators) and unserved buses. Empty => no
    // pinning.
    std::vector<char> pin;

    klu_common   Common;
    klu_numeric* Numeric = nullptr;
    bool factored = false;

    explicit SolveState(const Topology& topo) {
        Jx.assign(topo.Ji.size(), 0.0);
        Vm.assign(topo.n, 0.0);
        Va.assign(topo.n, 0.0);
        Vm_ls.assign(topo.n, 0.0);
        Va_ls.assign(topo.n, 0.0);
        Pspec.assign(topo.n, 0.0);
        Qspec.assign(topo.n, 0.0);
        F.assign(topo.m, 0.0);
        rhs.assign(topo.m, 0.0);
        klu_defaults(&Common);
    }
    ~SolveState() {
        if (Numeric) klu_free_numeric(&Numeric, &Common);
    }
    // non-copyable (owns a KLU numeric handle)
    SolveState(const SolveState&) = delete;
    SolveState& operator=(const SolveState&) = delete;
};

// find diagonal data index for each row of a CSR matrix
static void compute_diag_ix(const std::vector<int>& Yp, const std::vector<int>& Yj,
                            int n, std::vector<int>& Ydiag) {
    Ydiag.assign(n, -1);
    for (int r = 0; r < n; ++r)
        for (int k = Yp[r]; k < Yp[r + 1]; ++k)
            if (Yj[k] == r) { Ydiag[r] = k; break; }
}

// ----------------------------------------------------------------------------
// Build the constant CSC pattern of J and the per-nonzero source map.
// Column c of J corresponds to variable:
//   c <  npvpq        -> dVa at bus pvpq[c]
//   c >= npvpq        -> dVm at bus pq[c - npvpq]
// Row r of J corresponds to mismatch:
//   r <  npvpq        -> P at bus pvpq[r]
//   r >= npvpq        -> Q at bus pq[r - npvpq]
//
// A structural nonzero J(r,c) exists iff Ybus(busrow, buscol) != 0.
// We build it column-by-column (CSC) so KLU can consume it directly.
// ----------------------------------------------------------------------------
static void build_J_pattern(Topology& w) {
    const int npvpq = (int)w.pvpq.size();
    const int npq   = (int)w.pq.size();
    const int m = npvpq + npq;
    w.m = m;

    // We need, for a given bus column, the rows (in bus space) that connect to it.
    // Ybus is CSR (row-major). To iterate a column we need Ybus column access.
    // Build a transpose adjacency: for each bus col, list of (busrow, k) with Ybus(row,col).
    // Since the J pattern is symmetric-ish but we must be exact, gather from CSR.
    std::vector<std::vector<std::pair<int,int>>> col_entries(w.n); // bus col -> (bus row, csr k)
    for (int r = 0; r < w.n; ++r)
        for (int k = w.Yp[r]; k < w.Yp[r + 1]; ++k)
            col_entries[w.Yj[k]].push_back({r, k});

    w.Jp.assign(m + 1, 0);
    w.Ji.clear();
    w.src_k.clear();
    w.src_block.clear();

    auto emit_col = [&](int bus_col, bool col_is_vm) {
        // rows: P for pvpq (block dP/dVa if !col_is_vm else dP/dVm),
        //       Q for pq   (block dQ/dVa if !col_is_vm else dQ/dVm)
        for (auto& pr : col_entries[bus_col]) {
            int bus_row = pr.first;
            int k = pr.second;
            int rp = w.pvpq_pos[bus_row];
            if (rp != -1) {
                w.Ji.push_back(rp);                       // P-row
                w.src_k.push_back(k);
                w.src_block.push_back(col_is_vm ? 1 : 0); // dP/dVm : dP/dVa
            }
        }
        for (auto& pr : col_entries[bus_col]) {
            int bus_row = pr.first;
            int k = pr.second;
            int rq = w.pq_pos[bus_row];
            if (rq != -1) {
                w.Ji.push_back(npvpq + rq);               // Q-row
                w.src_k.push_back(k);
                w.src_block.push_back(col_is_vm ? 3 : 2); // dQ/dVm : dQ/dVa
            }
        }
    };

    int col = 0;
    for (int c = 0; c < npvpq; ++c) {        // dVa columns
        emit_col(w.pvpq[c], false);
        w.Jp[++col] = (int)w.Ji.size();
    }
    for (int c = 0; c < npq; ++c) {          // dVm columns
        emit_col(w.pq[c], true);
        w.Jp[++col] = (int)w.Ji.size();
    }
    // Jx is per-solve (SolveState), sized to w.Ji.size() when a SolveState is created.
}

// ----------------------------------------------------------------------------
// Fill J values for current Vm/Va. Polar derivatives.
//
// For an off-diagonal entry (i!=j) with delta = Va[i]-Ya[ij]-Va[j]:
//   dP_i/dVa_j = -Vm_i*Ym_ij*Vm_j*sin(delta)      (block 0, off-diag)  [note sign]
//   dQ_i/dVa_j =  Vm_i*Ym_ij*Vm_j*cos(delta)      (block 2)
//   dP_i/dVm_j =  Vm_i*Ym_ij*cos(delta)           (block 1)
//   dQ_i/dVm_j =  Vm_i*Ym_ij*sin(delta)           (block 3)
// Diagonal terms accumulate the negative row-sum (for dVa) and special dVm.
//
// We follow the convention used in p3s/loadflow_csr.cpp (verified against
// the scipy reference). To keep diagonal accumulation correct we compute, per
// Ybus row, the four "dPQ_dVma"-style data arrays exactly like loadflow_csr,
// then gather into J via (src_k, src_block).
// ----------------------------------------------------------------------------
// Fused: in ONE traversal of Ybus, compute both the bus power injections P/Q
// (-> mismatch F) and the four Jacobian derivative blocks, sharing one
// sincos(delta) per nonzero. This halves the trig work and the memory traffic
// vs computing mismatch and Jacobian separately.
//
//   Pcalc_i = sum_k Vm_i*Ym_ik*Vm_k*cos(delta_ik)
//   Qcalc_i = sum_k Vm_i*Ym_ik*Vm_k*sin(delta_ik),  delta_ik = Va_i-Ya_ik-Va_k
//
// Off-diagonal (i!=j) derivative entries and diagonal accumulation follow the
// polar formulas (validated against scipy to ~1e-11).
static void eval_F_and_J(const Topology& w, SolveState& st, const double* Vm, const double* Va,
                         const double* Pspec, const double* Qspec, double* F) {
    const int nnzY = (int)w.Yj.size();
    // Per-solve Ybus values override the topology's shared values when present (N-1
    // contingency: the case's outage removes branch stamps). Otherwise use topo's.
    const double* Ym = st.Ym.empty() ? w.Ym.data() : st.Ym.data();
    const double* Ya = st.Ya.empty() ? w.Ya.data() : st.Ya.data();
    static thread_local std::vector<double> buf, Pcalc, Qcalc;
    // Only GROW the buffers; do not zero-fill them. Every off-diagonal slot is assigned
    // unconditionally in the loop below (dP_dVa[k] = ... for row != col), and Pcalc/Qcalc
    // are written from the prow/qrow locals, so a blanket assign(4*nnzY, 0.0) was a dead
    // ~1.2 MB memset per iteration on pegase (4 * 37655 * 8 B). The ONLY slots that
    // accumulate (+=/-=) rather than being assigned are the four diagonals [dix], so
    // those -- and only those -- are cleared, below.
    if (buf.size() < (size_t)4 * nnzY) buf.resize((size_t)4 * nnzY);
    double* dP_dVa = buf.data();
    double* dQ_dVa = buf.data() + nnzY;
    double* dP_dVm = buf.data() + 2 * nnzY;
    double* dQ_dVm = buf.data() + 3 * nnzY;
    if ((int)Pcalc.size() < w.n) { Pcalc.resize(w.n); Qcalc.resize(w.n); }

    // Clear the accumulator slots (the per-row Ybus diagonal). Ydiag[row] is -1 only if a
    // row has no diagonal entry, which cannot happen for a real Ybus -- the loop below
    // already indexes buf[dix] unconditionally -- but guard anyway so a malformed input
    // degrades to a wrong answer rather than a stray write.
    for (int row = 0; row < w.n; ++row) {
        const int dix = w.Ydiag[row];
        if (dix < 0) continue;
        dP_dVa[dix] = 0.0; dQ_dVa[dix] = 0.0;
        dP_dVm[dix] = 0.0; dQ_dVm[dix] = 0.0;
    }

    for (int row = 0; row < w.n; ++row) {
        const int dix = w.Ydiag[row];
        double prow = 0.0, qrow = 0.0;
        for (int k = w.Yp[row]; k < w.Yp[row + 1]; ++k) {
            const int col = w.Yj[k];
            const double delta = Va[row] - Ya[k] - Va[col];
            double s, c;
            nr_sincos(delta, &s, &c);
            const double Vmrow_Ym = Vm[row] * Ym[k];
            const double vij = Vmrow_Ym * Vm[col];
            // power injection contribution (all entries, incl. diagonal)
            prow += vij * c;
            qrow += vij * s;
            if (row != col) {
                const double dpva =  vij * s;
                const double dqva = -vij * c;
                dP_dVa[k] =  dpva;
                dQ_dVa[k] =  dqva;
                dP_dVm[k] =  Vmrow_Ym * c;
                dQ_dVm[k] =  Vmrow_Ym * s;
                dP_dVa[dix] -= dpva;
                dQ_dVa[dix] -= dqva;
                dP_dVm[dix] += Ym[k] * Vm[col] * c;
                dQ_dVm[dix] += Ym[k] * Vm[col] * s;
            } else {
                // self term: delta = -Ya[diag], so c = cos(-Ya)=cos(Ya) (even, OK),
                // but s = sin(-Ya) = -sin(Ya). The formula needs sin(Ya), hence +s here
                // (the original loadflow_csr uses -2*Vm*Ym*sin(Ya) = +2*Vm*Ym*s).
                dP_dVm[dix] += 2.0 * Vm[row] * Ym[k] * c;
                dQ_dVm[dix] += 2.0 * Vm[row] * Ym[k] * s;
            }
        }
        Pcalc[row] = prow;
        Qcalc[row] = qrow;
    }

    // mismatch F = [ P(pvpq)-Pspec ; Q(pq)-Qspec ]
    const int npvpq = (int)w.pvpq.size();
    for (int ri = 0; ri < npvpq; ++ri) {
        const int row = w.pvpq[ri];
        F[ri] = Pcalc[row] - Pspec[row];
    }
    for (int ri = 0; ri < (int)w.pq.size(); ++ri) {
        const int row = w.pq[ri];
        F[npvpq + ri] = Qcalc[row] - Qspec[row];
    }

    // gather into J (CSC values) via source map -> the per-solve Jx in SolveState
    const int nzJ = (int)w.Ji.size();
    const double* blocks[4] = { dP_dVa, dP_dVm, dQ_dVa, dQ_dVm };
    double* Jx = st.Jx.data();
    for (int p = 0; p < nzJ; ++p)
        Jx[p] = blocks[(int)w.src_block[p]][w.src_k[p]];

    // --- pin step (N-1) -------------------------------------------------------
    // Freeze pinned buses by turning their Jacobian row(s) into identity rows and
    // zeroing their mismatch, so the Newton update for those variables is exactly 0.
    // This keeps each pinned bus at its start voltage WITHOUT changing the sparsity
    // pattern (and thus the shared symbolic factorization). A pinned bus has up to two
    // rows: its Va row in the pvpq block (always, if it is a pvpq bus) and its Vm row in
    // the pq block (if it is a pq bus). J is CSC, so we walk columns and rewrite entries
    // whose ROW is pinned: 0 off-diagonal, 1 on the (row==col) diagonal.
    if (!st.pin.empty()) {
        const char* pin = st.pin.data();
        // mark which Jacobian ROWS are pinned (row index -> pinned bus' var row)
        // rows: [0,npvpq) are Va of pvpq[r]; [npvpq, m) are Vm of pq[r-npvpq].
        for (int col = 0; col < w.m; ++col) {
            for (int p = w.Jp[col]; p < w.Jp[col + 1]; ++p) {
                const int r = w.Ji[p];
                // bus owning this row:
                int rbus;
                if (r < npvpq) rbus = w.pvpq[r];
                else           rbus = w.pq[r - npvpq];
                if (pin[rbus]) Jx[p] = (r == col) ? 1.0 : 0.0;
            }
        }
        // zero mismatch for pinned rows
        for (int r = 0; r < npvpq; ++r) if (pin[w.pvpq[r]]) F[r] = 0.0;
        for (int r = 0; r < (int)w.pq.size(); ++r) if (pin[w.pq[r]]) F[npvpq + r] = 0.0;
    }
}

// Mismatch-only infinity norm at a TRIAL voltage (Vm,Va). No Jacobian, one sincos per
// Ybus nonzero -- the cheapest primitive in the file. Used ONLY by the backtracking line
// search to score a candidate step.
//
// It must reproduce eval_F_and_J's mismatch exactly, so it mirrors two behaviours the old
// stale eval_F ignored:
//   * per-solve Ybus override: use st.Ym/st.Ya when non-empty (N-1 contingency) else topo's,
//   * pin mask: a pinned bus' mismatch rows are zeroed (frozen buses contribute nothing).
// Getting either wrong would make the line search accept/reject on a norm that disagrees
// with the actual convergence check, so keep this in lockstep with eval_F_and_J's F block.
static double eval_F_norm(const Topology& w, const SolveState& st,
                          const double* Vm, const double* Va) {
    const double* Ym = st.Ym.empty() ? w.Ym.data() : st.Ym.data();
    const double* Ya = st.Ya.empty() ? w.Ya.data() : st.Ya.data();
    const char* pin = st.pin.empty() ? nullptr : st.pin.data();
    const int npvpq = (int)w.pvpq.size();
    double nrm = 0.0;
    for (int ri = 0; ri < npvpq; ++ri) {
        const int row = w.pvpq[ri];
        if (pin && pin[row]) continue;              // pinned row -> mismatch is 0
        double p = 0.0;
        for (int k = w.Yp[row]; k < w.Yp[row + 1]; ++k) {
            const double delta = Va[row] - Ya[k] - Va[w.Yj[k]];
            p += Vm[row] * Ym[k] * Vm[w.Yj[k]] * std::cos(delta);
        }
        nrm = std::max(nrm, std::fabs(p - st.Pspec[row]));
    }
    for (int ri = 0; ri < (int)w.pq.size(); ++ri) {
        const int row = w.pq[ri];
        if (pin && pin[row]) continue;
        double q = 0.0;
        for (int k = w.Yp[row]; k < w.Yp[row + 1]; ++k) {
            const double delta = Va[row] - Ya[k] - Va[w.Yj[k]];
            q += Vm[row] * Ym[k] * Vm[w.Yj[k]] * std::sin(delta);
        }
        nrm = std::max(nrm, std::fabs(q - st.Qspec[row]));
    }
    return nrm;
}

// ----------------------------------------------------------------------------
// One full Newton-Raphson solve for a single operating point.
//
// Reads the shared (const) Topology and uses the per-solve SolveState `st` for all
// mutable work (Jx, KLU numeric, scratch). `st.Vm/Va` must be initialised with the
// start voltage and `st.Pspec/Qspec` with the specified injections before calling.
//
// Reuses the KLU symbolic factorization in `topo` (shared, read-only); the first
// iteration that needs a factorization calls klu_factor, later ones klu_refactor
// (cheap, pattern unchanged). When `st` is reused across operating points its
// Numeric handle persists, so even the first iteration of the 2nd+ solve refactors.
//
// Returns (iterations, converged). Thread-safe: touches only `topo` (const) and `st`.
// ----------------------------------------------------------------------------
struct NewtonResult { int iterations; bool converged; };

// Backtracking line-search parameters (compile-time; not exposed to Python).
//
// A full (alpha=1) step is ALWAYS tried first. It is accepted whenever it satisfies the
// Armijo SUFFICIENT-DECREASE test (below) -- true for good full steps -- so on well-
// behaved grids the search never backtracks and the only cost is one mismatch-norm
// evaluation per iteration, REUSED as the next iteration's convergence check (net zero
// extra work). Backtracking only engages when the full step overshoots.
//
// Acceptance is Armijo, not plain "any decrease": accept alpha iff
//     ||F(V + alpha*dV)||  <=  (1 - ARMIJO_C * alpha) * ||F(V)|| .
// Plain decrease (ARMIJO_C -> 0) lets the step shrink until the residual *just barely*
// dips, permitting arbitrarily small progress -> the iteration crawls and stalls on a
// plateau (observed on RTE grids). The (1 - c*alpha) bar forces progress PROPORTIONAL to
// the step actually taken, so accepted steps make provable headway and the search still
// terminates (dV = -J^-1 F is a descent direction, so small enough alpha always passes).
//
// If no trial down to alpha_min passes Armijo, commit the BEST (smallest-residual) trial
// seen rather than the first marginal one or the last (tiniest) alpha -- the least-bad
// step, at no extra cost since every trial's norm was computed anyway.
static constexpr double LS_BETA       = 0.5;   // step reduction factor per backtrack
static constexpr int    LS_MAX_TRIALS = 10;    // max backtracks (alpha down to ~1/1024)
static constexpr double ARMIJO_C      = 1e-4;  // sufficient-decrease constant

static NewtonResult run_newton(const Topology& topo, SolveState& st,
                               int max_iter, double tol, bool line_search = true) {
    const int npvpq = (int)topo.pvpq.size();
    const int npq   = (int)topo.pq.size();
    double* Vm = st.Vm.data();
    double* Va = st.Va.data();
    double* F  = st.F.data();
    double* rhs = st.rhs.data();
    double* Vm_ls = st.Vm_ls.data();
    double* Va_ls = st.Va_ls.data();

    // Current mismatch norm. eval_F_and_J fills F and the Jacobian at (Vm,Va); we track the
    // infinity-norm of F alongside so the line search can compare against it. After a step,
    // the accepted trial's norm becomes this value for the next iteration (fast-path reuse).
    eval_F_and_J(topo, st, Vm, Va, st.Pspec.data(), st.Qspec.data(), F);
    double nrm = 0; for (int i = 0; i < topo.m; ++i) nrm = std::max(nrm, std::fabs(F[i]));

    int it = 0;
    bool converged = false;
    while (it < max_iter) {
        if (nrm < tol) { converged = true; break; }
        ++it;

        // KLU factor (first time this state factors) or cheap refactor (pattern fixed).
        // A factorization failure here means the Jacobian became (numerically) singular
        // at this iterate -- e.g. a contingency that drives the grid to a degenerate /
        // non-convergent operating point (in N-1 some outages genuinely have no stable AC
        // solution; pandapower diverges on the same cases). That is a legitimate
        // "did not converge" outcome for THIS operating point, not a fatal error, so we
        // return non-converged rather than throwing -- which would abort an entire batch
        // for one bad contingency.
        if (!st.factored) {
            st.Numeric = klu_factor(const_cast<int*>(topo.Jp.data()),
                                    const_cast<int*>(topo.Ji.data()),
                                    st.Jx.data(), topo.Symbolic, &st.Common);
            if (!st.Numeric) return {it, false};
            st.factored = true;
        } else {
            int ok = klu_refactor(const_cast<int*>(topo.Jp.data()),
                                  const_cast<int*>(topo.Ji.data()),
                                  st.Jx.data(), topo.Symbolic, st.Numeric, &st.Common);
            if (!ok) {
                klu_free_numeric(&st.Numeric, &st.Common);
                st.Numeric = klu_factor(const_cast<int*>(topo.Jp.data()),
                                        const_cast<int*>(topo.Ji.data()),
                                        st.Jx.data(), topo.Symbolic, &st.Common);
                if (!st.Numeric) return {it, false};
            }
        }

        for (int i = 0; i < topo.m; ++i) rhs[i] = -F[i];
        if (!klu_solve(topo.Symbolic, st.Numeric, topo.m, 1, rhs, &st.Common))
            return {it, false};

        // --- step + guarded Armijo backtracking line search --------------------
        // Try the full Newton step first. If it passes the Armijo sufficient-decrease test
        // (the common case near the solution), accept it -- and reuse its norm as the next
        // iteration's convergence check, so the fast path adds no extra mismatch evals.
        // Only on overshoot do we backtrack, each trial costing one cheap eval_F_norm
        // (no Jacobian, no factor, no solve). A helper forms V + alpha*dV into Vm_ls/Va_ls.
        auto form_trial = [&](double a) {
            for (int i = 0; i < topo.n; ++i) { Vm_ls[i] = Vm[i]; Va_ls[i] = Va[i]; }
            for (int i = 0; i < npvpq; ++i) Va_ls[topo.pvpq[i]] += a * rhs[i];
            for (int i = 0; i < npq;   ++i) Vm_ls[topo.pq[i]]   += a * rhs[npvpq + i];
        };

        double alpha = 1.0;
        double nrm_new = nrm;
        double best_alpha = 1.0, best_nrm = std::numeric_limits<double>::infinity();
        for (int trial = 0; trial <= (line_search ? LS_MAX_TRIALS : 0); ++trial) {
            form_trial(alpha);
            nrm_new = eval_F_norm(topo, st, Vm_ls, Va_ls);
            if (nrm_new < best_nrm) { best_nrm = nrm_new; best_alpha = alpha; }
            // Fast path when disabled: take the single alpha=1 evaluation as-is.
            // Otherwise accept on Armijo sufficient decrease.
            if (!line_search || nrm_new <= (1.0 - ARMIJO_C * alpha) * nrm) break;
            if (trial == LS_MAX_TRIALS) {
                // No alpha passed Armijo -> commit the least-bad (smallest-residual) trial.
                if (best_alpha != alpha) { form_trial(best_alpha); nrm_new = best_nrm; }
                break;
            }
            alpha *= LS_BETA;
        }

        // commit the accepted trial voltage (currently held in Vm_ls/Va_ls)
        for (int i = 0; i < npvpq; ++i) Va[topo.pvpq[i]] = Va_ls[topo.pvpq[i]];
        for (int i = 0; i < npq;   ++i) Vm[topo.pq[i]]   = Vm_ls[topo.pq[i]];

        // Rebuild F and J at the accepted point for the next iteration's solve; the norm
        // is recomputed here (it equals nrm_new, but eval_F_and_J is needed anyway for J).
        eval_F_and_J(topo, st, Vm, Va, st.Pspec.data(), st.Qspec.data(), F);
        nrm = 0; for (int i = 0; i < topo.m; ++i) nrm = std::max(nrm, std::fabs(F[i]));
    }
    return {it, converged};
}

// Build the shared Topology (pattern + symbolic analyze) from CSR Ybus + pv/pq.
static void build_topology(Topology& topo,
                           const int* Yp, const int* Yj, const std::complex<double>* Yx,
                           int n, int nnzY,
                           const int* pv, int npv, const int* pq, int npq,
                           int ordering, int btf) {
    topo.n = n;
    topo.Yp.assign(Yp, Yp + n + 1);
    topo.Yj.assign(Yj, Yj + nnzY);
    topo.Ym.resize(nnzY); topo.Ya.resize(nnzY);
    for (int k = 0; k < nnzY; ++k) { topo.Ym[k] = std::abs(Yx[k]); topo.Ya[k] = std::arg(Yx[k]); }
    compute_diag_ix(topo.Yp, topo.Yj, n, topo.Ydiag);

    topo.pv.assign(pv, pv + npv);
    topo.pq.assign(pq, pq + npq);
    topo.pvpq = topo.pv; topo.pvpq.insert(topo.pvpq.end(), topo.pq.begin(), topo.pq.end());
    topo.pvpq_pos.assign(n, -1);
    for (int i = 0; i < (int)topo.pvpq.size(); ++i) topo.pvpq_pos[topo.pvpq[i]] = i;
    topo.pq_pos.assign(n, -1);
    for (int i = 0; i < (int)topo.pq.size(); ++i) topo.pq_pos[topo.pq[i]] = i;

    build_J_pattern(topo);
    klu_defaults(&topo.SymCommon);
    topo.SymCommon.ordering = ordering;
    topo.SymCommon.btf = btf;
    topo.Symbolic = klu_analyze(topo.m, topo.Jp.data(), topo.Ji.data(), &topo.SymCommon);
    if (!topo.Symbolic) throw std::runtime_error("klu_analyze failed");
}

// Run `body(t)` for t in [0, T): serially if n_threads<=1 or OpenMP is unavailable,
// otherwise with an OpenMP parallel-for using min(n_threads, T, hw_concurrency)
// threads (n_threads==0 means "all hardware threads"). The body must be independent
// across t (each builds its own SolveState), so the result is order-independent.
template <typename Body>
static void run_batch_loop(Body&& body, int T, int n_threads) {
#ifdef _OPENMP
    int hw = (int)std::thread::hardware_concurrency();
    if (hw <= 0) hw = 1;
    int nt = (n_threads <= 0) ? hw : n_threads;
    if (nt > T) nt = T;
    if (nt > hw) nt = hw;
    if (nt < 1) nt = 1;
    if (nt == 1) {
        for (int t = 0; t < T; ++t) body(t);
    } else {
        #pragma omp parallel for num_threads(nt) schedule(dynamic)
        for (int t = 0; t < T; ++t) body(t);
    }
#else
    (void)n_threads;
    for (int t = 0; t < T; ++t) body(t);
#endif
}

// ----------------------------------------------------------------------------
// pybind entry point. Inputs are numpy arrays (zero-copy where possible).
//   Yp,Yj : CSR structure of Ybus (int32)
//   Yx    : complex128 CSR data
//   Sbus  : complex128 power injections (specified P+jQ), length n
//   V0    : complex128 initial voltage, length n
//   pv,pq : int32 bus index sets
// Returns dict with V (complex128), iterations, converged, and timing.
// ----------------------------------------------------------------------------
py::dict solve_single(
    py::array_t<int, py::array::c_style | py::array::forcecast> Yp_in,
    py::array_t<int, py::array::c_style | py::array::forcecast> Yj_in,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> Yx_in,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> Sbus_in,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> V0_in,
    py::array_t<int, py::array::c_style | py::array::forcecast> pv_in,
    py::array_t<int, py::array::c_style | py::array::forcecast> pq_in,
    int max_iter = 30,
    double tol = 1e-8,
    int ordering = 0,   // KLU: 0=AMD, 1=COLAMD, 2=natural, 3=user/CHOLMOD
    int btf = 0,        // KLU: BTF pre-ordering. OFF is faster for connected PF grids
                        // (the Jacobian is one irreducible block, so BTF only adds cost).
    bool line_search = true)  // backtracking line search (damped Newton); see run_newton
{
    auto Yp = Yp_in.unchecked<1>();
    auto Yj = Yj_in.unchecked<1>();
    auto Yx = Yx_in.unchecked<1>();
    auto Sb = Sbus_in.unchecked<1>();
    auto V0 = V0_in.unchecked<1>();
    auto pvb = pv_in.unchecked<1>();
    auto pqb = pq_in.unchecked<1>();

    const int n = (int)Yp.shape(0) - 1;
    const int nnzY = (int)Yj.shape(0);

    // pack the numpy inputs into contiguous host buffers for build_topology
    std::vector<int> Yp_v(Yp.data(0), Yp.data(0) + n + 1);
    std::vector<int> Yj_v(Yj.data(0), Yj.data(0) + nnzY);
    std::vector<std::complex<double>> Yx_v(nnzY);
    for (int k = 0; k < nnzY; ++k) Yx_v[k] = Yx(k);
    std::vector<int> pv_v(pvb.shape(0)); for (int i = 0; i < (int)pvb.shape(0); ++i) pv_v[i] = pvb(i);
    std::vector<int> pq_v(pqb.shape(0)); for (int i = 0; i < (int)pqb.shape(0); ++i) pq_v[i] = pqb(i);

    auto t_setup0 = std::chrono::high_resolution_clock::now();
    Topology topo;
    build_topology(topo, Yp_v.data(), Yj_v.data(), Yx_v.data(), n, nnzY,
                   pv_v.data(), (int)pv_v.size(), pq_v.data(), (int)pq_v.size(),
                   ordering, btf);
    auto t_setup1 = std::chrono::high_resolution_clock::now();

    SolveState st(topo);
    for (int i = 0; i < n; ++i) { st.Pspec[i] = Sb(i).real(); st.Qspec[i] = Sb(i).imag(); }
    for (int i = 0; i < n; ++i) { st.Vm[i] = std::abs(V0(i)); st.Va[i] = std::arg(V0(i)); }

    auto t_nr0 = std::chrono::high_resolution_clock::now();
    NewtonResult nr = run_newton(topo, st, max_iter, tol, line_search);
    auto t_nr1 = std::chrono::high_resolution_clock::now();
    int it = nr.iterations;
    bool converged = nr.converged;

    // assemble complex result
    py::array_t<std::complex<double>> Vout(n);
    auto Vo = Vout.mutable_unchecked<1>();
    for (int i = 0; i < n; ++i) Vo(i) = std::polar(st.Vm[i], st.Va[i]);

    py::dict res;
    res["V"] = Vout;
    res["iterations"] = it;
    res["converged"] = converged;
    // Timing: setup (pattern + symbolic analyze) and the full Newton solve. The
    // previous fine-grained factor/refactor/klusolve split was removed when the
    // Newton loop moved into the shared run_newton(); t_assemble is kept as a key
    // (0.0) only for backward-compatible callers.
    res["t_setup_ms"]    = std::chrono::duration<double, std::milli>(t_setup1 - t_setup0).count();
    res["t_solve_ms"]    = std::chrono::duration<double, std::milli>(t_nr1 - t_nr0).count();
    res["t_assemble_ms"] = 0.0;
    return res;
}

// ----------------------------------------------------------------------------
// Stateful solver: amortizes topology-dependent setup (pattern build + KLU
// symbolic analyze) across many solves on the SAME grid topology. This is the
// common p3s case: one network, many load/generation profiles.
//
//   s = Solver(Yp, Yj, Yx, pv, pq, ordering=0, btf=0)   # analyze ONCE
//   V = s.solve(Sbus, V0)                                # many times, cheap
//
// The KLU Numeric handle persists between solves, so the 2nd+ call's first
// iteration uses klu_refactor (cheap) instead of klu_factor.
// ----------------------------------------------------------------------------
class Solver {
public:
    Solver(py::array_t<int, py::array::c_style | py::array::forcecast> Yp_in,
           py::array_t<int, py::array::c_style | py::array::forcecast> Yj_in,
           py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> Yx_in,
           py::array_t<int, py::array::c_style | py::array::forcecast> pv_in,
           py::array_t<int, py::array::c_style | py::array::forcecast> pq_in,
           int ordering = 0, int btf = 0)
    {
        auto Yp = Yp_in.unchecked<1>();
        auto Yj = Yj_in.unchecked<1>();
        auto Yx = Yx_in.unchecked<1>();
        auto pvb = pv_in.unchecked<1>();
        auto pqb = pq_in.unchecked<1>();

        const int n = (int)Yp.shape(0) - 1;
        const int nnzY = (int)Yj.shape(0);
        std::vector<int> Yp_v(Yp.data(0), Yp.data(0) + n + 1);
        std::vector<int> Yj_v(Yj.data(0), Yj.data(0) + nnzY);
        std::vector<std::complex<double>> Yx_v(nnzY);
        for (int k = 0; k < nnzY; ++k) Yx_v[k] = Yx(k);
        std::vector<int> pv_v(pvb.shape(0)); for (int i = 0; i < (int)pvb.shape(0); ++i) pv_v[i] = pvb(i);
        std::vector<int> pq_v(pqb.shape(0)); for (int i = 0; i < (int)pqb.shape(0); ++i) pq_v[i] = pqb(i);

        build_topology(topo, Yp_v.data(), Yj_v.data(), Yx_v.data(), n, nnzY,
                       pv_v.data(), (int)pv_v.size(), pq_v.data(), (int)pq_v.size(),
                       ordering, btf);
        // one persistent SolveState for single-grid sequential solve() calls; its
        // Numeric handle persists so 2nd+ solves refactor rather than factor.
        st = std::make_unique<SolveState>(topo);
    }

    // Refresh the Ybus values (magnitude/angle) without redoing the symbolic
    // analyze. Use when the topology (sparsity pattern + pv/pq) is unchanged but
    // the admittance values changed (e.g. transformer tap update). O(nnz).
    void update_Y(
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> Yx_in)
    {
        auto Yx = Yx_in.unchecked<1>();
        const int nnzY = (int)topo.Yj.size();
        if ((int)Yx.shape(0) != nnzY)
            throw std::runtime_error("update_Y: nnz mismatch (topology changed; rebuild Solver)");
        for (int k = 0; k < nnzY; ++k) {
            const cd y = Yx(k);
            topo.Ym[k] = std::abs(y);
            topo.Ya[k] = std::arg(y);
        }
    }

    py::dict solve(
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> Sbus_in,
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> V0_in,
        int max_iter = 30, double tol = 1e-8, bool line_search = true)
    {
        auto Sb = Sbus_in.unchecked<1>();
        auto V0 = V0_in.unchecked<1>();
        const int n = topo.n;
        for (int i = 0; i < n; ++i) {
            st->Pspec[i] = Sb(i).real(); st->Qspec[i] = Sb(i).imag();
            st->Vm[i] = std::abs(V0(i)); st->Va[i] = std::arg(V0(i));
        }
        NewtonResult nr = run_newton(topo, *st, max_iter, tol, line_search);

        py::array_t<std::complex<double>> Vout(n);
        auto Vo = Vout.mutable_unchecked<1>();
        for (int i = 0; i < n; ++i) Vo(i) = std::polar(st->Vm[i], st->Va[i]);

        py::dict res;
        res["V"] = Vout;
        res["iterations"] = nr.iterations;
        res["converged"] = nr.converged;
        return res;
    }

    // Solve a BATCH of operating points (columns of Sbus_mat) that share this
    // topology. Sbus_mat is (n, T) complex128; V0 is either (n, T) per-column starts
    // or (n,) a single start broadcast to all columns. Each column is an independent
    // Newton solve reusing the shared symbolic factorization.
    //
    // n_threads: 0 = serial (single-threaded loop); >=1 = OpenMP with that many
    // threads (capped to T and the hardware concurrency). Results are identical
    // regardless of thread count (columns are independent).
    //
    // Returns { V: (n, T) complex128, iterations: (T,) int32, converged: (T,) bool }.
    py::dict solve_batch(
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> Sbus_mat,
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> V0_in,
        int max_iter = 30, double tol = 1e-8, int n_threads = 0, bool line_search = true)
    {
        if (Sbus_mat.ndim() != 2)
            throw std::runtime_error("solve_batch: Sbus must be 2-D (n, T)");
        const int n = topo.n;
        if ((int)Sbus_mat.shape(0) != n)
            throw std::runtime_error("solve_batch: Sbus rows must equal n_bus");
        const int T = (int)Sbus_mat.shape(1);
        if (T == 0)
            throw std::runtime_error("solve_batch: T (time steps) must be > 0");

        if (max_iter <= 0)
            throw std::runtime_error("solve_batch: max_iter must be > 0");
        if (tol <= 0.0)
            throw std::runtime_error("solve_batch: tol must be > 0");
        if (n_threads < 0)
            throw std::runtime_error("solve_batch: n_threads must be >= 0");

        const bool v0_per_col = (V0_in.ndim() == 2);
        if (v0_per_col) {
            if ((int)V0_in.shape(0) != n || (int)V0_in.shape(1) != T)
                throw std::runtime_error("solve_batch: V0 (n,T) shape mismatch");
        } else if ((int)V0_in.shape(0) != n) {
            throw std::runtime_error("solve_batch: V0 (n,) length mismatch");
        }

        // copy inputs into contiguous host buffers (column-major access per step)
        auto Sb = Sbus_mat.unchecked<2>();
        std::vector<std::complex<double>> Sbuf((size_t)n * T), V0buf((size_t)n * (v0_per_col ? T : 1));
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < n; ++i) Sbuf[(size_t)t * n + i] = Sb(i, t);
        if (v0_per_col) {
            auto V0m = V0_in.unchecked<2>();
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < n; ++i) V0buf[(size_t)t * n + i] = V0m(i, t);
        } else {
            auto V0v = V0_in.unchecked<1>();
            for (int i = 0; i < n; ++i) V0buf[i] = V0v(i);
        }

        // outputs
        py::array_t<std::complex<double>> Vout({n, T});
        py::array_t<int> iters_out(T);
        py::array_t<bool> conv_out(T);
        auto Vo = Vout.mutable_unchecked<2>();
        auto io = iters_out.mutable_unchecked<1>();
        auto co = conv_out.mutable_unchecked<1>();

        // raw pointers for the (GIL-released) compute region
        std::complex<double>* Vptr = Vout.mutable_data();
        int* iptr = iters_out.mutable_data();
        bool* cptr = conv_out.mutable_data();
        const std::complex<double>* Sptr = Sbuf.data();
        const std::complex<double>* V0ptr = V0buf.data();
        const Topology& topo_ref = topo;
        std::string err;  // first error captured from any worker

        // Bounds check before parallel loop (using internal buffer sizes)
        if (Sbuf.size() < (size_t)n * T)
            throw std::runtime_error("solve_batch: Sbuf buffer too small");
        if (V0buf.size() < (size_t)n * (v0_per_col ? T : 1))
            throw std::runtime_error("solve_batch: V0buf buffer too small");
        // Vout buffer size is guaranteed by py::array_t constructor with correct shape

        // per-column work: build a fresh SolveState (own KLU numeric), solve, store.
        auto do_col = [&](int t) {
            try {
                // Validate column index
                if (t < 0 || t >= T)
                    throw std::runtime_error("solve_batch: invalid time-step index " + std::to_string(t));

                SolveState s(topo_ref);
                const std::complex<double>* Sc = Sptr + (size_t)t * n;
                const std::complex<double>* V0c = v0_per_col ? (V0ptr + (size_t)t * n) : V0ptr;
                for (int i = 0; i < n; ++i) {
                    // Validate voltage values before using
                    if (!std::isfinite(std::abs(V0c[i])) || !std::isfinite(std::arg(V0c[i])))
                        throw std::runtime_error("solve_batch: invalid initial voltage at bus " + std::to_string(i) + ", step " + std::to_string(t));
                    s.Pspec[i] = Sc[i].real(); s.Qspec[i] = Sc[i].imag();
                    s.Vm[i] = std::abs(V0c[i]); s.Va[i] = std::arg(V0c[i]);
                    // Ensure valid initial voltage magnitude
                    if (s.Vm[i] <= 0.0 || s.Vm[i] > 10.0)
                        throw std::runtime_error("solve_batch: invalid voltage magnitude " + std::to_string(s.Vm[i]) + " at bus " + std::to_string(i));
                }
                NewtonResult nr = run_newton(topo_ref, s, max_iter, tol, line_search);
                for (int i = 0; i < n; ++i)
                    Vptr[(size_t)i * T + t] = std::polar(s.Vm[i], s.Va[i]); // row-major (n,T)
                iptr[t] = nr.iterations;
                cptr[t] = nr.converged;
            } catch (const std::exception& e) {
                #pragma omp critical
                { if (err.empty()) err = e.what(); }
            }
        };

        // Release the GIL across the compute region: do_col touches only raw C++
        // buffers and KLU (no Python API), so worker threads can run truly parallel.
        {
            py::gil_scoped_release release;
            run_batch_loop(do_col, T, n_threads);
        }

        if (!err.empty())
            throw std::runtime_error("solve_batch: " + err);

        py::dict res;
        res["V"] = Vout;
        res["iterations"] = iters_out;
        res["converged"] = conv_out;
        return res;
    }

    // Solve a BATCH of N-1 contingencies that share this topology's sparsity pattern.
    // Each case is the base Ybus with one outage group's branch stamps removed -- a
    // values-only change -- so the KLU symbolic factorization is shared across the batch
    // exactly like solve_batch (the time-series path).
    //
    //   Yx_mag, Yx_ang : (nnz, L) float64  per-case Ybus magnitude/angle (nnz = Ybus CSR
    //                    nnz; same Yp/Yj as this Solver). Column c = case c.
    //   Sbus           : (n,) complex128   specified injections (constant across cases).
    //   V0             : (n, L) complex128 per-case start voltage. For pinned buses this
    //                    MUST hold the value to freeze them at (reference setpoint for a
    //                    re-slack generator; a placeholder for unserved buses).
    //   pin            : (n, L) uint8       per-case pin mask (1 = freeze this bus).
    //
    // Returns { V: (n, L) complex128, iterations: (L,) int32, converged: (L,) bool }.
    // Unserved buses are returned as solved here (at their placeholder); the Python
    // wrapper overwrites them with NaN using the served mask.
    py::dict solve_batch_contingency(
        py::array_t<double, py::array::c_style | py::array::forcecast> Yx_mag,
        py::array_t<double, py::array::c_style | py::array::forcecast> Yx_ang,
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> Sbus_in,
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> V0_in,
        py::array_t<unsigned char, py::array::c_style | py::array::forcecast> pin_in,
        int max_iter = 30, double tol = 1e-8, int n_threads = 0, bool line_search = true)
    {
        const int n = topo.n;
        const int nnzY = (int)topo.Yj.size();
        if (Yx_mag.ndim() != 2 || Yx_ang.ndim() != 2)
            throw std::runtime_error("solve_batch_contingency: Yx_mag/ang must be 2-D (nnz, L)");
        if ((int)Yx_mag.shape(0) != nnzY)
            throw std::runtime_error("solve_batch_contingency: Yx rows must equal Ybus nnz");
        const int L = (int)Yx_mag.shape(1);
        if ((int)Yx_ang.shape(0) != nnzY || (int)Yx_ang.shape(1) != L)
            throw std::runtime_error("solve_batch_contingency: Yx_mag/ang shape mismatch");
        if (V0_in.ndim() != 2 || (int)V0_in.shape(0) != n || (int)V0_in.shape(1) != L)
            throw std::runtime_error("solve_batch_contingency: V0 must be (n, L)");
        if (pin_in.ndim() != 2 || (int)pin_in.shape(0) != n || (int)pin_in.shape(1) != L)
            throw std::runtime_error("solve_batch_contingency: pin must be (n, L)");

        // copy inputs into contiguous per-case host buffers (column = case)
        auto Ymg = Yx_mag.unchecked<2>();
        auto Yag = Yx_ang.unchecked<2>();
        auto V0m = V0_in.unchecked<2>();
        auto pinm = pin_in.unchecked<2>();
        std::vector<double> Ymbuf((size_t)nnzY * L), Yabuf((size_t)nnzY * L);
        std::vector<std::complex<double>> V0buf((size_t)n * L);
        std::vector<char> pinbuf((size_t)n * L);
        for (int t = 0; t < L; ++t) {
            for (int k = 0; k < nnzY; ++k) {
                Ymbuf[(size_t)t * nnzY + k] = Ymg(k, t);
                Yabuf[(size_t)t * nnzY + k] = Yag(k, t);
            }
            for (int i = 0; i < n; ++i) {
                V0buf[(size_t)t * n + i] = V0m(i, t);
                pinbuf[(size_t)t * n + i] = (char)pinm(i, t);
            }
        }

        std::vector<std::complex<double>> Sbuf(n);
        auto Sb = Sbus_in.unchecked<1>();
        for (int i = 0; i < n; ++i) Sbuf[i] = Sb(i);

        py::array_t<std::complex<double>> Vout({n, L});
        py::array_t<int> iters_out(L);
        py::array_t<bool> conv_out(L);
        std::complex<double>* Vptr = Vout.mutable_data();
        int* iptr = iters_out.mutable_data();
        bool* cptr = conv_out.mutable_data();
        const double* Ymptr = Ymbuf.data();
        const double* Yaptr = Yabuf.data();
        const std::complex<double>* V0ptr = V0buf.data();
        const std::complex<double>* Sptr = Sbuf.data();
        const char* pinptr = pinbuf.data();
        const Topology& topo_ref = topo;
        std::string err;

        auto do_col = [&](int t) {
            try {
                SolveState s(topo_ref);
                // per-case Ybus values override the shared topo values
                s.Ym.assign(Ymptr + (size_t)t * nnzY, Ymptr + (size_t)(t + 1) * nnzY);
                s.Ya.assign(Yaptr + (size_t)t * nnzY, Yaptr + (size_t)(t + 1) * nnzY);
                s.pin.assign(pinptr + (size_t)t * n, pinptr + (size_t)(t + 1) * n);
                const std::complex<double>* V0c = V0ptr + (size_t)t * n;
                for (int i = 0; i < n; ++i) {
                    s.Pspec[i] = Sptr[i].real(); s.Qspec[i] = Sptr[i].imag();
                    s.Vm[i] = std::abs(V0c[i]); s.Va[i] = std::arg(V0c[i]);
                }
                NewtonResult nr = run_newton(topo_ref, s, max_iter, tol, line_search);
                for (int i = 0; i < n; ++i)
                    Vptr[(size_t)i * L + t] = std::polar(s.Vm[i], s.Va[i]);
                iptr[t] = nr.iterations;
                cptr[t] = nr.converged;
            } catch (const std::exception& e) {
                #pragma omp critical
                { if (err.empty()) err = e.what(); }
            }
        };

        {
            py::gil_scoped_release release;
            run_batch_loop(do_col, L, n_threads);
        }
        if (!err.empty())
            throw std::runtime_error("solve_batch_contingency: " + err);

        py::dict res;
        res["V"] = Vout;
        res["iterations"] = iters_out;
        res["converged"] = conv_out;
        return res;
    }

    // return (Jp, Ji, Jx) of the Jacobian built at V0 — for debugging vs scipy
    py::dict debug_J_at(
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> Sbus_in,
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> V0_in)
    {
        auto Sb = Sbus_in.unchecked<1>();
        auto V0 = V0_in.unchecked<1>();
        const int n = topo.n;
        SolveState s(topo);
        for (int i = 0; i < n; ++i) {
            s.Pspec[i] = Sb(i).real(); s.Qspec[i] = Sb(i).imag();
            s.Vm[i] = std::abs(V0(i)); s.Va[i] = std::arg(V0(i));
        }
        eval_F_and_J(topo, s, s.Vm.data(), s.Va.data(), s.Pspec.data(), s.Qspec.data(), s.F.data());
        py::array_t<int> Jp(topo.Jp.size()); auto jp = Jp.mutable_unchecked<1>();
        for (int i = 0; i < (int)topo.Jp.size(); ++i) jp(i) = topo.Jp[i];
        py::array_t<int> Ji(topo.Ji.size()); auto ji = Ji.mutable_unchecked<1>();
        for (int i = 0; i < (int)topo.Ji.size(); ++i) ji(i) = topo.Ji[i];
        py::array_t<double> Jx(s.Jx.size()); auto jx = Jx.mutable_unchecked<1>();
        for (int i = 0; i < (int)s.Jx.size(); ++i) jx(i) = s.Jx[i];
        py::dict r; r["Jp"] = Jp; r["Ji"] = Ji; r["Jx"] = Jx; r["m"] = topo.m; return r;
    }

private:
    Topology topo;                       // shared, read-only during solves
    std::unique_ptr<SolveState> st;      // persistent state for sequential solve()
};

PYBIND11_MODULE(nr_klu, m) {
    m.doc() = "Fast single-grid polar Newton-Raphson power flow (KLU)";
    m.def("solve_single", &solve_single,
          py::arg("Yp"), py::arg("Yj"), py::arg("Yx"), py::arg("Sbus"),
          py::arg("V0"), py::arg("pv"), py::arg("pq"),
          py::arg("max_iter") = 30, py::arg("tol") = 1e-8,
          py::arg("ordering") = 0, py::arg("btf") = 0,
          py::arg("line_search") = true);

    // debug helper: return the J (CSC) built at V0 for comparison with scipy
    m.def("debug_J", [](
        py::array_t<int, py::array::c_style | py::array::forcecast> Yp_in,
        py::array_t<int, py::array::c_style | py::array::forcecast> Yj_in,
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> Yx_in,
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> Sbus_in,
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> V0_in,
        py::array_t<int, py::array::c_style | py::array::forcecast> pv_in,
        py::array_t<int, py::array::c_style | py::array::forcecast> pq_in) {
        Solver s(Yp_in, Yj_in, Yx_in, pv_in, pq_in, 0, 0);
        return s.debug_J_at(Sbus_in, V0_in);
    });

    py::class_<Solver>(m, "Solver")
        .def(py::init<py::array_t<int, py::array::c_style | py::array::forcecast>,
                      py::array_t<int, py::array::c_style | py::array::forcecast>,
                      py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>,
                      py::array_t<int, py::array::c_style | py::array::forcecast>,
                      py::array_t<int, py::array::c_style | py::array::forcecast>,
                      int, int>(),
             py::arg("Yp"), py::arg("Yj"), py::arg("Yx"), py::arg("pv"), py::arg("pq"),
             py::arg("ordering") = 0, py::arg("btf") = 0)
        .def("update_Y", &Solver::update_Y, py::arg("Yx"),
             "Refresh Ybus values without redoing the symbolic analyze.")
        .def("solve", &Solver::solve,
             py::arg("Sbus"), py::arg("V0"), py::arg("max_iter") = 30, py::arg("tol") = 1e-8,
             py::arg("line_search") = true)
        .def("solve_batch", &Solver::solve_batch,
             py::arg("Sbus"), py::arg("V0"), py::arg("max_iter") = 30, py::arg("tol") = 1e-8,
             py::arg("n_threads") = 0, py::arg("line_search") = true,
             "Solve a batch of operating points (columns of Sbus, shape (n,T)) sharing "
             "this topology. n_threads: 0=all cores (OpenMP), 1=serial. line_search: "
             "backtracking damped Newton (default on), pass False to disable. Returns dict "
             "with V (n,T), iterations (T,), converged (T,).")
        .def("solve_batch_contingency", &Solver::solve_batch_contingency,
             py::arg("Yx_mag"), py::arg("Yx_ang"), py::arg("Sbus"), py::arg("V0"),
             py::arg("pin"), py::arg("max_iter") = 30, py::arg("tol") = 1e-8,
             py::arg("n_threads") = 0, py::arg("line_search") = true,
             "Solve a batch of N-1 contingencies sharing this topology's pattern. Each "
             "case has its own Ybus values (Yx_mag/Yx_ang, shape (nnz,L)) and pin mask "
             "(shape (n,L)); Sbus is constant, V0 is (n,L). Returns V (n,L), iterations "
             "(L,), converged (L,).");
}
