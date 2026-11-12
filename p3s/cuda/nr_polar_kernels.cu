// SPDX-FileCopyrightText: 2026 Fraunhofer IEE
//
// SPDX-License-Identifier: BSD-3-Clause

// Fully-resident batched polar Newton-Raphson GPU kernels.
//
// The Newton iteration is the one in p3s/cpp/nr_klu.cpp (same mismatch, same Jacobian,
// same Armijo line search), laid out for a GPU that solves a whole batch at once.
//
// LAYOUT. Every per-system array is BATCH-INTERLEAVED: item i of system c lives at
// [i * S + c] (S = stride >= number of systems B). A numpy (n, B) array in C order is
// exactly that, so inputs and results cross the bus without any host transpose. All
// kernels that work per item run one thread per (item, system) with the system as the
// fast index, so a warp is 32 systems doing the SAME item: its loads and stores are
// adjacent in memory (coalesced), it has no divergent branches, and the topology index
// arrays are fetched once per warp instead of once per system.
//
// TRIG. The mismatch and the Jacobian need cos/sin of (Va_i - Va_k - angle(Y_ik)) for
// every Ybus nonzero. With the admittance kept rectangular (G + jB) and the unit phasor
// (cos Va, sin Va) of every bus cached, that is four multiplications:
//     Ym cos(delta) = G cu + B su,   Ym sin(delta) = G su - B cu,
//     cu = ci ck + si sk,            su = si ck - ci sk.
// So one sincos per BUS and iteration replaces one per Ybus NONZERO, and a per-system
// Ybus needs no magnitude/angle conversion at all.
//
// STATUS. status[c] steers every kernel (run_mask / j_mask are bit sets over these):
//     0 ACTIVE     iterating; mismatch and Jacobian are evaluated at the current point
//     1 CONVERGED  done
//     2 FAILED     non-finite mismatch, or the factorization was rejected
//     3 BACKTRACK  line search: trial point, mismatch only
//     4 NEEDJ      line search finished; Jacobian still to be built at the accepted point
//
// PRECISION. vote[c] is the precision of the system's next factorization (mixed mode):
//     1 single, 0 double, -1 double from now on. See newton_decide and lean_factor_solve.
//
// Block ids (polar_topology.py / nr_klu.cpp): 0=dP/dVa, 1=dP/dVm, 2=dQ/dVa, 3=dQ/dVm.

#include <float.h>

#define ST_ACTIVE 0
#define ST_CONVERGED 1
#define ST_FAILED 2
#define ST_BACKTRACK 3
#define ST_NEEDJ 4

// nr_klu.cpp line-search constants
#define LS_MAX_TRIALS 10
#define ARMIJO_C 1e-4

#define IX(i) ((size_t)(i) * (size_t)S + (size_t)c)

typedef unsigned long long u64;

extern "C" {

// ---------------------------------------------------------------------------
// init_state: complex start voltage -> magnitude, angle and unit phasor.
// V0 is (n) shared by all systems (v0_shared) or (n, S) interleaved, complex128.
// One thread per (bus, system).
// ---------------------------------------------------------------------------
__global__ void init_state(
    const double* V0, int v0_shared,
    double* Vm, double* Va, double* cs, double* sn,
    int n, int S, int B)
{
    const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int bus = (int)(gid / B);
    if (bus >= n) return;
    const int c = (int)(gid % B);
    const size_t o = 2 * (v0_shared ? (size_t)bus : IX(bus));
    const double re = V0[o], im = V0[o + 1];
    const double vm = hypot(re, im);
    Vm[IX(bus)] = vm;
    Va[IX(bus)] = atan2(im, re);
    cs[IX(bus)] = vm > 0.0 ? re / vm : 1.0;
    sn[IX(bus)] = vm > 0.0 ? im / vm : 0.0;
}

// ---------------------------------------------------------------------------
// eval_FJ: mismatch and (optionally) Jacobian, one thread per (bus, system).
//
// The thread walks the bus's Ybus row once and produces
//   * the residual R = Sspec - Scalc of the bus's P row and Q row (R = -F, i.e. already
//     the right-hand side of J dx = R),
//   * the running infinity norm of R per system (atomic max on the bit pattern -- for
//     non-negative doubles integer order is numeric order, and a NaN compares above
//     everything, so a NaN mismatch can never pass for a small one),
//   * the Jacobian entries of that row, written STRAIGHT into the linear solver's value
//     array through posmap (4 slots per Ybus nonzero, -1 = no such entry): the factor
//     array of the lean LU, or the CSR values of cuDSS. The diagonal terms accumulate in
//     registers and are written once, so there are no atomics on the Jacobian.
//
// A pinned bus (N-1: island reference / unserved) gets identity rows and zero mismatch,
// exactly like nr_klu.cpp, which freezes it without touching the sparsity pattern.
//
// Output addressing is J[pos * sJp + c * sJc] and R[row * sRp + c * sRc], so the same
// kernel feeds interleaved (sp = S, sc = 1) and system-major (sp = 1, sc = len) buffers.
// ---------------------------------------------------------------------------
__global__ void eval_FJ(
    const int* Yp, const int* Yj, const int* Ydiag,
    const double* Y, int y_shared,          // complex G + jB: (nnzY) or (nnzY, S)
    const double* Vm, const double* cs, const double* sn,
    const double* Ssp, int s_shared,        // complex Sbus: (n) or (n, S)
    const int* row_p, const int* row_q,     // (n) residual row of the P / Q equation, -1 = none
    const unsigned char* pin,               // (n, S) or nullptr
    const int* status, int run_mask, int j_mask,
    const int* posmap, double* J, long long sJp, long long sJc,
    double* R, long long sRp, long long sRc,
    u64* resid,
    int n, int S, int B)
{
    const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int row = (int)(gid / B);
    if (row >= n) return;
    const int c = (int)(gid % B);
    const int st = status[c];
    if (!((run_mask >> st) & 1)) return;
    const bool wj = (j_mask >> st) & 1;
    const bool pinned = pin != nullptr && pin[IX(row)];

    const double vi = Vm[IX(row)], ci = cs[IX(row)], si = sn[IX(row)];
    const int dix = Ydiag[row];
    double prow = 0.0, qrow = 0.0;
    double dPVa = 0.0, dQVa = 0.0, dPVm = 0.0, dQVm = 0.0;
    const size_t jc = (size_t)c * (size_t)sJc;

#define PUT(b, v) { const int p_ = pm[b]; if (p_ >= 0) J[(size_t)p_ * (size_t)sJp + jc] = (v); }

    const int kend = Yp[row + 1];
    for (int k = Yp[row]; k < kend; ++k) {
        const int col = Yj[k];
        const size_t yo = 2 * (y_shared ? (size_t)k : IX(k));
        const double g = Y[yo], b = Y[yo + 1];
        if (col != row) {
            const double vk = Vm[IX(col)], ck = cs[IX(col)], sk = sn[IX(col)];
            const double cu = ci * ck + si * sk;
            const double su = si * ck - ci * sk;
            const double gc = g * cu + b * su;      // Ym cos(delta)
            const double gs = g * su - b * cu;      // Ym sin(delta)
            const double vgc = vi * gc, vgs = vi * gs;
            const double a = vgc * vk, bt = vgs * vk;
            prow += a;
            qrow += bt;
            dPVa -= bt;
            dQVa += a;
            dPVm += vk * gc;
            dQVm += vk * gs;
            if (wj) {
                const int* pm = posmap + 4 * (size_t)k;
                if (pinned) {
                    PUT(0, 0.0) PUT(1, 0.0) PUT(2, 0.0) PUT(3, 0.0)
                } else {
                    PUT(0, bt) PUT(1, vgc) PUT(2, -a) PUT(3, vgs)
                }
            }
        } else {
            // self term: delta = -angle(Y), so Ym cos = G and Ym sin = -B
            prow += vi * vi * g;
            qrow -= vi * vi * b;
            dPVm += 2.0 * vi * g;
            dQVm -= 2.0 * vi * b;
        }
    }
    if (wj && dix >= 0) {
        const int* pm = posmap + 4 * (size_t)dix;
        if (pinned) {
            PUT(0, 1.0) PUT(1, 0.0) PUT(2, 0.0) PUT(3, 1.0)
        } else {
            PUT(0, dPVa) PUT(1, dPVm) PUT(2, dQVa) PUT(3, dQVm)
        }
    }
#undef PUT

    const size_t so = 2 * (s_shared ? (size_t)row : IX(row));
    const size_t rc = (size_t)c * (size_t)sRc;
    double worst = 0.0;
    const int rp = row_p[row];
    if (rp >= 0) {
        const double r = pinned ? 0.0 : Ssp[so] - prow;
        R[(size_t)rp * (size_t)sRp + rc] = r;
        worst = fabs(r);
    }
    const int rq = row_q[row];
    if (rq >= 0) {
        const double r = pinned ? 0.0 : Ssp[so + 1] - qrow;
        R[(size_t)rq * (size_t)sRp + rc] = r;
        const double a = fabs(r);
        worst = (a > worst || a != a) ? a : worst;      // keep a NaN
    }
    const u64 bits = (u64)__double_as_longlong(worst);
    if (bits > resid[c]) atomicMax(&resid[c], bits);
}

// ---------------------------------------------------------------------------
// newton_decide: per-system bookkeeping after an eval pass -- convergence test and the
// Armijo backtracking line search of nr_klu.cpp. One thread per system.
//
//   phase 0  after the start-point evaluation
//   phase 1  after the full Newton step (alpha = 1) of iteration `it`
//   phase 2  after a backtracking trial
//   phase 3  after rebuilding F and J at the accepted backtracked point
//
// A step is accepted iff ||F(V + alpha dV)|| <= (1 - ARMIJO_C alpha) ||F(V)||; after
// LS_MAX_TRIALS halvings the best trial seen is taken. The voltage is updated in place:
// dalpha[c] is the correction apply_step has to add to move from the current alpha to
// the next one. A non-finite norm is replaced by DBL_MAX so that no comparison against it
// can succeed (nr_klu's NONFINITE_NORM).
//
// Mixed precision (mixed != 0). The mismatch is always evaluated in double, so a Newton
// step from single-precision factors is merely inexact: it lands within a relative error
// delta ~ cond(J) * 6e-8 of the exact step. That is invisible while the iteration is far
// from the solution, and close to it limits the contraction to delta per step. vote[c]
// picks the precision of the system's next factorization after every accepted step:
//   * single while the mismatch is above `mixed_switch`,
//   * double below it: the final, quadratically converging steps are exact,
//   * double from now on, if a single-precision step did not even halve the mismatch (too
//     large a delta shows as slow, linear convergence) or the line search had to cut one.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int classify(double nrm, double tol) {
    return nrm < tol ? ST_CONVERGED : (nrm >= DBL_MAX ? ST_FAILED : ST_ACTIVE);
}

__device__ __forceinline__ void vote_precision(int c, double old, double nw, double mixed_switch, int* vote)
{
    if (vote[c] < 0) return;
    if (vote[c] == 1 && !(nw <= 0.5 * old)) vote[c] = -1;
    else vote[c] = nw > mixed_switch ? 1 : 0;
}

__global__ void newton_decide(
    u64* resid, double* nrm, double* alpha, double* dalpha,
    double* best_alpha, double* best_nrm, int* trial,
    int* status, int* iters, int* vote,
    double tol, double mixed_switch, int line_search, int mixed, int phase, int it, int B)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= B) return;
    const int st = status[c];
    double nw = __longlong_as_double((long long)resid[c]);
    if (!(nw < DBL_MAX)) nw = DBL_MAX;                  // NaN / Inf

    if (phase == 0) {
        resid[c] = 0;
        nrm[c] = nw;
        iters[c] = 0;
        status[c] = classify(nw, tol);
        vote[c] = (mixed && nw > mixed_switch) ? 1 : 0;
    } else if (phase == 1) {
        if (st != ST_ACTIVE) return;
        resid[c] = 0;
        iters[c] = it;
        if (!line_search || nw <= (1.0 - ARMIJO_C) * nrm[c]) {
            if (mixed) vote_precision(c, nrm[c], nw, mixed_switch, vote);
            nrm[c] = nw;
            status[c] = classify(nw, tol);
        } else {
            best_alpha[c] = 1.0;
            best_nrm[c] = nw;
            alpha[c] = 0.5;
            dalpha[c] = -0.5;
            trial[c] = 1;
            status[c] = ST_BACKTRACK;
            if (mixed) vote[c] = -1;
        }
    } else if (phase == 2) {
        if (st != ST_BACKTRACK) return;
        resid[c] = 0;
        const double a = alpha[c];
        if (nw < best_nrm[c]) { best_nrm[c] = nw; best_alpha[c] = a; }
        if (nw <= (1.0 - ARMIJO_C * a) * nrm[c]) {
            dalpha[c] = 0.0;
            status[c] = ST_NEEDJ;
        } else if (trial[c] >= LS_MAX_TRIALS) {
            dalpha[c] = best_alpha[c] - a;
            alpha[c] = best_alpha[c];
            status[c] = ST_NEEDJ;
        } else {
            trial[c] += 1;
            alpha[c] = 0.5 * a;
            dalpha[c] = -0.5 * a;
        }
    } else {
        if (st != ST_NEEDJ) return;
        resid[c] = 0;
        nrm[c] = nw;
        status[c] = classify(nw, tol);
    }
}

// ---------------------------------------------------------------------------
// apply_step: V += a * dx, and refresh the unit phasor. One thread per (bus, system).
//   a = 1 for the full Newton step (dalpha == nullptr), else dalpha[c] (line search).
// dx is addressed as x[idx * sXp + c * sXc] with idx = xva[bus] / xvm[bus] (-1 = the bus
// has no such variable): the lean LU hands back the step in pivot order, interleaved;
// cuDSS in variable order, system-major.
// ---------------------------------------------------------------------------
__global__ void apply_step(
    const double* x, long long sXp, long long sXc,
    const int* xva, const int* xvm,
    const int* status, int run_mask, const double* dalpha,
    double* Vm, double* Va, double* cs, double* sn,
    int n, int S, int B)
{
    const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int bus = (int)(gid / B);
    if (bus >= n) return;
    const int c = (int)(gid % B);
    if (!((run_mask >> status[c]) & 1)) return;
    const double a = dalpha != nullptr ? dalpha[c] : 1.0;
    if (a == 0.0) return;
    const size_t xc = (size_t)c * (size_t)sXc;
    const int iva = xva[bus];
    if (iva >= 0) {
        const double va = Va[IX(bus)] + a * x[(size_t)iva * (size_t)sXp + xc];
        double s, co;
        sincos(va, &s, &co);
        Va[IX(bus)] = va;
        cs[IX(bus)] = co;
        sn[IX(bus)] = s;
    }
    const int ivm = xvm[bus];
    if (ivm >= 0) Vm[IX(bus)] += a * x[(size_t)ivm * (size_t)sXp + xc];
}

// ---------------------------------------------------------------------------
// finalize: polar state -> complex voltage, (n, S) interleaved complex128 -- the layout of
// the numpy (n, B) result. One thread per (bus, system).
// ---------------------------------------------------------------------------
__global__ void finalize(
    const double* Vm, const double* cs, const double* sn, double* Vout,
    int n, int S, int B)
{
    const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int bus = (int)(gid / B);
    if (bus >= n) return;
    const int c = (int)(gid % B);
    const double vm = Vm[IX(bus)];
    Vout[2 * IX(bus)] = vm * cs[IX(bus)];
    Vout[2 * IX(bus) + 1] = vm * sn[IX(bus)];
}

// ---------------------------------------------------------------------------
// expand_Y / apply_Y_delta: build the per-system Ybus of an N-1 batch ON THE DEVICE from
// the intact-grid values and the sparse outage stamps, Y[k, c] = Ybase[k] - delta[k, c].
// Each (k, c) appears at most once in the delta list, so the subtraction needs no atomics.
// ---------------------------------------------------------------------------
__global__ void expand_Y(const double* Ybase, double* Y, int nnzY, int S, int B)
{
    const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int k = (int)(gid / B);
    if (k >= nnzY) return;
    const int c = (int)(gid % B);
    Y[2 * IX(k)] = Ybase[2 * (size_t)k];
    Y[2 * IX(k) + 1] = Ybase[2 * (size_t)k + 1];
}

__global__ void apply_Y_delta(const int* dk, const int* dc, const double* dval, double* Y, int nd, int S)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nd) return;
    const size_t o = 2 * ((size_t)dk[e] * (size_t)S + (size_t)dc[e]);
    Y[o] -= dval[2 * (size_t)e];
    Y[o + 1] -= dval[2 * (size_t)e + 1];
}

} // extern "C"

// ---------------------------------------------------------------------------
// lean_factor_solve: batched static-pivot sparse LU + triangular solves, ONE launch.
//
// GPU port of p3s/cpp/lean_lu.hpp (frozen pivot order and pattern, numeric work only),
// reorganised so that a single factorization is parallel too -- see lean_lu_plan.py for
// the scheme and the index arrays. A block is 32 systems (lanes) x `slots` warps. Every
// warp executes its own task list of the current level for its 32 systems; the block
// meets at a barrier between levels:
//
//   factor   task = one entry of L or U: acc = a - sum L(i,t) U(t,k) over its pairs.
//            Diagonals are stored inverted; after a barrier the L entries of the level
//            are scaled by them.
//   forward  task = one row:  y_i = R[perm[i]] - sum_j L(i,j) y_j
//   backward task = one row:  y_i = (y_i - sum_j U(i,j) y_j) / U(i,i)
//
// J holds the matrix values eval_FJ scattered to the factor positions (fill entries start
// from 0); `fac` receives the factors. y ends up as the Newton step in pivot order.
//
//   double  fac == J: the factorization overwrites the Jacobian in place.
//   single  fac is a separate float array, J stays intact. Halving the bytes per operand
//           nearly doubles the throughput of this memory-bound kernel (and consumer GPUs
//           have little double-precision arithmetic to begin with); newton_decide keeps
//           the resulting inexact Newton steps away from the last iteration.
//
// Which systems a launch works on: ACTIVE ones, and with use_vote those whose vote asks
// for this kernel's precision (vote == 1: single).
//
// Pivot check (lean_refactor's): min|pivot| / max|pivot| of the system must reach
// rcond_min and every pivot must be finite and non-zero, otherwise the frozen pivot order
// does not suit this matrix. In single precision the system is then passed on to the
// double kernel of the same iteration (vote = -1). In double it is marked FAILED and left
// for the caller's fallback, as nr_klu hands such a column to KLU's pivoting factorization.
// pstat[2c], pstat[2c+1] return min and max |pivot|.
// ---------------------------------------------------------------------------
#define LEAN_MAX_THREADS 1024
#define TASK_DIAG -2
#define TASK_COPY -3

template <typename real>
__device__ __forceinline__ void lean_factor_solve_impl(
    const int* lptr, const int* tdst, const int* tcode, const int* tpp, const int* pa, const int* pb,
    const int* flptr, const int* frow, const int* fptr, const int* fcol, const int* fpos, const int* perm,
    const int* blptr, const int* brow, const int* base, const int* Lp, const int* Li,
    const double* J, real* fac, const double* R, double* y,
    int* status, int* vote, double* pstat, double rcond_min,
    int nlev, int nblev, int W, int S, int B, int do_factor, int do_solve, int use_vote)
{
    __shared__ double sh_min[LEAN_MAX_THREADS];
    __shared__ double sh_max[LEAN_MAX_THREADS];
    __shared__ double sh_sum[LEAN_MAX_THREADS];
    __shared__ int sh_bad[32];

    const bool single = sizeof(real) == 4;
    const int slot = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int c = blockIdx.x * 32 + lane;
    bool act = c < B && status[c] == ST_ACTIVE;
    if (act && (use_vote || single)) act = (vote[c] == 1) == single;

    if (do_factor) {
        double pmin = DBL_MAX, pmax = 0.0, psum = 0.0;
        for (int l = 0; l < nlev; ++l) {
            const int t0 = lptr[l * W + slot], t1 = lptr[l * W + slot + 1];
            if (act) {
                for (int t = t0; t < t1; ++t) {
                    const int enc = tdst[t];
                    const int dst = enc & 0x7fffffff;
                    const int code = tcode[t];
                    if (code == TASK_COPY) {            // entry of A that needs no arithmetic
                        if (single) fac[IX(dst)] = (real)J[IX(dst)];
                        continue;
                    }
                    real acc = enc < 0 ? (real)0.0 : (real)J[IX(dst)];
                    const int e1 = tpp[t + 1];
                    for (int e = tpp[t]; e < e1; ++e) acc -= fac[IX(pa[e])] * fac[IX(pb[e])];
                    if (code == TASK_DIAG) {
                        const double a = fabs((double)acc);
                        pmin = fmin(pmin, a);
                        pmax = fmax(pmax, a);
                        psum += a;                      // carries a NaN / Inf pivot
                        acc = (real)1.0 / acc;
                    }
                    fac[IX(dst)] = acc;
                }
            }
            __syncthreads();
            if (act) {
                for (int t = t0; t < t1; ++t) {
                    const int dg = tcode[t];
                    if (dg >= 0) fac[IX(tdst[t] & 0x7fffffff)] *= fac[IX(dg)];
                }
            }
            __syncthreads();
        }
        // reduce the pivot statistics over the slots of each lane (= system)
        sh_min[threadIdx.x] = pmin;
        sh_max[threadIdx.x] = pmax;
        sh_sum[threadIdx.x] = psum;
        __syncthreads();
        if (slot == 0) {
            int bad = 0;
            if (act) {
                for (int s = 1; s < W; ++s) {
                    pmin = fmin(pmin, sh_min[lane + 32 * s]);
                    pmax = fmax(pmax, sh_max[lane + 32 * s]);
                    psum += sh_sum[lane + 32 * s];
                }
                pstat[2 * c] = pmin;
                pstat[2 * c + 1] = pmax;
                const bool ok = psum < DBL_MAX && pmin > 0.0 && pmin >= rcond_min * pmax;
                if (!ok) {
                    bad = 1;
                    if (single) vote[c] = -1;           // retried by the double kernel
                    else status[c] = ST_FAILED;
                }
            }
            sh_bad[lane] = bad;
        }
        __syncthreads();
        if (sh_bad[lane]) act = false;
    }

    if (do_solve) {
        for (int l = 0; l < nlev; ++l) {
            const int t0 = flptr[l * W + slot], t1 = flptr[l * W + slot + 1];
            if (act) {
                for (int t = t0; t < t1; ++t) {
                    const int i = frow[t];
                    double acc = R[IX(perm[i])];
                    const int e1 = fptr[i + 1];
                    for (int e = fptr[i]; e < e1; ++e) acc -= (double)fac[IX(fpos[e])] * y[IX(fcol[e])];
                    y[IX(i)] = acc;
                }
            }
            __syncthreads();
        }
        for (int l = 0; l < nblev; ++l) {
            const int t0 = blptr[l * W + slot], t1 = blptr[l * W + slot + 1];
            if (act) {
                for (int t = t0; t < t1; ++t) {
                    const int i = brow[t];
                    double acc = y[IX(i)];
                    const int lo = Lp[i], hi = Lp[i + 1];
                    const int ub = base[i] + 1 - lo;
                    for (int e = lo; e < hi; ++e) acc -= (double)fac[IX(ub + e)] * y[IX(Li[e])];
                    y[IX(i)] = acc * (double)fac[IX(base[i])];
                }
            }
            __syncthreads();
        }
    }
}

extern "C" {

#define LEAN_ENTRY(name, real)                                                                              \
__global__ void name(                                                                                       \
    const int* lptr, const int* tdst, const int* tcode, const int* tpp, const int* pa, const int* pb,       \
    const int* flptr, const int* frow, const int* fptr, const int* fcol, const int* fpos, const int* perm,  \
    const int* blptr, const int* brow, const int* base, const int* Lp, const int* Li,                       \
    const double* J, real* fac, const double* R, double* y,                                                 \
    int* status, int* vote, double* pstat, double rcond_min,                                                \
    int nlev, int nblev, int W, int S, int B, int do_factor, int do_solve, int use_vote)                    \
{                                                                                                           \
    lean_factor_solve_impl<real>(lptr, tdst, tcode, tpp, pa, pb, flptr, frow, fptr, fcol, fpos, perm,       \
                                 blptr, brow, base, Lp, Li, J, fac, R, y, status, vote, pstat, rcond_min,   \
                                 nlev, nblev, W, S, B, do_factor, do_solve, use_vote);                      \
}

LEAN_ENTRY(lean_factor_solve_f64, double)
LEAN_ENTRY(lean_factor_solve_f32, float)

} // extern "C"
