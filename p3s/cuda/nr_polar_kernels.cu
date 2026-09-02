// SPDX-FileCopyrightText: 2026 Fraunhofer IEE
//
// SPDX-License-Identifier: BSD-3-Clause

// Fully-resident batched polar Newton-Raphson GPU kernels (Phase A).
//
// These mirror graviton/cpp/nr_klu.cpp (polar formulation) so GPU and CPU numerics are
// identical. All batch buffers are SYSTEM-MAJOR: system (column) c's data for a length-L
// quantity lives contiguously at [c*L .. c*L+L). This matches cuSolverRf's batch layout
// (system c's Jx at c*nnzJ, its rhs/x at c*n), so the assembly kernels write straight
// into the cuSolverRf batch buffers with no repack.
//
// One thread owns one (system, Ybus-row) pair. Because a single thread processes an
// entire Ybus row, it accumulates that row's diagonal derivative terms locally and writes
// them once -- NO atomicAdd (unlike the older rectangular jacobian_kernels.cu).
//
// Block ids (match polar_topology.py / nr_klu.cpp): 0=dP/dVa,1=dP/dVm,2=dQ/dVa,3=dQ/dVm.

extern "C" {

// ---------------------------------------------------------------------------
// yx_to_polar: convert a complex Ybus-values matrix to (magnitude, angle), transposing
// column-major (nnz, L) -> system-major (L, nnz) in ONE GPU pass. Replaces the host
// np.abs/np.angle + .T (~4 s of single-threaded CPU for pegase N-1). Input Yx_re/Yx_im are
// (nnz, L) column-major (element (k, c) at c*nnz + k -- i.e. already system-major if the
// caller uploads Yx.T... here we take the natural (nnz,L) so index (k,c) = k*L + c).
//   in:  Yx_re[k*L + c], Yx_im[k*L + c]   (nnz, L) row-major over k
//   out: Ym[c*nnz + k], Ya[c*nnz + k]     (L, nnz) system-major (what the solver wants)
// One thread per (k, c).
// ---------------------------------------------------------------------------
__global__ void yx_to_polar(
    const double* Yx_re, const double* Yx_im,
    double* Ym, double* Ya, int nnz, int L)
{
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)nnz * L;
    if (gid >= total) return;
    int k = (int)(gid / L);      // nonzero index
    int c = (int)(gid % L);      // system (column)
    double re = Yx_re[(long)k * L + c];
    double im = Yx_im[(long)k * L + c];
    long o = (long)c * nnz + k;  // system-major output
    Ym[o] = sqrt(re * re + im * im);
    Ya[o] = atan2(im, re);
}

// ---------------------------------------------------------------------------
// eval_F_and_J_polar
//   For each (system c, bus-row r):
//     * compute P/Q injection at r -> mismatch F[c, F_row]
//     * compute the 4 polar derivative blocks for row r's nonzeros, with local diagonal
//       accumulation, into a per-thread-visible scratch (dblocks), then GATHER into Jx.
//
// To avoid a huge global scratch, we do it in two device passes sharing the same launch:
//   pass 0 (this kernel): fill the 4 derivative blocks (system-major, 4*nnzY per system)
//                         AND the mismatch F.
//   then gather_J (below) maps blocks -> Jx via (src_block, src_k).
// This mirrors nr_klu's eval_F_and_J (blocks buffer of size 4*nnzY, then gather).
//
// dblocks layout (system-major): for system c, block b in {0..3}, nonzero k:
//     dblocks[c*(4*nnzY) + b*nnzY + k]
// ---------------------------------------------------------------------------
__global__ void eval_F_and_J_polar(
    const int*    Yp,         // (n+1)
    const int*    Yj,         // (nnzY)
    const int*    Ydiag,      // (n)
    const double* Ym,         // (nnzY) shared, or (B*nnzY) per-system if Ystride==nnzY
    const double* Ya,         // (nnzY) or (B*nnzY)
    long          Ystride,    // 0 = shared Ybus (time-series); nnzY = per-system (N-1)
    const double* Vm,         // (B*n) system-major
    const double* Va,         // (B*n)
    const double* Pspec,      // (n) shared, or (B*n) per-system if Pstride==n
    const double* Qspec,      // (n) or (B*n)
    long          Pstride,    // 0 = shared injections (N-1: Sbus const); n = per-system (TS)
    const int*    pvpq,       // (npvpq) bus indices
    const int*    pq,         // (npq)
    const int*    pvpq_pos,   // (n)  bus -> row in pvpq block, else -1
    const int*    pq_pos,     // (n)  bus -> row in pq block, else -1
    const unsigned char* pin, // (B*n) or nullptr; 1 = freeze this bus
    double*       dblocks,    // (B*4*nnzY) OUT derivative blocks
    double*       F,          // (B*m) OUT mismatch (system-major)
    int           n,
    int           nnzY,
    int           npvpq,
    int           npq,
    int           m,
    int           B)
{
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)B * n;
    if (gid >= total) return;

    int c   = (int)(gid / n);   // system (batch column)
    int row = (int)(gid % n);   // Ybus row / bus

    const double* Vm_c = Vm + (long)c * n;
    const double* Va_c = Va + (long)c * n;
    double* dP_dVa = dblocks + (long)c * 4 * nnzY + 0 * nnzY;
    double* dP_dVm = dblocks + (long)c * 4 * nnzY + 1 * nnzY;
    double* dQ_dVa = dblocks + (long)c * 4 * nnzY + 2 * nnzY;
    double* dQ_dVm = dblocks + (long)c * 4 * nnzY + 3 * nnzY;

    // Per-system Ybus values (N-1) override the shared ones (time-series) via Ystride.
    const double* Ym_c = Ym + (long)c * Ystride;
    const double* Ya_c = Ya + (long)c * Ystride;

    const int dix = Ydiag[row];
    double prow = 0.0, qrow = 0.0;
    double diagP_Va = 0.0, diagQ_Va = 0.0, diagP_Vm = 0.0, diagQ_Vm = 0.0;
    const double vm_row = Vm_c[row];
    const double va_row = Va_c[row];

    for (int k = Yp[row]; k < Yp[row + 1]; ++k) {
        const int col = Yj[k];
        const double delta = va_row - Ya_c[k] - Va_c[col];
        double s, cc;
        sincos(delta, &s, &cc);
        const double VmrowYm = vm_row * Ym_c[k];
        const double vij = VmrowYm * Vm_c[col];
        prow += vij * cc;
        qrow += vij * s;
        if (row != col) {
            const double dpva =  vij * s;
            const double dqva = -vij * cc;
            dP_dVa[k] = dpva;
            dQ_dVa[k] = dqva;
            dP_dVm[k] = VmrowYm * cc;
            dQ_dVm[k] = VmrowYm * s;
            diagP_Va -= dpva;
            diagQ_Va -= dqva;
            diagP_Vm += Ym_c[k] * Vm_c[col] * cc;
            diagQ_Vm += Ym_c[k] * Vm_c[col] * s;
        } else {
            // self term (delta = -Ya[diag]); see nr_klu.cpp note
            diagP_Vm += 2.0 * vm_row * Ym_c[k] * cc;
            diagQ_Vm += 2.0 * vm_row * Ym_c[k] * s;
        }
    }
    // write accumulated diagonal terms once
    if (dix >= 0) {
        dP_dVa[dix] = diagP_Va;
        dQ_dVa[dix] = diagQ_Va;
        dP_dVm[dix] = diagP_Vm;
        dQ_dVm[dix] = diagQ_Vm;
    }

    // mismatch: this bus contributes to at most one P-row (if pvpq) and one Q-row (if pq)
    const bool pinned = (pin != nullptr && pin[(long)c * n + row]);
    const double* Pspec_c = Pspec + (long)c * Pstride;   // stride 0 => shared across systems
    const double* Qspec_c = Qspec + (long)c * Pstride;
    const int rp = pvpq_pos[row];
    if (rp != -1)
        F[(long)c * m + rp] = pinned ? 0.0 : (prow - Pspec_c[row]);
    const int rq = pq_pos[row];
    if (rq != -1)
        F[(long)c * m + npvpq + rq] = pinned ? 0.0 : (qrow - Qspec_c[row]);
}


// ---------------------------------------------------------------------------
// gather_J_polar: Jx[c, p] = dblocks[c, src_block[p], src_k[p]]
// One thread per (system, Jacobian nonzero). Also applies the pin identity rows:
// a Jacobian row owned by a pinned bus becomes identity (1 on diagonal, 0 elsewhere).
// ---------------------------------------------------------------------------
__global__ void gather_J_polar(
    const double* dblocks,    // (B*4*nnzY)
    const int*    src_block,  // (nnzJ)
    const int*    src_k,      // (nnzJ)
    const int*    Jp,         // (m+1)      to find row owning each nonzero (for pin)
    const int*    Ji,         // (nnzJ)     column index (for pin diagonal test)
    const int*    pvpq,       // (npvpq)
    const int*    pq,         // (npq)
    const unsigned char* pin, // (B*n) or nullptr
    double*       Jx,         // (B*nnzJ) OUT system-major
    int           nnzY,
    int           nnzJ,
    int           npvpq,
    int           npq,
    int           n,
    int           B,
    const int*    row_of_nz)  // (nnzJ) precomputed J-row index of each nonzero
{
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)B * nnzJ;
    if (gid >= total) return;

    int c = (int)(gid / nnzJ);
    int p = (int)(gid % nnzJ);

    double val = dblocks[(long)c * 4 * nnzY + (long)src_block[p] * nnzY + src_k[p]];

    if (pin != nullptr) {
        int r = row_of_nz[p];                     // J row index
        int rbus = (r < npvpq) ? pvpq[r] : pq[r - npvpq];
        if (pin[(long)c * n + rbus]) {
            val = (Ji[p] == r) ? 1.0 : 0.0;       // identity row
        }
    }
    Jx[(long)c * nnzJ + p] = val;
}


// ---------------------------------------------------------------------------
// negate_into_rhs: rhs = -F  (cuSolverRf solves J*dx = rhs; Newton needs rhs=-F).
// One thread per (system, mismatch row).
// ---------------------------------------------------------------------------
__global__ void negate_F(const double* F, double* rhs, int m, int B) {
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)B * m;
    if (gid >= total) return;
    rhs[gid] = -F[gid];
}


// ---------------------------------------------------------------------------
// update_voltage_polar: apply the Newton step.
//   Va[c, pvpq[i]] += dx[c, i]                (i in [0, npvpq))
//   Vm[c, pq[j]]   += dx[c, npvpq + j]        (j in [0, npq))
// dx is the solved rhs, system-major (B*m). Skips systems already converged.
// One thread per (system, variable). We launch npvpq+npq = m variables per system.
// ---------------------------------------------------------------------------
__global__ void update_voltage_polar(
    const double* dx,          // (B*m)
    const int*    pvpq,        // (npvpq)
    const int*    pq,          // (npq)
    const unsigned char* converged, // (B) skip if 1
    double*       Vm,          // (B*n)
    double*       Va,          // (B*n)
    int           npvpq,
    int           npq,
    int           n,
    int           B)
{
    int m = npvpq + npq;
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)B * m;
    if (gid >= total) return;
    int c = (int)(gid / m);
    if (converged != nullptr && converged[c]) return;
    int v = (int)(gid % m);

    double step = dx[(long)c * m + v];
    if (v < npvpq) {
        Va[(long)c * n + pvpq[v]] += step;
    } else {
        Vm[(long)c * n + pq[v - npvpq]] += step;
    }
}


// ---------------------------------------------------------------------------
// inf_norm_per_column: residual[c] = max_i |F[c, i]|, then converged[c] = residual<tol.
// One block per system; block-stride reduction over that system's m entries.
// ---------------------------------------------------------------------------
__global__ void inf_norm_per_column(
    const double* F,          // (B*m)
    double*       residual,   // (B) OUT
    unsigned char* converged, // (B) OUT (1 if residual<tol)
    double        tol,
    int           m,
    int           B)
{
    extern __shared__ double sdata[];
    int c = blockIdx.x;
    if (c >= B) return;
    const double* Fc = F + (long)c * m;

    // NaN-PROPAGATING max: a singular contingency makes the RF solve emit a NaN dx, which
    // propagates into F. fmax() would DROP the NaN (fmax(NaN,x)=x) and report a small
    // residual -> false "converged". We must instead let the NaN survive so NaN<tol is
    // false and the column is correctly flagged non-converged (as KLU/CPU does).
    double local = 0.0;
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        double a = fabs(Fc[i]);
        local = (isnan(a) || a > local) ? a : local;   // propagate NaN
    }
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            double a = sdata[threadIdx.x], b = sdata[threadIdx.x + s];
            sdata[threadIdx.x] = (isnan(a) || isnan(b)) ? NAN : (a > b ? a : b);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        residual[c] = sdata[0];
        // (sdata[0] < tol) is false when sdata[0] is NaN -> non-converged, as desired.
        converged[c] = (sdata[0] < tol) ? 1 : 0;
    }
}

} // extern "C"
