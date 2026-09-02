// SPDX-FileCopyrightText: 2026 Fraunhofer IEE
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cuComplex.h>


// -- dSbus_dV_kernel --
__global__ void dSbus_dV_kernel(
    const cuDoubleComplex* Yx,
    const int* Yp,
    const int* Yj,
    const int* Yd,
    const cuDoubleComplex* voltage,
    const double* Vnorm,
    cuDoubleComplex* dS_dVm,
    cuDoubleComplex* dS_dVa,
    int n_buses
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n_buses) return;

    int diag_ix = Yd[row];
    cuDoubleComplex v_row = voltage[row];

    for (int data_ix = Yp[row]; data_ix < Yp[row + 1]; data_ix++) {
        int col = Yj[data_ix];
        cuDoubleComplex val = Yx[data_ix];
        cuDoubleComplex v_col = voltage[col];

        // Compute tmp = voltage[row] * conj(val * voltage[col])
        cuDoubleComplex val_times_v_col = cuCmul(val, v_col);
        cuDoubleComplex tmp = cuCmul(v_row, cuConj(val_times_v_col));

        // Compute tmp_j = tmp * 1j
        cuDoubleComplex tmp_j = make_cuDoubleComplex(-tmp.y, tmp.x);

        // Update dS_dVm
        cuDoubleComplex dVm_update1 = cuCdiv(tmp, make_cuDoubleComplex(Vnorm[col], 0.0));
        cuDoubleComplex dVm_update2 = cuCdiv(tmp, make_cuDoubleComplex(Vnorm[row], 0.0));

        atomicAdd((double*)&dS_dVm[data_ix].x, dVm_update1.x);
        atomicAdd((double*)&dS_dVm[data_ix].y, dVm_update1.y);
        atomicAdd((double*)&dS_dVm[diag_ix].x, dVm_update2.x);
        atomicAdd((double*)&dS_dVm[diag_ix].y, dVm_update2.y);

        // Update dS_dVa
        atomicAdd((double*)&dS_dVa[data_ix].x, -tmp_j.x);
        atomicAdd((double*)&dS_dVa[data_ix].y, -tmp_j.y);
        atomicAdd((double*)&dS_dVa[diag_ix].x, tmp_j.x);
        atomicAdd((double*)&dS_dVa[diag_ix].y, tmp_j.y);
    }
}


// -- create_J_kernel --
__global__ void create_J_kernel(
    const cuDoubleComplex* dVm_x,
    const cuDoubleComplex* dVa_x,
    const int* Yp,
    const int* Yj,
    const int* _pvpq,
    const int* _pq,
    const int* pvpq_pos,
    const int* pq_pos,
    double* Jx,
    int* Jj,
    int* Jp,
    int lpvpq,
    int lpq,
    int n_buses
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes one row
    if (tid < lpvpq + lpq) {
        int row_bus, is_top_block;
        int nnz_count = 0;

        if (tid < lpvpq) {
            // Top block rows: pvpq
            row_bus = _pvpq[tid];
            is_top_block = 1;
        } else {
            // Bottom block rows: pq
            row_bus = _pq[tid - lpvpq];
            is_top_block = 0;
        }

        int start_idx = Yp[row_bus];
        int end_idx = Yp[row_bus + 1];

        // Count non-zeros for this row first
        for (int k = start_idx; k < end_idx; k++) {
            int col_bus = Yj[k];
            int c_pvpq = pvpq_pos[col_bus];
            int c_pq = pq_pos[col_bus];

            if (is_top_block) {
                // Real parts
                if (c_pvpq != -1) nnz_count++;
                if (c_pq != -1) nnz_count++;
            } else {
                // Imaginary parts
                if (c_pvpq != -1) nnz_count++;
                if (c_pq != -1) nnz_count++;
            }
        }

        // Store count in Jp (will be converted to offsets later)
        Jp[tid + 1] = nnz_count;
    }
}

__global__ void fill_J_kernel(
    const cuDoubleComplex* dVm_x,
    const cuDoubleComplex* dVa_x,
    const int* Yp,
    const int* Yj,
    const int* _pvpq,
    const int* _pq,
    const int* pvpq_pos,
    const int* pq_pos,
    double* Jx,
    int* Jj,
    const int* Jp_offsets,
    int lpvpq,
    int lpq
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < lpvpq + lpq) {
        int row_bus, /*r,*/ is_top_block;

        if (tid < lpvpq) {
            // Top block rows: pvpq
            row_bus = _pvpq[tid];
            //r = tid;
            is_top_block = 1;
        } else {
            // Bottom block rows: pq
            row_bus = _pq[tid - lpvpq];
            //r = tid - lpvpq;
            is_top_block = 0;
        }

        // Starting position in J arrays for this row
        int j_idx = Jp_offsets[tid];

        for (int k = Yp[row_bus]; k < Yp[row_bus + 1]; k++) {
            int col_bus = Yj[k];
            int c_pvpq = pvpq_pos[col_bus];
            int c_pq = pq_pos[col_bus];

            if (is_top_block) {
                // Top block: real parts
                if (c_pvpq != -1) {
                    Jx[j_idx] = cuCreal(dVa_x[k]);
                    Jj[j_idx] = c_pvpq;
                    j_idx++;
                }
                if (c_pq != -1) {
                    Jx[j_idx] = cuCreal(dVm_x[k]);
                    Jj[j_idx] = lpvpq + c_pq;
                    j_idx++;
                }
            } else {
                // Bottom block: imaginary parts
                if (c_pvpq != -1) {
                    Jx[j_idx] = cuCimag(dVa_x[k]);
                    Jj[j_idx] = c_pvpq;
                    j_idx++;
                }
                if (c_pq != -1) {
                    Jx[j_idx] = cuCimag(dVm_x[k]);
                    Jj[j_idx] = lpvpq + c_pq;
                    j_idx++;
                }
            }
        }
    }
}


// -- convert_counts_to_offsets --
__global__ void convert_counts_to_offsets(int* Jp, int size, int* nnz_result) {
    // Simple prefix sum for small arrays
    for (int i = 1; i <= size; i++) {
        Jp[i] += Jp[i-1];
    }

    // Store the total nnz in the result location
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *nnz_result = Jp[size];
    }
}


// -- evaluate_results_kernel --
__global__ void evaluate_results_kernel(
    const double* dx,
    const double* voltage_real,
    const double* voltage_imag,
    const int* pq_indices,
    const int* pv_indices,
    const int* pvpq_indices,
    double* result_real,
    double* result_imag,
    int npq,
    int npv,
    int npvpq
    //int offset
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = 0;

    if (tid < npvpq) {
        int bus_idx = pvpq_indices[tid];

        // Calculate angle and magnitude
        double va = atan2(voltage_imag[bus_idx], voltage_real[bus_idx]);
        double vm = sqrt(voltage_real[bus_idx] * voltage_real[bus_idx] +
                         voltage_imag[bus_idx] * voltage_imag[bus_idx]);

        // Check if this bus is in pq (has magnitude update)
        bool is_pq = false;
        int pq_pos = -1;
        for (int i = 0; i < npq; i++) {
            if (pq_indices[i] == bus_idx) {
                is_pq = true;
                pq_pos = i;
                break;
            }
        }

        // Check if this bus is in pv (only angle update)
        bool is_pv = false;
        for (int i = 0; i < npv; i++) {
            if (pv_indices[i] == bus_idx) {
                is_pv = true;
                break;
            }
        }

        // Update angle
        if (is_pv || is_pq) {
            va += dx[offset + tid];
        }

        // Update magnitude (only for PQ buses)
        if (is_pq) {
            vm += dx[offset + npv + npq + pq_pos];
        }

        // Convert back to rectangular form
        result_real[bus_idx] = vm * cos(va);
        result_imag[bus_idx] = vm * sin(va);
    }
}

// -- evaluate_fx_kernel --
__global__ void evaluate_fx_kernel(
    const cuDoubleComplex* V,
    const cuDoubleComplex* Yx,
    const int* Yp,
    const int* Yj,
    const cuDoubleComplex* Sbus,
    const int* pv_indices,
    const int* pq_indices,
    double* F,
    const int npv,
    const int npq,
    const int n_buses
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < npv + 2 * npq) {
        // Compute Y*V product for mis calculation
        cuDoubleComplex yv_sum = make_cuDoubleComplex(0.0, 0.0);

        if (tid < npv) {
            // PV real part
            int bus_idx = pv_indices[tid];
            for (int k = Yp[bus_idx]; k < Yp[bus_idx + 1]; k++) {
                int col = Yj[k];
                yv_sum = cuCadd(yv_sum, cuCmul(Yx[k], V[col]));
            }
            cuDoubleComplex mis = cuCsub(cuCmul(V[bus_idx], cuConj(yv_sum)), Sbus[bus_idx]);
            F[tid] = cuCreal(mis);
        }
        else if (tid < npv + npq) {
            // PQ real part
            int idx = tid - npv;
            int bus_idx = pq_indices[idx];
            for (int k = Yp[bus_idx]; k < Yp[bus_idx + 1]; k++) {
                int col = Yj[k];
                yv_sum = cuCadd(yv_sum, cuCmul(Yx[k], V[col]));
            }
            cuDoubleComplex mis = cuCsub(cuCmul(V[bus_idx], cuConj(yv_sum)), Sbus[bus_idx]);
            F[tid] = cuCreal(mis);
        }
        else {
            // PQ imaginary part
            int idx = tid - npv - npq;
            int bus_idx = pq_indices[idx];
            for (int k = Yp[bus_idx]; k < Yp[bus_idx + 1]; k++) {
                int col = Yj[k];
                yv_sum = cuCadd(yv_sum, cuCmul(Yx[k], V[col]));
            }
            cuDoubleComplex mis = cuCsub(cuCmul(V[bus_idx], cuConj(yv_sum)), Sbus[bus_idx]);
            F[tid] = cuCimag(mis);
        }
    }
}


__device__ void atomicMaxDouble(double* addr, double val) {
    unsigned long long* addr_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_ull, assumed,
            __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (old != assumed);
}


__global__ void inf_norm_and_check_kernel(
    const double* F,
    const int     total,
    const double  tolerance,
    double*       norm_buffer,  // initialized to 0.0 before launch
    int*          result
) {
    extern __shared__ double sdata[];   // blockDim.x * sizeof(double)

    int tid       = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    int stride    = blockDim.x * gridDim.x;

    // ── Each thread accumulates its own max over multiple elements ────────
    double local_max = 0.0;
    for (int i = tid; i < total; i += stride)
        local_max = fmax(local_max, fabs(F[i]));

    // ── Block-level reduction in shared memory ────────────────────────────
    sdata[local_tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_tid < s)
            sdata[local_tid] = fmax(sdata[local_tid], sdata[local_tid + s]);
        __syncthreads();
    }

    // ── Each block's thread 0 atomically updates the global max ──────────
    if (local_tid == 0)
        atomicMaxDouble(norm_buffer, sdata[0]);

    // ── Final comparison: only one thread after all blocks finish ─────────
    // Use a separate check_convergence_kernel below (see note)
}

__global__ void check_convergence_kernel(
    const double* norm_buffer,
    const double  tolerance,
    int*          result
) {
    result[0] = (*norm_buffer < tolerance) ? 1 : 0;
}
