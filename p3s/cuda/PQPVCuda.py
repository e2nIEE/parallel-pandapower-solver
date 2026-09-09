# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.resources

import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from numpy.typing import NDArray
from pycuda.compiler import SourceModule


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


def _prepare_y_gpu(
    Yp, Yj, Yx, pv, pq
) -> tuple[
    gpuarray,
    gpuarray,
    gpuarray,
    gpuarray,
    gpuarray,
    gpuarray,
    gpuarray,
    gpuarray,
    gpuarray,
    gpuarray,
    int,
    int,
    int,
    int,
    int,
]:
    """
    This function takes an admittance matrix, calculates the diagonal array Yd and pvpq_pos / pq_pos.
    Then it transfers everything to the gpu and returns all the pointers.
    """
    lpv = len(pv)
    pvpq = np.r_[pv, pq]
    lpvpq = len(pvpq)
    lpq = len(pq)
    n_elements = len(Yx)
    n_buses = len(Yp) - 1

    Yd = get_ybus_diag_ix(Yj, Yp, n_buses)

    pvpq_pos: NDArray = -np.ones(n_buses, dtype=int)
    for pos, bus in enumerate(pvpq):
        pvpq_pos[bus] = pos

    pq_pos: NDArray = -np.ones(n_buses, dtype=int)
    for pos, bus in enumerate(pq):
        pq_pos[bus] = pos

    pv_pos: NDArray = -np.ones(n_buses, dtype=int)
    for pos, bus in enumerate(pv):
        pv_pos[bus] = pos

    # Convert to GPU-compatible types
    Yx_gpu = gpuarray.to_gpu(Yx.astype(np.complex128))
    Yp_gpu = gpuarray.to_gpu(Yp.astype(np.int32))
    Yj_gpu = gpuarray.to_gpu(Yj.astype(np.int32))
    Yd_gpu = gpuarray.to_gpu(Yd.astype(np.int32))

    # print(f"PV: {pv}")
    pv_gpu = gpuarray.to_gpu(np.array(pv, dtype=np.int32))
    pv_pos_gpu = gpuarray.to_gpu(np.array(pv_pos, dtype=np.int32))

    pvpq_gpu = gpuarray.to_gpu(np.array(pvpq, dtype=np.int32))
    pvpq_pos_gpu = gpuarray.to_gpu(np.array(pvpq_pos, dtype=np.int32))

    # print(f"PV: {pq}")
    pq_gpu = gpuarray.to_gpu(np.array(pq, dtype=np.int32))
    pq_pos_gpu = gpuarray.to_gpu(np.array(pq_pos, dtype=np.int32))

    return (
        Yx_gpu,
        Yp_gpu,
        Yj_gpu,
        Yd_gpu,
        pq_gpu,
        pq_pos_gpu,
        pv_gpu,
        pv_pos_gpu,
        pvpq_gpu,
        pvpq_pos_gpu,
        lpvpq,
        lpq,
        lpv,
        n_elements,
        n_buses,
    )


class PQPVCuda:
    def __init__(self, Yp, Yj, Yx, pv, pq, arch: str = "sm_86"):
        super().__init__()

        # -- read cuda code --
        with importlib.resources.files("p3s.cuda").joinpath("jacobian_kernels.cu").open("r") as f:
            cuda_source = f.read()

        # -- compile cuda code --
        options = ["-O3", "-rdc=false", "--use_fast_math", f"-gencode=arch=compute_{arch.split('_')[-1]},code={arch}"]

        self.cuda_module = SourceModule(cuda_source, options=options, arch=arch)

        self.evaluate_results_kernel = self.cuda_module.get_function("evaluate_results_kernel")
        self.evaluate_fx_kernel = self.cuda_module.get_function("evaluate_fx_kernel")
        self.inf_norm_and_check_kernel = self.cuda_module.get_function("inf_norm_and_check_kernel")
        self.check_convergence_kernel = self.cuda_module.get_function("check_convergence_kernel")

        # Get CUDA functions
        self.dSbus_dV_kernel = self.cuda_module.get_function("dSbus_dV_kernel")
        self.create_J_kernel = self.cuda_module.get_function("create_J_kernel")
        self.fill_J_kernel = self.cuda_module.get_function("fill_J_kernel")
        self.convert_counts_to_offsets = self.cuda_module.get_function("convert_counts_to_offsets")

        # -- init all variables --
        (
            self.Yx_gpu,
            self.Yp_gpu,
            self.Yj_gpu,
            self.Yd_gpu,
            self.pq_gpu,
            self.pq_pos_gpu,
            self.pv_gpu,
            self.pv_pos_gpu,
            self.pvpq_gpu,
            self.pvpq_pos_gpu,
            self.lpvpq,
            self.lpq,
            self.lpv,
            self.n_elements,
            self.n_buses,
        ) = _prepare_y_gpu(Yp, Yj, Yx, pv, pq)

    def create_J_cuda(self, voltage, Yp, Yj, Yx, pv, pq, update_only=False) -> tuple[gpuarray, gpuarray, gpuarray, int]:
        """
        calculate a new Jacobian via cuda. If update_only is set, it is assumed,
        that all the arrays / sparse matrices are already uploaded to the gpu.
        """

        # if the variables are not set and the user forget to set the update flag, it will be set automatically.
        if self.Yx_gpu is None:
            update_only = False

        # if only updating is selected, it is assumed, that all the arrays / matrices are already uploaded to the gpu
        if not update_only:
            (
                self.Yx_gpu,
                self.Yp_gpu,
                self.Yj_gpu,
                self.Yd_gpu,
                self.pq_gpu,
                self.pq_pos_gpu,
                self.pv_gpu,
                self.pv_pos_gpu,
                self.pvpq_gpu,
                self.pvpq_pos_gpu,
                self.lpvpq,
                self.lpq,
                self.lpv,
                self.n_elements,
                self.n_buses,
            ) = _prepare_y_gpu(Yp, Yj, Yx, pv, pq)

        # calculate the actual Jacobian on the gpu using jacobian_kernels.cu
        Jx_gpu, Jp_gpu, Jj_gpu, nnz = self._update_J_cuda(
            voltage=voltage,
            Yp_gpu=self.Yp_gpu,
            Yj_gpu=self.Yj_gpu,
            Yx_gpu=self.Yx_gpu,
            Yd_gpu=self.Yd_gpu,
            pvpq_gpu=self.pvpq_gpu,
            pvpq_pos_gpu=self.pvpq_pos_gpu,
            pq_gpu=self.pq_gpu,
            pq_pos_gpu=self.pq_pos_gpu,
            lpvpq=self.lpvpq,
            lpq=self.lpq,
            n_elements=self.n_elements,
            n_buses=self.n_buses,
            # cuda_module=self.cuda_module
        )

        return Jx_gpu, Jp_gpu, Jj_gpu, nnz

    def _update_J_cuda(
        self,
        voltage: NDArray,
        Yp_gpu: gpuarray,
        Yj_gpu: gpuarray,
        Yx_gpu: gpuarray,
        Yd_gpu: gpuarray,
        pvpq_gpu: gpuarray,
        pvpq_pos_gpu: gpuarray,
        pq_gpu: gpuarray,
        pq_pos_gpu: gpuarray,
        lpvpq: int,
        lpq: int,
        n_elements: int,
        n_buses: int,
        # cuda_module: SourceModule,
        block_size=256,
    ) -> tuple[gpuarray, gpuarray, gpuarray, int]:
        """Create Jacobian using CUDA acceleration."""

        # Prepare data
        Vnorm = np.abs(voltage)
        n_rows = lpvpq + lpq

        # Convert to GPU-compatible types
        voltage_gpu = gpuarray.to_gpu(voltage.astype(np.complex128))
        Vnorm_gpu = gpuarray.to_gpu(Vnorm.astype(np.float64))

        # Allocate derivative arrays on GPU
        dS_dVm_gpu = gpuarray.zeros(n_elements, dtype=np.complex128)
        dS_dVa_gpu = gpuarray.zeros(n_elements, dtype=np.complex128)

        # Launch derivative computation kernel
        grid_size = (n_buses + block_size - 1) // block_size

        self.dSbus_dV_kernel(
            Yx_gpu.gpudata,
            Yp_gpu.gpudata,
            Yj_gpu.gpudata,
            Yd_gpu.gpudata,
            voltage_gpu.gpudata,
            Vnorm_gpu.gpudata,
            dS_dVm_gpu.gpudata,
            dS_dVa_gpu.gpudata,
            np.int32(n_buses),
            block=(block_size, 1, 1),
            grid=(grid_size, 1),
        )

        # Allocate Jacobian CSR arrays on GPU
        # Reserve more space than needed to avoid reallocation
        reserve = max(1, n_elements * 4)
        Jx_gpu = gpuarray.empty(reserve, dtype=np.float64)
        Jj_gpu = gpuarray.empty(reserve, dtype=np.int32)
        Jp_gpu = gpuarray.zeros(n_rows + 1, dtype=np.int32)
        Jp_offsets_gpu = gpuarray.zeros(n_rows + 1, dtype=np.int32)

        nnz_gpu = gpuarray.zeros(1, dtype=np.int32)

        # First pass: count non-zeros per row
        grid_size_j = ((lpvpq + lpq) + block_size - 1) // block_size

        self.create_J_kernel(
            dS_dVm_gpu.gpudata,
            dS_dVa_gpu.gpudata,
            Yp_gpu.gpudata,
            Yj_gpu.gpudata,
            pvpq_gpu.gpudata,
            pq_gpu.gpudata,
            pvpq_pos_gpu.gpudata,
            pq_pos_gpu.gpudata,
            Jx_gpu.gpudata,
            Jj_gpu.gpudata,
            Jp_offsets_gpu.gpudata,
            np.int32(lpvpq),
            np.int32(lpq),
            np.int32(n_buses),
            block=(block_size, 1, 1),
            grid=(grid_size_j, 1),
        )

        # Convert counts to offsets
        self.convert_counts_to_offsets(
            Jp_offsets_gpu.gpudata, np.int32(n_rows), nnz_gpu.gpudata, block=(1, 1, 1), grid=(1, 1)
        )

        # Second pass: fill actual values
        self.fill_J_kernel(
            dS_dVm_gpu.gpudata,
            dS_dVa_gpu.gpudata,
            Yp_gpu.gpudata,
            Yj_gpu.gpudata,
            pvpq_gpu.gpudata,
            pq_gpu.gpudata,
            pvpq_pos_gpu.gpudata,
            pq_pos_gpu.gpudata,
            Jx_gpu.gpudata,
            Jj_gpu.gpudata,
            Jp_offsets_gpu.gpudata,
            np.int32(lpvpq),
            np.int32(lpq),
            block=(block_size, 1, 1),
            grid=(grid_size_j, 1),
        )

        # Copy final Jp back
        Jp_gpu[:] = Jp_offsets_gpu[:]

        # Get actual number of non-zeros
        nnz = int(nnz_gpu.get()[0])

        # Trim arrays to actual size and copy back to host
        Jx = Jx_gpu[:nnz].copy()
        Jj = Jj_gpu[:nnz].copy()
        Jp = Jp_gpu.copy()

        return Jx, Jp, Jj, nnz

    def evaluate_Results_cuda(self, dx, voltage, block_size=256):
        """CUDA version of evaluate_Results"""
        # npq = len(self._pq)
        npq = self.lpq
        # npv = len(self._pv)
        npv = self.lpv

        # Convert to GPU arrays
        dx_gpu = gpuarray.to_gpu(dx.astype(np.float64))

        # Split voltage into real/imag parts for easier processing
        voltage_real_gpu = gpuarray.to_gpu(np.real(voltage).astype(np.float64))
        voltage_imag_gpu = gpuarray.to_gpu(np.imag(voltage).astype(np.float64))

        # Result arrays
        result_real_gpu = gpuarray.empty(len(voltage), dtype=np.float64)
        result_imag_gpu = gpuarray.empty(len(voltage), dtype=np.float64)

        # Launch kernel
        grid_size = (self.lpvpq + block_size - 1) // block_size

        self.evaluate_results_kernel(
            dx_gpu.gpudata,
            voltage_real_gpu.gpudata,
            voltage_imag_gpu.gpudata,
            self.pq_gpu.gpudata,
            self.pv_gpu.gpudata,
            self.pvpq_gpu.gpudata,
            result_real_gpu.gpudata,
            result_imag_gpu.gpudata,
            np.int32(npq),
            np.int32(npv),
            np.int32(self.lpvpq),
            block=(block_size, 1, 1),
            grid=(grid_size, 1),
        )

        # Combine real and imaginary parts back to complex
        result_gpu = gpuarray.empty(len(voltage), dtype=np.complex128)
        result_gpu_real = result_gpu.real
        result_gpu_imag = result_gpu.imag

        # Copy results back (this could be optimized further)
        result_gpu_real.set(result_real_gpu.get())
        result_gpu_imag.set(result_imag_gpu.get())

        return result_gpu

    def evaluate_Fx_cuda(self, Sbus_gpu, voltage_gpu, block_size=256) -> tuple[gpuarray, bool]:
        """CUDA version of evaluate_Fx"""
        # Convert to GPU arrays
        npv = self.lpv
        npq = self.lpq

        # Result array
        F_gpu = gpuarray.empty(npv + 2 * npq, dtype=np.float64)

        # Launch kernel
        grid_size = ((npv + 2 * npq) + block_size - 1) // block_size

        self.evaluate_fx_kernel(
            voltage_gpu.gpudata,
            self.Yx_gpu.gpudata,
            self.Yp_gpu.gpudata,
            self.Yj_gpu.gpudata,
            Sbus_gpu.gpudata,
            self.pv_gpu.gpudata,
            self.pq_gpu.gpudata,
            F_gpu.gpudata,
            np.int32(npv),
            np.int32(npq),
            np.int32(self.n_buses),
            block=(block_size, 1, 1),
            grid=(grid_size, 1),
        )

        return F_gpu

    def calculate_norm(self, mismatch_gpu, tolerance, block_size=256):
        # Use a fixed, small number of blocks for the norm kernel

        # so each thread processes ceil(total / (BLOCKS*THREADS)) elements
        npv = self.lpv
        npq = self.lpq

        grid_size = ((npv + 2 * npq) + block_size - 1) // block_size

        norm_buffer = cuda.mem_alloc(8)
        cuda.memset_d8(norm_buffer, 0, 8)  # initialize to 0.0

        self.inf_norm_and_check_kernel(
            mismatch_gpu,
            np.int32(npv + 2 * npq),
            norm_buffer,
            block=(block_size, 1, 1),
            grid=(grid_size, 1),
            shared=block_size * 8,
        )

        # Kernel 3
        result_gpu = gpuarray.empty(1, dtype=np.int32)

        self.check_convergence_kernel(norm_buffer, np.float64(tolerance), result_gpu, block=(1, 1, 1), grid=(1, 1))
        return True if result_gpu.get()[0] == 1 else False
