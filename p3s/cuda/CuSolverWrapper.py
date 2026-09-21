# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

import ctypes
import glob
import os
import sys


def _load_cuda_lib(stem: str):
    """Load a CUDA library (cuSPARSE / cuSOLVER) on either Linux or Windows.

    On Linux the libraries are ``lib<stem>.so``. On Windows they are versioned
    ``<stem>64_<major>.dll`` and are not on the default loader path, so we search
    ``CUDA_PATH``/standard toolkit locations and load by absolute path. The
    cusolverRf and low-level host-LU symbols this module needs are present in both
    the CUDA 11 and CUDA 12/13 DLLs (verified on cusolver64_11 and cusolver64_12).
    """
    if sys.platform != "win32":
        # Linux: the unversioned ``lib<stem>.so`` is only present with the -dev package;
        # on many clusters only the versioned runtime (``lib<stem>.so.12`` / ``.so.11``)
        # is installed, so try those too. Load with RTLD_GLOBAL so cuSolver's internal
        # dependencies (cublas/cusparse) resolve; a plain load can leave them unbound and
        # segfault on the first call rather than fail cleanly here. Prefer the highest
        # version. Also honor CUDA_HOME/CUDA_PATH/LD_LIBRARY_PATH explicitly.
        names = [f"lib{stem}.so"]
        search = []
        for env in ("CUDA_HOME", "CUDA_PATH"):
            p = os.environ.get(env)
            if p:
                search += [os.path.join(p, "lib64"), os.path.join(p, "lib")]
        search += [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if d]
        for d in search:
            if os.path.isdir(d):
                names += sorted(glob.glob(os.path.join(d, f"lib{stem}.so.*")), reverse=True)
        # also try bare versioned names via the loader's own search path
        names += [f"lib{stem}.so.12", f"lib{stem}.so.11"]

        last_err = None
        for name in names:
            try:
                return ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                last_err = e
                continue
        raise OSError(
            f"could not load lib{stem}.so (tried {names}). "
            f"Add $CUDA_HOME/lib64 to LD_LIBRARY_PATH. Last error: {last_err}"
        )

    else:
        # Windows: try the bare loader first (works if the toolkit bin is on PATH),
        # then fall back to globbing known toolkit install locations. Prefer the
        # highest version found (sorted last).
        search_dirs = []
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path and os.path.isdir(cuda_path):
            search_dirs.append(cuda_path)
        search_dirs.append(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")

        candidates = []
        for base in search_dirs:
            if base and os.path.isdir(base):
                candidates += glob.glob(os.path.join(base, "**", f"{stem}64_*.dll"), recursive=True)

        for dll in sorted(set(candidates)):
            try:
                return ctypes.WinDLL(dll)
            except OSError:
                continue

        # Last resort: let the OS resolver try (relies on PATH).
        return ctypes.WinDLL(f"{stem}64_12.dll")


# cuSparse
_libcusparse = _load_cuda_lib("cusparse")


class cusparseMatDescr_t(ctypes.Structure):
    _fields_ = [
        ("MatrixType", ctypes.c_int),
        ("FillMode", ctypes.c_int),
        ("DiagType", ctypes.c_int),
        ("IndexBase", ctypes.c_int),
    ]


_libcusparse.cusparseCreate.restype = int
_libcusparse.cusparseCreate.argtypes = [ctypes.c_void_p]

_libcusparse.cusparseDestroy.restype = int
_libcusparse.cusparseDestroy.argtypes = [ctypes.c_void_p]

_libcusparse.cusparseCreateMatDescr.restype = int
_libcusparse.cusparseCreateMatDescr.argtypes = [ctypes.c_void_p]

_libcusparse.cusparseSetMatType.restype = int
_libcusparse.cusparseSetMatType.argtypes = [ctypes.c_void_p, ctypes.c_int]

_libcusparse.cusparseSetMatIndexBase.restype = int
_libcusparse.cusparseSetMatIndexBase.argtypes = [ctypes.c_void_p, ctypes.c_int]

# %%

# cuSOLVER
_libcusolver = _load_cuda_lib("cusolver")

_libcusolver.cusolverSpCreate.restype = int
_libcusolver.cusolverSpCreate.argtypes = [ctypes.c_void_p]

_libcusolver.cusolverSpDestroy.restype = int
_libcusolver.cusolverSpDestroy.argtypes = [ctypes.c_void_p]


_libcusolver.cusolverSpDcsrlsvqr.restype = int
_libcusolver.cusolverSpDcsrlsvqr.argtypes = [
    ctypes.c_void_p,  # cusolverSpHandle_t handle,
    ctypes.c_int,  # int m,
    ctypes.c_int,  # int nnz,
    ctypes.c_void_p,  # cusparseMatDescr_t descrA,
    ctypes.c_void_p,  # const double *csrValA,
    ctypes.c_void_p,  # const int *csrRowPtrA,
    ctypes.c_void_p,  # const int *csrColIndA,
    ctypes.c_void_p,  # const double *b,
    ctypes.c_double,  # double tol,
    ctypes.c_int,  # int reorder,
    ctypes.c_void_p,  # double *x,
    ctypes.c_void_p,  # int *singularity
]


_libcusolver.cusolverSpDcsrlsvluHost.restype = int
_libcusolver.cusolverSpDcsrlsvluHost.argtypes = [
    ctypes.c_void_p,  # cusolverSpHandle_t handle,
    ctypes.c_int,  # int m,
    ctypes.c_int,  # int nnz,
    ctypes.c_void_p,  # cusparseMatDescr_t descrA,
    ctypes.c_void_p,  # const double *csrValA,
    ctypes.c_void_p,  # const int *csrRowPtrA,
    ctypes.c_void_p,  # const int *csrColIndA,
    ctypes.c_void_p,  # const double *b,
    ctypes.c_double,  # double tol,
    ctypes.c_int,  # int reorder,
    ctypes.c_void_p,  # double *x,
    ctypes.c_void_p,  # int *singularity
]


_libcusolver.cusolverSpXcsrsymamdHost.restype = int
_libcusolver.cusolverSpXcsrsymamdHost.argtypes = [
    ctypes.c_void_p,  # cusolverSpHandle_t handle,
    ctypes.c_int,  # int n,
    ctypes.c_int,  # int nnzA,
    ctypes.c_void_p,  # const cusparseMatDescr_t descrA,
    ctypes.c_void_p,  # const int *csrRowPtrA,
    ctypes.c_void_p,  # const int *csrColIndA,
    ctypes.c_void_p,  # int *p
]

_libcusolver.cusolverSpXcsrsymrcmHost.restype = int  # cusolverStatus_t
_libcusolver.cusolverSpXcsrsymrcmHost.argtypes = [
    ctypes.c_void_p,  # cusolverSpHandle_t handle,
    ctypes.c_int,  # int n,
    ctypes.c_int,  # int nnzA,
    ctypes.c_void_p,  # const cusparseMatDescr_t descrA,
    ctypes.c_void_p,  # const int *csrRowPtrA,
    ctypes.c_void_p,  # const int *csrColIndA,
    ctypes.c_void_p,  # int *p
]

_libcusolver.cusolverSpXcsrperm_bufferSizeHost.restype = int  # cusolverStatus_t
_libcusolver.cusolverSpXcsrperm_bufferSizeHost.argtypes = [
    ctypes.c_void_p,  # cusolverSpHandle_t handle,
    ctypes.c_int,  # int m,
    ctypes.c_int,  # int n,
    ctypes.c_int,  # int nnzA,
    ctypes.c_void_p,  # const cusparseMatDescr_t descrA,
    ctypes.c_void_p,  # int *csrRowPtrA,
    ctypes.c_void_p,  # int *csrColIndA,
    ctypes.c_void_p,  # const int *p,
    ctypes.c_void_p,  # const int *q,
    ctypes.c_void_p,  # size_t *bufferSizeInBytes
]

_libcusolver.cusolverSpXcsrpermHost.restype = int  # cusolverStatus_t
_libcusolver.cusolverSpXcsrpermHost.argtypes = [
    ctypes.c_void_p,  # cusolverSpHandle_t handle,
    ctypes.c_int,  # int m,
    ctypes.c_int,  # int n,
    ctypes.c_int,  # int nnzA,
    ctypes.c_void_p,  # const cusparseMatDescr_t descrA,
    ctypes.c_void_p,  # int *csrRowPtrA,
    ctypes.c_void_p,  # int *csrColIndA,
    ctypes.c_void_p,  # const int *p,
    ctypes.c_void_p,  # const int *q,
    ctypes.c_void_p,  # int *map,
    ctypes.c_void_p,  # void *pBuffer
]

_libcusolver.cusolverSpCreateCsrluInfoHost.restype = int  # cusolverStatus_t
_libcusolver.cusolverSpCreateCsrluInfoHost.argtypes = [
    ctypes.c_void_p  # csrluInfoHost_t *info
]

_libcusolver.cusolverSpXcsrluAnalysisHost.restype = int  # cusolverStatus_t
_libcusolver.cusolverSpXcsrluAnalysisHost.argtypes = [
    ctypes.c_void_p,  # cusolverSpHandle_t handle,
    ctypes.c_int,  # int n,
    ctypes.c_int,  # int nnzA,
    ctypes.c_void_p,  # const cusparseMatDescr_t descrA,
    ctypes.c_void_p,  # const int *csrRowPtrA,
    ctypes.c_void_p,  # const int *csrColIndA,
    ctypes.c_void_p,  # csrluInfoHost_t info
]

_libcusolver.cusolverSpDcsrluBufferInfoHost.restype = int  # cusolverStatus_t
_libcusolver.cusolverSpDcsrluBufferInfoHost.argtypes = [
    ctypes.c_void_p,  # cusolverSpHandle_t handle,
    ctypes.c_int,  # int n,
    ctypes.c_int,  # int nnzA,
    ctypes.c_void_p,  # const cusparseMatDescr_t descrA,
    ctypes.c_void_p,  # const double *csrValA,
    ctypes.c_void_p,  # const int *csrRowPtrA,
    ctypes.c_void_p,  # const int *csrColIndA,
    ctypes.c_void_p,  # csrluInfoHost_t info,
    ctypes.c_void_p,  # size_t *internalDataInBytes,
    ctypes.c_void_p,  # size_t *workspaceInBytes
]

_libcusolver.cusolverSpDcsrluFactorHost.restype = int  # cusolverStatus_t
_libcusolver.cusolverSpDcsrluFactorHost.argtypes = [
    ctypes.c_void_p,  # cusolverSpHandle_t handle,
    ctypes.c_int,  # int n,
    ctypes.c_int,  # int nnzA,
    ctypes.c_void_p,  # const cusparseMatDescr_t descrA,
    ctypes.c_void_p,  # const double *csrValA,
    ctypes.c_void_p,  # const int *csrRowPtrA,
    ctypes.c_void_p,  # const int *csrColIndA,
    ctypes.c_void_p,  # csrluInfoHost_t info,
    ctypes.c_double,  # double pivot_threshold,
    ctypes.c_void_p,  # void *pBuffer
]

_libcusolver.cusolverSpDcsrluZeroPivotHost.restype = int  # cusolverStatus_t
_libcusolver.cusolverSpDcsrluZeroPivotHost.argtypes = [
    ctypes.c_void_p,  # cusolverSpHandle_t handle,
    ctypes.c_void_p,  # csrluInfo[Host]_t info,
    ctypes.c_double,  # double tol,
    ctypes.POINTER(ctypes.c_int),  # int *position
]

_libcusolver.cusolverSpDcsrluSolveHost.restype = int  # cusolverStatus_t
_libcusolver.cusolverSpDcsrluSolveHost.argtypes = [
    ctypes.c_void_p,  # cusolverSpHandle_t handle,
    ctypes.c_int,  # int n,
    ctypes.c_void_p,  # const double *b,
    ctypes.c_void_p,  # double *x,
    ctypes.c_void_p,  # csrluInfoHost_t info,
    ctypes.c_void_p,  # void *pBuffer
]

# nnz(L) and nnz(U) of the host LU factorization (needed before extraction)
_libcusolver.cusolverSpXcsrluNnzHost.restype = int  # cusolverStatus_t
_libcusolver.cusolverSpXcsrluNnzHost.argtypes = [
    ctypes.c_void_p,  # cusolverSpHandle_t handle,
    ctypes.POINTER(ctypes.c_int),  # int *nnzLRef,
    ctypes.POINTER(ctypes.c_int),  # int *nnzURef,
    ctypes.c_void_p,  # csrluInfoHost_t info
]

# Extract P, Q, L, U from Plu*B*Qlu^T = L*U (L has implicit unit diagonal)
_libcusolver.cusolverSpDcsrluExtractHost.restype = int  # cusolverStatus_t
_libcusolver.cusolverSpDcsrluExtractHost.argtypes = [
    ctypes.c_void_p,  # cusolverSpHandle_t handle,
    ctypes.c_void_p,  # int *P,
    ctypes.c_void_p,  # int *Q,
    ctypes.c_void_p,  # const cusparseMatDescr_t descrL,
    ctypes.c_void_p,  # double *csrValL,
    ctypes.c_void_p,  # int *csrRowPtrL,
    ctypes.c_void_p,  # int *csrColIndL,
    ctypes.c_void_p,  # const cusparseMatDescr_t descrU,
    ctypes.c_void_p,  # double *csrValU,
    ctypes.c_void_p,  # int *csrRowPtrU,
    ctypes.c_void_p,  # int *csrColIndU,
    ctypes.c_void_p,  # csrluInfoHost_t info,
    ctypes.c_void_p,  # void *pBuffer
]
# %%
# CuSolverRF Functions for refactorization
_libcusolver.cusolverRfCreate.restype = int  # cusolverStatus_t
_libcusolver.cusolverRfCreate.argtypes = [
    ctypes.c_void_p,  # cusolverRfHandle_t *handle
]

_libcusolver.cusolverRfDestroy.restype = int  # cusolverStatus_t
_libcusolver.cusolverRfDestroy.argtypes = [ctypes.c_void_p]  # cusolverRfHandle_t handle

_libcusolver.cusolverRfSetNumericProperties.restype = int  # cusolverStatus_t
_libcusolver.cusolverRfSetNumericProperties.argtypes = [
    ctypes.c_void_p,  # cusolverRfHandle_t handle,
    ctypes.c_double,  # double zero,
    ctypes.c_double,  # double boost
]

_libcusolver.cusolverRfSetAlgs.restype = int  # cusolverStatus_t
_libcusolver.cusolverRfSetAlgs.argtypes = [
    ctypes.c_void_p,  # cusolverRfHandle_t handle,
    ctypes.c_int,  # gluFactorization_t fact_alg [1, 2, 3]
    ctypes.c_int,  # gluTriangularSolve_t alg [1, 2, 3]
]

_libcusolver.cusolverRfSetMatrixFormat.restype = int  # cusolverStatus_t
_libcusolver.cusolverRfSetMatrixFormat.argtypes = [
    ctypes.c_void_p,  # cusolverRfHandle_t handle,
    ctypes.c_int,  # gluMatrixFormat_t format, CSR = 0, CSC = 1
    ctypes.c_int,  # gluUnitDiagonal_t diag, STORED_L = 0, STORED_U = 1, ASSUMED_L = 2, ASSUMED_U = 3
]

_libcusolver.cusolverRfSetResetValuesFastMode.restype = int  # cusolverStatus_t
_libcusolver.cusolverRfSetResetValuesFastMode.argtypes = [
    ctypes.c_void_p,  # cusolverRfHandle_t handle,
    ctypes.c_int,  # gluResetValuesFastMode_t fastMode, OFF = 0, ON = 1
]

_libcusolver.cusolverRfBatchSetupHost.restype = int  # cusolverStatus_t
_libcusolver.cusolverRfBatchSetupHost.argtypes = [
    ctypes.c_int,  # int batchSize,
    ctypes.c_int,  # int n,
    ctypes.c_int,  # int nnzA,
    ctypes.c_void_p,  # int* h_csrRowPtrA,
    ctypes.c_void_p,  # int* h_csrColIndA,
    ctypes.c_void_p,  # double *h_csrValA_array[],
    ctypes.c_int,  # int nnzL,
    ctypes.c_void_p,  # int* h_csrRowPtrL,
    ctypes.c_void_p,  # int* h_csrColIndL,
    ctypes.c_void_p,  # double *h_csrValL,
    ctypes.c_int,  # int nnzU,
    ctypes.c_void_p,  # int* h_csrRowPtrU,
    ctypes.c_void_p,  # int* h_csrColIndU,
    ctypes.c_void_p,  # double *h_csrValU,
    ctypes.c_void_p,  # int* h_P,
    ctypes.c_void_p,  # int* h_Q,
    # /* Output */
    ctypes.c_void_p,  # cusolverRfHandle_t handle
]

_libcusolver.cusolverRfBatchAnalyze.restype = int  # cusolverStatus_t
_libcusolver.cusolverRfBatchAnalyze.argtypes = [ctypes.c_void_p]  # cusolverRfHandle_t handle

_libcusolver.cusolverRfBatchResetValues.restype = int  # cusolverStatus_t
_libcusolver.cusolverRfBatchResetValues.argtypes = [  # /* Input (in the device memory) */
    ctypes.c_int,  # int batchSize,
    ctypes.c_int,  # int n,
    ctypes.c_int,  # int nnzA,
    ctypes.c_void_p,  # int* csrRowPtrA,
    ctypes.c_void_p,  # int* csrColIndA,
    ctypes.c_void_p,  # double* csrValA_array[],
    ctypes.c_void_p,  # int *P,
    ctypes.c_void_p,  # int *Q,
    # /* Output */
    ctypes.c_void_p,  # cusolverRfHandle_t handle
]

_libcusolver.cusolverRfBatchRefactor.restype = int  # cusolverStatus_t
_libcusolver.cusolverRfBatchRefactor.argtypes = [ctypes.c_void_p]  # cusolverRfHandle_t handle

_libcusolver.cusolverRfBatchSolve.restype = int  # cusolverStatus_t
_libcusolver.cusolverRfBatchSolve.argtypes = [  # /* Input (in the device memory) */
    ctypes.c_void_p,  # cusolverRfHandle_t handle,
    ctypes.c_void_p,  # int *P,
    ctypes.c_void_p,  # int *Q,
    ctypes.c_int,  # int nrhs,
    ctypes.c_void_p,  # double *Temp,
    ctypes.c_int,  # int ldt,
    # /* Input/Output (in the device memory) */
    ctypes.c_void_p,  # double *XF_array[],
    # /* Input */
    ctypes.c_int,  # int ldxf
]
