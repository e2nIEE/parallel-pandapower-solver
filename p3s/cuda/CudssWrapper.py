# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""ctypes bindings for NVIDIA cuDSS (the modern sparse direct solver).

cuDSS replaces the deprecated cusolverRf (which segfaults on CUDA 12.4).
This module loads ``libcudss.so`` (Linux) / ``cudss64_*.dll`` (Windows) and
declares argtypes/restype for the subset of the C API the batched Newton
solver needs. See ``p3s/cuda/cudss_batch.py`` for the ``CudssBatch``
backend built on top of it.

Enum values are the stable constants from ``cudss.h`` (unchanged across cuDSS 0.x). If a
future header renumbers them, override the module-level constants below.
"""
import ctypes
import glob
import os
import sys

# --- enums (cudss.h) --------------------------------------------------------
CUDSS_STATUS_SUCCESS = 0

# cudssPhase_t -- BIT FLAGS (verified from cudss_data_types.h). SOLVE is the OR of all five
# SOLVE_* sub-phases (fwd_perm | fwd | diag | bwd | bwd_perm). Getting these wrong is silent:
# a bad "SOLVE" value runs some other phase and never writes the solution.
CUDSS_PHASE_REORDERING = 1 << 0
CUDSS_PHASE_SYMBOLIC_FACTORIZATION = 1 << 1
CUDSS_PHASE_ANALYSIS = CUDSS_PHASE_REORDERING | CUDSS_PHASE_SYMBOLIC_FACTORIZATION  # 3
CUDSS_PHASE_FACTORIZATION = 1 << 2       # 4
CUDSS_PHASE_REFACTORIZATION = 1 << 3     # 8
CUDSS_PHASE_SOLVE = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8)  # 496 (all SOLVE_*)

# cudssMatrixType_t
CUDSS_MTYPE_GENERAL = 0
CUDSS_MTYPE_SYMMETRIC = 1
CUDSS_MTYPE_HERMITIAN = 2
CUDSS_MTYPE_SPD = 3
CUDSS_MTYPE_HPD = 4

# cudssMatrixViewType_t
CUDSS_MVIEW_FULL = 0
CUDSS_MVIEW_LOWER = 1
CUDSS_MVIEW_UPPER = 2

# cudssIndexBase_t
CUDSS_BASE_ZERO = 0
CUDSS_BASE_ONE = 1

# cudssLayout_t
CUDSS_LAYOUT_COL_MAJOR = 0
CUDSS_LAYOUT_ROW_MAJOR = 1

# cudaDataType_t (from CUDA library_types.h) -- verified values, NOT sequential:
CUDA_R_32F = 0
CUDA_R_64F = 1      # float64 (values, rhs, solution)
CUDA_R_32I = 10     # int32 (CSR offsets / column indices)

# cudssConfigParam_t -- sequential indices, verified against cudss_data_types.h (cuDSS 0.8,
# comments stripped before counting). Do NOT guess: a wrong index sets the wrong parameter,
# and cudssConfigSet returns status 3 (or silently mis-sets) -- e.g. UBATCH_SIZE is 18, not
# the naive count 12.
CUDSS_CONFIG_REORDERING_ALG = 0
CUDSS_CONFIG_FACTORIZATION_ALG = 1
CUDSS_CONFIG_SOLVE_ALG = 2
CUDSS_CONFIG_MATCHING_ALG = 3
CUDSS_CONFIG_SOLVE_MODE = 4
CUDSS_CONFIG_IR_N_STEPS = 5
CUDSS_CONFIG_IR_TOL = 6
CUDSS_CONFIG_PIVOT_TYPE = 7
CUDSS_CONFIG_PIVOT_THRESHOLD = 8
CUDSS_CONFIG_PIVOT_EPSILON = 9
CUDSS_CONFIG_MAX_LU_NNZ = 10
CUDSS_CONFIG_HYBRID_MEMORY_MODE = 11
CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT = 12
CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY = 13
CUDSS_CONFIG_HOST_NTHREADS = 14
CUDSS_CONFIG_HYBRID_EXECUTE_MODE = 15
CUDSS_CONFIG_PIVOT_EPSILON_ALG = 16
CUDSS_CONFIG_ND_NLEVELS = 17
CUDSS_CONFIG_UBATCH_SIZE = 18       # uniform-batch size (systems sharing one pattern)
CUDSS_CONFIG_UBATCH_INDEX = 19
CUDSS_CONFIG_USE_SUPERPANELS = 20
CUDSS_CONFIG_DEVICE_COUNT = 21
CUDSS_CONFIG_DEVICE_INDICES = 22
CUDSS_CONFIG_SCHUR_MODE = 23
CUDSS_CONFIG_DETERMINISTIC_MODE = 24
CUDSS_CONFIG_ND_UBFACTOR = 25

# cudssPivotType_t (verified from cudss_data_types.h)
CUDSS_PIVOT_AUTO = 0
CUDSS_PIVOT_NONE = 1        # disable pivot search -- much faster; safe for well-conditioned J


def _load_cudss():
    """Load libcudss, searching the pip-wheel path first (Linux) or the toolkit (Windows).

    The ``nvidia-cudss-cuXX`` wheel ships ``libcudss.so`` inside the package (not on the
    default loader path), so a bare name fails even when installed. Glob the wheel
    dir, then LD_LIBRARY_PATH, then bare names. RTLD_GLOBAL so its CUDA deps resolve.
    """
    if sys.platform == "win32":
        cands = []
        # pip wheel (nvidia-cudss-cuXX): nvidia/cuXX/bin/cudss64_*.dll
        try:
            import site
            for b in site.getsitepackages():
                cands += glob.glob(os.path.join(b, "nvidia", "**", "cudss64_*.dll"),
                                   recursive=True)
        except Exception:
            pass
        p = os.environ.get("CUDA_PATH")
        if p:
            cands += glob.glob(os.path.join(p, "**", "cudss64_*.dll"), recursive=True)
        cands = sorted(set(cands), reverse=True) + ["cudss64_0.dll", "cudss.dll"]
        # cuDSS depends on the CUDA runtime + cuBLAS (cudart/cublas) and an OpenMP runtime;
        # register likely dirs on the DLL search path so those resolve: each candidate's own
        # dir, the CUDA toolkit bin, and every nvidia-wheel bin.
        dep_dirs = [os.path.dirname(c) for c in cands if os.path.dirname(c)]
        p = os.environ.get("CUDA_PATH")
        if p:
            # CUDA 13 moved the runtime DLLs to bin\x64; older toolkits use bin.
            dep_dirs += [os.path.join(p, "bin", "x64"), os.path.join(p, "bin")]
        try:
            import site
            for b in site.getsitepackages():
                dep_dirs += glob.glob(os.path.join(b, "nvidia", "**", "bin"),
                                      recursive=True)
        except Exception:
            pass
        for d in dict.fromkeys(dep_dirs):     # de-dup, keep order
            if d and os.path.isdir(d):
                try:
                    os.add_dll_directory(d)
                except Exception:
                    pass
        last = None
        for c in cands:
            try:
                return ctypes.WinDLL(c)
            except OSError as e:
                last = e
        raise OSError(f"could not load cudss DLL (tried {cands}). Last error: {last}")

    # Linux: pip wheel dir, then LD_LIBRARY_PATH, then bare versioned names.
    cands = []
    try:
        import site
        bases = list(site.getsitepackages())
        try:
            bases.append(site.getusersitepackages())
        except Exception:
            pass
        for b in bases:
            cands += glob.glob(os.path.join(b, "nvidia", "**", "libcudss.so*"),
                               recursive=True)
    except Exception:
        pass
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if d:
            cands += glob.glob(os.path.join(d, "libcudss.so*"))
    # prefer versioned real files (sorted desc), then bare names via the loader path
    cands = sorted(set(cands), reverse=True) + ["libcudss.so", "libcudss.so.0"]
    last = None
    for c in cands:
        try:
            return ctypes.CDLL(c, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            last = e
    raise OSError(
        f"could not load libcudss.so (tried {cands}). "
        f"pip install nvidia-cudss-cu12 and/or add its lib dir to LD_LIBRARY_PATH. "
        f"Last error: {last}")


_libcudss = _load_cudss()

_VP = ctypes.c_void_p
_I = ctypes.c_int
_I64 = ctypes.c_int64


def _decl(name, restype, argtypes):
    fn = getattr(_libcudss, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


# --- lifecycle --------------------------------------------------------------
_decl("cudssCreate", _I, [_VP])                       # cudssHandle_t*
_decl("cudssDestroy", _I, [_VP])
_decl("cudssConfigCreate", _I, [_VP])
_decl("cudssConfigDestroy", _I, [_VP])
_decl("cudssDataCreate", _I, [_VP, _VP])              # handle, cudssData_t*
_decl("cudssDataDestroy", _I, [_VP, _VP])             # handle, cudssData_t
_decl("cudssSetStream", _I, [_VP, _VP])
# cudssConfigSet(config, param, value, sizeInBytes)
_decl("cudssConfigSet", _I, [_VP, _I, _VP, ctypes.c_size_t])

# --- version ----------------------------------------------------------------
try:
    _decl("cudssGetProperty", _I, [_I, ctypes.POINTER(_I)])
except AttributeError:
    pass

# --- matrix wrappers --------------------------------------------------------
# single CSR (used for the representative/analysis matrix if needed)
_decl("cudssMatrixCreateCsr", _I,
      [_VP, _I64, _I64, _I64, _VP, _VP, _VP, _VP, _I, _I, _I, _I, _I, _I])
_decl("cudssMatrixCreateDn", _I,
      [_VP, _I64, _I64, _I64, _VP, _I, _I])

# batched CSR: nrows/ncols/nnz are POINTERS to per-batch arrays; rowStart/rowEnd/colInd/
# values are ARRAYS OF DEVICE POINTERS (const void* const*). We pass all as c_void_p to
# device buffers we build in CudssBatch.
_decl("cudssMatrixCreateBatchCsr", _I,
      [_VP, _I64, _VP, _VP, _VP, _VP, _VP, _VP, _VP, _I, _I, _I, _I, _I, _I])
_decl("cudssMatrixCreateBatchDn", _I,
      [_VP, _I64, _VP, _VP, _VP, _VP, _I, _I, _I])
_decl("cudssMatrixDestroy", _I, [_VP])

# --- execute ----------------------------------------------------------------
# cudssExecute(handle, int phase, config, data, A, x(solution), b(rhs))
_decl("cudssExecute", _I, [_VP, _I, _VP, _VP, _VP, _VP, _VP])


def check(status, what):
    if status != CUDSS_STATUS_SUCCESS:
        raise RuntimeError(f"{what} failed with cudss status {status}")


def version():
    """(major, minor, patch) or None if cudssGetProperty is unavailable."""
    if not hasattr(_libcudss, "cudssGetProperty"):
        return None
    out = []
    for t in (0, 1, 2):  # MAJOR, MINOR, PATCH
        v = _I(-1)
        _libcudss.cudssGetProperty(t, ctypes.byref(v))
        out.append(v.value)
    return tuple(out)
