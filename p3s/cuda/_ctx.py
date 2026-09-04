# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared CUDA context setup for the GPU solver path.

Import this INSTEAD of ``pycuda.autoinit`` anywhere the resident polar solver, its kernels,
or a cuSolver/cuDSS backend run in the same process.

Why the PRIMARY context: cuDSS (and other CUDA libraries) operate on the device's *primary*
context. ``pycuda.autoinit`` instead creates a SEPARATE context; when cuDSS then activates
the primary context, any pycuda ``SourceModule`` compiled under autoinit's context becomes
invalid, and kernel launches fail with ``cuFuncSetBlockShape failed: invalid resource
handle``. Retaining the primary context (``pycuda.autoprimaryctx``) makes pycuda and cuDSS
share one context, so kernels and cuDSS interoperate. Falls back to ``autoinit`` on old
pycuda builds without ``autoprimaryctx`` (fine when cuDSS is not used).
"""

try:
    import pycuda.autoprimaryctx  # noqa: F401 (retains the device primary context)
except Exception:  # pragma: no cover - very old pycuda
    import pycuda.autoinit  # noqa: F401

import pycuda.driver as cuda  # noqa: F401 re-exported for convenience
