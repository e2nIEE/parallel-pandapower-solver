<!--
SPDX-FileCopyrightText: 2026 Fraunhofer IEE

SPDX-License-Identifier: BSD-3-Clause
-->

# `p3s.cpp` — fast single-grid Newton-Raphson power flow (C++ / KLU)

`nr_klu` is a self-contained C++ power-flow solver, exposed to Python via pybind11,
that solves a **single large grid** Newton-Raphson power flow much faster than the
SciPy/Numba reference. On `case9241pegase` it is roughly **8× faster stateless** and
**~14× faster warm** than the scipy `splu` Newton baseline, validated to ≤1.4e-11 against
scipy.

This directory holds everything for that solver: the C++ source, its build script,
reference-fixture generator, validation tests, and standalone benchmarks. It lives under
`p3s/cpp/` alongside the other solver code; the built extension is also copied one
level up into the `p3s` package so it imports as `from p3s import nr_klu`.

## Why it is fast (design)

- **Polar formulation.** `Ybus` is stored as constant magnitude/angle arrays
  (`|Y|`, `∠Y`). Each Newton iteration recomputes only **one `sincos(δ)` per nonzero**
  and assembles the four Jacobian blocks with real multiply-adds — no complex
  multiply/conjugate/divide, and roughly half the memory traffic of a complex
  formulation.
- **Fused mismatch + Jacobian.** `eval_F_and_J` computes the power-balance mismatch and
  fills the Jacobian in a single pass that shares the per-edge `sincos`.
- **Constant sparsity pattern.** The Jacobian structure does not change across NR
  iterations, so the CSR pattern is built **once**; the linear solver does its symbolic
  analyze once and only re-numbers values afterwards.
- **KLU linear solve** (SuiteSparse): `klu_analyze` once, `klu_factor` on iteration 1,
  then the much cheaper `klu_refactor` on every later iteration (reuses the pivot
  ordering). Tuned with `btf=0` (a connected PF grid is one irreducible block, so BTF
  only adds cost) and AMD ordering (denser, slower with COLAMD).

The Jacobian layout (unknowns `x = [Δθ(pvpq); ΔVm(pq)]`):

```
[ dP/dθ(pvpq,pvpq)   dP/dVm(pvpq,pq) ]
[ dQ/dθ(pq,  pvpq)   dQ/dVm(pq,  pq) ]
```

## Python API

```python
import nr_klu            # standalone (this dir on sys.path)
# or:  from p3s import nr_klu

# --- stateful (recommended for many profiles on one topology) ---
s = nr_klu.Solver(Yp, Yj, Yx, pv, pq)      # KLU analyze done ONCE here
r = s.solve(Sbus, V0, max_iter=30, tol=1e-8)
V          = r["V"]            # complex128 converged bus voltages
iters      = r["iterations"]
converged  = r["converged"]
s.update_Y(Yx_new)            # refresh Ybus values without re-analyzing

# --- batched / time-series: many operating points on ONE topology ---
# Sbus is (n, T) complex128; V0 is (n, T) per-column or (n,) broadcast. Each column is
# an independent Newton solve sharing the single KLU symbolic analyze.
rb = s.solve_batch(Sbus_mat, V0, max_iter=30, tol=1e-8, n_threads=0)
Vb         = rb["V"]            # complex128 (n, T)
iters      = rb["iterations"]   # int32  (T,)
converged  = rb["converged"]    # bool   (T,)
# n_threads: 0 = all CPU cores (OpenMP), 1 = serial. Results are thread-invariant.

# --- stateless one-shot (analyze + factor every call) ---
r = nr_klu.solve_single(Yp, Yj, Yx, Sbus, V0, pv, pq,
                        max_iter=30, tol=1e-8, ordering=0, btf=0)
# stateless result additionally reports timing: t_setup_ms, t_solve_ms.
```

### Batched / time-series (`solve_batch`)

Newton cannot share a single factorization across operating points the way SAM does
(KLU is single-matrix / single-RHS), so the batch speedup comes from (a) amortizing the
one-time `klu_analyze`, (b) the cheap `klu_refactor` reuse, and (c) **solving independent
time steps in parallel across CPU cores via OpenMP**. Internally one read-only `Topology`
(CSR pattern + symbolic factorization) is shared; each worker owns a `SolveState` (its own
`klu_numeric` + scratch). The Python driver
`p3s.NewtonPowerflowCpp.calculate_timeseries_cpp(net, timeseries, n_threads=0)` wraps
this, reusing the shared Sbus/DC-init helpers in `p3s/timeseries.py` (same convention
as the GPU `calculate_timeseries_cuda`).

On case9241pegase the batch is **~85–93× faster than a scipy Newton loop**, with ~4.4×
of that from OpenMP threading (1→8 cores; sub-linear beyond, memory-bandwidth bound).

**Inputs** (all 0-based, `Ybus` in CSR):
- `Yp` (`int32`, `n+1`), `Yj` (`int32`, `nnz`), `Yx` (`complex128`, `nnz`) — CSR `Ybus`.
- `Sbus` (`complex128`, `n`) — complex bus power injections (p.u.).
- `V0` (`complex128`, `n`) — initial voltage (keep ref/PV magnitudes and ref angle).
- `pv`, `pq` (`int32`) — PV and PQ bus indices. The slack/ref bus is whatever is in
  neither set.

The **stateful `Solver`** is the intended path for p3s's "many operating points on
one topology" workload: the ~6–8 ms `klu_analyze` is paid once and amortized across all
subsequent `solve` calls. `p3s/NewtonPowerflowCpp.py` uses exactly this — it caches
a `Solver` per topology and invalidates it when the `Ybus` structure changes.

## Files

| File | Purpose |
|------|---------|
| `nr_klu.cpp` | The solver (pybind module: `Solver` with `solve`/`solve_batch`, `solve_single`, `debug_J`). |
| `test_batch.py` | Validates `solve_batch` vs scipy + thread-invariance on the `case*.npz` fixtures. |
| `CMakeLists.txt` | Portable CMake build (pybind11 + KLU + optional OpenMP); used by `pip install p3s[cpp]`. |
| `pyproject.toml` | scikit-build-core definition for the `p3s-cpp` distribution. |
| `cmake/FindKLU.cmake` | KLU locator fallback for installs without a CMake config package. |
| `test_integration.py` | `NewtonPowerflowCpp.calculate` vs pandapower `runpp` (vm/va error + timing). |
| `test_selfconsistent.py` | Converged `V` drives p3s's *own* power-balance mismatch to ~0 (isolates solver correctness from any p3s-vs-pandapower model gap). |

## Building

The portable build uses **CMake + scikit-build-core** and is verified on Linux (g++) and
Windows (MSVC, Python 3.13 + conda-forge SuiteSparse). It compiles `nr_klu` and installs
it into the `p3s` package, so `from p3s import nr_klu` resolves with no path
juggling.

### Prerequisites
- A C++17 compiler (gcc/clang, or MSVC on Windows).
- **SuiteSparse/KLU.** Easiest cross-platform route is conda-forge, which ships the
  headers, libraries *and* CMake config packages used by the build:
  ```bash
  conda install -c conda-forge suitesparse
  ```
  (System `libsuitesparse-dev` also works on Linux — `cmake/FindKLU.cmake` locates it.)

### Install
```bash
# opt-in extra of the main package (pulls in the p3s-cpp distribution)
pip install p3s[cpp]

# from a source checkout, build this sub-package directly:
pip install ./p3s/cpp

# local dev build tuned for the host CPU (-march=native):
pip install ./p3s/cpp --config-settings=cmake.define.p3s_CPP_NATIVE=ON
```

Windows notes:
- Build and run from **within the activated conda env** that has SuiteSparse. The build
  finds KLU automatically (the CMake adds `%CONDA_PREFIX%\Library` to the search path),
  and importing from the env puts the KLU runtime DLLs (`klu`/`amd`/`btf`/`colamd`/
  `suitesparseconfig`, in `Library\bin`) on the loader path. Verified: `cp313` wheel
  builds with MSVC and `from p3s import nr_klu` solves correctly (single + batch).
- Do **not** put the source checkout on `PYTHONPATH` when testing the installed build —
  the source `p3s/` has no compiled `nr_klu` and would shadow the installed package.


## Benchmark results (case9241pegase, n=9241, nnz=37655, 6 iters)

| method                                     | time | speedup vs scipy |
|--------------------------------------------|------|------------------|
| `pandapower` baseline                     | ~207 ms | 1× |
| `solve_single` (analyze+factor every call) | ~24.5 ms | ~8.5× |
| `Solver` **warm** (analyze once, reuse)    | ~15 ms | ~14× |
| (one-time `Solver` analyze)                | ~6.4 ms | — |

All cases validate to ≤1.4e-11 vs pandapower.
