<!--
SPDX-FileCopyrightText: 2026 Fraunhofer IEE

SPDX-License-Identifier: BSD-3-Clause
-->

# p3s - parallel pandapower solver

![PyPI version](https://img.shields.io/pypi/v/p3s.svg)
[![Documentation Status](https://readthedocs.org/projects/p3s/badge/?version=latest)](https://p3s.readthedocs.io/en/latest/?version=latest)

CPU / GPU based AC/DC Powerflow solver

* PyPI package: https://pypi.org/project/p3s/
* Free software: BSD-3-Clause
* Documentation: https://p3s.readthedocs.io.

## Features

### Solvers
- C++ Newton-Raphson with KLU, single and multi-threaded (using openMP)
- GPU-resident polar Newton solver (cuDSS, cuSolverRF and cuSolverSp QR)
- reference python / numba implementation

### Study opportunities
- batched timeseries calculation
- N-1 contingency analysis with islanding detection (including a "reslack" feature)
- station controllers solved inside the Newton Raphson (still under review)

### Robustness
- Armijo damped Newton-Raphson
- DC power-flow initialisation with flat start fallback
- voltage-band plausibility check against non-physical roots
- per-case convergence and per-bus served masks

### Integration
- operates directly on pandapower networks (standalone)
- zero-copy numpy interface, using pybind11, no per-call marshalling
- Linux and Windows native builds; pip installable
- Will be directly integrated in pandapower

### Verification
- all results are verified against pandapower
- test pipeline automatically tests every change
- accuracy is 1e-11 vs pandapower

## Installation

Compiled C++/KLU Newton-Raphson solver (the `nr_klu` extension). Opt-in because it needs a C++17 compiler and SuiteSparse/KLU at build time.
Installs the separate `p3s-cpp` distribution, whose CMake build drops `nr_klu` into the p3s package so `from p3s import nr_klu` works.

```bash
pip install p3s[cpp]              # from an index (published p3s-cpp)
pip install .[cpp]                # from a checkout (see note below)
```

From a source checkout, p3s-cpp is not on an index, so build it explicitly:
  pip install ./p3s/cpp
(SuiteSparse must be available, e.g. `conda install -c conda-forge suitesparse`.)
(cuda-toolkit must be available, e.g. `conda install -c nvidia/label/cuda-12.4.0 cuda-toolkit`.)

Look into docs/installation.md for more details.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.


---

The software in this Github project only contains calls to NVIDIA software already installed by the user (e.g. CUDA);
this software must be obtained and licensed separately and independently by the user.
