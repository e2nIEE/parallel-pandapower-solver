<!--
SPDX-FileCopyrightText: 2026 Fraunhofer IEE

SPDX-License-Identifier: BSD-3-Clause
-->

# p3s - parallel pandapower solver

[![PyPI version](https://img.shields.io/pypi/v/parallel-pandapower-solver.svg)](https://pypi.org/project/parallel-pandapower-solver/)
[![Documentation Status](https://readthedocs.org/projects/parallel-pandapower-solver/badge/?version=latest)](https://parallel-pandapower-solver.readthedocs.io/en/latest/?version=latest)

CPU / GPU based AC/DC Powerflow solver

* PyPI package: https://pypi.org/project/parallel-pandapower-solver/
* Free software: BSD-3-Clause
* Documentation: https://parallel-pandapower-solver.readthedocs.io.

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
(cuda-toolkit[^1] must be available, e.g. `conda install -c nvidia/label/cuda-12.4.0 cuda-toolkit`.)

Look into docs/installation.md for more details.

## Acknowledgment

The code in this repository was created as part of the research project “GRAVITON”, supported by the German Federal Ministry for Economic Affairs and Climate Action (BMWE) on the basis of a decision by the German Bundestag (grant no. 03EIM4109).

<img alt="Supported by: Federal Ministry for Economic Affairs and Energy on the basis of a decision by the German Bundestag" src="https://github.com/e2nIEE/parallel-pandapower-solver/blob/main/docs/BMWE_gefoerdert_en_RGB.png" width="230" height="230" />

## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.

---
[^1]: CUDA libraries are not part of this repository and must be acquired separately from NVIDIA, under their respective license terms.
