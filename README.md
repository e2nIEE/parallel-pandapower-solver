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

* TODO

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
PtJ Grant ID
