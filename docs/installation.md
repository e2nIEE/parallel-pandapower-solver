<!--
SPDX-FileCopyrightText: 2026 Fraunhofer IEE

SPDX-License-Identifier: BSD-3-Clause
-->

# Installation

## Stable release

To install p3s, run this command in your terminal (without p3s-cpp):

```sh
uv add p3s
```

Or if you prefer to use `pip`:

```sh
pip install p3s
```

## From source

The source files for p3s can be downloaded from the [Github repo](https://github.com/e2nIEE/parallel-pandapower-solver).

You can clone the public repository:

```sh
git clone git://github.com/e2nIEE/parallel-pandapower-solver
```

Once you have a copy of the source, you can install it with:

```sh
cd p3s
uv pip install .
uv pip install p3s/cpp
```

Take a look into `docs/build.md` for more details.
