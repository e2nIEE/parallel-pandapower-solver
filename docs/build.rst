.. SPDX-FileCopyrightText: 2026 Fraunhofer IEE
..
.. SPDX-License-Identifier: BSD-3-Clause

Building the C++ Extension
==========================

This guide explains how to build the compiled C++ power-flow extension, ``nr_klu``
(a pybind11 module backed by SuiteSparse/KLU). It is shipped as a **separate
``p3s-cpp`` distribution** so the base ``p3s`` install stays pure-Python, and it is
opt-in because building it needs a C++17 compiler and SuiteSparse/KLU.

After a successful build, the module is installed **into the ``p3s`` package** by
CMake, so it resolves without any path juggling:

.. code-block:: python

   from p3s import nr_klu  # installed build
   # fallback: from p3s.cpp import nr_klu

Prerequisites
-------------

1. **C++17 compiler**

   - Windows: MSVC (Visual Studio, with the C++ desktop development workload)
   - Linux: ``g++`` or ``clang++``
   - macOS: Xcode Command Line Tools / clang

2. **SuiteSparse / KLU**

   - The easiest cross-platform route is conda-forge, which ships the headers,
     libraries **and** the CMake config packages the build uses:

     .. code-block:: bash

        conda install -c conda-forge suitesparse

   - A system ``libsuitesparse-dev`` also works on Linux; ``cmake/FindKLU.cmake``
     locates it.

3. **Python >= 3.10**

4. **pybind11, numpy and scikit-build-core** are pulled in automatically by the
   build backend — you do not need to install them yourself (see
   ``p3s/cpp/pyproject.toml``).

5. **OpenMP** (optional) — enables the multi-threaded batched solve
   (``Solver.solve_batch`` with ``n_threads > 1``). It is auto-detected; without it
   the batch path still works, just single-threaded.

Building
--------

The build is driven by **pip + scikit-build-core**, which runs the CMake build
in ``p3s/cpp/CMakeLists.txt`` for you.

From an index
~~~~~~~~~~~~~

.. code-block:: bash

   pip install parallel-pandapower-solver[cpp]        # pulls in the published p3s-cpp distribution

From a source checkout
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install ./p3s/cpp

Host-tuned local build (for development)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default the wheel is built without host-specific tuning so it stays portable.
For a local build tuned to your CPU (``-march=native``), re-enable it:

.. code-block:: bash

   pip install ./p3s/cpp --config-settings=cmake.define.p3s_CPP_NATIVE=ON

``p3s_CPP_NATIVE`` defaults to ``OFF`` in the published wheel; use ``ON`` only for local
development builds.

Output
------

The built ``nr_klu`` extension is installed into the ``p3s`` package by the CMake
``install(TARGETS ... DESTINATION p3s)`` rule. There is no manual copying: after
the install, ``from p3s import nr_klu`` just works.

Windows / conda notes
---------------------

- Build and run from **within the activated conda env** that has SuiteSparse. The
  build finds KLU automatically (CMake adds ``%CONDA_PREFIX%\\Library`` to its search
  path), and importing from the env puts the KLU runtime DLLs (``klu``, ``amd``,
  ``btf``, ``colamd``, ``suitesparseconfig`` in ``Library\\bin``) on the loader path.
- Verified: a ``cp313`` wheel builds with MSVC, and ``from p3s import nr_klu`` solves
  correctly (single and batch).
- Do **not** put the source checkout on ``PYTHONPATH`` when testing the installed
  build — the source ``p3s/`` has no compiled ``nr_klu`` and would shadow the
  installed package.

Advanced: raw CMake
-------------------

The supported way to build is via pip (above); scikit-build-core runs the CMake
build internally. For debugging or unusual setups you can drive CMake against
``p3s/cpp/CMakeLists.txt`` directly, but this is **not** the supported path and
you must provide the Python, pybind11, numpy and KLU discovery yourself:

.. code-block:: bash

   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<path-to-python>
   cmake --build . --config Release

CMake locates Python and pybind11 via ``find_package``, resolves the numpy headers
through the build interpreter, and finds KLU via the ``SuiteSparse::KLU`` config
package (falling back to ``cmake/FindKLU.cmake``). Use ``-DKLU_ROOT=...`` or
``-DSuiteSparse_ROOT=...`` if KLU is in a non-standard location.

Troubleshooting
---------------

KLU / SuiteSparse not found
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install SuiteSparse, preferably from conda-forge:

.. code-block:: bash

   conda install -c conda-forge suitesparse

Or point the build at your install. ``cmake/FindKLU.cmake`` honours ``KLU_ROOT``,
``SuiteSparse_ROOT`` and ``CONDA_PREFIX``:

.. code-block:: bash

   # Windows (PowerShell)
   $env:KLU_ROOT = "C:\path\to\suitesparse"

   # Linux/macOS
   export KLU_ROOT=/path/to/suitesparse

numpy headers not found
~~~~~~~~~~~~~~~~~~~~~~~

CMake locates the numpy headers by asking the build interpreter for
``numpy.get_include()``, which is robust to numpy's ``_core/include`` layout change.
If you build with ``--no-build-isolation``, make sure numpy is installed in the
active environment first:

.. code-block:: bash

   pip install numpy

Multi-threaded batch not available
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The batched solve runs single-threaded unless OpenMP is detected at build time.
Install an OpenMP-capable toolchain (MSVC supports it out of the box; on Linux
install ``libomp-dev`` or rely on your compiler's built-in OpenMP) and rebuild.

Testing
-------

After building, verify the module imports and run the validation tests:

.. code-block:: python

   from p3s import nr_klu

The test suite lives in ``p3s/cpp/tests/``:

- ``test_batch.py`` — ``solve_batch`` vs scipy and thread-invariance
- ``test_integration.py`` — ``NewtonPowerflowCpp.calculate`` vs pandapower ``runpp``
- ``test_selfconsistent.py`` — converged voltages drive p3s's own mismatch to ~0

.. code-block:: bash

   python -m pytest p3s/cpp/tests
