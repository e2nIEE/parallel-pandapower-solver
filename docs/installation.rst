.. SPDX-FileCopyrightText: 2026 Fraunhofer IEE
..
.. SPDX-License-Identifier: BSD-3-Clause

Installation
============

Stable release
--------------

To install the parallel pandapower solver , run this command in your terminal:

.. code-block:: sh

    uv add p3s

Or if you prefer to use ``pip``:

.. code-block:: sh

    pip install p3s

From source
-----------

The source files can be downloaded from the `Github repo <https://github.com/e2niee/parallel-pandapower-solver>`_.

You can either clone the public repository:

.. code-block:: sh

    git clone git://github.com/e2niee/parallel-pandapower-solver

Or download the `tarball <https://github.com/e2niee/parallel-pandapower-solver/tarball/master>`_:

.. code-block:: sh

    curl -OJL https://github.com/e2niee/parallel-pandapower-solver/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: sh

    cd parallel-pandapower-solver
    uv pip install .


Testing cuda hardware
---------------------

cuda-toolkit and cuDSS must be available,

.. code-block:: sh

    conda install -c nvidia/label/cuda-13.2.2 cuda-toolkit`

CUDA libraries are not part of this repository and must be acquired separately from NVIDIA, under their respective license terms.

Since many different cuda libraries exist, with different paths and setups, a test script exists:

.. code-block:: sh

    python -m p3s.cuda.check_cudss

this script will check if cudss is actually usable for p3s. Nvidia cuDSS is currently the newest sparse matrix solver,
available for Nvidia hardware.

.. code-block:: sh

    python -m p3s.cuda.diagnose_gpu

this script allows for in depth analysis if the powerflow solver actually works and that the hardware produces correct results.

