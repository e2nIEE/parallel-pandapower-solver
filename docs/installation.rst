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

    cd graviton
    uv pip install .
