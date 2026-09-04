.. SPDX-FileCopyrightText: 2026 Fraunhofer IEE
..
.. SPDX-License-Identifier: BSD-3-Clause

Usage
=====

To use parallel-pandapower-solver in a project:

.. code-block:: sh

    import p3s


Solvers
-------

- **Newton-Raphson** (:code:`p3s.NewtonPowerflow`) — the general AC power-flow solver implemented in Python
  (transmission, PV/gen buses, trafos).
- **CuDSS** (:code:`p3s.NewtonPowerflowCuda`) - also a general AC power-flow solver implemented in C++ and CUDA.
