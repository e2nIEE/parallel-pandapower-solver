Usage
=====

To use graviton in a project:

.. code-block:: sh

    import graviton


Solvers
-------

- **Newton-Raphson** (:code:`graviton.NewtonPowerflow`) — the general AC power-flow solver implemented in Python
  (transmission, PV/gen buses, trafos).
- **CuDSS** (:code:`graviton.NewtonPowerflowCuda`) - also a general AC power-flow solver implemented in C++ and CUDA.
