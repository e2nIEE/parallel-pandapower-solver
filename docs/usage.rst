.. SPDX-FileCopyrightText: 2026 Fraunhofer IEE
..
.. SPDX-License-Identifier: BSD-3-Clause

Usage
-----

p3s is a set of CPU / GPU based AC power-flow solvers for
[pandapower](https://pandapower.readthedocs.io) networks. Every solver consumes a
:code:`pandapowerNet` and writes pandapower-style `res_*` tables back into it.

.. code-block:: python

    import p3s

Most functionality lives in submodules, so import the solver you need explicitly:

.. code-block:: python

from p3s.NewtonPowerflow import NewtonPowerflow            # pure-Python CPU solver
from p3s.NewtonPowerflowCpp import NewtonPowerflow as NewtonPowerflowCpp  # C++/KLU CPU solver
from p3s.cuda.NewtonPowerflowCuda import NewtonPowerflowCUDA              # GPU solver


Transformers
============

Networks with transformers need a tap table before they can be solved.
Compute it once and store it on the network:

.. code-block:: python

    from p3s.calculateTrafoTapTable import calculateTrafoCharacteristic

    calculateTrafoCharacteristic(net, inplace=True)


With :code:`inplace=True` the result is written to :code:`net.trafo_characteristic_table`;
with the default :code:`inplace=False` it is returned instead. Call this before any of
the solvers below for networks that contain trafos.

Newton-Raphson solver (:code:`p3s.NewtonPowerflow`)
===================================================

The general AC power-flow solver for transmission networks, PV/generator buses
and trafos. This is the pure-Python implementation (NumPy/SciPy).

.. code-block:: python

    from p3s.NewtonPowerflow import NewtonPowerflow

    npf = NewtonPowerflow(net)
    voltage = npf.calculate(net, init="dc", tolerance=1e-8, max_iterations=30)


- :code:`init="dc"` (default) initialises the voltage vector with a DC power flow; use
  :code:`init="flat"` for a 1.0 pu / 0° flat start. DC init preserves the voltage-magnitude
  set-points at PV buses.
- :code:`tolerance` (default `1e-5`) is the convergence threshold (infinity norm of the
  power mismatch);
- :code:`max_iterations` defaults to `30`.

The solver writes results back into the network (:code:`net.res_bus.vm_pu`,
:code:`net.res_bus.va_degree`, plus the other `res_*` tables), matching pandapower
:code:`runpp`, and also returns the converged voltage vector `voltage`. Any extra keyword
arguments are passed through to :code:`scipy.sparse.linalg.spsolve`.

The number of iterations taken is stored in :code:`net["_ppc"]["iterations"]`.

C++ / KLU accelerated solver (:code:`p3s.NewtonPowerflowCpp`)
=============================================================

A drop-in accelerated variant of the Newton solver that offloads the sparse
factorisation to the compiled `nr_klu` extension. It is opt-in because building it
needs a C++17 compiler and SuiteSparse/KLU (and, for the CUDA paths, the CUDA
toolkit):

.. code-block:: bash

    pip install p3s[cpp]     # from an index
    pip install ./p3s/cpp    # from a source checkout


.. code-block:: python

    from p3s.NewtonPowerflowCpp import NewtonPowerflow as NewtonPowerflowCpp

    npf = NewtonPowerflowCpp(net)
    voltage = npf.calculate(net, tolerance=1e-8, max_iterations=30)


Batched time-series
~~~~~~~~~~~~~~~~~~~

The C++ solver can solve many operating points at once, sharing the symbolic
factorisation across the whole batch:

.. code-block:: python

    npf = NewtonPowerflowCpp(net)
    voltages = npf.calculate_timeseries_cpp(net, timeseries, n_threads=0)


`voltages` has shape `(n_bus, T)`, one complex voltage per bus per time step.
:code:`n_threads=0` (default) uses all available threads; set it to `1` for a single
thread. See [Time-series data](#time-series-data) for the `timeseries` format.

GPU solver (:code:`p3s.cuda.NewtonPowerflowCUDA`)
=================================================

A CUDA-accelerated Newton solver. Batched solves run fully on the GPU:

.. code-block:: python

from p3s.cuda.NewtonPowerflowCuda import NewtonPowerflowCUDA

npf = NewtonPowerflowCUDA(net)
voltages = npf.calculate_timeseries_cuda(net, timeseries, batch_size=5)


`voltages` has shape `(n_bus, T)`. `batch_size` controls how many operating points
are solved per GPU chunk. The underlying linear solve can be selected via a
`backend` argument (e.g. `"cudss"`).

Time-series data
================

The batched solvers (:code:`calculate_timeseries_cpp`, :code:`calculate_timeseries_cuda`, ...)
accept a `timeseries` dictionary mapping each controlled `(element, variable)` pair
to an array of shape `(n_element, T)`:

.. code-block:: python

    timeseries = {
        ("load", "p_mw"):   p_matrix,   # (n_load,  T)
        ("load", "q_mvar"): q_matrix,   # (n_load,  T)
        ("sgen", "p_mw"):   ...,        # (n_sgen,  T)
        ("gen",  "p_mw"):   ...,        # (n_gen,   T)
    }


Supported elements are `load`, `sgen` and `gen`, and supported variables `p_mw`
and `q_mvar`. Column `t` of every array is applied as the set-points for time step
`t`. A convenient way to build this from pandapower `ConstControl` controllers is
`extract_timeseries` (see `tests/get_load_gen_matrix.py`).

N-1 contingency analysis (:code:`p3s.contingency`)
==================================================

Batch analysis of single-element contingencies (N-1). Every line and transformer
belonging to an `outage_group` is removed one group at a time and the resulting
system is solved in parallel.

First tag the branches you want to consider with an `outage_group` column:

.. code-block:: python

    net.line["outage_group"] = net.line.index          # each line its own group
    net.trafo["outage_group"] = net.trafo.index


Branches with a null (`NaN`/`0`) `outage_group` never appear in any contingency;
branches sharing an `outage_group` value are taken out together (e.g. parallel
branches or multi-branch groups).

Case generation (optional, backend-agnostic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can generate the contingency cases (`outage_group`) up front, without solving, to inspect or
reuse the shared sparsity pattern:

.. code-block:: python

    from p3s.contingency.case_generator import generate_cases

    batch = generate_cases(net, reslack_islands=False)


`batch` is a `ContingencyBatch` exposing the shared CSR pattern (`Yp`, `Yj`), the
intact-grid values `Yx_base`, the list of `groups` and the per-case values. The
generator is pure NumPy/SciPy and needs no compiled backend.

:code:`reslack_islands=True` re-slacks islands that form after an outage, so that the
whole island is served rather than treated as unserved. When True an island with
no original slack but >=1 generator with valid short-circuit data is referenced
by its lowest-Z generator (the bus is recorded as a pinned reference and counts
as served). When False, such islands are entirely unserved.

### CPU solve

.. code-block:: python

from p3s.contingency.solver_cpp import solve_contingencies_cpp

res = solve_contingencies_cpp(net, reslack_islands=False, n_threads=1)


:code:`reslack_islands=True` same as before, re-slacks islands that form after an outage, so that the
whole island is served rather than treated as unserved.

`n_threads` defines how many threads should be used in parallel. If none is given, all cores are used.


GPU solve
~~~~~~~~~

.. code-block:: python

    from p3s.contingency.solver_cuda import solve_contingencies_cuda

    res = solve_contingencies_cuda(net, reslack_islands=False, backend="cudss")


:code:`reslack_islands=True` same as before, re-slacks islands that form after an outage, so that the
whole island is served rather than treated as unserved.

`backend` allows to use different backends, `cudss` (cudss backend), `rf` (old cusolverRF backend)
and `qr` (cuda QR algorithm, more stable, but slower).

### Result table

Both solvers return a :code:`ContingencyResultTable`:

.. code-block:: python

    res.groups       # list of contingency names, one per column
    res.V            # (n_bus, L) complex voltages; NaN where a bus is unserved
    res.vm           # |V| view of res.V
    res.va           # angle view of res.V
    res.served       # (n_bus, L) bool, exact served/unserved mask
    res.converged    # (L,) bool, per-contingency convergence
    res.iterations   # (L,) iterations per contingency
