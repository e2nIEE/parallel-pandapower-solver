.. SPDX-FileCopyrightText: 2026 Fraunhofer IEE
..
.. SPDX-License-Identifier: BSD-3-Clause

##################
p3s IEEE Benchmark
##################

The :code:`tests/benchmark` directory contains the IEEE test case power-flow benchmark for the p3s library.

Directory Structure
===================

- :code:`benchmark_batched.py` Main benchmark script
- :code:`compile_results.py` Result aggregation tool
- :code:`benchmark_all.sbatch` SLURM job for all IEEE cases
- :code:`benchmark_cpu.sbatch` SLURM job to run pegase with different core counts
- :code:`benchmark_gpu.bsbatch` SLURM job for all IEEE cases on gpu only
- :code:`benchmark_n_1_pegase.py` Standalone N-1 contingency-analysis benchmark on IEEE Pegase 9241
- :code:`benchmark_n-1.bsbatch` SLURM job for contingency analysis test
- :code:`cpu_benchmark_results.py` Results aggregation tool for the cpu benchmark

Quick Start
===========

1. Test Locally
.. code-block:: bash

    # Test case9 with all methods
    python tests/benchmark/benchmark_batched.py --case case9 --methods pp,p3s,cpp-1thr,cpp-Nthr,gpu

    # Test with specific T values
    python tests/benchmark/benchmark_batched.py --case case118 --method gpu --T 64 256 1024

2. Run on a Cluster
.. code-block:: bash

    # Submit all IEEE cases (job array: 0-29)
    sbatch tests/benchmark/benchmark_all.sbatch

    # Submit specific method
    METHOD=gpu sbatch tests/benchmark/benchmark_all.sbatch

3. Compile Results
.. code-block:: bash

    sbatch tests/resources/compile_results.sbatch
    python tests/compile_results.py --output-dir /mnt/home/user/p3s/results

4. Comparison of Results

Result comparison is optional but will be performed, if method pp is selected.

Available Methods
-----------------

+------------------+------------------+-----------------------+
| Method           | Implementation   | Notes                 |
+------------------+------------------+-----------------------+
| :code:`pp`       | pandapower.runpp | Pure Python baseline  |
| :code:`p3s`      | NewtonPowerflow  | Python Newton-Raphson |
| :code:`cpp`      | C++ KLU solver   | Multi-threaded        |
| :code:`cpp-1thr` | C++ KLU solver   | Single-threaded       |
| :code:`cpp-Nthr` | C++ KLU solver   | Custom threads        |
| :code:`gpu`      | CUDA             | GPU-accelerated       |
+------------------+------------------+-----------------------+

Configuration
=============

Environment Variables
---------------------

- :code:`P3S_BENCH_CUDA=1` Enable GPU benchmarking
- :code:`CUDA_VISIBLE_DEVICES` Set GPU device

T Values (Time Steps)
---------------------

Default: `[64, 256, 1024, 2048, 4096, 8760, 17520, 35040]`
    - 35040: 1 year in 15 min increments

Output
------

Results are saved as JSON files:
.. code-block:: none

   {output_dir}/{job_id}/{job_id}_{case}.json

Example:
output-dir: :code:`/mnt/home/user/p3s/results`
:code:`/mnt/home/user/p3s/results/123456/123456_case9.json`

Troubleshooting
===============

GPU not available
-----------------

- Set :code:`P3S_BENCH_CUDA=1` before running
- Check CUDA installation


Case doesn't converge
---------------------

- Some cases (especially large) may need adjusted tolerance

Results missing
---------------

- Check output directory exists
- Check log files for errors

Advanced Usage
==============

Custom Method Selection
-----------------------

.. code-block:: bash

    # Only test C++ and GPU
    python tests/benchmark/benchmark_batched.py --case case9 --methods cpp,gpu

    # Only single-threaded C++
    python tests/benchmark/benchmark_batched.py --case case14 --method cpp-1thr


Custom T Values
---------------

.. code-block:: bash

    # Test with different time steps
    python tests/benchmark/benchmark_batched.py --case case300 --T 100 500 1000 5000

Manual Result Compilation
-------------------------

.. code-block:: bash

    python compile_results.py \
      --include-partial \
      --include-error-unknown \
      --output-dir ./results/235449/

N-1 Contingency Benchmark
=========================

The :code:`benchmark_n_1_pegase.py` script is a standalone N-1 benchmark (not a pytest test,
as a full pegase N-1 takes minutes and would interfere with the test pipeline). It times a full
single-element N-1 contingency analysis -- every line **and** every transformer taken out one
at a time -- comparing:

- :code:`p3s`'s batched C++/KLU solver (:code:`solve_contingencies_cpp`, all cores), which
  shares one symbolic factorization across the whole batch, against
- a pandapower per-contingency loop (a fresh :code:`runpp` per outage), the reference an N-1
  study would otherwise run.

GPU backend (fully-resident polar cuSolverRf path) is also supported via
:code:`solve_contingencies_cuda`.

Quick Start
-----------

.. code-block:: bash

    # All lines + transformers
    python -m tests.benchmark.benchmark_n_1_pegase

    # First 200 contingencies (quick estimate)
    python -m tests.benchmark.benchmark_n_1_pegase --limit 200

    # p3s only (skip the slow pandapower loop)
    python -m tests.benchmark.benchmark_n_1_pegase --skip-pandapower

    # 8 threads + spot-check 25 contingencies against pandapower
    python -m tests.benchmark.benchmark_n_1_pegase --threads 8 --validate 25

GPU backend
-----------

.. code-block:: bash

    python -m tests.benchmark.benchmark_n_1_pegase --backend gpu --limit 2000
    python -m tests.benchmark.benchmark_n_1_pegase --backend gpu --gpu-max-chunk 512
    python -m tests.benchmark.benchmark_n_1_pegase --backend both --limit 2000  # CPU vs GPU

:code:`--backend gpu` needs pycuda + a CUDA GPU + nvcc on PATH (the polar kernels compile at
import). :code:`--backend both` times CPU and GPU on the same net and reports the GPU/CPU
speedup and their max voltage disagreement on mutually-converged cases. On a small GPU (e.g.
4 GB), cap the per-chunk batch with :code:`--gpu-max-chunk` (the solver also chunks by free
memory automatically). :code:`--backend cpp` (default) is the original nr_klu path.

Options
-------

- :code:`--limit N` Only benchmark the first N branches (default: all lines + trafos)
- :code:`--threads N` nr_klu OpenMP threads (0 = all cores, 1 = serial; default 0)
- :code:`--num-threads N...` Number of threads (only supported for method cpp)
- :code:`--skip-pandapower` Time p3s only (skip the slow pandapower loop)
- :code:`--validate N` Spot-check N contingencies vs pandapower (default 10)
- :code:`--chunk N` CPU only: solve in memory-bounded chunks of N contingencies
- :code:`--backend {cpp,gpu,both}` Which p3s solver to time (default cpp)
- :code:`--gpu-max-chunk N` GPU: cap the per-chunk batch size
- :code:`--gpu-backend {rf,qr,cudss}` GPU linear-solve backend (default cudss)
- :code:`--output-dir / -o` Output directory for results JSON
- :code:`--job-id / -j` Job ID for output file naming

Notes
-----

- :code:`init="flat"` is used: p3s's simplified DC model does a flat start and converges
  Pegase 9241 in ~6 iterations.
- The pandapower loop dominates wall-clock (~0.25 s/contingency). For the full ~16k
  contingencies that is ~1.1 hour, so use :code:`--limit` for a quick estimate or
  :code:`--skip-pandapower` to time only p3s.
- With :code:`--chunk`, the full result table (n_bus x L complex128, ~2.4 GB for pegase over
  all ~16k branches) is kept in memory one chunk at a time, but no result table is retained,
  which disables :code:`--validate`.

References
==========

- IEEE Test Cases from pandapower: https://pandapower.readthedocs.io/en/latest/networks/power_system_test_cases.html
- p3s Documentation: See project docs
