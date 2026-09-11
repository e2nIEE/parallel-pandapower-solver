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
- :code:`benchmark_n_1_pegase.py` Test script to calculate contingency analysis performance on IEEE Pegase 9241
- :code:`benchmark_n-1.bsbatch` SLURM job for contingency analysis test
- :code:`cpu_benchmark_results.py` Results aggregation tool for the cpu benchmark

Quick Start
===========

1. List Available Cases
.. code-block:: bash

   python tests/benchmark/benchmark_batched.py --list-cases


2. Test Locally
.. code-block:: bash

    # Test case9 with all methods
    python tests/benchmark/benchmark_batched.py --case case9 --methods pp,p3s,cpp-1thr,cpp-Nthr,gpu

    # Test with specific T values
    python tests/benchmark/benchmark_batched.py --case case118 --method gpu --T 64 256 1024

3. Run on Cluster
.. code-block:: bash

    # Submit all IEEE cases (job array: 0-29)
    sbatch tests/benchmark/benchmark_all.sbatch

    # Submit specific method
    METHOD=gpu sbatch tests/benchmark/benchmark_all.sbatch

4. Compile Results
.. code-block:: bash

    sbatch tests/resources/compile_results.sbatch
    python tests/compile_results.py --output-dir /mnt/home/user/p3s/results

5. Comparison of Results

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

Job fails with "Case not found"
-------------------------------

- Check case name spelling
- Use :code:`--list-cases` to see available cases


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

References
==========

- IEEE Test Cases from pandapower: https://pandapower.readthedocs.io/en/latest/networks/power_system_test_cases.html
- p3s Documentation: See project docs
