<!--
SPDX-FileCopyrightText: 2026 Fraunhofer IEE

SPDX-License-Identifier: BSD-3-Clause
-->

# p3s IEEE Benchmark

This directory contains the IEEE test case power-flow benchmark for the p3s library.

## Directory Structure

```
tests/benchmark/
├── benchmark_batched.py          # Main benchmark script
├── compile_results.py            # Result aggregation tool
├── benchmark_all.sbatch          # SLURM job for all IEEE cases
├── benchmark_cpu.sbatch          # SLURM job to run pegase with different core counts
├── benchmark_gpu.bsbatch         # SLURM job for all IEEE cases on gpu only
└── cpu_benchmark_results.py      # Results aggregation tool for the cpu benchmark
```

## Quick Start

### 1. List Available Cases
```bash
python tests/benchmark_batched.py --list-cases
```

### 2. Test Locally
```bash
# Test case9 with all methods
python tests/benchmark_batched.py --case case9 --methods pp,grav,cpp-1thr,cpp-Nthr,gpu

# Test with specific T values
python tests/benchmark_batched.py --case case118 --method gpu --T 64 256 1024
```

### 3. Run on Cluster
```bash
# Submit all IEEE cases (job array: 0-29)
sbatch tests/benchmark_all.sbatch

# Submit specific method
METHOD=gpu sbatch tests/benchmark_all.sbatch
```

### 4. Compile Results
```bash
sbatch tests/resources/compile_results.sbatch
python tests/compile_results.py --output-dir /mnt/home/user/p3s/results
```

## Available Methods

| Method     | Implementation   | Notes                 |
|------------|------------------|-----------------------|
| `pp`       | pandapower.runpp | Pure Python baseline  |
| `p3s`      | NewtonPowerflow  | Python Newton-Raphson |
| `cpp`      | C++ KLU solver   | Multi-threaded        |
| `cpp-1thr` | C++ KLU solver   | Single-threaded       |
| `cpp-Nthr` | C++ KLU solver   | Custom threads        |
| `gpu`      | CUDA             | GPU-accelerated       |

## Configuration

### Environment Variables

- `P3S_BENCH_CUDA=1` - Enable GPU benchmarking
- `CUDA_VISIBLE_DEVICES` - Set GPU device (SLURM sets automatically)

### SLURM Parameters

Default configuration in `benchmark_all.sbatch`:
- Nodes: 1
- CPUs: 64 (2×32-core CPUs)
- GPUs: 2 (NVIDIA)
- Memory: 256 GB
- Time limit: 6 hours

### T Values (Time Steps)

Default: `[64, 256, 1024, 2048, 4096, 8760, 17520, 35040]`
    - 35040: 1 year in 15 min increments

## Output

Results are saved as JSON files:
```
{output_dir}/{job_id}/{job_id}_{case}.json
```

Example:
output-dir: `/mnt/home/user/p3s/results`
```
/mnt/home/user/p3s/results/123456/123456_case9.json
```

## Troubleshooting

### Job fails with "Case not found"
- Check case name spelling
- Use `--list-cases` to see available cases

### GPU not available
- Set `P3S_BENCH_CUDA=1` before running
- Check CUDA installation

### Case doesn't converge
- Some cases (especially large) may need adjusted tolerance

### Results missing
- Check output directory exists
- Verify SLURM job completed successfully
- Check log files for errors

## Advanced Usage

### Custom Method Selection
```bash
# Only test C++ and GPU
python tests/benchmark_batched.py --case case9 --methods cpp,gpu

# Only single-threaded C++
python tests/benchmark_batched.py --case case14 --method cpp-1thr
```

### Custom T Values
```bash
# Test with different time steps
python tests/benchmark_batched.py --case case300 --T 100 500 1000 5000
```

### Manual Result Compilation
```bash
python compile_results.py \
  --include-partial \
  --include-error-unknown \
  --output-dir ./results/235449/
```

## References

- IEEE Test Cases from pandapower: https://pandapower.readthedocs.io/en/latest/networks/power_system_test_cases.html
- p3s Documentation: See project docs
- SLURM Manual: https://slurm.schedmd.com/
