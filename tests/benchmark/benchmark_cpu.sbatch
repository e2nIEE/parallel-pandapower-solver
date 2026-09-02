#!/bin/bash
# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

#SBATCH --job-name=p3s_bench
#SBATCH --partition=kes

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/mnt/home/%u/p3s/results/logs/benchmark_cpu_%A.out
#SBATCH --error=/mnt/home/%u/p3s/results/logs/benchmark_cpu_%A.err

EXIT_CODE=0
BASE_PATH="${HOME}/p3s"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_PATH/results}"
OUTPUT_DIR+="/${SLURM_ARRAY_JOB_ID}"

# Configuration
CPUS=${CPUS:-"1 2 4 8 16 32 64"}

# Load modules and activate conda environment
module load gcc12
export PYTHONBIN="${HOME}/.conda/envs/p3s"

echo "=== Job configuration ==="
echo "CPUS: ${CPUS}"
echo "Output Directory: ${OUTPUT_DIR}"

echo "=== SLURM Job Array ==="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"

echo "=== Starting benchmark ==="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run benchmark
cd "${p3s_DIR:-$(dirname $(dirname $(readlink -f $0)))}"

# Test specific method on this case
${PYTHONBIN}/bin/python ${BASE_PATH}/tests/benchmark/benchmark_batched.py \
    --num-threads $CPUS \
    --T 35040 \
    --output-dir "$OUTPUT_DIR" \
    --drop-results \
    --job-id "$SLURM_ARRAY_JOB_ID" || EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "=== Failed processing case: "case9241pegase" with ${CPU} cores ==="
fi

echo "=== Benchmark completed with exit code: $EXIT_CODE ==="

exit $EXIT_CODE
