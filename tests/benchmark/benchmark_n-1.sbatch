#!/bin/bash
# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

#SBATCH --job-name=p3s_n-1
#SBATCH --partition=kes

#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64

#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output=/mnt/home/%u/parallel-pandapower-solver/results/logs/benchmark_n-1_%A.out
#SBATCH --error=/mnt/home/%u/parallel-pandapower-solver/results/logs/benchmark_n-1_%A.err

EXIT_CODE=0
BASE_PATH="${HOME}/parallel-pandapower-solver"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_PATH/results}"
OUTPUT_DIR+="/${SLURM_ARRAY_JOB_ID}"
LIMIT=15000

# Configuration
CPUS=${CPUS:-"1 2 4 8 16 32 64"}
GPU_CHUNKS=${GPU_CHUNKS:-"64 128 256 512 1024 2028"}

# Load modules and activate conda environment
module purge
module load gcc12
export PYTHONBIN="${HOME}/.conda/envs/p3s"

echo "=== Job configuration ==="
echo "CPUS: ${CPUS}"
echo "Output Directory: ${OUTPUT_DIR}"

echo "=== SLURM Job Array ==="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"

echo "=== Starting n-1 benchmark ==="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run n-1 benchmark
cd "${p3s_DIR:-$(dirname $(dirname $(readlink -f $0)))}"

# Test specific method on this case
${PYTHONBIN}/bin/python ${BASE_PATH}/tests/benchmark/benchmark_n_1_pegase.py \
    --num-threads $CPUS \
    --gpu-max-chunk $GPU_CHUNKS \
    --validate 0 \
    --limit $LIMIT \
    --output-dir "$OUTPUT_DIR" \
    --job-id "$SLURM_ARRAY_JOB_ID" || EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "=== Failed processing case: "case9241pegase" with ${CPU} cores ==="
fi

echo "=== Benchmark completed with exit code: $EXIT_CODE ==="

exit $EXIT_CODE
