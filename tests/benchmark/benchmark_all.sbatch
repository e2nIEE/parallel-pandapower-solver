#!/bin/bash
# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause

#SBATCH --job-name=p3s_bench
#SBATCH --partition=kes

#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-26%14
#SBATCH --output=/mnt/home/%u/p3s/results/logs/%A/benchmark_%a.out
#SBATCH --error=/mnt/home/%u/p3s/results/logs/%A/benchmark_%a.err

EXIT_CODE=0
BASE_PATH="${HOME}/p3s"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_PATH/results}"
OUTPUT_DIR+="/${SLURM_ARRAY_JOB_ID}"

# Configuration
METHOD=${METHOD:-"all"}
T_VALUES=${T_VALUES:-"64 256 1024 2048 4096 8760 17520 35040"}

# Load modules and activate conda environment
module load gcc12
CUDA_HOME=/usr/local/cuda-12.4/

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONBIN="${HOME}/.conda/envs/p3s"
export P3S_BENCH_CUDA=1

# Case definitions
CASE_NAMES=("case4gs" "case5" "case6ww" "case9" "case14" "case24_ieee_rts" \
            "GBreducednetwork" "case30" "case_ieee30" "case33bw" "case39" "case57" "case89pegase" \
            "case118" "case145" "iceland" "case_illinois200" "case300" "case1354pegase" "case1888rte" \
            "GBnetwork" "case2848rte" "case2869pegase" "case3120sp" "case6470rte" "case6515rte" "case9241pegase")

CASE_FOR_WORKER="${CASE_NAMES[SLURM_ARRAY_TASK_ID]}"

echo "=== Job configuration ==="
echo "Total workers: $SLURM_ARRAY_TASK_COUNT"
echo "Method: ${METHOD}"
echo "T values: ${T_VALUES}"
echo "Output Directory: ${OUTPUT_DIR}"

echo "=== SLURM Job Array ==="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "GPUs: ${SLURM_JOB_GPUS}"
echo "Cases: ${CASE_FOR_WORKER}"

echo "=== Starting benchmark ==="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run benchmark
cd "${p3s_DIR:-$(dirname $(dirname $(readlink -f $0)))}"

if [ "$METHOD" == "all" ]; then
    # Test all methods on this case
    # method is currently excluded
    ${PYTHONBIN}/bin/python ${BASE_PATH}/tests/benchmark/benchmark_batched.py \
        --case "$CASE_FOR_WORKER" \
        --methods pp,grav,cpp,gpu \
        --T $T_VALUES \
        --output-dir "$OUTPUT_DIR" \
        --drop-results \
        --job-id "$SLURM_ARRAY_JOB_ID" || EXIT_CODE=$?
else
    # Test specific method on this case
    ${PYTHONBIN}/bin/python ${BASE_PATH}/tests/benchmark/benchmark_batched.py \
        --case "$CASE_FOR_WORKER" \
        --method "$METHOD" \
        --T $T_VALUES \
        --drop-results \
        --output-dir "$OUTPUT_DIR" \
        --job-id "$SLURM_ARRAY_JOB_ID" || EXIT_CODE=$?
fi

if [ $EXIT_CODE -ne 0 ]; then
    echo "=== Failed processing case: $CASE_NAME ==="
fi

echo "=== Benchmark completed with exit code: $EXIT_CODE ==="

exit $EXIT_CODE
