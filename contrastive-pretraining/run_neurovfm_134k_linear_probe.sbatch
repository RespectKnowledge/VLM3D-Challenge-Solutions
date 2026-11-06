#!/bin/bash
#SBATCH --job-name=nvfm_cls_134k
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/u6rm/my_projects/MR_RATE_EXPERIMENTS/logs/nvfm_cls_134k_%j.out
#SBATCH --error=/projects/u6rm/my_projects/MR_RATE_EXPERIMENTS/logs/nvfm_cls_134k_%j.err

set -euo pipefail

ROOT=/projects/u6rm/my_projects/foundation_models/VLM3D-Dockers/mr_challenges/MR-RATE-NeuroVFM/mr_challenges/abnclass_example_docker/MR-RATE/contrastive-pretraining

FEATURES=/projects/u6rm/my_projects/MR_RATE_EXPERIMENTS/lp_features_neurovfm_frozen_134k
RESULTS=/projects/u6rm/my_projects/MR_RATE_EXPERIMENTS/lp_results_neurovfm_frozen_134k

mkdir -p "$RESULTS"
mkdir -p /projects/u6rm/my_projects/MR_RATE_EXPERIMENTS/logs

cd "$ROOT"

echo "========================================"
echo "NeuroVFM MR-RATE classification"
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Features: $FEATURES"
echo "Results:  $RESULTS"
echo "========================================"

echo
echo "Feature directory:"
ls -lh "$FEATURES"

echo
echo "Checking cached feature files..."
find "$FEATURES" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort

echo
echo "Starting 14-label linear probe..."

python scripts/linear_probe.py \
    --features_dir "$FEATURES" \
    --results_dir "$RESULTS"

echo
echo "========================================"
echo "CLASSIFICATION FINISHED"
echo "Finished: $(date)"
echo "Results:"
find "$RESULTS" -maxdepth 2 -type f -printf '%p\n' | sort
echo "========================================"
