#!/bin/bash
# ── ASU SOL SLURM: UAV fault classification, stronger models ────────────────
# Step 6: HistGradientBoosting (CPU) + 1D-CNN (GPU) on DronePropA.
#
# This is a TEMPLATE. A human reviews and launches:
#     sbatch slurm/uav_fault_clf_strong.sbatch
#
# Smoke run first (htc, 4h cap, 30 epochs). Do NOT launch the full sweep
# until the smoke run produces a valid manifest.
#SBATCH --job-name=uav-strong
#SBATCH --partition=htc                 # htc (4h) for smoke; general (14d) for full
#SBATCH --gres=gpu:a100:1               # smallest GPU that fits; a30 if available
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00                 # 2h: 30-epoch CNN ~25 min on A100, + GB ~5 min
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

# Activate the SOL venv (Python 3.9 + torch + sklearn + scipy)
source /scratch/rpachgha/robeval/uav/env/venv/bin/activate

export DRONEPROPA_SOURCE_DIR="/scratch/rpachgha/robeval/uav/source/dronepropa/DronePropA Motion Trajectories Dataset for Commercial Drones with Defective Propellers/DronePropA Motion Trajectories Dataset"

cd /scratch/rpachgha/robeval/uav/repo

# Sync the latest code from the local repo (human runs rsync before sbatch).
# git_sha is passed explicitly so the manifest points at reproducible code.
GIT_SHA="${1:-f940e31}"

echo "Host: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}  git: ${GIT_SHA}"
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Run both models: gb (features) + cnn (raw channels, 30 epochs).
python experiments/uav/run_fault_clf_strong.py \
    --models gb,cnn \
    --cnn-epochs 30 \
    --cnn-batch-size 16 \
    --git-sha "${GIT_SHA}" \
    --output "results/uav_fault_clf_strong.json"

echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Validate the produced manifest — fail the job if provenance is missing.
python scripts/validate_results.py results/uav_fault_clf_strong.json
