#!/bin/bash
# ── ASU SOL SLURM: cross-dataset UAV fault detection (binary healthy vs faulty) ──
# 3 escalation levels: naive (18 features), time_broad (54 features), coral (54 + CORAL)
# Both directions: DronePropA -> TII, TII -> DronePropA
# Writes result manifests to results/uav_cross_dataset_{level}.json
#SBATCH --job-name=uav-xfer
#SBATCH --partition=htc
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs results

source /scratch/rpachgha/robeval/uav/env/venv/bin/activate

export DRONEPROPA_SOURCE_DIR="/scratch/rpachgha/robeval/uav/source/dronepropa/DronePropA Motion Trajectories Dataset for Commercial Drones with Defective Propellers/DronePropA Motion Trajectories Dataset"
export TII_SOURCE_DIR="/scratch/rpachgha/robeval/uav/source/tii/repo/Dataset"

cd /scratch/rpachgha/robeval/uav/repo

echo "Host: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}  Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

python experiments/uav/run_cross_dataset.py \
    --source-dir-dronepropa "$DRONEPROPA_SOURCE_DIR" \
    --source-dir-tii "$TII_SOURCE_DIR" \
    --seed 42 \
    --levels naive time_broad coral \
    --out-dir results

echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
