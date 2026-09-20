#!/bin/bash
# ── ASU SOL SLURM: f_rot diagnosis (why is DronePropA f_rot pegged at 48.83?) ──
# Top-5 PSD peaks + nperseg sweep + per-axis + motor_CMD DC + SP1/SP2 sampling.
# Writes a diagnostic artifact (NOT a result manifest) to results/.
#SBATCH --job-name=uav-frot-diag
#SBATCH --partition=htc
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

source /scratch/rpachgha/robeval/uav/env/venv/bin/activate

export DRONEPROPA_SOURCE_DIR="/scratch/rpachgha/robeval/uav/source/dronepropa/DronePropA Motion Trajectories Dataset for Commercial Drones with Defective Propellers/DronePropA Motion Trajectories Dataset"
export TII_SOURCE_DIR="/scratch/rpachgha/robeval/uav/source/tii/repo/Dataset"

cd /scratch/rpachgha/robeval/uav/repo

echo "Host: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}  Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

python experiments/uav/diagnose_f_rot.py \
    --source-dir-dronepropa "$DRONEPROPA_SOURCE_DIR" \
    --source-dir-tii "$TII_SOURCE_DIR" \
    --n-flights 12 \
    --out "results/uav_f_rot_diagnosis.json"

echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
