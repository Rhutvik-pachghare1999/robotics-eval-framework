#!/bin/bash
# ── ASU SOL SLURM: motor/ESC telemetry diagnosis (rows 47-54) ──────────────
# Check if 24.41 Hz is propeller RPM (tracks throttle) or control loop artifact.
# Writes a diagnostic artifact (NOT a result manifest) to results/.
#SBATCH --job-name=uav-motor-esc-diag
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

cd /scratch/rpachgha/robeval/uav/repo

echo "Host: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}  Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

python experiments/uav/diagnose_motor_esc.py \
    --source-dir-dronepropa "$DRONEPROPA_SOURCE_DIR" \
    --n-flights 12 \
    --out "results/uav_motor_esc_diagnosis.json"

echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
