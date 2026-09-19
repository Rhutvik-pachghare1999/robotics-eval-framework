#!/bin/bash
# ── ASU SOL SLURM template ───────────────────────────────────────────────────
# This is a TEMPLATE. It is NOT auto-submitted by any script in this repo.
# HPC jobs burn real allocation — a human reviews and launches:
#     sbatch slurm/experiment.sbatch <experiment_module> <config.yaml>
#
# Copy + edit per experiment. Keep --time/--gres honest to what the job needs.
#SBATCH --job-name=robeval
#SBATCH --partition=general           # public (7d) | general (14d) | htc (4h)
#SBATCH --gres=gpu:a100:1             # request the smallest GPU that fits
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

EXPERIMENT="${1:?usage: sbatch experiment.sbatch <experiment_module> <config.yaml>}"
CONFIG="${2:?missing config}"

module load mamba/latest 2>/dev/null || true
source activate robeval 2>/dev/null || {
  echo "Create the env first (login node):"
  echo "  mamba create -n robeval python=3.11 -y && mamba activate robeval && pip install -e ."
  exit 1
}

echo "Host: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}  git: $(git rev-parse --short HEAD)"

# The experiment MUST write a result manifest (schema/result_manifest.schema.json)
# with real metrics + provenance. It must not print/hand-write numbers.
srun python -m "$EXPERIMENT" --config "$CONFIG" --out "results/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.json"

# Validate the produced manifest immediately — fail the job if provenance missing.
python scripts/validate_results.py "results/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.json"
