#!/bin/bash
#SBATCH --job-name=uav-amplitude-perm
#SBATCH --partition=htc
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Adversarial re-run: amplitude ablation + correct permutation test.
# CPU-only (RandomForest, no GPU needed).
#
# Experiment 1: Amplitude ablation — rebuild features with ONLY shape/normalized
#   features (kurtosis, crest_factor, shape_factor, skew, spectral_centroid,
#   spectral_kurtosis, band_energy_ratio). Drop rms, absolute energy. Re-run
#   DP->TII RF. Does 0.929 survive?
#
# Experiment 2: Correct permutation test — shuffle SOURCE (DronePropA) labels,
#   RETRAIN RF, evaluate on TII. >=1000 perms. p=(k+1)/(n+1).
#   (The old scrutinize_cross_dataset.py shuffled TII test labels against
#   fixed predictions — invalid, does NOT retrain under the null.)
#
# Output:
#   results/uav_amplitude_ablation.json  (REAL-PUBLIC manifest, validated)
#   diagnostics/uav_permutation_test.json (diagnostic, NOT a manifest)

set -euo pipefail
mkdir -p logs results diagnostics

source /scratch/rpachgha/robeval/uav/env/venv/bin/activate

export DRONEPROPA_SOURCE_DIR="/scratch/rpachgha/robeval/uav/source/dronepropa/DronePropA Motion Trajectories Dataset for Commercial Drones with Defective Propellers/DronePropA Motion Trajectories Dataset"
export TII_SOURCE_DIR="/scratch/rpachgha/robeval/uav/source/tii/repo/Dataset"
export UAV_GIT_SHA="$(cd /scratch/rpachgha/robeval/uav/repo 2>/dev/null && git rev-parse --short=7 HEAD 2>/dev/null || echo unknown)"

cd /scratch/rpachgha/robeval/uav/repo

echo "Host: $(hostname)  Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "UAV_GIT_SHA: ${UAV_GIT_SHA}"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}  CPUs: ${SLURM_CPUS_PER_TASK:-?}  Mem: ${SLURM_MEM_PER_NODE:-?}"
echo ""

# ─── Experiment 1: Amplitude ablation ───────────────────────────────────────
echo "============================================================"
echo "  Experiment 1: Amplitude ablation (shape-only features)"
echo "  DROPS: rms, band_energy_low (absolute), band_energy_high (absolute)"
echo "  KEEPS: kurtosis, crest_factor, shape_factor, skew,"
echo "         spectral_centroid, spectral_kurtosis, band_energy_ratio"
echo "============================================================"
python experiments/uav/run_amplitude_ablation.py \
    --source-dir-dronepropa "$DRONEPROPA_SOURCE_DIR" \
    --source-dir-tii "$TII_SOURCE_DIR" \
    --seed 42 \
    --out results/uav_amplitude_ablation.json
echo ""
echo "Amplitude ablation manifest: results/uav_amplitude_ablation.json"
echo ""

# ─── Experiment 2: Correct permutation test (shuffle SOURCE labels, retrain) ─
echo "============================================================"
echo "  Experiment 2: Correct permutation test (shuffle SOURCE labels, RETRAIN)"
echo "  1000 permutations, p = (k+1)/(n+1)"
echo "============================================================"
python experiments/uav/run_permutation_test.py \
    --source-dir-dronepropa "$DRONEPROPA_SOURCE_DIR" \
    --source-dir-tii "$TII_SOURCE_DIR" \
    --n-permutations 1000 \
    --seed 42 \
    --out diagnostics/uav_permutation_test.json
echo ""
echo "Permutation test diagnostic: diagnostics/uav_permutation_test.json"
echo ""

# ─── Validate the amplitude ablation manifest ────────────────────────────────
echo "============================================================"
echo "  Validating amplitude ablation manifest"
echo "============================================================"
python scripts/validate_results.py results/uav_amplitude_ablation.json || true
echo ""

echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Done."
