#!/bin/bash
#SBATCH --job-name=uav-holdouts
#SBATCH --partition=htc
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Re-audit holdouts on the SHAPE-ONLY feature set (the honest 0.74, not the
# amplitude-inflated 0.929).
#
# Experiment 3: Frozen + bootstrap CI. Freeze the shape-only feature/model
#   choice, report TII macro-F1 ONCE with a bootstrap 95% CI over the 99
#   missions (2000 resamples).
# Experiment 4: Domain classifier. Train a DP-vs-TII classifier on shape-only
#   features; report how separable the domains are (confound risk).
# Experiment 5: Holdouts within DronePropA. Cross-trajectory (train t1-4 /
#   test t5, rotate), cross-speed (leave-one-speed-out), cross-drone
#   (leave-one-drone-out). Report macro-F1 each.
#
# All use the shape-only feature set (42 features: kurtosis, crest_factor,
# shape_factor, skew, spectral_centroid, spectral_kurtosis, band_energy_ratio).
# NO rms, NO absolute band energy (amplitude-ablated).
#
# Output:
#   results/uav_bootstrap_ci.json         (manifest, REAL-PUBLIC, real_to_real)
#   results/uav_domain_classifier.json    (manifest, REAL-PUBLIC, real_to_real)
#   results/uav_holdouts.json             (manifest, REAL-PUBLIC, real_to_real)

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

# ─── Experiments 3, 4, 5 (shape-only features, the honest 0.74) ─────────────
echo "============================================================"
echo "  Experiments 3-5: Bootstrap CI, domain classifier, holdouts"
echo "  Feature set: shape-only (42 features, amplitude-ablated)"
echo "  DROPS: rms, absolute band energy"
echo "  KEEPS: kurtosis, crest_factor, shape_factor, skew,"
echo "         spectral_centroid, spectral_kurtosis, band_energy_ratio"
echo "============================================================"
python experiments/uav/run_reaudit_holdouts.py \
    --source-dir-dronepropa "$DRONEPROPA_SOURCE_DIR" \
    --source-dir-tii "$TII_SOURCE_DIR" \
    --n-bootstrap 2000 \
    --seed 42 \
    --out-dir results
echo ""
echo "Experiments 3-5 complete."
echo ""

# ─── Validate manifests ─────────────────────────────────────────────────────
echo "============================================================"
echo "  Validating manifests"
echo "============================================================"
python scripts/validate_results.py results/uav_bootstrap_ci.json results/uav_domain_classifier.json results/uav_holdouts.json || true
echo ""

echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Done."
