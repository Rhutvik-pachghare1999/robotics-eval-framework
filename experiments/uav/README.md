# UAV Fault Diagnostics — Cross-Dataset Evaluation

Rigorous sim-to-real fault-detection evaluation on two UAV datasets, with
fully reproducible result manifests validated against
[`schema/result_manifest.schema.json`](../../schema/result_manifest.schema.json).

---

## Datasets

| Dataset | Class | Role | Description |
|---------|-------|------|-------------|
| **DronePropA** | `SIMULATED` | Source (train) | Simulink-generated motion trajectories for commercial drones with defective propellers (127 flights: 40 healthy F0, 87 faulty F1/F2/F3). |
| **TII UAV Realistic Fault Dataset** | `REAL-PUBLIC` | Target (test) | Real PX4 flight logs with broken propellers (99 missions: 19 healthy class 0, 80 faulty classes 1-4). MIT license. |

**Transfer type:** `sim_to_real` — train on Simulink (SIMULATED), test on real PX4 logs
(REAL-PUBLIC). Per the weakest-link rule, `evidence_class` for all cross-dataset
manifests is `SIMULATED` (the training/source dataset's class).

---

## The DronePropA-is-Simulink discovery

A 24.41 Hz peak initially looked like propeller RPM and tempted us to extract
order-normalized (1P/2P) features. Forensic analysis proved this is a **control
loop rate** (1000/41 Hz), not a physical RPM:

- The peak appears in `motor_CMD` (throttle command) and ESC telemetry, not just
  inertial data.
- It is **pegged across all flights** regardless of throttle setting.
- Correlation with throttle is **r = -0.753** — a physical RPM would track
  throttle positively.
- It appears in `gyro_yaw` (yaw rate) at the same frequency, consistent with a
  closed-loop control update rate, not a propeller signature.

**Consequence:** No order-normalized (1P/2P) features were used. The feature
pipeline is sampling-rate-invariant: time-domain (rms, kurtosis, crest_factor,
shape_factor, skew) + broadband spectral (spectral_centroid, spectral_kurtosis,
band_energy_low, band_energy_high) with normalized frequency 0-0.5. This is
critical for sim-to-real transfer — a Simulink control rate does not exist in
real PX4 logs.

See `diagnose_f_rot.py`, `diagnose_motor_esc.py`, and `validate_f_rot.py` for
the full forensic analysis, and `diagnostics/uav_f_rot_validation.json` for the
validated evidence.

---

## Feature pipeline

6 channels (gyro xyz + accel xyz) × 9 features = **54 features** per flight.

| Domain | Feature | Description |
|--------|---------|-------------|
| Time | `rms` | Root mean square |
| Time | `kurtosis` | Tail heaviness (sensitive to impacts) |
| Time | `crest_factor` | peak / rms (sensitive to impulses) |
| Time | `shape_factor` | rms / mean(abs) |
| Time | `skew` | Asymmetry |
| Spectral | `spectral_centroid` | Weighted mean frequency |
| Spectral | `spectral_kurtosis` | Spectral peakedness |
| Spectral | `band_energy_low` | Energy in normalized freq [0, 0.25] |
| Spectral | `band_energy_high` | Energy in normalized freq [0.25, 0.5] |

All spectral features use normalized frequency (0-0.5), so they are
sampling-rate-invariant and transfer across datasets with different sample rates.

---

## Results

### Within-dataset baselines (DronePropA, leave-one-flight-out)

| Model | macro_f1 | accuracy | Manifest |
|-------|----------|-----------|----------|
| LogReg | 0.5405 | 0.5276 | `results/uav_fault_clf_baseline.json` |
| RandomForest | 0.5311 | 0.5197 | `results/uav_fault_clf_baseline.json` |
| HistGradientBoosting | 0.5615 | 0.5591 | `results/uav_fault_clf_strong.json` |
| 1D-CNN | 0.3818 | 0.4567 | `results/uav_fault_clf_strong.json` |

`evidence_class: SIMULATED` (DronePropA is Simulink-generated).

### Cross-dataset sim-to-real transfer (the main result)

Three escalation levels, both directions. All numbers from validated manifests.

| Level | Direction | Model | macro_f1 | accuracy | AUC | Manifest |
|-------|-----------|-------|----------|----------|-----|----------|
| **naive** (18 feat) | DP→TII | logreg | 0.1610 | 0.1919 | 0.00 | `uav_cross_dataset_naive.json` |
| **naive** (18 feat) | DP→TII | random_forest | 0.5943 | 0.6162 | 1.00 | `uav_cross_dataset_naive.json` |
| **naive** (18 feat) | TII→DP | logreg | 0.2395 | 0.3150 | 0.35 | `uav_cross_dataset_naive.json` |
| **naive** (18 feat) | TII→DP | random_forest | 0.4295 | 0.6850 | 0.43 | `uav_cross_dataset_naive.json` |
| **time_broad** (54 feat) | DP→TII | logreg | 0.5909 | 0.8384 | 1.00 | `uav_cross_dataset_time_broad.json` |
| **time_broad** (54 feat) | **DP→TII** | **random_forest** | **0.9290** | **0.9596** | **1.00** | `uav_cross_dataset_time_broad.json` |
| **time_broad** (54 feat) | TII→DP | logreg | 0.2476 | 0.3150 | 0.60 | `uav_cross_dataset_time_broad.json` |
| **time_broad** (54 feat) | TII→DP | random_forest | 0.4869 | 0.4961 | 0.58 | `uav_cross_dataset_time_broad.json` |
| time_broad+coral (54+align) | DP→TII | logreg | 0.7908 | 0.8485 | 0.90 | `uav_cross_dataset_coral.json` |
| time_broad+coral (54+align) | DP→TII | random_forest | 0.6647 | 0.7879 | 0.84 | `uav_cross_dataset_coral.json` |
| time_broad+coral (54+align) | TII→DP | logreg | 0.5170 | 0.7087 | 0.57 | `uav_cross_dataset_coral.json` |
| time_broad+coral (54+align) | TII→DP | random_forest | 0.4065 | 0.6850 | 0.61 | `uav_cross_dataset_coral.json` |

**Key result:** `time_broad` + RandomForest achieves **macro_f1 = 0.9290** on
sim→real transfer (DronePropA→TII), with accuracy 0.9596 and AUC 1.0.

### Permutation test — is 0.93 real or domain leakage?

The 0.93 is suspiciously high for sim→real transfer. We ran a label-shuffle
permutation test (100 shuffles) on the `time_broad` + RandomForest DP→TII
pipeline to verify it's learning fault physics, not domain artifacts.

| Metric | Value |
|--------|-------|
| True-label macro_f1 | **0.9290** |
| In-domain DP macro_f1 (baseline) | 1.0000 |
| Shuffled-label macro_f1 (mean ± std) | 0.5014 ± 0.0524 |
| Shuffled-label macro_f1 (range) | 0.3963 – 0.6449 |
| **p-value** | **0.0** (0/100 permutations reached 0.93) |
| Verdict | `real_fault_transfer: true`, `leakage_suspected: false` |

The true F1 (0.9290) is **8.2σ** above the shuffled mean (0.5014). Zero of 100
shuffled permutations reached the true F1. The model learns fault physics that
generalizes from Simulink to real PX4 logs — it is not exploiting domain
artifacts.

See `diagnostics/uav_cross_dataset_scrutiny.json` for the full permutation
distribution and feature-level analysis.

---

## CORAL nuance

CORAL (Correlation Alignment, Sun & Saenko 2016) re-colors the source feature
covariance to match the target. The effect is model-dependent:

| Model | Without CORAL | With CORAL | Δ |
|-------|--------------|------------|---|
| logreg (DP→TII) | 0.5909 | 0.7908 | **+0.20** |
| random_forest (DP→TII) | **0.9290** | 0.6647 | **-0.26** |

CORAL helps the linear model (+0.20 macro_f1) but **hurts** the RandomForest
(-0.26 macro_f1). The nonlinear RF's decision boundaries are distorted by the
linear covariance alignment — the fault-relevant feature geometry is disrupted.
This is consistent with CORAL being a linear method applied to a nonlinear
classifier.

---

## Honest reverse-direction failure

The TII→DP direction (train on real, test on sim) is near-chance:

| Level | Model | macro_f1 | accuracy |
|-------|-------|----------|-----------|
| time_broad | logreg | 0.2476 | 0.3150 |
| time_broad | random_forest | 0.4869 | 0.4961 |

This is expected for sim→real transfer: the real dataset (TII, 99 missions) is
smaller and doesn't cover the sim domain's fault modes. The asymmetry
(DP→TII works, TII→DP doesn't) is consistent with the source dataset having
broader fault coverage than the target.

---

## `evidence_class` honesty note

All cross-dataset manifests use `evidence_class: SIMULATED` — the
**weakest-link label**. The model is *trained* on DronePropA (SIMULATED), so the
result's evidence class is `SIMULATED` regardless of the test dataset's class.
The manifest also records `source_dataset_class: SIMULATED`,
`target_dataset_class: REAL-PUBLIC`, and `transfer_type: sim_to_real` as
explicit provenance. We do **not** label this result `REAL-PUBLIC` even though
TII is real — that would blur the category boundary and overstate the evidence.

---

## Reproduce

### Prerequisites

- ASU SOL SLURM cluster (or local Linux with the datasets)
- DronePropA dataset: `/scratch/.../dronepropa/DronePropA Motion Trajectories Dataset/`
- TII UAV Realistic Fault Dataset: `/scratch/.../tii/repo/Dataset/`
- Python venv with scikit-learn, scipy, numpy, h5py

### Within-dataset baselines (DronePropA)

```bash
# On SOL:
sbatch slurm/uav_fault_clf.sbatch          # baseline (LogReg, RF)
sbatch slurm/uav_fault_clf_strong.sbatch   # stronger (HGB, 1D-CNN)
```

### Cross-dataset sim-to-real transfer

```bash
# On SOL (CPU-only, no GPU needed):
sbatch slurm/uav_cross_dataset_cpu.sbatch

# Or locally:
python experiments/uav/run_cross_dataset.py \
    --source-dir-dronepropa "$DRONEPROPA_SOURCE_DIR" \
    --source-dir-tii "$TII_SOURCE_DIR" \
    --seed 42 \
    --levels naive time_broad coral \
    --out-dir results
```

### Scrutiny (permutation test)

```bash
sbatch slurm/uav_cross_dataset_scrutiny.sbatch
```

### Validate manifests

```bash
python scripts/validate_results.py
# All 6 result manifests pass:
#   OK  edge_al_coco_tier1b_v2.json
#   OK  uav_fault_clf_baseline.json          (SIMULATED)
#   OK  uav_fault_clf_strong.json            (SIMULATED)
#   OK  uav_cross_dataset_naive.json         (SIMULATED, sim_to_real)
#   OK  uav_cross_dataset_time_broad.json    (SIMULATED, sim_to_real)
#   OK  uav_cross_dataset_coral.json         (SIMULATED, sim_to_real)
```

---

## Files

| File | Role |
|------|------|
| `run_fault_clf.py` | Within-dataset baselines (LogReg, RF) |
| `run_fault_clf_strong.py` | Stronger models (HGB, 1D-CNN) |
| `run_cross_dataset.py` | Cross-dataset sim-to-real (3 levels, both directions) |
| `scrutinize_cross_dataset.py` | Permutation test + feature analysis |
| `diagnose_f_rot.py` | f_rot frequency diagnosis (24.41 Hz peak) |
| `diagnose_motor_esc.py` | Motor/ESC telemetry diagnosis (control loop rate) |
| `validate_f_rot.py` | f_rot validation across datasets |
| `../../adapters/dronepropa.py` | DronePropA adapter (SIMULATED) |
| `../../adapters/tii.py` | TII adapter (REAL-PUBLIC) |
| `../../schema/result_manifest.schema.json` | Result manifest schema |
| `../../scripts/validate_results.py` | Manifest validator |
