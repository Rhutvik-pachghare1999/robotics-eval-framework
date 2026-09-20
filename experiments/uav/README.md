# UAV Fault Diagnostics — Cross-Dataset Evaluation

Cross-dataset fault-detection evaluation on two REAL UAV datasets, with
fully reproducible result manifests validated against
[`schema/result_manifest.schema.json`](../../schema/result_manifest.schema.json).

> **Re-audit notice (2026-09-20):** The 0.929 DP→TII RandomForest macro-F1
> (time_broad) is reported as a **provenance-tracked result under re-audit**.
> The permutation test in `diagnostics/uav_cross_dataset_scrutiny.json` only
> shuffles labels against fixed predictions — it does **not** retrain under the
> null, so it is **not** a valid significance test. A proper permutation test
> (retrain per shuffle) and domain-confound checks are pending. Treat 0.929 as
> provisional until that re-run completes.

---

## Datasets

| Dataset | Class | Role | Description |
|---------|-------|------|-------------|
| **DronePropA** | `REAL-PUBLIC` | Source (train) | REAL experimental flight logs (QDrone + OptiTrack motion capture, Mendeley CC BY 4.0). 127 .mat files (130 nominal, 3 missing — see exclusion manifest). F0=healthy (40), F1/F2/F3=faulty (87). |
| **TII UAV Realistic Fault Dataset** | `REAL-PUBLIC` | Target (test) | REAL PX4 flight logs with broken propellers (GitHub, MIT license). 99 missions: 19 healthy (class 0), 80 faulty (classes 1-4). |

**Transfer type:** `real_to_real` — train on DronePropA (REAL-PUBLIC), test on
TII (REAL-PUBLIC). Both datasets are real experimental data; the cross-dataset
evaluation tests generalization across two different real platforms (QDrone vs
PX4).

### 127 vs 130 file reconciliation

The DronePropA Mendeley record describes 130 flight sequences, but the download
contains 127 `.mat` files. The 3 missing files are all F3 (surface_cut) at
higher severities:

| Missing file | Fault | Severity | Speed | Trajectory |
|--------------|-------|----------|-------|------------|
| `F3_SV2_SP2_t5.mat` | F3 (surface_cut) | SV2 | SP2 | t5 |
| `F3_SV3_SP1_t5.mat` | F3 (surface_cut) | SV3 | SP1 | t5 |
| `F3_SV3_SP2_t3.mat` | F3 (surface_cut) | SV3 | SP2 | t3 |

See [`data/source/dronepropa_exclusion_manifest.json`](../../data/source/dronepropa_exclusion_manifest.json)
for the full exclusion manifest. Actual counts: F0=40, F1=30, F2=30, F3=27.

---

## The 24.41 Hz control-loop artifact

A 24.41 Hz peak initially looked like propeller RPM and tempted us to extract
order-normalized (1P/2P) features. Forensic analysis proved this is a **control
loop rate** (1000/41 Hz), not a physical RPM:

- The peak appears in `motor_CMD` (throttle command) and `gyro_yaw` (yaw rate),
  not just inertial data.
- It is **pegged across all flights** regardless of throttle setting.
- Correlation with throttle is **r = −0.753** — a physical RPM would track
  throttle positively.

**Consequence:** No order-normalized (1P/2P) features were used. The feature
pipeline is sampling-rate-invariant: time-domain (rms, kurtosis, crest_factor,
shape_factor, skew) + broadband spectral (spectral_centroid, spectral_kurtosis,
band_energy_low, band_energy_high) with normalized frequency 0–0.5. This is
critical for cross-dataset transfer — a QDrone control rate does not match the
PX4 control rate, so order-normalized features would not transfer.

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

All spectral features use normalized frequency (0–0.5), so they are
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

`evidence_class: REAL-PUBLIC` (DronePropA is real experimental data).

### Cross-dataset transfer (DronePropA → TII, the main result)

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

**Headline:** Cross-dataset (DronePropA→TII) RF macro-F1 0.929; provenance +
statistical validation under re-audit. The 0.929 is provisional — the permutation
test in `diagnostics/uav_cross_dataset_scrutiny.json` only shuffles labels against
fixed predictions and does **not** retrain under the null, so it is not a valid
significance test. A proper retrain-per-shuffle permutation test is pending.

### Permutation test (provisional — label-shuffle against fixed predictions, NOT a valid null)

| Metric | Value |
|--------|-------|
| True-label macro_f1 | 0.9290 |
| In-domain DP macro_f1 (baseline) | 1.0000 |
| Shuffled-label macro_f1 (mean ± std) | 0.5014 ± 0.0524 |
| Shuffled-label macro_f1 (range) | 0.3963 – 0.6449 |
| p-value (label-shuffle, fixed predictions) | 0.0 (0/100 reached 0.929) |

**Caveat:** This test shuffles labels and re-evaluates the *fixed* RandomForest
predictions without retraining. It shows the model's predictions are not
label-invariant, but it does **not** test whether the model learned fault physics
vs. a domain-confound. A proper permutation test must retrain the model per
shuffle. See `diagnostics/uav_cross_dataset_scrutiny.json` for the full
permutation distribution and feature-level analysis.

---

## CORAL nuance

CORAL (Correlation Alignment, Sun & Saenko 2016) re-colors the source feature
covariance to match the target. The effect is model-dependent:

| Model | Without CORAL | With CORAL | Δ |
|-------|--------------|------------|---|
| logreg (DP→TII) | 0.5909 | 0.7908 | **+0.20** |
| random_forest (DP→TII) | **0.9290** | 0.6647 | **−0.26** |

CORAL helps the linear model (+0.20 macro_f1) but **hurts** the RandomForest
(−0.26 macro_f1). The nonlinear RF's decision boundaries are distorted by the
linear covariance alignment — the fault-relevant feature geometry is disrupted.
This is consistent with CORAL being a linear method applied to a nonlinear
classifier.

---

## Reverse-direction failure

The TII→DP direction (train on TII, test on DronePropA) is near-chance:

| Level | Model | macro_f1 | accuracy |
|-------|-------|----------|-----------|
| time_broad | logreg | 0.2476 | 0.3150 |
| time_broad | random_forest | 0.4869 | 0.4961 |

This is expected for cross-dataset transfer: the target dataset (TII, 99
missions) is smaller and doesn't cover the source dataset's fault modes. The
asymmetry (DP→TII works, TII→DP doesn't) is consistent with the source dataset
having broader fault coverage (127 flights, 4 fault types) than the target
(99 missions, binary healthy/faulty).

---

## `evidence_class` honesty note

All manifests use `evidence_class: REAL-PUBLIC`. Both DronePropA and TII are
real experimental datasets (QDrone + OptiTrack and PX4 flight logs
respectively). The cross-dataset manifests also record
`source_dataset_class: REAL-PUBLIC`, `target_dataset_class: REAL-PUBLIC`, and
`transfer_type: real_to_real` as explicit provenance.

---

## Reproduce

### Prerequisites

- ASU SOL SLURM cluster (or local Linux with the datasets)
- DronePropA dataset: `/scratch/.../dronepropa/DronePropA Motion Trajectories Dataset/` (127 .mat files; 130 nominal, 3 missing — see exclusion manifest)
- TII UAV Realistic Fault Dataset: `/scratch/.../tii/repo/Dataset/` (99 missions)
- Python venv with scikit-learn, scipy, numpy, h5py

### Within-dataset baselines (DronePropA)

```bash
# On SOL:
sbatch slurm/uav_fault_clf.sbatch          # baseline (LogReg, RF)
sbatch slurm/uav_fault_clf_strong.sbatch   # stronger (HGB, 1D-CNN)
```

### Cross-dataset transfer

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

### Scrutiny (provisional permutation test — label-shuffle only, NOT a valid null)

```bash
sbatch slurm/uav_cross_dataset_scrutiny.sbatch
```

### Validate manifests

```bash
python scripts/validate_results.py
# All 6 result manifests pass:
#   OK  edge_al_coco_tier1b_v2.json
#   OK  uav_fault_clf_baseline.json          (REAL-PUBLIC)
#   OK  uav_fault_clf_strong.json            (REAL-PUBLIC)
#   OK  uav_cross_dataset_naive.json         (REAL-PUBLIC, real_to_real)
#   OK  uav_cross_dataset_time_broad.json    (REAL-PUBLIC, real_to_real)
#   OK  uav_cross_dataset_coral.json         (REAL-PUBLIC, real_to_real)
```

---

## Pending re-audit

The following are pending before the 0.929 result can be considered validated:

1. **Proper permutation test**: retrain the model per shuffle (not just shuffle
   labels against fixed predictions). The current test in
   `diagnostics/uav_cross_dataset_scrutiny.json` is label-shuffle-only.
2. **Domain-confound checks**: verify the 0.929 is not driven by a domain
   artifact (e.g., sample rate, flight duration, drone identity) that leaks
   across the train/test split.
3. **Frozen test set + held-out flights**: the current cross-dataset split uses
   all of TII as the test set; a held-out flight-level split would be stronger.
4. **Feature importance audit**: RF feature importances are mixed (fault physics
   + domain markers); need to disentangle which features drive the 0.929.

---

## Files

| File | Role |
|------|------|
| `run_fault_clf.py` | Within-dataset baselines (LogReg, RF) |
| `run_fault_clf_strong.py` | Stronger models (HGB, 1D-CNN) |
| `run_cross_dataset.py` | Cross-dataset transfer (3 levels, both directions) |
| `scrutinize_cross_dataset.py` | Provisional permutation test + feature analysis |
| `diagnose_f_rot.py` | f_rot frequency diagnosis (24.41 Hz peak) |
| `diagnose_motor_esc.py` | Motor/ESC telemetry diagnosis (control loop rate) |
| `validate_f_rot.py` | f_rot validation across datasets |
| `../../adapters/dronepropa.py` | DronePropA adapter (REAL-PUBLIC) |
| `../../adapters/tii.py` | TII adapter (REAL-PUBLIC) |
| `../../data/source/dronepropa_exclusion_manifest.json` | 3 missing F3 files (127 vs 130) |
| `../../schema/result_manifest.schema.json` | Result manifest schema |
| `../../scripts/validate_results.py` | Manifest validator |
