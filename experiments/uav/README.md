# UAV Fault Diagnostics — Cross-Dataset Evaluation

Cross-dataset fault-detection evaluation on two REAL UAV datasets, with
fully reproducible result manifests validated against
[`schema/result_manifest.schema.json`](../../schema/result_manifest.schema.json).

> **Re-audit complete (2026-09-20) — final verdict: cross-dataset transfer NOT
> ESTABLISHED.** The model learns **real fault structure on DronePropA**
> (permutation test p=0.003, 1000 perms, retrain under null; cross-trajectory
> holdout 0.73) and generalizes **across trajectories within DronePropA**, but
> **cross-DATASET transfer to TII is not cleanly established**: a domain
> classifier separates DP from TII with **macro-F1 = 1.0** even on shape-only
> features (the datasets are 100% separable), cross-dataset healthy-recall on
> TII is **9/19 (near chance)**, and cross-speed holdout within DronePropA
> collapses to **0.46 (chance)**. The naive 0.929 was **amplitude-inflated**
> (~40% from rms + absolute band energy, partly cross-domain signal-scale
> differences); the 0.74 shape-only score **remains domain-confounded**. We
> report cross-dataset fault transfer as **NOT ESTABLISHED** on this data.
>
> **Contribution (reframed):** the valuable result is the **methodology** — a
> rigorous adversarial-validation pipeline (amplitude ablation, correct
> permutation test with retrain-under-null, bootstrap CI, domain classifier,
> within-dataset holdouts) that **detects domain confound in cross-dataset
> fault diagnosis**. The pipeline caught that the 0.929 was amplitude-inflated
> and that the 0.74 is domain-confounded (domain classifier macro-F1 = 1.0).

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

**Headline (honest, re-audited):** Cross-dataset (DronePropA→TII,
real→real) fault detection is **NOT ESTABLISHED** on this data. The model
learns **real fault structure on DronePropA** (permutation test p=0.003,
1000 perms, retrain under null; cross-trajectory holdout 0.73) but
**cross-DATASET transfer to TII is confounded**: a domain classifier
separates DP from TII with **macro-F1 = 1.0** even on shape-only features
(the datasets are 100% separable), cross-dataset healthy-recall on TII is
**9/19 (near chance)**, and cross-speed holdout within DronePropA collapses
to **0.46 (chance)**. The naive 0.929 was **amplitude-inflated** (~40% from
rms + absolute band energy, partly cross-domain signal-scale differences);
the 0.74 shape-only score **remains domain-confounded**. The contribution is
the **methodology** — a rigorous adversarial-validation pipeline that
detects domain confound in cross-dataset fault diagnosis.

| Level | Direction | Model | macro_f1 | accuracy | AUC | Note | Manifest |
|-------|-----------|-------|----------|----------|-----|------|----------|
| **naive** (18 feat) | DP→TII | logreg | 0.1610 | 0.1919 | 0.00 | | `uav_cross_dataset_naive.json` |
| **naive** (18 feat) | DP→TII | random_forest | 0.5943 | 0.6162 | 1.00 | | `uav_cross_dataset_naive.json` |
| **naive** (18 feat) | TII→DP | logreg | 0.2395 | 0.3150 | 0.35 | | `uav_cross_dataset_naive.json` |
| **naive** (18 feat) | TII→DP | random_forest | 0.4295 | 0.6850 | 0.43 | | `uav_cross_dataset_naive.json` |
| **time_broad** (54 feat) | DP→TII | logreg | 0.5909 | 0.8384 | 1.00 | | `uav_cross_dataset_time_broad.json` |
| **time_broad** (54 feat) | **DP→TII** | **random_forest** | **0.9290** | **0.9596** | **1.00** | ⚠ **amplitude-inflated** | `uav_cross_dataset_time_broad.json` |
| **time_broad** (42 feat, shape-only) | **DP→TII** | **random_forest** | **0.7391** | **0.8586** | **0.93** | ✅ **honest (amplitude-ablated)** | `uav_amplitude_ablation.json` |
| **time_broad** (54 feat) | TII→DP | logreg | 0.2476 | 0.3150 | 0.60 | | `uav_cross_dataset_time_broad.json` |
| **time_broad** (54 feat) | TII→DP | random_forest | 0.4869 | 0.4961 | 0.58 | | `uav_cross_dataset_time_broad.json` |
| time_broad+coral (54+align) | DP→TII | logreg | 0.7908 | 0.8485 | 0.90 | | `uav_cross_dataset_coral.json` |
| time_broad+coral (54+align) | DP→TII | random_forest | 0.6647 | 0.7879 | 0.84 | | `uav_cross_dataset_coral.json` |
| time_broad+coral (54+align) | TII→DP | logreg | 0.5170 | 0.7087 | 0.57 | | `uav_cross_dataset_coral.json` |
| time_broad+coral (54+align) | TII→DP | random_forest | 0.4065 | 0.6850 | 0.61 | | `uav_cross_dataset_coral.json` |

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

### Amplitude ablation (shape-only features)

```bash
# On SOL (CPU-only, no GPU needed):
sbatch slurm/uav_amplitude_permutation.sbatch
# Or locally:
python experiments/uav/run_amplitude_ablation.py \
    --source-dir-dronepropa "$DRONEPROPA_SOURCE_DIR" \
    --source-dir-tii "$TII_SOURCE_DIR" \
    --seed 42 \
    --out results/uav_amplitude_ablation.json
```

### Correct permutation test (shuffle source labels, retrain under null)

```bash
# On SOL (CPU-only, ~8 min for 1000 perms):
sbatch slurm/uav_amplitude_permutation.sbatch
# Or locally:
python experiments/uav/run_permutation_test.py \
    --source-dir-dronepropa "$DRONEPROPA_SOURCE_DIR" \
    --source-dir-tii "$TII_SOURCE_DIR" \
    --n-permutations 1000 \
    --seed 42 \
    --out diagnostics/uav_permutation_test.json
```

### Scrutiny (old, label-shuffle only — NOT a valid null)

```bash
sbatch slurm/uav_cross_dataset_scrutiny.sbatch
```

### Validate manifests

```bash
python scripts/validate_results.py
# All 7 result manifests pass:
#   OK  edge_al_coco_tier1b_v2.json
#   OK  uav_fault_clf_baseline.json          (REAL-PUBLIC)
#   OK  uav_fault_clf_strong.json            (REAL-PUBLIC)
#   OK  uav_cross_dataset_naive.json         (REAL-PUBLIC, real_to_real)
#   OK  uav_cross_dataset_time_broad.json    (REAL-PUBLIC, real_to_real)
#   OK  uav_cross_dataset_coral.json         (REAL-PUBLIC, real_to_real)
#   OK  uav_amplitude_ablation.json          (REAL-PUBLIC, real_to_real, shape-only)
```

### Amplitude ablation (shape-only features)

The 0.929 all-feature result was re-audited for amplitude confounding. An
amplitude ablation rebuilds the feature pipeline with ONLY shape/normalized
features — kurtosis, crest factor, shape factor, skew (time domain) and
spectral centroid, spectral kurtosis, band-energy *ratio* (spectral domain) —
dropping rms and absolute band energy (which carry cross-domain signal-scale
differences).

| Feature set | DP→TII RF macro_f1 | Δ vs all-feature |
|-------------|--------------------|-----------------|
| all features (54, time_broad) | 0.9290 | — |
| **shape-only (42, amplitude-ablated)** | **0.7391** | **−0.1899** |

**Verdict:** ~40% of the 0.929 was amplitude/magnitude signal (rms, absolute
band energy) that is partly confounded with cross-domain scale differences
(DronePropA QDrone vs TII PX4 have different signal magnitudes). The honest
transfer estimate is **0.74** (shape-only, amplitude-ablated). Manifest:
`results/uav_amplitude_ablation.json`.

### Correct permutation test (shuffle source labels, retrain under null)

The old permutation test in `diagnostics/uav_cross_dataset_scrutiny.json`
shuffled TII test labels and re-evaluated *fixed* predictions — it did **not**
retrain under the null, so it was not a valid significance test. The correct
test shuffles **source** (DronePropA) labels, **retrains** the RF per shuffle,
and evaluates on the fixed, unshuffled TII test set.

| Metric | Value |
|--------|-------|
| True-label DP→TII RF macro_f1 | 0.9290 |
| In-domain DP→DP RF macro_f1 | 1.0000 |
| Null (shuffled source labels) mean ± std | 0.4540 ± 0.0479 |
| Null min / max | 0.4310 / 1.0000 |
| Null p25 / median / p75 / p95 | 0.4469 / 0.4469 / 0.4469 / 0.4469 |
| k null ≥ true | 2 / 1000 |
| **p-value** (k+1)/(n+1) | **0.002997** |
| z-score | 9.91σ |
| Verdict | `real_fault_transfer` |

**Verdict:** 0.9290 is statistically significant (p=0.003, z=9.91σ). The null
collapses to chance (~0.45). The 2/1000 nulls that reached 1.0 are random label
permutations that happened to align — expected under H₀ with 1000 trials.
Diagnostic: `diagnostics/uav_permutation_test.json` (machine-readable:
`p_value_ge`, `null_mean_f1`, `null_std_f1`, `k_null_ge_true`).

---

## Re-audit complete — final verdict

The adversarial re-audit is complete. The five checks (amplitude ablation,
correct permutation test, bootstrap CI, domain classifier, within-dataset
holdouts) converge on a single honest conclusion: **cross-dataset fault
transfer from DronePropA to TII is NOT cleanly established on this data.**

### The numbers

| Check | Metric | Value | Interpretation |
|-------|--------|-------|----------------|
| Naive all-feature RF | DP→TII macro-F1 | 0.9290 | ⚠ amplitude-inflated (~40% from rms + absolute band energy) |
| Shape-only RF (amplitude-ablated) | DP→TII macro-F1 | 0.7391 | ✅ honest point estimate, but domain-confounded |
| Bootstrap 95% CI (shape-only) | macro-F1 | [0.6005, 0.8439] | 0.929 is outside the CI → amplitude-inflated |
| Permutation test (1000 perms, retrain under null) | p-value | 0.003 | fault signal is real (null mean 0.454 ± 0.048, z=9.91σ) |
| **Domain classifier (shape-only)** | DP-vs-TII macro-F1 | **1.0000** | ⚠ **datasets are 100% separable → confound** |
| Cross-trajectory holdout (within DP, shape-only) | macro-F1 | 0.7279 ± 0.0584 | ✅ generalizes across trajectories within DP |
| Cross-speed holdout (within DP, shape-only) | macro-F1 | 0.4642 ± 0.0748 | ⚠ collapses to chance across speeds |
| Cross-dataset healthy recall on TII | 9 / 19 | 0.47 | ⚠ near chance on the minority class |

### Why the 0.74 is NOT clean cross-dataset transfer

1. **Domain separability (decisive):** a RandomForest on shape-only features
   separates DronePropA from TII with **macro-F1 = 1.0** (confusion matrix
   [[127, 0], [0, 99]], zero errors). The two datasets are linearly separable
   in feature space — any cross-dataset classifier can exploit domain identity
   (QDrone vs PX4, indoor OptiTrack vs outdoor, different signal scales) as a
   proxy for the label. The 0.74 macro-F1 is partly domain identity, not fault
   physics.
2. **Healthy recall on TII is near chance:** the shape-only RF gets 9/19
   healthy missions correct on TII (47% recall) — the macro-F1 of 0.74 is
   carried by faulty recall (76/80 = 95%), not by distinguishing healthy from
   faulty. The model mostly predicts "faulty" on TII.
3. **Cross-speed holdout collapses to chance:** within DronePropA, leaving one
   speed out drops macro-F1 to 0.46 — the model does not generalize across
   speeds even within the source dataset. Cross-dataset transfer (which also
   crosses speed distributions) is therefore unlikely to be clean.
4. **Cross-trajectory holdout is the one positive:** 0.73 within DronePropA
   shows the model does learn real fault structure that generalizes across
   trajectories (same drones, same speeds). This is the honest positive
   result: the permutation test (p=0.003) and cross-trajectory holdout (0.73)
   together show the model learns real fault signatures — but they do not
   transfer cleanly across datasets.

### What IS established

- **Real fault structure on DronePropA:** permutation test p=0.003 (1000
  perms, retrain under null, null mean 0.454 ± 0.048, z=9.91σ). The model is
  not memorizing label noise — the fault signal is statistically real.
- **Cross-trajectory generalization within DronePropA:** 0.7279 ± 0.0584
  macro-F1 (leave-one-trajectory-out, 5 folds). The model generalizes across
  trajectories within the source dataset.
- **The 0.929 was amplitude-inflated:** amplitude ablation drops it to 0.74,
  and the 0.929 is outside the shape-only bootstrap 95% CI [0.60, 0.84]. ~40%
  of the 0.929 was rms + absolute band energy, partly confounded with
  cross-domain signal-scale differences.

### What is NOT established

- **Cross-dataset fault transfer (DP→TII):** NOT ESTABLISHED. The 0.74
  shape-only score is domain-confounded (domain classifier macro-F1 = 1.0,
  healthy recall 9/19, cross-speed 0.46). We cannot attribute the 0.74 to
  fault physics that transfers across platforms — it may be domain identity.
- **Cross-speed generalization:** NOT ESTABLISHED. Within DronePropA,
  leave-one-speed-out drops to 0.46 (chance).
- **Cross-drone generalization:** NOT ESTABLISHED. Leave-one-drone-out could
  not be run (single-class test sets: D1 has 30 healthy-only, D2/D3 have 5
  faulty-only).

### Contribution (reframed)

The contribution is the **methodology**, not a cross-dataset transfer
number. This is a rigorous adversarial-validation pipeline that **detects
domain confound in cross-dataset fault diagnosis**:

1. **Amplitude ablation** — drops amplitude features (rms, absolute band
   energy) that confound cross-domain signal scale; reveals the 0.929 was
   ~40% amplitude-inflated.
2. **Correct permutation test** — shuffles **source** labels and **retrains
   the RF per shuffle** (the old label-shuffle test did not retrain under the
   null and was invalid); confirms the fault signal is real (p=0.003).
3. **Bootstrap 95% CI** — 2000 resamples over the 99 TII missions; the 0.929
   is outside [0.60, 0.84], confirming amplitude inflation.
4. **Domain classifier** — trains a DP-vs-TII classifier on the same
   features; macro-F1 = 1.0 means the datasets are 100% separable and any
   cross-dataset score is domain-confounded.
5. **Within-dataset holdouts** — cross-trajectory (0.73, positive),
   cross-speed (0.46, chance), cross-drone (skipped, single-class test
   sets); shows the fault signal is real but does not cross speeds or drones.

This pipeline is the reusable artifact: it turns a 0.929 "cross-dataset
transfer" claim into an honest "NOT ESTABLISHED, here is why" verdict with
five converging checks. The same pipeline applies to any cross-dataset fault
diagnosis claim.

---

## Files

| File | Role |
|------|------|
| `run_fault_clf.py` | Within-dataset baselines (LogReg, RF) |
| `run_fault_clf_strong.py` | Stronger models (HGB, 1D-CNN) |
| `run_cross_dataset.py` | Cross-dataset transfer (3 levels, both directions) |
| `run_amplitude_ablation.py` | Amplitude ablation (shape-only features, 42 features) |
| `run_permutation_test.py` | Correct permutation test (shuffle source labels, retrain under null) |
| `scrutinize_cross_dataset.py` | Old permutation test (label-shuffle only, NOT a valid null) |
| `diagnose_f_rot.py` | f_rot frequency diagnosis (24.41 Hz peak) |
| `diagnose_motor_esc.py` | Motor/ESC telemetry diagnosis (control loop rate) |
| `validate_f_rot.py` | f_rot validation across datasets |
| `../../adapters/dronepropa.py` | DronePropA adapter (REAL-PUBLIC) |
| `../../adapters/tii.py` | TII adapter (REAL-PUBLIC) |
| `../../data/source/dronepropa_exclusion_manifest.json` | 3 missing F3 files (127 vs 130) |
| `../../schema/result_manifest.schema.json` | Result manifest schema |
| `../../scripts/validate_results.py` | Manifest validator |
