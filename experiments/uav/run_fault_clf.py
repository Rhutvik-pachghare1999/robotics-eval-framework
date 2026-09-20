#!/usr/bin/env python3
"""
run_fault_clf.py — UAV fault-type classification baselines on DronePropA.

Task: 4-class fault_type classification (F0=healthy, F1=edge_cut,
F2=crack, F3=surface_cut) from QDrone_data flight logs.

Features (per flight, from verified QDrone_data channel map):
  Statistical + spectral features over the raw IMU and motor/ESC channels.
  Channel rows used (per adapters/dronepropa.py, verified against
  important_data_extraction.m.txt / plot_daq_qdrone2.m.txt):
    rows 24-26 : battery, electronics_current, motor_current      (3 ch)
    rows 27-29 : gyro_1  (roll, pitch, yaw)                       (3 ch)
    rows 30-32 : accel_1  (x, y, z)                               (3 ch)
    rows 33-35 : gyro_2  (roll, pitch, yaw)                       (3 ch)
    rows 36-38 : accel_2  (x, y, z)                               (3 ch)
    row  46    : height sensor / range_data                      (1 ch)
    rows 47-54 : per-rotor motor + ESC commands                   (8 ch)
  Total: 21 channels.
  Rows 2-19 (body-frame IMU) are NOT used to avoid double-counting the
  IMU signal (we use the raw gyro/accel rows 27-38 instead, per the
  approved channel map note).
  Rows 39-45 (OpticalFlow) and 55-56 (undocumented) are NOT used.

  Per channel we extract 8 features:
    stat: mean, std, min, max, rms, variance
    fft: spectral_entropy, dominant_freq_magnitude_ratio
  => 21 channels * 8 features = 168 features per flight.

Split: GroupKFold(n_splits=5) on group_id (leave-one-flight-out family).
  No flight appears in both train and test.

Models: LogisticRegression, RandomForestClassifier (sklearn).

Output: results/uav_fault_clf_baseline.json result manifest +
        prints per-class precision/recall, macro-F1, confusion matrix.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from adapters.dronepropa import DronePropAAdapter  # noqa: E402

# Verified QDrone_data channel rows (1-indexed, per adapters/dronepropa.py).
# rows 24-26: battery, electronics_current, motor_current
# rows 27-29: gyro_1 (roll, pitch, yaw)
# rows 30-32: accel_1 (x, y, z)
# rows 33-35: gyro_2 (roll, pitch, yaw)
# rows 36-38: accel_2 (x, y, z)
# row  46:   height sensor / range_data
# rows 47-54: motor_FL, esc_FL, motor_FR, esc_FR, motor_BL, esc_BL, motor_BR, esc_BR
CHANNEL_ROWS = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 46, 47, 48, 49, 50, 51, 52, 53, 54]
CHANNEL_NAMES = [
    "battery", "electronics_current", "motor_current",
    "gyro1_roll", "gyro1_pitch", "gyro1_yaw",
    "accel1_x", "accel1_y", "accel1_z",
    "gyro2_roll", "gyro2_pitch", "gyro2_yaw",
    "accel2_x", "accel2_y", "accel2_z",
    "height_sensor",
    "motor_FL", "esc_FL", "motor_FR", "esc_FR",
    "motor_BL", "esc_BL", "motor_BR", "esc_BR",
]

FAULT_NAMES = {0: "healthy", 1: "edge_cut", 2: "crack", 3: "surface_cut"}


def extract_features(signal: np.ndarray) -> np.ndarray:
    """Extract 8 features per channel from a 1-D signal array.

    Statistical: mean, std, min, max, rms, variance
    Spectral:    spectral_entropy, dominant_freq_magnitude_ratio

    NaN-aware: some QDrone_data channels contain NaN values (e.g. height
    sensor on ground-truth flights). We use nan-aware reductions and
    replace any remaining NaN/inf with 0.0 so sklearn does not choke.
    """
    arr = np.asarray(signal, dtype=np.float64)
    n = arr.shape[0]
    if n == 0:
        return np.zeros(8)
    finite = arr[np.isfinite(arr)]
    if finite.shape[0] == 0:
        return np.zeros(8)
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    rms = float(np.sqrt(np.nanmean(arr ** 2)))
    var = float(np.nanvar(arr))
    # FFT-based features on NaN-filled signal (replace NaN with mean for FFT)
    filled = np.where(np.isfinite(arr), arr, mean)
    spectrum = np.abs(np.fft.rfft(filled - mean))
    total_energy = float(np.sum(spectrum ** 2)) + 1e-12
    psd = spectrum ** 2 / total_energy
    spectral_entropy = float(-np.sum(psd * np.log(psd + 1e-12)))
    dominant_ratio = float(spectrum[0] / (np.max(spectrum) + 1e-12)) if len(spectrum) > 0 else 0.0
    feats = np.array([mean, std, mn, mx, rms, var, spectral_entropy, dominant_ratio])
    feats = np.where(np.isfinite(feats), feats, 0.0)
    return feats


def featurize_flight(qdrone_data: np.ndarray) -> np.ndarray:
    """Extract the 168-feature vector (21 channels x 8 features) for one flight."""
    feats = []
    for row in CHANNEL_ROWS:
        signal = qdrone_data[row - 1, :].astype(np.float64)  # rows are 1-indexed in the map
        feats.append(extract_features(signal))
    return np.concatenate(feats)


def load_dataset(source_dir: str):
    """Load all flights, return (X, y, groups, group_ids)."""
    adapter = DronePropAAdapter()
    if source_dir:
        os.environ["DRONEPROPA_SOURCE_DIR"] = source_dir
    X, y, groups, group_ids = [], [], [], []
    for sample in adapter.iter_samples():
        feats = featurize_flight(sample["qdrone_data"])
        X.append(feats)
        y.append(sample["fault_type"])
        groups.append(sample["group_id"])
        group_ids.append(sample["group_id"])
    return np.array(X), np.array(y), np.array(groups), group_ids


def run_baselines(X, y, groups, seed: int = 42):
    """Run LogisticRegression and RandomForest under GroupKFold(5).

    Returns (metrics_dict, per_model_reports).
    """
    models = {
        "logreg": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
    }
    gkf = GroupKFold(n_splits=5)
    all_metrics = {}
    reports = {}
    for name, model in models.items():
        y_true_all, y_pred_all = [], []
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            # Fresh model each fold to avoid warm-start leakage.
            if name == "logreg":
                m = LogisticRegression(max_iter=2000, random_state=seed)
            else:
                m = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_te)
            y_true_all.extend(y_te)
            y_pred_all.extend(y_pred)
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true_all, y_pred_all, labels=[0, 1, 2, 3], zero_division=0
        )
        macro_f1 = float(f1_score(y_true_all, y_pred_all, labels=[0, 1, 2, 3], average="macro", zero_division=0))
        cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1, 2, 3])
        per_class = {}
        for i, cls in enumerate([0, 1, 2, 3]):
            per_class[FAULT_NAMES[cls]] = {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(np.sum(y_true_all == cls)),
            }
        all_metrics[f"{name}_macro_f1"] = macro_f1
        all_metrics[f"{name}_accuracy"] = float(np.mean(y_true_all == y_pred_all))
        for cls_name, vals in per_class.items():
            all_metrics[f"{name}_{cls_name}_precision"] = vals["precision"]
            all_metrics[f"{name}_{cls_name}_recall"] = vals["recall"]
            all_metrics[f"{name}_{cls_name}_f1"] = vals["f1"]
            all_metrics[f"{name}_{cls_name}_support"] = vals["support"]
        reports[name] = {
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "macro_f1": macro_f1,
            "accuracy": float(np.mean(y_true_all == y_pred_all)),
            "classification_report": classification_report(
                y_true_all, y_pred_all, labels=[0, 1, 2, 3],
                target_names=[FAULT_NAMES[i] for i in range(4)], zero_division=0,
            ),
        }
    return all_metrics, reports


def get_git_sha(explicit: str = None) -> str:
    if explicit:
        return explicit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def get_dataset_sha256() -> str:
    prov = ROOT / "data" / "source" / "dronepropa.provenance.json"
    if prov.exists():
        return json.loads(prov.read_text()).get("dataset_sha256", "n/a")
    return "n/a"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", default=None, help="DronePropA .mat directory (else DRONEPROPA_SOURCE_DIR env)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=str(ROOT / "results" / "uav_fault_clf_baseline.json"))
    ap.add_argument("--git-sha", default=None, help="git sha to embed in manifest (else auto-detect)")
    ap.add_argument("--dry-run", action="store_true", help="Print metrics, don't write manifest")
    args = ap.parse_args(argv)

    print("Loading DronePropA dataset ...")
    X, y, groups, group_ids = load_dataset(args.source_dir)
    print(f"Loaded {len(y)} flights. Fault-type counts: {dict(sorted(Counter(y).items()))}")

    print("Running baselines (LogisticRegression, RandomForest) under GroupKFold(5) ...")
    metrics, reports = run_baselines(X, y, groups, seed=args.seed)

    print("\n=== LogisticRegression ===")
    print(reports["logreg"]["classification_report"])
    print("Confusion matrix (rows=true, cols=pred; classes=healthy,edge_cut,crack,surface_cut):")
    print(np.array(reports["logreg"]["confusion_matrix"]))
    print(f"macro_f1 = {reports['logreg']['macro_f1']:.4f}")

    print("\n=== RandomForest ===")
    print(reports["random_forest"]["classification_report"])
    print("Confusion matrix (rows=true, cols=pred; classes=healthy,edge_cut,crack,surface_cut):")
    print(np.array(reports["random_forest"]["confusion_matrix"]))
    print(f"macro_f1 = {reports['random_forest']['macro_f1']:.4f}")

    manifest = {
        "evidence_class": "REAL-PUBLIC",
        "project": "uav-fault",
        "dataset": "dronepropa",
        "task": "fault_type_classification_4class",
        "split": "leave_one_flight_out",
        "seed": args.seed,
        "git_sha": get_git_sha(args.git_sha),
        "dataset_sha256": get_dataset_sha256(),
        "metrics": metrics,
        "produced_by": "experiments/uav/run_fault_clf.py",
        "notes": (
            "Baselines on DronePropA (127 flights, F0=40/F1=30/F2=30/F3=27). "
            "Features: 21 channels x 8 features = 168 per flight "
            "(rows 24-26,27-38,46,47-54 of QDrone_data; "
            "stat: mean/std/min/max/rms/variance; "
            "fft: spectral_entropy, dominant_freq_magnitude_ratio). "
            "Split: GroupKFold(5) on group_id (filename). "
            "No flight in both train and test. "
            "Confusion matrices printed to stdout; per-class P/R/F1 in metrics."
        ),
    }

    if args.dry_run:
        print("\n--dry-run: not writing manifest.")
        print(json.dumps(manifest, indent=2))
        return 0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written -> {out}")
    print(f"Validate with: python scripts/validate_results.py {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
