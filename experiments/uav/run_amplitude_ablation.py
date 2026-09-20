#!/usr/bin/env python3
"""
run_amplitude_ablation.py — Amplitude-ablation: does 0.929 survive WITHOUT
absolute-magnitude features?

Rebuilds the feature pipeline with ONLY shape/normalized features:
  - Time-domain: kurtosis, crest_factor, shape_factor, skew  (4 per channel)
  - Spectral:    spectral_centroid, spectral_kurtosis, band_energy_ratio  (3 per channel)
    band_energy_ratio = band_energy_high / (band_energy_low + band_energy_high)
    (normalized — drops absolute energy)

DROPS: rms (absolute magnitude), band_energy_low (absolute), band_energy_high
(absolute), raw amplitude.

6 channels (gyro xyz + accel xyz) x 7 features = 42 features.

If 0.929 drops substantially (e.g. to ~0.5-0.7), the original result was
driven by amplitude (domain marker). If it survives (~0.9), the result is
driven by shape (fault physics).

Output: results/uav_amplitude_ablation.json manifest (REAL-PUBLIC, real_to_real).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from adapters.dronepropa import DronePropAAdapter  # noqa: E402
from adapters.tii import TIIUAVAdapter  # noqa: E402

# Reuse channel layout + dataset loaders from run_cross_dataset
from experiments.uav.run_cross_dataset import (  # noqa: E402
    DRONEPROPA_GYRO_ROWS,
    DRONEPROPA_ACCEL_ROWS,
    CHANNEL_NAMES,
    get_git_sha,
    get_dataset_sha256,
)


# ─── Shape-only features (amplitude-ablated) ───────────────────────────────

TIME_SHAPE_NAMES = ["kurtosis", "crest_factor", "shape_factor", "skew"]
SPEC_SHAPE_NAMES = ["spectral_centroid", "spectral_kurtosis", "band_energy_ratio"]
SHAPE_FEATURE_NAMES = []
for ch in CHANNEL_NAMES:
    for fn in TIME_SHAPE_NAMES + SPEC_SHAPE_NAMES:
        SHAPE_FEATURE_NAMES.append(f"{ch}_{fn}")


def _time_shape_features(signal: np.ndarray) -> np.ndarray:
    """Extract 4 shape/normalized time-domain features (NO rms, NO absolute).

    kurtosis:      tail heaviness (shape, amplitude-invariant)
    crest_factor:  peak / rms (shape, normalized)
    shape_factor:  rms / mean(|x|) (shape, normalized)
    skew:          asymmetry (shape, amplitude-invariant)
    """
    arr = np.asarray(signal, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    n = finite.shape[0]
    if n == 0:
        return np.zeros(4)
    rms = float(np.sqrt(np.mean(finite ** 2)))
    abs_mean = float(np.mean(np.abs(finite)))
    peak = float(np.max(np.abs(finite)))
    kurt = float(scipy_kurtosis(finite, fisher=True, bias=False)) if n > 3 else 0.0
    crest = peak / (rms + 1e-12)
    shape = rms / (abs_mean + 1e-12)
    sk = float(scipy_skew(finite, bias=False)) if n > 2 else 0.0
    feats = np.array([kurt, crest, shape, sk], dtype=np.float64)
    return np.where(np.isfinite(feats), feats, 0.0)


def _spectral_shape_features(signal: np.ndarray, fs: float) -> np.ndarray:
    """Extract 3 shape/normalized spectral features (NO absolute energy).

    spectral_centroid:  weighted mean of normalized freq (shape, amplitude-invariant)
    spectral_kurtosis:  kurtosis of normalized PSD (shape, amplitude-invariant)
    band_energy_ratio:  E_high / (E_low + E_high)  (normalized, amplitude-invariant)
    """
    arr = np.asarray(signal, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    n = finite.shape[0]
    if n < 32 or not np.isfinite(fs) or fs <= 0:
        return np.zeros(3)
    mean = float(np.mean(finite))
    filled = np.where(np.isfinite(arr), arr, mean)
    nperseg = min(256, n)
    freqs, psd = welch(filled - mean, fs=fs, nperseg=nperseg, window="hann")
    total_energy = float(np.sum(psd)) + 1e-12
    norm_freqs = freqs / fs  # normalized 0-0.5
    centroid = float(np.sum(norm_freqs * psd) / total_energy)
    psd_norm = psd / total_energy
    if psd_norm.shape[0] > 3:
        spec_kurt = float(scipy_kurtosis(psd_norm, fisher=True, bias=False))
    else:
        spec_kurt = 0.0
    nyquist = 0.5
    low_mask = norm_freqs <= 0.1 * nyquist
    high_mask = (norm_freqs > 0.1 * nyquist) & (norm_freqs <= 0.5 * nyquist)
    e_low = float(np.sum(psd[low_mask]))
    e_high = float(np.sum(psd[high_mask]))
    band_ratio = e_high / (e_low + e_high + 1e-12)
    feats = np.array([centroid, spec_kurt, band_ratio], dtype=np.float64)
    return np.where(np.isfinite(feats), feats, 0.0)


def extract_shape_features_dronepropa(qdrone_data: np.ndarray, fs: float) -> np.ndarray:
    """Extract shape-only features from DronePropA QDrone_data (56 x N)."""
    signals = []
    for row in DRONEPROPA_GYRO_ROWS:
        signals.append(qdrone_data[row - 1, :])
    for row in DRONEPROPA_ACCEL_ROWS:
        signals.append(qdrone_data[row - 1, :])
    feats = []
    for sig in signals:
        feats.append(_time_shape_features(sig))
        feats.append(_spectral_shape_features(sig, fs))
    return np.concatenate(feats)


def extract_shape_features_tii(gyro: np.ndarray, accel: np.ndarray, fs: float) -> np.ndarray:
    """Extract shape-only features from TII SensorCombined data."""
    signals = [gyro[:, 0], gyro[:, 1], gyro[:, 2],
               accel[:, 0], accel[:, 1], accel[:, 2]]
    feats = []
    for sig in signals:
        feats.append(_time_shape_features(sig))
        feats.append(_spectral_shape_features(sig, fs))
    return np.concatenate(feats)


def load_dronepropa_shape(source_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load DronePropA with shape-only features. Returns (X, y_binary, group_ids)."""
    if source_dir:
        os.environ["DRONEPROPA_SOURCE_DIR"] = source_dir
    adapter = DronePropAAdapter()
    X, y, gids = [], [], []
    for sample in adapter.iter_samples():
        qd = sample["qdrone_data"]
        time_s = qd[0, :]
        dt = np.diff(time_s)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        fs = 1.0 / np.mean(dt) if dt.size > 0 else 1000.0
        feats = extract_shape_features_dronepropa(qd, fs)
        X.append(feats)
        y.append(0 if sample["fault_type"] == 0 else 1)
        gids.append(sample["group_id"])
    return np.array(X), np.array(y), gids


def load_tii_shape(source_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load TII with shape-only features. Returns (X, y_binary, group_ids)."""
    if source_dir:
        os.environ["TII_SOURCE_DIR"] = source_dir
    adapter = TIIUAVAdapter()
    X, y, gids = [], [], []
    for sample in adapter.iter_samples():
        feats = extract_shape_features_tii(sample["gyro_rad"], sample["accel_m_s2"],
                                            sample["fs_hz"])
        X.append(feats)
        y.append(0 if sample["fault_type"] == 0 else 1)
        gids.append(sample["group_id"])
    return np.array(X), np.array(y), gids


def cross_dataset_eval_shape(X_train, y_train, X_test, y_test,
                              seed: int = 42) -> Dict[str, Any]:
    """Train on source, test on target with shape-only features."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    models = {
        "logreg": LogisticRegression(max_iter=2000, random_state=seed,
                                       class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed,
                                                  class_weight="balanced", n_jobs=-1),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else y_pred
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=[0, 1], zero_division=0
        )
        macro_f1 = float(f1_score(y_test, y_pred, labels=[0, 1],
                                   average="macro", zero_division=0))
        acc = float(accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        try:
            auc = float(roc_auc_score(y_test, y_prob))
        except (ValueError, IndexError):
            auc = float("nan")
        results[name] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "auc": auc,
            "healthy_precision": float(prec[0]),
            "healthy_recall": float(rec[0]),
            "healthy_f1": float(f1[0]),
            "healthy_support": int(np.sum(y_test == 0)),
            "faulty_precision": float(prec[1]),
            "faulty_recall": float(rec[1]),
            "faulty_f1": float(f1[1]),
            "faulty_support": int(np.sum(y_test == 1)),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y_test, y_pred, labels=[0, 1],
                target_names=["healthy", "faulty"], zero_division=0,
            ),
        }
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir-dronepropa", default=None)
    ap.add_argument("--source-dir-tii", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "results" / "uav_amplitude_ablation.json"))
    args = ap.parse_args(argv)

    dp_dir = args.source_dir_dronepropa or os.environ.get("DRONEPROPA_SOURCE_DIR")
    tii_dir = args.source_dir_tii or os.environ.get("TII_SOURCE_DIR")
    if not dp_dir or not tii_dir:
        print("ERROR: need --source-dir-dronepropa and --source-dir-tii")
        return 1

    print("=" * 70)
    print("  AMPLITUDE ABLATION: shape-only features (NO rms, NO absolute energy)")
    print("=" * 70)

    print("\nLoading DronePropA (shape-only features) ...")
    X_dp, y_dp, gids_dp = load_dronepropa_shape(dp_dir)
    print(f"  DP: {len(y_dp)} flights, healthy={int(np.sum(y_dp==0))}, "
          f"faulty={int(np.sum(y_dp==1))}, features={X_dp.shape[1]}")

    print("Loading TII (shape-only features) ...")
    X_tii, y_tii, gids_tii = load_tii_shape(tii_dir)
    print(f"  TII: {len(y_tii)} missions, healthy={int(np.sum(y_tii==0))}, "
          f"faulty={int(np.sum(y_tii==1))}, features={X_tii.shape[1]}")

    # Direction 1: DP -> TII
    print("\n  Direction 1: DronePropA -> TII (shape-only)")
    r_dp2tii = cross_dataset_eval_shape(X_dp, y_dp, X_tii, y_tii, args.seed)
    for mname, mres in r_dp2tii.items():
        print(f"    {mname}: macro_f1={mres['macro_f1']:.4f}  "
              f"acc={mres['accuracy']:.4f}  auc={mres['auc']:.4f}")
        print(f"      confusion (rows=true healthy/faulty): {mres['confusion_matrix']}")

    # Direction 2: TII -> DP
    print("\n  Direction 2: TII -> DronePropA (shape-only)")
    r_tii2dp = cross_dataset_eval_shape(X_tii, y_tii, X_dp, y_dp, args.seed)
    for mname, mres in r_tii2dp.items():
        print(f"    {mname}: macro_f1={mres['macro_f1']:.4f}  "
              f"acc={mres['accuracy']:.4f}  auc={mres['auc']:.4f}")
        print(f"      confusion (rows=true healthy/faulty): {mres['confusion_matrix']}")

    # Original 0.929 for comparison
    print("\n" + "=" * 70)
    print("  COMPARISON: original time_broad (54 feat) DP->TII RF macro_f1 = 0.9290")
    print(f"  AMPLITUDE ABLATION (shape-only, {X_dp.shape[1]} feat) DP->TII RF macro_f1 = "
          f"{r_dp2tii['random_forest']['macro_f1']:.4f}")
    delta = r_dp2tii['random_forest']['macro_f1'] - 0.9290
    print(f"  DELTA: {delta:+.4f}")
    if r_dp2tii['random_forest']['macro_f1'] < 0.70:
        print("  VERDICT: 0.929 does NOT survive amplitude ablation → driven by amplitude (domain marker)")
    elif r_dp2tii['random_forest']['macro_f1'] < 0.85:
        print("  VERDICT: 0.929 partially survives → mixed amplitude + shape")
    else:
        print("  VERDICT: 0.929 survives amplitude ablation → driven by shape (fault physics)")
    print("=" * 70)

    # Build manifest
    dp_sha = get_dataset_sha256("dronepropa")
    tii_sha = get_dataset_sha256("tii-uav-fault")
    metrics = {}
    for direction_name, direction_data in [("dp_to_tii", r_dp2tii), ("tii_to_dp", r_tii2dp)]:
        for model_name, model_res in direction_data.items():
            prefix = f"{direction_name}_{model_name}"
            metrics[f"{prefix}_macro_f1"] = model_res["macro_f1"]
            metrics[f"{prefix}_accuracy"] = model_res["accuracy"]
            metrics[f"{prefix}_auc"] = model_res["auc"]
            metrics[f"{prefix}_healthy_precision"] = model_res["healthy_precision"]
            metrics[f"{prefix}_healthy_recall"] = model_res["healthy_recall"]
            metrics[f"{prefix}_healthy_f1"] = model_res["healthy_f1"]
            metrics[f"{prefix}_healthy_support"] = model_res["healthy_support"]
            metrics[f"{prefix}_faulty_precision"] = model_res["faulty_precision"]
            metrics[f"{prefix}_faulty_recall"] = model_res["faulty_recall"]
            metrics[f"{prefix}_faulty_f1"] = model_res["faulty_f1"]
            metrics[f"{prefix}_faulty_support"] = model_res["faulty_support"]
            cm = model_res["confusion_matrix"]
            metrics[f"{prefix}_cm_healthy_pred_healthy"] = cm[0][0]
            metrics[f"{prefix}_cm_healthy_pred_faulty"] = cm[0][1]
            metrics[f"{prefix}_cm_faulty_pred_healthy"] = cm[1][0]
            metrics[f"{prefix}_cm_faulty_pred_faulty"] = cm[1][1]
    metrics["dp_to_tii_n_train"] = float(len(y_dp))
    metrics["dp_to_tii_n_test"] = float(len(y_tii))
    metrics["dp_to_tii_train_healthy"] = float(np.sum(y_dp == 0))
    metrics["dp_to_tii_train_faulty"] = float(np.sum(y_dp == 1))
    metrics["dp_to_tii_test_healthy"] = float(np.sum(y_tii == 0))
    metrics["dp_to_tii_test_faulty"] = float(np.sum(y_tii == 1))
    metrics["tii_to_dp_n_train"] = float(len(y_tii))
    metrics["tii_to_dp_n_test"] = float(len(y_dp))
    metrics["original_time_broad_dp_to_tii_rf_macro_f1"] = 0.9290
    metrics["amplitude_ablation_delta"] = float(delta)

    manifest = {
        "evidence_class": "REAL-PUBLIC",
        "source_dataset_class": "REAL-PUBLIC",
        "target_dataset_class": "REAL-PUBLIC",
        "transfer_type": "real_to_real",
        "project": "uav-fault",
        "dataset": "dronepropa->tii-uav-fault",
        "task": "cross_dataset_fault_detection_binary_amplitude_ablation",
        "split": "cross_dataset",
        "level": "shape_only",
        "feature_dim": X_dp.shape[1],
        "channels": CHANNEL_NAMES,
        "seed": args.seed,
        "git_sha": get_git_sha(),
        "dataset_sha256": dp_sha,
        "source_dataset_sha256": dp_sha,
        "target_dataset_sha256": tii_sha,
        "metrics": metrics,
        "produced_by": "experiments/uav/run_amplitude_ablation.py",
        "notes": (
            f"Amplitude ablation: rebuild features with ONLY shape/normalized features. "
            f"DROPS rms (absolute magnitude), band_energy_low (absolute), band_energy_high (absolute). "
            f"KEEPS kurtosis, crest_factor, shape_factor, skew (time shape), spectral_centroid, "
            f"spectral_kurtosis, band_energy_ratio (spectral shape, normalized). "
            f"6 channels x 7 features = {X_dp.shape[1]} features. "
            f"If 0.929 drops substantially, the original result was driven by amplitude (domain marker). "
            f"If it survives, the result is driven by shape (fault physics). "
            f"Original time_broad DP->TII RF macro_f1 = 0.9290 (54 features, includes rms + absolute energy). "
            f"Amplitude ablation DP->TII RF macro_f1 = {r_dp2tii['random_forest']['macro_f1']:.4f} "
            f"({X_dp.shape[1]} features, shape-only). "
            f"Delta = {delta:+.4f}. "
            f"Transfer type: real_to_real (source=DronePropA REAL-PUBLIC, target=TII REAL-PUBLIC). "
            f"DronePropA: 127 flights (130 nominal, 3 missing — see exclusion manifest). "
            f"TII: 99 missions."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"\n  Manifest written -> {out}")
    print(f"  Validate: python scripts/validate_results.py {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
