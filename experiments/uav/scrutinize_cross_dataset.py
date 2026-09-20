#!/usr/bin/env python3
"""
scrutinize_cross_dataset.py — Investigate the 0.93 macro_F1 for domain leakage.

The DP->TII direction achieved macro_F1=0.93 with 80/80 faulty perfect but
only 15/19 healthy correct, and the reverse TII->DP is only 0.49. This
asymmetry suggests the model may be learning "is this TII?" (domain marker)
rather than "is this faulty?" (fault physics).

Three tests:
  1. Feature distributions: DP-healthy vs DP-faulty vs TII-healthy vs TII-faulty
     - Per feature: mean, std
     - Domain separability: |mean_DP - mean_TII| / pooled_std  (does it separate by dataset?)
     - Fault separability: |mean_healthy - mean_faulty| / pooled_std  (does it separate by fault?)
     - If domain_sep >> fault_sep for top features → domain leakage

  2. Permutation test: shuffle TII test labels, re-evaluate RF trained on DP
     - If macro_F1 stays high → model exploits domain structure, not fault
     - If macro_F1 drops to ~0.5 → model learned real fault signal

  3. RF feature importances: rank features
     - Domain markers: rms (signal magnitude), band_energy (absolute levels)
     - Fault physics: kurtosis, crest_factor, skew, spectral_kurtosis
     - If top features are domain markers → leakage

Output: results/uav_cross_dataset_scrutiny.json (diagnostic, NOT a result manifest)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from adapters.dronepropa import DronePropAAdapter  # noqa: E402
from adapters.tii import TIIUAVAdapter  # noqa: E402

# Reuse feature extraction from run_cross_dataset
from experiments.uav.run_cross_dataset import (  # noqa: E402
    extract_features_dronepropa,
    extract_features_tii,
    feature_dim,
    CHANNEL_NAMES,
    DRONEPROPA_GYRO_ROWS,
    DRONEPROPA_ACCEL_ROWS,
    coral_align,
)

# Feature names (6 channels x 9 features = 54)
TIME_DOMAIN_NAMES = ["rms", "kurtosis", "crest_factor", "shape_factor", "skew"]
BROADBAND_NAMES = ["spectral_centroid", "spectral_kurtosis", "band_energy_low", "band_energy_high"]
FEATURE_NAMES = []
for ch in CHANNEL_NAMES:
    for fn in TIME_DOMAIN_NAMES + BROADBAND_NAMES:
        FEATURE_NAMES.append(f"{ch}_{fn}")


def load_datasets(dp_dir: str, tii_dir: str, level: str = "time_broad"):
    """Load DP and TII with features. Returns X_dp, y_dp, X_tii, y_tii."""
    os.environ["DRONEPROPA_SOURCE_DIR"] = dp_dir
    os.environ["TII_SOURCE_DIR"] = tii_dir
    dp_adapter = DronePropAAdapter()
    tii_adapter = TIIUAVAdapter()

    X_dp, y_dp = [], []
    for s in dp_adapter.iter_samples():
        qd = s["qdrone_data"]
        time_s = qd[0, :]
        dt = np.diff(time_s)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        fs = 1.0 / np.mean(dt) if dt.size > 0 else 1000.0
        feats = extract_features_dronepropa(qd, fs, level)
        X_dp.append(feats)
        y_dp.append(0 if s["fault_type"] == 0 else 1)
    X_dp = np.array(X_dp)
    y_dp = np.array(y_dp)

    X_tii, y_tii = [], []
    for s in tii_adapter.iter_samples():
        feats = extract_features_tii(s["gyro_rad"], s["accel_m_s2"], s["fs_hz"], level)
        X_tii.append(feats)
        y_tii.append(0 if s["fault_type"] == 0 else 1)
    X_tii = np.array(X_tii)
    y_tii = np.array(y_tii)

    return X_dp, y_dp, X_tii, y_tii


def feature_distributions(X_dp: np.ndarray, y_dp: np.ndarray,
                          X_tii: np.ndarray, y_tii: np.ndarray) -> List[Dict]:
    """Per-feature: mean/std for DP-healthy, DP-faulty, TII-healthy, TII-faulty.

    Also compute:
      - domain_sep: |mean_DP - mean_TII| / pooled_std  (separates by dataset?)
      - fault_sep_dp: |mean_DP_h - mean_DP_f| / pooled_std  (separates by fault in DP?)
      - fault_sep_tii: |mean_TII_h - mean_TII_f| / pooled_std  (separates by fault in TII?)
    """
    dp_h = X_dp[y_dp == 0]
    dp_f = X_dp[y_dp == 1]
    tii_h = X_tii[y_tii == 0]
    tii_f = X_tii[y_tii == 1]
    results = []
    for i, fname in enumerate(FEATURE_NAMES):
        col_dp_h = dp_h[:, i]
        col_dp_f = dp_f[:, i]
        col_tii_h = tii_h[:, i]
        col_tii_f = tii_f[:, i]
        mean_dp_h = float(np.nanmean(col_dp_h))
        mean_dp_f = float(np.nanmean(col_dp_f))
        mean_tii_h = float(np.nanmean(col_tii_h))
        mean_tii_f = float(np.nanmean(col_tii_f))
        std_dp_h = float(np.nanstd(col_dp_h))
        std_dp_f = float(np.nanstd(col_dp_f))
        std_tii_h = float(np.nanstd(col_tii_h))
        std_tii_f = float(np.nanstd(col_tii_f))
        # Domain separability: DP (all) vs TII (all)
        pooled_std_domain = np.sqrt((np.nanvar(X_dp[:, i]) + np.nanvar(X_tii[:, i])) / 2) + 1e-12
        domain_sep = abs(np.nanmean(X_dp[:, i]) - np.nanmean(X_tii[:, i])) / pooled_std_domain
        # Fault separability within DP
        pooled_std_dp = np.sqrt((std_dp_h ** 2 + std_dp_f ** 2) / 2) + 1e-12
        fault_sep_dp = abs(mean_dp_h - mean_dp_f) / pooled_std_dp
        # Fault separability within TII
        pooled_std_tii = np.sqrt((std_tii_h ** 2 + std_tii_f ** 2) / 2) + 1e-12
        fault_sep_tii = abs(mean_tii_h - mean_tii_f) / pooled_std_tii
        # Domain separability for healthy vs faulty separately
        # (does the feature separate DP-healthy from TII-healthy? → domain, not fault)
        pooled_std_h = np.sqrt((std_dp_h ** 2 + std_tii_h ** 2) / 2) + 1e-12
        domain_sep_healthy = abs(mean_dp_h - mean_tii_h) / pooled_std_h
        pooled_std_f = np.sqrt((std_dp_f ** 2 + std_tii_f ** 2) / 2) + 1e-12
        domain_sep_faulty = abs(mean_dp_f - mean_tii_f) / pooled_std_f
        results.append({
            "feature": fname,
            "index": i,
            "dp_healthy_mean": mean_dp_h, "dp_healthy_std": std_dp_h,
            "dp_faulty_mean": mean_dp_f, "dp_faulty_std": std_dp_f,
            "tii_healthy_mean": mean_tii_h, "tii_healthy_std": std_tii_h,
            "tii_faulty_mean": mean_tii_f, "tii_faulty_std": std_tii_f,
            "domain_sep": float(domain_sep),
            "domain_sep_healthy": float(domain_sep_healthy),
            "domain_sep_faulty": float(domain_sep_faulty),
            "fault_sep_dp": float(fault_sep_dp),
            "fault_sep_tii": float(fault_sep_tii),
            "leakage_ratio": float(domain_sep / (fault_sep_dp + fault_sep_tii + 1e-12)),
        })
    return results


def permutation_test(X_dp, y_dp, X_tii, y_tii, n_permutations: int = 100,
                     seed: int = 42) -> Dict:
    """Shuffle TII test labels, re-evaluate RF trained on DP.

    If macro_F1 stays high with shuffled labels → domain leakage.
    If macro_F1 drops to ~0.5 → real fault transfer.
    """
    rng = np.random.RandomState(seed)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_dp)
    X_te = scaler.transform(X_tii)
    # True-label baseline
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced", n_jobs=-1)
    rf.fit(X_tr, y_dp)
    y_pred_true = rf.predict(X_te)
    f1_true = float(f1_score(y_tii, y_pred_true, labels=[0, 1], average="macro", zero_division=0))
    acc_true = float(accuracy_score(y_tii, y_pred_true))
    cm_true = confusion_matrix(y_tii, y_pred_true, labels=[0, 1]).tolist()
    # Permuted labels
    f1s_perm = []
    accs_perm = []
    for i in range(n_permutations):
        y_tii_perm = rng.permutation(y_tii)
        f1_perm = float(f1_score(y_tii_perm, y_pred_true, labels=[0, 1],
                                  average="macro", zero_division=0))
        acc_perm = float(accuracy_score(y_tii_perm, y_pred_true))
        f1s_perm.append(f1_perm)
        accs_perm.append(acc_perm)
    f1s_perm = np.array(f1s_perm)
    accs_perm = np.array(accs_perm)
    # Also: train on DP, predict DP (in-domain sanity check)
    y_pred_dp = rf.predict(X_tr)
    f1_dp = float(f1_score(y_dp, y_pred_dp, labels=[0, 1], average="macro", zero_division=0))
    acc_dp = float(accuracy_score(y_dp, y_pred_dp))
    cm_dp = confusion_matrix(y_dp, y_pred_dp, labels=[0, 1]).tolist()
    return {
        "true_label_f1": f1_true,
        "true_label_acc": acc_true,
        "true_label_cm": cm_true,
        "in_domain_dp_f1": f1_dp,
        "in_domain_dp_acc": acc_dp,
        "in_domain_dp_cm": cm_dp,
        "permutation_mean_f1": float(np.mean(f1s_perm)),
        "permutation_std_f1": float(np.std(f1s_perm)),
        "permutation_min_f1": float(np.min(f1s_perm)),
        "permutation_max_f1": float(np.max(f1s_perm)),
        "permutation_mean_acc": float(np.mean(accs_perm)),
        "permutation_std_acc": float(np.std(accs_perm)),
        "permutation_p_value": float(np.mean(f1s_perm >= f1_true)),
        "n_permutations": n_permutations,
        "permutation_f1_values": [float(x) for x in f1s_perm[:20]],
    }


def feature_importances(X_dp, y_dp, seed: int = 42) -> List[Tuple[str, float]]:
    """Train RF on DP, return feature importances ranked descending."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_dp)
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced", n_jobs=-1)
    rf.fit(X_tr, y_dp)
    importances = rf.feature_importances_
    ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
    return [(str(n), float(v)) for n, v in ranked]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir-dronepropa", default=None)
    ap.add_argument("--source-dir-tii", default=None)
    ap.add_argument("--level", default="time_broad", choices=["naive", "time_broad"])
    ap.add_argument("--n-permutations", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/uav_cross_dataset_scrutiny.json")
    args = ap.parse_args()

    dp_dir = args.source_dir_dronepropa or os.environ.get("DRONEPROPA_SOURCE_DIR")
    tii_dir = args.source_dir_tii or os.environ.get("TII_SOURCE_DIR")
    if not dp_dir or not tii_dir:
        print("ERROR: need --source-dir-dronepropa and --source-dir-tii")
        return 1

    print("=" * 70)
    print(f"  CROSS-DATASET SCRUTINY: investigating 0.93 macro_F1 for domain leakage")
    print(f"  Level: {args.level}  |  Permutations: {args.n_permutations}")
    print("=" * 70)

    print("\nLoading datasets ...")
    X_dp, y_dp, X_tii, y_tii = load_datasets(dp_dir, tii_dir, args.level)
    print(f"  DP:  {len(y_dp)} flights (healthy={int(np.sum(y_dp==0))}, faulty={int(np.sum(y_dp==1))})")
    print(f"  TII: {len(y_tii)} missions (healthy={int(np.sum(y_tii==0))}, faulty={int(np.sum(y_tii==1))})")
    print(f"  Features: {X_dp.shape[1]}")

    # Test 1: Feature distributions
    print("\n--- Test 1: Feature distributions ---")
    dists = feature_distributions(X_dp, y_dp, X_tii, y_tii)
    # Sort by leakage ratio (domain_sep / fault_sep) descending
    dists_sorted = sorted(dists, key=lambda d: -d["leakage_ratio"])
    print(f"\n  Top 15 features by LEAKAGE RATIO (domain_sep / fault_sep):")
    print(f"  {'feature':<35} {'dom_sep':>8} {'fault_dp':>8} {'fault_tii':>8} {'leak_rat':>8}  "
          f"DP_h→DP_f  TII_h→TII_f  DP→TII")
    for d in dists_sorted[:15]:
        print(f"  {d['feature']:<35} {d['domain_sep']:>8.2f} {d['fault_sep_dp']:>8.2f} "
              f"{d['fault_sep_tii']:>8.2f} {d['leakage_ratio']:>8.2f}  "
              f"{d['dp_healthy_mean']:>8.3f}→{d['dp_faulty_mean']:>8.3f}  "
              f"{d['tii_healthy_mean']:>8.3f}→{d['tii_faulty_mean']:>8.3f}  "
              f"{d['dp_healthy_mean']:>8.3f}vs{d['tii_healthy_mean']:>8.3f}")

    print(f"\n  Top 15 features by FAULT separability in TII (fault_sep_tii):")
    dists_by_fault = sorted(dists, key=lambda d: -d["fault_sep_tii"])
    print(f"  {'feature':<35} {'dom_sep':>8} {'fault_dp':>8} {'fault_tii':>8} {'leak_rat':>8}")
    for d in dists_by_fault[:15]:
        print(f"  {d['feature']:<35} {d['domain_sep']:>8.2f} {d['fault_sep_dp']:>8.2f} "
              f"{d['fault_sep_tii']:>8.2f} {d['leakage_ratio']:>8.2f}")

    # Test 2: Permutation test
    print(f"\n--- Test 2: Permutation test ({args.n_permutations} shuffles) ---")
    perm = permutation_test(X_dp, y_dp, X_tii, y_tii, args.n_permutations, args.seed)
    print(f"  True-label DP->TII:  macro_f1={perm['true_label_f1']:.4f}  acc={perm['true_label_acc']:.4f}")
    print(f"  In-domain DP->DP:    macro_f1={perm['in_domain_dp_f1']:.4f}  acc={perm['in_domain_dp_acc']:.4f}")
    print(f"  Permuted DP->TII:    macro_f1={perm['permutation_mean_f1']:.4f} ± {perm['permutation_std_f1']:.4f}  "
          f"(min={perm['permutation_min_f1']:.4f}, max={perm['permutation_max_f1']:.4f})")
    print(f"  Permutation p-value (frac shuffled >= true): {perm['permutation_p_value']:.4f}")
    print(f"  True-label confusion (DP->TII): {perm['true_label_cm']}")
    print(f"  In-domain confusion (DP->DP):   {perm['in_domain_dp_cm']}")
    if perm["permutation_mean_f1"] > 0.7:
        print("  ⚠ LEAKAGE SUSPECTED: shuffled labels still high → model exploits domain, not fault")
    elif perm["permutation_mean_f1"] < 0.55:
        print("  ✓ REAL FAULT TRANSFER: shuffled labels drop to chance → model learned fault signal")
    else:
        print("  ? PARTIAL: shuffled labels moderate → some domain structure exploited")

    # Test 3: Feature importances
    print("\n--- Test 3: RF feature importances (trained on DP) ---")
    importances = feature_importances(X_dp, y_dp, args.seed)
    print(f"  Top 20 features by RF importance:")
    for i, (name, imp) in enumerate(importances[:20]):
        # Classify as domain marker or fault physics
        tag = ""
        if "rms" in name or "band_energy" in name:
            tag = "DOMAIN_MARKER (magnitude/energy)"
        elif "kurtosis" in name or "crest" in name or "skew" in name or "shape" in name:
            tag = "FAULT_PHYSICS (shape)"
        elif "spectral_centroid" in name or "spectral_kurtosis" in name:
            tag = "SPECTRAL (shape)"
        print(f"    {i+1:>2}. {name:<35} {imp:.4f}  {tag}")

    # Write JSON
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "level": args.level,
        "n_dp": len(y_dp), "n_dp_healthy": int(np.sum(y_dp == 0)), "n_dp_faulty": int(np.sum(y_dp == 1)),
        "n_tii": len(y_tii), "n_tii_healthy": int(np.sum(y_tii == 0)), "n_tii_faulty": int(np.sum(y_tii == 1)),
        "feature_dim": X_dp.shape[1],
        "test1_feature_distributions": dists,
        "test1_top_leakage": dists_sorted[:20],
        "test1_top_fault_sep_tii": dists_by_fault[:20],
        "test2_permutation": perm,
        "test3_feature_importances": importances[:30],
        "verdict": {
            "permutation_mean_f1": perm["permutation_mean_f1"],
            "permutation_p_value": perm["permutation_p_value"],
            "true_label_f1": perm["true_label_f1"],
            "leakage_suspected": perm["permutation_mean_f1"] > 0.7,
            "real_fault_transfer": perm["permutation_mean_f1"] < 0.55,
        },
    }
    out.write_text(json.dumps(result, indent=2))
    print(f"\nScrutiny JSON written -> {out}")
    print("\nThis is a DIAGNOSTIC artifact, NOT a result manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
