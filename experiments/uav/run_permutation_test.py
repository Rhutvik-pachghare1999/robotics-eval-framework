#!/usr/bin/env python3
"""
run_permutation_test.py — CORRECT permutation test for cross-dataset transfer.

The old scrutinize_cross_dataset.py shuffled TII TEST labels and re-evaluated
FIXED predictions — that is NOT a valid null. It tests whether the predictions
match random labels, not whether the model learned fault physics.

This script does the CORRECT permutation test:
  1. Shuffle SOURCE (DronePropA) TRAINING labels.
  2. RETRAIN the RandomForest on the shuffled-label source.
  3. Evaluate on the FIXED, unshuffled TII test set.
  4. Repeat for n_permutations (>= 1000).
  5. Report: true_f1 vs null mean/std/min/max + p = (k+1)/(n+1).

If the model learned fault physics that transfers DP->TII, shuffling the
training labels destroys that signal and the null distribution collapses to
chance (~0.5). If the model exploited a domain artifact, the null stays high.

This is a proper permutation test of the NULL HYPOTHESIS that the source labels
are exchangeable. A low p-value (p < 0.05) means we reject the null — the
source labels carry information that transfers to the target.

Output: diagnostics/uav_permutation_test.json (diagnostic, NOT a result manifest).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

# Reuse feature extraction + channel layout from run_cross_dataset
from experiments.uav.run_cross_dataset import (  # noqa: E402
    extract_features_dronepropa,
    extract_features_tii,
    CHANNEL_NAMES,
)


def load_time_broad(dp_dir: str, tii_dir: str) -> Tuple[np.ndarray, np.ndarray,
                                                          np.ndarray, np.ndarray]:
    """Load DP and TII with time_broad features (54 features per sample)."""
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
        feats = extract_features_dronepropa(qd, fs, level="time_broad")
        X_dp.append(feats)
        y_dp.append(0 if s["fault_type"] == 0 else 1)
    X_dp = np.array(X_dp)
    y_dp = np.array(y_dp)

    X_tii, y_tii = [], []
    for s in tii_adapter.iter_samples():
        feats = extract_features_tii(s["gyro_rad"], s["accel_m_s2"],
                                     s["fs_hz"], level="time_broad")
        X_tii.append(feats)
        y_tii.append(0 if s["fault_type"] == 0 else 1)
    X_tii = np.array(X_tii)
    y_tii = np.array(y_tii)

    return X_dp, y_dp, X_tii, y_tii


def train_and_eval(X_train, y_train, X_test, y_test, seed: int = 42) -> Dict[str, Any]:
    """Train RF on (X_train, y_train), evaluate on (X_test, y_test)."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    rf = RandomForestClassifier(n_estimators=200, random_state=seed,
                                 class_weight="balanced", n_jobs=-1)
    rf.fit(X_tr, y_train)
    y_pred = rf.predict(X_te)
    macro_f1 = float(f1_score(y_test, y_pred, labels=[0, 1],
                               average="macro", zero_division=0))
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
    return {"macro_f1": macro_f1, "accuracy": acc, "confusion_matrix": cm}


def permutation_test(X_dp, y_dp, X_tii, y_tii, n_permutations: int = 1000,
                     seed: int = 42) -> Dict[str, Any]:
    """Correct permutation test: shuffle SOURCE labels, RETRAIN, eval on TII.

    Null hypothesis: source labels are exchangeable (model didn't learn fault
    physics). If true, shuffling labels shouldn't change the test F1.

    p-value = (k + 1) / (n + 1), where k = # permutations >= true_f1.
    """
    rng = np.random.RandomState(seed)

    # True-label baseline: train on real DP labels, eval on TII
    print("  Training true-label model (DP -> TII) ...")
    t0 = time.time()
    true_res = train_and_eval(X_dp, y_dp, X_tii, y_tii, seed=seed)
    true_f1 = true_res["macro_f1"]
    true_acc = true_res["accuracy"]
    true_cm = true_res["confusion_matrix"]
    print(f"    true_label_f1 = {true_f1:.4f}  acc = {true_acc:.4f}  "
          f"cm = {true_cm}  ({time.time()-t0:.1f}s)")

    # In-domain sanity: train on DP, predict DP
    print("  Training in-domain model (DP -> DP) ...")
    in_domain = train_and_eval(X_dp, y_dp, X_dp, y_dp, seed=seed)
    print(f"    in_domain_f1 = {in_domain['macro_f1']:.4f}  "
          f"acc = {in_domain['accuracy']:.4f}")

    # Permutation: shuffle SOURCE (DP) labels, retrain, eval on TII
    print(f"  Running {n_permutations} permutations (shuffle DP labels, retrain, eval TII) ...")
    null_f1s = []
    null_accs = []
    t_perm_start = time.time()
    for i in range(n_permutations):
        y_dp_perm = rng.permutation(y_dp)
        res = train_and_eval(X_dp, y_dp_perm, X_tii, y_tii, seed=seed)
        null_f1s.append(res["macro_f1"])
        null_accs.append(res["accuracy"])
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_perm_start
            rate = (i + 1) / elapsed
            eta = (n_permutations - i - 1) / rate
            print(f"    [{i+1}/{n_permutations}] null_f1 mean so far = "
                  f"{np.mean(null_f1s):.4f}  (rate={rate:.1f}/s, eta={eta:.0f}s)")
    null_f1s = np.array(null_f1s)
    null_accs = np.array(null_accs)

    # p-value: (k + 1) / (n + 1) where k = # null >= true
    k_ge = int(np.sum(null_f1s >= true_f1))
    k_gt = int(np.sum(null_f1s > true_f1))
    p_ge = (k_ge + 1) / (n_permutations + 1)
    p_gt = (k_gt + 1) / (n_permutations + 1)

    return {
        "test": "permutation_test_source_labels_retrain",
        "n_permutations": n_permutations,
        "true_label_f1": true_f1,
        "true_label_acc": true_acc,
        "true_label_cm": true_cm,
        "in_domain_f1": in_domain["macro_f1"],
        "in_domain_acc": in_domain["accuracy"],
        "null_mean_f1": float(np.mean(null_f1s)),
        "null_std_f1": float(np.std(null_f1s)),
        "null_min_f1": float(np.min(null_f1s)),
        "null_max_f1": float(np.max(null_f1s)),
        "null_median_f1": float(np.median(null_f1s)),
        "null_p25_f1": float(np.percentile(null_f1s, 25)),
        "null_p75_f1": float(np.percentile(null_f1s, 75)),
        "null_p95_f1": float(np.percentile(null_f1s, 95)),
        "null_mean_acc": float(np.mean(null_accs)),
        "null_std_acc": float(np.std(null_accs)),
        "k_null_ge_true": k_ge,
        "k_null_gt_true": k_gt,
        "p_value_ge": float(p_ge),
        "p_value_gt": float(p_gt),
        "z_score": float((true_f1 - np.mean(null_f1s)) / (np.std(null_f1s) + 1e-12)),
        "null_f1_values": [float(x) for x in null_f1s[:50]],
        "verdict": (
            "real_fault_transfer" if p_ge < 0.05 else
            "domain_artifact" if np.mean(null_f1s) > 0.7 else
            "inconclusive"
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir-dronepropa", default=None)
    ap.add_argument("--source-dir-tii", default=None)
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "diagnostics" / "uav_permutation_test.json"))
    args = ap.parse_args()

    dp_dir = args.source_dir_dronepropa or os.environ.get("DRONEPROPA_SOURCE_DIR")
    tii_dir = args.source_dir_tii or os.environ.get("TII_SOURCE_DIR")
    if not dp_dir or not tii_dir:
        print("ERROR: need --source-dir-dronepropa and --source-dir-tii")
        return 1

    print("=" * 70)
    print(f"  CORRECT PERMUTATION TEST (shuffle SOURCE labels, RETRAIN RF)")
    print(f"  n_permutations = {args.n_permutations}  seed = {args.seed}")
    print("=" * 70)

    print("\nLoading datasets (time_broad, 54 features) ...")
    X_dp, y_dp, X_tii, y_tii = load_time_broad(dp_dir, tii_dir)
    print(f"  DP:  {len(y_dp)} flights (healthy={int(np.sum(y_dp==0))}, "
          f"faulty={int(np.sum(y_dp==1))}), features={X_dp.shape[1]}")
    print(f"  TII: {len(y_tii)} missions (healthy={int(np.sum(y_tii==0))}, "
          f"faulty={int(np.sum(y_tii==1))}), features={X_tii.shape[1]}")

    print(f"\nRunning permutation test ({args.n_permutations} permutations) ...")
    result = permutation_test(X_dp, y_dp, X_tii, y_tii,
                               n_permutations=args.n_permutations, seed=args.seed)

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  True-label DP->TII RF macro_f1:  {result['true_label_f1']:.4f}")
    print(f"  In-domain DP->DP RF macro_f1:     {result['in_domain_f1']:.4f}")
    print(f"  Null (shuffled source labels):    {result['null_mean_f1']:.4f} "
          f"± {result['null_std_f1']:.4f}")
    print(f"    null min = {result['null_min_f1']:.4f}")
    print(f"    null max = {result['null_max_f1']:.4f}")
    print(f"    null p25 = {result['null_p25_f1']:.4f}")
    print(f"    null median = {result['null_median_f1']:.4f}")
    print(f"    null p75 = {result['null_p75_f1']:.4f}")
    print(f"    null p95 = {result['null_p95_f1']:.4f}")
    print(f"  k null >= true: {result['k_null_ge_true']} / {result['n_permutations']}")
    print(f"  p-value (k+1)/(n+1) [null >= true]: {result['p_value_ge']:.6f}")
    print(f"  p-value (k+1)/(n+1) [null > true]:  {result['p_value_gt']:.6f}")
    print(f"  z-score: {result['z_score']:.2f}σ")
    print(f"  verdict: {result['verdict']}")
    print("=" * 70)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result["n_dp"] = int(len(y_dp))
    result["n_dp_healthy"] = int(np.sum(y_dp == 0))
    result["n_dp_faulty"] = int(np.sum(y_dp == 1))
    result["n_tii"] = int(len(y_tii))
    result["n_tii_healthy"] = int(np.sum(y_tii == 0))
    result["n_tii_faulty"] = int(np.sum(y_tii == 1))
    result["feature_dim"] = int(X_dp.shape[1])
    result["feature_level"] = "time_broad"
    result["source_dataset"] = "dronepropa"
    result["target_dataset"] = "tii-uav-fault"
    result["model"] = "RandomForestClassifier(n_estimators=200, class_weight=balanced)"
    result["permutation_method"] = "shuffle_source_labels_retrain"
    result["seed"] = args.seed
    out.write_text(json.dumps(result, indent=2))
    print(f"\n  Permutation test JSON written -> {out}")
    print("  This is a DIAGNOSTIC artifact, NOT a result manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
