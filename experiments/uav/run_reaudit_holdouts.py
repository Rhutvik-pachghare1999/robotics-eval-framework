#!/usr/bin/env python3
"""
run_reaudit_holdouts.py — Re-audit holdouts on the SHAPE-ONLY feature set (the
honest 0.74, not the amplitude-inflated 0.929).

Three experiments, each emitting a validated manifest (REAL-PUBLIC, real_to_real):

  3. FROZEN + BOOTSTRAP CI: freeze the shape-only feature/model choice, report
     TII macro-F1 ONCE with a bootstrap 95% CI over the 99 missions.

  4. DOMAIN CLASSIFIER: train a DP-vs-TII classifier on shape-only features;
     report how separable the domains are (quantifies confound risk).

  5. HOLDOUTS within DronePropA: cross-trajectory (train t1-4 / test t5, rotate),
     cross-speed, cross-drone. Report macro-F1 each.

All use the shape-only feature set (42 features: kurtosis, crest_factor,
shape_factor, skew, spectral_centroid, spectral_kurtosis, band_energy_ratio).
NO rms, NO absolute band energy (amplitude-ablated).

Output:
  results/uav_bootstrap_ci.json        (manifest, #3)
  results/uav_domain_classifier.json    (manifest, #4)
  results/uav_holdouts.json             (manifest, #5)
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.utils import resample

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from adapters.dronepropa import DronePropAAdapter  # noqa: E402
from adapters.tii import TIIUAVAdapter  # noqa: E402

from experiments.uav.run_amplitude_ablation import (  # noqa: E402
    extract_shape_features_dronepropa,
    extract_shape_features_tii,
    load_dronepropa_shape,
    load_tii_shape,
    SHAPE_FEATURE_NAMES,
)
from experiments.uav.run_cross_dataset import (  # noqa: E402
    get_git_sha,
    get_dataset_sha256,
    CHANNEL_NAMES,
)


# ─── Experiment 3: Frozen + Bootstrap CI ─────────────────────────────────────

def bootstrap_ci(X_dp, y_dp, X_tii, y_tii, n_bootstrap: int = 2000,
                 seed: int = 42) -> Dict[str, Any]:
    """Freeze the shape-only feature/model choice, report TII macro-F1 ONCE
    with a bootstrap 95% CI over the 99 TII missions.

    Bootstrap procedure:
      1. Train RF on DP (frozen, full DP training set).
      2. Evaluate on TII (frozen, full TII test set) → point estimate.
      3. Bootstrap: resample TII test set with replacement (n=99), compute
         macro-F1 on each resample. Repeat n_bootstrap times.
      4. Report 95% CI as the 2.5th and 97.5th percentiles of the bootstrap
         distribution.
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_dp)
    X_te = scaler.transform(X_tii)
    rf = RandomForestClassifier(n_estimators=200, random_state=seed,
                                 class_weight="balanced", n_jobs=-1)
    rf.fit(X_tr, y_dp)
    y_pred = rf.predict(X_te)
    y_prob = rf.predict_proba(X_te)[:, 1]

    point_f1 = float(f1_score(y_tii, y_pred, labels=[0, 1], average="macro",
                               zero_division=0))
    point_acc = float(accuracy_score(y_tii, y_pred))
    try:
        point_auc = float(roc_auc_score(y_tii, y_prob))
    except (ValueError, IndexError):
        point_auc = float("nan")

    rng = np.random.RandomState(seed)
    n_tii = len(y_tii)
    boot_f1s = []
    boot_accs = []
    for i in range(n_bootstrap):
        idx = rng.choice(n_tii, size=n_tii, replace=True)
        y_tii_boot = y_tii[idx]
        y_pred_boot = y_pred[idx]
        # Skip if bootstrap sample has only one class
        if len(np.unique(y_tii_boot)) < 2:
            continue
        f1_b = float(f1_score(y_tii_boot, y_pred_boot, labels=[0, 1],
                               average="macro", zero_division=0))
        acc_b = float(accuracy_score(y_tii_boot, y_pred_boot))
        boot_f1s.append(f1_b)
        boot_accs.append(acc_b)
    boot_f1s = np.array(boot_f1s)
    boot_accs = np.array(boot_accs)

    ci_lo_f1 = float(np.percentile(boot_f1s, 2.5))
    ci_hi_f1 = float(np.percentile(boot_f1s, 97.5))
    ci_lo_acc = float(np.percentile(boot_accs, 2.5))
    ci_hi_acc = float(np.percentile(boot_accs, 97.5))

    return {
        "point_macro_f1": point_f1,
        "point_accuracy": point_acc,
        "point_auc": point_auc,
        "bootstrap_n": len(boot_f1s),
        "bootstrap_mean_f1": float(np.mean(boot_f1s)),
        "bootstrap_std_f1": float(np.std(boot_f1s)),
        "bootstrap_ci95_lo_f1": ci_lo_f1,
        "bootstrap_ci95_hi_f1": ci_hi_f1,
        "bootstrap_mean_acc": float(np.mean(boot_accs)),
        "bootstrap_std_acc": float(np.std(boot_accs)),
        "bootstrap_ci95_lo_acc": ci_lo_acc,
        "bootstrap_ci95_hi_acc": ci_hi_acc,
        "n_tii": int(n_tii),
        "n_tii_healthy": int(np.sum(y_tii == 0)),
        "n_tii_faulty": int(np.sum(y_tii == 1)),
        "n_dp_train": int(len(y_dp)),
        "n_dp_healthy": int(np.sum(y_dp == 0)),
        "n_dp_faulty": int(np.sum(y_dp == 1)),
        "cm": confusion_matrix(y_tii, y_pred, labels=[0, 1]).tolist(),
        "bootstrap_f1_values": [float(x) for x in boot_f1s[:50]],
    }


# ─── Experiment 4: Domain Classifier ─────────────────────────────────────────

def domain_classifier(X_dp, X_tii, seed: int = 42) -> Dict[str, Any]:
    """Train a DP-vs-TII classifier on shape-only features.

    If the domains are highly separable, the cross-dataset transfer may be
    exploiting domain identity rather than fault physics. Reports:
      - macro-F1, accuracy, AUC for DP-vs-TII classification
      - feature importances (top 10)
    """
    X = np.vstack([X_dp, X_tii])
    y_domain = np.concatenate([np.zeros(len(X_dp)), np.ones(len(X_tii))])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=200, random_state=seed,
                                 class_weight="balanced", n_jobs=-1)
    rf.fit(X_s, y_domain)
    y_pred = rf.predict(X_s)
    y_prob = rf.predict_proba(X_s)[:, 1]
    macro_f1 = float(f1_score(y_domain, y_pred, labels=[0, 1], average="macro",
                               zero_division=0))
    acc = float(accuracy_score(y_domain, y_pred))
    try:
        auc = float(roc_auc_score(y_domain, y_prob))
    except (ValueError, IndexError):
        auc = float("nan")
    cm = confusion_matrix(y_domain, y_pred, labels=[0, 1]).tolist()
    importances = rf.feature_importances_
    ranked = sorted(zip(SHAPE_FEATURE_NAMES, importances), key=lambda x: -x[1])
    top_features = [(str(n), float(v)) for n, v in ranked[:10]]
    return {
        "domain_macro_f1": macro_f1,
        "domain_accuracy": acc,
        "domain_auc": auc,
        "domain_cm": cm,
        "n_dp": int(len(X_dp)),
        "n_tii": int(len(X_tii)),
        "n_total": int(len(X)),
        "top_features": top_features,
        "verdict": (
            "highly_separable" if macro_f1 > 0.95 else
            "separable" if macro_f1 > 0.80 else
            "moderately_separable" if macro_f1 > 0.65 else
            "weakly_separable" if macro_f1 > 0.55 else
            "not_separable"
        ),
    }


# ─── Experiment 5: Holdouts within DronePropA ────────────────────────────────

def parse_group_id(gid: str) -> Dict[str, str]:
    """Parse DronePropA group_id 'F{fault}_SV{sev}_SP{speed}_t{traj}_D{drone}'
    into components."""
    parts = gid.split("_")
    out = {}
    for p in parts:
        if p.startswith("F"):
            out["fault"] = p
        elif p.startswith("SV"):
            out["severity"] = p
        elif p.startswith("SP"):
            out["speed"] = p
        elif p.startswith("t"):
            out["trajectory"] = p
        elif p.startswith("D"):
            out["drone"] = p
    return out


def holdout_cross_trajectory(X_dp, y_dp, gids_dp, seed: int = 42) -> Dict[str, Any]:
    """Cross-trajectory holdout: train t1-4, test t5 (rotate)."""
    results = []
    trajectories = ["t1", "t2", "t3", "t4", "t5"]
    for test_traj in trajectories:
        train_idx = []
        test_idx = []
        for i, gid in enumerate(gids_dp):
            parts = parse_group_id(gid)
            traj = parts.get("trajectory", "")
            if traj == test_traj:
                test_idx.append(i)
            elif traj in trajectories:
                train_idx.append(i)
        if not train_idx or not test_idx:
            continue
        X_tr, y_tr = X_dp[train_idx], y_dp[train_idx]
        X_te, y_te = X_dp[test_idx], y_dp[test_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            results.append({"test_trajectory": test_traj, "skipped": True,
                             "n_train": len(y_tr), "n_test": len(y_te)})
            continue
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        rf = RandomForestClassifier(n_estimators=200, random_state=seed,
                                     class_weight="balanced", n_jobs=-1)
        rf.fit(X_tr_s, y_tr)
        y_pred = rf.predict(X_te_s)
        f1 = float(f1_score(y_te, y_pred, labels=[0, 1], average="macro",
                             zero_division=0))
        acc = float(accuracy_score(y_te, y_pred))
        cm = confusion_matrix(y_te, y_pred, labels=[0, 1]).tolist()
        results.append({
            "test_trajectory": test_traj,
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "n_train_healthy": int(np.sum(y_tr == 0)),
            "n_train_faulty": int(np.sum(y_tr == 1)),
            "n_test_healthy": int(np.sum(y_te == 0)),
            "n_test_faulty": int(np.sum(y_te == 1)),
            "macro_f1": f1,
            "accuracy": acc,
            "cm": cm,
        })
    f1s = [r["macro_f1"] for r in results if "macro_f1" in r]
    return {
        "holdout": "cross_trajectory",
        "method": "leave-one-trajectory-out (train t1-4, test t5, rotate)",
        "per_trajectory": results,
        "mean_macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "std_macro_f1": float(np.std(f1s)) if f1s else 0.0,
        "n_folds": len(f1s),
    }


def holdout_cross_speed(X_dp, y_dp, gids_dp, seed: int = 42) -> Dict[str, Any]:
    """Cross-speed holdout: train on all but one speed, test on held-out speed."""
    results = []
    speeds = ["SP1", "SP2", "SP3"]
    for test_speed in speeds:
        train_idx = []
        test_idx = []
        for i, gid in enumerate(gids_dp):
            parts = parse_group_id(gid)
            speed = parts.get("speed", "")
            if speed == test_speed:
                test_idx.append(i)
            elif speed in speeds:
                train_idx.append(i)
        if not train_idx or not test_idx:
            continue
        X_tr, y_tr = X_dp[train_idx], y_dp[train_idx]
        X_te, y_te = X_dp[test_idx], y_dp[test_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            results.append({"test_speed": test_speed, "skipped": True,
                             "n_train": len(y_tr), "n_test": len(y_te)})
            continue
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        rf = RandomForestClassifier(n_estimators=200, random_state=seed,
                                     class_weight="balanced", n_jobs=-1)
        rf.fit(X_tr_s, y_tr)
        y_pred = rf.predict(X_te_s)
        f1 = float(f1_score(y_te, y_pred, labels=[0, 1], average="macro",
                             zero_division=0))
        acc = float(accuracy_score(y_te, y_pred))
        cm = confusion_matrix(y_te, y_pred, labels=[0, 1]).tolist()
        results.append({
            "test_speed": test_speed,
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "n_train_healthy": int(np.sum(y_tr == 0)),
            "n_train_faulty": int(np.sum(y_tr == 1)),
            "n_test_healthy": int(np.sum(y_te == 0)),
            "n_test_faulty": int(np.sum(y_te == 1)),
            "macro_f1": f1,
            "accuracy": acc,
            "cm": cm,
        })
    f1s = [r["macro_f1"] for r in results if "macro_f1" in r]
    return {
        "holdout": "cross_speed",
        "method": "leave-one-speed-out (train on 2 speeds, test on held-out speed)",
        "per_speed": results,
        "mean_macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "std_macro_f1": float(np.std(f1s)) if f1s else 0.0,
        "n_folds": len(f1s),
    }


def holdout_cross_drone(X_dp, y_dp, gids_dp, seed: int = 42) -> Dict[str, Any]:
    """Cross-drone holdout: train on all but one drone, test on held-out drone."""
    results = []
    drones = sorted(set(parse_group_id(g).get("drone", "") for g in gids_dp))
    for test_drone in drones:
        if not test_drone:
            continue
        train_idx = []
        test_idx = []
        for i, gid in enumerate(gids_dp):
            parts = parse_group_id(gid)
            drone = parts.get("drone", "")
            if drone == test_drone:
                test_idx.append(i)
            elif drone in drones:
                train_idx.append(i)
        if not train_idx or not test_idx:
            continue
        X_tr, y_tr = X_dp[train_idx], y_dp[train_idx]
        X_te, y_te = X_dp[test_idx], y_dp[test_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            results.append({"test_drone": test_drone, "skipped": True,
                             "n_train": len(y_tr), "n_test": len(y_te)})
            continue
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        rf = RandomForestClassifier(n_estimators=200, random_state=seed,
                                     class_weight="balanced", n_jobs=-1)
        rf.fit(X_tr_s, y_tr)
        y_pred = rf.predict(X_te_s)
        f1 = float(f1_score(y_te, y_pred, labels=[0, 1], average="macro",
                             zero_division=0))
        acc = float(accuracy_score(y_te, y_pred))
        cm = confusion_matrix(y_te, y_pred, labels=[0, 1]).tolist()
        results.append({
            "test_drone": test_drone,
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "n_train_healthy": int(np.sum(y_tr == 0)),
            "n_train_faulty": int(np.sum(y_tr == 1)),
            "n_test_healthy": int(np.sum(y_te == 0)),
            "n_test_faulty": int(np.sum(y_te == 1)),
            "macro_f1": f1,
            "accuracy": acc,
            "cm": cm,
        })
    f1s = [r["macro_f1"] for r in results if "macro_f1" in r]
    return {
        "holdout": "cross_drone",
        "method": "leave-one-drone-out (train on all but one drone, test on held-out drone)",
        "per_drone": results,
        "mean_macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "std_macro_f1": float(np.std(f1s)) if f1s else 0.0,
        "n_folds": len(f1s),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir-dronepropa", default=None)
    ap.add_argument("--source-dir-tii", default=None)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    args = ap.parse_args()

    dp_dir = args.source_dir_dronepropa or os.environ.get("DRONEPROPA_SOURCE_DIR")
    tii_dir = args.source_dir_tii or os.environ.get("TII_SOURCE_DIR")
    if not dp_dir or not tii_dir:
        print("ERROR: need --source-dir-dronepropa and --source-dir-tii")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  RE-AUDIT HOLDOUTS (shape-only features, the honest 0.74)")
    print("  DROPS: rms, absolute band energy (amplitude-ablated)")
    print("  KEEPS: kurtosis, crest_factor, shape_factor, skew,")
    print("         spectral_centroid, spectral_kurtosis, band_energy_ratio")
    print("=" * 70)

    print("\nLoading DronePropA (shape-only features) ...")
    X_dp, y_dp, gids_dp = load_dronepropa_shape(dp_dir)
    print(f"  DP: {len(y_dp)} flights, healthy={int(np.sum(y_dp==0))}, "
          f"faulty={int(np.sum(y_dp==1))}, features={X_dp.shape[1]}")

    print("Loading TII (shape-only features) ...")
    X_tii, y_tii, gids_tii = load_tii_shape(tii_dir)
    print(f"  TII: {len(y_tii)} missions, healthy={int(np.sum(y_tii==0))}, "
          f"faulty={int(np.sum(y_tii==1))}, features={X_tii.shape[1]}")

    dp_sha = get_dataset_sha256("dronepropa")
    tii_sha = get_dataset_sha256("tii-uav-fault")
    git_sha = get_git_sha()

    # ─── Experiment 3: Bootstrap CI ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Experiment 3: Frozen + Bootstrap CI (n_bootstrap={args.n_bootstrap})")
    print(f"{'='*70}")
    boot = bootstrap_ci(X_dp, y_dp, X_tii, y_tii, n_bootstrap=args.n_bootstrap,
                        seed=args.seed)
    print(f"  Point estimate: macro_f1={boot['point_macro_f1']:.4f}  "
          f"acc={boot['point_accuracy']:.4f}  auc={boot['point_auc']:.4f}")
    print(f"  Bootstrap 95% CI (macro_f1): [{boot['bootstrap_ci95_lo_f1']:.4f}, "
          f"{boot['bootstrap_ci95_hi_f1']:.4f}]  "
          f"(mean={boot['bootstrap_mean_f1']:.4f} ± {boot['bootstrap_std_f1']:.4f})")
    print(f"  Bootstrap 95% CI (accuracy): [{boot['bootstrap_ci95_lo_acc']:.4f}, "
          f"{boot['bootstrap_ci95_hi_acc']:.4f}]")
    print(f"  Bootstrap samples: {boot['bootstrap_n']}")
    print(f"  Confusion matrix (rows=true healthy/faulty): {boot['cm']}")

    boot_manifest = {
        "evidence_class": "REAL-PUBLIC",
        "source_dataset_class": "REAL-PUBLIC",
        "target_dataset_class": "REAL-PUBLIC",
        "transfer_type": "real_to_real",
        "project": "uav-fault",
        "dataset": "dronepropa->tii-uav-fault",
        "task": "cross_dataset_fault_detection_binary_bootstrap_ci",
        "split": "cross_dataset",
        "level": "shape_only",
        "feature_dim": X_dp.shape[1],
        "channels": CHANNEL_NAMES,
        "seed": args.seed,
        "git_sha": git_sha,
        "dataset_sha256": dp_sha,
        "source_dataset_sha256": dp_sha,
        "target_dataset_sha256": tii_sha,
        "metrics": {
            "point_macro_f1": boot["point_macro_f1"],
            "point_accuracy": boot["point_accuracy"],
            "point_auc": boot["point_auc"],
            "bootstrap_n": float(boot["bootstrap_n"]),
            "bootstrap_mean_f1": boot["bootstrap_mean_f1"],
            "bootstrap_std_f1": boot["bootstrap_std_f1"],
            "bootstrap_ci95_lo_f1": boot["bootstrap_ci95_lo_f1"],
            "bootstrap_ci95_hi_f1": boot["bootstrap_ci95_hi_f1"],
            "bootstrap_mean_acc": boot["bootstrap_mean_acc"],
            "bootstrap_std_acc": boot["bootstrap_std_acc"],
            "bootstrap_ci95_lo_acc": boot["bootstrap_ci95_lo_acc"],
            "bootstrap_ci95_hi_acc": boot["bootstrap_ci95_hi_acc"],
            "n_tii": float(boot["n_tii"]),
            "n_tii_healthy": float(boot["n_tii_healthy"]),
            "n_tii_faulty": float(boot["n_tii_faulty"]),
            "n_dp_train": float(boot["n_dp_train"]),
            "n_dp_healthy": float(boot["n_dp_healthy"]),
            "n_dp_faulty": float(boot["n_dp_faulty"]),
            "cm_healthy_pred_healthy": boot["cm"][0][0],
            "cm_healthy_pred_faulty": boot["cm"][0][1],
            "cm_faulty_pred_healthy": boot["cm"][1][0],
            "cm_faulty_pred_faulty": boot["cm"][1][1],
        },
        "produced_by": "experiments/uav/run_reaudit_holdouts.py",
        "notes": (
            f"Frozen + bootstrap CI on shape-only features (the honest 0.74). "
            f"Train RF on DP (shape-only, 42 features), evaluate on TII (99 missions). "
            f"Bootstrap: resample TII test set with replacement (n=99), compute macro-F1 "
            f"per resample, report 95% CI as 2.5/97.5 percentiles. "
            f"n_bootstrap={args.n_bootstrap}, seed={args.seed}. "
            f"Feature set: shape_only_no_amplitude (kurtosis, crest_factor, shape_factor, "
            f"skew, spectral_centroid, spectral_kurtosis, band_energy_ratio). "
            f"DROPS: rms, absolute band energy (amplitude-ablated). "
            f"Transfer type: real_to_real (source=DronePropA REAL-PUBLIC, target=TII REAL-PUBLIC). "
            f"DronePropA: 127 flights (130 nominal, 3 missing). TII: 99 missions. "
            f"Point estimate: macro_f1={boot['point_macro_f1']:.4f}, "
            f"95% CI=[{boot['bootstrap_ci95_lo_f1']:.4f}, {boot['bootstrap_ci95_hi_f1']:.4f}]."
        ),
    }
    boot_out = out_dir / "uav_bootstrap_ci.json"
    boot_out.write_text(json.dumps(boot_manifest, indent=2))
    print(f"\n  Manifest written -> {boot_out}")

    # ─── Experiment 4: Domain Classifier ───────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Experiment 4: Domain classifier (DP vs TII, shape-only features)")
    print(f"{'='*70}")
    dom = domain_classifier(X_dp, X_tii, seed=args.seed)
    print(f"  Domain macro_f1: {dom['domain_macro_f1']:.4f}")
    print(f"  Domain accuracy: {dom['domain_accuracy']:.4f}")
    print(f"  Domain AUC: {dom['domain_auc']:.4f}")
    print(f"  Confusion matrix (rows=true DP/TII): {dom['domain_cm']}")
    print(f"  Verdict: {dom['verdict']}")
    print(f"  Top 10 features (domain separability):")
    for name, imp in dom["top_features"]:
        print(f"    {name:<35} {imp:.4f}")

    dom_manifest = {
        "evidence_class": "REAL-PUBLIC",
        "source_dataset_class": "REAL-PUBLIC",
        "target_dataset_class": "REAL-PUBLIC",
        "transfer_type": "real_to_real",
        "project": "uav-fault",
        "dataset": "dronepropa-vs-tii-domain",
        "task": "domain_classification_dp_vs_tii",
        "split": "cross_dataset",
        "level": "shape_only",
        "feature_dim": X_dp.shape[1],
        "channels": CHANNEL_NAMES,
        "seed": args.seed,
        "git_sha": git_sha,
        "dataset_sha256": dp_sha,
        "source_dataset_sha256": dp_sha,
        "target_dataset_sha256": tii_sha,
        "metrics": {
            "domain_macro_f1": dom["domain_macro_f1"],
            "domain_accuracy": dom["domain_accuracy"],
            "domain_auc": dom["domain_auc"],
            "domain_cm_dp_pred_dp": dom["domain_cm"][0][0],
            "domain_cm_dp_pred_tii": dom["domain_cm"][0][1],
            "domain_cm_tii_pred_dp": dom["domain_cm"][1][0],
            "domain_cm_tii_pred_tii": dom["domain_cm"][1][1],
            "n_dp": float(dom["n_dp"]),
            "n_tii": float(dom["n_tii"]),
            "n_total": float(dom["n_total"]),
        },
        "produced_by": "experiments/uav/run_reaudit_holdouts.py",
        "notes": (
            f"Domain classifier: train RF on shape-only features to classify DP vs TII. "
            f"If macro_f1 > 0.95, domains are highly separable (confound risk). "
            f"If < 0.65, domains are not separable (low confound risk). "
            f"Feature set: shape_only_no_amplitude (42 features, amplitude-ablated). "
            f"DP: {dom['n_dp']} flights, TII: {dom['n_tii']} missions, total: {dom['n_total']}. "
            f"Verdict: {dom['verdict']} (macro_f1={dom['domain_macro_f1']:.4f}). "
            f"Top 10 features (by RF importance): "
            + ", ".join(f"{n}={v:.4f}" for n, v in dom["top_features"])
            + "."
        ),
    }
    dom_out = out_dir / "uav_domain_classifier.json"
    dom_out.write_text(json.dumps(dom_manifest, indent=2))
    print(f"\n  Manifest written -> {dom_out}")

    # ─── Experiment 5: Holdouts within DronePropA ─────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Experiment 5: Holdouts within DronePropA (shape-only features)")
    print(f"{'='*70}")

    print("\n  Cross-trajectory (leave-one-trajectory-out, train t1-4 / test t5, rotate):")
    ct = holdout_cross_trajectory(X_dp, y_dp, gids_dp, seed=args.seed)
    print(f"    mean macro_f1 = {ct['mean_macro_f1']:.4f} ± {ct['std_macro_f1']:.4f}  "
          f"(n_folds={ct['n_folds']})")
    for r in ct["per_trajectory"]:
        if "macro_f1" in r:
            print(f"      test={r['test_trajectory']}: macro_f1={r['macro_f1']:.4f}  "
                  f"acc={r['accuracy']:.4f}  n_train={r['n_train']}  n_test={r['n_test']}  "
                  f"cm={r['cm']}")
        else:
            print(f"      test={r['test_trajectory']}: SKIPPED (n_train={r['n_train']}, "
                  f"n_test={r['n_test']})")

    print("\n  Cross-speed (leave-one-speed-out):")
    cs = holdout_cross_speed(X_dp, y_dp, gids_dp, seed=args.seed)
    print(f"    mean macro_f1 = {cs['mean_macro_f1']:.4f} ± {cs['std_macro_f1']:.4f}  "
          f"(n_folds={cs['n_folds']})")
    for r in cs["per_speed"]:
        if "macro_f1" in r:
            print(f"      test={r['test_speed']}: macro_f1={r['macro_f1']:.4f}  "
                  f"acc={r['accuracy']:.4f}  n_train={r['n_train']}  n_test={r['n_test']}  "
                  f"cm={r['cm']}")
        else:
            print(f"      test={r['test_speed']}: SKIPPED (n_train={r['n_train']}, "
                  f"n_test={r['n_test']})")

    print("\n  Cross-drone (leave-one-drone-out):")
    cd = holdout_cross_drone(X_dp, y_dp, gids_dp, seed=args.seed)
    print(f"    mean macro_f1 = {cd['mean_macro_f1']:.4f} ± {cd['std_macro_f1']:.4f}  "
          f"(n_folds={cd['n_folds']})")
    for r in cd["per_drone"]:
        if "macro_f1" in r:
            print(f"      test={r['test_drone']}: macro_f1={r['macro_f1']:.4f}  "
                  f"acc={r['accuracy']:.4f}  n_train={r['n_train']}  n_test={r['n_test']}  "
                  f"cm={r['cm']}")
        else:
            print(f"      test={r['test_drone']}: SKIPPED (n_train={r['n_train']}, "
                  f"n_test={r['n_test']})")

    hold_manifest = {
        "evidence_class": "REAL-PUBLIC",
        "source_dataset_class": "REAL-PUBLIC",
        "target_dataset_class": "REAL-PUBLIC",
        "transfer_type": "real_to_real",
        "project": "uav-fault",
        "dataset": "dronepropa-holdouts",
        "task": "within_dataset_holdouts_shape_only",
        "split": "leave_one_group_out",
        "level": "shape_only",
        "feature_dim": X_dp.shape[1],
        "channels": CHANNEL_NAMES,
        "seed": args.seed,
        "git_sha": git_sha,
        "dataset_sha256": dp_sha,
        "source_dataset_sha256": dp_sha,
        "target_dataset_sha256": dp_sha,
        "metrics": {
            "cross_trajectory_mean_macro_f1": ct["mean_macro_f1"],
            "cross_trajectory_std_macro_f1": ct["std_macro_f1"],
            "cross_trajectory_n_folds": float(ct["n_folds"]),
            "cross_speed_mean_macro_f1": cs["mean_macro_f1"],
            "cross_speed_std_macro_f1": cs["std_macro_f1"],
            "cross_speed_n_folds": float(cs["n_folds"]),
            "cross_drone_mean_macro_f1": cd["mean_macro_f1"],
            "cross_drone_std_macro_f1": cd["std_macro_f1"],
            "cross_drone_n_folds": float(cd["n_folds"]),
        },
        "produced_by": "experiments/uav/run_reaudit_holdouts.py",
        "notes": (
            f"Within-DronePropA holdouts on shape-only features (the honest 0.74 feature set). "
            f"Cross-trajectory: leave-one-trajectory-out (train t1-4, test t5, rotate). "
            f"Cross-speed: leave-one-speed-out (train on 2 speeds, test on held-out). "
            f"Cross-drone: leave-one-drone-out (train on all but one drone, test on held-out). "
            f"Feature set: shape_only_no_amplitude (42 features, amplitude-ablated). "
            f"DP: 127 flights (130 nominal, 3 missing). "
            f"Cross-trajectory: {ct['mean_macro_f1']:.4f} ± {ct['std_macro_f1']:.4f} "
            f"(n_folds={ct['n_folds']}). "
            f"Cross-speed: {cs['mean_macro_f1']:.4f} ± {cs['std_macro_f1']:.4f} "
            f"(n_folds={cs['n_folds']}). "
            f"Cross-drone: {cd['mean_macro_f1']:.4f} ± {cd['std_macro_f1']:.4f} "
            f"(n_folds={cd['n_folds']}). "
            f"Per-fold details in diagnostics/uav_holdouts_detail.json."
        ),
    }
    hold_out = out_dir / "uav_holdouts.json"
    hold_out.write_text(json.dumps(hold_manifest, indent=2))
    print(f"\n  Manifest written -> {hold_out}")

    # Write per-fold details to diagnostics/ (not a manifest, doesn't need schema validation)
    diag_dir = ROOT / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    hold_detail = {
        "experiment": "within_dronepropa_holdouts_shape_only",
        "feature_set": "shape_only_no_amplitude",
        "feature_dim": X_dp.shape[1],
        "cross_trajectory": ct,
        "cross_speed": cs,
        "cross_drone": cd,
        "produced_by": "experiments/uav/run_reaudit_holdouts.py",
    }
    hold_detail_out = diag_dir / "uav_holdouts_detail.json"
    hold_detail_out.write_text(json.dumps(hold_detail, indent=2))
    print(f"  Per-fold details written -> {hold_detail_out}")

    # ─── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY (shape-only features, the honest 0.74)")
    print(f"{'='*70}")
    print(f"  #3 Bootstrap CI:     point={boot['point_macro_f1']:.4f}  "
          f"95% CI=[{boot['bootstrap_ci95_lo_f1']:.4f}, {boot['bootstrap_ci95_hi_f1']:.4f}]")
    print(f"  #4 Domain classifier: macro_f1={dom['domain_macro_f1']:.4f}  "
          f"verdict={dom['verdict']}")
    print(f"  #5 Holdouts:")
    print(f"     cross_trajectory: {ct['mean_macro_f1']:.4f} ± {ct['std_macro_f1']:.4f}")
    print(f"     cross_speed:       {cs['mean_macro_f1']:.4f} ± {cs['std_macro_f1']:.4f}")
    print(f"     cross_drone:       {cd['mean_macro_f1']:.4f} ± {cd['std_macro_f1']:.4f}")
    print(f"{'='*70}")
    print(f"\n  Manifests:")
    print(f"    {boot_out}")
    print(f"    {dom_out}")
    print(f"    {hold_out}")
    print(f"\n  Validate: python scripts/validate_results.py {boot_out} {dom_out} {hold_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
