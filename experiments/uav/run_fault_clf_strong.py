#!/usr/bin/env python3
"""
run_fault_clf_strong.py — stronger UAV fault-type baselines on DronePropA.

Adds two stronger models on top of the LogReg/RF baselines
(experiments/uav/run_fault_clf.py):

  1. HistGradientBoostingClassifier on the 192 hand-crafted features
     (24 channels x 8 stat/fft features). CPU, sklearn.
  2. 1D-CNN over the raw verified channels (24 channels, resampled to a
     fixed length). PyTorch, CPU or GPU.

Same split as the baselines: GroupKFold(5) on group_id (leave-one-flight-out
family). No flight appears in both train and test.

Same manifest format: evidence_class REAL-PUBLIC, dataset_sha256 from
provenance, git_sha, seed, split=leave_one_flight_out, per-class P/R/F1,
macro-F1, confusion matrix.

Usage:
    python experiments/uav/run_fault_clf_strong.py                    # both models
    python experiments/uav/run_fault_clf_strong.py --models gb         # boosting only
    python experiments/uav/run_fault_clf_strong.py --models cnn        # 1D-CNN only
    python experiments/uav/run_fault_clf_strong.py --git-sha <sha>     # embed commit
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from adapters.dronepropa import DronePropAAdapter  # noqa: E402

# Verified QDrone_data channel rows (1-indexed, per adapters/dronepropa.py).
# rows 24-26: battery, electronics_current, motor_current
# rows 27-38: gyro_1/accel_1/gyro_2/accel_2 (raw IMU, 12 ch)
# row  46:   height sensor
# rows 47-54: motor_FL/esc_FL/motor_FR/esc_FR/motor_BL/esc_BL/motor_BR/esc_BR
CHANNEL_ROWS = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 46, 47, 48, 49, 50, 51, 52, 53, 54]
N_CHANNELS = len(CHANNEL_ROWS)  # 24
FAULT_NAMES = {0: "healthy", 1: "edge_cut", 2: "crack", 3: "surface_cut"}
FIXED_LEN = 1024  # resample each flight to this many time steps for the CNN


# ── Feature extraction (shared with run_fault_clf.py) ────────────────────────

def extract_features(signal: np.ndarray) -> np.ndarray:
    """8 features per channel: mean, std, min, max, rms, variance,
    spectral_entropy, dominant_freq_magnitude_ratio. NaN-aware."""
    arr = np.asarray(signal, dtype=np.float64)
    if arr.shape[0] == 0:
        return np.zeros(8)
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    rms = float(np.sqrt(np.nanmean(arr ** 2)))
    var = float(np.nanvar(arr))
    filled = np.where(np.isfinite(arr), arr, mean)
    spectrum = np.abs(np.fft.rfft(filled - mean))
    total_energy = float(np.sum(spectrum ** 2)) + 1e-12
    psd = spectrum ** 2 / total_energy
    spectral_entropy = float(-np.sum(psd * np.log(psd + 1e-12)))
    dominant_ratio = float(spectrum[0] / (np.max(spectrum) + 1e-12)) if len(spectrum) > 0 else 0.0
    feats = np.array([mean, std, mn, mx, rms, var, spectral_entropy, dominant_ratio])
    return np.where(np.isfinite(feats), feats, 0.0)


def featurize_flight(qdrone_data: np.ndarray) -> np.ndarray:
    """192-feature vector (24 channels x 8 features) for one flight."""
    feats = []
    for row in CHANNEL_ROWS:
        signal = qdrone_data[row - 1, :].astype(np.float64)
        feats.append(extract_features(signal))
    return np.concatenate(feats)


# ── Raw-channel resampling for the 1D-CNN ─────────────────────────────────────

def resample_signal(signal: np.ndarray, target_len: int = FIXED_LEN) -> np.ndarray:
    """Resample a 1-D signal to target_len via linear interpolation.
    NaNs are replaced with the channel mean before resampling."""
    arr = np.asarray(signal, dtype=np.float64)
    n = arr.shape[0]
    if n == 0:
        return np.zeros(target_len)
    finite = arr[np.isfinite(arr)]
    fill = float(np.nanmean(arr)) if finite.shape[0] > 0 else 0.0
    arr = np.where(np.isfinite(arr), arr, fill)
    if n == target_len:
        return arr
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, arr)


def raw_channels_flight(qdrone_data: np.ndarray, target_len: int = FIXED_LEN) -> np.ndarray:
    """Return (N_CHANNELS, target_len) array of resampled verified channels."""
    out = np.zeros((N_CHANNELS, target_len), dtype=np.float32)
    for i, row in enumerate(CHANNEL_ROWS):
        out[i] = resample_signal(qdrone_data[row - 1, :], target_len)
    return out


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(source_dir: str):
    """Load all flights. Returns (X_feat, X_raw, y, groups, group_ids)."""
    adapter = DronePropAAdapter()
    if source_dir:
        os.environ["DRONEPROPA_SOURCE_DIR"] = source_dir
    X_feat, X_raw, y, groups = [], [], [], []
    for sample in adapter.iter_samples():
        X_feat.append(featurize_flight(sample["qdrone_data"]))
        X_raw.append(raw_channels_flight(sample["qdrone_data"]))
        y.append(sample["fault_type"])
        groups.append(sample["group_id"])
    return (
        np.array(X_feat, dtype=np.float64),
        np.array(X_raw, dtype=np.float32),
        np.array(y, dtype=np.int64),
        np.array(groups),
    )


# ── Gradient boosting on features ─────────────────────────────────────────────

def run_gradient_boosting(X, y, groups, seed: int = 42):
    """HistGradientBoostingClassifier, GroupKFold(5)."""
    gkf = GroupKFold(n_splits=5)
    y_true_all, y_pred_all = [], []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_depth=None,
            l2_regularization=1.0, random_state=seed,
        )
        m.fit(X_tr, y[train_idx])
        y_pred = m.predict(X_te)
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(y_pred)
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    return y_true_all, y_pred_all


# ── 1D-CNN over raw channels ──────────────────────────────────────────────────

def run_cnn(X_raw, y, groups, seed: int = 42, epochs: int = 30, batch_size: int = 16):
    """Small 1D-CNN, GroupKFold(5). PyTorch."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gkf = GroupKFold(n_splits=5)
    y_true_all, y_pred_all = [], []

    n_classes = 4
    in_ch = N_CHANNELS

    class CNN1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(in_ch, 32, 7, padding=3)
            self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
            self.conv3 = nn.Conv1d(64, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(64, 32)
            self.fc2 = nn.Linear(32, n_classes)
            self.drop = nn.Dropout(0.3)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = self.pool(x).squeeze(-1)
            x = self.drop(x)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_raw, y, groups)):
        torch.manual_seed(seed)
        # Per-channel standardization using train statistics
        X_tr = X_raw[train_idx].copy()
        X_te = X_raw[test_idx].copy()
        mean = X_tr.mean(axis=(0, 2), keepdims=True)
        std = X_tr.std(axis=(0, 2), keepdims=True) + 1e-8
        X_tr = (X_tr - mean) / std
        X_te = (X_te - mean) / std

        model = CNN1D().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        X_tr_t = torch.from_numpy(X_tr).float().to(device)
        y_tr_t = torch.from_numpy(y[train_idx]).long().to(device)
        X_te_t = torch.from_numpy(X_te).float().to(device)

        ds = TensorDataset(X_tr_t, y_tr_t)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        model.train()
        for ep in range(epochs):
            for xb, yb in dl:
                opt.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_te_t)
            y_pred = logits.argmax(dim=1).cpu().numpy()
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(y_pred)

    return np.array(y_true_all), np.array(y_pred_all)


# ── Metrics + manifest ─────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, prefix: str):
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
    )
    macro_f1 = float(f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist()
    metrics = {
        f"{prefix}_macro_f1": macro_f1,
        f"{prefix}_accuracy": float(np.mean(y_true == y_pred)),
    }
    for i, cls in enumerate([0, 1, 2, 3]):
        metrics[f"{prefix}_{FAULT_NAMES[cls]}_precision"] = float(prec[i])
        metrics[f"{prefix}_{FAULT_NAMES[cls]}_recall"] = float(rec[i])
        metrics[f"{prefix}_{FAULT_NAMES[cls]}_f1"] = float(f1[i])
        metrics[f"{prefix}_{FAULT_NAMES[cls]}_support"] = int(np.sum(y_true == cls))
    return metrics, cm, classification_report(
        y_true, y_pred, labels=[0, 1, 2, 3],
        target_names=[FAULT_NAMES[i] for i in range(4)], zero_division=0,
    )


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
    ap.add_argument("--source-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", default="gb,cnn", help="comma-separated: gb,cnn")
    ap.add_argument("--cnn-epochs", type=int, default=30)
    ap.add_argument("--cnn-batch-size", type=int, default=16)
    ap.add_argument("--output", default=str(ROOT / "results" / "uav_fault_clf_strong.json"))
    ap.add_argument("--git-sha", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    model_set = set(args.models.split(","))

    print("Loading DronePropA dataset ...")
    X_feat, X_raw, y, groups = load_dataset(args.source_dir)
    print(f"Loaded {len(y)} flights. Fault-type counts: {dict(sorted(Counter(y).items()))}")
    print(f"X_feat shape: {X_feat.shape}  X_raw shape: {X_raw.shape}")

    all_metrics = {}
    reports = {}

    if "gb" in model_set:
        print("\n=== HistGradientBoosting (features, GroupKFold=5) ===")
        y_true, y_pred = run_gradient_boosting(X_feat, y, groups, seed=args.seed)
        m, cm, rpt = compute_metrics(y_true, y_pred, "gb")
        all_metrics.update(m)
        reports["gb"] = {"confusion_matrix": cm, "report": rpt}
        print(rpt)
        print("Confusion matrix (rows=true, cols=pred; healthy,edge_cut,crack,surface_cut):")
        print(np.array(cm))
        print(f"macro_f1 = {m['gb_macro_f1']:.4f}")

    if "cnn" in model_set:
        print("\n=== 1D-CNN (raw channels, GroupKFold=5) ===")
        y_true, y_pred = run_cnn(
            X_raw, y, groups, seed=args.seed,
            epochs=args.cnn_epochs, batch_size=args.cnn_batch_size,
        )
        m, cm, rpt = compute_metrics(y_true, y_pred, "cnn")
        all_metrics.update(m)
        reports["cnn"] = {"confusion_matrix": cm, "report": rpt}
        print(rpt)
        print("Confusion matrix (rows=true, cols=pred; healthy,edge_cut,crack,surface_cut):")
        print(np.array(cm))
        print(f"macro_f1 = {m['cnn_macro_f1']:.4f}")

    manifest = {
        "evidence_class": "SIMULATED",
        "project": "uav-fault",
        "dataset": "dronepropa",
        "task": "fault_type_classification_4class",
        "split": "leave_one_flight_out",
        "seed": args.seed,
        "git_sha": get_git_sha(args.git_sha),
        "dataset_sha256": get_dataset_sha256(),
        "metrics": all_metrics,
        "produced_by": "experiments/uav/run_fault_clf_strong.py",
        "notes": (
            "Stronger models on DronePropA (127 flights, F0=40/F1=30/F2=30/F3=27). "
            "gb: HistGradientBoostingClassifier on 192 features (24 ch x 8 stat/fft). "
            "cnn: 1D-CNN over 24 raw verified channels (rows 24-26,27-38,46,47-54), "
            "resampled to 1024 time steps, 30 epochs, batch 16. "
            "Split: GroupKFold(5) on group_id (filename). No flight in both train and test. "
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sys.exit(main())
