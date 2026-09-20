#!/usr/bin/env python3
"""
diagnose_motor_esc.py — Check rows 47-54 (motor/ESC telemetry) for propeller RPM.

Hypothesis: 24.41 Hz is the propeller rotation frequency (1P), not a control loop
artifact. Evidence:
  - motor_CMD (rows 20-23) oscillates at 24.41 Hz
  - gz (yaw rate, row 29) oscillates at 23.93 Hz ~ 24.41 Hz
  - 48.83 Hz = 2 x 24.41 Hz = blade-passing frequency (2P, 2-blade propeller)
  - 1465 RPM = 24.41 Hz x 60 is plausible for a small quadrotor at hover

Test: Check if rows 47-54 (motor_FL, esc_FL, motor_FR, esc_FR, motor_BL, esc_BL,
motor_BR, esc_BR) show a 24.41 Hz peak, and whether it tracks motor_CMD DC
(throttle). If the ESC telemetry oscillates at 24.41 Hz, it's the propeller RPM.
If the ESC telemetry is flat (DC only) or shows a different frequency, 24.41 Hz
might be a control loop artifact.

Also check: does the 24.41 Hz peak frequency shift with motor_CMD DC level?
If higher throttle -> higher frequency, it's RPM. If frequency is pegged regardless
of throttle, it's a control loop rate.

Run on a compute node via sbatch. NO model training, NO manifest.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# DronePropA channel rows (1-indexed)
ROW_TIME = 1
ROWS_MOTOR_CMD = [20, 21, 22, 23]       # throttle command [0,1]
ROW_BATTERY = 24
ROWS_GYRO_1 = [27, 28, 29]              # gyro_1 (roll, pitch, yaw rad/s)
# Rows 47-54: motor + ESC telemetry (Volt %)
ROWS_MOTOR_TELEM = [47, 49, 51, 53]     # motor_FL, motor_FR, motor_BL, motor_BR
ROWS_ESC_TELEM = [48, 50, 52, 54]       # esc_FL, esc_FR, esc_BL, esc_BR
MOTOR_NAMES = ["motor_FL", "motor_FR", "motor_BL", "motor_BR"]
ESC_NAMES = ["esc_FL", "esc_FR", "esc_BL", "esc_BR"]

F_MIN = 5.0
F_MAX = 200.0


def dronepropa_sampling_rate(time_s: np.ndarray) -> Tuple[float, float]:
    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(dt)), float(1.0 / np.mean(dt))


def welch_peaks(signal: np.ndarray, fs: float, nperseg: int = 4096,
                f_min: float = F_MIN, f_max: float = F_MAX,
                n_peaks: int = 5) -> List[Tuple[float, float, int]]:
    n = len(signal)
    if n < nperseg or not np.isfinite(fs) or fs <= 0:
        return []
    freqs, psd = welch(signal, fs=fs, nperseg=min(nperseg, n), window="hann")
    mask = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(mask):
        return []
    f_m = freqs[mask]
    p_m = psd[mask]
    idx_m = np.where(mask)[0]
    order = np.argsort(p_m)[::-1][:n_peaks]
    return [(float(f_m[i]), float(p_m[i]), int(idx_m[i])) for i in sorted(order)]


def signal_stats(signal: np.ndarray) -> Dict:
    return {
        "dc_mean": round(float(np.nanmean(signal)), 4),
        "std": round(float(np.nanstd(signal)), 4),
        "min": round(float(np.nanmin(signal)), 4),
        "max": round(float(np.nanmax(signal)), 4),
        "ac_std": round(float(np.nanstd(signal - np.nanmean(signal))), 6),
    }


def diagnose_motor_esc(source_dir: str, n_flights: int = 12) -> List[Dict]:
    """Check motor/ESC telemetry (rows 47-54) for propeller RPM peak."""
    from adapters.dronepropa import DronePropAAdapter
    os.environ["DRONEPROPA_SOURCE_DIR"] = source_dir
    adapter = DronePropAAdapter()
    flights = list(adapter.iter_samples())
    # Sample across speed (SP1, SP2) and fault type
    by_sp_fault = defaultdict(list)
    for s in flights:
        key = (s.get("speed", 1), s["fault_type"])
        by_sp_fault[key].append(s)
    sampled = []
    for key in sorted(by_sp_fault):
        for s in by_sp_fault[key][:2]:
            sampled.append(s)
            if len(sampled) >= n_flights:
                break
        if len(sampled) >= n_flights:
            break
    results = []
    for s in sampled:
        qd = s["qdrone_data"]
        time_s = qd[ROW_TIME - 1, :]
        _, fs = dronepropa_sampling_rate(time_s)
        motor_cmd = [qd[r - 1, :] for r in ROWS_MOTOR_CMD]
        motor_cmd_dc = [float(np.nanmean(m)) for m in motor_cmd]
        gyro_yaw = qd[ROWS_GYRO_1[2] - 1, :]  # yaw rate (row 29)
        motor_telem = [qd[r - 1, :] for r in ROWS_MOTOR_TELEM]
        esc_telem = [qd[r - 1, :] for r in ROWS_ESC_TELEM]
        # PSD peaks for each channel
        gyro_peaks = welch_peaks(gyro_yaw, fs, nperseg=4096, n_peaks=3)
        motor_cmd_peaks = [welch_peaks(m, fs, nperseg=4096, n_peaks=3) for m in motor_cmd]
        motor_telem_peaks = [welch_peaks(m, fs, nperseg=4096, n_peaks=3) for m in motor_telem]
        esc_telem_peaks = [welch_peaks(e, fs, nperseg=4096, n_peaks=3) for e in esc_telem]
        results.append({
            "group_id": s["group_id"],
            "fault_type": s["fault_type"],
            "speed": s.get("speed", 1),
            "n_samples": int(qd.shape[1]),
            "fs_hz": round(fs, 2),
            "motor_cmd_dc": [round(d, 4) for d in motor_cmd_dc],
            "motor_cmd_mean_dc": round(float(np.mean(motor_cmd_dc)), 4),
            "gyro_yaw_stats": signal_stats(gyro_yaw),
            "gyro_yaw_peaks": [(round(f, 3), round(p, 6), b) for f, p, b in gyro_peaks],
            "motor_cmd_peaks": [
                {"name": MOTOR_NAMES[i] if i < len(MOTOR_NAMES) else f"motor_cmd_{i}",
                 "dc": round(float(np.nanmean(motor_cmd[i])), 4),
                 "peaks": [(round(f, 3), round(p, 6), b) for f, p, b in motor_cmd_peaks[i]]}
                for i in range(4)
            ],
            "motor_telem_stats": [
                {"name": MOTOR_NAMES[i], **signal_stats(motor_telem[i]),
                 "peaks": [(round(f, 3), round(p, 6), b) for f, p, b in motor_telem_peaks[i]]}
                for i in range(4)
            ],
            "esc_telem_stats": [
                {"name": ESC_NAMES[i], **signal_stats(esc_telem[i]),
                 "peaks": [(round(f, 3), round(p, 6), b) for f, p, b in esc_telem_peaks[i]]}
                for i in range(4)
            ],
        })
    return results


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir-dronepropa", default=None)
    p.add_argument("--n-flights", type=int, default=12)
    p.add_argument("--out", default="results/uav_motor_esc_diagnosis.json")
    args = p.parse_args()

    print("=" * 80)
    print("MOTOR/ESC TELEMETRY DIAGNOSIS (rows 47-54)")
    print("Hypothesis: 24.41 Hz = propeller rotation frequency (1P)")
    print("=" * 80)

    dp_dir = args.source_dir_dronepropa or os.environ.get("DRONEPROPA_SOURCE_DIR")
    print(f"\nsource: {dp_dir}")
    results = diagnose_motor_esc(dp_dir, n_flights=args.n_flights) if dp_dir else []

    print(f"\n{'flight':<40} {'SP':>2} {'F':>2} {'mot_DC':>7} "
          f"{'gyro_yaw_pk':>12} {'motor_cmd_pk':>12} {'motor_telem_pk':>14} {'esc_telem_pk':>12}")
    print("-" * 110)
    for r in results:
        gyro_pk = r["gyro_yaw_peaks"][0][0] if r["gyro_yaw_peaks"] else float("nan")
        mcmd_pk = r["motor_cmd_peaks"][0]["peaks"][0][0] if r["motor_cmd_peaks"][0]["peaks"] else float("nan")
        mtl_pk = r["motor_telem_stats"][0]["peaks"][0][0] if r["motor_telem_stats"][0]["peaks"] else float("nan")
        esc_pk = r["esc_telem_stats"][0]["peaks"][0][0] if r["esc_telem_stats"][0]["peaks"] else float("nan")
        print(f"{r['group_id']:<40} {r['speed']:>2} {r['fault_type']:>2} "
              f"{r['motor_cmd_mean_dc']:>7.4f} {gyro_pk:>12.3f} {mcmd_pk:>12.3f} "
              f"{mtl_pk:>14.3f} {esc_pk:>12.3f}")

    # Detailed per-flight report
    print(f"\n\n{'='*80}")
    print("DETAILED REPORT (top-3 PSD peaks per channel, nperseg=4096)")
    print(f"{'='*80}")
    for r in results:
        print(f"\n--- {r['group_id']} (SP{r['speed']}, F{r['fault_type']}, "
              f"n={r['n_samples']}, fs={r['fs_hz']:.1f} Hz, motor_CMD DC={r['motor_cmd_mean_dc']:.4f}) ---")
        print(f"  motor_CMD DC: {r['motor_cmd_dc']}")
        print(f"  gyro_yaw (row 29): stats={r['gyro_yaw_stats']}")
        print(f"    peaks: {r['gyro_yaw_peaks']}")
        for mc in r["motor_cmd_peaks"]:
            print(f"  {mc['name']} (motor_CMD): DC={mc['dc']:.4f}, peaks={mc['peaks']}")
        for mt in r["motor_telem_stats"]:
            print(f"  {mt['name']} (motor telem): stats={ {k:v for k,v in mt.items() if k != 'peaks'} }")
            print(f"    peaks: {mt['peaks']}")
        for et in r["esc_telem_stats"]:
            print(f"  {et['name']} (ESC telem): stats={ {k:v for k,v in et.items() if k != 'peaks'} }")
            print(f"    peaks: {et['peaks']}")

    # Summary: does 24.41 Hz track throttle?
    print(f"\n\n{'='*80}")
    print("THROTTLE TRACKING SUMMARY")
    print(f"{'='*80}")
    if results:
        dcs = [r["motor_cmd_mean_dc"] for r in results]
        gyro_pks = [r["gyro_yaw_peaks"][0][0] if r["gyro_yaw_peaks"] else float("nan") for r in results]
        mcmd_pks = [r["motor_cmd_peaks"][0]["peaks"][0][0] if r["motor_cmd_peaks"][0]["peaks"] else float("nan") for r in results]
        mtl_pks = [r["motor_telem_stats"][0]["peaks"][0][0] if r["motor_telem_stats"][0]["peaks"] else float("nan") for r in results]
        esc_pks = [r["esc_telem_stats"][0]["peaks"][0][0] if r["esc_telem_stats"][0]["peaks"] else float("nan") for r in results]
        print(f"  motor_CMD DC range: {min(dcs):.4f} - {max(dcs):.4f} (spread {max(dcs)-min(dcs):.4f})")
        print(f"  gyro_yaw top peak:  {min(gyro_pks):.3f} - {max(gyro_pks):.3f} Hz (spread {max(gyro_pks)-min(gyro_pks):.3f})")
        print(f"  motor_CMD top peak: {min(mcmd_pks):.3f} - {max(mcmd_pks):.3f} Hz (spread {max(mcmd_pks)-min(mcmd_pks):.3f})")
        print(f"  motor_telem top peak: {min(mtl_pks):.3f} - {max(mtl_pks):.3f} Hz (spread {max(mtl_pks)-min(mtl_pks):.3f})")
        print(f"  esc_telem top peak:  {min(esc_pks):.3f} - {max(esc_pks):.3f} Hz (spread {max(esc_pks)-min(esc_pks):.3f})")
        # Correlation: does frequency track throttle?
        if np.std(dcs) > 1e-6 and np.std(gyro_pks) > 1e-6:
            corr_gyro = np.corrcoef(dcs, gyro_pks)[0, 1]
            print(f"  motor_CMD DC vs gyro_yaw peak freq: r={corr_gyro:.3f}")
        if np.std(dcs) > 1e-6 and np.std(mtl_pks) > 1e-6:
            corr_mtl = np.corrcoef(dcs, mtl_pks)[0, 1]
            print(f"  motor_CMD DC vs motor_telem peak freq: r={corr_mtl:.3f}")
        # Is 24.41 Hz present in motor/ESC telemetry?
        mtl_24 = [any(abs(p[0] - 24.41) < 1.0 for p in mt["peaks"]) for mt in [r["motor_telem_stats"][0] for r in results]]
        esc_24 = [any(abs(p[0] - 24.41) < 1.0 for p in et["peaks"]) for et in [r["esc_telem_stats"][0] for r in results]]
        print(f"  24.41 Hz in motor_telem (FL): {sum(mtl_24)}/{len(mtl_24)} flights")
        print(f"  24.41 Hz in esc_telem (FL):   {sum(esc_24)}/{len(esc_24)} flights")
        # Is 48.83 Hz (2P) present?
        mtl_48 = [any(abs(p[0] - 48.83) < 1.0 for p in mt["peaks"]) for mt in [r["motor_telem_stats"][0] for r in results]]
        print(f"  48.83 Hz (2P) in motor_telem (FL): {sum(mtl_48)}/{len(mtl_48)} flights")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"dronepropa": results, "n_flights": len(results)}, f, indent=2)
    print(f"\nDiagnostic JSON written -> {out_path}")
    print("\nThis is a DIAGNOSTIC artifact, NOT a result manifest. No model metrics.")


if __name__ == "__main__":
    main()
