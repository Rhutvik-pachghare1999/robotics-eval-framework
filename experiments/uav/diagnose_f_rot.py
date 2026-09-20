#!/usr/bin/env python3
"""
diagnose_f_rot.py — Diagnose why DronePropA f_rot is pegged at 48.83 Hz.

Investigates three hypotheses:
  H1: 48.83 Hz is a Welch bin artifact (bin 50 of nperseg=1024 at fs=1000).
  H2: 48.83 Hz is 50 Hz mains interference (present on all axes, healthy too).
  H3: 48.83 Hz is the real propeller rotation rate, constant because all
      sampled flights are SP1 (speed 1) — need to sample SP2 to see variation.

For each flight:
  - Top-5 Welch PSD peaks (Hz, power) for gyro magnitude, and per-axis (x/y/z).
  - Welch with nperseg = 512, 1024, 2048, 4096 (does the peak move or stay?).
  - Motor_CMD DC level per motor (rows 20-23) + std.
  - Motor_CMD Welch peak (if any) — motor_CMD is a throttle command [0,1],
    but its spectrum may reveal the control loop rate or motor dynamics.
  - Speed setting (SP1 vs SP2) — does f_rot vary with speed?

For TII (5 missions): top-5 Welch PSD peaks per-axis + nperseg sweep.

Run on a compute node via sbatch. Prints a report; writes JSON summary.
NO model training, NO result manifest — this is a diagnostic gate.
"""
from __future__ import annotations

import argparse
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

# DronePropA channel rows (1-indexed, per adapters/dronepropa.py)
ROW_TIME = 1
ROWS_MOTOR_CMD = [20, 21, 22, 23]   # motor_CMD (FL, FR, BL, BR), [0,1] throttle
ROW_BATTERY = 24
ROWS_GYRO_1 = [27, 28, 29]          # gyro_1 (roll, pitch, yaw rad/s)
ROWS_ACCEL_1 = [30, 31, 32]         # accel_1 (x, y, z m/s^2)

F_ROT_MIN_HZ = 10.0
F_ROT_MAX_HZ = 500.0


def dronepropa_sampling_rate(time_s: np.ndarray) -> Tuple[float, float]:
    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(dt)), float(1.0 / np.mean(dt))


def tii_sampling_rate(ts_us: np.ndarray, gyro_dt_us: float) -> Tuple[float, float]:
    dt_us = np.diff(ts_us.astype(float))
    dt_us = dt_us[np.isfinite(dt_us) & (dt_us > 0)]
    if dt_us.size == 0:
        return float(gyro_dt_us) / 1e6, float(1e6 / gyro_dt_us)
    return float(np.mean(dt_us)) / 1e6, float(1e6 / np.mean(dt_us))


def welch_peaks(signal: np.ndarray, fs: float, nperseg: int,
                f_min: float = F_ROT_MIN_HZ, f_max: float = F_ROT_MAX_HZ,
                n_peaks: int = 5) -> List[Tuple[float, float, int]]:
    """Top-n Welch PSD peaks in [f_min, f_max]. Returns [(freq_hz, power, bin_idx), ...]."""
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


def diagnose_dronepropa(source_dir: str, n_flights: int = 10) -> List[Dict]:
    """Sample flights across SP1/SP2 and fault types; top-5 peaks + nperseg sweep."""
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
    # Take 1-2 per (speed, fault) combo, up to n_flights total
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
        gx = qd[ROWS_GYRO_1[0] - 1, :]
        gy = qd[ROWS_GYRO_1[1] - 1, :]
        gz = qd[ROWS_GYRO_1[2] - 1, :]
        gmag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        ax = qd[ROWS_ACCEL_1[0] - 1, :]
        ay = qd[ROWS_ACCEL_1[1] - 1, :]
        az = qd[ROWS_ACCEL_1[2] - 1, :]
        amag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
        motors = [qd[r - 1, :] for r in ROWS_MOTOR_CMD]
        motor_dc = [float(np.nanmean(m)) for m in motors]
        motor_std = [float(np.nanstd(m)) for m in motors]
        # nperseg sweep on gyro magnitude
        sweep = {}
        for nps in [512, 1024, 2048, 4096]:
            peaks = welch_peaks(gmag, fs, nps, n_peaks=3)
            sweep[nps] = {
                "bin_width_hz": round(fs / nps, 4),
                "peaks": [(round(f, 3), round(p, 6), b) for f, p, b in peaks],
            }
        # Per-axis top-3 peaks (nperseg=2048 for good resolution)
        per_axis = {}
        for name, sig in [("gx", gx), ("gy", gy), ("gz", gz),
                          ("ax", ax), ("ay", ay), ("az", az),
                          ("gmag", gmag), ("amag", amag)]:
            per_axis[name] = [(round(f, 3), round(p, 6), b)
                              for f, p, b in welch_peaks(sig, fs, 2048, n_peaks=3)]
        # Motor_CMD top-3 peaks (nperseg=2048)
        motor_peaks = {}
        for i, m in enumerate(motors):
            motor_peaks[f"motor_{i}"] = [(round(f, 3), round(p, 6), b)
                                         for f, p, b in welch_peaks(m, fs, 2048, n_peaks=3)]
        # Check if 50 Hz / 48.83 Hz is mains or bin artifact
        bin_50_at_1024 = 50 * (1024 / fs)  # bin index for 50 Hz at nperseg=1024
        freq_at_bin_50_1024 = 50 * (fs / 1024)  # freq at bin 50 at nperseg=1024
        results.append({
            "group_id": s["group_id"],
            "fault_type": s["fault_type"],
            "severity": s.get("severity", 0),
            "speed": s.get("speed", 1),
            "trajectory": s.get("trajectory", 1),
            "n_samples": int(qd.shape[1]),
            "fs_hz": round(fs, 2),
            "duration_s": round(float(time_s[-1] - time_s[0]), 2),
            "motor_cmd_dc": [round(d, 4) for d in motor_dc],
            "motor_cmd_std": [round(s_, 4) for s_ in motor_std],
            "bin_50_freq_at_nperseg1024": round(freq_at_bin_50_1024, 4),
            "nperseg_sweep": {str(k): v for k, v in sweep.items()},
            "per_axis_top3_peaks_2048": per_axis,
            "motor_cmd_top3_peaks_2048": motor_peaks,
        })
    return results


def diagnose_tii(source_dir: str, n_missions: int = 5) -> List[Dict]:
    """Sample TII missions; top-5 peaks + nperseg sweep."""
    root = Path(source_dir)
    if not (root / "0").exists():
        root = root / "Dataset"
    missions = []
    for cls_dir in sorted(root.iterdir()):
        if cls_dir.is_dir() and cls_dir.name.isdigit():
            for mdir in sorted(cls_dir.iterdir()):
                if mdir.is_dir():
                    missions.append(mdir)
    by_class = defaultdict(list)
    for m in missions:
        cls = int(m.parent.name)
        by_class[cls].append(m)
    sampled = []
    for cls in sorted(by_class):
        for m in by_class[cls][: max(1, n_missions // 5)]:
            sampled.append(m)
    results = []
    for mdir in sampled:
        sc_files = sorted(mdir.glob("*-SensorCombined.jsonl"))
        if not sc_files:
            continue
        ts_us, gx, gy, gz, ax, ay, az, idt = [], [], [], [], [], [], [], []
        for line in open(sc_files[0]):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ts_us.append(obj["timestamp"])
            g = obj.get("gyro_rad", [0, 0, 0])
            a = obj.get("accelerometer_m_s2", [0, 0, 0])
            gx.append(g[0]); gy.append(g[1]); gz.append(g[2])
            ax.append(a[0]); ay.append(a[1]); az.append(a[2])
            idt.append(obj.get("gyro_integral_dt", 5000))
        if len(ts_us) < 100:
            continue
        ts_us = np.array(ts_us, dtype=float)
        gx = np.array(gx); gy = np.array(gy); gz = np.array(gz)
        ax = np.array(ax); ay = np.array(ay); az = np.array(az)
        idt_arr = np.array(idt, dtype=float)
        _, fs = tii_sampling_rate(ts_us, float(np.median(idt_arr)))
        gmag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        amag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
        sweep = {}
        for nps in [256, 512, 1024, 2048]:
            peaks = welch_peaks(gmag, fs, nps, n_peaks=3)
            sweep[nps] = {
                "bin_width_hz": round(fs / nps, 4),
                "peaks": [(round(f, 3), round(p, 6), b) for f, p, b in peaks],
            }
        per_axis = {}
        for name, sig in [("gx", gx), ("gy", gy), ("gz", gz),
                          ("ax", ax), ("ay", ay), ("az", az),
                          ("gmag", gmag), ("amag", amag)]:
            per_axis[name] = [(round(f, 3), round(p, 6), b)
                              for f, p, b in welch_peaks(sig, fs, 512, n_peaks=3)]
        results.append({
            "group_id": mdir.name,
            "class": int(mdir.parent.name),
            "n_samples": int(len(ts_us)),
            "fs_hz": round(fs, 2),
            "duration_s": round(float(ts_us[-1] - ts_us[0]) / 1e6, 2),
            "nperseg_sweep": {str(k): v for k, v in sweep.items()},
            "per_axis_top3_peaks_512": per_axis,
        })
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir-dronepropa", default=None)
    p.add_argument("--source-dir-tii", default=None)
    p.add_argument("--n-flights", type=int, default=10)
    p.add_argument("--out", default="results/uav_f_rot_diagnosis.json")
    args = p.parse_args()

    print("=" * 80)
    print("f_rot DIAGNOSIS — why is DronePropA f_rot pegged at 48.83 Hz?")
    print("=" * 80)

    # ── DronePropA ──
    dp_dir = args.source_dir_dronepropa or os.environ.get("DRONEPROPA_SOURCE_DIR")
    print("\n## DronePropA diagnosis (sampled across SP1/SP2 × fault types)")
    print(f"   source: {dp_dir}")
    dp_results = diagnose_dronepropa(dp_dir, n_flights=args.n_flights) if dp_dir else []
    for r in dp_results:
        print(f"\n   --- {r['group_id']} (fault={r['fault_type']}, speed=SP{r['speed']}, "
              f"traj=t{r['trajectory']}, n={r['n_samples']}, fs={r['fs_hz']:.1f} Hz, "
              f"dur={r['duration_s']:.1f}s) ---")
        print(f"   motor_CMD DC:  {r['motor_cmd_dc']}  std: {r['motor_cmd_std']}")
        print(f"   bin 50 @ nperseg=1024 -> {r['bin_50_freq_at_nperseg1024']:.4f} Hz "
              f"(is 48.83 = bin 50? {'YES' if abs(r['bin_50_freq_at_nperseg1024'] - 48.83) < 0.1 else 'no'})")
        print(f"   nperseg sweep (gyro magnitude, top-3 peaks):")
        for nps in [512, 1024, 2048, 4096]:
            s = r["nperseg_sweep"][str(nps)]
            pk = s["peaks"]
            pk_str = ", ".join(f"{f:.2f}Hz(p={p:.2e},bin={b})" for f, p, b in pk)
            print(f"     nperseg={nps:4d} (binw={s['bin_width_hz']:.4f} Hz): {pk_str}")
        print(f"   per-axis top-3 peaks (nperseg=2048):")
        for axis in ["gx", "gy", "gz", "gmag"]:
            pk = r["per_axis_top3_peaks_2048"][axis]
            pk_str = ", ".join(f"{f:.2f}Hz(p={p:.2e})" for f, p, b in pk)
            print(f"     {axis}: {pk_str}")
        print(f"   motor_CMD top-3 peaks (nperseg=2048):")
        for k, pk in r["motor_cmd_top3_peaks_2048"].items():
            pk_str = ", ".join(f"{f:.2f}Hz(p={p:.2e})" for f, p, b in pk)
            print(f"     {k}: {pk_str}")

    # ── TII ──
    tii_dir = args.source_dir_tii or os.environ.get("TII_SOURCE_DIR")
    print("\n\n## TII diagnosis (5 missions across classes)")
    print(f"   source: {tii_dir}")
    tii_results = diagnose_tii(tii_dir, n_missions=5) if tii_dir else []
    for r in tii_results:
        print(f"\n   --- {r['group_id']} (class={r['class']}, n={r['n_samples']}, "
              f"fs={r['fs_hz']:.1f} Hz, dur={r['duration_s']:.1f}s) ---")
        print(f"   nperseg sweep (gyro magnitude, top-3 peaks):")
        for nps in [256, 512, 1024, 2048]:
            s = r["nperseg_sweep"][str(nps)]
            pk = s["peaks"]
            pk_str = ", ".join(f"{f:.2f}Hz(p={p:.2e},bin={b})" for f, p, b in pk)
            print(f"     nperseg={nps:4d} (binw={s['bin_width_hz']:.4f} Hz): {pk_str}")
        print(f"   per-axis top-3 peaks (nperseg=512):")
        for axis in ["gx", "gy", "gz", "gmag"]:
            pk = r["per_axis_top3_peaks_512"][axis]
            pk_str = ", ".join(f"{f:.2f}Hz(p={p:.2e})" for f, p, b in pk)
            print(f"     {axis}: {pk_str}")

    # ── Summary ──
    print("\n\n## Summary")
    if dp_results:
        print(f"\nDronePropA: {len(dp_results)} flights (SP1+SP2 × fault types)")
        speeds = set(r["speed"] for r in dp_results)
        print(f"   speeds sampled: {sorted(speeds)}")
        # Does f_rot (top peak) vary with nperseg?
        for nps in [512, 1024, 2048, 4096]:
            top_peaks = [r["nperseg_sweep"][str(nps)]["peaks"][0][0]
                         for r in dp_results
                         if r["nperseg_sweep"][str(nps)]["peaks"]]
            print(f"   nperseg={nps}: top-peak freqs = {[round(f, 2) for f in top_peaks]}")
        # Per-axis: is 50 Hz on all axes (mains) or just 1-2 (rotation)?
        print(f"   per-axis top peak (nperseg=2048):")
        for axis in ["gx", "gy", "gz", "gmag"]:
            top = [r["per_axis_top3_peaks_2048"][axis][0][0] for r in dp_results
                   if r["per_axis_top3_peaks_2048"][axis]]
            print(f"     {axis}: {[round(f, 2) for f in top]}")
        # Motor_CMD DC variation
        dcs = [np.mean(r["motor_cmd_dc"]) for r in dp_results]
        print(f"   motor_CMD DC mean: per-flight avg = {[round(d, 3) for d in dcs]}")
        print(f"   motor_CMD DC range: {min(dcs):.4f} - {max(dcs):.4f} "
              f"(spread {max(dcs) - min(dcs):.4f})")
    if tii_results:
        print(f"\nTII: {len(tii_results)} missions")
        for nps in [256, 512, 1024, 2048]:
            top_peaks = [r["nperseg_sweep"][str(nps)]["peaks"][0][0]
                         for r in tii_results
                         if r["nperseg_sweep"][str(nps)]["peaks"]]
            print(f"   nperseg={nps}: top-peak freqs = {[round(f, 2) for f in top_peaks]}")

    # ── Write JSON ──
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "dronepropa": dp_results,
            "tii": tii_results,
            "n_flights_dronepropa": len(dp_results),
            "n_missions_tii": len(tii_results),
        }, f, indent=2)
    print(f"\nDiagnostic JSON written -> {out_path}")
    print("\nThis is a DIAGNOSTIC artifact, NOT a result manifest. No model metrics.")


if __name__ == "__main__":
    main()
