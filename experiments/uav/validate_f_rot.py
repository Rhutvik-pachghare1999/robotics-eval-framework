#!/usr/bin/env python3
"""
validate_f_rot.py — STEP-3 gate: validate f_rot estimation + sampling rates.

Measures (NO model training, NO manifest — this is a validation gate):
  1. Actual IMU sampling rate of BOTH datasets (from timestamps).
  2. f_rot estimation via cepstrum + harmonic-product-spectrum (HPS) on gyro
     magnitude — robust to harmonics/BPF.
  3. Cross-check on DronePropA: gyro-estimated f_rot vs motor_CMD-derived
     rotation rate (spectral peak + DC-throttle proxy).
  4. Sanity report: ~10 flights per dataset, f_rot stability, Nyquist check.

Run on a compute node via sbatch. Prints a report to stdout; writes a
JSON summary to results/uav_f_rot_validation.json (NOT a result manifest —
a validation artifact, no model metrics).

Usage:
    python experiments/uav/validate_f_rot.py [--source-dir-dronepropa DIR] [--source-dir-tii DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# DronePropA channel rows (1-indexed, per adapters/dronepropa.py)
ROW_TIME = 1          # time_s
ROWS_MOTOR_CMD = [20, 21, 22, 23]   # motor_CMD (FL, FR, BL, BR)
ROW_BATTERY = 24      # battery voltage
ROWS_GYRO_1 = [27, 28, 29]   # gyro_1 (roll, pitch, yaw rad/s)
ROWS_ACCEL_1 = [30, 31, 32]  # accel_1 (x, y, z m/s^2)

# f_rot search band (consumer drone propeller RPM: 600-30000 RPM -> 10-500 Hz)
F_ROT_MIN_HZ = 10.0
F_ROT_MAX_HZ = 500.0


# ── Sampling rate ────────────────────────────────────────────────────────────

def dronepropa_sampling_rate(time_s: np.ndarray) -> Tuple[float, float]:
    """Return (mean_dt_s, mean_fs_hz) from DronePropA row-0 timestamps."""
    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.shape[0] == 0:
        return float("nan"), float("nan")
    mean_dt = float(np.median(dt))
    return mean_dt, 1.0 / mean_dt


def tii_sampling_rate(timestamps_us: np.ndarray, integral_dt_us: float) -> Tuple[float, float]:
    """Return (mean_dt_s, mean_fs_hz) from TII SensorCombined timestamps (µs)."""
    dt_us = np.diff(timestamps_us)
    dt_us = dt_us[np.isfinite(dt_us) & (dt_us > 0)]
    if dt_us.shape[0] == 0:
        mean_dt_us = integral_dt_us
    else:
        mean_dt_us = float(np.median(dt_us))
    mean_dt_s = mean_dt_us / 1e6
    return mean_dt_s, 1.0 / mean_dt_s


# ── f_rot estimation: cepstrum + HPS ─────────────────────────────────────────

def cepstrum_f0(signal: np.ndarray, fs: float, f_min: float = F_ROT_MIN_HZ,
                f_max: float = F_ROT_MAX_HZ) -> float:
    """Estimate fundamental frequency via cepstrum (robust to harmonics).

    cepstrum = IFFT(log(|FFT(signal)|))
    Peak in cepstrum (quefrency) -> f0 = fs / peak_quefrency
    """
    n = len(signal)
    if n < 64 or not np.isfinite(fs) or fs <= 0:
        return float("nan")
    spectrum = np.fft.rfft(signal * np.hanning(n))
    log_mag = np.log(np.abs(spectrum) + 1e-12)
    cepstrum = np.fft.irfft(log_mag)
    # quefrency bin -> period (samples); f0 = fs / period
    # search in [fs/f_max, fs/f_min] samples
    min_period = max(1, int(np.ceil(fs / f_max)))
    max_period = min(n // 2, int(np.floor(fs / f_min)))
    if max_period <= min_period:
        return float("nan")
    peak_idx = min_period + int(np.argmax(cepstrum[min_period:max_period + 1]))
    if peak_idx <= 0:
        return float("nan")
    return float(fs / peak_idx)


def hps_f0(signal: np.ndarray, fs: float, n_harmonics: int = 3,
           f_min: float = F_ROT_MIN_HZ, f_max: float = F_ROT_MAX_HZ) -> float:
    """Estimate fundamental frequency via Harmonic Product Spectrum.

    HPS(f) = |X(f)| * |X(2f)| * |X(3f)| ... (downsampled + multiplied)
    Reinforces the fundamental, suppresses harmonics.
    """
    n = len(signal)
    if n < 64 or not np.isfinite(fs) or fs <= 0:
        return float("nan")
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    hps = spectrum.copy()
    for h in range(2, n_harmonics + 1):
        ds = spectrum[::h][:len(hps)]
        hps[:len(ds)] *= ds
    mask = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(mask):
        return float("nan")
    candidates = np.where(mask)[0]
    peak = candidates[int(np.argmax(hps[mask]))]
    return float(freqs[peak])


def welch_f0(signal: np.ndarray, fs: float, f_min: float = F_ROT_MIN_HZ,
             f_max: float = F_ROT_MAX_HZ, nperseg: int = 1024) -> float:
    """Estimate f_rot via Welch PSD peak (robust to noise vs raw FFT).

    Returns the frequency of the dominant peak in [f_min, f_max].
    """
    n = len(signal)
    if n < nperseg or not np.isfinite(fs) or fs <= 0:
        return float("nan")
    freqs, psd = welch(signal, fs=fs, nperseg=min(nperseg, n), window="hann")
    mask = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(mask):
        return float("nan")
    candidates = np.where(mask)[0]
    peak = candidates[int(np.argmax(psd[mask]))]
    return float(freqs[peak])


def top_n_psd_peaks(signal: np.ndarray, fs: float, n: int = 5,
                     f_min: float = F_ROT_MIN_HZ,
                     f_max: float = F_ROT_MAX_HZ,
                     nperseg: int = 1024) -> List[Tuple[float, float]]:
    """Return top-n (freq, power) peaks from Welch PSD in [f_min, f_max]."""
    n_samp = len(signal)
    if n_samp < nperseg or not np.isfinite(fs) or fs <= 0:
        return []
    freqs, psd = welch(signal, fs=fs, nperseg=min(nperseg, n_samp),
                        window="hann")
    mask = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(mask):
        return []
    f_masked = freqs[mask]
    psd_masked = psd[mask]
    idx = np.argsort(psd_masked)[::-1][:n]
    return [(float(f_masked[i]), float(psd_masked[i])) for i in sorted(idx)]


def gyro_magnitude(rows: List[np.ndarray]) -> np.ndarray:
    """Stack gyro rows into (T, 3) and return magnitude sqrt(gx^2+gy^2+gz^2)."""
    arr = np.stack(rows, axis=1)  # (T, 3)
    return np.sqrt(np.sum(arr ** 2, axis=1))


def motor_cmd_mean(motor_rows: List[np.ndarray]) -> np.ndarray:
    """Mean across 4 motor_CMD channels (T,)."""
    arr = np.stack(motor_rows, axis=1)  # (T, 4)
    return np.mean(arr, axis=1)


# ── DronePropA validation ────────────────────────────────────────────────────

def validate_dronepropa(source_dir: str, n_flights: int = 10) -> List[Dict]:
    """Sample n_flights from DronePropA, estimate f_rot from gyro_1 (rows 27-29).

    Cross-check: motor_CMD (rows 20-23) is a throttle COMMAND [0,1], not a feedback
    signal — its spectrum does NOT give f_rot. Instead, validate by checking that
    motor_CMD DC level (throttle) correlates with f_rot across flights (higher
    throttle -> higher RPM -> higher f_rot).
    """
    from adapters.dronepropa import DronePropAAdapter
    os.environ["DRONEPROPA_SOURCE_DIR"] = source_dir
    adapter = DronePropAAdapter()
    results = []
    flights = list(adapter.iter_samples())
    # sample evenly across fault types
    by_fault = defaultdict(list)
    for s in flights:
        by_fault[s["fault_type"]].append(s)
    sampled = []
    for ft in sorted(by_fault):
        for s in by_fault[ft][: max(1, n_flights // 4)]:
            sampled.append(s)
    for s in sampled:
        qd = s["qdrone_data"]
        time_s = qd[ROW_TIME - 1, :]
        gyro = gyro_magnitude([qd[r - 1, :] for r in ROWS_GYRO_1])
        motor = motor_cmd_mean([qd[r - 1, :] for r in ROWS_MOTOR_CMD])
        battery = float(np.nanmean(qd[ROW_BATTERY - 1, :]))
        mean_dt, fs = dronepropa_sampling_rate(time_s)
        f_rot_cep = cepstrum_f0(gyro, fs)
        f_rot_hps = hps_f0(gyro, fs)
        f_rot_welch = welch_f0(gyro, fs)
        top_peaks = top_n_psd_peaks(gyro, fs, n=3)
        motor_dc = float(np.nanmean(motor))
        motor_std = float(np.nanstd(motor))
        nyquist = fs / 2.0
        max_order = nyquist / f_rot_cep if f_rot_cep > 0 else float("nan")
        results.append({
            "group_id": s["group_id"],
            "fault_type": s["fault_type"],
            "severity": s["severity"],
            "n_samples": int(qd.shape[1]),
            "fs_hz": round(fs, 2),
            "dt_ms": round(mean_dt * 1000, 3),
            "duration_s": round(float(time_s[-1] - time_s[0]), 2),
            "battery_v": round(battery, 3),
            "motor_cmd_dc": round(motor_dc, 4),
            "motor_cmd_std": round(motor_std, 4),
            "f_rot_gyro_cepstrum_hz": round(f_rot_cep, 2),
            "f_rot_gyro_hps_hz": round(f_rot_hps, 2),
            "f_rot_gyro_welch_hz": round(f_rot_welch, 2),
            "top3_psd_peaks_hz": [(round(f, 2), round(p, 6)) for f, p in top_peaks],
            "nyquist_hz": round(nyquist, 2),
            "max_order_at_nyquist": round(max_order, 1),
        })
    return results


# ── TII validation ───────────────────────────────────────────────────────────

def load_tii_sensor_combined(mission_dir: Path) -> Dict:
    """Load SensorCombined.jsonl from a TII mission directory."""
    sc_files = sorted(mission_dir.glob("*-SensorCombined.jsonl"))
    if not sc_files:
        return {}
    f = sc_files[0]
    ts, gx, gy, gz, ax, ay, az, idt = [], [], [], [], [], [], [], []
    for line in open(f):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        ts.append(d["timestamp"])
        g = d["gyro_rad"]
        a = d["accelerometer_m_s2"]
        gx.append(g[0]); gy.append(g[1]); gz.append(g[2])
        ax.append(a[0]); ay.append(a[1]); az.append(a[2])
        idt.append(d.get("gyro_integral_dt", 0))
    if not ts:
        return {}
    return {
        "timestamp_us": np.array(ts),
        "gyro": np.stack([gx, gy, gz], axis=1),   # (T, 3)
        "accel": np.stack([ax, ay, az], axis=1),  # (T, 3)
        "gyro_integral_dt_us": float(np.median(idt)),
    }


def validate_tii(source_dir: str, n_missions: int = 10) -> List[Dict]:
    """Sample n_missions from TII, estimate f_rot from SensorCombined gyro."""
    # source_dir may be .../repo/Dataset (has class dirs 0-4) or .../repo (has Dataset/)
    root = Path(source_dir)
    if not (root / "0").is_dir() and (root / "Dataset").is_dir():
        root = root / "Dataset"
    results = []
    by_class = defaultdict(list)
    for cls in sorted(root.iterdir()):
        if cls.is_dir() and cls.name.isdigit():
            for mission in sorted(cls.iterdir()):
                if mission.is_dir():
                    by_class[int(cls.name)].append(mission)
    for cls in sorted(by_class):
        for mission in by_class[cls][: max(1, n_missions // 5)]:
            sc = load_tii_sensor_combined(mission)
            if not sc:
                continue
            ts_us = sc["timestamp_us"]
            gyro = np.sqrt(np.sum(sc["gyro"] ** 2, axis=1))  # magnitude
            mean_dt, fs = tii_sampling_rate(ts_us, sc["gyro_integral_dt_us"])
            f_rot_cep = cepstrum_f0(gyro, fs)
            f_rot_hps = hps_f0(gyro, fs)
            f_rot_welch = welch_f0(gyro, fs)
            top_peaks = top_n_psd_peaks(gyro, fs, n=3)
            nyquist = fs / 2.0
            max_order = nyquist / f_rot_cep if f_rot_cep > 0 else float("nan")
            results.append({
                "group_id": mission.name,
                "class": cls,
                "n_samples": int(len(ts_us)),
                "fs_hz": round(fs, 2),
                "dt_ms": round(mean_dt * 1000, 3),
                "duration_s": round(float(ts_us[-1] - ts_us[0]) / 1e6, 2),
                "gyro_integral_dt_us": round(sc["gyro_integral_dt_us"], 1),
                "f_rot_gyro_cepstrum_hz": round(f_rot_cep, 2),
                "f_rot_gyro_hps_hz": round(f_rot_hps, 2),
                "f_rot_gyro_welch_hz": round(f_rot_welch, 2),
                "top3_psd_peaks_hz": [(round(f, 2), round(p, 6)) for f, p in top_peaks],
                "nyquist_hz": round(nyquist, 2),
                "max_order_at_nyquist": round(max_order, 1),
            })
    return results


# ── Report ───────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir-dronepropa", default=None)
    ap.add_argument("--source-dir-tii", default=None)
    ap.add_argument("--n-flights", type=int, default=10)
    ap.add_argument("--output", default=str(ROOT / "results" / "uav_f_rot_validation.json"))
    args = ap.parse_args(argv)

    print("=" * 80)
    print("f_rot VALIDATION REPORT (STEP-3 gate)")
    print("=" * 80)

    # ── DronePropA ──
    dp_dir = args.source_dir_dronepropa or os.environ.get("DRONEPROPA_SOURCE_DIR")
    print("\n## DronePropA (gyro_1 rows 27-29, motor_CMD rows 20-23, battery row 24)")
    print(f"   source: {dp_dir}")
    dp_results = validate_dronepropa(dp_dir, n_flights=args.n_flights) if dp_dir else []
    if dp_results:
        fs_vals = [r["fs_hz"] for r in dp_results]
        print(f"   sampling rate: mean={np.mean(fs_vals):.2f} Hz, "
              f"min={np.min(fs_vals):.2f}, max={np.max(fs_vals):.2f}")
        print(f"   {'group_id':<40} {'fault':>5} {'sev':>3} {'fs_Hz':>7} "
              f"{'dur_s':>6} {'bat_V':>6} {'mot_DC':>7} {'mot_std':>7} "
              f"{'f_cep':>7} {'f_hps':>7} {'f_welch':>8} {'max_ord':>7} "
              f"{'top3_psd_peaks_Hz':<30}")
        for r in dp_results:
            peaks_str = ", ".join(f"{f:.1f}({p:.2e})" for f, p in r["top3_psd_peaks_hz"][:3])
            print(f"   {r['group_id']:<40} {r['fault_type']:>5} {r['severity']:>3} "
                  f"{r['fs_hz']:>7.2f} {r['duration_s']:>6.1f} {r['battery_v']:>6.2f} "
                  f"{r['motor_cmd_dc']:>7.4f} {r['motor_cmd_std']:>7.4f} "
                  f"{r['f_rot_gyro_cepstrum_hz']:>7.2f} {r['f_rot_gyro_hps_hz']:>7.2f} "
                  f"{r['f_rot_gyro_welch_hz']:>8.2f} {r['max_order_at_nyquist']:>7.1f} "
                  f"{peaks_str:<30}")
        # estimator agreement
        cep_vals = np.array([r["f_rot_gyro_cepstrum_hz"] for r in dp_results])
        hps_vals = np.array([r["f_rot_gyro_hps_hz"] for r in dp_results])
        welch_vals = np.array([r["f_rot_gyro_welch_hz"] for r in dp_results])
        motor_dc = np.array([r["motor_cmd_dc"] for r in dp_results])
        plausible = np.sum((cep_vals >= 10) & (cep_vals <= 500))
        print(f"\n   f_rot estimator agreement (gyro):")
        print(f"     cepstrum vs HPS:     mean |diff|={np.mean(np.abs(cep_vals-hps_vals)):.2f} Hz, "
              f"median |diff|={np.median(np.abs(cep_vals-hps_vals)):.2f} Hz")
        print(f"     cepstrum vs Welch:   mean |diff|={np.mean(np.abs(cep_vals-welch_vals)):.2f} Hz, "
              f"median |diff|={np.median(np.abs(cep_vals-welch_vals)):.2f} Hz")
        print(f"     HPS vs Welch:        mean |diff|={np.mean(np.abs(hps_vals-welch_vals)):.2f} Hz, "
              f"median |diff|={np.median(np.abs(hps_vals-welch_vals)):.2f} Hz")
        print(f"   f_rot in [10, 500] Hz (plausible): {plausible}/{len(dp_results)}")
        # motor_CMD DC vs f_rot correlation (throttle should correlate with RPM -> f_rot)
        if np.std(motor_dc) > 1e-6 and np.std(cep_vals) > 1e-6:
            corr = np.corrcoef(motor_dc, cep_vals)[0, 1]
            print(f"   motor_CMD DC vs f_rot_gyro correlation: r={corr:.3f} "
                  f"(positive expected: higher throttle -> higher RPM -> higher f_rot)")
        # f_rot stability across flights (hover should be steady)
        print(f"   f_rot_gyro_cepstrum: mean={np.mean(cep_vals):.2f} Hz, "
              f"std={np.std(cep_vals):.2f} Hz, median={np.median(cep_vals):.2f} Hz")
        print(f"   f_rot_gyro_welch:    mean={np.mean(welch_vals):.2f} Hz, "
              f"std={np.std(welch_vals):.2f} Hz, median={np.median(welch_vals):.2f} Hz")

    # ── TII ──
    tii_dir = args.source_dir_tii or os.environ.get("TII_SOURCE_DIR")
    print("\n## TII (SensorCombined gyro_rad[3], accelerometer_m_s2[3])")
    print(f"   source: {tii_dir}")
    tii_results = validate_tii(tii_dir, n_missions=args.n_flights) if tii_dir else []
    if tii_results:
        fs_vals = [r["fs_hz"] for r in tii_results]
        print(f"   sampling rate: mean={np.mean(fs_vals):.2f} Hz, "
              f"min={np.min(fs_vals):.2f}, max={np.max(fs_vals):.2f}")
        print(f"   {'group_id':<40} {'class':>5} {'fs_Hz':>7} "
              f"{'dur_s':>6} {'idt_us':>7} "
              f"{'f_cep':>7} {'f_hps':>7} {'f_welch':>8} {'max_ord':>7} "
              f"{'top3_psd_peaks_Hz':<30}")
        for r in tii_results:
            peaks_str = ", ".join(f"{f:.1f}({p:.2e})" for f, p in r["top3_psd_peaks_hz"][:3])
            print(f"   {r['group_id']:<40} {r['class']:>5} {r['fs_hz']:>7.2f} "
                  f"{r['duration_s']:>6.1f} {r['gyro_integral_dt_us']:>7.1f} "
                  f"{r['f_rot_gyro_cepstrum_hz']:>7.2f} {r['f_rot_gyro_hps_hz']:>7.2f} "
                  f"{r['f_rot_gyro_welch_hz']:>8.2f} {r['max_order_at_nyquist']:>7.1f} "
                  f"{peaks_str:<30}")
        cep_vals_t = np.array([r["f_rot_gyro_cepstrum_hz"] for r in tii_results])
        hps_vals_t = np.array([r["f_rot_gyro_hps_hz"] for r in tii_results])
        welch_vals_t = np.array([r["f_rot_gyro_welch_hz"] for r in tii_results])
        plausible_t = np.sum((cep_vals_t >= 10) & (cep_vals_t <= 500))
        print(f"\n   f_rot estimator agreement (gyro):")
        print(f"     cepstrum vs HPS:     mean |diff|={np.mean(np.abs(cep_vals_t-hps_vals_t)):.2f} Hz, "
              f"median |diff|={np.median(np.abs(cep_vals_t-hps_vals_t)):.2f} Hz")
        print(f"     cepstrum vs Welch:   mean |diff|={np.mean(np.abs(cep_vals_t-welch_vals_t)):.2f} Hz, "
              f"median |diff|={np.median(np.abs(cep_vals_t-welch_vals_t)):.2f} Hz")
        print(f"     HPS vs Welch:        mean |diff|={np.mean(np.abs(hps_vals_t-welch_vals_t)):.2f} Hz, "
              f"median |diff|={np.median(np.abs(hps_vals_t-welch_vals_t)):.2f} Hz")
        print(f"   f_rot in [10, 500] Hz (plausible): {plausible_t}/{len(tii_results)}")
        print(f"   f_rot_gyro_cepstrum: mean={np.mean(cep_vals_t):.2f} Hz, "
              f"std={np.std(cep_vals_t):.2f} Hz, median={np.median(cep_vals_t):.2f} Hz")
        print(f"   f_rot_gyro_welch:    mean={np.mean(welch_vals_t):.2f} Hz, "
              f"std={np.std(welch_vals_t):.2f} Hz, median={np.median(welch_vals_t):.2f} Hz")

    # ── Nyquist / order-band check ──
    print("\n## Nyquist check (max observable order at f_rot)")
    if dp_results:
        dp_fs = np.mean([r["fs_hz"] for r in dp_results])
        dp_frot = np.median([r["f_rot_gyro_welch_hz"] for r in dp_results])
        dp_nyq = dp_fs / 2
        print(f"   DronePropA: fs={dp_fs:.1f} Hz, Nyquist={dp_nyq:.1f} Hz, "
              f"median f_rot(welch)={dp_frot:.1f} Hz, max order={dp_nyq/dp_frot:.1f}P")
    if tii_results:
        tii_fs = np.mean([r["fs_hz"] for r in tii_results])
        tii_frot = np.median([r["f_rot_gyro_welch_hz"] for r in tii_results])
        tii_nyq = tii_fs / 2
        print(f"   TII:       fs={tii_fs:.1f} Hz, Nyquist={tii_nyq:.1f} Hz, "
              f"median f_rot(welch)={tii_frot:.1f} Hz, max order={tii_nyq/tii_frot:.1f}P")
    if dp_results and tii_results:
        min_nyq = min(dp_nyq, tii_nyq)
        dp_frot_med = np.median([r["f_rot_gyro_welch_hz"] for r in dp_results])
        tii_frot_med = np.median([r["f_rot_gyro_welch_hz"] for r in tii_results])
        max_order_both = min(min_nyq / dp_frot_med, min_nyq / tii_frot_med)
        print(f"   BOTH datasets: max comparable order = {max_order_both:.1f}P")
        print(f"   -> order bands 1P-{int(max_order_both)}P are physically observable on both")

    # ── JSON summary (validation artifact, NOT a result manifest) ──
    summary = {
        "type": "f_rot_validation",
        "dronepropa": {
            "n_flights": len(dp_results),
            "fs_hz_mean": float(np.mean(fs_vals)) if dp_results else None,
            "fs_hz_min": float(np.min(fs_vals)) if dp_results else None,
            "fs_hz_max": float(np.max(fs_vals)) if dp_results else None,
            "flights": dp_results,
        },
        "tii": {
            "n_missions": len(tii_results),
            "fs_hz_mean": float(np.mean(fs_vals)) if tii_results else None,
            "fs_hz_min": float(np.min(fs_vals)) if tii_results else None,
            "fs_hz_max": float(np.max(fs_vals)) if tii_results else None,
            "missions": tii_results,
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nValidation summary written -> {out}")
    print("\nThis is a VALIDATION artifact, NOT a result manifest. No model metrics.")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sys.exit(main())
