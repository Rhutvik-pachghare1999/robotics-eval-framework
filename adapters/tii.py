"""TII UAV Realistic Fault Dataset adapter.

Verified facts (from TII GitHub repo, ICRA 2023):
  - 99 missions across 5 class directories (Dataset/0..4).
  - Class = number of broken propellers: 0=Normal, 1-4=faulty.
  - Each mission is a rosbag2_* subdirectory containing JSONL sensor logs.
  - SensorCombined.jsonl: gyro_rad[3], accelerometer_m_s2[3], timestamp (us),
    gyro_integral_dt, accelerometer_integral_dt.
  - 99/99 missions have SensorCombined; 9/99 also have Phidget Imu/MagneticField.
  - Sampling rate ~128 Hz (gyro_integral_dt ~7800 us, measured mean ~128 Hz).

For cross-dataset transfer we binarize: class 0 = healthy, classes 1-4 = faulty.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Iterator, List

import numpy as np

from .base import BaseAdapter


class TIIUAVAdapter(BaseAdapter):
    REGISTRY_ID = "tii-uav-fault"

    VERIFIED = True

    LABEL_SEMANTICS = (
        "TII UAV Realistic Fault Dataset: 5 classes (0=Normal/healthy, "
        "1-4 = number of broken propellers). For cross-dataset transfer with "
        "DronePropA, we binarize: class 0 = healthy (label 0), classes 1-4 = "
        "faulty (label 1). Label is encoded in the parent directory name "
        "(Dataset/0..4/rosbag2_*). Each rosbag2_* subdirectory is one mission "
        "(take-off to landing). SensorCombined.jsonl provides gyro_rad[3] "
        "(rad/s) and accelerometer_m_s2[3] (m/s^2) at ~128 Hz."
    )

    def source_dir(self) -> Path:
        override = os.environ.get("TII_SOURCE_DIR")
        if override:
            return Path(override)
        return super().source_dir()

    @staticmethod
    def parse_class_from_dir(mission_dir: Path) -> int:
        """Extract class label from parent directory name (0..4)."""
        parent = mission_dir.parent.name
        if not parent.isdigit():
            raise ValueError(f"TII mission directory parent is not a class label: {parent}")
        return int(parent)

    @staticmethod
    def load_sensor_combined(mission_dir: Path) -> Dict[str, np.ndarray]:
        """Load SensorCombined.jsonl from a mission directory.

        Returns dict with keys: timestamp_us, gyro_rad (N,3), accel_m_s2 (N,3),
        fs_hz (scalar).
        """
        sc_files = sorted(mission_dir.glob("*-SensorCombined.jsonl"))
        if not sc_files:
            raise FileNotFoundError(f"No SensorCombined.jsonl in {mission_dir}")
        ts, gx, gy, gz, ax, ay, az = [], [], [], [], [], [], []
        dt_us_vals = []
        for line in open(sc_files[0]):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts.append(obj["timestamp"])
            g = obj.get("gyro_rad", [0.0, 0.0, 0.0])
            a = obj.get("accelerometer_m_s2", [0.0, 0.0, 0.0])
            gx.append(g[0]); gy.append(g[1]); gz.append(g[2])
            ax.append(a[0]); ay.append(a[1]); az.append(a[2])
            dt_us_vals.append(obj.get("gyro_integral_dt", 7800))
        ts = np.array(ts, dtype=np.float64)
        gyro = np.column_stack([gx, gy, gz]).astype(np.float64)
        accel = np.column_stack([ax, ay, az]).astype(np.float64)
        dt_us = np.median(dt_us_vals) if dt_us_vals else 7800.0
        if len(ts) > 1:
            mean_dt = float(np.mean(np.diff(ts)))
            fs = 1e6 / mean_dt if mean_dt > 0 else 1e6 / dt_us
        else:
            fs = 1e6 / dt_us
        return {
            "timestamp_us": ts,
            "gyro_rad": gyro,
            "accel_m_s2": accel,
            "fs_hz": float(fs),
            "gyro_integral_dt_us": float(dt_us),
        }

    def iter_samples(self) -> Iterator[Dict[str, Any]]:
        """Yield one standardized sample per mission.

        Each yielded dict contains:
          - group_id: mission directory name (rosbag2_*)
          - fault_type: 0 (healthy) or 1 (faulty, binarized from class 1-4)
          - n_broken_propellers: original class (0-4)
          - gyro_rad: (N, 3) array
          - accel_m_s2: (N, 3) array
          - timestamp_us: (N,) array
          - fs_hz: sampling rate
        """
        self._guard()
        src = self.source_dir()
        if not src.exists():
            raise FileNotFoundError(f"TII source directory not found: {src}")
        # Source dir may be .../Dataset or the parent containing 0..4
        if (src / "Dataset").exists():
            src = src / "Dataset"
        for cls_dir in sorted(src.iterdir()):
            if not cls_dir.is_dir() or not cls_dir.name.isdigit():
                continue
            cls = int(cls_dir.name)
            for mdir in sorted(cls_dir.iterdir()):
                if not mdir.is_dir():
                    continue
                try:
                    sc = self.load_sensor_combined(mdir)
                except FileNotFoundError:
                    continue
                yield {
                    "group_id": mdir.name,
                    "fault_type": 0 if cls == 0 else 1,
                    "n_broken_propellers": cls,
                    "gyro_rad": sc["gyro_rad"],
                    "accel_m_s2": sc["accel_m_s2"],
                    "timestamp_us": sc["timestamp_us"],
                    "fs_hz": sc["fs_hz"],
                }

    def load(self) -> List[Dict[str, Any]]:
        return list(self.iter_samples())
