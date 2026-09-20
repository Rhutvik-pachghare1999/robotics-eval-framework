"""DronePropA adapter for the UAV fault-diagnostics project.

Verified facts (from DronePropA data article / Mendeley record / GitHub processing
scripts by Ismail et al.):
  - 130 indoor flight sequences in raw MATLAB .mat format (127 present on SOL).
  - Each .mat contains three variables:
      QDrone_data      : 56 x N  (rows=channels, cols=time)
      commander_data   : 37 x N
      stabilizer_data  : 21 x N
  - Fault labels are encoded in the FILENAME, not inside the .mat variables:
      F{0-3}   : fault type   (0=healthy, 1=edge_cut, 2=crack, 3=surface_cut)
      SV{0-3}  : severity     (0=baseline, 1-3=increasing severity)
      SP{1,2}  : speed setting
      t{1-5}   : trajectory id (grouping field for splits)
      D#       : drone id
      R#       : repeat/run id

QDrone_data channel mapping transcribed from the published MATLAB scripts:
  Row  1: time / timestamp (s)
  Rows 2-10:  IMU_0BF (IMU 1) — 9 channels:
              roll(2), pitch(3), yaw(4),
              roll_rate(5), pitch_rate(6), yaw_rate(7),
              roll_acc(8), pitch_acc(9), yaw_acc(10)
  Rows 11-19: IMU_1BF (IMU 2) — 9 channels:
              roll(11), pitch(12), yaw(13),
              roll_rate(14), pitch_rate(15), yaw_rate(16),
              roll_acc(17), pitch_acc(18), yaw_acc(19)
  Rows 20-23: motor_CMD — 4 channels (front-left, front-right, back-left, back-right)
  Row 24:     battery voltage
  Row 25:     electronics current
  Row 26:     motor current
  Rows 27-29: gyro_1 — roll, pitch, yaw (rad/s)
  Rows 30-32: accel_1 — x, y, z (m/s^2)
  Rows 33-35: gyro_2 — roll, pitch, yaw (rad/s)
  Rows 36-38: accel_2 — x, y, z (m/s^2)
  Rows 39-45: OpticalFlow — 7 channels (defined in plot_daq_qdrone2.m.txt)
  Row 46:     height sensor / range_data
  Rows 47-54: per-rotor motor + ESC commands:
              motor_FL(47), esc_FL(48), motor_FR(49), esc_FR(50),
              motor_BL(51), esc_BL(52), motor_BR(53), esc_BR(54)
  Rows 55-56: unused / VERIFY — genuinely outside the documented (2:55) log range.

There are TWO IMU representations in rows 2-38.  Use either the body-frame
IMU_0BF/IMU_1BF rows (2-19, includes angular acceleration) OR the raw gyro+accel
rows (27-38), not both, and document which representation a feature set uses.

Group id for splits: full filename (leave-one-flight-out) or trajectory t{1-5}
(leave-one-trajectory-out). Do NOT use random row / window splits for headline
metrics.

Sources:
  - Mendeley Data record: https://data.mendeley.com/datasets/ftdyxrr3c5/1
    DOI 10.17632/ftdyxrr3c5.1
  - Data in Brief article: 10.1016/j.dib.2025.111589
  - Processing/visualization code: https://github.com/DrIsmailCode/DronePropA-Motion-Trajectories-Dataset
    (important_data_extraction.m.txt, plot_daq_qdrone2.m.txt)
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from .base import BaseAdapter


class DronePropAAdapter(BaseAdapter):
    REGISTRY_ID = "dronepropa"

    # Mapping approved by human review of the source MATLAB scripts.
    VERIFIED = True

    LABEL_SEMANTICS = (
        "DronePropA fault labels are encoded in the .mat filename, not in internal "
        "fields. Healthy files include a _D#_R# suffix "
        "(e.g. F0_SV0_SP1_t1_D1_R1.mat); faulty files omit that suffix "
        "(e.g. F1_SV1_SP1_t1.mat). "
        "Pattern: F{0-3}_SV{0-3}_SP{1,2}_t{1-5}[_D#][_R#].mat. "
        "F0=healthy, F1=edge_cut, F2=crack, F3=surface_cut. "
        "SV0=baseline severity, SV1-3=increasing severity. "
        "Group id for leave-one-flight-out splits is the full filename; "
        "trajectory t{1-5} can be used for leave-one-trajectory-out splits. "
        "QDrone_data rows 55-56 are unused because they are outside the "
        "documented 2:55 log range."
    )

    # Filename token pattern:
    #   Healthy flights:  F0_SV0_SP1_t1_D1_R1.mat  (includes _D#_R#)
    #   Faulty flights:   F1_SV1_SP1_t1.mat         (no _D#_R# suffix)
    _FILENAME_RE = re.compile(
        r"F(?P<fault>\d+)"
        r"_SV(?P<severity>\d+)"
        r"_SP(?P<speed>\d+)"
        r"_t(?P<trajectory>\d+)"
        r"(?:_D(?P<drone>\d+))?"
        r"(?:_R(?P<repeat>\d+))?"
        r"\.mat$",
        re.IGNORECASE,
    )

    def source_dir(self) -> Path:
        """Return the directory holding the raw .mat files.

        Defaults to ROOT/data/source/dronepropa, but may be overridden with the
        DRONEPROPA_SOURCE_DIR environment variable for HPC/SOL storage.
        """
        override = os.environ.get("DRONEPROPA_SOURCE_DIR")
        if override:
            return Path(override)
        return super().source_dir()

    @staticmethod
    def parse_filename(path: Path) -> Dict[str, Any]:
        """Extract fault, severity, speed, trajectory, drone, repeat from filename.

        Healthy flights include _D#_R# (drone/repeat); faulty flights omit both.
        Missing drone/repeat default to None so callers can treat them as
        single-run fault recordings.
        """
        m = DronePropAAdapter._FILENAME_RE.match(path.name)
        if not m:
            raise ValueError(f"Filename does not match DronePropA convention: {path.name}")
        drone = m.group("drone")
        repeat = m.group("repeat")
        return {
            "fault_type": int(m.group("fault")),
            "severity": int(m.group("severity")),
            "speed": int(m.group("speed")),
            "trajectory": int(m.group("trajectory")),
            "drone": int(drone) if drone is not None else None,
            "repeat": int(repeat) if repeat is not None else None,
        }

    def iter_samples(self):
        """Yield one standardized sample per .mat flight log (memory-efficient).

        Each yielded dict contains:
          - group_id: filename stem (leave-one-flight-out unit)
          - fault_type, severity, speed, trajectory, drone, repeat
          - qdrone_data:     np.ndarray, shape (56, T)
          - commander_data:  np.ndarray, shape (37, T)
          - stabilizer_data: np.ndarray, shape (21, T)
          - time_s:          np.ndarray, shape (T,)
        """
        self._guard()
        src = self.source_dir()
        if not src.exists():
            raise FileNotFoundError(f"DronePropA source directory not found: {src}")

        files = sorted(src.glob("*.mat"))
        if not files:
            raise FileNotFoundError(f"No .mat files found under {src}")

        from scipy.io import loadmat  # lazy import: only needed when loading data

        for fpath in files:
            meta = self.parse_filename(fpath)
            mat = loadmat(str(fpath))
            for key in ("QDrone_data", "commander_data", "stabilizer_data"):
                if key not in mat:
                    raise KeyError(f"{fpath.name} missing required variable '{key}'")

            qdrone = np.asarray(mat["QDrone_data"], dtype=np.float64)
            commander = np.asarray(mat["commander_data"], dtype=np.float64)
            stabilizer = np.asarray(mat["stabilizer_data"], dtype=np.float64)

            if qdrone.shape[0] != 56:
                raise ValueError(
                    f"{fpath.name}: QDrone_data has {qdrone.shape[0]} rows, expected 56"
                )
            if commander.shape[0] != 37:
                raise ValueError(
                    f"{fpath.name}: commander_data has {commander.shape[0]} rows, expected 37"
                )
            if stabilizer.shape[0] != 21:
                raise ValueError(
                    f"{fpath.name}: stabilizer_data has {stabilizer.shape[0]} rows, expected 21"
                )

            time_s = qdrone[0, :]
            yield {
                "group_id": fpath.stem,
                **meta,
                "qdrone_data": qdrone,
                "commander_data": commander,
                "stabilizer_data": stabilizer,
                "time_s": time_s,
            }

    def load(self) -> List[Dict[str, Any]]:
        """Load every DronePropA .mat flight log as a standardized sample.

        Warning: this materializes all flight logs in memory. For the full
        DronePropA dataset that is ~10 GB. Prefer :meth:`iter_samples` for
        large-scale or HPC work.
        """
        return list(self.iter_samples())
