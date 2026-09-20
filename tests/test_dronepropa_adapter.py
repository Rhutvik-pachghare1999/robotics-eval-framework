"""Adapter tests for DronePropA (UAV fault diagnostics).

These tests require the raw DronePropA .mat files. On SOL, point
DRONEPROPA_SOURCE_DIR to the unzipped dataset directory; locally the test
skips if no data is present.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from adapters.dronepropa import DronePropAAdapter


@pytest.fixture
def adapter():
    return DronePropAAdapter()


def _source_dir() -> Optional[Path]:
    env = __import__("os").environ.get("DRONEPROPA_SOURCE_DIR")
    if env:
        p = Path(env)
    else:
        p = Path(__file__).resolve().parents[1] / "data" / "source" / "dronepropa"
    if not p.exists() or not any(p.glob("*.mat")):
        return None
    return p


def test_adapter_is_verified_and_documents_semantics(adapter):
    """Once VERIFIED is flipped, LABEL_SEMANTICS must be non-empty."""
    assert adapter.VERIFIED, "DronePropA adapter should be VERIFIED after label review"
    assert adapter.LABEL_SEMANTICS.strip(), "LABEL_SEMANTICS must document the source meaning"


def test_parse_filename_roundtrip():
    """Filename tokens must map to the documented label scheme."""
    meta = DronePropAAdapter.parse_filename(Path("F2_SV3_SP1_t4_D2_R5.mat"))
    assert meta == {
        "fault_type": 2,
        "severity": 3,
        "speed": 1,
        "trajectory": 4,
        "drone": 2,
        "repeat": 5,
    }


def test_parse_filename_no_repeat():
    meta = DronePropAAdapter.parse_filename(Path("F1_SV1_SP2_t2_D3.mat"))
    assert meta["repeat"] is None
    assert meta["fault_type"] == 1


def test_parse_filename_fault_no_drone_repeat():
    """Faulty flights omit the _D#_R# suffix entirely."""
    meta = DronePropAAdapter.parse_filename(Path("F1_SV1_SP1_t1.mat"))
    assert meta == {
        "fault_type": 1,
        "severity": 1,
        "speed": 1,
        "trajectory": 1,
        "drone": None,
        "repeat": None,
    }


@pytest.mark.skipif(_source_dir() is None, reason="DronePropA .mat data not available")
def test_load_all_real_files(adapter, capsys):
    """Load ALL 127 DronePropA .mat files and verify every flight loads.

    This is a deliberate end-to-end check of the adapter against the full
    dataset on SOL. It uses iter_samples() and only keeps small metadata
    (counts, group ids); full flight arrays are released each iteration so the
    test does not materialise the whole ~10 GB dataset at once.
    """
    from collections import Counter

    src = _source_dir()
    files = sorted(src.glob("*.mat"))
    assert len(files) == 127, f"expected 127 .mat files, found {len(files)}"
    expected_ids = {f.stem for f in files}

    fault_counts = Counter()
    severity_counts = Counter()
    loaded_ids = set()
    total = 0

    for s in adapter.iter_samples():
        total += 1
        loaded_ids.add(s["group_id"])
        fault_counts[s["fault_type"]] += 1
        severity_counts[s["severity"]] += 1

        # Per-flight shape + label checks.
        assert s["qdrone_data"].shape[0] == 56, "QDrone_data must have 56 channels"
        assert s["commander_data"].shape[0] == 37, "commander_data must have 37 channels"
        assert s["stabilizer_data"].shape[0] == 21, "stabilizer_data must have 21 channels"
        assert s["qdrone_data"].shape[1] == s["time_s"].shape[0], "time length must match columns"
        assert s["time_s"][0] == pytest.approx(0.0, abs=1e-9), "time should start at 0"
        assert 0 <= s["fault_type"] <= 3, "fault_type must be in {0,1,2,3}"
        assert 0 <= s["severity"] <= 3, "severity must be in {0,1,2,3}"
        assert s["trajectory"] in {1, 2, 3, 4, 5}, "trajectory must be in {1..5}"

    with capsys.disabled():
        print("\n=== DronePropA full-load summary ===")
        print(f"Total flights loaded: {total}")
        print(f"Fault-type counts:    {dict(sorted(fault_counts.items()))}")
        print(f"Severity counts:      {dict(sorted(severity_counts.items()))}")
        print("====================================\n")

    assert total == len(files), "iter_samples() must yield one sample per .mat file"
    assert loaded_ids == expected_ids, "every file stem must appear as a group_id"

    # All four classes (healthy + three fault types) must be present.
    assert fault_counts[0] > 0, "no healthy (F0) flights loaded"
    assert fault_counts[1] > 0, "no edge_cut (F1) flights loaded"
    assert fault_counts[2] > 0, "no crack (F2) flights loaded"
    assert fault_counts[3] > 0, "no surface_cut (F3) flights loaded"
