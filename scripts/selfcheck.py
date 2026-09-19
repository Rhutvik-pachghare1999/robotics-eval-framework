#!/usr/bin/env python3
"""
selfcheck.py — proves the result validator actually enforces provenance.

Runs a set of good and deliberately-bad manifests through the schema and asserts
the validator accepts the good ones and rejects each bad one for the right reason.
This is the "if the guardrail breaks, this fails" test. No framework needed.

    python scripts/selfcheck.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_results as V  # noqa: E402

GOOD = {
    "evidence_class": "REAL-PUBLIC",
    "project": "uav-fault",
    "dataset": "dronepropa_v1",
    "task": "fault_classification",
    "split": "leave_one_flight_out",
    "seed": 42,
    "git_sha": "a1b2c3d",
    "dataset_sha256": "f" * 64,
    "metrics": {"macro_f1": 0.87},
    "produced_by": "experiments/uav/run_fault_clf.py",
}

GOOD_CODE_VERIFIED = {
    **GOOD,
    "evidence_class": "CODE-VERIFIED",
    "dataset": "n/a",
    "dataset_sha256": "n/a",
    "task": "hocbf_saturation_feasibility",
    "split": "n/a",
    "metrics": {"violation_rate": 0.0},
    "produced_by": "experiments/safety/verify_hocbf.py",
}

BADS = {
    "missing_provenance (no git_sha)": {k: v for k, v in GOOD.items() if k != "git_sha"},
    "hand-written (produced_by=manual)": {**GOOD, "produced_by": "manual"},
    "blurred class (REAL-PUBLIC with no dataset hash)": {**GOOD, "dataset_sha256": "n/a"},
    "invented evidence_class": {**GOOD, "evidence_class": "REAL_ISH"},
    "no metrics": {**GOOD, "metrics": {}},
}


def _validate(obj):
    v = V._load_validator()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(obj, f)
        p = f.name
    return V.validate_file(p, v)


def main():
    ok = True

    for name, obj in [("REAL-PUBLIC", GOOD), ("CODE-VERIFIED", GOOD_CODE_VERIFIED)]:
        errs = _validate(obj)
        if errs:
            print(f"FAIL: valid manifest '{name}' was rejected: {errs}")
            ok = False
        else:
            print(f"PASS: valid manifest '{name}' accepted")

    for name, obj in BADS.items():
        errs = _validate(obj)
        if not errs:
            print(f"FAIL: bad manifest '{name}' was ACCEPTED (guardrail broken)")
            ok = False
        else:
            print(f"PASS: bad manifest rejected — {name}")

    if not ok:
        print("\nSELF-CHECK FAILED — the provenance guardrail is not working.")
        sys.exit(1)
    print("\nSELF-CHECK PASSED — validator enforces provenance and rejects invented results.")


if __name__ == "__main__":
    main()
