#!/usr/bin/env python3
"""
validate_results.py — fail CI if any committed result manifest lacks provenance.

A result is valid only if it declares an evidence_class and carries git_sha,
dataset_sha256 (except CODE-VERIFIED), seed, split, and a machine `produced_by`.
This is the gate that stops invented/hand-written numbers from entering the repo.

Usage:
    python scripts/validate_results.py            # validate everything under results/
    python scripts/validate_results.py path.json  # validate one file
Exit code 1 on any invalid manifest.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schema" / "result_manifest.schema.json"


def _load_validator():
    try:
        import jsonschema
    except ImportError:
        sys.exit("jsonschema not installed: pip install jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())
    return jsonschema.Draft7Validator(schema)


def validate_file(path, validator):
    try:
        data = json.loads(Path(path).read_text())
    except Exception as e:
        return [f"{path}: not valid JSON ({e})"]
    errors = []
    for err in validator.iter_errors(data):
        loc = "/".join(str(p) for p in err.path) or "(root)"
        errors.append(f"{path}: [{loc}] {err.message}")
    return errors


def main(argv):
    validator = _load_validator()
    if len(argv) > 1:
        targets = [Path(argv[1])]
    else:
        results_dir = ROOT / "results"
        targets = sorted(results_dir.rglob("*.json")) if results_dir.is_dir() else []

    if not targets:
        print("No result manifests to validate (results/ empty). OK.")
        return 0

    all_errors = []
    for t in targets:
        errs = validate_file(t, validator)
        if errs:
            all_errors.extend(errs)
        else:
            try:
                shown = t.relative_to(ROOT)
            except ValueError:
                shown = t
            print(f"OK  {shown}")

    if all_errors:
        print("\nINVALID RESULTS (missing/blurred provenance):", file=sys.stderr)
        for e in all_errors:
            print("  " + e, file=sys.stderr)
        return 1
    print(f"\nAll {len(targets)} result manifest(s) valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
