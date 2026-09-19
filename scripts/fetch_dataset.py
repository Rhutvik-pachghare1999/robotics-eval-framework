#!/usr/bin/env python3
"""
fetch_dataset.py — download a registered dataset, checksum it, record provenance.

Datasets are NEVER committed. This records where the data came from, its license,
and a sha256 so results can cite dataset_sha256. Many datasets require manual
acceptance of terms (login/click-through) — this script will print instructions
and refuse to invent a download for those.

    python scripts/fetch_dataset.py <registry_id>
    python scripts/fetch_dataset.py --list
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "data" / "registry"
SOURCE = ROOT / "data" / "source"


def load_registry():
    return {p.stem: yaml.safe_load(p.read_text()) for p in REGISTRY.glob("*.yaml")}


def sha256_dir(path: Path) -> str:
    """Deterministic hash over file contents + relative names."""
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(path)).encode())
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("registry_id", nargs="?")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    reg = load_registry()
    if args.list or not args.registry_id:
        print("Registered datasets:")
        for k, v in reg.items():
            print(f"  {k:24s} [{v.get('license','?')}] access={v.get('access','?')}")
        return
    if args.registry_id not in reg:
        sys.exit(f"Unknown dataset '{args.registry_id}'. Use --list.")

    entry = reg[args.registry_id]
    dest = SOURCE / args.registry_id
    dest.mkdir(parents=True, exist_ok=True)

    if entry.get("access") == "manual":
        print(f"'{args.registry_id}' requires MANUAL download (license acceptance / login).")
        print("Steps:")
        for i, step in enumerate(entry.get("manual_steps", ["See url below."]), 1):
            print(f"  {i}. {step}")
        print(f"URL: {entry.get('url')}")
        print(f"After downloading, place files under: {dest}")
        print("Then re-run this script to checksum + record provenance.")
        if not any(dest.iterdir()):
            return  # nothing downloaded yet

    if not any(dest.iterdir()):
        print(f"No files under {dest}. Nothing to checksum yet.")
        print("(Automated download for this dataset is intentionally not implemented —")
        print(" verify the source + license, download manually, then re-run.)")
        return

    digest = sha256_dir(dest)
    prov = {
        "registry_id": args.registry_id,
        "url": entry.get("url"),
        "license": entry.get("license"),
        "citation": entry.get("citation"),
        "dataset_sha256": digest,
        "local_path": str(dest),
    }
    prov_path = SOURCE / f"{args.registry_id}.provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2))
    print(f"Recorded provenance -> {prov_path}")
    print(f"dataset_sha256 = {digest}")
    print("Use this dataset_sha256 in any result manifest built from this data.")


if __name__ == "__main__":
    main()
