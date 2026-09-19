"""
Base dataset adapter. Every dataset adapter subclasses this and stays HARD-FAILED
until a human sets VERIFIED = True after confirming the source label/target
semantics against the dataset's own paper/record.

This is deliberate: it prevents an agent (or a rushed human) from silently
inventing a label conversion just to make the pipeline run.
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "source"


class AdapterNotVerified(NotImplementedError):
    pass


class BaseAdapter:
    #: registry id (matches data/registry/<id>.yaml)
    REGISTRY_ID: str = ""

    #: Flip to True ONLY after a human has read the source paper/record and
    #: documented the real target semantics in `LABEL_SEMANTICS` below.
    VERIFIED: bool = False

    #: Human-written, cited description of exactly what the label/target means in
    #: the SOURCE data, and how (if at all) it maps to this project's target.
    LABEL_SEMANTICS: str = ""

    def _guard(self):
        if not self.VERIFIED:
            raise AdapterNotVerified(
                f"[{self.REGISTRY_ID}] adapter is not verified. Before implementing:\n"
                f"  1. Read the source dataset's paper/record.\n"
                f"  2. Document the real label/target semantics in LABEL_SEMANTICS.\n"
                f"  3. Decide: convert mathematically (documented) OR change the model\n"
                f"     to predict the source's actual target. NO invented conversion.\n"
                f"  4. Set VERIFIED = True and implement load().\n"
            )
        if not self.LABEL_SEMANTICS.strip():
            raise AdapterNotVerified(
                f"[{self.REGISTRY_ID}] VERIFIED=True but LABEL_SEMANTICS is empty. "
                f"Document the verified label meaning before use."
            )

    def source_dir(self) -> Path:
        return SOURCE / self.REGISTRY_ID

    def load(self):
        """Return standardized samples. Subclass implements AFTER _guard() passes."""
        self._guard()
        raise NotImplementedError(f"{self.REGISTRY_ID}.load() not implemented yet.")
