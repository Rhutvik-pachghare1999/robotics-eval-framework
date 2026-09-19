"""Proves every dataset adapter hard-fails until a human verifies label semantics."""
import importlib
import pkgutil

import pytest

import adapters
from adapters.base import BaseAdapter, AdapterNotVerified


def _all_adapter_classes():
    classes = []
    pkg = adapters
    for _, name, _ in pkgutil.iter_modules(pkg.__path__):
        if name == "base":
            continue
        mod = importlib.import_module(f"adapters.{name}")
        for obj in vars(mod).values():
            if isinstance(obj, type) and issubclass(obj, BaseAdapter) and obj is not BaseAdapter:
                classes.append(obj)
    return classes


def test_there_are_adapters():
    assert _all_adapter_classes(), "no adapter subclasses found"


@pytest.mark.parametrize("cls", _all_adapter_classes())
def test_unverified_adapter_refuses_to_load(cls):
    """An unverified adapter must raise, never silently return invented data."""
    inst = cls()
    if not inst.VERIFIED:
        with pytest.raises(AdapterNotVerified):
            inst.load()
    else:
        # If someone flipped VERIFIED, they must have documented semantics.
        assert inst.LABEL_SEMANTICS.strip(), f"{cls.__name__} VERIFIED but no LABEL_SEMANTICS"
