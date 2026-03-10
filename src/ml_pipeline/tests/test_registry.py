from __future__ import annotations

import pytest

from ml_pipeline.core.registry import ProcessorRegistry


class DummyProcessor:
    pass


def test_registry_register_and_get() -> None:
    """test-ut-registry-1"""
    registry = ProcessorRegistry()
    registry.register("dummy", DummyProcessor)

    assert registry.get("dummy") is DummyProcessor
    assert "dummy" in registry
    assert len(registry) == 1
    assert registry.list_all() == ["dummy"]


def test_registry_get_missing() -> None:
    """test-ut-registry-2 (detailed assertion in test_registry_extended)"""
    registry = ProcessorRegistry()
    with pytest.raises(KeyError):
        registry.get("missing")
