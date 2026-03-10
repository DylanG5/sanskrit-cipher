"""Extended registry tests – discover(), list_all() on empty registry."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from ml_pipeline.core.registry import ProcessorRegistry


def test_list_all_empty():
    reg = ProcessorRegistry()
    assert reg.list_all() == []


def test_register_overwrites():
    """Registering the same name twice keeps the latest."""
    reg = ProcessorRegistry()

    class A:
        pass

    class B:
        pass

    reg.register("x", A)
    reg.register("x", B)
    assert reg.get("x") is B


def test_get_missing_error_message():
    """KeyError message should list available processors."""
    reg = ProcessorRegistry()
    reg.register("alpha", object)
    with pytest.raises(KeyError, match="alpha"):
        reg.get("beta")


def test_discover_import_error():
    """If the processors package can't be imported, discover handles it."""
    reg = ProcessorRegistry()
    with patch("importlib.import_module", side_effect=ImportError("no module")):
        reg.discover()  # should not raise
    assert reg.list_all() == []


def test_discover_finds_processor():
    """Cover registry.py lines 41-57 – successful discover loop.

    Patches importlib/pkgutil/inspect at the registry MODULE level so that
    the discover() method sees the mocks when it looks them up.
    """
    from ml_pipeline.core.processor import BaseProcessor
    import ml_pipeline.core.registry as reg_mod

    class FakeNewProcessor(BaseProcessor):
        def _setup(self): pass
        def get_metadata(self): pass
        def process(self, f, d): pass
        def cleanup(self): pass

    fake_module = MagicMock()
    fake_module.__path__ = ["/fake"]

    proc_module = MagicMock()

    def _import(name):
        if name == "ml_pipeline.processors":
            return fake_module
        return proc_module

    with (
        patch.object(reg_mod.importlib, "import_module", side_effect=_import),
        patch.object(reg_mod.pkgutil, "iter_modules", return_value=[(None, "fake_proc", False)]),
        patch.object(reg_mod.inspect, "getmembers", return_value=[("FakeNewProcessor", FakeNewProcessor)]),
        patch.object(reg_mod.inspect, "isabstract", return_value=False),
    ):
        reg = ProcessorRegistry()
        reg.discover()

    assert "fakenew" in reg.list_all()
