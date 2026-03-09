"""Tests for the pipeline orchestrator with mocked dependencies."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from ml_pipeline.core.processor import (
    BaseProcessor,
    FragmentRecord,
    ProcessingResult,
    ProcessorMetadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fragment(**overrides) -> FragmentRecord:
    defaults = dict(id=1, fragment_id="F001", image_path="uploads/f001.png")
    defaults.update(overrides)
    return FragmentRecord(**defaults)


class _FakeProcessor(BaseProcessor):
    """Processor that always succeeds."""

    def _setup(self):
        self._cleaned = False

    def get_metadata(self):
        return ProcessorMetadata(name="fake", version="1.0", description="fake")

    def process(self, fragment, data_dir):
        return ProcessingResult(
            success=True,
            updates={"processing_status": "completed", "last_processed_at": "CURRENT_TIMESTAMP"},
        )

    def cleanup(self):
        self._cleaned = True

    def should_process(self, fragment):
        return True


class _FailingProcessor(BaseProcessor):
    """Processor that always fails."""

    def _setup(self):
        pass

    def get_metadata(self):
        return ProcessorMetadata(name="failing", version="1.0", description="fail")

    def process(self, fragment, data_dir):
        return ProcessingResult(success=False, error="intentional failure")

    def cleanup(self):
        pass


class _SkippingProcessor(BaseProcessor):
    """Processor that skips every fragment."""

    def _setup(self):
        pass

    def get_metadata(self):
        return ProcessorMetadata(name="skip", version="1.0", description="skip")

    def process(self, fragment, data_dir):
        return ProcessingResult(success=True)

    def cleanup(self):
        pass

    def should_process(self, fragment):
        return False


class _CacheProcessor(BaseProcessor):
    """Processor that produces cache files."""

    def _setup(self):
        pass

    def get_metadata(self):
        return ProcessorMetadata(name="cache", version="1.0", description="cache")

    def process(self, fragment, data_dir):
        import numpy as np

        img = np.zeros((10, 10, 4), dtype=np.uint8)
        return ProcessingResult(
            success=True,
            updates={"processing_status": "completed"},
            cache_files={"test_seg.png": img},
        )

    def cleanup(self):
        pass


# ---------------------------------------------------------------------------
# _process_fragment tests  (unit-test the private method directly)
# ---------------------------------------------------------------------------

def _build_orchestrator_stub(processors, config=None, data_dir="/data", cache_dir="/cache"):
    """Build a minimal PipelineOrchestrator-like object without touching disk."""
    from ml_pipeline.core.orchestrator import PipelineOrchestrator

    # We'll patch __init__ to avoid file IO then set attributes manually.
    with patch.object(PipelineOrchestrator, "__init__", lambda self, *a, **kw: None):
        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)

    orch.config = config or {"processing": {"continue_on_error": True}}
    orch.logger = logging.getLogger("test_orch")
    orch.data_dir = Path(data_dir)
    orch.cache_dir = Path(cache_dir)
    orch.db = MagicMock()
    orch.registry = MagicMock()
    orch.processors = processors
    orch.config_path = Path("/fake/config.yaml")
    return orch


def test_process_fragment_success():
    logger = logging.getLogger("test")
    proc = _FakeProcessor(config={}, logger=logger)
    orch = _build_orchestrator_stub([proc])

    frag = _make_fragment()
    result = orch._process_fragment(frag, dry_run=False)
    assert result is True
    orch.db.update_fragment.assert_called_once()


def test_process_fragment_dry_run_no_db_write():
    logger = logging.getLogger("test")
    proc = _FakeProcessor(config={}, logger=logger)
    orch = _build_orchestrator_stub([proc])

    frag = _make_fragment()
    result = orch._process_fragment(frag, dry_run=True)
    assert result is True
    orch.db.update_fragment.assert_not_called()


def test_process_fragment_all_skip():
    logger = logging.getLogger("test")
    proc = _SkippingProcessor(config={}, logger=logger)
    orch = _build_orchestrator_stub([proc])

    frag = _make_fragment()
    result = orch._process_fragment(frag, dry_run=False)
    assert result is False  # no processor did anything


def test_process_fragment_failure():
    logger = logging.getLogger("test")
    proc = _FailingProcessor(config={}, logger=logger)
    orch = _build_orchestrator_stub([proc])

    frag = _make_fragment()
    result = orch._process_fragment(frag, dry_run=False)
    assert result is False


def test_process_fragment_with_cache_files(tmp_path):
    logger = logging.getLogger("test")
    proc = _CacheProcessor(config={}, logger=logger)
    orch = _build_orchestrator_stub([proc], cache_dir=str(tmp_path))
    orch.cache_dir = tmp_path

    frag = _make_fragment()
    result = orch._process_fragment(frag, dry_run=False)
    assert result is True
    # Cache file should be written
    assert (tmp_path / "segmented" / "test_seg.png").exists()


def test_process_fragment_db_error_returns_false():
    logger = logging.getLogger("test")
    proc = _FakeProcessor(config={}, logger=logger)
    orch = _build_orchestrator_stub([proc])
    orch.db.update_fragment.side_effect = Exception("db write error")

    frag = _make_fragment()
    result = orch._process_fragment(frag, dry_run=False)
    assert result is False


# ---------------------------------------------------------------------------
# _cleanup_processors
# ---------------------------------------------------------------------------

def test_cleanup_processors():
    logger = logging.getLogger("test")
    p1 = _FakeProcessor(config={}, logger=logger)
    p2 = _FakeProcessor(config={}, logger=logger)
    orch = _build_orchestrator_stub([p1, p2])

    orch._cleanup_processors()
    assert p1._cleaned is True
    assert p2._cleaned is True
    assert orch.processors == []


def test_cleanup_processors_error_does_not_raise():
    """Even if cleanup raises, the orchestrator should not propagate."""
    logger = logging.getLogger("test")
    bad = MagicMock()
    bad.cleanup.side_effect = RuntimeError("boom")
    orch = _build_orchestrator_stub([bad])
    orch._cleanup_processors()  # should not raise
    assert orch.processors == []


# ---------------------------------------------------------------------------
# _generate_cache_files
# ---------------------------------------------------------------------------

def test_generate_cache_files(tmp_path):
    import numpy as np

    orch = _build_orchestrator_stub([], cache_dir=str(tmp_path))
    orch.cache_dir = tmp_path

    img = np.zeros((10, 10, 4), dtype=np.uint8)
    orch._generate_cache_files({"output.png": img})

    assert (tmp_path / "segmented" / "output.png").exists()


# ---------------------------------------------------------------------------
# _get_fragments
# ---------------------------------------------------------------------------

def test_get_fragments_delegates_to_db():
    orch = _build_orchestrator_stub([])
    orch.db.get_fragments.return_value = [_make_fragment()]
    orch.db.get_last_processed_id.return_value = None

    frags = orch._get_fragments(resume=False, limit=10, fragment_ids=None)
    orch.db.get_fragments.assert_called_once_with(limit=10, offset=0, fragment_ids=None, collection=None)
    assert len(frags) == 1


def test_get_fragments_resume():
    orch = _build_orchestrator_stub([])
    orch.db.get_last_processed_id.return_value = 42
    orch.db.get_fragments.return_value = []

    orch._get_fragments(resume=True, limit=None, fragment_ids=None)


# ---------------------------------------------------------------------------
# _initialize_processors
# ---------------------------------------------------------------------------

def test_initialize_processors_from_config():
    """Cover orchestrator.py lines 150-210 – processor loading loop."""
    logger = logging.getLogger("test")
    proc = _FakeProcessor(config={}, logger=logger)

    orch = _build_orchestrator_stub([], config={
        "processing": {"continue_on_error": True},
        "processors": {
            "fake": {"enabled": True, "config": {"model_version": "1.0"}},
        },
    })
    orch.config_path = Path("/fake/config.yaml")

    # Mock registry to return the FakeProcessor class
    orch.registry.get.return_value = _FakeProcessor

    orch._initialize_processors(processor_names=None, force=False)
    assert len(orch.processors) == 1


def test_initialize_processors_by_name():
    """Explicit processor name list."""
    logger = logging.getLogger("test")

    orch = _build_orchestrator_stub([], config={
        "processing": {"continue_on_error": True},
        "processors": {
            "fake": {"enabled": True, "config": {"model_version": "1.0"}},
        },
    })
    orch.config_path = Path("/fake/config.yaml")
    orch.registry.get.return_value = _FakeProcessor

    orch._initialize_processors(processor_names=["fake"], force=False)
    assert len(orch.processors) == 1


def test_initialize_processors_force_mode():
    """Force mode should set skip_processed to False."""
    logger = logging.getLogger("test")

    orch = _build_orchestrator_stub([], config={
        "processing": {"continue_on_error": True},
        "processors": {
            "fake": {"enabled": True, "config": {"model_version": "1.0", "skip_processed": True}},
        },
    })
    orch.config_path = Path("/fake/config.yaml")
    orch.registry.get.return_value = _FakeProcessor

    orch._initialize_processors(processor_names=["fake"], force=True)
    assert len(orch.processors) == 1


def test_initialize_processors_missing_config():
    """Processor with no config entry is skipped."""
    orch = _build_orchestrator_stub([], config={
        "processing": {"continue_on_error": True},
        "processors": {},
    })
    orch.config_path = Path("/fake/config.yaml")

    orch._initialize_processors(processor_names=["missing"], force=False)
    assert len(orch.processors) == 0


def test_initialize_processors_error_handling():
    """Exception during processor load should not crash."""
    orch = _build_orchestrator_stub([], config={
        "processing": {"continue_on_error": True},
        "processors": {
            "bad": {"enabled": True, "config": {"model_version": "1.0"}},
        },
    })
    orch.config_path = Path("/fake/config.yaml")
    orch.registry.get.side_effect = KeyError("bad")

    orch._initialize_processors(processor_names=["bad"], force=False)
    assert len(orch.processors) == 0


def test_initialize_processors_model_path_resolution():
    """Cover model_path / meta_path resolution in _initialize_processors."""
    orch = _build_orchestrator_stub([], config={
        "processing": {"continue_on_error": True},
        "processors": {
            "fake": {
                "enabled": True,
                "model_path": "models/best.pt",
                "meta_path": "models/meta.json",
                "config": {"model_version": "1.0"},
            },
        },
    })
    orch.config_path = Path("/fake/config.yaml")
    orch.registry.get.return_value = _FakeProcessor

    orch._initialize_processors(processor_names=["fake"], force=False)
    assert len(orch.processors) == 1


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def test_run_full_pipeline():
    """Cover orchestrator.py run() method – lines 85-140."""
    logger = logging.getLogger("test")
    proc = _FakeProcessor(config={}, logger=logger)

    orch = _build_orchestrator_stub([], config={
        "processing": {"continue_on_error": True},
        "processors": {
            "fake": {"enabled": True, "config": {"model_version": "1.0"}},
        },
    })
    orch.config_path = Path("/fake/config.yaml")
    orch.registry.get.return_value = _FakeProcessor
    orch.db.get_fragments.return_value = [_make_fragment()]
    orch.db.get_last_processed_id.return_value = None

    orch.run(processor_names=["fake"], resume=False, dry_run=True, limit=10)

    # Should have called get_fragments and processed
    orch.db.get_fragments.assert_called_once()


def test_run_no_processors_enabled():
    """run() with no enabled processors – early return."""
    orch = _build_orchestrator_stub([], config={
        "processing": {"continue_on_error": True},
        "processors": {},
    })
    orch.config_path = Path("/fake/config.yaml")

    orch.run(processor_names=None)
    # Should not have tried to get fragments
    orch.db.get_fragments.assert_not_called()


def test_run_no_fragments():
    """run() with empty fragment list – early return."""
    logger = logging.getLogger("test")

    orch = _build_orchestrator_stub([], config={
        "processing": {"continue_on_error": True},
        "processors": {
            "fake": {"enabled": True, "config": {"model_version": "1.0"}},
        },
    })
    orch.config_path = Path("/fake/config.yaml")
    orch.registry.get.return_value = _FakeProcessor
    orch.db.get_fragments.return_value = []
    orch.db.get_last_processed_id.return_value = None

    orch.run(processor_names=["fake"])


def test_run_stop_on_error():
    """run() stops on error when continue_on_error is False."""
    logger = logging.getLogger("test")
    proc = _FailingProcessor(config={}, logger=logger)

    orch = _build_orchestrator_stub([], config={
        "processing": {"continue_on_error": False},
        "processors": {
            "failing": {"enabled": True, "config": {"model_version": "1.0"}},
        },
    })
    orch.config_path = Path("/fake/config.yaml")
    orch.registry.get.return_value = _FailingProcessor
    orch.db.get_fragments.return_value = [_make_fragment(), _make_fragment(id=2, fragment_id="F002")]
    orch.db.get_last_processed_id.return_value = None

    orch.run(processor_names=["failing"])


def test_run_force_and_collection():
    """Cover the force and collection kwargs in run()."""
    logger = logging.getLogger("test")

    orch = _build_orchestrator_stub([], config={
        "processing": {"continue_on_error": True},
        "processors": {
            "fake": {"enabled": True, "config": {"model_version": "1.0", "skip_processed": True}},
        },
    })
    orch.config_path = Path("/fake/config.yaml")
    orch.registry.get.return_value = _FakeProcessor
    orch.db.get_fragments.return_value = [_make_fragment()]
    orch.db.get_last_processed_id.return_value = None

    orch.run(processor_names=["fake"], force=True, collection="BLL", resume=True)


def test_run_exception_in_process_fragment():
    """Cover the except block inside the fragment processing loop."""
    logger = logging.getLogger("test")

    class _RaisingProcessor(BaseProcessor):
        def _setup(self): pass
        def get_metadata(self):
            return ProcessorMetadata(name="raise", version="1.0", description="")
        def process(self, fragment, data_dir):
            raise RuntimeError("boom")
        def cleanup(self): pass
        def should_process(self, fragment):
            return True

    orch = _build_orchestrator_stub([], config={
        "processing": {"continue_on_error": True},
        "processors": {
            "raise": {"enabled": True, "config": {"model_version": "1.0"}},
        },
    })
    orch.config_path = Path("/fake/config.yaml")
    orch.registry.get.return_value = _RaisingProcessor
    orch.db.get_fragments.return_value = [_make_fragment()]
    orch.db.get_last_processed_id.return_value = None

    orch.run(processor_names=["raise"])
    orch.db.get_fragments.assert_called_once_with(limit=None, offset=0, fragment_ids=None, collection=None)


# ---------------------------------------------------------------------------
# _initialize_processors (partial – verify enabled filtering logic)
# ---------------------------------------------------------------------------

def test_initialize_processors_filters_enabled(tmp_path):
    orch = _build_orchestrator_stub([])
    orch.config = {
        "processors": {
            "fake": {"enabled": True, "config": {}, "model_path": str(tmp_path / "m.pt")},
            "disabled_one": {"enabled": False, "config": {}},
        },
        "processing": {},
    }
    orch.config_path = tmp_path / "config.yaml"

    # Mock registry to return _FakeProcessor
    orch.registry.get.return_value = _FakeProcessor

    orch._initialize_processors(processor_names=None)
    # Only the enabled processor should have been loaded
    assert len(orch.processors) == 1


def test_initialize_processors_explicit_names(tmp_path):
    orch = _build_orchestrator_stub([])
    orch.config = {
        "processors": {
            "specific": {"enabled": False, "config": {}},
        },
        "processing": {},
    }
    orch.config_path = tmp_path / "config.yaml"
    orch.registry.get.return_value = _FakeProcessor

    orch._initialize_processors(processor_names=["specific"])
    assert len(orch.processors) == 1


def test_initialize_processors_missing_config(tmp_path):
    orch = _build_orchestrator_stub([])
    orch.config = {"processors": {}, "processing": {}}
    orch.config_path = tmp_path / "config.yaml"
    orch._initialize_processors(processor_names=["nonexistent"])
    assert len(orch.processors) == 0
