"""Tests for processor.py – dataclasses and BaseProcessor default behaviour."""

from __future__ import annotations

import logging

import pytest

from ml_pipeline.core.processor import (
    BaseProcessor,
    FragmentRecord,
    ProcessingResult,
    ProcessorMetadata,
)


# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------

def test_processor_metadata_defaults():
    m = ProcessorMetadata(name="test", version="1.0", description="desc")
    assert m.model_path is None
    assert m.requires_gpu is False
    assert m.batch_size == 1


def test_fragment_record_defaults():
    f = FragmentRecord(id=1, fragment_id="F001", image_path="a.png")
    assert f.processing_status is None
    assert f.line_count is None
    assert f.has_circle is None
    assert f.scale_unit is None
    assert f.line_detection_data is None
    assert f.script_type_classification_data is None


def test_processing_result_defaults():
    r = ProcessingResult(success=True)
    assert r.updates == {}
    assert r.cache_files is None
    assert r.error is None
    assert r.metadata is None


def test_processing_result_with_values():
    r = ProcessingResult(
        success=False,
        updates={"line_count": 5},
        cache_files={"f.png": b"data"},
        error="some error",
        metadata={"confidence": 0.9},
    )
    assert r.success is False
    assert r.updates["line_count"] == 5
    assert r.error == "some error"


# ---------------------------------------------------------------------------
# Concrete subclass to test BaseProcessor default should_process
# ---------------------------------------------------------------------------

class _StubProcessor(BaseProcessor):
    def _setup(self) -> None:
        pass

    def get_metadata(self) -> ProcessorMetadata:
        return ProcessorMetadata(name="stub", version="0.1", description="stub")

    def process(self, fragment, data_dir):
        return ProcessingResult(success=True)

    def cleanup(self) -> None:
        pass


def test_base_processor_should_process_returns_true():
    logger = logging.getLogger("test_stub")
    p = _StubProcessor(config={}, logger=logger)
    frag = FragmentRecord(id=1, fragment_id="F001", image_path="x.png")
    assert p.should_process(frag) is True





def test_stub_processor_metadata():
    logger = logging.getLogger("test_stub")
    p = _StubProcessor(config={}, logger=logger)
    m = p.get_metadata()
    assert m.name == "stub"
    assert m.version == "0.1"
