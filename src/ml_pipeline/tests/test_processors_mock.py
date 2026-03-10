"""Tests for ML model processors using mocks for external dependencies.

Covers: segmentation, circle detection, classification, line detection,
scale detection, and script type classification processors.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from ml_pipeline.core.processor import FragmentRecord, ProcessingResult, ProcessorMetadata


def _make_fragment(**kw):
    defaults = dict(id=1, fragment_id="F001", image_path="uploads/f001.png")
    defaults.update(kw)
    return FragmentRecord(**defaults)


def _create_test_image(tmp_path, rel_path="uploads/f001.png"):
    """Create a small test image and return the data_dir."""
    import cv2

    img_full = tmp_path / rel_path
    img_full.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_full), img)
    return str(tmp_path)


# ===========================================================================
# Segmentation Processor
# ===========================================================================

class TestSegmentationProcessor:
    @pytest.fixture(autouse=True)
    def _mock_yolo(self):
        """Mock YOLO so we don't need ultralytics installed."""
        mock_yolo_class = MagicMock()

        # Build a fake result
        mask_data = np.ones((100, 100), dtype=np.float32)
        box_data = np.array([10, 10, 90, 90, 0.95, 0])

        fake_result = MagicMock()
        fake_result.masks.data = [MagicMock()]
        fake_result.masks.data[0].cpu.return_value.numpy.return_value = mask_data
        fake_result.masks.__len__ = lambda self: 1
        fake_result.boxes.data = [MagicMock()]
        fake_result.boxes.data[0].cpu.return_value.numpy.return_value = box_data

        mock_model = MagicMock()
        mock_model.predict.return_value = [fake_result]
        mock_yolo_class.return_value = mock_model
        self._mock_model = mock_model

        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_class)}):
            # Re-import to pick up mocked YOLO
            import importlib
            import ml_pipeline.processors.segmentation_processor as mod
            mod.YOLO = mock_yolo_class
            self.mod = mod
            yield

    def _make_processor(self, tmp_path):
        model_path = tmp_path / "best.pt"
        model_path.touch()

        config = {
            "model_path": str(model_path),
            "config": {
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45,
                "model_version": "1.0",
            },
        }
        logger = logging.getLogger("test_seg")
        return self.mod.SegmentationProcessor(config, logger)

    def test_get_metadata(self, tmp_path):
        p = self._make_processor(tmp_path)
        m = p.get_metadata()
        assert m.name == "segmentation"
        assert m.version == "1.0"

    def test_should_process_no_coords(self, tmp_path):
        p = self._make_processor(tmp_path)
        frag = _make_fragment(segmentation_coords=None)
        assert p.should_process(frag) is True

    def test_should_process_same_version(self, tmp_path):
        p = self._make_processor(tmp_path)
        frag = _make_fragment(
            segmentation_coords='{"contours":[]}',
            segmentation_model_version="1.0",
        )
        assert p.should_process(frag) is False

    def test_should_process_skip_disabled(self, tmp_path):
        model_path = tmp_path / "best.pt"
        model_path.touch()
        config = {
            "model_path": str(model_path),
            "skip_processed": False,
            "config": {
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45,
                "model_version": "1.0",
            },
        }
        logger = logging.getLogger("test_seg")
        p = self.mod.SegmentationProcessor(config, logger)
        frag = _make_fragment(segmentation_coords='{}', segmentation_model_version="1.0")
        assert p.should_process(frag) is True

    def test_process_image_not_found(self, tmp_path):
        p = self._make_processor(tmp_path)
        frag = _make_fragment(image_path="missing.png")
        result = p.process(frag, str(tmp_path))
        assert result.success is False

    def test_process_success(self, tmp_path):
        p = self._make_processor(tmp_path)
        data_dir = _create_test_image(tmp_path)
        frag = _make_fragment()
        result = p.process(frag, data_dir)
        assert result.success is True
        assert "segmentation_coords" in result.updates
        assert result.cache_files is not None

    def test_process_null_image_read(self, tmp_path):
        """Cover segmentation_processor.py line 99: cv2.imread returns None."""
        import cv2 as real_cv2
        p = self._make_processor(tmp_path)
        data_dir = _create_test_image(tmp_path)
        frag = _make_fragment()
        with patch.object(self.mod.cv2, "imread", return_value=None):
            result = p.process(frag, data_dir)
        assert result.success is False
        assert "Failed to read image" in result.error

    def test_process_no_masks(self, tmp_path):
        """Cover segmentation_processor.py line 117: no masks detected."""
        p = self._make_processor(tmp_path)
        data_dir = _create_test_image(tmp_path)
        frag = _make_fragment()
        # Override model prediction to return no masks
        fake_result = MagicMock()
        fake_result.masks = None
        self._mock_model.predict.return_value = [fake_result]
        result = p.process(frag, data_dir)
        assert result.success is False
        assert "No masks detected" in result.error

    def test_process_exception(self, tmp_path):
        """Cover the except block in process()."""
        p = self._make_processor(tmp_path)
        data_dir = _create_test_image(tmp_path)
        self._mock_model.predict.side_effect = RuntimeError("model error")
        result = p.process(_make_fragment(), data_dir)
        assert result.success is False
        assert "model error" in result.error

    def test_cleanup(self, tmp_path):
        p = self._make_processor(tmp_path)
        p.cleanup()


# ===========================================================================
# Circle Detection Processor
# ===========================================================================

class TestCircleDetectionProcessor:
    @pytest.fixture(autouse=True)
    def _mock_yolo(self):
        mock_yolo_class = MagicMock()

        fake_result = MagicMock()
        # Simulate one detection: class 0 (circle), conf 0.9
        fake_result.boxes.cls = [MagicMock()]
        fake_result.boxes.cls[0].__int__ = lambda self: 0
        fake_result.boxes.conf = [0.9]
        fake_result.boxes.__len__ = lambda s: 1

        mock_model = MagicMock()
        mock_model.predict.return_value = [fake_result]
        mock_yolo_class.return_value = mock_model

        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_class)}):
            import ml_pipeline.processors.circle_detection_processor as mod
            mod.YOLO = mock_yolo_class
            self.mod = mod
            yield

    def _make_processor(self, tmp_path):
        model_path = tmp_path / "circle.pt"
        model_path.touch()
        config = {
            "model_path": str(model_path),
            "config": {
                "confidence_threshold": 0.25,
                "model_version": "1.0",
            },
        }
        logger = logging.getLogger("test_circle")
        return self.mod.CircleDetectionProcessor(config, logger)

    def test_get_metadata(self, tmp_path):
        m = self._make_processor(tmp_path).get_metadata()
        assert m.name == "circledetection"

    def test_should_process_no_data(self, tmp_path):
        p = self._make_processor(tmp_path)
        assert p.should_process(_make_fragment(has_circle=None)) is True

    def test_should_process_already_done(self, tmp_path):
        p = self._make_processor(tmp_path)
        assert p.should_process(_make_fragment(has_circle=True)) is False

    def test_process_image_not_found(self, tmp_path):
        p = self._make_processor(tmp_path)
        result = p.process(_make_fragment(image_path="nope.png"), str(tmp_path))
        assert result.success is False

    def test_process_circle_detected(self, tmp_path):
        p = self._make_processor(tmp_path)
        data_dir = _create_test_image(tmp_path)
        result = p.process(_make_fragment(), data_dir)
        assert result.success is True
        assert result.updates["has_circle"] == 1

    def test_cleanup(self, tmp_path):
        p = self._make_processor(tmp_path)
        p.cleanup()
        assert p.model is None


# ===========================================================================
# Line Detection Processor
# ===========================================================================

class TestLineDetectionProcessor:
    @pytest.fixture(autouse=True)
    def _mock_yolo(self):
        mock_yolo_class = MagicMock()

        fake_result = MagicMock()
        # 2 detected lines
        fake_result.boxes.data.cpu.return_value.numpy.return_value = np.array([
            [10, 20, 90, 30, 0.9, 0],
            [10, 40, 90, 50, 0.85, 0],
        ])
        fake_result.boxes.__len__ = lambda s: 2
        fake_result.boxes.__bool__ = lambda s: True

        mock_model = MagicMock()
        mock_model.predict.return_value = [fake_result]
        mock_yolo_class.return_value = mock_model

        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_class)}):
            import ml_pipeline.processors.line_detection_processor as mod
            mod.YOLO = mock_yolo_class
            self.mod = mod
            yield

    def _make_processor(self, tmp_path):
        model_path = tmp_path / "line.pt"
        model_path.touch()
        config = {
            "model_path": str(model_path),
            "config": {
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45,
                "model_version": "1.0",
                "skip_processed": True,
            },
        }
        logger = logging.getLogger("test_line")
        return self.mod.LineDetectionProcessor(config, logger)

    def test_get_metadata(self, tmp_path):
        m = self._make_processor(tmp_path).get_metadata()
        assert m.name == "linedetection"

    def test_should_process_no_version(self, tmp_path):
        p = self._make_processor(tmp_path)
        assert p.should_process(_make_fragment(line_detection_model_version=None)) is True

    def test_should_process_same_version(self, tmp_path):
        p = self._make_processor(tmp_path)
        assert p.should_process(_make_fragment(line_detection_model_version="1.0")) is False

    def test_should_process_skip_disabled(self, tmp_path):
        model_path = tmp_path / "line.pt"
        model_path.touch()
        config = {
            "model_path": str(model_path),
            "config": {"confidence_threshold": 0.25, "iou_threshold": 0.45, "model_version": "1.0", "skip_processed": False},
        }
        logger = logging.getLogger("test_line")
        p = self.mod.LineDetectionProcessor(config, logger)
        assert p.should_process(_make_fragment(line_detection_model_version="1.0")) is True

    def test_process_image_not_found(self, tmp_path):
        p = self._make_processor(tmp_path)
        result = p.process(_make_fragment(image_path="nope.png"), str(tmp_path))
        assert result.success is False

    def test_process_success(self, tmp_path):
        p = self._make_processor(tmp_path)
        data_dir = _create_test_image(tmp_path)
        result = p.process(_make_fragment(), data_dir)
        assert result.success is True
        assert result.updates["line_count"] == 2
        data = json.loads(result.updates["line_detection_data"])
        assert data["num_lines"] == 2

    def test_process_no_lines(self, tmp_path):
        """Test when no boxes detected."""
        p = self._make_processor(tmp_path)
        # Override the mock to return empty boxes
        fake_result = MagicMock()
        fake_result.boxes = None
        p.model.predict.return_value = [fake_result]

        data_dir = _create_test_image(tmp_path)
        result = p.process(_make_fragment(), data_dir)
        assert result.success is True
        assert result.updates["line_count"] == 0

    def test_cleanup(self, tmp_path):
        p = self._make_processor(tmp_path)
        p.cleanup()
        assert p.model is None


# ===========================================================================
# Scale Detection Processor
# ===========================================================================

class TestScaleDetectionProcessor:
    @pytest.fixture(autouse=True)
    def _mock_detect(self):
        with patch(
            "ml_pipeline.processors.scale_detection_processor.detect_scale_ratio"
        ) as mock_fn:
            self._mock_detect_fn = mock_fn
            yield

    def _make_processor(self):
        config = {
            "config": {
                "model_version": "1.0",
                "visualize": False,
                "skip_processed": True,
                "ocr_enabled": True,
            },
        }
        logger = logging.getLogger("test_scale")
        from ml_pipeline.processors.scale_detection_processor import ScaleDetectionProcessor
        return ScaleDetectionProcessor(config, logger)

    def test_get_metadata(self):
        p = self._make_processor()
        m = p.get_metadata()
        assert m.name == "scale_detection"

    def test_should_process_no_data(self):
        p = self._make_processor()
        assert p.should_process(_make_fragment(scale_unit=None)) is True

    def test_should_process_same_version(self):
        p = self._make_processor()
        assert p.should_process(_make_fragment(scale_unit="cm", scale_model_version="1.0")) is False

    def test_should_process_different_version(self):
        p = self._make_processor()
        assert p.should_process(_make_fragment(scale_unit="cm", scale_model_version="0.5")) is True

    def test_should_process_skip_disabled(self):
        config = {
            "config": {
                "model_version": "1.0",
                "visualize": False,
                "skip_processed": False,
                "ocr_enabled": True,
            },
        }
        logger = logging.getLogger("test_scale")
        from ml_pipeline.processors.scale_detection_processor import ScaleDetectionProcessor
        p = ScaleDetectionProcessor(config, logger)
        assert p.should_process(_make_fragment(scale_unit="cm", scale_model_version="1.0")) is True

    def test_process_image_not_found(self, tmp_path):
        p = self._make_processor()
        result = p.process(_make_fragment(image_path="nope.png"), str(tmp_path))
        assert result.success is False

    def test_process_success(self, tmp_path):
        p = self._make_processor()
        data_dir = _create_test_image(tmp_path)
        self._mock_detect_fn.return_value = {
            "status": "success",
            "unit": "cm",
            "pixels_per_unit": 120.5,
            "num_ticks": 5,
            "tick_positions": [10, 30, 50, 70, 90],
        }
        result = p.process(_make_fragment(), data_dir)
        assert result.success is True
        assert result.updates["scale_unit"] == "cm"
        assert result.updates["pixels_per_unit"] == 120.5

    def test_process_detection_failed(self, tmp_path):
        p = self._make_processor()
        data_dir = _create_test_image(tmp_path)
        self._mock_detect_fn.return_value = {
            "status": "error",
            "message": "No ruler found",
        }
        result = p.process(_make_fragment(), data_dir)
        assert result.success is True  # still "success" but with None values
        assert result.updates["scale_unit"] is None

    def test_process_exception(self, tmp_path):
        p = self._make_processor()
        data_dir = _create_test_image(tmp_path)
        self._mock_detect_fn.side_effect = RuntimeError("boom")
        result = p.process(_make_fragment(), data_dir)
        assert result.success is False
        assert "boom" in result.error

    def test_cleanup(self):
        p = self._make_processor()
        p.cleanup()  # should not raise


# ===========================================================================
# Classification Processor (line count CNN)
# ===========================================================================

class TestClassificationProcessor:
    @pytest.fixture(autouse=True)
    def _mock_deps(self, tmp_path):
        """Mock torch and LineCountCNN."""
        self.tmp_path = tmp_path

        mock_torch = MagicMock()
        mock_torch.device.return_value = "cpu"
        mock_torch.cuda.is_available.return_value = False
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        # softmax returns tensor with probs
        probs = MagicMock()
        probs.argmax.return_value.item.return_value = 5  # predicted 5 lines
        probs.max.return_value.item.return_value = 0.92
        mock_torch.softmax.return_value = probs

        self._mock_torch = mock_torch
        self._probs = probs

        mock_model = MagicMock()
        mock_model.return_value = MagicMock()  # output tensor

        mock_cnn_class = MagicMock(return_value=mock_model)

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "torchvision": MagicMock(),
            "torchvision.transforms": MagicMock(),
            "PIL": MagicMock(),
            "PIL.Image": MagicMock(),
        }):
            import importlib
            import ml_pipeline.processors.classification_processor as mod
            mod.torch = mock_torch
            mod.LineCountCNN = mock_cnn_class
            mod.T = MagicMock()
            mod.Image = MagicMock()

            self.mod = mod
            yield

    def _make_processor(self):
        model_path = self.tmp_path / "model.pt"
        model_path.touch()
        meta_path = self.tmp_path / "meta.json"
        meta_path.write_text('{"max_lines": 15, "img_size": 224}')

        config = {
            "model_path": str(model_path),
            "meta_path": str(meta_path),
            "skip_processed": True,
            "config": {"model_version": "1.0", "batch_size": 16},
        }
        logger = logging.getLogger("test_cls")
        return self.mod.ClassificationProcessor(config, logger)

    def test_get_metadata(self):
        m = self._make_processor().get_metadata()
        assert m.name == "classification"

    def test_should_process_no_data(self):
        p = self._make_processor()
        assert p.should_process(_make_fragment(line_count=None)) is True

    def test_should_process_same_version(self):
        p = self._make_processor()
        assert p.should_process(_make_fragment(line_count=5, classification_model_version="1.0")) is False

    def test_process_image_not_found(self):
        p = self._make_processor()
        result = p.process(_make_fragment(image_path="nope.png"), str(self.tmp_path))
        assert result.success is False

    def test_process_success(self):
        """Cover classification_processor.py lines 126-153 (inference path)."""
        p = self._make_processor()
        data_dir = _create_test_image(self.tmp_path)
        # PIL.Image.open needs to return a mock with .convert().unsqueeze().to()
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        self.mod.Image.open.return_value = mock_img

        result = p.process(_make_fragment(), data_dir)
        assert result.success is True
        assert result.updates["line_count"] == 5
        assert result.updates["classification_model_version"] == "1.0"
        assert result.metadata["confidence"] == 0.92

    def test_process_exception(self):
        """Cover the except block in process()."""
        p = self._make_processor()
        data_dir = _create_test_image(self.tmp_path)
        self.mod.Image.open.side_effect = RuntimeError("corrupt image")
        result = p.process(_make_fragment(), data_dir)
        assert result.success is False
        assert "corrupt image" in result.error

    def test_cleanup_with_cuda(self):
        """Cover cleanup() when cuda is available."""
        p = self._make_processor()
        self._mock_torch.cuda.is_available.return_value = True
        p.cleanup()
        self._mock_torch.cuda.empty_cache.assert_called_once()

    def test_should_process_different_version(self):
        """Cover the version-mismatch branch in should_process."""
        p = self._make_processor()
        frag = _make_fragment(line_count=3, classification_model_version="0.5")
        assert p.should_process(frag) is True


# ===========================================================================
# Script Type Classification Processor
# ===========================================================================

class TestScriptTypeClassificationProcessor:
    @pytest.fixture(autouse=True)
    def _mock_deps(self, tmp_path):
        self.tmp_path = tmp_path

        mock_torch = MagicMock()
        mock_torch.device.return_value = "cpu"
        mock_torch.cuda.is_available.return_value = False
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        probs_tensor = MagicMock()
        probs_tensor.argmax.return_value.item.return_value = 0
        probs_tensor.max.return_value.item.return_value = 0.85
        probs_tensor.__getitem__ = lambda self, idx: MagicMock(
            __getitem__=lambda s, i: MagicMock(item=lambda: 0.85 if i == 0 else 0.05)
        )
        mock_torch.softmax.return_value = probs_tensor

        mock_load_model = MagicMock(return_value=MagicMock())
        mock_efficient = MagicMock()

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "torchvision": MagicMock(),
            "torchvision.transforms": MagicMock(),
            "PIL": MagicMock(),
            "PIL.Image": MagicMock(),
            "yaml": pytest.importorskip("yaml"),
        }):
            import ml_pipeline.processors.script_type_classification_processor as mod
            mod.torch = mock_torch
            mod.load_model = mock_load_model
            mod.EfficientScriptTypeClassifier = mock_efficient
            mod.T = MagicMock()
            mod.Image = MagicMock()

            self.mod = mod
            yield

    def _make_processor(self):
        model_path = self.tmp_path / "model.pt"
        model_path.touch()
        meta_path = self.tmp_path / "metadata.yaml"
        meta_path.write_text("class_names:\n  - Brahmi\n  - Kharosthi\nimg_size: 224\n")

        config = {
            "model_path": str(model_path),
            "meta_path": str(meta_path),
            "config": {
                "model_version": "2.0",
                "batch_size": 16,
                "confidence_threshold": 0.0,
                "skip_processed": True,
            },
        }
        logger = logging.getLogger("test_script")
        return self.mod.ScriptTypeClassificationProcessor(config, logger)

    def test_get_metadata(self):
        m = self._make_processor().get_metadata()
        assert m.name == "script_type_classification"

    def test_should_process_no_data(self):
        p = self._make_processor()
        assert p.should_process(_make_fragment(script_type=None)) is True

    def test_should_process_same_version(self):
        p = self._make_processor()
        frag = _make_fragment(
            script_type="Brahmi",
            script_type_classification_model_version="2.0",
        )
        assert p.should_process(frag) is False

    def test_process_image_not_found(self):
        p = self._make_processor()
        result = p.process(_make_fragment(image_path="nope.png"), str(self.tmp_path))
        assert result.success is False

    def test_process_success(self):
        """Cover script_type_classification_processor.py lines 138-193."""
        p = self._make_processor()
        data_dir = _create_test_image(self.tmp_path)

        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        self.mod.Image.open.return_value = mock_img

        # probs[0][i].item() needs to work for range(num_classes)
        probs_tensor = self.mod.torch.softmax.return_value
        row = MagicMock()
        row.__getitem__ = lambda s, i: MagicMock(item=lambda: 0.85 if i == 0 else 0.15)
        probs_tensor.__getitem__ = lambda s, idx: row

        result = p.process(_make_fragment(), data_dir)
        assert result.success is True
        assert result.updates["script_type"] == "Brahmi"
        assert "script_type_classification_data" in result.updates
        assert result.metadata["confidence"] == 0.85

    def test_process_exception(self):
        """Cover the except block in process()."""
        p = self._make_processor()
        data_dir = _create_test_image(self.tmp_path)
        self.mod.Image.open.side_effect = RuntimeError("bad")
        result = p.process(_make_fragment(), data_dir)
        assert result.success is False
        assert "bad" in result.error

    def test_cleanup_with_cuda(self):
        """Cover cleanup() when cuda is available."""
        p = self._make_processor()
        self.mod.torch.cuda.is_available.return_value = True
        p.cleanup()
        self.mod.torch.cuda.empty_cache.assert_called_once()

    def test_should_process_different_version(self):
        """Cover version-mismatch branch."""
        p = self._make_processor()
        frag = _make_fragment(script_type="Brahmi", script_type_classification_model_version="1.0")
        assert p.should_process(frag) is True

    def test_cleanup(self):
        p = self._make_processor()
        p.cleanup()
