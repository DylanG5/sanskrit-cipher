"""Tests for edge_detection_processor.py – EdgeExtractor and EdgeDetectionProcessor.

These tests use synthetic contours and masks to exercise the pure-CV logic
without needing actual manuscript images or ML models.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from ml_pipeline.core.processor import FragmentRecord, ProcessingResult


# Import EdgeExtractor and EdgeDetectionProcessor
from ml_pipeline.processors.edge_detection_processor import (
    EdgeDetectionProcessor,
    EdgeExtractor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rect_contour(x0, y0, x1, y1):
    """Return a dense contour (Nx1x2 int32) tracing a rectangle."""
    pts = []
    # top edge left→right
    for x in range(x0, x1 + 1):
        pts.append([x, y0])
    # right edge top→bottom
    for y in range(y0 + 1, y1 + 1):
        pts.append([x1, y])
    # bottom edge right→left
    for x in range(x1 - 1, x0 - 1, -1):
        pts.append([x, y1])
    # left edge bottom→top
    for y in range(y1 - 1, y0, -1):
        pts.append([x0, y])
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def _rect_mask(h, w, x0, y0, x1, y1):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1 + 1, x0:x1 + 1] = 255
    return mask


def _make_fragment(**kw):
    defaults = dict(id=1, fragment_id="F001", image_path="uploads/f001.png")
    defaults.update(kw)
    return FragmentRecord(**defaults)


# ---------------------------------------------------------------------------
# EdgeExtractor unit tests
# ---------------------------------------------------------------------------

class TestEdgeExtractor:
    def setup_method(self):
        self.ext = EdgeExtractor()

    def test_classify_edges_full_rectangle(self):
        """A perfect rectangle should have at least some borders detected."""
        contour = _rect_contour(10, 10, 200, 150)
        mask = _rect_mask(200, 250, 10, 10, 200, 150)
        result = self.ext.classify_edges(contour, mask)
        # At least some sides should be border edges
        assert len(result["border_edges"]) >= 2
        assert result["piece_type"] in ("corner", "edge", "interior")

    def test_classify_edges_returns_expected_keys(self):
        contour = _rect_contour(10, 10, 100, 80)
        mask = _rect_mask(120, 150, 10, 10, 100, 80)
        result = self.ext.classify_edges(contour, mask)
        for key in ("contour", "bbox", "border_edges", "tear_edges", "scores", "piece_type"):
            assert key in result

    def test_classify_edges_hull_segments_full_rectangle(self):
        """Hull-segment method on a rectangle."""
        contour = _rect_contour(10, 10, 200, 150)
        mask = _rect_mask(200, 250, 10, 10, 200, 150)
        result = self.ext.classify_edges_hull_segments(contour, mask)
        assert "piece_type" in result
        assert result["piece_type"] in ("corner", "edge", "interior")

    def test_classify_edges_hull_segments_returns_expected_keys(self):
        contour = _rect_contour(10, 10, 200, 150)
        mask = _rect_mask(200, 250, 10, 10, 200, 150)
        result = self.ext.classify_edges_hull_segments(contour, mask)
        for key in ("contour", "bbox", "border_edges", "tear_edges", "scores", "piece_type"):
            assert key in result

    def test_classify_edges_oriented_runs_full_rectangle(self):
        contour = _rect_contour(10, 10, 200, 150)
        mask = _rect_mask(200, 250, 10, 10, 200, 150)
        result = self.ext.classify_edges_oriented_runs(contour, mask)
        assert set(result["border_edges"]) == {
            "top_edge",
            "bottom_edge",
            "left_edge",
            "right_edge",
        }
        assert result["piece_type"] == "corner"

    def test_classify_edges_oriented_runs_sparse_contour_uses_mask(self):
        mask = _rect_mask(200, 250, 10, 10, 200, 150)
        sparse_contour = np.array(
            [[10, 10], [10, 150], [200, 150], [200, 10]],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        result = self.ext.classify_edges_oriented_runs(sparse_contour, mask)
        assert set(result["border_edges"]) == {
            "top_edge",
            "bottom_edge",
            "left_edge",
            "right_edge",
        }

    def test_classify_edges_oriented_runs_rotated_rectangle(self):
        mask = np.zeros((300, 300), dtype=np.uint8)
        rr = cv2.boxPoints(((150, 150), (140, 80), 25)).astype(np.int32)
        cv2.fillPoly(mask, [rr], 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        result = self.ext.classify_edges_oriented_runs(contour, mask)
        assert set(result["border_edges"]) == {
            "top_edge",
            "bottom_edge",
            "left_edge",
            "right_edge",
        }

    def test_classify_edges_oriented_runs_concave_notch_rejects_torn_side(self):
        mask = np.zeros((300, 300), dtype=np.uint8)
        poly = np.array(
            [(40, 40), (220, 40), (220, 90), (170, 90), (170, 150), (220, 150), (220, 220), (40, 220)],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [poly], 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        result = self.ext.classify_edges_oriented_runs(contour, mask)
        assert "top_edge" in result["border_edges"]
        assert "bottom_edge" in result["border_edges"]
        assert "left_edge" in result["border_edges"]
        assert "right_edge" not in result["border_edges"]

    def test_classify_edges_oriented_runs_scaled_full_rectangle(self):
        contour = _rect_contour(10, 10, 200, 150)
        mask = _rect_mask(200, 250, 10, 10, 200, 150)
        result = self.ext.classify_edges_oriented_runs_scaled(
            contour,
            mask,
            pixels_per_unit=220.0,
            target_pixels_per_unit=150.0,
        )
        assert set(result["border_edges"]) == {
            "top_edge",
            "bottom_edge",
            "left_edge",
            "right_edge",
        }
        assert "scale_normalization" in result

    def test_contiguous_runs(self):
        mask_bool = np.array([False, True, True, False, True, False])
        runs = self.ext._contiguous_runs(mask_bool)
        assert runs == [(1, 2), (4, 4)]

    def test_contiguous_runs_empty(self):
        mask_bool = np.array([False, False, False])
        assert self.ext._contiguous_runs(mask_bool) == []

    def test_merge_wraparound_runs(self):
        runs = [(0, 2), (5, 7), (8, 9)]
        merged = self.ext._merge_wraparound_runs(runs, n=10)
        assert merged[0] == (8, 2)

    def test_merge_wraparound_no_wrap(self):
        runs = [(2, 5)]
        assert self.ext._merge_wraparound_runs(runs, n=10) == [(2, 5)]

    def test_run_span_along_side_horizontal(self):
        pts = np.array([[10, 5], [50, 5], [100, 5]])
        span = self.ext._run_span_along_side(pts, "top_edge")
        assert span == 90.0

    def test_run_span_along_side_vertical(self):
        pts = np.array([[5, 10], [5, 50], [5, 100]])
        span = self.ext._run_span_along_side(pts, "left_edge")
        assert span == 90.0

    def test_mean_line_fit_error_straight(self):
        """Perfectly collinear points should have ~0 error."""
        pts = np.array([[i, i] for i in range(50)], dtype=np.float32)
        assert self.ext._mean_line_fit_error(pts) < 1.0

    def test_mean_line_fit_error_noisy(self):
        """Noisy points should have larger error."""
        rng = np.random.RandomState(42)
        pts = np.column_stack([
            np.arange(50),
            np.arange(50) + rng.normal(0, 10, 50),
        ]).astype(np.float32)
        assert self.ext._mean_line_fit_error(pts) > 1.0

    def test_extract_straight_segments(self):
        # A straight segment
        hpts = np.array([[i, 0] for i in range(100)])
        segs = self.ext._extract_straight_segments(hpts, err_thresh=5.0, min_len_pts=10)
        assert len(segs) >= 1
        assert len(segs[0]) >= 10

    def test_line_angle_deg_horizontal(self):
        pts = np.array([[0, 50], [100, 50]], dtype=np.float32)
        ang = self.ext._line_angle_deg(pts)
        assert ang < 5.0  # nearly horizontal

    def test_line_angle_deg_vertical(self):
        pts = np.array([[50, 0], [50, 100]], dtype=np.float32)
        ang = self.ext._line_angle_deg(pts)
        assert ang > 85.0  # nearly vertical

    def test_classify_edges_empty_mask_raises(self):
        contour = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int32).reshape(-1, 1, 2)
        mask = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="Empty mask"):
            self.ext.classify_edges(contour, mask)


# ---------------------------------------------------------------------------
# EdgeDetectionProcessor tests (mock image loading)
# ---------------------------------------------------------------------------

class TestEdgeDetectionProcessor:
    def _make_processor(self, edge_method="bbox_runs"):
        config = {
            "config": {
                "model_version": "1.0",
                "edge_method": edge_method,
                "near_frac": 0.05,
                "min_run_frac": 0.05,
                "max_mean_line_err": 18.0,
                "hull_err_thresh": 3.5,
                "hull_min_len_pts": 35,
                "hull_min_cov": 0.45,
                "hull_max_err": 4.5,
                "hull_near_frac": 0.06,
                "hull_angle_h": 12.0,
                "hull_angle_v": 78.0,
                "oriented_near_frac": 0.04,
                "oriented_min_run_frac": 0.40,
                "oriented_max_line_err_frac": 0.02,
                "oriented_max_side_offset_frac": 0.025,
                "oriented_angle_tol_deg": 18.0,
                "oriented_min_confidence": 0.60,
                "oriented_smoothing_window": 7,
                "oriented_trim_frac": 0.10,
                "oriented_scaled_target_pixels_per_unit": 150.0,
                "oriented_scaled_crop_pad": 4,
                "oriented_scaled_max_dimension": 8192,
            }
        }
        logger = logging.getLogger("test_edge")
        return EdgeDetectionProcessor(config, logger)

    def test_get_metadata(self):
        p = self._make_processor()
        m = p.get_metadata()
        assert m.name == "edgedetection"
        assert m.version == "1.0"

    def test_should_process_always_true(self):
        p = self._make_processor()
        frag = _make_fragment()
        assert p.should_process(frag) is True

    def test_process_image_not_found(self, tmp_path):
        p = self._make_processor()
        frag = _make_fragment(image_path="missing.png")
        result = p.process(frag, str(tmp_path))
        assert result.success is False
        assert "not found" in result.error

    def test_process_no_segmentation_coords(self, tmp_path):
        """If fragment has no segmentation_coords, should fail gracefully."""
        p = self._make_processor()
        # Create a dummy image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img_path = tmp_path / "uploads" / "f001.png"
        img_path.parent.mkdir(parents=True)
        cv2.imwrite(str(img_path), img)

        frag = _make_fragment(segmentation_coords=None)
        result = p.process(frag, str(tmp_path))
        assert result.success is False
        assert "segmentation_coords" in result.error.lower() or "No segmentation" in result.error

    def test_process_empty_contours(self, tmp_path):
        """segmentation_coords with empty contours list should fail."""
        p = self._make_processor()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img_path = tmp_path / "uploads" / "f001.png"
        img_path.parent.mkdir(parents=True)
        cv2.imwrite(str(img_path), img)

        seg_data = json.dumps({"contours": []})
        frag = _make_fragment(segmentation_coords=seg_data)
        result = p.process(frag, str(tmp_path))
        assert result.success is False

    def test_process_bbox_runs_method(self, tmp_path):
        """Full process with bbox_runs method on a rectangular fragment."""
        p = self._make_processor("bbox_runs")
        h, w = 200, 250

        # Create image
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img_path = tmp_path / "uploads" / "f001.png"
        img_path.parent.mkdir(parents=True)
        cv2.imwrite(str(img_path), img)

        # Create rectangular contour data
        contour = _rect_contour(10, 10, 200, 150)
        contours_list = contour.squeeze().tolist()
        seg_data = json.dumps({"contours": [contours_list]})

        frag = _make_fragment(segmentation_coords=seg_data)
        result = p.process(frag, str(tmp_path))
        assert result.success is True
        assert "has_top_edge" in result.updates
        assert "has_bottom_edge" in result.updates
        assert "has_left_edge" in result.updates
        assert "has_right_edge" in result.updates
        assert result.updates["processing_status"] == "completed"

    def test_process_hull_segments_method(self, tmp_path):
        """Full process with hull_segments method."""
        p = self._make_processor("hull_segments")
        h, w = 200, 250

        img = np.zeros((h, w, 3), dtype=np.uint8)
        img_path = tmp_path / "uploads" / "f001.png"
        img_path.parent.mkdir(parents=True)
        cv2.imwrite(str(img_path), img)

        contour = _rect_contour(10, 10, 200, 150)
        contours_list = contour.squeeze().tolist()
        seg_data = json.dumps({"contours": [contours_list]})

        frag = _make_fragment(segmentation_coords=seg_data)
        result = p.process(frag, str(tmp_path))
        assert result.success is True
        assert "has_top_edge" in result.updates

    def test_process_oriented_runs_method(self, tmp_path):
        p = self._make_processor("oriented_runs")
        h, w = 200, 250

        img = np.zeros((h, w, 3), dtype=np.uint8)
        img_path = tmp_path / "uploads" / "f001.png"
        img_path.parent.mkdir(parents=True)
        cv2.imwrite(str(img_path), img)

        contour = _rect_contour(10, 10, 200, 150)
        contours_list = contour.squeeze().tolist()
        seg_data = json.dumps({"contours": [contours_list]})

        frag = _make_fragment(segmentation_coords=seg_data)
        result = p.process(frag, str(tmp_path))
        assert result.success is True
        assert result.updates["has_top_edge"] is True
        assert result.updates["has_bottom_edge"] is True
        assert result.updates["has_left_edge"] is True
        assert result.updates["has_right_edge"] is True

    def test_process_oriented_runs_scaled_method(self, tmp_path):
        p = self._make_processor("oriented_runs_scaled")
        h, w = 200, 250

        img = np.zeros((h, w, 3), dtype=np.uint8)
        img_path = tmp_path / "uploads" / "f001.png"
        img_path.parent.mkdir(parents=True)
        cv2.imwrite(str(img_path), img)

        contour = _rect_contour(10, 10, 200, 150)
        contours_list = contour.squeeze().tolist()
        seg_data = json.dumps({"contours": [contours_list]})

        frag = _make_fragment(segmentation_coords=seg_data, pixels_per_unit=220.0)
        result = p.process(frag, str(tmp_path))
        assert result.success is True
        assert result.updates["has_top_edge"] is True
        assert result.updates["has_bottom_edge"] is True
        assert result.updates["has_left_edge"] is True
        assert result.updates["has_right_edge"] is True

    def test_cleanup_does_nothing(self):
        p = self._make_processor()
        p.cleanup()  # should not raise
