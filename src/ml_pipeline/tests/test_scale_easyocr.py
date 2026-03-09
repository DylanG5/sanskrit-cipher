"""Tests for scale_detection_easyocr.py – utility functions for ruler detection."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import pytest

from ml_pipeline.processors.scale_detection_easyocr import (
    calculate_pixels_per_unit,
    detect_scale_ratio,
    find_scale_text_region,
)


# ---------------------------------------------------------------------------
# calculate_pixels_per_unit
# ---------------------------------------------------------------------------

def test_calculate_pixels_per_unit_normal():
    ticks = [100, 200, 300, 400]
    result = calculate_pixels_per_unit(ticks)
    assert result == 100.0


def test_calculate_pixels_per_unit_single_tick():
    result = calculate_pixels_per_unit([100])
    assert result is None


def test_calculate_pixels_per_unit_empty():
    result = calculate_pixels_per_unit([])
    assert result is None


def test_calculate_pixels_per_unit_two_ticks():
    result = calculate_pixels_per_unit([50, 150])
    assert result == 100.0


def test_calculate_pixels_per_unit_uneven_spacing():
    """Median should be robust to one outlier gap."""
    ticks = [100, 200, 300, 500]  # gaps: 100, 100, 200
    result = calculate_pixels_per_unit(ticks)
    assert result == 100.0  # median of [100, 100, 200]


# ---------------------------------------------------------------------------
# find_scale_text_region (mocked OCR)
# ---------------------------------------------------------------------------

def test_find_scale_text_region_fallback_no_ocr():
    """When OCR is unavailable, fallback should return 'cm'."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    with patch("ml_pipeline.processors.scale_detection_easyocr.OCR_AVAILABLE", False):
        unit, x_end = find_scale_text_region(img)
    assert unit == "cm"
    assert isinstance(x_end, int)


def test_find_scale_text_region_ocr_finds_cm():
    """OCR finds 'cm' text."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    mock_reader = MagicMock()
    # EasyOCR result format: [([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], text, conf), ...]
    mock_reader.readtext.return_value = [
        ([[5, 5], [30, 5], [30, 20], [5, 20]], "cm", 0.95),
    ]
    with (
        patch("ml_pipeline.processors.scale_detection_easyocr.OCR_AVAILABLE", True),
        patch("ml_pipeline.processors.scale_detection_easyocr.get_ocr_reader", return_value=mock_reader),
    ):
        unit, x_end = find_scale_text_region(img)
    assert unit == "cm"


def test_find_scale_text_region_ocr_finds_mm():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [
        ([[5, 5], [30, 5], [30, 20], [5, 20]], "mm", 0.90),
    ]
    with (
        patch("ml_pipeline.processors.scale_detection_easyocr.OCR_AVAILABLE", True),
        patch("ml_pipeline.processors.scale_detection_easyocr.get_ocr_reader", return_value=mock_reader),
    ):
        unit, x_end = find_scale_text_region(img)
    assert unit == "mm"


def test_find_scale_text_region_ocr_no_match():
    """OCR doesn't find cm/mm – falls through to fallback."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [
        ([[5, 5], [30, 5], [30, 20], [5, 20]], "abcdef", 0.90),
    ]
    with (
        patch("ml_pipeline.processors.scale_detection_easyocr.OCR_AVAILABLE", True),
        patch("ml_pipeline.processors.scale_detection_easyocr.get_ocr_reader", return_value=mock_reader),
    ):
        unit, x_end = find_scale_text_region(img)
    assert unit == "cm"  # fallback default


def test_find_scale_text_region_ocr_exception():
    """OCR raises – should fall through to fallback."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    mock_reader = MagicMock()
    mock_reader.readtext.side_effect = RuntimeError("ocr fail")
    with (
        patch("ml_pipeline.processors.scale_detection_easyocr.OCR_AVAILABLE", True),
        patch("ml_pipeline.processors.scale_detection_easyocr.get_ocr_reader", return_value=mock_reader),
    ):
        unit, x_end = find_scale_text_region(img)
    assert unit == "cm"


# ---------------------------------------------------------------------------
# detect_scale_ratio (integration – uses synthetic images)
# ---------------------------------------------------------------------------

def _create_ruler_image(tmp_path, tick_positions, height=200, width=500, filename="ruler.png"):
    """Create a synthetic image with white ticks on a dark ruler bar."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Dark ruler bar at the bottom
    img[height - 50:, :] = 20
    # Draw white tick marks
    for x in tick_positions:
        if 0 <= x < width:
            cv2.line(img, (x, height - 50), (x, height), (255, 255, 255), 2)
    path = tmp_path / filename
    cv2.imwrite(str(path), img)
    return str(path)


def test_detect_scale_ratio_image_not_found(tmp_path):
    result = detect_scale_ratio(str(tmp_path / "nonexistent.png"))
    assert result["status"] == "error"
    assert "load" in result["message"].lower()


def test_detect_scale_ratio_success(tmp_path):
    """Ruler with clear ticks should return success."""
    ticks = [100, 200, 300, 400]
    img_path = _create_ruler_image(tmp_path, ticks)

    with patch("ml_pipeline.processors.scale_detection_easyocr.find_scale_text_region", return_value=("cm", 10)):
        with patch("ml_pipeline.processors.scale_detection_easyocr.detect_ruler_ticks_simple", return_value=ticks):
            result = detect_scale_ratio(img_path)
    assert result["status"] == "success"
    assert result["unit"] == "cm"
    assert result["pixels_per_unit"] == 100.0
    assert result["num_ticks"] == 4


def test_detect_scale_ratio_not_enough_ticks(tmp_path):
    """Too few ticks should return error."""
    img_path = _create_ruler_image(tmp_path, [200])

    with patch("ml_pipeline.processors.scale_detection_easyocr.find_scale_text_region", return_value=("cm", 10)):
        with patch("ml_pipeline.processors.scale_detection_easyocr.detect_ruler_ticks_simple", return_value=[200]):
            result = detect_scale_ratio(img_path)
    assert result["status"] == "error"
    assert "ticks" in result["message"].lower()


def test_detect_scale_ratio_with_visualization(tmp_path):
    """Test that visualization is saved when enabled."""
    ticks = [100, 200, 300]
    img_path = _create_ruler_image(tmp_path, ticks)
    vis_dir = str(tmp_path / "vis")

    with patch("ml_pipeline.processors.scale_detection_easyocr.find_scale_text_region", return_value=("cm", 10)):
        with patch("ml_pipeline.processors.scale_detection_easyocr.detect_ruler_ticks_simple", return_value=ticks):
            result = detect_scale_ratio(img_path, visualize=True, output_dir=vis_dir)
    assert result["status"] == "success"
    # Visualization file should exist
    assert (tmp_path / "vis" / "ruler_detection.jpg").exists()


def test_detect_scale_ratio_unit_none_fallback(tmp_path):
    """When find_scale_text_region returns None, should default to cm."""
    ticks = [100, 200, 300]
    img_path = _create_ruler_image(tmp_path, ticks)

    with patch("ml_pipeline.processors.scale_detection_easyocr.find_scale_text_region", return_value=(None, 50)):
        with patch("ml_pipeline.processors.scale_detection_easyocr.detect_ruler_ticks_simple", return_value=ticks):
            result = detect_scale_ratio(img_path)
    assert result["status"] == "success"
    assert result["unit"] == "cm"
