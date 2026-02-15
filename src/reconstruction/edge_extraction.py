"""
Edge extraction with scale-aware descriptors.

Adapted from edge_matching_v2/enhanced_edge_matching.py (git commit f3a3a3c).
Reuses ContourSegmenter and feature extraction; adds ScaleAwareDescriptor
that carries real-world (cm-space) measurements.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

from .config import (
    CURVATURE_HISTOGRAM_BINS,
    CURVATURE_WINDOW,
    CORNER_THRESHOLD,
    MIN_CORNER_SPACING_RATIO,
    MIN_EDGE_LENGTH_CM,
    MIN_SEGMENT_LENGTH,
    NUM_FOURIER_COEFFICIENTS,
    NUM_RESAMPLE_POINTS,
    STRAIGHTNESS_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EdgeSegment:
    """A single contiguous edge segment from contour segmentation."""
    segment_id: int
    start_idx: int
    end_idx: int
    points: np.ndarray
    edge_type: str          # 'tear' or 'border'
    position_label: str     # 'top', 'bottom', 'left', 'right', 'corner'
    length: float
    straightness_ratio: float
    dominant_direction: float
    centroid: np.ndarray


@dataclass
class ScaleAwareDescriptor:
    """Edge descriptor carrying both normalised features and real-world measurements."""

    # Identity
    fragment_id: str
    edge_name: str              # e.g. "left_seg2"
    edge_type: str              # 'tear' or 'border'
    position_label: str         # 'top', 'bottom', 'left', 'right'

    # Shape features (scale-invariant, from v2)
    normalized_points: np.ndarray   # NUM_RESAMPLE_POINTS pts, centered + normalized
    fourier_descriptors: np.ndarray # NUM_FOURIER_COEFFICIENTS coeffs
    curvature_histogram: np.ndarray # CURVATURE_HISTOGRAM_BINS bins
    complexity_score: float
    orientation: float

    # Pixel-space (raw)
    length_px: float
    points_px: np.ndarray

    # Real-world scale
    pixels_per_unit: float
    scale_unit: str             # 'cm' or 'mm'
    length_cm: float
    points_cm: np.ndarray
    centroid_cm: np.ndarray

    # Fragment metadata (for filtering)
    line_count: Optional[int] = None
    script_type: Optional[str] = None
    image_width_px: int = 0
    image_height_px: int = 0


# ---------------------------------------------------------------------------
# ContourSegmenter  (ported from v2 as-is)
# ---------------------------------------------------------------------------

class ContourSegmenter:
    """Curvature-based contour segmentation into edge segments."""

    def __init__(
        self,
        curvature_window: int = CURVATURE_WINDOW,
        corner_threshold: float = CORNER_THRESHOLD,
        min_segment_length: int = MIN_SEGMENT_LENGTH,
        straightness_threshold: float = STRAIGHTNESS_THRESHOLD,
        min_corner_spacing_ratio: float = MIN_CORNER_SPACING_RATIO,
    ):
        self.curvature_window = curvature_window
        self.corner_threshold = corner_threshold
        self.min_segment_length = min_segment_length
        self.straightness_threshold = straightness_threshold
        self.min_corner_spacing_ratio = min_corner_spacing_ratio

    # -- curvature / corner detection --

    def compute_curvature(self, contour: np.ndarray) -> np.ndarray:
        n = len(contour)
        if n < self.curvature_window:
            return np.zeros(n)
        half_window = self.curvature_window // 2
        curvature = np.zeros(n)
        for i in range(n):
            idx_before = (i - half_window) % n
            idx_after = (i + half_window) % n
            v1 = contour[i] - contour[idx_before]
            v2 = contour[idx_after] - contour[i]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
                curvature[i] = np.pi - np.arccos(cos_angle)
        return curvature

    def detect_corners(self, curvature: np.ndarray) -> List[int]:
        n = len(curvature)
        min_distance = max(int(n * self.min_corner_spacing_ratio), 10)
        peaks, _ = find_peaks(
            curvature,
            height=self.corner_threshold,
            distance=min_distance,
            prominence=self.corner_threshold * 0.5,
        )
        if len(peaks) > 0:
            corners = peaks[np.argsort(curvature[peaks])[::-1]].tolist()
        else:
            corners = []
        if len(corners) < 2:
            peaks, _ = find_peaks(
                curvature,
                height=self.corner_threshold * 0.5,
                distance=min_distance,
            )
            if len(peaks) >= 2:
                corners = peaks[np.argsort(curvature[peaks])[::-1]].tolist()
        return sorted(corners)

    # -- segmentation --

    def segment_contour(
        self, contour: np.ndarray, corners: List[int]
    ) -> List[Tuple[int, int, np.ndarray]]:
        n = len(contour)
        if len(corners) < 2:
            return [(0, n - 1, contour.copy())]
        segments = []
        for i in range(len(corners)):
            start_idx = corners[i]
            end_idx = corners[(i + 1) % len(corners)]
            if end_idx > start_idx:
                points = contour[start_idx : end_idx + 1].copy()
            else:
                points = np.vstack([contour[start_idx:], contour[: end_idx + 1]])
            if len(points) >= self.min_segment_length:
                segments.append((start_idx, end_idx, points))
        return segments

    # -- geometric properties --

    @staticmethod
    def compute_segment_properties(
        points: np.ndarray,
    ) -> Tuple[float, float, float, np.ndarray]:
        if len(points) < 2:
            return 0.0, 1.0, 0.0, np.array([0.0, 0.0])
        diffs = np.diff(points, axis=0)
        length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        straight_dist = np.linalg.norm(points[-1] - points[0])
        straightness_ratio = length / straight_dist if straight_dist > 0 else float('inf')
        centered = points - points.mean(axis=0)
        if len(centered) > 1:
            cov = np.cov(centered.T)
            if cov.ndim == 2:
                eigvals, eigvecs = np.linalg.eig(cov)
                principal = eigvecs[:, np.argmax(eigvals)]
                dominant_direction = float(np.arctan2(principal[1], principal[0]))
            else:
                dominant_direction = 0.0
        else:
            dominant_direction = 0.0
        centroid = points.mean(axis=0)
        return length, straightness_ratio, dominant_direction, centroid

    # -- classification helpers --

    @staticmethod
    def classify_segment_position(
        centroid: np.ndarray,
        dominant_direction: float,
        mask_shape: Tuple[int, int],
    ) -> str:
        h, w = mask_shape
        cx, cy = centroid
        nx, ny = cx / w, cy / h
        abs_dir = abs(dominant_direction)
        is_horizontal = abs_dir < np.pi / 4 or abs_dir > 3 * np.pi / 4
        is_vertical = np.pi / 4 <= abs_dir <= 3 * np.pi / 4
        if is_horizontal:
            if ny < 0.35:
                return 'top'
            if ny > 0.65:
                return 'bottom'
        elif is_vertical:
            if nx < 0.35:
                return 'left'
            if nx > 0.65:
                return 'right'
        if ny < 0.25:
            return 'top'
        if ny > 0.75:
            return 'bottom'
        if nx < 0.25:
            return 'left'
        if nx > 0.75:
            return 'right'
        return 'corner'

    def classify_edge_type(self, straightness_ratio: float) -> str:
        return 'tear' if straightness_ratio > self.straightness_threshold else 'border'

    # -- main entry --

    def segment_and_classify(
        self, contour: np.ndarray, mask_shape: Tuple[int, int]
    ) -> List[EdgeSegment]:
        contour = contour.reshape(-1, 2).astype(np.float32)
        curvature = self.compute_curvature(contour)
        corners = self.detect_corners(curvature)
        raw_segments = self.segment_contour(contour, corners)
        edge_segments: List[EdgeSegment] = []
        for seg_id, (start_idx, end_idx, points) in enumerate(raw_segments):
            length, straightness, direction, centroid = self.compute_segment_properties(points)
            position_label = self.classify_segment_position(centroid, direction, mask_shape)
            edge_type = self.classify_edge_type(straightness)
            edge_segments.append(
                EdgeSegment(
                    segment_id=seg_id,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    points=points,
                    edge_type=edge_type,
                    position_label=position_label,
                    length=length,
                    straightness_ratio=straightness,
                    dominant_direction=direction,
                    centroid=centroid,
                )
            )
        return edge_segments


# ---------------------------------------------------------------------------
# Feature helpers  (ported from v2 EnhancedEdgeExtractor)
# ---------------------------------------------------------------------------

def _resample_points(points: np.ndarray, num_samples: int) -> Optional[np.ndarray]:
    if points is None or len(points) < 2:
        return None
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    total_length = cumulative[-1]
    if total_length == 0:
        return None
    target = np.linspace(0, total_length, num_samples)
    resampled = np.zeros((num_samples, 2), dtype=np.float32)
    resampled[:, 0] = np.interp(target, cumulative, points[:, 0])
    resampled[:, 1] = np.interp(target, cumulative, points[:, 1])
    return resampled


def _compute_complexity(points: np.ndarray) -> float:
    curvatures = []
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 0 and n2 > 0:
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            curvatures.append(np.arccos(cos_a))
    return float(np.std(curvatures)) if curvatures else 0.0


def _compute_curvature_histogram(points: np.ndarray, bins: int = CURVATURE_HISTOGRAM_BINS) -> np.ndarray:
    curvatures = []
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 0 and n2 > 0:
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            curvatures.append(np.arccos(cos_a))
    if not curvatures:
        return np.zeros(bins)
    hist, _ = np.histogram(curvatures, bins=bins, range=(0, np.pi))
    hist = hist.astype(float)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def _compute_fourier_descriptors(points: np.ndarray, n_coeffs: int = NUM_FOURIER_COEFFICIENTS) -> np.ndarray:
    complex_pts = points[:, 0] + 1j * points[:, 1]
    fft = np.fft.fft(complex_pts)
    desc = np.abs(fft[1 : n_coeffs + 1])
    if desc[0] > 0:
        desc = desc / desc[0]
    return desc


# ---------------------------------------------------------------------------
# ScaleAwareEdgeExtractor
# ---------------------------------------------------------------------------

class ScaleAwareEdgeExtractor:
    """Extract edges from DB segmentation_coords — no image loading needed."""

    def __init__(self):
        self.segmenter = ContourSegmenter()

    def extract_from_segmentation_coords(
        self,
        segmentation_coords_json: str,
        image_shape: Tuple[int, int],   # (height, width)
        fragment_id: str,
        pixels_per_unit: float,
        scale_unit: str,
        line_count: Optional[int] = None,
        script_type: Optional[str] = None,
    ) -> List[ScaleAwareDescriptor]:
        """
        Extract ScaleAwareDescriptors from the segmentation_coords stored in DB.

        The segmentation_coords column stores JSON like:
          {"contours": [[[x,y], [x,y], ...]], "confidence": 0.95, ...}
        We use contours[0] (the primary contour).
        """
        # Parse contour points from JSON
        data = json.loads(segmentation_coords_json)
        contours = data.get('contours', [])
        if not contours or not contours[0]:
            return []
        raw_points = np.array(contours[0], dtype=np.float32).reshape(-1, 2)
        if len(raw_points) < 20:
            return []

        height, width = image_shape

        # Segment the contour
        segments = self.segmenter.segment_and_classify(raw_points, (height, width))

        # Compute px→cm conversion
        px_per_cm = self._pixels_per_cm(pixels_per_unit, scale_unit)

        descriptors: List[ScaleAwareDescriptor] = []
        for seg in segments:
            if seg.edge_type != 'tear':
                continue
            if seg.position_label == 'corner':
                continue

            desc = self._build_descriptor(
                seg, fragment_id, px_per_cm, scale_unit,
                pixels_per_unit, line_count, script_type, width, height,
            )
            if desc is not None:
                descriptors.append(desc)

        return descriptors

    # ------------------------------------------------------------------

    @staticmethod
    def _pixels_per_cm(pixels_per_unit: float, scale_unit: str) -> float:
        if scale_unit == 'mm':
            return pixels_per_unit * 10.0   # 10 mm = 1 cm
        return pixels_per_unit              # already px/cm

    def _build_descriptor(
        self,
        seg: EdgeSegment,
        fragment_id: str,
        px_per_cm: float,
        scale_unit: str,
        pixels_per_unit: float,
        line_count: Optional[int],
        script_type: Optional[str],
        width: int,
        height: int,
    ) -> Optional[ScaleAwareDescriptor]:
        points_px = seg.points.astype(np.float32)
        length_px = seg.length

        # Convert to cm
        length_cm = length_px / px_per_cm
        if length_cm < MIN_EDGE_LENGTH_CM:
            return None

        points_cm = points_px / px_per_cm
        centroid_cm = seg.centroid / px_per_cm

        # Resample + normalise (scale-invariant features)
        resampled = _resample_points(points_px, NUM_RESAMPLE_POINTS)
        if resampled is None:
            return None
        centered = resampled - resampled.mean(axis=0, keepdims=True)
        scale = np.max(np.linalg.norm(centered, axis=1))
        if scale == 0:
            scale = 1.0
        normalized = centered / scale

        edge_name = f"{seg.position_label}_seg{seg.segment_id}"

        return ScaleAwareDescriptor(
            fragment_id=fragment_id,
            edge_name=edge_name,
            edge_type=seg.edge_type,
            position_label=seg.position_label,
            normalized_points=normalized,
            fourier_descriptors=_compute_fourier_descriptors(normalized),
            curvature_histogram=_compute_curvature_histogram(resampled),
            complexity_score=_compute_complexity(resampled),
            orientation=seg.dominant_direction,
            length_px=length_px,
            points_px=points_px,
            pixels_per_unit=pixels_per_unit,
            scale_unit=scale_unit,
            length_cm=length_cm,
            points_cm=points_cm,
            centroid_cm=centroid_cm,
            line_count=line_count,
            script_type=script_type,
            image_width_px=width,
            image_height_px=height,
        )
