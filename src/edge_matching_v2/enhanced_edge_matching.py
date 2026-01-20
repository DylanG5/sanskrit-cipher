"""
Enhanced Edge Matching System for Sanskrit Cipher
==================================================

This module provides a scalable, hierarchical edge matching algorithm
for ancient manuscript fragment reconstruction.

Features:
- Hierarchical matching (coarse -> fine)
- Multiple geometric features
- Fast candidate retrieval with indexing
- Comprehensive evaluation metrics
- Curvature-based contour segmentation (v2)
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from scipy.stats import entropy
from scipy.signal import find_peaks


@dataclass
class EdgeSegment:
    """Represents a single contiguous edge segment from contour segmentation."""

    # Segment identification
    segment_id: int
    start_idx: int  # Index in original contour
    end_idx: int    # Index in original contour

    # Points (contiguous, properly ordered)
    points: np.ndarray

    # Classification
    edge_type: str  # 'tear' or 'border'
    position_label: str  # 'top', 'bottom', 'left', 'right', or 'corner'

    # Geometric properties
    length: float
    straightness_ratio: float
    dominant_direction: float  # Angle in radians
    centroid: np.ndarray

    def is_matchable(self) -> bool:
        """Return True if this edge should be considered for matching."""
        return self.edge_type == 'tear'


@dataclass
class EdgeDescriptor:
    """Enhanced edge descriptor with multiple feature levels."""

    # Basic identification
    fragment_id: str
    edge_name: str
    edge_type: str  # 'tear' or 'border'

    # Level 1: Coarse features (for fast filtering)
    length: float
    complexity_score: float
    orientation: float  # Dominant direction in radians
    curvature_histogram: np.ndarray  # 8-bin histogram

    # Level 2: Geometric features (for precise matching)
    normalized_points: np.ndarray  # 80 resampled points
    fourier_descriptors: np.ndarray  # 16 coefficients

    # Computed buckets for indexing
    length_bucket: int
    complexity_bucket: int

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        # Convert numpy arrays to lists
        d['curvature_histogram'] = self.curvature_histogram.tolist()
        d['normalized_points'] = self.normalized_points.tolist()
        d['fourier_descriptors'] = self.fourier_descriptors.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'EdgeDescriptor':
        """Reconstruct from dict."""
        d['curvature_histogram'] = np.array(d['curvature_histogram'])
        d['normalized_points'] = np.array(d['normalized_points'])
        d['fourier_descriptors'] = np.array(d['fourier_descriptors'])
        return cls(**d)


@dataclass
class EdgeMatch:
    """Represents a potential edge match."""

    query_fragment: str
    query_edge: str
    match_fragment: str
    match_edge: str
    score: float
    match_details: dict


class ContourSegmenter:
    """
    Curvature-based contour segmentation for proper edge detection.

    This class properly segments a contour into distinct, contiguous edge segments
    by detecting corners (high curvature points) and splitting at those locations.
    Each segment is classified as 'tear' or 'border' based on its straightness.
    """

    def __init__(
        self,
        curvature_window: int = 11,
        corner_threshold: float = 0.4,
        min_segment_length: int = 50,
        straightness_threshold: float = 1.10,
        min_corner_spacing_ratio: float = 0.03,
    ):
        """
        Initialize the ContourSegmenter.

        Args:
            curvature_window: Window size for curvature calculation (odd number)
            corner_threshold: Minimum curvature (radians) to be considered a corner
            min_segment_length: Minimum number of points for a valid segment
            straightness_threshold: Ratio above which edge is considered 'tear'
                                   (path_length / straight_distance)
            min_corner_spacing_ratio: Minimum spacing between corners as fraction
                                      of total contour length
        """
        self.curvature_window = curvature_window
        self.corner_threshold = corner_threshold
        self.min_segment_length = min_segment_length
        self.straightness_threshold = straightness_threshold
        self.min_corner_spacing_ratio = min_corner_spacing_ratio

    def compute_curvature(self, contour: np.ndarray) -> np.ndarray:
        """
        Compute curvature at each point along the contour.

        Uses angle between vectors from neighboring points.
        High curvature = sharp turn = potential corner.

        Args:
            contour: Nx2 array of contour points

        Returns:
            Array of curvature values (in radians, 0 to pi)
        """
        n = len(contour)
        if n < self.curvature_window:
            return np.zeros(n)

        half_window = self.curvature_window // 2
        curvature = np.zeros(n)

        for i in range(n):
            # Get points before and after (with wrapping for closed contour)
            idx_before = (i - half_window) % n
            idx_after = (i + half_window) % n

            # Vector from before to current
            v1 = contour[i] - contour[idx_before]
            # Vector from current to after
            v2 = contour[idx_after] - contour[i]

            # Compute angle between vectors
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                # High angle = low curvature (straight), low angle = high curvature (corner)
                # We want curvature, so use pi - angle
                angle = np.arccos(cos_angle)
                curvature[i] = np.pi - angle  # Transform so corners have HIGH values

        return curvature

    def detect_corners(self, curvature: np.ndarray) -> List[int]:
        """
        Detect corner indices from curvature array.

        Uses peak detection with filtering for minimum spacing.

        Args:
            curvature: Array of curvature values

        Returns:
            List of indices where corners are detected
        """
        n = len(curvature)
        min_distance = max(int(n * self.min_corner_spacing_ratio), 10)

        # Find peaks in curvature
        peaks, properties = find_peaks(
            curvature,
            height=self.corner_threshold,
            distance=min_distance,
            prominence=self.corner_threshold * 0.5
        )

        # Sort by curvature value (strongest corners first)
        if len(peaks) > 0:
            sorted_idx = np.argsort(curvature[peaks])[::-1]
            corners = peaks[sorted_idx].tolist()
        else:
            corners = []

        # If no corners detected, try with lower threshold
        if len(corners) < 2:
            peaks, _ = find_peaks(
                curvature,
                height=self.corner_threshold * 0.5,
                distance=min_distance
            )
            if len(peaks) >= 2:
                sorted_idx = np.argsort(curvature[peaks])[::-1]
                corners = peaks[sorted_idx].tolist()

        return sorted(corners)  # Return in contour order

    def segment_contour(
        self,
        contour: np.ndarray,
        corners: List[int]
    ) -> List[Tuple[int, int, np.ndarray]]:
        """
        Split contour into segments at corner points.

        Args:
            contour: Nx2 array of contour points
            corners: List of corner indices

        Returns:
            List of (start_idx, end_idx, points) tuples
        """
        n = len(contour)
        segments = []

        if len(corners) < 2:
            # No clear corners - treat entire contour as one segment
            return [(0, n - 1, contour.copy())]

        # Create segments between consecutive corners
        for i in range(len(corners)):
            start_idx = corners[i]
            end_idx = corners[(i + 1) % len(corners)]

            # Handle wrap-around
            if end_idx > start_idx:
                points = contour[start_idx:end_idx + 1].copy()
            else:
                # Wraps around the contour
                points = np.vstack([
                    contour[start_idx:],
                    contour[:end_idx + 1]
                ])

            if len(points) >= self.min_segment_length:
                segments.append((start_idx, end_idx, points))

        return segments

    def compute_segment_properties(
        self,
        points: np.ndarray
    ) -> Tuple[float, float, float, np.ndarray]:
        """
        Compute geometric properties of a segment.

        Args:
            points: Nx2 array of segment points

        Returns:
            (length, straightness_ratio, dominant_direction, centroid)
        """
        if len(points) < 2:
            return 0.0, 1.0, 0.0, np.array([0.0, 0.0])

        # Length: sum of distances between consecutive points
        diffs = np.diff(points, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        length = float(np.sum(distances))

        # Straight distance: start to end
        straight_dist = np.linalg.norm(points[-1] - points[0])

        # Straightness ratio
        if straight_dist > 0:
            straightness_ratio = length / straight_dist
        else:
            straightness_ratio = float('inf')  # Loop back to start

        # Dominant direction using PCA
        centered = points - points.mean(axis=0)
        if len(centered) > 1:
            cov_matrix = np.cov(centered.T)
            if cov_matrix.ndim == 2:
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                principal_idx = np.argmax(eigenvalues)
                principal = eigenvectors[:, principal_idx]
                dominant_direction = float(np.arctan2(principal[1], principal[0]))
            else:
                dominant_direction = 0.0
        else:
            dominant_direction = 0.0

        # Centroid
        centroid = points.mean(axis=0)

        return length, straightness_ratio, dominant_direction, centroid

    def classify_segment_position(
        self,
        centroid: np.ndarray,
        dominant_direction: float,
        mask_shape: Tuple[int, int]
    ) -> str:
        """
        Classify segment position based on centroid location and direction.

        Args:
            centroid: [x, y] center of segment
            dominant_direction: Dominant direction in radians
            mask_shape: (height, width) of the mask

        Returns:
            Position label: 'top', 'bottom', 'left', 'right', or 'corner'
        """
        h, w = mask_shape
        cx, cy = centroid

        # Normalize coordinates to [0, 1]
        nx = cx / w
        ny = cy / h

        # Direction classification
        # Horizontal edges (top/bottom): direction near 0 or pi
        # Vertical edges (left/right): direction near pi/2 or -pi/2
        abs_dir = abs(dominant_direction)
        is_horizontal = abs_dir < np.pi / 4 or abs_dir > 3 * np.pi / 4
        is_vertical = np.pi / 4 <= abs_dir <= 3 * np.pi / 4

        # Position classification based on centroid
        if is_horizontal:
            if ny < 0.35:
                return 'top'
            elif ny > 0.65:
                return 'bottom'
        elif is_vertical:
            if nx < 0.35:
                return 'left'
            elif nx > 0.65:
                return 'right'

        # Fallback: use position only
        if ny < 0.25:
            return 'top'
        elif ny > 0.75:
            return 'bottom'
        elif nx < 0.25:
            return 'left'
        elif nx > 0.75:
            return 'right'

        return 'corner'  # Segment is in a corner region

    def classify_edge_type(self, straightness_ratio: float) -> str:
        """
        Classify edge as 'tear' or 'border' based on straightness.

        Args:
            straightness_ratio: path_length / straight_distance

        Returns:
            'tear' if irregular, 'border' if straight
        """
        if straightness_ratio > self.straightness_threshold:
            return 'tear'
        return 'border'

    def segment_and_classify(
        self,
        contour: np.ndarray,
        mask_shape: Tuple[int, int]
    ) -> List[EdgeSegment]:
        """
        Main method: segment contour and classify each segment.

        Args:
            contour: Nx1x2 or Nx2 array of contour points
            mask_shape: (height, width) of the fragment mask

        Returns:
            List of EdgeSegment objects
        """
        # Ensure contour is Nx2
        contour = contour.reshape(-1, 2).astype(np.float32)

        # Step 1: Compute curvature
        curvature = self.compute_curvature(contour)

        # Step 2: Detect corners
        corners = self.detect_corners(curvature)

        # Step 3: Segment at corners
        raw_segments = self.segment_contour(contour, corners)

        # Step 4: Create EdgeSegment objects with classification
        edge_segments = []
        for seg_id, (start_idx, end_idx, points) in enumerate(raw_segments):
            length, straightness, direction, centroid = self.compute_segment_properties(points)

            position_label = self.classify_segment_position(
                centroid, direction, mask_shape
            )
            edge_type = self.classify_edge_type(straightness)

            segment = EdgeSegment(
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
            edge_segments.append(segment)

        return edge_segments

    def get_debug_info(
        self,
        contour: np.ndarray,
    ) -> Dict:
        """
        Get debug information for visualization.

        Returns curvature array and corner indices for plotting.
        """
        contour = contour.reshape(-1, 2).astype(np.float32)
        curvature = self.compute_curvature(contour)
        corners = self.detect_corners(curvature)
        return {
            'curvature': curvature,
            'corners': corners,
            'contour': contour,
        }


class EnhancedEdgeExtractor:
    """Extract and analyze fragment edges with rich feature descriptors."""

    def __init__(
        self,
        smoothing_iterations: int = 2,
        num_samples: int = 80,
        use_curvature_segmentation: bool = True,
    ):
        self.smoothing_iterations = smoothing_iterations
        self.num_samples = num_samples
        self.length_bucket_size = 50  # pixels per bucket
        self.complexity_bucket_size = 0.05
        self.use_curvature_segmentation = use_curvature_segmentation

        # Initialize the curvature-based segmenter
        self.segmenter = ContourSegmenter(
            curvature_window=11,
            corner_threshold=0.4,
            min_segment_length=50,
            straightness_threshold=1.10,
        )

    def load_fragment(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load fragment and extract binary mask."""
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Extract alpha channel as binary mask
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return img, mask

    def extract_contour(self, mask: np.ndarray) -> np.ndarray:
        """Extract and smooth the outer contour."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            raise ValueError("No contours found in mask")

        contour = max(contours, key=cv2.contourArea)

        # Apply smoothing
        for _ in range(self.smoothing_iterations):
            contour = self._smooth_contour(contour)

        return contour

    def _smooth_contour(self, contour: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Smooth contour using moving average filter."""
        contour = contour.reshape(-1, 2)
        smoothed = np.zeros_like(contour, dtype=np.float32)

        for i in range(len(contour)):
            indices = [(i + j - kernel_size // 2) % len(contour) for j in range(kernel_size)]
            smoothed[i] = np.mean(contour[indices], axis=0)

        return smoothed.reshape(-1, 1, 2).astype(np.int32)

    def classify_edges(self, contour: np.ndarray, mask_shape: Tuple[int, int]) -> Dict:
        """
        Classify contour segments as tear edges or original borders.

        Uses curvature-based segmentation by default (v2) which properly:
        1. Detects corners via curvature analysis
        2. Segments contour into contiguous edge segments
        3. Classifies each segment as 'tear' or 'border' based on straightness

        Args:
            contour: Contour points from cv2.findContours
            mask_shape: (height, width) of the fragment mask

        Returns:
            Dict with edge data including segments, tear_edges, and border_edges
        """
        if self.use_curvature_segmentation:
            return self._classify_edges_curvature(contour, mask_shape)
        else:
            return self._classify_edges_legacy(contour, mask_shape)

    def _classify_edges_curvature(self, contour: np.ndarray, mask_shape: Tuple[int, int]) -> Dict:
        """
        Curvature-based edge classification (v2 - RECOMMENDED).

        Properly segments contour at detected corners and classifies each
        contiguous segment individually.
        """
        # Use the ContourSegmenter to get properly segmented edges
        segments = self.segmenter.segment_and_classify(contour, mask_shape)

        # Build edge_data structure compatible with existing code
        edge_data = {
            "contour": contour,
            "segments": segments,  # New: list of EdgeSegment objects
            "top_edge": [],
            "bottom_edge": [],
            "left_edge": [],
            "right_edge": [],
            "tear_edges": [],
            "border_edges": [],
        }

        # Populate position-based edge lists from segments
        for segment in segments:
            # Create edge point list compatible with old format
            edge_points = [(i, pt) for i, pt in enumerate(segment.points)]

            # Map position labels to edge names
            position_to_edge = {
                'top': 'top_edge',
                'bottom': 'bottom_edge',
                'left': 'left_edge',
                'right': 'right_edge',
            }

            edge_name = position_to_edge.get(segment.position_label)
            if edge_name:
                # Store segment reference with unique name
                seg_edge_name = f"{edge_name}_seg{segment.segment_id}"
                edge_data[seg_edge_name] = edge_points

                # Track tear vs border
                if segment.edge_type == 'tear':
                    edge_data["tear_edges"].append(seg_edge_name)
                else:
                    edge_data["border_edges"].append(seg_edge_name)

        return edge_data

    def _classify_edges_legacy(self, contour: np.ndarray, mask_shape: Tuple[int, int]) -> Dict:
        """
        Legacy position-based edge classification.

        WARNING: This method has known bugs:
        - Points can belong to multiple edges (corner overlap)
        - Edge points are non-contiguous
        - Straightness calculation is broken for non-contiguous points

        Kept for backward compatibility only. Use curvature-based segmentation instead.
        """
        height, width = mask_shape
        contour_points = contour.reshape(-1, 2)

        edge_data = {
            "contour": contour,
            "top_edge": [],
            "bottom_edge": [],
            "left_edge": [],
            "right_edge": [],
            "tear_edges": [],
            "border_edges": [],
        }

        # Segment contour into regions (BUGGY: points overlap at corners)
        for i, point in enumerate(contour_points):
            x, y = point

            if y < height / 4:
                edge_data["top_edge"].append((i, point))
            elif y > 3 * height / 4:
                edge_data["bottom_edge"].append((i, point))

            if x < width / 4:
                edge_data["left_edge"].append((i, point))
            elif x > 3 * width / 4:
                edge_data["right_edge"].append((i, point))

        # Classify each edge segment
        for edge_name in ["top_edge", "bottom_edge", "left_edge", "right_edge"]:
            edge_points = edge_data[edge_name]
            if len(edge_points) > 10:
                is_tear = self._is_tear_edge(edge_points)
                if is_tear:
                    edge_data["tear_edges"].append(edge_name)
                else:
                    edge_data["border_edges"].append(edge_name)

        return edge_data

    def _is_tear_edge(
        self,
        edge_points: List[Tuple[int, np.ndarray]],
        straightness_threshold: float = 0.15,
    ) -> bool:
        """Determine if an edge is a tear (irregular) or border (straight)."""
        if len(edge_points) < 3:
            return False

        points = np.array([p[1] for p in edge_points])
        path_length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        straight_distance = np.linalg.norm(points[-1] - points[0])

        if straight_distance == 0:
            return True

        straightness_ratio = path_length / straight_distance
        return straightness_ratio > (1.0 + straightness_threshold)

    def compute_edge_descriptor(
        self,
        edge_points: List[Tuple[int, np.ndarray]],
        fragment_id: str,
        edge_name: str,
    ) -> Optional[EdgeDescriptor]:
        """Generate comprehensive edge descriptor with multi-level features."""
        if not edge_points or len(edge_points) < 5:
            return None

        # Extract and resample points
        sorted_points = sorted(edge_points, key=lambda x: x[0])
        points = np.array([pt for _, pt in sorted_points], dtype=np.float32)

        # Original length
        length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

        # Resample to fixed number of points
        resampled = self._resample_points(points, self.num_samples)
        if resampled is None:
            return None

        # Normalize for scale invariance
        centered = resampled - resampled.mean(axis=0, keepdims=True)
        scale = np.max(np.linalg.norm(centered, axis=1))
        if scale == 0:
            scale = 1.0
        normalized = centered / scale

        # Compute coarse features
        complexity_score = self._compute_complexity(resampled)
        orientation = self._compute_orientation(resampled)
        curvature_hist = self._compute_curvature_histogram(resampled)

        # Compute Fourier descriptors for rotation invariance
        fourier_desc = self._compute_fourier_descriptors(normalized, n_coeffs=16)

        # Compute buckets for indexing
        length_bucket = int(length / self.length_bucket_size)
        complexity_bucket = int(complexity_score / self.complexity_bucket_size)

        return EdgeDescriptor(
            fragment_id=fragment_id,
            edge_name=edge_name,
            edge_type='tear',
            length=length,
            complexity_score=complexity_score,
            orientation=orientation,
            curvature_histogram=curvature_hist,
            normalized_points=normalized,
            fourier_descriptors=fourier_desc,
            length_bucket=length_bucket,
            complexity_bucket=complexity_bucket,
        )

    def compute_descriptor_from_segment(
        self,
        segment: EdgeSegment,
        fragment_id: str,
    ) -> Optional[EdgeDescriptor]:
        """
        Generate edge descriptor from an EdgeSegment (v2 method).

        This uses the properly segmented, contiguous edge points
        from curvature-based segmentation.

        Args:
            segment: EdgeSegment from ContourSegmenter
            fragment_id: Fragment identifier

        Returns:
            EdgeDescriptor or None if segment is too small
        """
        points = segment.points

        if len(points) < 5:
            return None

        # Use segment's pre-computed length
        length = segment.length

        # Resample to fixed number of points
        resampled = self._resample_points(points, self.num_samples)
        if resampled is None:
            return None

        # Normalize for scale invariance
        centered = resampled - resampled.mean(axis=0, keepdims=True)
        scale = np.max(np.linalg.norm(centered, axis=1))
        if scale == 0:
            scale = 1.0
        normalized = centered / scale

        # Compute coarse features
        complexity_score = self._compute_complexity(resampled)
        orientation = segment.dominant_direction  # Use pre-computed direction
        curvature_hist = self._compute_curvature_histogram(resampled)

        # Compute Fourier descriptors for rotation invariance
        fourier_desc = self._compute_fourier_descriptors(normalized, n_coeffs=16)

        # Compute buckets for indexing
        length_bucket = int(length / self.length_bucket_size)
        complexity_bucket = int(complexity_score / self.complexity_bucket_size)

        # Create unique edge name: position_segID (e.g., "top_seg0", "left_seg2")
        edge_name = f"{segment.position_label}_seg{segment.segment_id}"

        return EdgeDescriptor(
            fragment_id=fragment_id,
            edge_name=edge_name,
            edge_type=segment.edge_type,
            length=length,
            complexity_score=complexity_score,
            orientation=orientation,
            curvature_histogram=curvature_hist,
            normalized_points=normalized,
            fourier_descriptors=fourier_desc,
            length_bucket=length_bucket,
            complexity_bucket=complexity_bucket,
        )

    def extract_tear_descriptors(
        self,
        image_path: str,
        fragment_id: str,
    ) -> Tuple[List[EdgeDescriptor], List[EdgeSegment]]:
        """
        Extract edge descriptors for all tear edges in a fragment.

        This is the recommended high-level method for processing fragments.
        Uses curvature-based segmentation and returns only matchable tear edges.

        Args:
            image_path: Path to fragment image
            fragment_id: Unique identifier for the fragment

        Returns:
            (list of EdgeDescriptors for tear edges, list of all EdgeSegments)
        """
        # Load fragment
        img, mask = self.load_fragment(image_path)

        # Extract contour
        contour = self.extract_contour(mask)

        # Segment using curvature-based method
        segments = self.segmenter.segment_and_classify(contour, mask.shape)

        # Generate descriptors for tear edges only
        descriptors = []
        for segment in segments:
            if segment.edge_type == 'tear':  # Only matchable edges
                descriptor = self.compute_descriptor_from_segment(segment, fragment_id)
                if descriptor is not None:
                    descriptors.append(descriptor)

        return descriptors, segments

    def _resample_points(self, points: np.ndarray, num_samples: int) -> Optional[np.ndarray]:
        """Resample a polyline to a fixed number of points."""
        if points is None or len(points) < 2:
            return None

        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(distances)))
        total_length = cumulative[-1]

        if total_length == 0:
            return None

        target_distances = np.linspace(0, total_length, num_samples)
        resampled = np.zeros((num_samples, 2), dtype=np.float32)
        resampled[:, 0] = np.interp(target_distances, cumulative, points[:, 0])
        resampled[:, 1] = np.interp(target_distances, cumulative, points[:, 1])

        return resampled

    def _compute_complexity(self, points: np.ndarray) -> float:
        """
        Compute edge complexity/irregularity score.
        Higher values indicate more irregular (torn) edges.
        """
        # Compute curvature at each point
        curvatures = []
        for i in range(1, len(points) - 1):
            p1, p2, p3 = points[i-1], points[i], points[i+1]

            # Vectors
            v1 = p2 - p1
            v2 = p3 - p2

            # Angle between vectors
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                curvatures.append(angle)

        if not curvatures:
            return 0.0

        # Use standard deviation of curvature as complexity measure
        return float(np.std(curvatures))

    def _compute_orientation(self, points: np.ndarray) -> float:
        """Compute dominant orientation of edge using PCA."""
        centered = points - points.mean(axis=0)
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Principal direction
        principal_direction = eigenvectors[:, np.argmax(eigenvalues)]
        orientation = np.arctan2(principal_direction[1], principal_direction[0])

        return float(orientation)

    def _compute_curvature_histogram(self, points: np.ndarray, bins: int = 8) -> np.ndarray:
        """
        Compute histogram of curvature values.
        Provides a coarse shape signature for fast filtering.
        """
        curvatures = []
        for i in range(1, len(points) - 1):
            p1, p2, p3 = points[i-1], points[i], points[i+1]
            v1 = p2 - p1
            v2 = p3 - p2

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                curvatures.append(angle)

        if not curvatures:
            return np.zeros(bins)

        hist, _ = np.histogram(curvatures, bins=bins, range=(0, np.pi))
        # Normalize to make it a probability distribution
        hist = hist.astype(float)
        if hist.sum() > 0:
            hist /= hist.sum()

        return hist

    def _compute_fourier_descriptors(self, points: np.ndarray, n_coeffs: int = 16) -> np.ndarray:
        """
        Compute Fourier descriptors for rotation and scale invariant matching.
        """
        # Convert to complex numbers
        complex_points = points[:, 0] + 1j * points[:, 1]

        # FFT
        fft_coeffs = np.fft.fft(complex_points)

        # Take magnitude of coefficients (rotation invariant)
        # Skip DC component (index 0) as it represents translation
        descriptors = np.abs(fft_coeffs[1:n_coeffs+1])

        # Normalize by first coefficient for scale invariance
        if descriptors[0] > 0:
            descriptors = descriptors / descriptors[0]

        return descriptors


class EdgeIndex:
    """Multi-level index for fast edge matching."""

    def __init__(self):
        self.length_index: Dict[int, List[EdgeDescriptor]] = defaultdict(list)
        self.complexity_index: Dict[int, List[EdgeDescriptor]] = defaultdict(list)
        self.all_edges: List[EdgeDescriptor] = []

    def build_index(self, edges: List[EdgeDescriptor]):
        """Build multi-level index from edge descriptors."""
        self.all_edges = edges

        for edge in edges:
            # Index by length bucket
            self.length_index[edge.length_bucket].append(edge)

            # Index by complexity bucket
            self.complexity_index[edge.complexity_bucket].append(edge)

    def find_candidates(
        self,
        query_edge: EdgeDescriptor,
        k: int = 50,
        length_tolerance: int = 2,
        complexity_tolerance: int = 2,
    ) -> List[EdgeDescriptor]:
        """
        Fast candidate retrieval using hierarchical filtering.

        Args:
            query_edge: The edge to find matches for
            k: Maximum number of candidates to return
            length_tolerance: Number of adjacent buckets to search
            complexity_tolerance: Number of adjacent buckets to search

        Returns:
            List of candidate edges (up to k edges)
        """
        candidates = []

        # Search adjacent length buckets
        for bucket_offset in range(-length_tolerance, length_tolerance + 1):
            bucket = query_edge.length_bucket + bucket_offset
            candidates.extend(self.length_index[bucket])

        # Filter by complexity
        complexity_filtered = []
        min_complexity_bucket = query_edge.complexity_bucket - complexity_tolerance
        max_complexity_bucket = query_edge.complexity_bucket + complexity_tolerance

        for edge in candidates:
            if min_complexity_bucket <= edge.complexity_bucket <= max_complexity_bucket:
                # Don't match edge with itself
                if edge.fragment_id != query_edge.fragment_id:
                    complexity_filtered.append(edge)

        # If we have more than k candidates, filter by curvature histogram similarity
        if len(complexity_filtered) > k:
            # Compute histogram similarities
            scored_candidates = []
            for edge in complexity_filtered:
                # Jensen-Shannon divergence (lower is more similar)
                hist_dist = distance.jensenshannon(
                    query_edge.curvature_histogram,
                    edge.curvature_histogram
                )
                scored_candidates.append((hist_dist, edge))

            # Sort by similarity and take top k
            scored_candidates.sort(key=lambda x: x[0])
            return [edge for _, edge in scored_candidates[:k]]

        return complexity_filtered


class EdgeMatcher:
    """Hierarchical edge matching with multiple scoring methods."""

    def __init__(self, num_samples: int = 80):
        self.num_samples = num_samples

    @staticmethod
    def _get_base_piece_id(fragment_id: str) -> str:
        """
        Extract base piece ID from fragment identifier.

        For example:
        - "OR15013_382B_L [BLL250]_fragment_1_conf0.98" -> "OR15013_382"
        - "OR15013_382A_L [BLL250]_fragment_1_conf0.97" -> "OR15013_382"

        The pattern is: {edition}_{piece_number}{A/B}_{rest}
        We want to keep {edition}_{piece_number} and strip the A/B suffix.
        """
        # Split by underscore
        parts = fragment_id.split('_')
        if len(parts) < 2:
            return fragment_id

        # Extract edition (e.g., "OR15013") and piece number (e.g., "382B")
        edition = parts[0]
        piece_with_suffix = parts[1]

        # Remove A/B suffix from piece number (e.g., "382B" -> "382")
        # The last character is typically A or B
        if piece_with_suffix and piece_with_suffix[-1] in 'AB':
            piece_base = piece_with_suffix[:-1]
        else:
            piece_base = piece_with_suffix

        return f"{edition}_{piece_base}"

    def match_edges(
        self,
        query_edge: EdgeDescriptor,
        candidate_edges: List[EdgeDescriptor],
        top_k: int = 10,
    ) -> List[EdgeMatch]:
        """
        Match a query edge against candidate edges.

        Args:
            query_edge: The edge to find matches for
            candidate_edges: List of candidate edges to compare against
            top_k: Number of top matches to return

        Returns:
            List of EdgeMatch objects, sorted by score (lower is better)
        """
        matches = []
        query_base_id = self._get_base_piece_id(query_edge.fragment_id)

        for candidate in candidate_edges:
            # Filter out self-matches (same base piece ID)
            # E.g., OR15013_382A should not match with OR15013_382B
            candidate_base_id = self._get_base_piece_id(candidate.fragment_id)
            if query_base_id == candidate_base_id:
                continue

            # Compute multi-level match score
            score, details = self._compute_match_score(query_edge, candidate)

            matches.append(EdgeMatch(
                query_fragment=query_edge.fragment_id,
                query_edge=query_edge.edge_name,
                match_fragment=candidate.fragment_id,
                match_edge=candidate.edge_name,
                score=score,
                match_details=details,
            ))

        # Sort by score (lower is better)
        matches.sort(key=lambda x: x.score)

        return matches[:top_k]

    def _compute_match_score(
        self,
        edge_a: EdgeDescriptor,
        edge_b: EdgeDescriptor,
    ) -> Tuple[float, dict]:
        """
        Compute comprehensive match score between two edges.

        Returns:
            (score, details) where score is lower for better matches
        """
        details = {}

        # 1. Geometric shape matching (complementary edges)
        shape_score = self._compute_shape_score(edge_a, edge_b)
        details['shape_score'] = shape_score

        # 2. Fourier descriptor matching
        fourier_score = self._compute_fourier_score(edge_a, edge_b)
        details['fourier_score'] = fourier_score

        # 3. Length compatibility
        length_penalty = abs(edge_a.length - edge_b.length) / max(edge_a.length, edge_b.length, 1.0)
        details['length_penalty'] = length_penalty

        # 4. Curvature histogram similarity
        hist_dist = distance.jensenshannon(edge_a.curvature_histogram, edge_b.curvature_histogram)
        details['histogram_distance'] = hist_dist

        # Combined score (weighted sum)
        combined_score = (
            0.40 * shape_score +
            0.25 * fourier_score +
            0.20 * length_penalty +
            0.15 * hist_dist
        )

        details['combined_score'] = combined_score

        return combined_score, details

    def _compute_shape_score(self, edge_a: EdgeDescriptor, edge_b: EdgeDescriptor) -> float:
        """
        Compute shape matching score using normalized point-to-point distance.
        For puzzle pieces, we want complementary edges (negative correlation).
        """
        base = edge_a.normalized_points

        # Generate orientation variants
        variants = self._generate_variants(edge_b.normalized_points)

        best_score = float('inf')

        for variant in variants:
            # For complementary jigsaw fits, one edge should be inverse of the other
            complement_variant = -variant

            # Compute RMSE
            rmse = np.sqrt(np.mean((base - complement_variant) ** 2))

            if rmse < best_score:
                best_score = rmse

        return best_score

    def _compute_fourier_score(self, edge_a: EdgeDescriptor, edge_b: EdgeDescriptor) -> float:
        """Compare Fourier descriptors (rotation/scale invariant)."""
        # Euclidean distance between Fourier coefficient magnitudes
        return float(np.linalg.norm(edge_a.fourier_descriptors - edge_b.fourier_descriptors))

    def _generate_variants(self, points: np.ndarray) -> List[np.ndarray]:
        """Generate orientation variants for edge comparison."""
        variants = []

        # Original
        variants.append(points.copy())

        # Flip X
        flipped_x = points.copy()
        flipped_x[:, 0] *= -1
        variants.append(flipped_x)

        # Reversed
        reversed_points = points[::-1].copy()
        variants.append(reversed_points)

        # Reversed + Flip X
        reversed_flipped = reversed_points.copy()
        reversed_flipped[:, 0] *= -1
        variants.append(reversed_flipped)

        return variants


def process_fragment(
    fragment_path: Path,
    extractor: EnhancedEdgeExtractor,
) -> Tuple[str, List[EdgeDescriptor]]:
    """
    Process a single fragment and extract all tear edge descriptors.

    Uses curvature-based segmentation (v2) by default for proper edge detection.

    Returns:
        (fragment_id, list of edge descriptors)
    """
    fragment_id = fragment_path.stem

    try:
        if extractor.use_curvature_segmentation:
            # NEW: Use curvature-based segmentation (recommended)
            descriptors, segments = extractor.extract_tear_descriptors(
                str(fragment_path),
                fragment_id
            )
            return fragment_id, descriptors
        else:
            # LEGACY: Use old position-based segmentation (buggy)
            img, mask = extractor.load_fragment(str(fragment_path))
            contour = extractor.extract_contour(mask)
            edge_data = extractor.classify_edges(contour, mask.shape)

            descriptors = []
            for edge_name in edge_data['tear_edges']:
                edge_points = edge_data.get(edge_name, [])
                if edge_points:
                    descriptor = extractor.compute_edge_descriptor(
                        edge_points,
                        fragment_id,
                        edge_name
                    )
                    if descriptor is not None:
                        descriptors.append(descriptor)

            return fragment_id, descriptors

    except Exception as e:
        print(f"Error processing {fragment_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return fragment_id, []


def save_descriptors(descriptors: List[EdgeDescriptor], output_path: Path):
    """Save edge descriptors to JSON file."""
    descriptor_dicts = [desc.to_dict() for desc in descriptors]

    with open(output_path, 'w') as f:
        json.dump(descriptor_dicts, f, indent=2)

    print(f"Saved {len(descriptors)} edge descriptors to {output_path}")


def load_descriptors(input_path: Path) -> List[EdgeDescriptor]:
    """Load edge descriptors from JSON file."""
    with open(input_path, 'r') as f:
        descriptor_dicts = json.load(f)

    descriptors = [EdgeDescriptor.from_dict(d) for d in descriptor_dicts]
    print(f"Loaded {len(descriptors)} edge descriptors from {input_path}")

    return descriptors


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Edge Matching for Sanskrit Cipher"
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Extract command
    extract_parser = subparsers.add_parser(
        'extract',
        help='Extract edge descriptors from fragment images'
    )
    extract_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Directory containing fragment images'
    )
    extract_parser.add_argument(
        '--output',
        type=str,
        default='output/edge_descriptors.json',
        help='Output JSON file for descriptors'
    )
    extract_parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of fragments to process (for testing)'
    )

    # Match command
    match_parser = subparsers.add_parser(
        'match',
        help='Find matching edges for a target fragment'
    )
    match_parser.add_argument(
        '--descriptors',
        type=str,
        required=True,
        help='Path to edge descriptors JSON file'
    )
    match_parser.add_argument(
        '--fragment',
        type=str,
        required=True,
        help='Fragment ID to find matches for'
    )
    match_parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of top matches to return per edge'
    )
    match_parser.add_argument(
        '--output',
        type=str,
        default='output/matches',
        help='Output directory for match visualizations'
    )

    args = parser.parse_args()

    if args.command == 'extract':
        extract_edges(args)
    elif args.command == 'match':
        match_edges(args)


def extract_edges(args):
    """Extract edge descriptors from fragments."""
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find fragment images
    fragment_files = []
    for pattern in ('*.png', '*.jpg', '*.jpeg'):
        fragment_files.extend(input_path.glob(pattern))
    fragment_files = sorted(fragment_files)

    if args.limit:
        fragment_files = fragment_files[:args.limit]

    if not fragment_files:
        print(f"No fragment images found in {input_path}")
        return

    print(f"Processing {len(fragment_files)} fragments...")
    print("=" * 60)

    extractor = EnhancedEdgeExtractor()
    all_descriptors = []

    start_time = time.time()

    for i, fragment_path in enumerate(fragment_files, 1):
        print(f"[{i}/{len(fragment_files)}] Processing {fragment_path.name}")

        fragment_id, descriptors = process_fragment(fragment_path, extractor)
        all_descriptors.extend(descriptors)

        print(f"  → Found {len(descriptors)} tear edges")

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"Extraction complete!")
    print(f"  Total fragments: {len(fragment_files)}")
    print(f"  Total edges: {len(all_descriptors)}")
    print(f"  Time elapsed: {elapsed:.2f}s")
    print(f"  Avg time per fragment: {elapsed/len(fragment_files):.2f}s")

    # Save descriptors
    save_descriptors(all_descriptors, output_path)


def match_edges(args):
    """Find matching edges for a target fragment."""
    descriptors_path = Path(args.descriptors)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load descriptors
    all_descriptors = load_descriptors(descriptors_path)

    # Find target fragment edges
    target_descriptors = [d for d in all_descriptors if d.fragment_id == args.fragment]

    if not target_descriptors:
        print(f"Fragment '{args.fragment}' not found in descriptors")
        return

    # Build index from candidate edges (exclude target fragment)
    candidate_descriptors = [d for d in all_descriptors if d.fragment_id != args.fragment]

    print(f"Building index from {len(candidate_descriptors)} candidate edges...")
    index = EdgeIndex()
    index.build_index(candidate_descriptors)

    # Match each edge
    matcher = EdgeMatcher()

    print(f"\nFinding matches for fragment '{args.fragment}'...")
    print("=" * 60)

    all_matches = {}

    for target_edge in target_descriptors:
        print(f"\nEdge: {target_edge.edge_name}")
        print(f"  Length: {target_edge.length:.1f}px")
        print(f"  Complexity: {target_edge.complexity_score:.3f}")

        # Find candidates using index
        candidates = index.find_candidates(target_edge, k=50)
        print(f"  → Found {len(candidates)} candidates after filtering")

        # Match against candidates
        matches = matcher.match_edges(target_edge, candidates, top_k=args.top_k)

        print(f"  Top {len(matches)} matches:")
        for i, match in enumerate(matches, 1):
            print(f"    {i}. {match.match_fragment} [{match.match_edge}]")
            print(f"       Score: {match.score:.4f} (lower is better)")
            print(f"       Details: shape={match.match_details['shape_score']:.4f}, "
                  f"fourier={match.match_details['fourier_score']:.4f}, "
                  f"length={match.match_details['length_penalty']:.4f}")

        all_matches[target_edge.edge_name] = matches

    # Save results
    results_file = output_dir / f"matches_{args.fragment}.json"
    results_data = {
        'fragment': args.fragment,
        'matches': {
            edge_name: [
                {
                    'match_fragment': m.match_fragment,
                    'match_edge': m.match_edge,
                    'score': m.score,
                    'details': m.match_details,
                }
                for m in matches
            ]
            for edge_name, matches in all_matches.items()
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()
