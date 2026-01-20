"""
Advanced Fragment Alignment and Validation
===========================================

Performs actual geometric alignment of matched fragments and creates
composite images showing the physical fit with proper transparency and rotation testing.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from enhanced_edge_matching import (
    EdgeDescriptor,
    EnhancedEdgeExtractor,
    load_descriptors,
)


class FragmentAligner:
    """Performs geometric alignment of matched fragments with rotation testing."""

    def __init__(self, fragments_dir: Path):
        self.fragments_dir = fragments_dir
        self.extractor = EnhancedEdgeExtractor()

    def load_fragment(self, fragment_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load fragment image with RGBA channels.

        Returns:
            (rgb_image, alpha_channel)
        """
        for ext in ['.png', '.jpg', '.jpeg']:
            path = self.fragments_dir / f"{fragment_id}{ext}"
            if path.exists():
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if img.shape[2] == 4:
                        # PNG with alpha channel
                        rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
                        alpha = img[:, :, 3]
                        return rgb, alpha
                    else:
                        # No alpha channel - create one from grayscale threshold
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        _, alpha = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                        return rgb, alpha

        raise FileNotFoundError(f"Fragment not found: {fragment_id}")

    def extract_edge_points(
        self,
        fragment_id: str,
        edge_name: str
    ) -> Optional[np.ndarray]:
        """
        Extract the actual edge points from a fragment.

        Handles both old format ("top_edge") and new segment format ("top_seg0").
        """
        rgb, alpha = self.load_fragment(fragment_id)

        # Create binary mask from alpha
        _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

        # Extract contour
        contour = self.extractor.extract_contour(mask)

        # Check if using new segment-based naming (e.g., "top_seg0", "left_seg3")
        if '_seg' in edge_name:
            # New format: use curvature-based segmentation
            segments = self.extractor.segmenter.segment_and_classify(contour, mask.shape)

            # Parse segment ID from edge name (e.g., "top_seg0" -> 0)
            import re
            match = re.search(r'_seg(\d+)$', edge_name)
            if match:
                seg_id = int(match.group(1))
                for segment in segments:
                    if segment.segment_id == seg_id:
                        return segment.points.astype(np.float32)

            # Fallback: try to match by position and approximate segment
            position = edge_name.split('_seg')[0]  # e.g., "top"
            matching_segments = [s for s in segments if s.position_label == position]
            if matching_segments:
                return matching_segments[0].points.astype(np.float32)

            return None
        else:
            # Old format: use legacy classification
            edge_data = self.extractor._classify_edges_legacy(contour, mask.shape)

            # Get edge points
            edge_points = edge_data.get(edge_name, [])
            if not edge_points:
                return None

            # Convert to array
            points = np.array([pt for _, pt in edge_points], dtype=np.float32)
            return points

    def rotate_image_and_points(
        self,
        image: np.ndarray,
        alpha: np.ndarray,
        angle_degrees: float,
        points: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Rotate image, alpha channel, and optionally points by given angle.

        Returns:
            (rotated_image, rotated_alpha, rotated_points, rotation_matrix)
        """
        h, w = image.shape[:2]
        center = (w / 2, h / 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

        # Calculate new image size to fit rotated image
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Adjust rotation matrix for new size
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2

        # Rotate image
        rotated_image = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # Rotate alpha channel
        rotated_alpha = cv2.warpAffine(
            alpha,
            rotation_matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # Rotate points if provided
        rotated_points = None
        if points is not None and len(points) > 0:
            # Add homogeneous coordinate
            points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
            # Apply rotation
            rotated_points = (rotation_matrix @ points_homogeneous.T).T

        return rotated_image, rotated_alpha, rotated_points, rotation_matrix

    def compute_alignment_transform(
        self,
        edge1_points: np.ndarray,
        edge2_points: np.ndarray,
        edge1_name: str,
        edge2_name: str
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Compute the transformation to align edge2 to edge1.
        Positions fragments edge-to-edge with separation to prevent body overlap.

        Returns:
            (transform_matrix, quality_score, should_flip)
        """
        # Resample both edges to same number of points
        n_points = min(len(edge1_points), len(edge2_points), 50)
        edge1_resampled = self._resample_edge(edge1_points, n_points)
        edge2_resampled = self._resample_edge(edge2_points, n_points)

        # Determine if edge2 should be flipped based on edge names
        should_flip = self._should_flip_for_complementary(edge1_name, edge2_name)

        if should_flip:
            edge2_resampled = edge2_resampled[::-1]

        # Find best transformation using ICP-like approach
        best_transform = None
        best_error = float('inf')
        best_offset = 0
        best_shifted_edge2 = None

        # Try different starting alignments
        for start_offset in range(0, n_points, max(1, n_points // 10)):
            # Circularly shift edge2 points
            shifted_edge2 = np.roll(edge2_resampled, start_offset, axis=0)

            # Compute transformation
            transform, error = self._compute_transform_least_squares(
                edge1_resampled,
                shifted_edge2
            )

            if error < best_error:
                best_error = error
                best_transform = transform
                best_offset = start_offset
                best_shifted_edge2 = shifted_edge2

        # Apply edge-to-edge separation to prevent body overlap
        # Transform edge2 points to aligned position
        edge2_aligned = (best_transform @ np.column_stack([best_shifted_edge2, np.ones(len(best_shifted_edge2))]).T).T[:, :2]

        # Compute separation vector to position fragments on opposite sides
        # This separation is in the DESTINATION coordinate space (where edge1 lives)
        separation = self._compute_separation_for_edge_meeting(
            edge1_name, edge2_name, edge1_resampled, edge2_aligned
        )

        # Apply separation by subtracting (moving fragment2 AWAY from fragment1)
        # The separation vector points in the direction fragment1's body is relative to its edge.
        # We subtract to move fragment2 in the OPPOSITE direction (away from fragment1's body)
        best_transform[0, 2] -= separation[0]
        best_transform[1, 2] -= separation[1]

        return best_transform, best_error, should_flip

    def compute_complementary_fit(
        self,
        edge1_points: np.ndarray,
        edge2_points: np.ndarray,
        transform: np.ndarray
    ) -> float:
        """
        Compute complementary fit score between edges.

        For true puzzle-piece fit: edge1 protrusions should fill edge2 indentations.
        This measures how well the edges interlock.

        Args:
            edge1_points: Points along edge1 (on fragment1)
            edge2_points: Points along edge2 (on fragment2), before transformation
            transform: Transformation matrix to apply to edge2_points

        Returns:
            fit_score: 0.0 = perfect interlock, higher = poor fit
        """
        # Transform edge2 points to align with edge1
        edge2_homogeneous = np.column_stack([edge2_points, np.ones(len(edge2_points))])
        edge2_transformed = (transform @ edge2_homogeneous.T).T[:, :2]

        # Compute signed distances from edge2 points to edge1 line segments
        distances = []
        for pt in edge2_transformed:
            # Find minimum distance to any edge1 line segment
            min_dist = float('inf')
            for i in range(len(edge1_points) - 1):
                p1 = edge1_points[i]
                p2 = edge1_points[i + 1]

                # Distance from point to line segment
                segment = p2 - p1
                segment_length = np.linalg.norm(segment)
                if segment_length < 1e-6:
                    dist = np.linalg.norm(pt - p1)
                else:
                    t = np.clip(np.dot(pt - p1, segment) / (segment_length ** 2), 0, 1)
                    projection = p1 + t * segment
                    dist = np.linalg.norm(pt - projection)

                min_dist = min(min_dist, dist)

            distances.append(min_dist)

        # Complementary fit score: average distance
        fit_score = np.mean(distances)
        return fit_score

    def compute_overlap_and_gap_metrics(
        self,
        fragment1_alpha: np.ndarray,
        fragment2_alpha: np.ndarray,
        transform: np.ndarray,
        canvas_shape: Tuple[int, int],
        fragment1_offset: Tuple[int, int]
    ) -> Tuple[int, float, float]:
        """
        Compute overlap and gap metrics between aligned fragments.

        Args:
            fragment1_alpha: Alpha channel of fragment 1
            fragment2_alpha: Alpha channel of fragment 2
            transform: Transformation matrix for fragment 2
            canvas_shape: (height, width) of canvas
            fragment1_offset: (x_offset, y_offset) of fragment 1 on canvas

        Returns:
            (overlap_pixels, overlap_percentage, edge_gap_distance)
        """
        canvas_h, canvas_w = canvas_shape
        x1_offset, y1_offset = fragment1_offset

        # Create masks on canvas
        mask1 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        h1, w1 = fragment1_alpha.shape
        mask1[y1_offset:y1_offset+h1, x1_offset:x1_offset+w1] = (fragment1_alpha > 127).astype(np.uint8)

        # Transform fragment 2 mask
        mask2_transformed = cv2.warpAffine(
            (fragment2_alpha > 127).astype(np.uint8),
            transform[:2, :],
            (canvas_w, canvas_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # Compute overlap
        overlap_mask = (mask1 > 0) & (mask2_transformed > 0)
        overlap_pixels = np.sum(overlap_mask)

        # Compute overlap percentage relative to smaller fragment
        area1 = np.sum(mask1 > 0)
        area2 = np.sum(mask2_transformed > 0)
        min_area = min(area1, area2)
        overlap_percentage = (overlap_pixels / min_area * 100) if min_area > 0 else 0.0

        # Compute minimum gap at edge boundary
        # Find edge pixels (pixels adjacent to background)
        kernel = np.ones((3, 3), dtype=np.uint8)
        edge1 = cv2.dilate(mask1, kernel) - mask1
        edge2 = cv2.dilate(mask2_transformed, kernel) - mask2_transformed

        # Compute distance transform from each edge
        if np.any(edge1) and np.any(edge2):
            dist1 = cv2.distanceTransform(1 - edge1, cv2.DIST_L2, 5)
            dist2 = cv2.distanceTransform(1 - edge2, cv2.DIST_L2, 5)

            # Minimum distance between edges
            edge1_coords = np.argwhere(edge1 > 0)
            edge2_coords = np.argwhere(edge2 > 0)

            if len(edge1_coords) > 0 and len(edge2_coords) > 0:
                # Sample points to avoid expensive computation
                sample_size = min(100, len(edge1_coords), len(edge2_coords))
                edge1_sample = edge1_coords[np.random.choice(len(edge1_coords), sample_size, replace=False)]
                edge2_sample = edge2_coords[np.random.choice(len(edge2_coords), sample_size, replace=False)]

                # Compute pairwise distances
                from scipy.spatial.distance import cdist
                distances = cdist(edge1_sample, edge2_sample, metric='euclidean')
                edge_gap_distance = np.min(distances)
            else:
                edge_gap_distance = 0.0
        else:
            edge_gap_distance = 0.0

        return overlap_pixels, overlap_percentage, edge_gap_distance

    def _should_flip_for_complementary(self, edge1_name: str, edge2_name: str) -> bool:
        """Determine if edge2 should be flipped for complementary matching."""
        # For complementary puzzle-piece matching:
        # - Same edge names (top-top, right-right): flip to face each other
        # - Different edge names: depends on configuration

        if edge1_name == edge2_name:
            return True

        # For different edges, check if they are opposite
        opposite_pairs = [
            ('top_edge', 'bottom_edge'),
            ('bottom_edge', 'top_edge'),
            ('left_edge', 'right_edge'),
            ('right_edge', 'left_edge')
        ]

        if (edge1_name, edge2_name) in opposite_pairs:
            return False

        return True

    def _resample_edge(self, points: np.ndarray, n_points: int) -> np.ndarray:
        """Resample edge to fixed number of points."""
        if len(points) < 2:
            return points

        # Compute cumulative distance
        distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        cumulative = np.concatenate(([0], np.cumsum(distances)))
        total_length = cumulative[-1]

        if total_length == 0:
            return points

        # Resample at uniform intervals
        target_distances = np.linspace(0, total_length, n_points)
        resampled = np.zeros((n_points, 2), dtype=np.float32)
        resampled[:, 0] = np.interp(target_distances, cumulative, points[:, 0])
        resampled[:, 1] = np.interp(target_distances, cumulative, points[:, 1])

        return resampled

    def _compute_transform_least_squares(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute optimal transformation (rotation, translation, scale) using least squares.

        Returns:
            (transform_matrix, alignment_error)
        """
        # Center both point sets
        src_center = src_points.mean(axis=0)
        dst_center = dst_points.mean(axis=0)

        src_centered = src_points - src_center
        dst_centered = dst_points - dst_center

        # Compute scale
        src_scale = np.sqrt(np.sum(src_centered**2) / len(src_centered))
        dst_scale = np.sqrt(np.sum(dst_centered**2) / len(dst_centered))

        if dst_scale > 0:
            scale = src_scale / dst_scale
        else:
            scale = 1.0

        # Normalize
        src_normalized = src_centered / (src_scale + 1e-6)
        dst_normalized = dst_centered / (dst_scale + 1e-6)

        # Compute rotation using SVD
        H = dst_normalized.T @ src_normalized
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Build full transformation matrix
        # T = translation to center, R = rotation, S = scale, T_back = translation back
        transform = np.eye(3)
        transform[:2, :2] = scale * R
        transform[:2, 2] = src_center - scale * R @ dst_center

        # Compute alignment error
        transformed_dst = (transform @ np.column_stack([dst_points, np.ones(len(dst_points))]).T).T
        error = np.mean(np.linalg.norm(src_points - transformed_dst[:, :2], axis=1))

        return transform, error

    def _compute_separation_for_edge_meeting(
        self,
        edge1_name: str,
        edge2_name: str,
        edge1_points: np.ndarray,
        edge2_points_aligned: np.ndarray
    ) -> np.ndarray:
        """
        Compute translation to position fragments edge-to-edge (touching but not overlapping).

        Strategy:
        1. Compute average edge tangent and perpendicular normal direction
        2. Determine which side fragment2 should be on based on edge names
        3. Translate fragment2 perpendicular to edge to eliminate body overlap

        The separation distance must be large enough to prevent body overlap.
        Fragment bodies are typically 500-1200 pixels, so we need substantial separation.

        Args:
            edge1_name: Name of edge1 (e.g., 'left_edge', 'top_edge')
            edge2_name: Name of edge2
            edge1_points: Points along edge1 (reference edge)
            edge2_points_aligned: Points along edge2 after alignment transformation

        Returns:
            separation_vector: [dx, dy] translation to apply to fragment2
        """
        # Compute edge tangent as average direction along edge1
        edge1_tangents = np.diff(edge1_points, axis=0)
        avg_tangent = np.mean(edge1_tangents, axis=0)
        avg_tangent = avg_tangent / (np.linalg.norm(avg_tangent) + 1e-6)

        # Compute perpendicular normal (two possible directions)
        # Rotate tangent by 90 degrees: [x, y] -> [-y, x] or [y, -x]
        normal1 = np.array([-avg_tangent[1], avg_tangent[0]])
        normal2 = np.array([avg_tangent[1], -avg_tangent[0]])

        # Determine separation direction based on edge names
        # Key insight: fragments must be on OPPOSITE sides of the edge boundary
        separation_direction = self._get_separation_direction(edge1_name, edge2_name, normal1, normal2)

        # CRITICAL FIX: Separation must account for FRAGMENT BODY SIZE, not just edge length.
        #
        # The problem: Edge segments are typically 50-200 pixels, but fragment bodies
        # are 500-1200 pixels. Using edge length as a reference leads to massive overlap.
        #
        # Better approach: Use the BOUNDING BOX of the edge points to estimate fragment size.
        # The edge sits on the boundary of the fragment, so its extent gives us a lower bound
        # on fragment dimensions in that direction.

        # Estimate fragment dimensions from edge extent
        edge1_bbox = edge1_points.max(axis=0) - edge1_points.min(axis=0)
        edge2_bbox = edge2_points_aligned.max(axis=0) - edge2_points_aligned.min(axis=0)

        # The larger dimension of the edge bbox is along the edge itself
        # We need separation perpendicular to the edge, so use fragment body estimates
        # Typical manuscript fragments have body "depth" of 300-800 pixels
        # We use a fixed minimum separation plus edge-based adjustment

        edge_length = np.sum(np.linalg.norm(edge1_tangents, axis=1))

        # Base separation: fragments typically extend 3-5x the edge length perpendicular to edge
        # Use conservative estimate to ensure no overlap
        # Also add fixed minimum to handle small edges
        MIN_SEPARATION = 200  # Minimum separation in pixels
        BODY_DEPTH_FACTOR = 3.0  # Assume fragment body extends ~3x edge length perpendicular

        base_separation = max(MIN_SEPARATION, edge_length * BODY_DEPTH_FACTOR)

        # Check current distance between edges (after alignment transform)
        distances = np.linalg.norm(edge1_points - edge2_points_aligned, axis=1)
        avg_distance = np.mean(distances)

        # Add the current distance to ensure we're pushing fragments apart
        # The transform already tried to align edges, so avg_distance is typically small
        separation_distance = base_separation + avg_distance

        # Compute final separation vector
        separation_vector = separation_direction * separation_distance

        return separation_vector

    def _get_separation_direction(
        self,
        edge1_name: str,
        edge2_name: str,
        normal1: np.ndarray,
        normal2: np.ndarray
    ) -> np.ndarray:
        """
        Determine which direction fragment2 should be separated from fragment1.

        CORRECTED Logic:
        The key insight is that fragment bodies are on SPECIFIC sides of their edges:
        - top_edge: Fragment body is BELOW the edge line (larger y values in image coordinates)
        - bottom_edge: Fragment body is ABOVE the edge line (smaller y values)
        - left_edge: Fragment body is to the RIGHT of the edge line (larger x values)
        - right_edge: Fragment body is to the LEFT of the edge line (smaller x values)

        For edge-to-edge meeting WITHOUT overlap:
        - Fragment2 must be positioned on the OPPOSITE side of edge1 from fragment1's body
        - If edge1 is top_edge (body below), fragment2 must go ABOVE → negative y
        - If edge1 is bottom_edge (body above), fragment2 must go BELOW → positive y
        - If edge1 is left_edge (body right), fragment2 must go LEFT → negative x
        - If edge1 is right_edge (body left), fragment2 must go RIGHT → positive x

        Returns:
            unit_vector: direction to separate fragment2
        """
        # Determine separation based on edge1_name (the reference edge)
        # Image coordinates: x increases right, y increases down
        if 'top' in edge1_name:
            # Fragment1 body is below (positive y direction from edge)
            # Fragment2 should go above the edge (negative y direction)
            target_direction = np.array([0, -1.0])
        elif 'bottom' in edge1_name:
            # Fragment1 body is above (negative y direction from edge)
            # Fragment2 should go below the edge (positive y direction)
            target_direction = np.array([0, 1.0])
        elif 'left' in edge1_name:
            # Fragment1 body is to the right (positive x direction from edge)
            # Fragment2 should go left of the edge (negative x direction)
            target_direction = np.array([-1.0, 0])
        elif 'right' in edge1_name:
            # Fragment1 body is to the left (negative x direction from edge)
            # Fragment2 should go right of the edge (positive x direction)
            target_direction = np.array([1.0, 0])
        else:
            # Default: use normal1
            print(f"  Warning: Unknown edge name '{edge1_name}', using normal1")
            return normal1

        # Normalize target direction
        target_direction = target_direction / np.linalg.norm(target_direction)

        # Choose the normal that best aligns with target direction
        dot1 = np.dot(normal1, target_direction)
        dot2 = np.dot(normal2, target_direction)

        if abs(dot1) > abs(dot2):
            chosen = normal1 if dot1 > 0 else -normal1
        else:
            chosen = normal2 if dot2 > 0 else -normal2

        return chosen

    def test_multiple_rotations(
        self,
        fragment1_id: str,
        edge1_name: str,
        fragment2_id: str,
        edge2_name: str,
        fragment1_rgb: np.ndarray,
        fragment1_alpha: np.ndarray,
        fragment2_rgb: np.ndarray,
        fragment2_alpha: np.ndarray,
        edge1_points: np.ndarray,
        edge2_points: np.ndarray
    ) -> Tuple[int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Test multiple rotation angles (0, 90, 180, 270) and return the best one.

        Returns:
            (best_angle, best_error, best_rgb, best_alpha, best_edge_points, best_transform)
        """
        rotation_angles = [0, 90, 180, 270]
        best_angle = 0
        best_error = float('inf')
        best_rgb = fragment2_rgb
        best_alpha = fragment2_alpha
        best_edge_points = edge2_points
        best_transform = None

        print(f"    Testing rotations: ", end='')

        for angle in rotation_angles:
            # Rotate fragment 2
            if angle == 0:
                rotated_rgb = fragment2_rgb
                rotated_alpha = fragment2_alpha
                rotated_edge_points = edge2_points
            else:
                rotated_rgb, rotated_alpha, rotated_edge_points, _ = self.rotate_image_and_points(
                    fragment2_rgb, fragment2_alpha, angle, edge2_points
                )

            # Compute alignment with rotated fragment
            try:
                transform, error, _ = self.compute_alignment_transform(
                    edge1_points,
                    rotated_edge_points,
                    edge1_name,
                    edge2_name
                )

                print(f"{angle}°={error:.1f}px ", end='')

                if error < best_error:
                    best_error = error
                    best_angle = angle
                    best_rgb = rotated_rgb
                    best_alpha = rotated_alpha
                    best_edge_points = rotated_edge_points
                    best_transform = transform
            except Exception as e:
                print(f"{angle}°=FAIL ", end='')

        print(f"| Best: {best_angle}° ({best_error:.1f}px)")

        return best_angle, best_error, best_rgb, best_alpha, best_edge_points, best_transform

    def compute_body_overlap_percentage(
        self,
        alpha1: np.ndarray,
        alpha2: np.ndarray,
        transform: np.ndarray,
        canvas_size: Tuple[int, int],
        frag1_offset: Tuple[int, int]
    ) -> Tuple[float, int, int]:
        """
        Compute the actual body overlap between two aligned fragments.

        This is the critical check: if bodies overlap, the match is INVALID
        because two physical fragments cannot occupy the same space.

        Args:
            alpha1: Alpha channel of fragment 1
            alpha2: Alpha channel of fragment 2 (before transformation)
            transform: Full transformation matrix for fragment 2
            canvas_size: (height, width) of the canvas
            frag1_offset: (x_offset, y_offset) where fragment1 is placed

        Returns:
            (overlap_percentage, overlap_pixels, smaller_fragment_pixels)
        """
        canvas_h, canvas_w = canvas_size
        x1_offset, y1_offset = frag1_offset

        # Create mask for fragment 1 on canvas
        mask1 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        h1, w1 = alpha1.shape
        y1_end = min(y1_offset + h1, canvas_h)
        x1_end = min(x1_offset + w1, canvas_w)
        mask1[y1_offset:y1_end, x1_offset:x1_end] = alpha1[:y1_end-y1_offset, :x1_end-x1_offset]

        # Transform fragment 2 mask
        mask2_transformed = cv2.warpAffine(
            alpha2,
            transform[:2, :],
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # Threshold to binary
        mask1_binary = (mask1 > 127).astype(np.uint8)
        mask2_binary = (mask2_transformed > 127).astype(np.uint8)

        # Compute overlap
        overlap = mask1_binary & mask2_binary
        overlap_pixels = int(np.sum(overlap))

        # Get fragment sizes
        frag1_pixels = int(np.sum(mask1_binary))
        frag2_pixels = int(np.sum(mask2_binary))
        smaller_pixels = min(frag1_pixels, frag2_pixels)

        if smaller_pixels == 0:
            return 0.0, 0, 0

        overlap_percentage = 100.0 * overlap_pixels / smaller_pixels

        return overlap_percentage, overlap_pixels, smaller_pixels

    def is_valid_match(
        self,
        overlap_percentage: float,
        overlap_threshold: float = 1.0
    ) -> Tuple[bool, str]:
        """
        Determine if a match is physically valid based on overlap.

        Args:
            overlap_percentage: Percentage of smaller fragment that overlaps
            overlap_threshold: Maximum allowed overlap (default 1%)

        Returns:
            (is_valid, reason)
        """
        if overlap_percentage > overlap_threshold:
            return False, f"DISQUALIFIED: {overlap_percentage:.1f}% body overlap (threshold: {overlap_threshold}%)"
        return True, "Valid"

    def blend_with_alpha(
        self,
        canvas: np.ndarray,
        overlay_rgb: np.ndarray,
        overlay_alpha: np.ndarray,
        blend_ratio: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Blend overlay onto canvas using alpha channel with proper transparency.

        Returns:
            (blended_canvas, overlap_mask)
        """
        # Normalize alpha to 0-1 range
        overlay_alpha_norm = overlay_alpha.astype(float) / 255.0

        # Track where overlay is present
        overlay_mask = overlay_alpha > 127

        # Track overlapping regions (both canvas and overlay have content)
        canvas_has_content = np.any(canvas < 255, axis=2)
        overlap_mask = canvas_has_content & overlay_mask

        # Blend the images
        result = canvas.copy()

        for c in range(3):
            # Where only overlay is present (no overlap)
            only_overlay = overlay_mask & ~canvas_has_content
            result[:, :, c] = np.where(
                only_overlay,
                overlay_rgb[:, :, c],
                result[:, :, c]
            )

            # Where there's overlap - blend with transparency
            alpha_blend = overlay_alpha_norm * blend_ratio
            result[:, :, c] = np.where(
                overlap_mask,
                (result[:, :, c].astype(float) * (1 - alpha_blend) +
                 overlay_rgb[:, :, c].astype(float) * alpha_blend).astype(np.uint8),
                result[:, :, c]
            )

        return result, overlap_mask

    def create_aligned_composite(
        self,
        fragment1_id: str,
        edge1_name: str,
        fragment2_id: str,
        edge2_name: str,
        score: float,
        output_path: Path,
        canvas_scale: float = 2.5,
        test_rotations: bool = True
    ) -> bool:
        """
        Create composite image with fragment2 aligned to fragment1.
        Tests multiple rotations to find best alignment.

        Returns:
            True if alignment succeeded, False otherwise
        """
        try:
            # Load fragments
            rgb1, alpha1 = self.load_fragment(fragment1_id)
            rgb2, alpha2 = self.load_fragment(fragment2_id)

            # Extract edge points
            edge1_points = self.extract_edge_points(fragment1_id, edge1_name)
            edge2_points = self.extract_edge_points(fragment2_id, edge2_name)

            if edge1_points is None or edge2_points is None:
                print(f"  Could not extract edge points")
                return False

            # Test multiple rotations if enabled
            if test_rotations:
                best_angle, alignment_error, rgb2_rotated, alpha2_rotated, edge2_rotated, transform = \
                    self.test_multiple_rotations(
                        fragment1_id, edge1_name,
                        fragment2_id, edge2_name,
                        rgb1, alpha1,
                        rgb2, alpha2,
                        edge1_points, edge2_points
                    )
            else:
                # No rotation testing - use original
                best_angle = 0
                rgb2_rotated = rgb2
                alpha2_rotated = alpha2
                edge2_rotated = edge2_points
                transform, alignment_error, _ = self.compute_alignment_transform(
                    edge1_points, edge2_points, edge1_name, edge2_name
                )

            # Create large canvas for overlay
            h1, w1 = rgb1.shape[:2]
            h2, w2 = rgb2_rotated.shape[:2]
            max_dim = max(h1, w1, h2, w2)
            canvas_h = int(max_dim * canvas_scale)
            canvas_w = int(max_dim * canvas_scale)

            # Create figure with 3 panels
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))

            # Panel 1: Fragment 1
            ax1 = axes[0]
            # Show with transparency
            frag1_display = np.dstack([rgb1, alpha1])
            ax1.imshow(frag1_display)
            ax1.set_title(f'Fragment 1: {fragment1_id}\nEdge: {edge1_name}',
                         fontsize=12, fontweight='bold')
            ax1.axis('off')

            # Highlight edge
            if edge1_points is not None:
                ax1.plot(edge1_points[:, 0], edge1_points[:, 1],
                        'b-', linewidth=3, label='Edge 1', alpha=0.8)
                ax1.scatter(edge1_points[0, 0], edge1_points[0, 1],
                           c='green', s=100, zorder=5, label='Start')
                ax1.scatter(edge1_points[-1, 0], edge1_points[-1, 1],
                           c='red', s=100, zorder=5, label='End')
                ax1.legend(loc='upper right')

            # Panel 2: Fragment 2 (with best rotation applied)
            ax2 = axes[1]
            frag2_display = np.dstack([rgb2_rotated, alpha2_rotated])
            ax2.imshow(frag2_display)
            rotation_text = f' (Rotated {best_angle}°)' if best_angle != 0 else ''
            ax2.set_title(f'Fragment 2: {fragment2_id}{rotation_text}\nEdge: {edge2_name}',
                         fontsize=12, fontweight='bold')
            ax2.axis('off')

            # Highlight edge
            if edge2_rotated is not None:
                ax2.plot(edge2_rotated[:, 0], edge2_rotated[:, 1],
                        'r-', linewidth=3, label='Edge 2', alpha=0.8)
                ax2.scatter(edge2_rotated[0, 0], edge2_rotated[0, 1],
                           c='green', s=100, zorder=5, label='Start')
                ax2.scatter(edge2_rotated[-1, 0], edge2_rotated[-1, 1],
                           c='red', s=100, zorder=5, label='End')
                ax2.legend(loc='upper right')

            # Panel 3: Aligned composite with proper transparency overlay
            ax3 = axes[2]

            # Create composite canvas (white background)
            composite = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

            # Place fragment 1 in center with alpha blending
            y1_offset = (canvas_h - h1) // 2
            x1_offset = (canvas_w - w1) // 2

            # Blend fragment 1 onto canvas
            alpha1_norm = alpha1.astype(float) / 255.0
            for c in range(3):
                composite[y1_offset:y1_offset+h1, x1_offset:x1_offset+w1, c] = \
                    (composite[y1_offset:y1_offset+h1, x1_offset:x1_offset+w1, c] * (1 - alpha1_norm) +
                     rgb1[:, :, c] * alpha1_norm).astype(np.uint8)

            # Transform fragment 2 using computed transformation
            # Adjust transform to account for canvas offset
            offset_transform = np.eye(3)
            offset_transform[:2, 2] = [x1_offset, y1_offset]
            full_transform = offset_transform @ transform

            # Apply transformation to fragment 2
            transformed_rgb2 = cv2.warpAffine(
                rgb2_rotated,
                full_transform[:2, :],
                (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )

            transformed_alpha2 = cv2.warpAffine(
                alpha2_rotated,
                full_transform[:2, :],
                (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            # Blend fragment 2 onto composite with transparency
            composite, overlap_mask = self.blend_with_alpha(
                composite,
                transformed_rgb2,
                transformed_alpha2,
                blend_ratio=0.5
            )

            # Compute ACTUAL body overlap using proper mask intersection
            body_overlap_pct, overlap_pixels, smaller_pixels = self.compute_body_overlap_percentage(
                alpha1,
                alpha2_rotated,
                full_transform,
                (canvas_h, canvas_w),
                (x1_offset, y1_offset)
            )

            # Check if match is valid (overlap > 1% = DISQUALIFIED)
            is_valid, validity_reason = self.is_valid_match(body_overlap_pct, overlap_threshold=1.0)

            # Compute comprehensive metrics (for display)
            _, _, edge_gap = self.compute_overlap_and_gap_metrics(
                alpha1,
                alpha2_rotated,
                full_transform,
                (canvas_h, canvas_w),
                (x1_offset, y1_offset)
            )

            # Compute complementary fit score
            if edge1_points is not None and edge2_rotated is not None:
                edge1_canvas = edge1_points + np.array([x1_offset, y1_offset])
                complementary_fit = self.compute_complementary_fit(
                    edge1_canvas,
                    edge2_rotated,
                    full_transform
                )
            else:
                complementary_fit = 0.0

            # Compute edge length for length-weighted scoring
            edge1_length = float(np.sum(np.linalg.norm(np.diff(edge1_points, axis=0), axis=1)))
            edge2_length = float(np.sum(np.linalg.norm(np.diff(edge2_rotated, axis=0), axis=1)))
            avg_edge_length = (edge1_length + edge2_length) / 2

            # Length penalty: shorter edges get penalized
            # No penalty for edges >= 150px, increasing penalty below that
            if avg_edge_length >= 150:
                length_penalty = 0.0
            else:
                length_penalty = (150 - avg_edge_length) / 150 * 0.1  # Max 0.1 penalty

            # Compute final adjusted score
            adjusted_score = score + length_penalty

            # Show composite
            ax3.imshow(composite)

            # Highlight overlap region with red overlay (body overlap is BAD)
            if np.any(overlap_mask):
                overlap_vis = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
                overlap_vis[overlap_mask, :] = [255, 0, 0, 100]
                ax3.imshow(overlap_vis)

            # Determine quality category based on validity and metrics
            if not is_valid:
                quality = f"DISQUALIFIED ({body_overlap_pct:.1f}% overlap)"
                quality_color = "red"
            elif body_overlap_pct < 0.5 and complementary_fit < 5.0:
                quality = "Excellent - Valid Match"
                quality_color = "green"
            elif body_overlap_pct < 1.0 and complementary_fit < 10.0:
                quality = "Good - Valid Match"
                quality_color = "blue"
            else:
                quality = "Fair - Valid Match"
                quality_color = "orange"

            # Add comprehensive metrics
            length_info = f"Edge Length: {avg_edge_length:.0f}px"
            if length_penalty > 0:
                length_info += f" (penalty: +{length_penalty:.3f})"

            ax3.set_title(
                f'Aligned Composite: {quality}\n'
                f'Original Score: {score:.4f} | Adjusted: {adjusted_score:.4f} | {length_info}\n'
                f'Body Overlap: {body_overlap_pct:.2f}% | Complementary Fit: {complementary_fit:.1f}px | Gap: {edge_gap:.1f}px\n'
                f'Rotation: {best_angle}° | {"✓ VALID" if is_valid else "✗ INVALID - BODY OVERLAP"}',
                fontsize=10,
                fontweight='bold',
                color=quality_color
            )
            ax3.axis('off')

            plt.tight_layout()
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close()

            # Return tuple: (success, is_valid, adjusted_score, body_overlap_pct)
            return True, is_valid, adjusted_score, body_overlap_pct

        except Exception as e:
            print(f"  Error creating alignment: {e}")
            import traceback
            traceback.print_exc()
            return False, False, float('inf'), 100.0


def generate_alignments_for_best_matches(args):
    """Generate aligned composites for best matches from evaluation."""
    all_matches_path = Path(args.evaluation_matches)
    fragments_dir = Path(args.fragments_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING ALIGNED COMPOSITES FOR BEST MATCHES")
    print("=" * 60)

    # Load evaluation results
    with open(all_matches_path, 'r') as f:
        all_matches = json.load(f)

    print(f"\nFound {len(all_matches)} fragments in evaluation")
    print(f"Threshold: score < {args.score_threshold}")
    print(f"Rotation testing: {'Enabled' if args.test_rotations else 'Disabled'}")

    aligner = FragmentAligner(fragments_dir)
    success_count = 0
    valid_count = 0
    disqualified_count = 0
    total_attempts = 0

    # Track results for summary
    valid_matches = []
    disqualified_matches = []

    # For each fragment, create alignments for excellent matches
    for fragment_id, edges in all_matches.items():
        print(f"\n{fragment_id}:")

        for edge_name, matches in edges.items():
            if not matches:
                continue

            # Get best match
            best_match = matches[0]
            match_fragment = best_match['match_fragment']
            match_edge = best_match['match_edge']
            score = best_match['score']

            # Only process if score is below threshold
            if score >= args.score_threshold:
                continue

            total_attempts += 1

            output_file = output_dir / f"ALIGNED_{fragment_id}_{edge_name}_{match_fragment}_{match_edge}.png"

            success, is_valid, adjusted_score, overlap_pct = aligner.create_aligned_composite(
                fragment_id,
                edge_name,
                match_fragment,
                match_edge,
                score,
                output_file,
                test_rotations=args.test_rotations
            )

            if success:
                success_count += 1
                if is_valid:
                    valid_count += 1
                    status = "✓ VALID"
                    valid_matches.append({
                        'query': f"{fragment_id}[{edge_name}]",
                        'match': f"{match_fragment}[{match_edge}]",
                        'score': score,
                        'adjusted_score': adjusted_score,
                        'overlap_pct': overlap_pct
                    })
                else:
                    disqualified_count += 1
                    status = f"✗ DISQUALIFIED ({overlap_pct:.1f}% overlap)"
                    disqualified_matches.append({
                        'query': f"{fragment_id}[{edge_name}]",
                        'match': f"{match_fragment}[{match_edge}]",
                        'score': score,
                        'overlap_pct': overlap_pct
                    })
                print(f"  {status} | {edge_name} -> {match_fragment}[{match_edge}] | orig:{score:.4f} adj:{adjusted_score:.4f}")
            else:
                print(f"  ✗ FAILED | {edge_name} -> {match_fragment}[{match_edge}]")

    print("\n" + "=" * 60)
    print(f"ALIGNMENT GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total processed: {success_count}/{total_attempts}")
    print(f"  ✓ Valid matches: {valid_count}")
    print(f"  ✗ Disqualified (overlap): {disqualified_count}")
    print(f"Output directory: {output_dir}")

    # Save summary JSON
    summary = {
        'total_processed': success_count,
        'valid_matches': valid_count,
        'disqualified_matches': disqualified_count,
        'valid_match_list': sorted(valid_matches, key=lambda x: x['adjusted_score']),
        'disqualified_list': disqualified_matches
    }
    summary_path = output_dir / 'validation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    if valid_count > 0:
        print(f"\n🏆 TOP 10 VALID MATCHES (by adjusted score):")
        for i, m in enumerate(sorted(valid_matches, key=lambda x: x['adjusted_score'])[:10], 1):
            print(f"  {i:2d}. {m['adjusted_score']:.4f} | {m['query']} ↔ {m['match']}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate aligned composite images for matched fragments"
    )
    parser.add_argument(
        '--evaluation-matches',
        type=str,
        required=True,
        help='Path to all_matches.json from evaluation'
    )
    parser.add_argument(
        '--fragments-dir',
        type=str,
        required=True,
        help='Directory containing fragment images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/aligned_composites',
        help='Output directory for aligned images'
    )
    parser.add_argument(
        '--score-threshold',
        type=float,
        default=0.25,
        help='Only create alignments for matches below this score (default: 0.25)'
    )
    parser.add_argument(
        '--test-rotations',
        action='store_true',
        default=True,
        help='Test multiple rotation angles (0, 90, 180, 270) to find best alignment'
    )
    parser.add_argument(
        '--no-test-rotations',
        dest='test_rotations',
        action='store_false',
        help='Disable rotation testing'
    )

    args = parser.parse_args()
    generate_alignments_for_best_matches(args)


if __name__ == '__main__':
    main()
