"""
Position computation: compute where to place fragment B relative to fragment A
when their edges match.

Adapted from edge_matching_v2/align_and_validate.py FragmentAligner (f3a3a3c).
Works entirely in cm-space so the result can be converted to canvas coordinates
with  canvas_px = relative_cm * gridScale.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import NUM_RESAMPLE_POINTS
from .edge_extraction import ScaleAwareDescriptor, _resample_points


@dataclass
class PlacementResult:
    relative_x_cm: float
    relative_y_cm: float
    rotation_deg: float
    alignment_error: float  # cm-space RMSE


class PositionComputer:
    """Compute placement of fragment B relative to A from matched edges."""

    def compute_placement(
        self,
        edge_a: ScaleAwareDescriptor,
        edge_b: ScaleAwareDescriptor,
    ) -> PlacementResult:
        """
        1. Get edge points in cm-space
        2. Flip edge_b for complementary fit
        3. Least-squares rotation + translation to align B's edge onto A's edge
        4. Return relative position of B's image origin vs A's image origin

        The transform maps B's edge points onto A's edge points. Since both
        edges are contour boundaries, the fragment bodies naturally extend to
        opposite sides of the aligned edge — no additional separation is needed.
        """
        n_pts = min(len(edge_a.points_cm), len(edge_b.points_cm), 50)
        pts_a = self._resample(edge_a.points_cm, n_pts)
        pts_b = self._resample(edge_b.points_cm, n_pts)

        # Determine complementary flip
        if self._should_flip(edge_a.position_label, edge_b.position_label):
            pts_b = pts_b[::-1]

        # Find best circular-shift alignment
        best_transform = None
        best_error = float('inf')

        for offset in range(0, n_pts, max(1, n_pts // 10)):
            shifted = np.roll(pts_b, offset, axis=0)
            transform, error = self._least_squares_transform(pts_a, shifted)
            if error < best_error:
                best_error = error
                best_transform = transform

        # Compute relative offset of B's origin (0,0 in cm = top-left of image)
        origin_b_cm = np.array([0.0, 0.0])
        origin_b_in_a = self._apply_transform(best_transform, origin_b_cm.reshape(1, 2))[0]

        relative_x = float(origin_b_in_a[0])
        relative_y = float(origin_b_in_a[1])

        # Extract rotation from transform
        rotation_rad = np.arctan2(best_transform[1, 0], best_transform[0, 0])
        rotation_deg = float(np.degrees(rotation_rad))

        return PlacementResult(
            relative_x_cm=relative_x,
            relative_y_cm=relative_y,
            rotation_deg=rotation_deg,
            alignment_error=best_error,
        )

    # ------------------------------------------------------------------
    # Helpers (adapted from v2 FragmentAligner)
    # ------------------------------------------------------------------

    @staticmethod
    def _resample(points: np.ndarray, n: int) -> np.ndarray:
        r = _resample_points(points.astype(np.float32), n)
        if r is None:
            return points[:n] if len(points) >= n else points
        return r

    @staticmethod
    def _should_flip(pos_a: str, pos_b: str) -> bool:
        """For complementary edges (left↔right), flip b so curves face each other."""
        opposite = {
            ('left', 'right'), ('right', 'left'),
            ('top', 'bottom'), ('bottom', 'top'),
        }
        return (pos_a, pos_b) not in opposite

    @staticmethod
    def _least_squares_transform(
        src: np.ndarray, dst: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Rotation + translation + uniform scale via SVD."""
        src_c = src.mean(axis=0)
        dst_c = dst.mean(axis=0)
        s = src - src_c
        d = dst - dst_c
        src_scale = np.sqrt(np.sum(s ** 2) / len(s))
        dst_scale = np.sqrt(np.sum(d ** 2) / len(d))
        scale = src_scale / dst_scale if dst_scale > 0 else 1.0
        s_n = s / (src_scale + 1e-9)
        d_n = d / (dst_scale + 1e-9)
        H = d_n.T @ s_n
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        T = np.eye(3)
        T[:2, :2] = scale * R
        T[:2, 2] = src_c - scale * R @ dst_c
        # Error
        transformed = (T @ np.column_stack([dst, np.ones(len(dst))]).T).T[:, :2]
        error = float(np.mean(np.linalg.norm(src - transformed, axis=1)))
        return T, error

    @staticmethod
    def _apply_transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
        h = np.column_stack([pts, np.ones(len(pts))])
        return (T @ h.T).T[:, :2]
