import os
import cv2
import numpy as np
from ml_pipeline.core.processor import BaseProcessor, ProcessorMetadata, FragmentRecord, ProcessingResult

import cv2
import numpy as np
from typing import Dict, List, Tuple

class EdgeExtractor:
    def __init__(self, smoothing_iterations: int = 2, epsilon_factor: float = 0.001):
        self.smoothing_iterations = smoothing_iterations
        self.epsilon_factor = epsilon_factor

    def classify_edges(
        self,
        contour: np.ndarray,
        mask: np.ndarray,
        near_frac: float = 0.05,          # how close points must be to bbox side (fraction of bbox size)
        min_run_frac: float = 0.10,       # minimum run size vs side length to count
        max_mean_line_err: float = 4.0,   # pixels: lower => stricter straight border
    ) -> Dict:
        """
        Robustly classify which sides are true borders (straight) vs tears (irregular).
        Uses fragment bounding box, contiguous runs, and line-fit residual.
        """
        pts = contour.reshape(-1, 2)
        n = len(pts)

        # Tight bounding box of the fragment (use mask so bbox is stable)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            raise ValueError("Empty mask")
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        bw = max(1, x1 - x0 + 1)
        bh = max(1, y1 - y0 + 1)

        # pixel tolerance for being "near" a bbox side
        tol_x = max(2, int(bw * near_frac))
        tol_y = max(2, int(bh * near_frac))

        # side tests
        is_top    = np.abs(pts[:, 1] - y0) <= tol_y
        is_bottom = np.abs(pts[:, 1] - y1) <= tol_y
        is_left   = np.abs(pts[:, 0] - x0) <= tol_x
        is_right  = np.abs(pts[:, 0] - x1) <= tol_x

        side_masks = {
            "top_edge": is_top,
            "bottom_edge": is_bottom,
            "left_edge": is_left,
            "right_edge": is_right,
        }

        edge_data = {
            "contour": contour,
            "bbox": (x0, y0, x1, y1),
            "top_edge": [],
            "bottom_edge": [],
            "left_edge": [],
            "right_edge": [],
            "tear_edges": [],
            "border_edges": [],
            "scores": {},  # debug: per-side metrics
        }

        # classify each side based on contiguous runs
        for side, mask_bool in side_masks.items():
            runs = self._contiguous_runs(mask_bool)
            # merge wrap-around runs (contour is circular)
            runs = self._merge_wraparound_runs(runs, n)

            best = None  # best run by coverage
            for (a, b) in runs:
                run_pts = pts[a:b+1]
                if len(run_pts) < 20:
                    continue

                # coverage vs side length (approx)
                side_len = bw if side in ("top_edge", "bottom_edge") else bh
                coverage = self._run_span_along_side(run_pts, side) / max(1.0, float(side_len))

                if coverage < min_run_frac:
                    continue

                mean_err = self._mean_line_fit_error(run_pts)

                cand = (coverage, mean_err, (a, b))
                if best is None or coverage > best[0]:
                    best = cand

            # store the actual edge points (for viz)
            if best is not None:
                _, mean_err, (a, b) = best
                idx_pts = [(i, pts[i]) for i in range(a, b+1)]
                edge_data[side] = idx_pts

                edge_data["scores"][side] = {
                    "coverage": float(best[0]),
                    "mean_line_err": float(mean_err),
                }

                # classify border vs tear
                if mean_err <= max_mean_line_err:
                    edge_data["border_edges"].append(side)
                else:
                    edge_data["tear_edges"].append(side)
            else:
                edge_data["scores"][side] = {
                    "coverage": 0.0,
                    "mean_line_err": None,
                }

        # overall piece type
        borders = set(edge_data["border_edges"])
        if len(borders) == 0:
            edge_data["piece_type"] = "interior"
        elif len(borders) == 1:
            edge_data["piece_type"] = "edge"
        else:
            # if two adjacent borders -> corner
            adj = {frozenset(("top_edge","left_edge")),
                   frozenset(("top_edge","right_edge")),
                   frozenset(("bottom_edge","left_edge")),
                   frozenset(("bottom_edge","right_edge"))}
            if any(frozenset(pair) <= borders for pair in adj):
                edge_data["piece_type"] = "corner"
            else:
                edge_data["piece_type"] = "edge"  # could be weird shape / partial border

        return edge_data

    def _contiguous_runs(self, mask_bool: np.ndarray) -> List[Tuple[int, int]]:
        """Return contiguous True runs as inclusive index ranges [(start,end), ...]."""
        runs = []
        n = len(mask_bool)
        i = 0
        while i < n:
            if not mask_bool[i]:
                i += 1
                continue
            start = i
            while i + 1 < n and mask_bool[i + 1]:
                i += 1
            end = i
            runs.append((start, end))
            i += 1
        return runs

    def _merge_wraparound_runs(self, runs: List[Tuple[int,int]], n: int) -> List[Tuple[int,int]]:
        """
        If a run exists at the start and end, merge them (circular contour).
        """
        if not runs:
            return runs
        if runs[0][0] == 0 and runs[-1][1] == n - 1:
            merged = (runs[-1][0], runs[0][1])
            return [merged] + runs[1:-1]
        return runs

    def _run_span_along_side(self, run_pts: np.ndarray, side: str) -> float:
        """
        Approx span along the side direction: for top/bottom use x-range, for left/right use y-range.
        """
        if side in ("top_edge", "bottom_edge"):
            return float(run_pts[:, 0].max() - run_pts[:, 0].min())
        else:
            return float(run_pts[:, 1].max() - run_pts[:, 1].min())

    def _mean_line_fit_error(self, pts: np.ndarray) -> float:
        """
        Fit a line and compute mean orthogonal distance (pixels).
        More stable than endpoint straightness ratio when runs are long/curvy.
        """
        pts_f = pts.astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts_f, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)

        # orth distance from point p to line through (x0,y0) direction (vx,vy)
        dx = pts_f[:, 0] - x0
        dy = pts_f[:, 1] - y0
        # |(p - p0) x v| / |v|
        cross = np.abs(dx * vy - dy * vx)
        denom = max(1e-6, np.sqrt(vx*vx + vy*vy))
        return float(np.mean(cross / denom))


class EdgeDetectionProcessor(BaseProcessor):
    def _setup(self) -> None:
        self.extractor = EdgeExtractor()

        # optional: thresholds from config
        self.near_frac = self.config['config'].get('near_frac', 0.03)
        self.min_run_frac = self.config['config'].get('min_run_frac', 0.15)
        self.max_mean_line_err = self.config['config'].get('max_mean_line_err', 2.5)

        self.version = self.config['config'].get('model_version', '1.0')

    def get_metadata(self) -> ProcessorMetadata:
        return ProcessorMetadata(
            name="edgedetection",
            version=self.version,
            description="Detects fragment border edges (top/bottom/left/right)",
            requires_gpu=False,
            batch_size=1,
        )

    def should_process(self, fragment: FragmentRecord) -> bool:
        # Only process if edges not set
        #return fragment.has_top_edge is None and fragment.has_bottom_edge is None
        return True

    def process(self, fragment: FragmentRecord, data_dir: str) -> ProcessingResult:
        img_path = os.path.join(data_dir, fragment.image_path)
        if not os.path.exists(img_path):
            return ProcessingResult(success=False, updates={}, error=f"Image not found: {img_path}")

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            return ProcessingResult(success=False, updates={}, error=f"Failed to read image: {img_path}")

        # TODO: build mask + contour
        # If you already have segmentation masks, use them.
        # Otherwise, create a simple mask from the image (threshold).
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contour = self._largest_contour(mask)

        edge_data = self.extractor.classify_edges(
            contour=contour,
            mask=mask,
            near_frac=self.near_frac,
            min_run_frac=self.min_run_frac,
            max_mean_line_err=self.max_mean_line_err,
        )

        # Set edge flags
        top = 1 if "top_edge" in edge_data["border_edges"] else 0
        bottom = 1 if "bottom_edge" in edge_data["border_edges"] else 0
        left = 1 if "left_edge" in edge_data["border_edges"] else 0
        right = 1 if "right_edge" in edge_data["border_edges"] else 0

        edge_piece = 1 if (top or bottom or left or right) else 0

        return ProcessingResult(
            success=True,
            updates={
                "has_top_edge": top,
                "has_bottom_edge": bottom,
                "edge_piece": edge_piece,
                "processing_status": "completed",
                "last_processed_at": "CURRENT_TIMESTAMP",
                "processing_error": None,
            },
            metadata={
                "piece_type": edge_data["piece_type"],
                "borders": edge_data["border_edges"],
                "tears": edge_data["tear_edges"],
            }
        )

    def _largest_contour(self, mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contour found")
        return max(contours, key=cv2.contourArea)

    def cleanup(self) -> None:
        pass
