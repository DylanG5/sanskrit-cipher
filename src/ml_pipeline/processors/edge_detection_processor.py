import os
import json
import math
import cv2
import numpy as np
from ml_pipeline.core.processor import BaseProcessor, ProcessorMetadata, FragmentRecord, ProcessingResult
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
        min_run_frac: float = 0.05,       # minimum run size vs side length to count
        max_mean_line_err: float = 18.0,   # pixels: lower => stricter straight border
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

    def classify_edges_hull_segments(
        self,
        contour: np.ndarray,
        mask: np.ndarray,
        err_thresh: float = 3.5,
        min_len_pts: int = 35,
        min_cov: float = 0.45,
        max_err: float = 4.5,
        near_frac: float = 0.06,
        angle_h: float = 12.0,
        angle_v: float = 78.0,
    ) -> Dict:
        """
        Classify edges using convex hull segmentation + line-fit angle/error.
        """
        pts = contour.reshape(-1, 2)
        if len(pts) < 3:
            raise ValueError("Contour too small")

        h, w = mask.shape[:2]
        hull_poly = cv2.convexHull(contour)
        hull_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(hull_mask, [hull_poly], 255)

        hull_contours, _ = cv2.findContours(
            hull_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not hull_contours:
            raise ValueError("No hull contour found")
        hull_dense = max(hull_contours, key=cv2.contourArea).reshape(-1, 2)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            raise ValueError("Empty mask")
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        xs_c = pts[:, 0]
        ys_c = pts[:, 1]
        x_left = np.percentile(xs_c, 1)
        x_right = np.percentile(xs_c, 99)
        y_top = np.percentile(ys_c, 1)
        y_bot = np.percentile(ys_c, 99)
        bw = max(1.0, x_right - x_left)
        bh = max(1.0, y_bot - y_top)

        min_len_pts = max(15, int(0.08 * len(hull_dense)))
        segs = self._extract_straight_segments(
            hull_dense, err_thresh=err_thresh, min_len_pts=min_len_pts
        )

        sides = ("left_edge", "right_edge", "top_edge", "bottom_edge")
        best = {side: None for side in sides}
        best_seg = {side: None for side in sides}

        def update(side: str, cov: float, err: float, seg: np.ndarray) -> None:
            cur = best[side]
            if cur is None or cov > cur[0]:
                best[side] = (cov, err)
                best_seg[side] = seg

        near_x = near_frac * bw
        near_y = near_frac * bh

        for seg in segs:
            err = self._mean_line_fit_error(seg)
            ang = self._line_angle_deg(seg)

            x_med = float(np.median(seg[:, 0]))
            y_med = float(np.median(seg[:, 1]))

            cov_h = (seg[:, 0].max() - seg[:, 0].min()) / bw
            cov_v = (seg[:, 1].max() - seg[:, 1].min()) / bh

            if ang < angle_h and cov_h >= min_cov:
                if abs(y_med - y_top) <= near_y:
                    update("top_edge", cov_h, err, seg)
                if abs(y_med - y_bot) <= near_y:
                    update("bottom_edge", cov_h, err, seg)

            if ang > angle_v and cov_v >= min_cov:
                if abs(x_med - x_left) <= near_x:
                    update("left_edge", cov_v, err, seg)
                if abs(x_med - x_right) <= near_x:
                    update("right_edge", cov_v, err, seg)

        edge_data = {
            "contour": contour,
            "bbox": (x0, y0, x1, y1),
            "top_edge": None,
            "bottom_edge": None,
            "left_edge": None,
            "right_edge": None,
            "tear_edges": [],
            "border_edges": [],
            "scores": {},
        }

        for side in sides:
            if best[side] is None:
                edge_data["scores"][side] = {
                    "coverage": 0.0,
                    "mean_line_err": None,
                }
                continue

            cov, err = best[side]
            edge_data[side] = best_seg[side]
            edge_data["scores"][side] = {
                "coverage": float(cov),
                "mean_line_err": float(err),
            }
            if err <= max_err:
                edge_data["border_edges"].append(side)
            else:
                edge_data["tear_edges"].append(side)

        borders = set(edge_data["border_edges"])
        if len(borders) == 0:
            edge_data["piece_type"] = "interior"
        elif len(borders) == 1:
            edge_data["piece_type"] = "edge"
        else:
            adj = {
                frozenset(("top_edge", "left_edge")),
                frozenset(("top_edge", "right_edge")),
                frozenset(("bottom_edge", "left_edge")),
                frozenset(("bottom_edge", "right_edge")),
            }
            if any(frozenset(pair) <= borders for pair in adj):
                edge_data["piece_type"] = "corner"
            else:
                edge_data["piece_type"] = "edge"

        return edge_data

    def _extract_straight_segments(
        self,
        hpts: np.ndarray,
        err_thresh: float,
        min_len_pts: int,
    ) -> List[np.ndarray]:
        segs = []
        if len(hpts) == 0:
            return segs
        cur = [hpts[0]]
        for i in range(1, len(hpts)):
            cur.append(hpts[i])
            if len(cur) >= min_len_pts:
                err = self._mean_line_fit_error(np.array(cur))
                if err > err_thresh:
                    cur.pop()
                    if len(cur) >= min_len_pts:
                        segs.append(np.array(cur))
                    cur = [hpts[i]]
        if len(cur) >= min_len_pts:
            segs.append(np.array(cur))
        return segs

    def _line_angle_deg(self, seg_pts: np.ndarray) -> float:
        pts_f = seg_pts.astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts_f, cv2.DIST_L2, 0, 0.01, 0.01)
        ang = abs(math.degrees(math.atan2(float(vy), float(vx))))
        if ang > 90:
            ang = 180 - ang
        return float(ang)

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
        self.edge_method = self.config['config'].get('edge_method', 'bbox_runs')

        self.hull_err_thresh = self.config['config'].get('hull_err_thresh', 3.5)
        self.hull_min_len_pts = self.config['config'].get('hull_min_len_pts', 35)
        self.hull_min_cov = self.config['config'].get('hull_min_cov', 0.45)
        self.hull_max_err = self.config['config'].get('hull_max_err', 4.5)
        self.hull_near_frac = self.config['config'].get('hull_near_frac', 0.06)
        self.hull_angle_h = self.config['config'].get('hull_angle_h', 12.0)
        self.hull_angle_v = self.config['config'].get('hull_angle_v', 78.0)

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

        # Build mask + contour from segmentation_coords
        if not fragment.segmentation_coords:
            return ProcessingResult(
                success=False,
                updates={"processing_error": "No segmentation_coords found"},
                error="No segmentation_coords found"
            )

        seg = json.loads(fragment.segmentation_coords)
        contours = seg.get("contours", [])
        if not contours:
            return ProcessingResult(
                success=False,
                updates={"processing_error": "segmentation_coords missing contours"},
                error="segmentation_coords missing contours"
            )

        # pick the largest contour by area
        all_contours = [np.array(c, dtype=np.int32) for c in contours if len(c) >= 3]
        if not all_contours:
            return ProcessingResult(
                success=False,
                updates={"processing_error": "No valid contours"},
                error="No valid contours"
            )

        contour = max(all_contours, key=lambda c: cv2.contourArea(c))

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)


        if self.edge_method == "hull_segments":
            edge_data = self.extractor.classify_edges_hull_segments(
                contour=contour,
                mask=mask,
                err_thresh=self.hull_err_thresh,
                min_len_pts=self.hull_min_len_pts,
                min_cov=self.hull_min_cov,
                max_err=self.hull_max_err,
                near_frac=self.hull_near_frac,
                angle_h=self.hull_angle_h,
                angle_v=self.hull_angle_v,
            )
        else:
            edge_data = self.extractor.classify_edges(
                contour=contour,
                mask=mask,
                near_frac=self.near_frac,
                min_run_frac=self.min_run_frac,
                max_mean_line_err=self.max_mean_line_err,
            )
        
        print("EDGE SCORES:", fragment.fragment_id, edge_data["scores"])
        print("BORDERS:", edge_data["border_edges"])
        
        scores = edge_data["scores"]

        def ok(side, min_cov=0.50, max_err=12.0):
            cov = scores[side]["coverage"]
            err = scores[side]["mean_line_err"]
            return cov is not None and cov >= min_cov and err is not None and err <= max_err

        top_ok = ok("top_edge")
        bottom_ok = ok("bottom_edge")

        # optional: keep left/right off for now
        left_ok = False
        right_ok = False

        top = 1 if top_ok else 0
        bottom = 1 if bottom_ok else 0
        left = 1 if left_ok else 0
        right = 1 if right_ok else 0

        edge_piece = 1 if (top or bottom) else 0



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
