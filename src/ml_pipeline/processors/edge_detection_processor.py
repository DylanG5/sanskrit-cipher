import os
import json
import math
from pathlib import Path
import cv2
import numpy as np
import torch
from ml_pipeline.core.processor import BaseProcessor, ProcessorMetadata, FragmentRecord, ProcessingResult
from typing import Any, Dict, List, Optional, Sequence, Tuple

from edge_classification import load_edge_classifier
from edge_classification.dataset import build_edge_image_transform, render_fragment_image

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

    def classify_edges_oriented_runs(
        self,
        contour: np.ndarray,
        mask: np.ndarray,
        near_frac: float = 0.04,
        min_run_frac: float = 0.40,
        max_line_err_frac: float = 0.02,
        max_side_offset_frac: float = 0.025,
        angle_tol_deg: float = 18.0,
        min_confidence: float = 0.60,
        smoothing_window: int = 7,
        trim_frac: float = 0.10,
    ) -> Dict:
        """
        Classify edges in the fragment's oriented frame using the actual dense contour.

        Improvements over the older heuristics:
        - Uses a dense contour extracted from the mask, not the sparse stored contour.
        - Works in the fragment's min-area-rect frame instead of raw image axes.
        - Scores sides using real contour support, normalized line error, offset, and angle.
        - Avoids convex-hull hallucinations on torn / concave edges.
        """
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            raise ValueError("Empty mask")

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        dense_contour = self._largest_mask_contour(mask, chain_approx=cv2.CHAIN_APPROX_NONE)
        dense_pts = dense_contour.reshape(-1, 2).astype(np.float32)
        if len(dense_pts) < 20:
            raise ValueError("Contour too small")

        smooth_pts = self._smooth_closed_contour(
            dense_pts,
            window=max(3, int(smoothing_window)),
            iterations=max(1, int(self.smoothing_iterations)),
        )

        rect = cv2.minAreaRect(smooth_pts.reshape(-1, 1, 2).astype(np.float32))
        local_pts, width, height = self._project_to_min_area_rect_frame(smooth_pts, rect)

        width = max(width, 1.0)
        height = max(height, 1.0)
        scale = max(1.0, min(width, height))

        tol_x = max(2.0, width * near_frac)
        tol_y = max(2.0, height * near_frac)

        side_masks = {
            "top_edge": np.abs(local_pts[:, 1] - 0.0) <= tol_y,
            "bottom_edge": np.abs(local_pts[:, 1] - height) <= tol_y,
            "left_edge": np.abs(local_pts[:, 0] - 0.0) <= tol_x,
            "right_edge": np.abs(local_pts[:, 0] - width) <= tol_x,
        }
        side_targets = {
            "top_edge": 0.0,
            "bottom_edge": height,
            "left_edge": 0.0,
            "right_edge": width,
        }

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

        for side, mask_bool in side_masks.items():
            runs = self._merge_wraparound_runs(self._contiguous_runs(mask_bool), len(local_pts))
            best = None

            for a, b in runs:
                run_local = self._extract_circular_run_points(local_pts, a, b)
                run_global = self._extract_circular_run_points(smooth_pts, a, b)
                if len(run_local) < 12:
                    continue

                score_local = self._trim_run_points(run_local, trim_frac=trim_frac)
                score_global = self._trim_run_points(run_global, trim_frac=trim_frac)
                if len(score_local) < 8:
                    score_local = run_local
                    score_global = run_global

                side_len = width if side in ("top_edge", "bottom_edge") else height
                coverage = self._run_span_along_side(run_local, side) / max(1.0, side_len)
                if coverage < min_run_frac:
                    continue

                line_err = self._mean_line_fit_error(score_local)
                norm_line_err = line_err / scale

                if side in ("top_edge", "bottom_edge"):
                    side_offset = float(np.mean(np.abs(score_local[:, 1] - side_targets[side])))
                    ang = self._line_angle_deg(score_local)
                    angle_err = min(abs(ang), abs(180.0 - ang))
                else:
                    side_offset = float(np.mean(np.abs(score_local[:, 0] - side_targets[side])))
                    ang = self._line_angle_deg(score_local)
                    angle_err = abs(90.0 - ang)

                norm_side_offset = side_offset / scale
                confidence = (
                    0.55 * min(1.0, coverage)
                    + 0.20 * max(0.0, 1.0 - (norm_line_err / max(max_line_err_frac, 1e-6)))
                    + 0.15 * max(0.0, 1.0 - (norm_side_offset / max(max_side_offset_frac, 1e-6)))
                    + 0.10 * max(0.0, 1.0 - (angle_err / max(angle_tol_deg, 1e-6)))
                )

                candidate = {
                    "coverage": float(coverage),
                    "mean_line_err": float(line_err),
                    "norm_line_err": float(norm_line_err),
                    "side_offset": float(side_offset),
                    "norm_side_offset": float(norm_side_offset),
                    "angle_error_deg": float(angle_err),
                    "confidence": float(confidence),
                    "run_local": score_local,
                    "run_global": score_global,
                }

                if best is None or candidate["confidence"] > best["confidence"]:
                    best = candidate

            if best is None:
                edge_data["scores"][side] = {
                    "coverage": 0.0,
                    "mean_line_err": None,
                    "norm_line_err": None,
                    "side_offset": None,
                    "norm_side_offset": None,
                    "angle_error_deg": None,
                    "confidence": 0.0,
                }
                continue

            edge_data[side] = best["run_global"]
            edge_data["scores"][side] = {
                "coverage": best["coverage"],
                "mean_line_err": best["mean_line_err"],
                "norm_line_err": best["norm_line_err"],
                "side_offset": best["side_offset"],
                "norm_side_offset": best["norm_side_offset"],
                "angle_error_deg": best["angle_error_deg"],
                "confidence": best["confidence"],
            }

            is_border = (
                best["coverage"] >= min_run_frac
                and best["norm_line_err"] <= max_line_err_frac
                and best["norm_side_offset"] <= max_side_offset_frac
                and best["angle_error_deg"] <= angle_tol_deg
                and best["confidence"] >= min_confidence
            )

            if is_border:
                edge_data["border_edges"].append(side)
            else:
                edge_data["tear_edges"].append(side)

        edge_data["piece_type"] = self._classify_piece_type(edge_data["border_edges"])
        return edge_data

    def classify_edges_oriented_runs_scaled(
        self,
        contour: np.ndarray,
        mask: np.ndarray,
        pixels_per_unit: float,
        target_pixels_per_unit: float = 100.0,
        crop_pad: int = 4,
        max_dimension: int = 8192,
        **kwargs,
    ) -> Dict:
        """
        Normalize the fragment mask to a canonical physical scale before edge scoring.

        This is useful when the same edge geometry appears at very different
        pixels-per-unit ratios across collections.
        """
        if pixels_per_unit is None or pixels_per_unit <= 0:
            return self.classify_edges_oriented_runs(contour=contour, mask=mask, **kwargs)

        scale_factor = float(target_pixels_per_unit) / float(pixels_per_unit)
        scaled_mask, effective_scale_factor = self._scale_fragment_mask(
            mask=mask,
            scale_factor=scale_factor,
            crop_pad=crop_pad,
            max_dimension=max_dimension,
        )
        scaled_contour = self._largest_mask_contour(scaled_mask, chain_approx=cv2.CHAIN_APPROX_SIMPLE)
        edge_data = self.classify_edges_oriented_runs(
            contour=scaled_contour,
            mask=scaled_mask,
            **kwargs,
        )
        edge_data["scale_normalization"] = {
            "pixels_per_unit": float(pixels_per_unit),
            "target_pixels_per_unit": float(target_pixels_per_unit),
            "requested_scale_factor": float(scale_factor),
            "effective_scale_factor": float(effective_scale_factor),
        }
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
        vx, vy, x0, y0 = cv2.fitLine(pts_f, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        ang = abs(math.degrees(math.atan2(float(vy), float(vx))))
        if ang > 90:
            ang = 180 - ang
        return float(ang)

    def _largest_mask_contour(
        self,
        mask: np.ndarray,
        chain_approx: int = cv2.CHAIN_APPROX_NONE,
    ) -> np.ndarray:
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            chain_approx,
        )
        if not contours:
            raise ValueError("No contour found")
        return max(contours, key=cv2.contourArea)

    def _scale_fragment_mask(
        self,
        mask: np.ndarray,
        scale_factor: float,
        crop_pad: int = 4,
        max_dimension: int = 8192,
    ) -> Tuple[np.ndarray, float]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            raise ValueError("Empty mask")

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        crop_pad = max(0, int(crop_pad))
        x0 = max(0, int(x0) - crop_pad)
        y0 = max(0, int(y0) - crop_pad)
        x1 = min(mask.shape[1] - 1, int(x1) + crop_pad)
        y1 = min(mask.shape[0] - 1, int(y1) + crop_pad)

        cropped = mask[y0:y1 + 1, x0:x1 + 1]
        if cropped.size == 0:
            raise ValueError("Empty cropped mask")

        scale_factor = max(0.05, float(scale_factor))
        out_h = max(3, int(round(cropped.shape[0] * scale_factor)))
        out_w = max(3, int(round(cropped.shape[1] * scale_factor)))

        if max_dimension > 0:
            longest = max(out_h, out_w)
            if longest > max_dimension:
                cap = max_dimension / float(longest)
                scale_factor *= cap
                out_h = max(3, int(round(cropped.shape[0] * scale_factor)))
                out_w = max(3, int(round(cropped.shape[1] * scale_factor)))

        scaled = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        scaled = ((scaled > 0).astype(np.uint8)) * 255
        return scaled, scale_factor

    def _smooth_closed_contour(
        self,
        pts: np.ndarray,
        window: int = 7,
        iterations: int = 1,
    ) -> np.ndarray:
        pts_f = pts.astype(np.float32)
        if len(pts_f) == 0:
            return pts_f

        window = max(1, int(window))
        if window % 2 == 0:
            window += 1
        if window == 1 or len(pts_f) < window:
            return pts_f

        smoothed = pts_f.copy()
        radius = window // 2
        for _ in range(max(1, int(iterations))):
            acc = np.zeros_like(smoothed)
            for shift in range(-radius, radius + 1):
                acc += np.roll(smoothed, shift=shift, axis=0)
            smoothed = acc / float((2 * radius) + 1)
        return smoothed

    def _project_to_min_area_rect_frame(
        self,
        pts: np.ndarray,
        rect: Tuple[Tuple[float, float], Tuple[float, float], float],
    ) -> Tuple[np.ndarray, float, float]:
        box = cv2.boxPoints(rect).astype(np.float32)
        edge0 = box[1] - box[0]
        edge1 = box[2] - box[1]
        len0 = float(np.linalg.norm(edge0))
        len1 = float(np.linalg.norm(edge1))

        if len0 < 1e-6 or len1 < 1e-6:
            raise ValueError("Degenerate oriented box")

        if abs(float(edge0[0])) >= abs(float(edge0[1])):
            x_axis = edge0 / len0
            y_axis = edge1 / len1
        else:
            x_axis = edge1 / len1
            y_axis = -edge0 / len0

        width = float(np.max(box @ x_axis) - np.min(box @ x_axis))
        height = float(np.max(box @ y_axis) - np.min(box @ y_axis))
        if width < 1e-6 or height < 1e-6:
            raise ValueError("Degenerate oriented box")

        pts_f = pts.astype(np.float32)
        x_proj = (pts_f[:, 0] * float(x_axis[0])) + (pts_f[:, 1] * float(x_axis[1]))
        y_proj = (pts_f[:, 0] * float(y_axis[0])) + (pts_f[:, 1] * float(y_axis[1]))
        box_x_proj = (box[:, 0] * float(x_axis[0])) + (box[:, 1] * float(x_axis[1]))
        box_y_proj = (box[:, 0] * float(y_axis[0])) + (box[:, 1] * float(y_axis[1]))
        local_x = x_proj - np.min(box_x_proj)
        local_y = y_proj - np.min(box_y_proj)
        return np.column_stack([local_x, local_y]), width, height

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

    def _extract_circular_run_points(self, pts: np.ndarray, start: int, end: int) -> np.ndarray:
        if start <= end:
            return pts[start:end + 1]
        return np.concatenate([pts[start:], pts[:end + 1]], axis=0)

    def _trim_run_points(self, pts: np.ndarray, trim_frac: float = 0.10) -> np.ndarray:
        if len(pts) == 0:
            return pts
        trim = int(len(pts) * max(0.0, trim_frac))
        if trim == 0 or len(pts) <= (2 * trim) + 5:
            return pts
        return pts[trim:-trim]

    def _classify_piece_type(self, border_edges: List[str]) -> str:
        borders = set(border_edges)
        if len(borders) == 0:
            return "interior"
        if len(borders) == 1:
            return "edge"

        adj = {
            frozenset(("top_edge", "left_edge")),
            frozenset(("top_edge", "right_edge")),
            frozenset(("bottom_edge", "left_edge")),
            frozenset(("bottom_edge", "right_edge")),
        }
        if any(frozenset(pair) <= borders for pair in adj):
            return "corner"
        return "edge"

    def _mean_line_fit_error(self, pts: np.ndarray) -> float:
        """
        Fit a line and compute mean orthogonal distance (pixels).
        More stable than endpoint straightness ratio when runs are long/curvy.
        """
        pts_f = pts.astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts_f, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
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
        self.oriented_near_frac = self.config['config'].get('oriented_near_frac', 0.04)
        self.oriented_min_run_frac = self.config['config'].get('oriented_min_run_frac', 0.40)
        self.oriented_max_line_err_frac = self.config['config'].get('oriented_max_line_err_frac', 0.02)
        self.oriented_max_side_offset_frac = self.config['config'].get('oriented_max_side_offset_frac', 0.025)
        self.oriented_angle_tol_deg = self.config['config'].get('oriented_angle_tol_deg', 18.0)
        self.oriented_min_confidence = self.config['config'].get('oriented_min_confidence', 0.60)
        self.oriented_smoothing_window = self.config['config'].get('oriented_smoothing_window', 7)
        self.oriented_trim_frac = self.config['config'].get('oriented_trim_frac', 0.10)
        self.oriented_scaled_target_pixels_per_unit = self.config['config'].get('oriented_scaled_target_pixels_per_unit', 100.0)
        self.oriented_scaled_crop_pad = self.config['config'].get('oriented_scaled_crop_pad', 4)
        self.oriented_scaled_max_dimension = self.config['config'].get('oriented_scaled_max_dimension', 8192)
        self.classifier_device_name = self.config['config'].get('classifier_device', 'auto')
        self.classifier_fallback_method = self.config['config'].get('classifier_fallback_method', 'oriented_runs')
        self.classifier_thresholds_override = self.config['config'].get('classifier_thresholds')
        self.classifier_img_size_override = self.config['config'].get('classifier_img_size')
        self.classifier_input_mode_override = self.config['config'].get('classifier_input_mode')
        self.classifier_target_pixels_per_unit_override = self.config['config'].get('classifier_target_pixels_per_unit')
        self.classifier_crop_pad_override = self.config['config'].get('classifier_crop_pad')
        self.classifier_max_scaled_longest_side_override = self.config['config'].get('classifier_max_scaled_longest_side')
        self.edge_classifier_model = None
        self.edge_classifier_device = None
        self.edge_classifier_thresholds: List[float] = [0.5, 0.5, 0.5, 0.5]
        self.edge_classifier_transform = None
        self.edge_classifier_input_mode = "masked_rgb"
        self.edge_classifier_target_pixels_per_unit = 100.0
        self.edge_classifier_crop_pad = 6
        self.edge_classifier_max_scaled_longest_side = 1024
        self.edge_classifier_meta: Dict[str, Any] = {}

        self.version = self.config['config'].get('model_version', '1.0')
        if self.edge_method in {"edge_classifier", "ml_classifier"}:
            self._setup_edge_classifier()

    def get_metadata(self) -> ProcessorMetadata:
        return ProcessorMetadata(
            name="edgedetection",
            version=self.version,
            description="Detects fragment border edges (top/bottom/left/right)",
            model_path=self.config.get('model_path'),
            requires_gpu=bool(
                self.edge_method in {"edge_classifier", "ml_classifier"}
                and self.edge_classifier_device is not None
                and self.edge_classifier_device.type == "cuda"
            ),
            batch_size=1,
        )

    def should_process(self, fragment: FragmentRecord) -> bool:
        # Only process if edges not set
        #return fragment.has_top_edge is None and fragment.has_bottom_edge is None
        return True

    def _resolve_torch_device(self, requested: str) -> torch.device:
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_classifier_meta_path(self, model_path: Path) -> Optional[Path]:
        explicit_meta = self.config.get("meta_path")
        if explicit_meta:
            candidate = Path(explicit_meta)
            if candidate.exists():
                return candidate

        sibling = model_path.with_name("metadata.json")
        if sibling.exists():
            return sibling

        parent = model_path.parent.parent / "metadata.json"
        if parent.exists():
            return parent

        return None

    def _normalize_classifier_thresholds(self, value: Optional[Sequence[float]]) -> List[float]:
        if value is None:
            return [0.5, 0.5, 0.5, 0.5]

        if isinstance(value, str):
            raw_values = [part.strip() for part in value.split(",") if part.strip()]
            thresholds = [float(part) for part in raw_values]
        else:
            thresholds = [float(item) for item in value]

        if len(thresholds) != 4:
            raise ValueError("classifier_thresholds must contain exactly 4 values")
        return thresholds

    def _setup_edge_classifier(self) -> None:
        model_path = self.config.get("model_path")
        if not model_path:
            raise ValueError("edgedetection model_path is required when edge_method=edge_classifier")

        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Edge classifier model not found: {model_path_obj}")

        meta_path = self._resolve_classifier_meta_path(model_path_obj)
        metadata: Dict[str, Any] = {}
        if meta_path is not None and meta_path.exists():
            metadata = json.loads(meta_path.read_text())

        model_config = metadata.get("config", {})
        thresholds = self.classifier_thresholds_override
        if thresholds is None:
            thresholds = metadata.get("best_thresholds")
        self.edge_classifier_thresholds = self._normalize_classifier_thresholds(thresholds)

        self.edge_classifier_input_mode = str(
            self.classifier_input_mode_override
            if self.classifier_input_mode_override is not None
            else model_config.get("input_mode", "masked_rgb")
        )
        self.edge_classifier_target_pixels_per_unit = float(
            self.classifier_target_pixels_per_unit_override
            if self.classifier_target_pixels_per_unit_override is not None
            else model_config.get("target_pixels_per_unit", 100.0)
        )
        self.edge_classifier_crop_pad = int(
            self.classifier_crop_pad_override
            if self.classifier_crop_pad_override is not None
            else model_config.get("crop_pad", 6)
        )
        self.edge_classifier_max_scaled_longest_side = int(
            self.classifier_max_scaled_longest_side_override
            if self.classifier_max_scaled_longest_side_override is not None
            else model_config.get("max_scaled_longest_side", 1024)
        )
        classifier_img_size = int(
            self.classifier_img_size_override
            if self.classifier_img_size_override is not None
            else model_config.get("img_size", 224)
        )
        aux_piece_head = bool(model_config.get("aux_piece_head", True))
        dropout_rate = float(model_config.get("dropout_rate", 0.3))

        self.edge_classifier_device = self._resolve_torch_device(self.classifier_device_name)
        self.edge_classifier_model = load_edge_classifier(
            model_path=str(model_path_obj),
            device=str(self.edge_classifier_device),
            pretrained=False,
            aux_piece_head=aux_piece_head,
            dropout_rate=dropout_rate,
            freeze_backbone_fraction=0.0,
        )
        self.edge_classifier_transform = build_edge_image_transform(
            img_size=classifier_img_size,
            augment=False,
        )
        self.edge_classifier_meta = metadata
        self.logger.info(
            "Loaded edge classifier model from %s on %s (thresholds=%s)",
            model_path_obj,
            self.edge_classifier_device,
            [round(value, 3) for value in self.edge_classifier_thresholds],
        )

    def _bbox_from_mask(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            raise ValueError("Empty mask")
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def _predict_edges_with_classifier(
        self,
        img: np.ndarray,
        fragment: FragmentRecord,
        contour: np.ndarray,
        mask: np.ndarray,
    ) -> Dict[str, Any]:
        if self.edge_classifier_model is None or self.edge_classifier_transform is None or self.edge_classifier_device is None:
            raise RuntimeError("Edge classifier is not initialized")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rendered = render_fragment_image(
            rgb=rgb,
            segmentation_coords=fragment.segmentation_coords,
            pixels_per_unit=getattr(fragment, "pixels_per_unit", None),
            crop_pad=self.edge_classifier_crop_pad,
            target_pixels_per_unit=self.edge_classifier_target_pixels_per_unit,
            max_scaled_longest_side=self.edge_classifier_max_scaled_longest_side,
            input_mode=self.edge_classifier_input_mode,
        )
        image_tensor = self.edge_classifier_transform(rendered).unsqueeze(0).to(self.edge_classifier_device)

        with torch.no_grad():
            outputs = self.edge_classifier_model(image_tensor)
            edge_probs = torch.sigmoid(outputs["edge_logits"]).cpu().numpy()[0]
            piece_logits = outputs.get("piece_logits")
            piece_probs = None
            if piece_logits is not None:
                piece_probs = torch.softmax(piece_logits, dim=1).cpu().numpy()[0]

        side_names = ["top_edge", "bottom_edge", "left_edge", "right_edge"]
        border_edges = [
            side_name
            for side_name, probability, threshold in zip(side_names, edge_probs, self.edge_classifier_thresholds)
            if float(probability) >= float(threshold)
        ]
        piece_type = self.extractor._classify_piece_type(border_edges)
        piece_prediction_aux = None
        if piece_probs is not None:
            piece_classes = ["interior", "edge", "corner"]
            piece_prediction_aux = piece_classes[int(np.argmax(piece_probs))]

        x0, y0, x1, y1 = self._bbox_from_mask(mask)
        edge_data: Dict[str, Any] = {
            "contour": contour,
            "bbox": (x0, y0, x1, y1),
            "top_edge": None,
            "bottom_edge": None,
            "left_edge": None,
            "right_edge": None,
            "tear_edges": [side for side in side_names if side not in border_edges],
            "border_edges": border_edges,
            "piece_type": piece_type,
            "scores": {
                side_name: {
                    "probability": float(probability),
                    "threshold": float(threshold),
                    "prediction": float(probability) >= float(threshold),
                }
                for side_name, probability, threshold in zip(side_names, edge_probs, self.edge_classifier_thresholds)
            },
            "classifier": {
                "edge_probabilities": {
                    side_name: float(probability)
                    for side_name, probability in zip(side_names, edge_probs)
                },
                "thresholds": {
                    side_name: float(threshold)
                    for side_name, threshold in zip(side_names, self.edge_classifier_thresholds)
                },
                "input_mode": self.edge_classifier_input_mode,
                "target_pixels_per_unit": self.edge_classifier_target_pixels_per_unit,
                "crop_pad": self.edge_classifier_crop_pad,
                "max_scaled_longest_side": self.edge_classifier_max_scaled_longest_side,
            },
        }
        if piece_probs is not None:
            piece_classes = ["interior", "edge", "corner"]
            edge_data["classifier"]["piece_probabilities"] = {
                piece_name: float(probability)
                for piece_name, probability in zip(piece_classes, piece_probs)
            }
            edge_data["classifier"]["piece_prediction_aux"] = piece_prediction_aux
        return edge_data

    def _run_heuristic_method(
        self,
        method: str,
        contour: np.ndarray,
        mask: np.ndarray,
        fragment: FragmentRecord,
    ) -> Dict[str, Any]:
        if method == "hull_segments":
            return self.extractor.classify_edges_hull_segments(
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
        if method == "oriented_runs":
            return self.extractor.classify_edges_oriented_runs(
                contour=contour,
                mask=mask,
                near_frac=self.oriented_near_frac,
                min_run_frac=self.oriented_min_run_frac,
                max_line_err_frac=self.oriented_max_line_err_frac,
                max_side_offset_frac=self.oriented_max_side_offset_frac,
                angle_tol_deg=self.oriented_angle_tol_deg,
                min_confidence=self.oriented_min_confidence,
                smoothing_window=self.oriented_smoothing_window,
                trim_frac=self.oriented_trim_frac,
            )
        if method == "oriented_runs_scaled":
            return self.extractor.classify_edges_oriented_runs_scaled(
                contour=contour,
                mask=mask,
                pixels_per_unit=getattr(fragment, "pixels_per_unit", None),
                target_pixels_per_unit=self.oriented_scaled_target_pixels_per_unit,
                crop_pad=self.oriented_scaled_crop_pad,
                max_dimension=self.oriented_scaled_max_dimension,
                near_frac=self.oriented_near_frac,
                min_run_frac=self.oriented_min_run_frac,
                max_line_err_frac=self.oriented_max_line_err_frac,
                max_side_offset_frac=self.oriented_max_side_offset_frac,
                angle_tol_deg=self.oriented_angle_tol_deg,
                min_confidence=self.oriented_min_confidence,
                smoothing_window=self.oriented_smoothing_window,
                trim_frac=self.oriented_trim_frac,
            )
        if method == "bbox_runs":
            return self.extractor.classify_edges(
                contour=contour,
                mask=mask,
                near_frac=self.near_frac,
                min_run_frac=self.min_run_frac,
                max_mean_line_err=self.max_mean_line_err,
            )
        raise ValueError(f"Unsupported edge method: {method}")

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
        try:
            if self.edge_method in {"edge_classifier", "ml_classifier"}:
                try:
                    edge_data = self._predict_edges_with_classifier(
                        img=img,
                        fragment=fragment,
                        contour=contour,
                        mask=mask,
                    )
                except Exception as exc:
                    fallback_method = self.classifier_fallback_method
                    if fallback_method in {None, "", "none"}:
                        raise
                    self.logger.warning(
                        "Edge classifier failed for %s: %s. Falling back to %s.",
                        fragment.fragment_id,
                        exc,
                        fallback_method,
                    )
                    edge_data = self._run_heuristic_method(
                        method=fallback_method,
                        contour=contour,
                        mask=mask,
                        fragment=fragment,
                    )
                    edge_data.setdefault("classifier", {})
                    edge_data["classifier"]["fallback_method"] = fallback_method
                    edge_data["classifier"]["fallback_error"] = str(exc)
            else:
                edge_data = self._run_heuristic_method(
                    method=self.edge_method,
                    contour=contour,
                    mask=mask,
                    fragment=fragment,
                )
        except Exception as exc:
            return ProcessingResult(
                success=False,
                updates={"processing_error": str(exc)},
                error=str(exc),
            )

        self.logger.debug("Edge scores for %s: %s", fragment.fragment_id, edge_data.get("scores"))
        self.logger.debug("Border edges for %s: %s", fragment.fragment_id, edge_data.get("border_edges"))

        # Set edge flags as booleans
        has_top = "top_edge" in edge_data["border_edges"]
        has_bottom = "bottom_edge" in edge_data["border_edges"]
        has_left = "left_edge" in edge_data["border_edges"]
        has_right = "right_edge" in edge_data["border_edges"]

        is_edge_piece = has_top or has_bottom or has_left or has_right

        return ProcessingResult(
            success=True,
            updates={
                "has_top_edge": has_top,
                "has_bottom_edge": has_bottom,
                "has_left_edge": has_left,
                "has_right_edge": has_right,
                "edge_piece": is_edge_piece,
                "processing_status": "completed",
                "last_processed_at": "CURRENT_TIMESTAMP",
                "processing_error": None,
            },
            metadata={
                "piece_type": edge_data["piece_type"],
                "borders": edge_data["border_edges"],
                "tears": edge_data["tear_edges"],
                "scores": edge_data.get("scores"),
                "classifier": edge_data.get("classifier"),
            }
        )

    def _largest_contour(self, mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contour found")
        return max(contours, key=cv2.contourArea)

    def cleanup(self) -> None:
        self.edge_classifier_model = None
        self.edge_classifier_transform = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
