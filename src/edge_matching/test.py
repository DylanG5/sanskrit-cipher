import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class EdgeSegment:
    piece_name: str
    edge_name: str  # "top", "bottom", "left", "right"
    points: np.ndarray  # (N, 2) float32, in image coords
    length: float       # polyline length in pixels
    is_tear: bool       # True if jagged (tear), False if almost straight
    lab_patches: Optional[np.ndarray] = None  # (P, 3) mean Lab colours along edge


# ----------------------------
# Geometry helpers
# ----------------------------

def resample_polyline(points: np.ndarray, n_samples: int = 200) -> np.ndarray:
    """
    Resample a 2D polyline (N,2) to exactly n_samples points with uniform arc-length spacing.
    """
    if len(points) < 2:
        return np.repeat(points.astype(np.float32), n_samples, axis=0)

    # cumulative arc length
    diffs = np.diff(points, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cumlen = np.cumsum(seg_lengths)
    total_len = float(cumlen[-1])
    if total_len == 0.0:
        return np.repeat(points[:1].astype(np.float32), n_samples, axis=0)

    target = np.linspace(0, total_len, n_samples, dtype=np.float32)

    # insert 0 at start for convenience
    cumlen = np.concatenate(([0.0], cumlen))

    resampled = []
    j = 0
    for t in target:
        # find segment where this arc-length lies
        while j < len(seg_lengths) and cumlen[j + 1] < t:
            j += 1
        if j >= len(seg_lengths):
            resampled.append(points[-1])
            continue
        t0, t1 = cumlen[j], cumlen[j + 1]
        alpha = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        p = (1 - alpha) * points[j] + alpha * points[j + 1]
        resampled.append(p)
    return np.asarray(resampled, dtype=np.float32)


def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.sqrt((diffs ** 2).sum(axis=1)).sum())


def classify_tear(points: np.ndarray, straightness_threshold: float = 0.15) -> bool:
    """
    Simple straight vs tear classification based on path_length / straight_line_distance.
    > 1 + threshold -> tear (jagged).
    """
    if len(points) < 2:
        return False
    path_len = polyline_length(points)
    start, end = points[0], points[-1]
    straight = float(np.linalg.norm(end - start))
    if straight == 0.0:
        return True
    ratio = path_len / straight
    return ratio > (1.0 + straightness_threshold)


def assign_edges_by_nearest_side(contour: np.ndarray,
                                 width: int,
                                 height: int) -> Dict[str, List[np.ndarray]]:
    """
    Assign every contour point to the nearest image border (top/bottom/left/right),
    so all segments are classified as one of the four edges.
    Returns dictionary edge_name -> list of (x,y) points in order along contour.
    """
    contour = contour.reshape(-1, 2).astype(np.float32)
    edges = {"top": [], "bottom": [], "left": [], "right": []}

    for (x, y) in contour:
        dist_top = y
        dist_bottom = height - 1 - y
        dist_left = x
        dist_right = width - 1 - x
        dists = np.array([dist_top, dist_bottom, dist_left, dist_right], dtype=np.float32)
        side = int(np.argmin(dists))
        if side == 0:
            edges["top"].append([x, y])
        elif side == 1:
            edges["bottom"].append([x, y])
        elif side == 2:
            edges["left"].append([x, y])
        else:
            edges["right"].append([x, y])

    # convert to numpy arrays (may be empty)
    for k in edges:
        if len(edges[k]) > 0:
            edges[k] = np.asarray(edges[k], dtype=np.float32)
        else:
            edges[k] = np.zeros((0, 2), dtype=np.float32)
    return edges


# ----------------------------
# Colour helpers
# ----------------------------

def compute_lab_patches(img_bgr: np.ndarray,
                        mask: np.ndarray,
                        edge_points: np.ndarray,
                        num_patches: int = 5,
                        patch_radius: int = 3) -> np.ndarray:
    """
    Sample small patches inside the mask along the edge and compute mean Lab colour for each.
    Returns (P, 3) array.
    """
    if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        # grayscale-only case: just return zeros
        return np.zeros((num_patches, 3), dtype=np.float32)

    h, w = mask.shape
    # convert entire image once to Lab
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    if len(edge_points) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # pick indices along the edge
    idxs = np.linspace(0, len(edge_points) - 1, num_patches, dtype=int)
    patch_cols = []

    for idx in idxs:
        x, y = edge_points[idx]
        x = int(round(x))
        y = int(round(y))

        # define patch, clipped to image
        x0 = max(0, x - patch_radius)
        x1 = min(w, x + patch_radius + 1)
        y0 = max(0, y - patch_radius)
        y1 = min(h, y + patch_radius + 1)

        patch_mask = mask[y0:y1, x0:x1]
        if patch_mask.size == 0:
            patch_cols.append([0.0, 0.0, 0.0])
            continue

        # keep only pixels inside the piece
        inside = patch_mask > 0
        if not np.any(inside):
            patch_cols.append([0.0, 0.0, 0.0])
            continue

        patch_lab = img_lab[y0:y1, x0:x1][inside]
        mean_lab = patch_lab.mean(axis=0)
        patch_cols.append(mean_lab)

    return np.asarray(patch_cols, dtype=np.float32)


# ----------------------------
# Edge extraction from an image
# ----------------------------

def extract_piece_edges(img_path: str) -> List[EdgeSegment]:
    """
    Load an image, segment the main piece, find contour, split into four edges,
    classify tear vs border, and compute Lab patches for each edge.
    """
    piece_name = os.path.basename(img_path)
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {img_path}")

    h, w = img_bgr.shape[:2]

    # simple binary mask via Otsu threshold on grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # decide whether we need to invert (piece should be foreground = 255)
    if np.count_nonzero(mask) < (mask.size // 2):
        mask = cv2.bitwise_not(mask)

    # largest contour as the piece boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea)  # (N,1,2)

    edge_points_dict = assign_edges_by_nearest_side(contour, w, h)

    segments: List[EdgeSegment] = []
    for edge_name, pts in edge_points_dict.items():
        if pts.shape[0] < 2:
            continue
        length = polyline_length(pts)
        is_tear = classify_tear(pts)
        lab_patches = compute_lab_patches(img_bgr, mask, pts)
        segments.append(
            EdgeSegment(
                piece_name=piece_name,
                edge_name=edge_name,
                points=pts,
                length=length,
                is_tear=is_tear,
                lab_patches=lab_patches,
            )
        )
    return segments


# ----------------------------
# Matching / scoring
# ----------------------------

def normalize_descriptor(points: np.ndarray,
                         n_samples: int = 200) -> np.ndarray:
    """
    Convert a polyline into a normalized descriptor:
    - resampled to n_samples
    - centered
    - isotropically scaled to unit RMS distance
    """
    pts = resample_polyline(points, n_samples)
    pts = pts.astype(np.float32)
    pts -= pts.mean(axis=0, keepdims=True)
    rms = float(np.sqrt((pts ** 2).sum(axis=1).mean()))
    if rms > 0:
        pts /= rms
    return pts


def procrustes_align(base: np.ndarray, cand: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orthogonal Procrustes alignment: find rotation that best aligns cand to base
    (both already centered & similarly scaled).
    """
    # both (N,2) and mean~0
    H = cand.T @ base
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    cand_aligned = cand @ R
    return base, cand_aligned


def edge_shape_distance(desc_a: np.ndarray, desc_b: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute best shape distance between two descriptors, considering
    reversing and complement (negation), with Procrustes alignment.
    Returns (rmse, base_points, cand_points_aligned).
    """
    variants = []

    # B as-is and reversed
    variants.append(desc_b)
    variants.append(desc_b[::-1, :])

    # complement (negate) variants
    variants.append(-desc_b)
    variants.append((-desc_b)[::-1, :])

    best_rmse = float("inf")
    best_cand = None

    base = desc_a - desc_a.mean(axis=0, keepdims=True)

    for v in variants:
        v = v - v.mean(axis=0, keepdims=True)
        base_aligned, cand_aligned = procrustes_align(base, v)
        rmse = float(np.sqrt(((base_aligned - cand_aligned) ** 2).mean()))
        if rmse < best_rmse:
            best_rmse = rmse
            best_cand = cand_aligned

    return best_rmse, base, best_cand


def lab_delta_e(patches_a: np.ndarray, patches_b: np.ndarray) -> float:
    """
    Basic ΔE distance between two sets of Lab patches (mean of pairwise).
    Uses simple L2 in Lab space.
    """
    if patches_a is None or patches_b is None:
        return 0.0
    if len(patches_a) == 0 or len(patches_b) == 0:
        return 0.0

    # resample to same number of patches
    P = min(len(patches_a), len(patches_b))
    idx_a = np.linspace(0, len(patches_a) - 1, P, dtype=int)
    idx_b = np.linspace(0, len(patches_b) - 1, P, dtype=int)
    A = patches_a[idx_a].astype(np.float32)
    B = patches_b[idx_b].astype(np.float32)

    diffs = A - B
    dE = np.sqrt((diffs ** 2).sum(axis=1))
    return float(dE.mean())


def match_score(edge_a: EdgeSegment,
                edge_b: EdgeSegment,
                desc_a: np.ndarray,
                desc_b: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute overall match score between two tear edges.
    Lower is better.
    """
    # length penalty
    len_a, len_b = edge_a.length, edge_b.length
    length_penalty = abs(len_a - len_b) / (max(len_a, len_b) + 1e-6)

    # shape term
    shape_rmse, base_aligned, cand_aligned = edge_shape_distance(desc_a, desc_b)

    # colour term (Lab)
    dE = lab_delta_e(edge_a.lab_patches, edge_b.lab_patches)

    # weights (tune as needed)
    w_shape = 1.0
    w_len = 1.5
    w_color = 0.1

    score = w_shape * shape_rmse + w_len * length_penalty + w_color * (dE / 50.0)
    return score, base_aligned, cand_aligned


# ----------------------------
# Main matching routine
# ----------------------------

def collect_edges(input_dir: str) -> List[EdgeSegment]:
    all_segments: List[EdgeSegment] = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        path = os.path.join(input_dir, fname)
        segs = extract_piece_edges(path)
        all_segments.extend(segs)
    return all_segments


def run_match(input_dir: str,
              fragment_name: str,
              top_k: int,
              viz_dir: Optional[str],
              only_tears: bool = True) -> None:
    segments = collect_edges(input_dir)

    # filter by tear vs border
    if only_tears:
        segments = [s for s in segments if s.is_tear]

    if not segments:
        print("No edges found.")
        return

    target_segments = [s for s in segments if s.piece_name == fragment_name]
    if not target_segments:
        print(f"No segments found for fragment '{fragment_name}'. "
              f"Available pieces: {sorted({s.piece_name for s in segments})}")
        return

    other_segments = [s for s in segments if s.piece_name != fragment_name]

    # precompute descriptors
    desc_cache: Dict[Tuple[str, str], np.ndarray] = {}
    for s in segments:
        key = (s.piece_name, s.edge_name)
        desc_cache[key] = normalize_descriptor(s.points)

    if viz_dir:
        os.makedirs(viz_dir, exist_ok=True)

    for t_seg in target_segments:
        print(f"\n=== Matches for {t_seg.piece_name} [{t_seg.edge_name}] ===")
        t_desc = desc_cache[(t_seg.piece_name, t_seg.edge_name)]

        scores = []
        for c_seg in other_segments:
            c_desc = desc_cache[(c_seg.piece_name, c_seg.edge_name)]
            score, base_aligned, cand_aligned = match_score(
                t_seg, c_seg, t_desc, c_desc
            )
            scores.append((score, c_seg, base_aligned, cand_aligned))

        scores.sort(key=lambda x: x[0])
        best = scores[:top_k]

        for rank, (score, c_seg, base_aligned, cand_aligned) in enumerate(best, start=1):
            print(f"{rank}. {c_seg.piece_name} [{c_seg.edge_name}]  "
                  f"score={score:.4f}")

            if viz_dir:
                plt.figure(figsize=(4, 4))
                plt.plot(base_aligned[:, 0], base_aligned[:, 1], label="Target edge")
                plt.plot(cand_aligned[:, 0], cand_aligned[:, 1], label="Candidate edge")
                plt.gca().invert_yaxis()
                plt.axis("equal")
                plt.legend()
                title = (f"{t_seg.piece_name} ({t_seg.edge_name}) "
                         f"vs {c_seg.piece_name} ({c_seg.edge_name})\n"
                         f"score={score:.4f}")
                plt.title(title)
                out_name = f"{t_seg.piece_name}_{t_seg.edge_name}__" \
                           f"{c_seg.piece_name}_{c_seg.edge_name}__{rank}.png"
                out_path = os.path.join(viz_dir, out_name)
                plt.tight_layout()
                plt.savefig(out_path, dpi=200)
                plt.close()


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Edge-based fragment matcher (shape + colour)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    match_p = subparsers.add_parser("match", help="Find matches for a fragment edge.")
    match_p.add_argument("--input_dir", type=str, required=True,
                         help="Directory containing fragment images.")
    match_p.add_argument("--fragment_name", type=str, required=True,
                         help="File name of the target fragment (e.g. '0_0.png').")
    match_p.add_argument("--top_k", type=int, default=5,
                         help="How many best matches to show.")
    match_p.add_argument("--viz_dir", type=str, default=None,
                         help="Directory to save overlay plots (optional).")
    match_p.add_argument("--include_borders", action="store_true",
                         help="Include straight (border) edges as candidates as well.")

    args = parser.parse_args()

    if args.command == "match":
        run_match(
            input_dir=args.input_dir,
            fragment_name=args.fragment_name,
            top_k=args.top_k,
            viz_dir=args.viz_dir,
            only_tears=not args.include_borders,
        )


if __name__ == "__main__":
    main()
