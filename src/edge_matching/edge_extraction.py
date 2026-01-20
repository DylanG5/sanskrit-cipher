"""
Edge Extraction for Manuscript Fragment Matching
Stage 1: Extract contours from fragment masks, classify edges,
and suggest likely complementary tear edges.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


class EdgeExtractor:
    def __init__(self, smoothing_iterations: int = 2, epsilon_factor: float = 0.001):
        """
        Initialize edge extractor with parameters.

        Args:
            smoothing_iterations: Number of times to smooth the contour
            epsilon_factor: Factor for contour approximation (lower = more detail)
        """
        self.smoothing_iterations = smoothing_iterations
        self.epsilon_factor = epsilon_factor

    def load_fragment(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load fragment image and extract binary mask.

        Args:
            image_path: Path to fragment image (PNG with alpha or JPEG)

        Returns:
            img: Image array
            mask: Binary mask from alpha channel or grayscale threshold
        """
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
            _, mask = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )

        return img, mask

    def extract_contour(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract the outer contour from binary mask.

        Args:
            mask: Binary mask

        Returns:
            contour: Largest contour as numpy array
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            raise ValueError("No contours found in mask")

        # Get the largest contour (should be the fragment boundary)
        contour = max(contours, key=cv2.contourArea)

        # Apply smoothing to reduce noise
        for _ in range(self.smoothing_iterations):
            contour = self._smooth_contour(contour)

        return contour

    def _smooth_contour(self, contour: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Smooth contour using moving average filter.

        Args:
            contour: Input contour
            kernel_size: Size of smoothing kernel

        Returns:
            Smoothed contour
        """
        contour = contour.reshape(-1, 2)
        smoothed = np.zeros_like(contour, dtype=np.float32)

        for i in range(len(contour)):
            # Circular indexing for moving average
            indices = [
                (i + j - kernel_size // 2) % len(contour)
                for j in range(kernel_size)
            ]
            smoothed[i] = np.mean(contour[indices], axis=0)

        return smoothed.reshape(-1, 1, 2).astype(np.int32)

    def classify_edges(self, contour: np.ndarray, mask_shape: Tuple[int, int]) -> Dict:
        """
        Classify contour segments as tear edges or original borders.
        Uses straightness and complexity metrics.

        Args:
            contour: Fragment contour
            mask_shape: Shape of the mask (height, width)

        Returns:
            Dictionary with edge classification results
        """
        height, width = mask_shape
        contour_points = contour.reshape(-1, 2)

        # Calculate properties for edge classification
        edge_data = {
            "contour": contour,
            "top_edge": [],
            "bottom_edge": [],
            "left_edge": [],
            "right_edge": [],
            "tear_edges": [],
            "border_edges": [],
        }

        # Segment contour into regions
        # Top quarter: y < height/4
        # Bottom quarter: y > 3*height/4
        # Left quarter: x < width/4
        # Right quarter: x > 3*width/4

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
            if len(edge_points) > 10:  # Need sufficient points
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
        """
        Determine if an edge is a tear (irregular) or border (straight).

        Args:
            edge_points: List of (index, point) tuples
            straightness_threshold: Threshold for straightness ratio

        Returns:
            True if edge is a tear, False if border
        """
        if len(edge_points) < 3:
            return False

        points = np.array([p[1] for p in edge_points])

        # Calculate actual path length
        path_length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))

        # Calculate straight-line distance between endpoints
        straight_distance = np.linalg.norm(points[-1] - points[0])

        if straight_distance == 0:
            return True  # Degenerate case, treat as tear

        # Straightness ratio: 1.0 = perfectly straight, >1.0 = curved/irregular
        straightness_ratio = path_length / straight_distance

        # If ratio is significantly > 1, it's irregular (tear edge)
        return straightness_ratio > (1.0 + straightness_threshold)

    def visualize_edges(
        self, img: np.ndarray, edge_data: Dict, output_path: str = None
    ) -> np.ndarray:
        """
        Visualize contour with color-coded edges.

        Args:
            img: Original RGBA image
            edge_data: Edge classification data
            output_path: Optional path to save visualization

        Returns:
            Visualization image
        """
        # Create visualization on white background
        vis = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255

        # Paste fragment onto white background
        if img.shape[2] == 4:
            alpha = img[:, :, 3:4] / 255.0
            rgb = img[:, :, :3]
            vis = (vis * (1 - alpha) + rgb * alpha).astype(np.uint8)

        contour = edge_data["contour"]
        contour_points = contour.reshape(-1, 2)

        # Draw full contour in gray first
        cv2.drawContours(vis, [contour], -1, (128, 128, 128), 2)

        # Color code different edges
        edge_colors = {
            "top_edge": (255, 0, 0),  # Blue
            "bottom_edge": (0, 255, 0),  # Green
            "left_edge": (0, 0, 255),  # Red
            "right_edge": (255, 0, 255),  # Magenta
        }

        # Draw each edge segment in its color
        for edge_name, color in edge_colors.items():
            edge_points = edge_data[edge_name]
            if edge_points:
                # Only draw lines between strictly adjacent contour points
                for i in range(len(edge_points) - 1):
                    idx1, pt1 = edge_points[i]
                    idx2, pt2 = edge_points[i + 1]
                    # Only draw if indices are exactly adjacent (diff of 1)
                    if idx2 - idx1 == 1:
                        cv2.line(vis, tuple(pt1), tuple(pt2), color, 3)

        # Highlight tear edges with thicker lines
        for edge_name in edge_data["tear_edges"]:
            edge_points = edge_data[edge_name]
            if edge_points:
                # Only draw lines between strictly adjacent contour points
                for i in range(len(edge_points) - 1):
                    idx1, pt1 = edge_points[i]
                    idx2, pt2 = edge_points[i + 1]
                    # Only draw if indices are exactly adjacent (diff of 1)
                    if idx2 - idx1 == 1:
                        cv2.line(
                            vis,
                            tuple(pt1),
                            tuple(pt2),
                            edge_colors[edge_name],
                            5,
                        )

        # Add legend
        legend_y = 30
        cv2.putText(
            vis,
            "Edge Classification:",
            (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
        legend_y += 25
        cv2.putText(
            vis,
            f"Tear edges: {', '.join(edge_data['tear_edges'])}",
            (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        legend_y += 20
        cv2.putText(
            vis,
            f"Border edges: {', '.join(edge_data['border_edges'])}",
            (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

        if output_path:
            cv2.imwrite(output_path, vis)

        return vis


def extract_edge_points(
    edge_points: List[Tuple[int, np.ndarray]], min_points: int = 5
) -> Optional[np.ndarray]:
    """
    Convert contour edge tuples into an ordered point array.
    """
    if not edge_points or len(edge_points) < min_points:
        return None

    sorted_points = sorted(edge_points, key=lambda x: x[0])
    points = np.array([pt for _, pt in sorted_points], dtype=np.float32)
    return points


def resample_points(points: np.ndarray, num_samples: int = 80) -> Optional[np.ndarray]:
    """
    Resample a polyline to a fixed number of points using linear interpolation.
    """
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


def compute_edge_descriptor(points: np.ndarray, num_samples: int = 80) -> Optional[Dict]:
    """
    Generate a normalized descriptor for a tear edge polyline.
    """
    resampled = resample_points(points, num_samples)
    if resampled is None:
        return None

    centered = resampled - resampled.mean(axis=0, keepdims=True)
    magnitudes = np.linalg.norm(centered, axis=1)
    scale = np.max(magnitudes)
    if scale == 0:
        scale = 1.0

    normalized = centered / scale
    length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

    return {
        "normalized_points": normalized,
        "length": length,
    }


def aligned_variants(points: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Generate orientation variants for edge comparison.

    Returns:
        List of (label, points) tuples.
    """
    variants: List[Tuple[str, np.ndarray]] = []

    variants.append(("original", points))

    flipped_x = points.copy()
    flipped_x[:, 0] *= -1
    variants.append(("flip_x", flipped_x))

    reversed_points = points[::-1]
    variants.append(("reversed", reversed_points))

    reversed_flipped = reversed_points.copy()
    reversed_flipped[:, 0] *= -1
    variants.append(("reversed_flip_x", reversed_flipped))

    return variants


def compute_match_details(
    desc_a: Dict, desc_b: Dict
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compare descriptors and return a score along with aligned point sets.
    """
    base = desc_a["normalized_points"]
    candidate_variants = aligned_variants(desc_b["normalized_points"])

    best_variant = None
    best_shape_score = float("inf")

    for _, variant in candidate_variants:
        # For complementary jigsaw fits we want one edge to be the inverse of the other.
        complement_variant = -variant
        mse = float(np.sqrt(np.mean((base - complement_variant) ** 2)))
        if mse < best_shape_score:
            best_shape_score = mse
            best_variant = complement_variant

    if best_variant is None:
        # Fallback to the first complementary variant if no match is found
        best_variant = -candidate_variants[0][1]

    length_a = desc_a["length"]
    length_b = desc_b["length"]
    max_length = max(length_a, length_b, 1e-6)
    length_penalty = abs(length_a - length_b) / max_length

    score = best_shape_score + 0.5 * length_penalty

    # Return copies so downstream plotting cannot mutate the stored descriptors
    return score, base.copy(), best_variant.copy()


@dataclass
class TearEdgeRecord:
    fragment: str
    edge_name: str
    descriptor: Dict


def analyze_fragment(
    fragment_path: Path,
    extractor: EdgeExtractor,
    output_file: Optional[Path] = None,
) -> Dict:
    """
    Run the full edge extraction pipeline for a single fragment.
    """
    img, mask = extractor.load_fragment(str(fragment_path))
    contour = extractor.extract_contour(mask)
    edge_data = extractor.classify_edges(contour, mask.shape)

    if output_file is not None:
        extractor.visualize_edges(img, edge_data, str(output_file))

    return {
        "filename": fragment_path.name,
        "contour": contour,
        "edge_data": edge_data,
    }


def process_fragments(input_dir: str, output_dir: str):
    """
    Process all fragments in directory and extract/visualize edges.

    Args:
        input_dir: Directory containing fragment images (PNG/JPEG)
        output_dir: Directory to save edge visualizations
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    extractor = EdgeExtractor(smoothing_iterations=2, epsilon_factor=0.001)

    # Find all supported fragment images
    fragment_files: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        fragment_files.extend(input_path.glob(pattern))
    fragment_files = sorted(fragment_files)

    if not fragment_files:
        print(f"No fragment files found in {input_dir}")
        return

    print(f"Processing {len(fragment_files)} fragments...")
    print("-" * 60)

    results = []

    for fragment_path in fragment_files:
        print(f"\nProcessing: {fragment_path.name}")

        try:
            output_file = output_path / f"edges_{fragment_path.stem}.jpg"
            result = analyze_fragment(fragment_path, extractor, output_file)
            edge_data = result["edge_data"]

            # Print edge classification
            print(f"  Tear edges: {edge_data['tear_edges']}")
            print(f"  Border edges: {edge_data['border_edges']}")
            print(f"  Saved: {output_file}")

            results.append(result)

        except Exception as e:
            print(f"  Error processing {fragment_path.name}: {e}")

    print("\n" + "=" * 60)
    print(f"Completed! Processed {len(results)}/{len(fragment_files)} fragments")
    print(f"Visualizations saved to: {output_path}")

    return results


def build_edge_records(result: Dict, num_samples: int) -> List[TearEdgeRecord]:
    """
    Convert tear edges from a fragment into descriptor records.
    """
    records: List[TearEdgeRecord] = []
    edge_data = result["edge_data"]

    for edge_name in edge_data["tear_edges"]:
        raw_points = extract_edge_points(edge_data.get(edge_name, []))
        if raw_points is None:
            continue

        descriptor = compute_edge_descriptor(raw_points, num_samples=num_samples)
        if descriptor is None:
            continue

        records.append(
            TearEdgeRecord(
                fragment=result["filename"],
                edge_name=edge_name,
                descriptor=descriptor,
            )
        )

    return records


def save_match_visualization(
    target_points: np.ndarray,
    candidate_points: np.ndarray,
    output_path: Path,
    title: str,
):
    """
    Save an overlay plot for a pair of matched edges.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.plot(
        target_points[:, 0],
        target_points[:, 1],
        label="Target edge",
        linewidth=2,
    )
    plt.plot(
        candidate_points[:, 0],
        candidate_points[:, 1],
        label="Candidate edge",
        linewidth=2,
    )
    plt.legend(loc="best")
    plt.title(title)
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.2, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def match_fragment_edges(
    input_dir: str,
    fragment_name: str,
    top_k: int = 3,
    num_samples: int = 80,
    viz_dir: Optional[str] = None,
):
    """
    Compare tear edges from a target fragment to all others
    and print the best match candidates for each edge.
    """
    input_path = Path(input_dir)
    extractor = EdgeExtractor(smoothing_iterations=2, epsilon_factor=0.001)

    fragment_files: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        fragment_files.extend(input_path.glob(pattern))
    fragment_files = sorted(fragment_files)

    if not fragment_files:
        print(f"No fragment files found in {input_dir}")
        return

    analyses: List[Dict] = []
    for fragment_path in fragment_files:
        try:
            analyses.append(analyze_fragment(fragment_path, extractor))
        except Exception as exc:
            print(f"  Error processing {fragment_path.name}: {exc}")

    if not analyses:
        print("No fragments could be analyzed for matching.")
        return

    fragment_name_lower = fragment_name.lower()
    target_results = [
        res
        for res in analyses
        if res["filename"].lower() == fragment_name_lower
        or Path(res["filename"]).stem.lower()
        == Path(fragment_name).stem.lower()
    ]

    if not target_results:
        print(f"Target fragment '{fragment_name}' not found in {input_dir}.")
        return

    target_records: List[TearEdgeRecord] = []
    candidate_records: List[TearEdgeRecord] = []

    for res in analyses:
        records = build_edge_records(res, num_samples=num_samples)
        if not records:
            continue

        if res in target_results:
            target_records.extend(records)
        else:
            candidate_records.extend(records)

    if viz_dir == "":
        viz_path = None
    else:
        viz_path = Path(viz_dir) if viz_dir else None
        if viz_path:
            viz_path.mkdir(parents=True, exist_ok=True)

    if not target_records:
        print(f"No tear edges available for fragment '{fragment_name}'.")
        return

    if not candidate_records:
        print("No candidate tear edges found in the dataset.")
        return

    print(f"Matching tear edges for fragment '{fragment_name}'...")
    print("-" * 60)

    for target_record in target_records:
        scored_candidates = []
        for candidate in candidate_records:
            score, base_points, candidate_points = compute_match_details(
                target_record.descriptor,
                candidate.descriptor,
            )
            scored_candidates.append((score, candidate, base_points, candidate_points))

        scored_candidates.sort(key=lambda item: item[0])
        top_matches = scored_candidates[:top_k]

        print(f"\nEdge: {target_record.edge_name}")
        if not top_matches:
            print("  No candidates available.")
            continue

        for rank, (score, candidate, base_points, candidate_points) in enumerate(
            top_matches, start=1
        ):
            print(
                f"  {rank}. {candidate.fragment} [{candidate.edge_name}] "
                f"-> score={score:.4f}"
            )

            if viz_path:
                target_stem = Path(target_record.fragment).stem
                candidate_stem = Path(candidate.fragment).stem
                filename = (
                    f"{target_stem}_{target_record.edge_name}_rank{rank}_"
                    f"{candidate_stem}_{candidate.edge_name}.png"
                ).replace(" ", "_")
                output_path = viz_path / filename
                title = (
                    f"{target_record.fragment} ({target_record.edge_name}) vs "
                    f"{candidate.fragment} ({candidate.edge_name})\n"
                    f"score={score:.4f}"
                )
                save_match_visualization(base_points, candidate_points, output_path, title)
                print(f"     ↳ saved plot: {output_path}")


def main():
    base_dir = Path(__file__).resolve().parent
    default_input = base_dir / "datasets" / "masks"
    default_output = base_dir / "datasets" / "edge_visualizations"
    default_viz = base_dir / "datasets" / "match_visualizations"

    parser = argparse.ArgumentParser(
        description="Edge extraction and matching tool"
    )
    subparsers = parser.add_subparsers(dest="command")

    process_parser = subparsers.add_parser(
        "process", help="Extract edges and generate visualizations"
    )
    process_parser.add_argument(
        "--input",
        type=str,
        default=str(default_input),
        help="Input directory with fragment images",
    )
    process_parser.add_argument(
        "--output",
        type=str,
        default=str(default_output),
        help="Directory to store edge visualizations",
    )

    match_parser = subparsers.add_parser(
        "match", help="Find matching tear edges for a specific fragment"
    )
    match_parser.add_argument(
        "--input",
        type=str,
        default=str(default_input),
        help="Input directory with fragment images",
    )
    match_parser.add_argument(
        "--fragment",
        type=str,
        required=True,
        help="Filename (or stem) of the fragment to match against",
    )
    match_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top candidates to display per edge",
    )
    match_parser.add_argument(
        "--samples",
        type=int,
        default=80,
        help="Number of resampled points per edge descriptor",
    )
    match_parser.add_argument(
        "--viz-dir",
        type=str,
        default=str(default_viz),
        help="Directory to save visualization plots (set to '' to disable)",
    )

    parser.set_defaults(command="process")
    args = parser.parse_args()

    if args.command == "match":
        match_fragment_edges(
            input_dir=args.input,
            fragment_name=args.fragment,
            top_k=args.top_k,
            num_samples=args.samples,
            viz_dir=args.viz_dir,
        )
    else:
        process_fragments(input_dir=args.input, output_dir=args.output)


if __name__ == "__main__":
    main()
