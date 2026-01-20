"""
Evaluation and Testing Suite for Edge Matching Algorithm
=========================================================

Provides comprehensive evaluation metrics and visualizations
to validate the edge matching algorithm's performance.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from enhanced_edge_matching import (
    EdgeDescriptor,
    EdgeIndex,
    EdgeMatcher,
    EnhancedEdgeExtractor,
    load_descriptors,
    process_fragment,
)


class MatchingEvaluator:
    """Evaluate edge matching performance with metrics and visualizations."""

    def __init__(self, fragments_dir: Path):
        self.fragments_dir = fragments_dir
        self.extractor = EnhancedEdgeExtractor()

    def load_fragment_image(self, fragment_id: str) -> np.ndarray:
        """Load fragment image by ID."""
        # Try different extensions
        for ext in ['.png', '.jpg', '.jpeg']:
            path = self.fragments_dir / f"{fragment_id}{ext}"
            if path.exists():
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    return img

        raise FileNotFoundError(f"Fragment image not found: {fragment_id}")

    def visualize_edge_overlay(
        self,
        query_descriptor: EdgeDescriptor,
        match_descriptor: EdgeDescriptor,
        output_path: Path,
        match_score: float,
    ):
        """
        Visualize two matching edges overlaid on each other.
        Shows how well the complementary edges align.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Query edge
        ax1 = axes[0]
        query_points = query_descriptor.normalized_points
        ax1.plot(query_points[:, 0], query_points[:, 1], 'b-', linewidth=2, label='Query edge')
        ax1.scatter(query_points[0, 0], query_points[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
        ax1.scatter(query_points[-1, 0], query_points[-1, 1], c='red', s=100, marker='s', label='End', zorder=5)
        ax1.set_title(f"Query: {query_descriptor.fragment_id}\n{query_descriptor.edge_name}")
        ax1.axis('equal')
        ax1.grid(alpha=0.3)
        ax1.legend()
        ax1.invert_yaxis()

        # Plot 2: Match edge
        ax2 = axes[1]
        match_points = match_descriptor.normalized_points
        ax2.plot(match_points[:, 0], match_points[:, 1], 'r-', linewidth=2, label='Match edge')
        ax2.scatter(match_points[0, 0], match_points[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
        ax2.scatter(match_points[-1, 0], match_points[-1, 1], c='red', s=100, marker='s', label='End', zorder=5)
        ax2.set_title(f"Match: {match_descriptor.fragment_id}\n{match_descriptor.edge_name}")
        ax2.axis('equal')
        ax2.grid(alpha=0.3)
        ax2.legend()
        ax2.invert_yaxis()

        # Plot 3: Overlay (query edge vs complementary match edge)
        ax3 = axes[2]

        # Find best variant (matching the algorithm's logic)
        best_variant = self._find_best_variant(query_points, match_points)

        ax3.plot(query_points[:, 0], query_points[:, 1], 'b-', linewidth=2, label='Query edge', alpha=0.7)
        ax3.plot(best_variant[:, 0], best_variant[:, 1], 'r--', linewidth=2, label='Match edge (complementary)', alpha=0.7)

        # Draw correspondence lines
        for i in range(0, len(query_points), len(query_points)//10):
            ax3.plot(
                [query_points[i, 0], best_variant[i, 0]],
                [query_points[i, 1], best_variant[i, 1]],
                'gray', alpha=0.3, linewidth=0.5
            )

        ax3.set_title(f"Overlay - Score: {match_score:.4f}\n(Lower is better)")
        ax3.axis('equal')
        ax3.grid(alpha=0.3)
        ax3.legend()
        ax3.invert_yaxis()

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

    def _find_best_variant(self, query_points: np.ndarray, match_points: np.ndarray) -> np.ndarray:
        """Find the best orientation variant of match_points to align with query_points."""
        variants = [
            match_points.copy(),
            match_points[::-1].copy(),
            match_points.copy() * np.array([-1, 1]),
            (match_points[::-1].copy()) * np.array([-1, 1]),
        ]

        best_score = float('inf')
        best_variant = None

        for variant in variants:
            # Complementary for puzzle matching
            complement = -variant
            score = np.sqrt(np.mean((query_points - complement) ** 2))

            if score < best_score:
                best_score = score
                best_variant = complement

        return best_variant if best_variant is not None else -variants[0]

    def visualize_fragment_with_edges(
        self,
        fragment_id: str,
        edge_descriptors: List[EdgeDescriptor],
        output_path: Path,
    ):
        """
        Visualize a fragment with its detected tear edges highlighted.
        """
        try:
            img = self.load_fragment_image(fragment_id)
        except FileNotFoundError:
            print(f"Warning: Could not load image for {fragment_id}")
            return

        # Convert to RGB if needed
        if img.shape[2] == 4:
            # Create white background
            rgb = img[:, :, :3]
            alpha = img[:, :, 3:4] / 255.0
            white_bg = np.ones_like(rgb) * 255
            img_rgb = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img_rgb)

        colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']

        for i, edge_desc in enumerate(edge_descriptors):
            color = colors[i % len(colors)]

            # Draw edge label
            ax.text(
                10, 30 + i * 25,
                f"{edge_desc.edge_name}: len={edge_desc.length:.0f}px, "
                f"complexity={edge_desc.complexity_score:.3f}",
                color=color,
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

        ax.set_title(f"Fragment: {fragment_id}\nDetected {len(edge_descriptors)} tear edge(s)")
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

    def compute_retrieval_metrics(
        self,
        all_matches: Dict[str, List],
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict:
        """
        Compute retrieval metrics for edge matching.

        Since we don't have ground truth, we compute:
        - Score distributions
        - Diversity metrics
        - Performance statistics
        """
        metrics = {
            'total_queries': len(all_matches),
            'avg_candidates_per_query': 0,
            'score_statistics': {},
            'top_k_analysis': {},
        }

        all_scores = []
        all_candidate_counts = []

        for edge_name, matches in all_matches.items():
            if not matches:
                continue

            all_candidate_counts.append(len(matches))
            scores = [m['score'] for m in matches]
            all_scores.extend(scores)

            # Top-k score gaps (measure discriminability)
            for k in k_values:
                if len(matches) >= k:
                    top_k_scores = scores[:k]
                    if k not in metrics['top_k_analysis']:
                        metrics['top_k_analysis'][k] = {
                            'score_ranges': [],
                            'avg_top_score': [],
                        }

                    metrics['top_k_analysis'][k]['score_ranges'].append(
                        max(top_k_scores) - min(top_k_scores)
                    )
                    metrics['top_k_analysis'][k]['avg_top_score'].append(
                        top_k_scores[0]
                    )

        # Aggregate statistics
        if all_scores:
            metrics['score_statistics'] = {
                'min': float(np.min(all_scores)),
                'max': float(np.max(all_scores)),
                'mean': float(np.mean(all_scores)),
                'median': float(np.median(all_scores)),
                'std': float(np.std(all_scores)),
                'percentiles': {
                    '25': float(np.percentile(all_scores, 25)),
                    '75': float(np.percentile(all_scores, 75)),
                    '90': float(np.percentile(all_scores, 90)),
                }
            }

        if all_candidate_counts:
            metrics['avg_candidates_per_query'] = float(np.mean(all_candidate_counts))

        # Average top-k analysis
        for k in k_values:
            if k in metrics['top_k_analysis']:
                data = metrics['top_k_analysis'][k]
                data['avg_score_range'] = float(np.mean(data['score_ranges']))
                data['avg_top_score'] = float(np.mean(data['avg_top_score']))
                # Remove raw lists to keep JSON clean
                del data['score_ranges']

        return metrics

    def generate_evaluation_report(
        self,
        all_descriptors: List[EdgeDescriptor],
        sample_matches: Dict,
        metrics: Dict,
        output_dir: Path,
    ):
        """Generate comprehensive evaluation report with visualizations."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Dataset statistics
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)

        fragment_ids = set(d.fragment_id for d in all_descriptors)
        print(f"Total fragments: {len(fragment_ids)}")
        print(f"Total edges: {len(all_descriptors)}")
        print(f"Avg edges per fragment: {len(all_descriptors) / len(fragment_ids):.2f}")

        # Edge statistics
        lengths = [d.length for d in all_descriptors]
        complexities = [d.complexity_score for d in all_descriptors]

        print(f"\nEdge lengths:")
        print(f"  Min: {np.min(lengths):.1f}px")
        print(f"  Max: {np.max(lengths):.1f}px")
        print(f"  Mean: {np.mean(lengths):.1f}px")
        print(f"  Median: {np.median(lengths):.1f}px")

        print(f"\nEdge complexity scores:")
        print(f"  Min: {np.min(complexities):.3f}")
        print(f"  Max: {np.max(complexities):.3f}")
        print(f"  Mean: {np.mean(complexities):.3f}")
        print(f"  Median: {np.median(complexities):.3f}")

        # 2. Matching performance
        print("\n" + "=" * 60)
        print("MATCHING PERFORMANCE")
        print("=" * 60)

        print(f"Total queries: {metrics['total_queries']}")
        print(f"Avg candidates per query: {metrics['avg_candidates_per_query']:.1f}")

        if metrics['score_statistics']:
            stats = metrics['score_statistics']
            print(f"\nMatch scores (lower is better):")
            print(f"  Min: {stats['min']:.4f}")
            print(f"  Max: {stats['max']:.4f}")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Median: {stats['median']:.4f}")
            print(f"  Std: {stats['std']:.4f}")
            print(f"  25th percentile: {stats['percentiles']['25']:.4f}")
            print(f"  75th percentile: {stats['percentiles']['75']:.4f}")

        if metrics['top_k_analysis']:
            print(f"\nTop-K analysis:")
            for k, data in sorted(metrics['top_k_analysis'].items()):
                print(f"  Top-{k}:")
                print(f"    Avg best score: {data['avg_top_score']:.4f}")
                print(f"    Avg score range: {data['avg_score_range']:.4f}")

        # 3. Visualizations
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        # Plot: Length distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(lengths, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Edge Length (pixels)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Edge Lengths')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'length_distribution.png', dpi=150)
        plt.close()
        print("  ✓ Length distribution")

        # Plot: Complexity distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(complexities, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Complexity Score')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Edge Complexity Scores')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'complexity_distribution.png', dpi=150)
        plt.close()
        print("  ✓ Complexity distribution")

        # Plot: Length vs Complexity scatter
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(lengths, complexities, alpha=0.5, s=20)
        ax.set_xlabel('Edge Length (pixels)')
        ax.set_ylabel('Complexity Score')
        ax.set_title('Edge Length vs Complexity')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'length_vs_complexity.png', dpi=150)
        plt.close()
        print("  ✓ Length vs complexity scatter")

        # Save metrics to JSON
        metrics_file = output_dir / 'evaluation_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  ✓ Metrics saved to {metrics_file}")

        print("\n" + "=" * 60)
        print(f"Evaluation report saved to {output_dir}/")
        print("=" * 60)


def run_evaluation(args):
    """Run comprehensive evaluation on the edge matching system."""
    descriptors_path = Path(args.descriptors)
    fragments_dir = Path(args.fragments_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EDGE MATCHING EVALUATION")
    print("=" * 60)

    # Load descriptors
    print("\nLoading edge descriptors...")
    all_descriptors = load_descriptors(descriptors_path)

    # Get unique fragments
    fragment_ids = list(set(d.fragment_id for d in all_descriptors))
    print(f"Found {len(fragment_ids)} unique fragments")

    # Select test fragments
    if args.test_fragments:
        test_fragments = args.test_fragments.split(',')
    else:
        # Randomly select N fragments for testing
        np.random.seed(42)
        n_test = min(args.num_test, len(fragment_ids))
        test_fragments = np.random.choice(fragment_ids, size=n_test, replace=False).tolist()

    print(f"Testing on {len(test_fragments)} fragments: {test_fragments}")

    # Build index
    print("\nBuilding edge index...")
    index = EdgeIndex()
    index.build_index(all_descriptors)

    # Run matching for each test fragment
    matcher = EdgeMatcher()
    evaluator = MatchingEvaluator(fragments_dir)

    all_results = {}
    timing_stats = []

    for i, fragment_id in enumerate(test_fragments, 1):
        print(f"\n[{i}/{len(test_fragments)}] Processing {fragment_id}...")

        # Get edges for this fragment
        fragment_edges = [d for d in all_descriptors if d.fragment_id == fragment_id]

        if not fragment_edges:
            print(f"  No edges found for {fragment_id}")
            continue

        fragment_matches = {}

        for edge_desc in fragment_edges:
            # Find candidates
            start_time = time.time()
            candidates = index.find_candidates(edge_desc, k=50)
            index_time = time.time() - start_time

            # Match
            start_time = time.time()
            matches = matcher.match_edges(edge_desc, candidates, top_k=args.top_k)
            match_time = time.time() - start_time

            total_time = index_time + match_time
            timing_stats.append(total_time)

            print(f"  {edge_desc.edge_name}: {len(matches)} matches in {total_time*1000:.1f}ms")

            # Store results
            fragment_matches[edge_desc.edge_name] = [
                {
                    'match_fragment': m.match_fragment,
                    'match_edge': m.match_edge,
                    'score': m.score,
                    'details': m.match_details,
                }
                for m in matches
            ]

            # Visualize top match
            if matches and args.visualize:
                top_match = matches[0]
                match_desc = next(
                    d for d in all_descriptors
                    if d.fragment_id == top_match.match_fragment and d.edge_name == top_match.match_edge
                )

                viz_path = output_dir / 'visualizations' / f"{fragment_id}_{edge_desc.edge_name}_top_match.png"
                viz_path.parent.mkdir(parents=True, exist_ok=True)

                evaluator.visualize_edge_overlay(
                    edge_desc,
                    match_desc,
                    viz_path,
                    top_match.score
                )

        all_results[fragment_id] = fragment_matches

        # Visualize fragment with edges
        if args.visualize:
            viz_path = output_dir / 'fragments' / f"{fragment_id}_edges.png"
            viz_path.parent.mkdir(parents=True, exist_ok=True)
            evaluator.visualize_fragment_with_edges(fragment_id, fragment_edges, viz_path)

    # Compute metrics
    print("\n" + "=" * 60)
    print("COMPUTING METRICS")
    print("=" * 60)

    # Flatten all matches for metrics
    all_matches_flat = {}
    for fragment_id, fragment_matches in all_results.items():
        for edge_name, matches in fragment_matches.items():
            key = f"{fragment_id}_{edge_name}"
            all_matches_flat[key] = matches

    metrics = evaluator.compute_retrieval_metrics(all_matches_flat)

    # Add timing statistics
    if timing_stats:
        metrics['timing'] = {
            'avg_query_time_ms': float(np.mean(timing_stats) * 1000),
            'median_query_time_ms': float(np.median(timing_stats) * 1000),
            'max_query_time_ms': float(np.max(timing_stats) * 1000),
            'total_queries': len(timing_stats),
        }

    # Generate report
    evaluator.generate_evaluation_report(
        all_descriptors,
        all_results,
        metrics,
        output_dir / 'report'
    )

    # Save all results
    results_file = output_dir / 'all_matches.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ All results saved to {results_file}")

    # Timing summary
    if timing_stats:
        print("\n" + "=" * 60)
        print("TIMING PERFORMANCE")
        print("=" * 60)
        print(f"Total queries: {len(timing_stats)}")
        print(f"Avg query time: {np.mean(timing_stats)*1000:.1f}ms")
        print(f"Median query time: {np.median(timing_stats)*1000:.1f}ms")
        print(f"Max query time: {np.max(timing_stats)*1000:.1f}ms")
        print(f"Min query time: {np.min(timing_stats)*1000:.1f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate edge matching performance"
    )
    parser.add_argument(
        '--descriptors',
        type=str,
        required=True,
        help='Path to edge descriptors JSON file'
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
        default='output/evaluation',
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--num-test',
        type=int,
        default=10,
        help='Number of fragments to test'
    )
    parser.add_argument(
        '--test-fragments',
        type=str,
        default=None,
        help='Comma-separated list of specific fragment IDs to test'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of top matches to retrieve per edge'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations (slower but more informative)'
    )

    args = parser.parse_args()
    run_evaluation(args)


if __name__ == '__main__':
    main()
