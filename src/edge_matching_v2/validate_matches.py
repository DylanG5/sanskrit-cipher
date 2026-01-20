"""
Visual Validation Tool for Edge Matches
========================================

Shows actual fragment images with matched edges aligned,
allowing manual verification of match quality.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

from enhanced_edge_matching import (
    EdgeDescriptor,
    load_descriptors,
)


class MatchValidator:
    """Visual validation of edge matches using actual fragment images."""

    def __init__(self, fragments_dir: Path):
        self.fragments_dir = fragments_dir

    def load_fragment_image(self, fragment_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load fragment image and mask."""
        # Try different extensions
        for ext in ['.png', '.jpg', '.jpeg']:
            path = self.fragments_dir / f"{fragment_id}{ext}"
            if path.exists():
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if img.shape[2] == 4:
                        # RGBA image
                        rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
                        mask = img[:, :, 3]
                        return rgb, mask
                    else:
                        # RGB image
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # Create mask from brightness
                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                        return rgb, mask

        raise FileNotFoundError(f"Fragment image not found: {fragment_id}")

    def get_edge_region(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        edge_name: str,
        margin: int = 50
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract the region of image near the specified edge.

        Returns:
            (cropped_img, (x1, y1, x2, y2)) - cropped image and bounding box
        """
        height, width = img.shape[:2]

        # Define edge regions
        if edge_name == 'top_edge':
            y1, y2 = 0, min(height // 3, height)
            x1, x2 = 0, width
        elif edge_name == 'bottom_edge':
            y1, y2 = max(2 * height // 3, 0), height
            x1, x2 = 0, width
        elif edge_name == 'left_edge':
            y1, y2 = 0, height
            x1, x2 = 0, min(width // 3, width)
        elif edge_name == 'right_edge':
            y1, y2 = 0, height
            x1, x2 = max(2 * width // 3, 0), width
        else:
            return img, (0, 0, width, height)

        # Add margin
        y1 = max(0, y1 - margin)
        y2 = min(height, y2 + margin)
        x1 = max(0, x1 - margin)
        x2 = min(width, x2 + margin)

        cropped = img[y1:y2, x1:x2]
        return cropped, (x1, y1, x2, y2)

    def get_edge_orientation(self, edge_name: str) -> str:
        """Get expected orientation for edge type."""
        if edge_name in ['top_edge', 'bottom_edge']:
            return 'horizontal'
        else:
            return 'vertical'

    def align_fragments(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        edge1_name: str,
        edge2_name: str,
        gap: int = 20
    ) -> np.ndarray:
        """
        Create side-by-side visualization of two fragments with their
        matching edges adjacent.

        Args:
            img1, img2: Fragment images
            edge1_name, edge2_name: Which edges are matching
            gap: Pixels between fragments

        Returns:
            Combined image
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Determine layout based on edge types
        orientation = self.get_edge_orientation(edge1_name)

        if orientation == 'horizontal':
            # Stack vertically (for top/bottom edges)
            if edge1_name == 'bottom_edge':
                # img1 on top, img2 on bottom
                max_width = max(w1, w2)
                canvas_height = h1 + h2 + gap
                canvas = np.ones((canvas_height, max_width, 3), dtype=np.uint8) * 255

                # Center img1
                x1_offset = (max_width - w1) // 2
                canvas[:h1, x1_offset:x1_offset+w1] = img1

                # Center img2
                x2_offset = (max_width - w2) // 2
                canvas[h1+gap:, x2_offset:x2_offset+w2] = img2

            else:  # top_edge
                # img2 on top, img1 on bottom
                max_width = max(w1, w2)
                canvas_height = h1 + h2 + gap
                canvas = np.ones((canvas_height, max_width, 3), dtype=np.uint8) * 255

                x2_offset = (max_width - w2) // 2
                canvas[:h2, x2_offset:x2_offset+w2] = img2

                x1_offset = (max_width - w1) // 2
                canvas[h2+gap:, x1_offset:x1_offset+w1] = img1

        else:  # vertical edges
            # Stack horizontally (for left/right edges)
            if edge1_name == 'right_edge':
                # img1 on left, img2 on right
                max_height = max(h1, h2)
                canvas_width = w1 + w2 + gap
                canvas = np.ones((max_height, canvas_width, 3), dtype=np.uint8) * 255

                # Center img1
                y1_offset = (max_height - h1) // 2
                canvas[y1_offset:y1_offset+h1, :w1] = img1

                # Center img2
                y2_offset = (max_height - h2) // 2
                canvas[y2_offset:y2_offset+h2, w1+gap:] = img2

            else:  # left_edge
                # img2 on left, img1 on right
                max_height = max(h1, h2)
                canvas_width = w1 + w2 + gap
                canvas = np.ones((max_height, canvas_width, 3), dtype=np.uint8) * 255

                y2_offset = (max_height - h2) // 2
                canvas[y2_offset:y2_offset+h2, :w2] = img2

                y1_offset = (max_height - h1) // 2
                canvas[y1_offset:y1_offset+h1, w2+gap:] = img1

        return canvas

    def create_validation_panel(
        self,
        query_fragment: str,
        query_edge: str,
        match_fragment: str,
        match_edge: str,
        score: float,
        details: dict,
        output_path: Path,
    ):
        """
        Create comprehensive validation panel showing:
        1. Full query fragment with edge highlighted
        2. Full match fragment with edge highlighted
        3. Side-by-side alignment of matched edges
        4. Scoring details
        """
        # Load images
        query_img, query_mask = self.load_fragment_image(query_fragment)
        match_img, match_mask = self.load_fragment_image(match_fragment)

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Full query fragment
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(query_img)
        self._highlight_edge(ax1, query_img.shape, query_edge, 'blue')
        ax1.set_title(f'Query Fragment\n{query_fragment}', fontsize=12, fontweight='bold')
        ax1.axis('off')

        # 2. Full match fragment
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(match_img)
        self._highlight_edge(ax2, match_img.shape, match_edge, 'red')
        ax2.set_title(f'Match Fragment\n{match_fragment}', fontsize=12, fontweight='bold')
        ax2.axis('off')

        # 3. Score details
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        self._render_score_details(ax3, query_edge, match_edge, score, details)

        # 4. Query edge closeup
        ax4 = fig.add_subplot(gs[1, 0])
        query_crop, query_bbox = self.get_edge_region(query_img, query_mask, query_edge)
        ax4.imshow(query_crop)
        ax4.set_title(f'Query Edge: {query_edge}', fontsize=11)
        ax4.axis('off')
        ax4.add_patch(Rectangle((0, 0), query_crop.shape[1]-1, query_crop.shape[0]-1,
                                fill=False, edgecolor='blue', linewidth=3))

        # 5. Match edge closeup
        ax5 = fig.add_subplot(gs[1, 1])
        match_crop, match_bbox = self.get_edge_region(match_img, match_mask, match_edge)
        ax5.imshow(match_crop)
        ax5.set_title(f'Match Edge: {match_edge}', fontsize=11)
        ax5.axis('off')
        ax5.add_patch(Rectangle((0, 0), match_crop.shape[1]-1, match_crop.shape[0]-1,
                                fill=False, edgecolor='red', linewidth=3))

        # 6. Quality indicator
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        self._render_quality_indicator(ax6, score)

        # 7-9. Side-by-side alignment (spans bottom row)
        ax7 = fig.add_subplot(gs[2, :])
        aligned = self.align_fragments(query_img, match_img, query_edge, match_edge, gap=30)
        ax7.imshow(aligned)
        ax7.set_title('Aligned Fragments (Gap shows how edges would meet)', fontsize=12, fontweight='bold')
        ax7.axis('off')

        # Add arrows/indicators at the gap
        self._add_alignment_indicators(ax7, aligned.shape, query_edge)

        # Overall title
        quality = self._get_quality_label(score)
        fig.suptitle(
            f'Edge Match Validation - {quality}\n'
            f'Score: {score:.4f} (lower is better)',
            fontsize=16,
            fontweight='bold'
        )

        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

    def _highlight_edge(self, ax, img_shape, edge_name: str, color: str):
        """Add colored box highlighting the edge region."""
        height, width = img_shape[:2]

        if edge_name == 'top_edge':
            rect = Rectangle((0, 0), width, height//4, fill=False,
                           edgecolor=color, linewidth=3, linestyle='--')
        elif edge_name == 'bottom_edge':
            rect = Rectangle((0, 3*height//4), width, height//4, fill=False,
                           edgecolor=color, linewidth=3, linestyle='--')
        elif edge_name == 'left_edge':
            rect = Rectangle((0, 0), width//4, height, fill=False,
                           edgecolor=color, linewidth=3, linestyle='--')
        elif edge_name == 'right_edge':
            rect = Rectangle((3*width//4, 0), width//4, height, fill=False,
                           edgecolor=color, linewidth=3, linestyle='--')
        else:
            return

        ax.add_patch(rect)

    def _render_score_details(self, ax, query_edge: str, match_edge: str, score: float, details: dict):
        """Render detailed scoring breakdown."""
        text_parts = [
            f"Match Details",
            f"=" * 40,
            f"",
            f"Query Edge:  {query_edge}",
            f"Match Edge:  {match_edge}",
            f"",
            f"Overall Score: {score:.4f}",
            f"",
            f"Component Scores:",
            f"  Shape (40%):      {details.get('shape_score', 0):.4f}",
            f"  Fourier (25%):    {details.get('fourier_score', 0):.4f}",
            f"  Length (20%):     {details.get('length_penalty', 0):.4f}",
            f"  Histogram (15%):  {details.get('histogram_distance', 0):.4f}",
            f"",
            f"Interpretation:",
            f"  {self._get_quality_description(score)}"
        ]

        text = "\n".join(text_parts)
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def _render_quality_indicator(self, ax, score: float):
        """Visual quality indicator with gauge."""
        # Quality thresholds
        thresholds = [
            (0.0, 0.2, 'Excellent', 'green'),
            (0.2, 0.4, 'Good', 'yellowgreen'),
            (0.4, 0.6, 'Fair', 'orange'),
            (0.6, 1.0, 'Poor', 'red'),
        ]

        quality_label = self._get_quality_label(score)
        quality_color = 'gray'
        for min_s, max_s, label, color in thresholds:
            if min_s <= score < max_s:
                quality_label = label
                quality_color = color
                break

        # Draw gauge
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Background bars
        bar_height = 0.15
        y_pos = 0.6
        for i, (min_s, max_s, label, color) in enumerate(thresholds):
            width = max_s - min_s
            rect = Rectangle((min_s, y_pos - i * bar_height), width, bar_height * 0.8,
                           facecolor=color, alpha=0.3, edgecolor='black')
            ax.add_patch(rect)
            ax.text(min_s + width/2, y_pos - i * bar_height + bar_height*0.4,
                   label, ha='center', va='center', fontsize=9)

        # Score indicator
        ax.plot([score, score], [0.1, 0.9], 'k-', linewidth=4, marker='v',
               markersize=15, markerfacecolor='black')

        # Score text
        ax.text(0.5, 0.95, f'Match Quality: {quality_label}',
               ha='center', va='top', fontsize=14, fontweight='bold',
               color=quality_color)
        ax.text(0.5, 0.05, f'Score: {score:.4f}',
               ha='center', va='bottom', fontsize=12)

        ax.axis('off')

    def _add_alignment_indicators(self, ax, canvas_shape, edge_name: str):
        """Add arrows/text showing where edges meet."""
        height, width = canvas_shape[:2]

        orientation = self.get_edge_orientation(edge_name)

        if orientation == 'horizontal':
            # Horizontal line in middle
            y = height // 2
            ax.axhline(y=y, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax.text(width//2, y-20, '← Edges meet here →',
                   ha='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        else:
            # Vertical line in middle
            x = width // 2
            ax.axvline(x=x, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax.text(x+20, height//2, '↑\nEdges\nmeet\nhere\n↓',
                   ha='left', va='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    def _get_quality_label(self, score: float) -> str:
        """Get quality label from score."""
        if score < 0.2:
            return "Excellent Match"
        elif score < 0.4:
            return "Good Match"
        elif score < 0.6:
            return "Fair Match"
        else:
            return "Poor Match"

    def _get_quality_description(self, score: float) -> str:
        """Get quality description from score."""
        if score < 0.2:
            return "High confidence - edges likely fit well"
        elif score < 0.4:
            return "Reasonable match - worth investigating"
        elif score < 0.6:
            return "Possible match - manual verification needed"
        else:
            return "Unlikely match - significant differences"


def validate_top_matches(args):
    """Generate validation visualizations for top matches."""
    descriptors_path = Path(args.descriptors)
    matches_path = Path(args.matches)
    fragments_dir = Path(args.fragments_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MATCH VALIDATION")
    print("=" * 60)

    # Load descriptors
    print("\nLoading edge descriptors...")
    all_descriptors = load_descriptors(descriptors_path)
    descriptor_map = {
        (d.fragment_id, d.edge_name): d
        for d in all_descriptors
    }

    # Load match results
    print(f"Loading match results from {matches_path}...")
    with open(matches_path, 'r') as f:
        match_data = json.load(f)

    fragment_id = match_data['fragment']
    matches = match_data['matches']

    print(f"\nValidating matches for fragment: {fragment_id}")
    print(f"Found {len(matches)} edges with matches")

    validator = MatchValidator(fragments_dir)

    # Generate validation for each edge's top matches
    total_validations = 0

    for edge_name, edge_matches in matches.items():
        print(f"\n{edge_name}:")

        # Get top N matches
        top_matches = edge_matches[:args.top_n]

        for i, match in enumerate(top_matches, 1):
            match_fragment = match['match_fragment']
            match_edge = match['match_edge']
            score = match['score']
            details = match['details']

            print(f"  [{i}] {match_fragment} [{match_edge}] - Score: {score:.4f}")

            try:
                output_file = output_dir / f"{fragment_id}_{edge_name}_rank{i}_{match_fragment}_{match_edge}.png"

                validator.create_validation_panel(
                    fragment_id,
                    edge_name,
                    match_fragment,
                    match_edge,
                    score,
                    details,
                    output_file
                )

                print(f"      ✓ Saved: {output_file.name}")
                total_validations += 1

            except Exception as e:
                print(f"      ✗ Error: {e}")

    print("\n" + "=" * 60)
    print(f"VALIDATION COMPLETE")
    print("=" * 60)
    print(f"Generated {total_validations} validation panels")
    print(f"Output directory: {output_dir}")
    print("\nReview the validation images to verify match quality!")


def validate_best_matches_from_evaluation(args):
    """Generate validations for best matches from evaluation results."""
    all_matches_path = Path(args.evaluation_matches)
    fragments_dir = Path(args.fragments_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VALIDATION OF BEST MATCHES FROM EVALUATION")
    print("=" * 60)

    # Load evaluation results
    with open(all_matches_path, 'r') as f:
        all_matches = json.load(f)

    print(f"\nFound {len(all_matches)} fragments in evaluation")

    validator = MatchValidator(fragments_dir)
    total_validations = 0

    # For each fragment, validate top match for each edge
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
            details = best_match['details']

            quality = "⭐" if score < 0.2 else "✓" if score < 0.4 else "?"
            print(f"  {quality} {edge_name} → {match_fragment}[{match_edge}] ({score:.4f})")

            try:
                output_file = output_dir / f"{fragment_id}_{edge_name}_BEST_{match_fragment}_{match_edge}.png"

                validator.create_validation_panel(
                    fragment_id,
                    edge_name,
                    match_fragment,
                    match_edge,
                    score,
                    details,
                    output_file
                )

                total_validations += 1

            except Exception as e:
                print(f"      ✗ Error: {e}")

    print("\n" + "=" * 60)
    print(f"VALIDATION COMPLETE")
    print("=" * 60)
    print(f"Generated {total_validations} validation panels")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visual validation of edge matches"
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Validate from match results
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate matches from match results file'
    )
    validate_parser.add_argument(
        '--descriptors',
        type=str,
        required=True,
        help='Path to edge descriptors JSON'
    )
    validate_parser.add_argument(
        '--matches',
        type=str,
        required=True,
        help='Path to match results JSON (from enhanced_edge_matching.py match)'
    )
    validate_parser.add_argument(
        '--fragments-dir',
        type=str,
        required=True,
        help='Directory containing fragment images'
    )
    validate_parser.add_argument(
        '--output',
        type=str,
        default='output/validation',
        help='Output directory for validation images'
    )
    validate_parser.add_argument(
        '--top-n',
        type=int,
        default=3,
        help='Number of top matches to validate per edge'
    )

    # Validate best from evaluation
    best_parser = subparsers.add_parser(
        'validate-best',
        help='Validate best matches from evaluation results'
    )
    best_parser.add_argument(
        '--evaluation-matches',
        type=str,
        required=True,
        help='Path to all_matches.json from evaluation'
    )
    best_parser.add_argument(
        '--fragments-dir',
        type=str,
        required=True,
        help='Directory containing fragment images'
    )
    best_parser.add_argument(
        '--output',
        type=str,
        default='output/validation_best',
        help='Output directory for validation images'
    )

    args = parser.parse_args()

    if args.command == 'validate':
        validate_top_matches(args)
    elif args.command == 'validate-best':
        validate_best_matches_from_evaluation(args)


if __name__ == '__main__':
    main()
