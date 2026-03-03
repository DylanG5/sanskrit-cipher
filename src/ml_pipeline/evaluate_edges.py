"""
Edge Detection Evaluation Script

Usage:
1. Generate predictions for manual labeling:
   python -m ml_pipeline.evaluate_edges generate --sample-size 50

2. After filling in ground_truth columns in the CSV, calculate accuracy:
   python -m ml_pipeline.evaluate_edges evaluate --csv edge_evaluation.csv
"""

import argparse
import csv
import os
import sqlite3
import random
from pathlib import Path
from typing import List, Dict, Any, Optional


def get_db_path() -> str:
    """Get the database path."""
    return str(Path(__file__).parent.parent / "web/web-canvas/electron/resources/database/fragments.db")


def get_data_dir() -> str:
    """Get the data directory path."""
    return str(Path(__file__).parent.parent / "web/web-canvas/data")


def generate_evaluation_csv(sample_size: int, output_path: str, collection: Optional[str] = None) -> None:
    """
    Generate a CSV with edge detection predictions for manual labeling.

    Args:
        sample_size: Number of fragments to sample
        output_path: Path to output CSV file
        collection: Optional collection filter (e.g., "BLL" or "CUL")
    """
    db_path = get_db_path()
    data_dir = get_data_dir()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get fragments that have been processed with edge detection
    query = """
        SELECT fragment_id, image_path,
               has_top_edge, has_bottom_edge, has_left_edge, has_right_edge, edge_piece
        FROM fragments
        WHERE segmentation_coords IS NOT NULL
    """
    params = []

    if collection:
        query += " AND image_path LIKE ?"
        params.append(f"{collection}%")

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    if len(rows) == 0:
        print("No fragments found with segmentation data.")
        return

    # Sample randomly
    if len(rows) > sample_size:
        rows = random.sample(rows, sample_size)

    print(f"Generating evaluation CSV with {len(rows)} fragments...")

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'fragment_id',
            'image_path',
            'full_image_path',
            # Predictions (from algorithm)
            'pred_top', 'pred_bottom', 'pred_left', 'pred_right', 'pred_edge_piece',
            # Ground truth (fill these in manually)
            'gt_top', 'gt_bottom', 'gt_left', 'gt_right',
            # Notes
            'notes'
        ])

        for row in rows:
            full_path = os.path.join(data_dir, row['image_path'])
            writer.writerow([
                row['fragment_id'],
                row['image_path'],
                full_path,
                # Predictions
                1 if row['has_top_edge'] else 0,
                1 if row['has_bottom_edge'] else 0,
                1 if row['has_left_edge'] else 0,
                1 if row['has_right_edge'] else 0,
                1 if row['edge_piece'] else 0,
                # Ground truth (empty - to be filled)
                '', '', '', '',
                # Notes
                ''
            ])

    print(f"CSV saved to: {output_path}")
    print("\nInstructions:")
    print("1. Open the CSV in a spreadsheet application")
    print("2. For each row, view the image at 'full_image_path'")
    print("3. Fill in gt_top, gt_bottom, gt_left, gt_right with 1 (has edge) or 0 (no edge)")
    print("4. Run: python -m ml_pipeline.evaluate_edges evaluate --csv " + output_path)


def evaluate_predictions(csv_path: str) -> None:
    """
    Evaluate edge detection predictions against ground truth labels.

    Args:
        csv_path: Path to CSV with predictions and ground truth
    """
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter rows that have ground truth labels
    labeled_rows = [r for r in rows if r.get('gt_top', '').strip() != '']

    if len(labeled_rows) == 0:
        print("No ground truth labels found in CSV. Please fill in gt_top, gt_bottom, gt_left, gt_right columns.")
        return

    print(f"Evaluating {len(labeled_rows)} labeled fragments...\n")

    edges = ['top', 'bottom', 'left', 'right']
    metrics: Dict[str, Dict[str, int]] = {
        edge: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for edge in edges
    }

    # Also track edge_piece accuracy
    edge_piece_correct = 0
    edge_piece_total = 0

    misclassified: List[Dict[str, Any]] = []

    for row in labeled_rows:
        has_any_error = False

        for edge in edges:
            pred = int(row[f'pred_{edge}'])
            gt = int(row[f'gt_{edge}'])

            if pred == 1 and gt == 1:
                metrics[edge]['tp'] += 1
            elif pred == 1 and gt == 0:
                metrics[edge]['fp'] += 1
                has_any_error = True
            elif pred == 0 and gt == 0:
                metrics[edge]['tn'] += 1
            else:  # pred == 0 and gt == 1
                metrics[edge]['fn'] += 1
                has_any_error = True

        # Calculate expected edge_piece from ground truth
        gt_edge_piece = 1 if any(int(row[f'gt_{e}']) == 1 for e in edges) else 0
        pred_edge_piece = int(row['pred_edge_piece'])

        if gt_edge_piece == pred_edge_piece:
            edge_piece_correct += 1
        edge_piece_total += 1

        if has_any_error:
            misclassified.append({
                'fragment_id': row['fragment_id'],
                'predictions': {e: int(row[f'pred_{e}']) for e in edges},
                'ground_truth': {e: int(row[f'gt_{e}']) for e in edges},
                'notes': row.get('notes', '')
            })

    # Print results
    print("=" * 60)
    print("EDGE DETECTION EVALUATION RESULTS")
    print("=" * 60)

    for edge in edges:
        m = metrics[edge]
        total = m['tp'] + m['fp'] + m['tn'] + m['fn']
        accuracy = (m['tp'] + m['tn']) / total if total > 0 else 0
        precision = m['tp'] / (m['tp'] + m['fp']) if (m['tp'] + m['fp']) > 0 else 0
        recall = m['tp'] / (m['tp'] + m['fn']) if (m['tp'] + m['fn']) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n{edge.upper()} EDGE:")
        print(f"  Accuracy:  {accuracy:.1%}")
        print(f"  Precision: {precision:.1%} (of predicted edges, how many are correct)")
        print(f"  Recall:    {recall:.1%} (of actual edges, how many were detected)")
        print(f"  F1 Score:  {f1:.3f}")
        print(f"  Confusion: TP={m['tp']}, FP={m['fp']}, TN={m['tn']}, FN={m['fn']}")

    print(f"\n{'=' * 60}")
    print(f"EDGE_PIECE (any edge detected):")
    print(f"  Accuracy: {edge_piece_correct / edge_piece_total:.1%} ({edge_piece_correct}/{edge_piece_total})")

    if misclassified:
        print(f"\n{'=' * 60}")
        print(f"MISCLASSIFIED FRAGMENTS ({len(misclassified)}):")
        print("-" * 60)
        for item in misclassified[:10]:  # Show first 10
            print(f"\n  {item['fragment_id']}")
            print(f"    Pred: top={item['predictions']['top']}, bottom={item['predictions']['bottom']}, "
                  f"left={item['predictions']['left']}, right={item['predictions']['right']}")
            print(f"    GT:   top={item['ground_truth']['top']}, bottom={item['ground_truth']['bottom']}, "
                  f"left={item['ground_truth']['left']}, right={item['ground_truth']['right']}")
            if item['notes']:
                print(f"    Notes: {item['notes']}")

        if len(misclassified) > 10:
            print(f"\n  ... and {len(misclassified) - 10} more")


def main():
    parser = argparse.ArgumentParser(description='Edge Detection Evaluation')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate evaluation CSV')
    gen_parser.add_argument('--sample-size', type=int, default=50, help='Number of fragments to sample')
    gen_parser.add_argument('--output', type=str, default='edge_evaluation.csv', help='Output CSV path')
    gen_parser.add_argument('--collection', type=str, help='Filter by collection (e.g., BLL, CUL)')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate predictions against ground truth')
    eval_parser.add_argument('--csv', type=str, required=True, help='Path to CSV with ground truth labels')

    args = parser.parse_args()

    if args.command == 'generate':
        generate_evaluation_csv(args.sample_size, args.output, args.collection)
    elif args.command == 'evaluate':
        evaluate_predictions(args.csv)


if __name__ == '__main__':
    main()
