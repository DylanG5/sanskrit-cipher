"""
Create a validation project in the DB with auto-placed matched fragment pairs.

Usage:
    python -m reconstruction.validate \
        --db-path src/web/web-canvas/electron/resources/database/fragments.db \
        --project-name "Edge Match Validation" \
        --top-n 10 --grid-scale 25
"""

import argparse
import json
import sqlite3

from .config import DEFAULT_GRID_SCALE
from .db import (
    create_validation_project,
    ensure_edge_matches_table,
    get_fragment_scale,
    get_top_matches,
    insert_project_fragment,
)


def _calculate_display_size(
    pixels_per_unit: float,
    scale_unit: str,
    original_width: int,
    original_height: int,
    grid_scale: int,
) -> tuple[float, float]:
    """Mirror the CanvasPage.tsx calculateDisplaySize logic."""
    mm_to_cm = 0.1
    if scale_unit == 'mm':
        w_cm = (original_width / pixels_per_unit) * mm_to_cm
        h_cm = (original_height / pixels_per_unit) * mm_to_cm
    else:
        w_cm = original_width / pixels_per_unit
        h_cm = original_height / pixels_per_unit
    w_px = max(50, min(2000, round(w_cm * grid_scale)))
    h_px = max(50, min(2000, round(h_cm * grid_scale)))
    return w_px, h_px


def _get_image_dims(conn: sqlite3.Connection, fragment_id: str) -> tuple[int, int]:
    """Get original image dimensions from segmentation_coords bounding box."""
    row = conn.execute(
        "SELECT segmentation_coords FROM fragments WHERE fragment_id = ?",
        (fragment_id,),
    ).fetchone()
    if not row or not row[0]:
        return 800, 600
    try:
        data = json.loads(row[0])
        if 'image_shape' in data:
            return data['image_shape'][1], data['image_shape'][0]  # (w, h)
        contours = data.get('contours', [])
        if contours and contours[0]:
            pts = contours[0]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return int(max(xs) * 1.1) + 1, int(max(ys) * 1.1) + 1
    except Exception:
        pass
    return 800, 600


def run_validate(
    db_path: str,
    project_name: str = "Edge Match Validation",
    top_n: int = 10,
    grid_scale: int = DEFAULT_GRID_SCALE,
) -> None:
    conn = sqlite3.connect(db_path)
    ensure_edge_matches_table(conn)

    matches = get_top_matches(conn, limit=top_n)
    print(f"[validate] Found {len(matches)} rank-1 matches")

    if not matches:
        print("[validate] No matches to validate.")
        conn.close()
        return

    project_id = create_validation_project(
        conn, project_name,
        f"Auto-placed top-{top_n} edge matches for visual validation",
    )
    print(f"[validate] Created project '{project_name}' (id={project_id})")

    x_cursor = 100.0
    pair_spacing = 500.0
    y_base = 200.0

    for idx, m in enumerate(matches):
        frag_a_id, edge_a, frag_b_id, edge_b, score, rel_x, rel_y, rot, _ = m

        # Get scale info for display sizing
        scale_a = get_fragment_scale(conn, frag_a_id)
        scale_b = get_fragment_scale(conn, frag_b_id)

        # Get original image dimensions
        w_a_orig, h_a_orig = _get_image_dims(conn, frag_a_id)
        w_b_orig, h_b_orig = _get_image_dims(conn, frag_b_id)

        # Compute display sizes
        if scale_a:
            w_a, h_a = _calculate_display_size(scale_a[0], scale_a[1], w_a_orig, h_a_orig, grid_scale)
        else:
            w_a, h_a = min(300, w_a_orig), min(300, h_a_orig)

        if scale_b:
            w_b, h_b = _calculate_display_size(scale_b[0], scale_b[1], w_b_orig, h_b_orig, grid_scale)
        else:
            w_b, h_b = min(300, w_b_orig), min(300, h_b_orig)

        # Place fragment A
        ax, ay = x_cursor, y_base
        insert_project_fragment(conn, project_id, frag_a_id, ax, ay, w_a, h_a,
                                rotation=0, z_index=idx * 2)

        # Place fragment B relative to A using stored offsets
        bx = ax + (rel_x or 0) * grid_scale
        by = ay + (rel_y or 0) * grid_scale
        insert_project_fragment(conn, project_id, frag_b_id, bx, by, w_b, h_b,
                                rotation=rot or 0, z_index=idx * 2 + 1)

        print(f"  Pair {idx + 1}: {frag_a_id}[{edge_a}] <-> {frag_b_id}[{edge_b}] "
              f"score={score:.4f}")

        x_cursor += pair_spacing

    conn.commit()
    conn.close()
    print(f"[validate] Placed {len(matches)} pairs in project '{project_name}'")


def main():
    parser = argparse.ArgumentParser(description="Create validation project from edge matches")
    parser.add_argument("--db-path", required=True, help="Path to fragments.db")
    parser.add_argument("--project-name", default="Edge Match Validation")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--grid-scale", type=int, default=DEFAULT_GRID_SCALE)
    args = parser.parse_args()
    run_validate(args.db_path, args.project_name, args.top_n, args.grid_scale)


if __name__ == "__main__":
    main()
