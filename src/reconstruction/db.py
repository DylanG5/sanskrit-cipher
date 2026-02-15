"""
Database helpers: read fragments, write edge_matches, create validation projects.
"""

import json
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data types returned by queries
# ---------------------------------------------------------------------------

@dataclass
class FragmentRow:
    fragment_id: str
    image_path: str
    pixels_per_unit: float
    scale_unit: str
    segmentation_coords: str
    line_count: Optional[int]
    script_type: Optional[str]
    image_width: int     # extracted from segmentation_coords bounding box
    image_height: int


@dataclass
class MatchRow:
    fragment_a_id: str
    edge_a_name: str
    fragment_b_id: str
    edge_b_name: str
    score: float
    rank: int
    confidence: float
    score_details: str
    relative_x_cm: float
    relative_y_cm: float
    rotation_deg: float
    algorithm_version: str


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

EDGE_MATCHES_DDL = """
CREATE TABLE IF NOT EXISTS edge_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fragment_a_id TEXT NOT NULL,
    edge_a_name TEXT NOT NULL,
    fragment_b_id TEXT NOT NULL,
    edge_b_name TEXT NOT NULL,
    score REAL NOT NULL,
    rank INTEGER NOT NULL,
    confidence REAL,
    score_details TEXT,
    relative_x_cm REAL,
    relative_y_cm REAL,
    rotation_deg REAL DEFAULT 0,
    algorithm_version TEXT,
    computed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fragment_a_id) REFERENCES fragments(fragment_id),
    FOREIGN KEY (fragment_b_id) REFERENCES fragments(fragment_id)
);
CREATE INDEX IF NOT EXISTS idx_edge_matches_a ON edge_matches(fragment_a_id);
CREATE INDEX IF NOT EXISTS idx_edge_matches_b ON edge_matches(fragment_b_id);
"""


def ensure_edge_matches_table(conn: sqlite3.Connection) -> None:
    conn.executescript(EDGE_MATCHES_DDL)


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def _image_dims_from_seg(seg_json: str) -> Tuple[int, int]:
    """Best-effort extraction of image dimensions from segmentation_coords.

    Returns (height, width).  Falls back to bounding box of contour points.
    """
    try:
        data = json.loads(seg_json)
        # Some versions store image_shape directly
        if 'image_shape' in data:
            return tuple(data['image_shape'][:2])  # (h, w)
        contours = data.get('contours', [])
        if contours and contours[0]:
            pts = [(p[0], p[1]) for p in contours[0]]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            # Estimate image dims as slightly larger than contour bounds
            return int(max(ys) * 1.1) + 1, int(max(xs) * 1.1) + 1
    except Exception:
        pass
    return 1000, 1000  # safe fallback


def get_eligible_fragments(conn: sqlite3.Connection) -> List[FragmentRow]:
    """Fragments with both scale detection and segmentation data."""
    rows = conn.execute("""
        SELECT fragment_id, image_path, pixels_per_unit, scale_unit,
               segmentation_coords, line_count, script_type
        FROM fragments
        WHERE pixels_per_unit IS NOT NULL
          AND segmentation_coords IS NOT NULL
          AND segmentation_coords != ''
    """).fetchall()

    result: List[FragmentRow] = []
    for r in rows:
        h, w = _image_dims_from_seg(r[4])
        result.append(FragmentRow(
            fragment_id=r[0],
            image_path=r[1],
            pixels_per_unit=r[2],
            scale_unit=r[3],
            segmentation_coords=r[4],
            line_count=r[5],
            script_type=r[6],
            image_width=w,
            image_height=h,
        ))
    return result


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def clear_matches(conn: sqlite3.Connection, version: str) -> None:
    conn.execute("DELETE FROM edge_matches WHERE algorithm_version = ?", (version,))
    conn.commit()


def insert_matches(conn: sqlite3.Connection, matches: List[MatchRow]) -> int:
    sql = """
        INSERT INTO edge_matches
            (fragment_a_id, edge_a_name, fragment_b_id, edge_b_name,
             score, rank, confidence, score_details,
             relative_x_cm, relative_y_cm, rotation_deg, algorithm_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    rows = [
        (m.fragment_a_id, m.edge_a_name, m.fragment_b_id, m.edge_b_name,
         m.score, m.rank, m.confidence, m.score_details,
         m.relative_x_cm, m.relative_y_cm, m.rotation_deg, m.algorithm_version)
        for m in matches
    ]
    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Validation project helpers
# ---------------------------------------------------------------------------

def create_validation_project(
    conn: sqlite3.Connection,
    project_name: str,
    description: str,
) -> int:
    cur = conn.execute(
        "INSERT INTO projects (project_name, description) VALUES (?, ?)",
        (project_name, description),
    )
    conn.commit()
    return cur.lastrowid


def insert_project_fragment(
    conn: sqlite3.Connection,
    project_id: int,
    fragment_id: str,
    x: float,
    y: float,
    width: Optional[float],
    height: Optional[float],
    rotation: float = 0,
    z_index: int = 0,
    show_segmented: int = 1,
) -> None:
    conn.execute("""
        INSERT INTO project_fragments
            (project_id, fragment_id, x, y, width, height,
             rotation, scale_x, scale_y, is_locked, z_index, show_segmented)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1, 1, 0, ?, ?)
    """, (project_id, fragment_id, x, y, width, height, rotation, z_index, show_segmented))


def get_top_matches(
    conn: sqlite3.Connection,
    limit: int = 10,
) -> list:
    """Get best rank-1 matches ordered by score."""
    return conn.execute("""
        SELECT fragment_a_id, edge_a_name, fragment_b_id, edge_b_name,
               score, relative_x_cm, relative_y_cm, rotation_deg, score_details
        FROM edge_matches
        WHERE rank = 1
        ORDER BY score ASC
        LIMIT ?
    """, (limit,)).fetchall()


def get_fragment_scale(conn: sqlite3.Connection, fragment_id: str) -> Optional[Tuple[float, str]]:
    """Return (pixels_per_unit, scale_unit) or None."""
    row = conn.execute(
        "SELECT pixels_per_unit, scale_unit FROM fragments WHERE fragment_id = ?",
        (fragment_id,),
    ).fetchone()
    if row and row[0] is not None:
        return row[0], row[1]
    return None
