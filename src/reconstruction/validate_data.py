"""
Validate fragment metadata integrity in the SQLite database.

Checks:
  1) Required fields are present and non-null (fragment_id, image_path, created_at).
  2) Image files exist on disk (relative to data root) and are non-empty files.
  3) Referential integrity (no orphaned records in tables that reference fragments).

Usage:
  python reconstruction/validate_data.py
  python reconstruction/validate_data.py --db-path path/to/fragments.db --data-root path/to/data --report report.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List


def _resolve_path(data_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return data_root / path


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def validate(
    db_path: Path,
    data_root: Path,
) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    missing_required_fields: List[Dict[str, Any]] = []
    missing_images: List[Dict[str, Any]] = []
    duplicate_fragment_ids: List[Dict[str, Any]] = []
    orphaned_records: List[Dict[str, Any]] = []

    fragment_rows = conn.execute(
        "SELECT id, fragment_id, image_path, created_at FROM fragments"
    ).fetchall()

    for row in fragment_rows:
        issues: List[str] = []
        fragment_id = row["fragment_id"]
        image_path = row["image_path"]
        created_at = row["created_at"]

        if fragment_id is None or str(fragment_id).strip() == "":
            issues.append("fragment_id")
        if image_path is None or str(image_path).strip() == "":
            issues.append("image_path")
        if created_at is None or str(created_at).strip() == "":
            issues.append("created_at")

        if issues:
            missing_required_fields.append(
                {
                    "id": row["id"],
                    "fragment_id": fragment_id,
                    "missing": issues,
                }
            )

        if image_path is not None and str(image_path).strip() != "":
            resolved = _resolve_path(data_root, str(image_path))
            if not resolved.exists():
                missing_images.append(
                    {
                        "id": row["id"],
                        "fragment_id": fragment_id,
                        "image_path": str(image_path),
                        "resolved_path": str(resolved),
                        "reason": "missing",
                    }
                )
            elif resolved.is_dir():
                missing_images.append(
                    {
                        "id": row["id"],
                        "fragment_id": fragment_id,
                        "image_path": str(image_path),
                        "resolved_path": str(resolved),
                        "reason": "is_directory",
                    }
                )
            else:
                try:
                    if resolved.stat().st_size == 0:
                        missing_images.append(
                            {
                                "id": row["id"],
                                "fragment_id": fragment_id,
                                "image_path": str(image_path),
                                "resolved_path": str(resolved),
                                "reason": "empty_file",
                            }
                        )
                except OSError as exc:
                    missing_images.append(
                        {
                            "id": row["id"],
                            "fragment_id": fragment_id,
                            "image_path": str(image_path),
                            "resolved_path": str(resolved),
                            "reason": f"stat_error: {exc}",
                        }
                    )

    duplicate_rows = conn.execute(
        """
        SELECT fragment_id, COUNT(*) AS count
        FROM fragments
        GROUP BY fragment_id
        HAVING COUNT(*) > 1
        """
    ).fetchall()
    for row in duplicate_rows:
        duplicate_fragment_ids.append(
            {"fragment_id": row["fragment_id"], "count": row["count"]}
        )

    # Referential integrity checks for tables that reference fragments
    if _table_exists(conn, "project_fragments"):
        rows = conn.execute(
            """
            SELECT pf.id, pf.fragment_id
            FROM project_fragments pf
            LEFT JOIN fragments f ON f.fragment_id = pf.fragment_id
            WHERE f.fragment_id IS NULL
            """
        ).fetchall()
        for row in rows:
            orphaned_records.append(
                {
                    "table": "project_fragments",
                    "column": "fragment_id",
                    "id": row["id"],
                    "value": row["fragment_id"],
                }
            )

    if _table_exists(conn, "edge_matches"):
        rows = conn.execute(
            """
            SELECT em.id, em.fragment_a_id AS fragment_id
            FROM edge_matches em
            LEFT JOIN fragments f ON f.fragment_id = em.fragment_a_id
            WHERE f.fragment_id IS NULL
            """
        ).fetchall()
        for row in rows:
            orphaned_records.append(
                {
                    "table": "edge_matches",
                    "column": "fragment_a_id",
                    "id": row["id"],
                    "value": row["fragment_id"],
                }
            )

        rows = conn.execute(
            """
            SELECT em.id, em.fragment_b_id AS fragment_id
            FROM edge_matches em
            LEFT JOIN fragments f ON f.fragment_id = em.fragment_b_id
            WHERE f.fragment_id IS NULL
            """
        ).fetchall()
        for row in rows:
            orphaned_records.append(
                {
                    "table": "edge_matches",
                    "column": "fragment_b_id",
                    "id": row["id"],
                    "value": row["fragment_id"],
                }
            )

    conn.close()

    report = {
        "summary": {
            "total_fragments": len(fragment_rows),
            "missing_required_fields": len(missing_required_fields),
            "missing_images": len(missing_images),
            "duplicate_fragment_ids": len(duplicate_fragment_ids),
            "orphaned_records": len(orphaned_records),
        },
        "missing_required_fields": missing_required_fields,
        "missing_images": missing_images,
        "duplicate_fragment_ids": duplicate_fragment_ids,
        "orphaned_records": orphaned_records,
    }
    return report


def main() -> None:
    script_root = Path(__file__).resolve().parents[1]
    default_db = script_root / "web" / "web-canvas" / "electron" / "resources" / "database" / "fragments.db"
    default_data_root = script_root / "web" / "web-canvas" / "data"
    default_report = Path.cwd() / "validation_report.json"

    parser = argparse.ArgumentParser(description="Validate fragment data integrity.")
    parser.add_argument("--db-path", type=Path, default=default_db, help="Path to fragments.db")
    parser.add_argument("--data-root", type=Path, default=default_data_root, help="Root folder for fragment images")
    parser.add_argument("--report", type=Path, default=default_report, help="Output report JSON path")
    args = parser.parse_args()

    report = validate(args.db_path, args.data_root)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report["summary"]
    print("[validate-data] Summary")
    print(f"  Total fragments:        {summary['total_fragments']}")
    print(f"  Missing required fields:{summary['missing_required_fields']}")
    print(f"  Missing images:         {summary['missing_images']}")
    print(f"  Duplicate fragment IDs: {summary['duplicate_fragment_ids']}")
    print(f"  Orphaned records:       {summary['orphaned_records']}")
    print(f"[validate-data] Report written to: {args.report}")

    failures = (
        summary["missing_required_fields"]
        + summary["missing_images"]
        + summary["duplicate_fragment_ids"]
        + summary["orphaned_records"]
    )
    if failures:
        print("[validate-data] FAILED: data integrity issues found.")
        sys.exit(1)
    print("[validate-data] PASSED: data integrity checks complete.")


if __name__ == "__main__":
    main()
