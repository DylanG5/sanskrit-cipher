from __future__ import annotations

import sqlite3
from pathlib import Path

from ml_pipeline.core.database import DatabaseManager


def _create_test_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE fragments (
            id INTEGER PRIMARY KEY,
            fragment_id TEXT NOT NULL,
            image_path TEXT NOT NULL,
            edge_piece INTEGER,
            has_top_edge INTEGER,
            has_bottom_edge INTEGER,
            has_left_edge INTEGER,
            has_right_edge INTEGER,
            line_count INTEGER,
            script_type TEXT,
            segmentation_coords TEXT,
            notes TEXT,
            processing_status TEXT,
            segmentation_model_version TEXT,
            classification_model_version TEXT,
            last_processed_at TEXT,
            processing_error TEXT,
            has_circle INTEGER,
            scale_unit TEXT,
            pixels_per_unit REAL,
            scale_detection_status TEXT,
            scale_model_version TEXT,
            line_detection_data TEXT,
            line_detection_model_version TEXT,
            line_detection_confidence REAL,
            script_type_classification_data TEXT,
            script_type_classification_model_version TEXT,
            script_type_confidence REAL
        )
        """
    )

    conn.execute(
        "INSERT INTO fragments (id, fragment_id, image_path, processing_status) "
        "VALUES (1, 'F001', 'uploads/f001.png', 'pending')"
    )
    conn.execute(
        "INSERT INTO fragments (id, fragment_id, image_path, processing_status, last_processed_at) "
        "VALUES (2, 'F002', 'uploads/f002.png', 'completed', '2025-01-01T00:00:00')"
    )
    conn.execute(
        "INSERT INTO fragments (id, fragment_id, image_path, processing_status, last_processed_at) "
        "VALUES (3, 'F003', 'uploads/f003.png', 'failed', '2025-01-02T00:00:00')"
    )

    conn.commit()
    conn.close()


def test_get_fragments_returns_records(tmp_path: Path) -> None:
    """test-ut-db-1"""
    db_path = tmp_path / "test.db"
    _create_test_db(db_path)

    db = DatabaseManager(str(db_path))
    fragments = db.get_fragments()

    assert len(fragments) == 3
    assert fragments[0].fragment_id == "F001"


def test_update_fragment_persists_changes(tmp_path: Path) -> None:
    """test-ut-db-2"""
    db_path = tmp_path / "test.db"
    _create_test_db(db_path)

    db = DatabaseManager(str(db_path))
    db.update_fragment(1, {"line_count": 5, "processing_status": "completed"})

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT line_count, processing_status FROM fragments WHERE id = 1"
    ).fetchone()
    conn.close()

    assert row == (5, "completed")


def test_get_processing_stats(tmp_path: Path) -> None:
    """test-ut-db-3"""
    db_path = tmp_path / "test.db"
    _create_test_db(db_path)

    db = DatabaseManager(str(db_path))
    stats = db.get_processing_stats()

    assert stats["total"] == 3
    assert stats["completed"] == 1
    assert stats["pending"] == 1
    assert stats["failed"] == 1


def test_get_last_processed_id(tmp_path: Path) -> None:
    """test-ut-db-4"""
    db_path = tmp_path / "test.db"
    _create_test_db(db_path)

    db = DatabaseManager(str(db_path))
    assert db.get_last_processed_id() == 2
