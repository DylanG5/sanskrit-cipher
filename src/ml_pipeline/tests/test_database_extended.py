"""Extended database tests – migrations, context manager, filters, edge cases."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ml_pipeline.core.database import DatabaseManager


def _create_minimal_db(db_path: Path) -> None:
    """Create a DB with only the base fragments table (pre-migration)."""
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
            line_count INTEGER,
            script_type TEXT,
            segmentation_coords TEXT,
            notes TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO fragments (id, fragment_id, image_path) VALUES (1, 'F001', 'uploads/f001.png')"
    )
    conn.execute(
        "INSERT INTO fragments (id, fragment_id, image_path) VALUES (2, 'F002', 'BLL/f002.png')"
    )
    conn.commit()
    conn.close()


def _create_full_db(db_path: Path) -> None:
    """Create fully migrated DB with sample data."""
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
            processing_status TEXT DEFAULT 'pending',
            segmentation_model_version TEXT,
            classification_model_version TEXT,
            last_processed_at DATETIME,
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
        "VALUES (2, 'F002', 'BLL/f002.png', 'completed', '2025-06-01T00:00:00')"
    )
    conn.execute(
        "INSERT INTO fragments (id, fragment_id, image_path, processing_status) "
        "VALUES (3, 'F003', 'BLX/f003.png', 'failed')"
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Database not found
# ---------------------------------------------------------------------------

def test_db_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DatabaseManager(str(tmp_path / "nonexistent.db"))


# ---------------------------------------------------------------------------
# connect / disconnect
# ---------------------------------------------------------------------------

def test_context_manager(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_full_db(db_path)

    with DatabaseManager(str(db_path)) as db:
        frags = db.get_fragments()
        assert len(frags) == 3
    # After exit, connection should be closed
    assert db.conn is None


# ---------------------------------------------------------------------------
# run_migration
# ---------------------------------------------------------------------------

def test_run_migration_adds_columns(tmp_path: Path) -> None:
    """Migration should add all processing columns to a bare table."""
    db_path = tmp_path / "test.db"
    _create_minimal_db(db_path)

    db = DatabaseManager(str(db_path))
    db.run_migration()

    # Verify columns were added
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("PRAGMA table_info(fragments)")
    col_names = {row[1] for row in cursor.fetchall()}
    conn.close()

    assert "processing_status" in col_names
    assert "has_circle" in col_names
    assert "scale_unit" in col_names
    assert "has_left_edge" in col_names
    assert "line_detection_data" in col_names
    assert "script_type_classification_data" in col_names
    db.disconnect()


def test_run_migration_idempotent(tmp_path: Path) -> None:
    """Running migration twice should not raise."""
    db_path = tmp_path / "test.db"
    _create_full_db(db_path)

    db = DatabaseManager(str(db_path))
    db.run_migration()  # already migrated, should skip
    db.run_migration()
    db.disconnect()


# ---------------------------------------------------------------------------
# get_fragments filters
# ---------------------------------------------------------------------------

def test_get_fragments_with_limit(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_full_db(db_path)
    db = DatabaseManager(str(db_path))

    frags = db.get_fragments(limit=1)
    assert len(frags) == 1
    db.disconnect()


def test_get_fragments_with_offset(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_full_db(db_path)
    db = DatabaseManager(str(db_path))

    frags = db.get_fragments(limit=1, offset=1)
    assert len(frags) == 1
    assert frags[0].fragment_id == "F002"
    db.disconnect()


def test_get_fragments_by_ids(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_full_db(db_path)
    db = DatabaseManager(str(db_path))

    frags = db.get_fragments(fragment_ids=["F002"])
    assert len(frags) == 1
    assert frags[0].fragment_id == "F002"
    db.disconnect()


def test_get_fragments_by_processing_status(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_full_db(db_path)
    db = DatabaseManager(str(db_path))

    frags = db.get_fragments(processing_status="failed")
    assert len(frags) == 1
    assert frags[0].fragment_id == "F003"
    db.disconnect()


def test_get_fragments_by_collection(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_full_db(db_path)
    db = DatabaseManager(str(db_path))

    frags = db.get_fragments(collection="BLL")
    assert len(frags) == 1
    assert frags[0].fragment_id == "F002"
    db.disconnect()


# ---------------------------------------------------------------------------
# update_fragment
# ---------------------------------------------------------------------------

def test_update_fragment_current_timestamp(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_full_db(db_path)
    db = DatabaseManager(str(db_path))

    db.update_fragment(1, {
        "processing_status": "completed",
        "last_processed_at": "CURRENT_TIMESTAMP",
    })

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT processing_status, last_processed_at FROM fragments WHERE id = 1"
    ).fetchone()
    conn.close()

    assert row[0] == "completed"
    assert row[1] is not None
    db.disconnect()


# ---------------------------------------------------------------------------
# get_processing_stats
# ---------------------------------------------------------------------------

def test_get_processing_stats_all_statuses(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_full_db(db_path)
    db = DatabaseManager(str(db_path))
    stats = db.get_processing_stats()
    assert stats["total"] == 3
    assert stats["completed"] == 1
    assert stats["pending"] == 1
    assert stats["failed"] == 1
    db.disconnect()


# ---------------------------------------------------------------------------
# get_last_processed_id
# ---------------------------------------------------------------------------

def test_get_last_processed_id_none(tmp_path: Path) -> None:
    """No completed fragments → None."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE fragments (
            id INTEGER PRIMARY KEY,
            fragment_id TEXT NOT NULL,
            image_path TEXT NOT NULL,
            processing_status TEXT,
            last_processed_at DATETIME
        )
        """
    )
    conn.execute(
        "INSERT INTO fragments (id, fragment_id, image_path, processing_status) "
        "VALUES (1, 'F001', 'a.png', 'pending')"
    )
    conn.commit()
    conn.close()

    db = DatabaseManager(str(db_path))
    assert db.get_last_processed_id() is None
    db.disconnect()


def test_get_last_processed_id_returns_id(tmp_path: Path) -> None:
    """Completed fragment → returns its id."""
    db_path = tmp_path / "test.db"
    _create_full_db(db_path)
    db = DatabaseManager(str(db_path))
    last = db.get_last_processed_id()
    assert last == 2  # F002 is the completed fragment
    db.disconnect()
