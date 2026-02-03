"""
Database layer for ML pipeline.

Handles SQLite database connections, migrations, and CRUD operations
for fragment processing.
"""

import sqlite3
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime

from ml_pipeline.core.processor import FragmentRecord


class DatabaseManager:
    """
    Manages SQLite database connections and operations.

    Provides methods for querying fragments, updating processing results,
    and tracking pipeline progress.
    """

    def __init__(self, db_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
            logger: Optional logger for database operations
        """
        self.db_path = Path(db_path)
        self.logger = logger or logging.getLogger(__name__)
        self.conn: Optional[sqlite3.Connection] = None

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

    def connect(self) -> None:
        """Establish database connection."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            self.logger.info(f"Connected to database: {self.db_path}")

    def disconnect(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.logger.info("Disconnected from database")

    def run_migration(self) -> None:
        """
        Run database migration to add processing fields.

        Adds fields needed for pipeline tracking:
        - processing_status
        - segmentation_model_version
        - classification_model_version
        - last_processed_at
        - processing_error
        """
        self.connect()

        cursor = self.conn.cursor()

        # Check if migration already ran by testing for processing_status column
        cursor.execute("PRAGMA table_info(fragments)")
        columns = [row[1] for row in cursor.fetchall()]

        migration_needed = False

        if 'processing_status' not in columns:
            migration_needed = True
            self.logger.info("Running database migration for processing fields...")

            try:
                # Add processing columns
                cursor.execute("ALTER TABLE fragments ADD COLUMN processing_status TEXT DEFAULT 'pending'")
                cursor.execute("ALTER TABLE fragments ADD COLUMN segmentation_model_version TEXT")
                cursor.execute("ALTER TABLE fragments ADD COLUMN classification_model_version TEXT")
                cursor.execute("ALTER TABLE fragments ADD COLUMN last_processed_at DATETIME")
                cursor.execute("ALTER TABLE fragments ADD COLUMN processing_error TEXT")

                # Create index for processing_status
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_processing_status ON fragments(processing_status)")

                self.conn.commit()
                self.logger.info("Processing fields migration completed successfully")
            except sqlite3.Error as e:
                self.conn.rollback()
                self.logger.error(f"Processing fields migration failed: {e}")
                raise

        # Check if scale columns exist
        cursor.execute("PRAGMA table_info(fragments)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'scale_unit' not in columns:
            migration_needed = True
            self.logger.info("Running database migration for scale detection fields...")

            try:
                # Add scale detection columns
                cursor.execute("ALTER TABLE fragments ADD COLUMN scale_unit TEXT")
                cursor.execute("ALTER TABLE fragments ADD COLUMN pixels_per_unit REAL")
                cursor.execute("ALTER TABLE fragments ADD COLUMN scale_detection_status TEXT")
                cursor.execute("ALTER TABLE fragments ADD COLUMN scale_model_version TEXT")

                # Create index for scale_detection_status
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_scale_detection ON fragments(scale_detection_status)")

                self.conn.commit()
                self.logger.info("Scale detection fields migration completed successfully")
            except sqlite3.Error as e:
                self.conn.rollback()
                self.logger.error(f"Scale detection fields migration failed: {e}")
                raise

        # Check if edge detection columns exist
        cursor.execute("PRAGMA table_info(fragments)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'has_left_edge' not in columns:
            migration_needed = True
            self.logger.info("Running database migration for edge detection fields...")

            try:
                # Add left/right edge columns
                cursor.execute("ALTER TABLE fragments ADD COLUMN has_left_edge BOOLEAN DEFAULT NULL")
                cursor.execute("ALTER TABLE fragments ADD COLUMN has_right_edge BOOLEAN DEFAULT NULL")

                self.conn.commit()
                self.logger.info("Edge detection fields migration completed successfully")
            except sqlite3.Error as e:
                self.conn.rollback()
                self.logger.error(f"Edge detection fields migration failed: {e}")
                raise

        if not migration_needed:
            self.logger.info("Migration already applied, skipping")

    def get_fragments(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        fragment_ids: Optional[List[str]] = None,
        processing_status: Optional[str] = None
    ) -> List[FragmentRecord]:
        """
        Query fragments from database.

        Args:
            limit: Maximum number of fragments to return
            offset: Number of fragments to skip
            fragment_ids: Optional list of specific fragment IDs to fetch
            processing_status: Optional filter by processing status

        Returns:
            List of FragmentRecord objects
        """
        self.connect()

        query = "SELECT * FROM fragments WHERE 1=1"
        params: List[Any] = []

        if fragment_ids:
            placeholders = ','.join('?' * len(fragment_ids))
            query += f" AND fragment_id IN ({placeholders})"
            params.extend(fragment_ids)

        if processing_status:
            query += " AND processing_status = ?"
            params.append(processing_status)

        query += " ORDER BY id"

        if limit:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        cursor = self.conn.cursor()
        cursor.execute(query, params)

        fragments = []
        for row in cursor.fetchall():
            # Convert row to dict to safely access potentially missing columns
            row_dict = dict(row)
            fragments.append(FragmentRecord(
                id=row_dict['id'],
                fragment_id=row_dict['fragment_id'],
                image_path=row_dict['image_path'],
                edge_piece=row_dict.get('edge_piece'),
                has_top_edge=row_dict.get('has_top_edge'),
                has_bottom_edge=row_dict.get('has_bottom_edge'),
                has_left_edge=row_dict.get('has_left_edge'),
                has_right_edge=row_dict.get('has_right_edge'),
                line_count=row_dict.get('line_count'),
                script_type=row_dict.get('script_type'),
                segmentation_coords=row_dict.get('segmentation_coords'),
                notes=row_dict.get('notes'),
                processing_status=row_dict.get('processing_status'),
                segmentation_model_version=row_dict.get('segmentation_model_version'),
                classification_model_version=row_dict.get('classification_model_version'),
                last_processed_at=row_dict.get('last_processed_at'),
                processing_error=row_dict.get('processing_error'),
                scale_unit=row_dict.get('scale_unit'),
                pixels_per_unit=row_dict.get('pixels_per_unit'),
                scale_detection_status=row_dict.get('scale_detection_status'),
                scale_model_version=row_dict.get('scale_model_version')
            ))

        return fragments

    def update_fragment(self, fragment_id: int, updates: Dict[str, Any]) -> None:
        """
        Update fragment fields.

        Args:
            fragment_id: Fragment database ID
            updates: Dictionary of field names to values

        Example:
            db.update_fragment(123, {
                'line_count': 5,
                'processing_status': 'completed',
                'last_processed_at': 'CURRENT_TIMESTAMP'
            })
        """
        if not updates:
            return

        self.connect()

        # Build UPDATE query
        set_clauses = []
        params = []

        for field, value in updates.items():
            if value == 'CURRENT_TIMESTAMP':
                set_clauses.append(f"{field} = CURRENT_TIMESTAMP")
            else:
                set_clauses.append(f"{field} = ?")
                params.append(value)

        params.append(fragment_id)

        query = f"UPDATE fragments SET {', '.join(set_clauses)} WHERE id = ?"

        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            self.conn.commit()
        except sqlite3.Error as e:
            self.conn.rollback()
            self.logger.error(f"Failed to update fragment {fragment_id}: {e}")
            raise

    def get_processing_stats(self) -> Dict[str, int]:
        """
        Get processing statistics.

        Returns:
            Dictionary with counts:
            - total: Total fragments
            - completed: Successfully processed
            - pending: Not yet processed
            - failed: Processing failed
        """
        self.connect()

        cursor = self.conn.cursor()

        # Total count
        cursor.execute("SELECT COUNT(*) FROM fragments")
        total = cursor.fetchone()[0]

        # Completed count
        cursor.execute("SELECT COUNT(*) FROM fragments WHERE processing_status = 'completed'")
        completed = cursor.fetchone()[0]

        # Pending count
        cursor.execute("SELECT COUNT(*) FROM fragments WHERE processing_status = 'pending' OR processing_status IS NULL")
        pending = cursor.fetchone()[0]

        # Failed count
        cursor.execute("SELECT COUNT(*) FROM fragments WHERE processing_status = 'failed'")
        failed = cursor.fetchone()[0]

        return {
            'total': total,
            'completed': completed,
            'pending': pending,
            'failed': failed
        }

    def get_last_processed_id(self) -> Optional[int]:
        """
        Get the ID of the last successfully processed fragment.

        Returns:
            Fragment ID or None if no fragments processed
        """
        self.connect()

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id FROM fragments
            WHERE processing_status = 'completed'
            ORDER BY last_processed_at DESC
            LIMIT 1
        """)

        row = cursor.fetchone()
        return row[0] if row else None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
