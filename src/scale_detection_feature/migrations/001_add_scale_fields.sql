-- Migration: Add scale detection fields to fragments table
-- Purpose: Store physical scale information (pixels per cm/mm) for auto-scaling fragments
-- Date: 2026-01-19

-- Add scale_unit column (cm, mm, or NULL if detection failed)
ALTER TABLE fragments ADD COLUMN scale_unit TEXT DEFAULT NULL;

-- Add pixels_per_unit column (ratio of pixels to physical units in original image)
ALTER TABLE fragments ADD COLUMN pixels_per_unit REAL DEFAULT NULL;

-- Add scale_detection_status column (success, error, or pending)
ALTER TABLE fragments ADD COLUMN scale_detection_status TEXT DEFAULT 'pending';

-- Add scale_model_version column (tracking which version of detection algorithm was used)
ALTER TABLE fragments ADD COLUMN scale_model_version TEXT DEFAULT NULL;

-- Create index for filtering by scale availability
CREATE INDEX IF NOT EXISTS idx_scale_unit ON fragments(scale_unit);

-- Verify migration
SELECT 'Migration completed successfully. Scale fields added to fragments table.' AS status;
