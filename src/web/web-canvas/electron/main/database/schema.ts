// Database schema for Sanskrit Cipher fragments

export const SCHEMA_SQL = `
-- Fragments table: stores metadata for each manuscript fragment image
CREATE TABLE IF NOT EXISTS fragments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fragment_id TEXT UNIQUE NOT NULL,
    image_path TEXT NOT NULL,              -- Relative path from data/ folder
    
    -- Classification metadata (populated by ML models later)
    edge_piece BOOLEAN DEFAULT 0,
    has_top_edge BOOLEAN DEFAULT 0,
    has_bottom_edge BOOLEAN DEFAULT 0,
    line_count INTEGER,
    script_type TEXT,
    segmentation_coords TEXT,              -- JSON: polygon coordinates

    -- Scale detection metadata (populated by ML models)
    scale_unit TEXT,                       -- 'cm' or 'mm'
    pixels_per_unit REAL,                  -- Pixels per physical unit
    scale_detection_status TEXT,           -- 'success' or 'error: message'
    scale_model_version TEXT,              -- Version of scale detection model used

    -- User-editable metadata
    notes TEXT,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_fragment_id ON fragments(fragment_id);
CREATE INDEX IF NOT EXISTS idx_line_count ON fragments(line_count);
CREATE INDEX IF NOT EXISTS idx_script_type ON fragments(script_type);
CREATE INDEX IF NOT EXISTS idx_edge_piece ON fragments(edge_piece);
CREATE INDEX IF NOT EXISTS idx_scale_detection ON fragments(scale_detection_status);

-- Projects table: stores saved canvas projects
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_name TEXT NOT NULL,
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Project fragments: stores fragment positions on canvas for each project
CREATE TABLE IF NOT EXISTS project_fragments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    fragment_id TEXT NOT NULL,
    x REAL NOT NULL DEFAULT 0,
    y REAL NOT NULL DEFAULT 0,
    width REAL,
    height REAL,
    rotation REAL DEFAULT 0,
    scale_x REAL DEFAULT 1,
    scale_y REAL DEFAULT 1,
    is_locked BOOLEAN DEFAULT 0,
    z_index INTEGER DEFAULT 0,
    
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY (fragment_id) REFERENCES fragments(fragment_id)
);

CREATE INDEX IF NOT EXISTS idx_project_fragments_project ON project_fragments(project_id);

-- Project notes: stores notes for each project
CREATE TABLE IF NOT EXISTS project_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL UNIQUE,
    content TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
`;

export default SCHEMA_SQL;
