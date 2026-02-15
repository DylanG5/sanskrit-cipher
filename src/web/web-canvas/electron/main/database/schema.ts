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
    has_left_edge BOOLEAN DEFAULT NULL,
    has_right_edge BOOLEAN DEFAULT NULL,
    has_circle BOOLEAN DEFAULT NULL,
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
    show_segmented BOOLEAN DEFAULT 1,

    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY (fragment_id) REFERENCES fragments(fragment_id)
);

-- Project notes: stores notes for each project
CREATE TABLE IF NOT EXISTS project_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL UNIQUE,
    content TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

-- Custom filters: user-defined metadata fields
CREATE TABLE IF NOT EXISTS custom_filters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filter_key TEXT UNIQUE NOT NULL,
    label TEXT NOT NULL,
    type TEXT NOT NULL, -- 'dropdown' or 'text'
    options TEXT,       -- JSON array for dropdown options
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Edge matches: pairwise edge match results from reconstruction pipeline
CREATE TABLE IF NOT EXISTS edge_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fragment_a_id TEXT NOT NULL,
    edge_a_name TEXT NOT NULL,
    fragment_b_id TEXT NOT NULL,
    edge_b_name TEXT NOT NULL,
    score REAL NOT NULL,
    rank INTEGER NOT NULL,
    confidence REAL,
    score_details TEXT,        -- JSON breakdown
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
`;

export default SCHEMA_SQL;
