#!/usr/bin/env tsx
import Database from 'better-sqlite3';
import * as fs from 'fs';
import * as path from 'path';

const DB_PATH = './electron/resources/database/fragments.db';
const SCHEMA_SQL = `
-- Fragment metadata table
CREATE TABLE IF NOT EXISTS fragments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fragment_id TEXT UNIQUE NOT NULL,
    image_path TEXT NOT NULL,

    -- Optional metadata (NULL until ML models populate)
    edge_piece BOOLEAN,
    has_top_edge BOOLEAN,
    has_bottom_edge BOOLEAN,
    line_count INTEGER,
    script_type TEXT,
    segmentation_coords TEXT,

    -- User metadata
    notes TEXT,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fragment_id ON fragments(fragment_id);
CREATE INDEX IF NOT EXISTS idx_line_count ON fragments(line_count);
CREATE INDEX IF NOT EXISTS idx_script_type ON fragments(script_type);
CREATE INDEX IF NOT EXISTS idx_edge_piece ON fragments(edge_piece);

CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_name TEXT NOT NULL,
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_opened_at DATETIME
);

CREATE TABLE IF NOT EXISTS project_fragments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    fragment_id TEXT NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    width REAL NOT NULL,
    height REAL NOT NULL,
    rotation REAL DEFAULT 0,
    scale_x REAL DEFAULT 1,
    scale_y REAL DEFAULT 1,
    is_locked BOOLEAN DEFAULT 0,
    z_index INTEGER DEFAULT 0,

    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY (fragment_id) REFERENCES fragments(fragment_id)
);

CREATE TABLE IF NOT EXISTS project_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    content TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
`;

console.log('Creating empty database for testing...');

// Ensure directory exists
fs.mkdirSync(path.dirname(DB_PATH), { recursive: true });

// Create database
const db = new Database(DB_PATH);

// Run schema
db.exec(SCHEMA_SQL);

// Add a few test fragments so we can see something in the UI
const insert = db.prepare(`
  INSERT INTO fragments (fragment_id, image_path)
  VALUES (?, ?)
`);

// Add some sample entries (these paths won't exist yet, but that's okay for testing)
insert.run('test-fragment-1', 'BLL1/test1.jpg');
insert.run('test-fragment-2', 'BLL1/test2.jpg');
insert.run('test-fragment-3', 'BLL1/test3.jpg');

db.close();

console.log('✓ Empty database created at:', DB_PATH);
console.log('✓ Added 3 test fragments');
