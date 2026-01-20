/**
 * Database Seeding Script
 * 
 * Scans the data/ directory for all image files and populates the fragments table.
 * Run with: npm run seed:database
 */

import Database from 'better-sqlite3';
import fs from 'node:fs';
import path from 'node:path';
import { glob } from 'glob';

// Paths
const PROJECT_ROOT = process.cwd();
const DATA_DIR = path.join(PROJECT_ROOT, 'data');
const DB_DIR = path.join(PROJECT_ROOT, 'electron', 'resources', 'database');
const DB_PATH = path.join(DB_DIR, 'fragments.db');

// Schema SQL (duplicated here to avoid import issues with tsx)
const SCHEMA_SQL = `
CREATE TABLE IF NOT EXISTS fragments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fragment_id TEXT UNIQUE NOT NULL,
    image_path TEXT NOT NULL,
    edge_piece BOOLEAN DEFAULT 0,
    has_top_edge BOOLEAN DEFAULT 0,
    has_bottom_edge BOOLEAN DEFAULT 0,
    line_count INTEGER,
    script_type TEXT,
    segmentation_coords TEXT,
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
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

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

CREATE TABLE IF NOT EXISTS project_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL UNIQUE,
    content TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);
`;

async function seedDatabase() {
  console.log('Starting database seeding...');
  console.log(`Data directory: ${DATA_DIR}`);
  console.log(`Database path: ${DB_PATH}`);

  // Check if data directory exists
  if (!fs.existsSync(DATA_DIR)) {
    console.error(`Error: Data directory not found at ${DATA_DIR}`);
    process.exit(1);
  }

  // Create database directory if it doesn't exist
  if (!fs.existsSync(DB_DIR)) {
    fs.mkdirSync(DB_DIR, { recursive: true });
    console.log(`Created database directory: ${DB_DIR}`);
  }

  // Remove existing database to start fresh
  if (fs.existsSync(DB_PATH)) {
    fs.unlinkSync(DB_PATH);
    console.log('Removed existing database');
  }

  // Create database and run schema
  const db = new Database(DB_PATH);
  db.pragma('foreign_keys = ON');
  db.exec(SCHEMA_SQL);
  console.log('Created database schema');

  // Find all image files
  console.log('Scanning for images...');
  const images = await glob('**/*.{jpg,jpeg,png,JPG,JPEG,PNG}', { 
    cwd: DATA_DIR,
    nodir: true 
  });
  
  console.log(`Found ${images.length} images`);

  if (images.length === 0) {
    console.warn('Warning: No images found in data directory');
    db.close();
    return;
  }

  // Prepare insert statement
  const insert = db.prepare(`
    INSERT INTO fragments (fragment_id, image_path)
    VALUES (?, ?)
  `);

  // Insert in batches using transactions for better performance
  const BATCH_SIZE = 1000;
  let inserted = 0;
  let skipped = 0;

  const insertBatch = db.transaction((batch: { id: string; path: string }[]) => {
    for (const item of batch) {
      try {
        insert.run(item.id, item.path);
        inserted++;
      } catch (error: any) {
        if (error.code === 'SQLITE_CONSTRAINT_UNIQUE') {
          skipped++;
        } else {
          throw error;
        }
      }
    }
  });

  // Process images in batches
  for (let i = 0; i < images.length; i += BATCH_SIZE) {
    const batch = images.slice(i, i + BATCH_SIZE).map(imgPath => {
      // Extract fragment ID from filename (without extension)
      const filename = path.basename(imgPath);
      const fragmentId = filename.replace(/\.(jpg|jpeg|png)$/i, '');
      
      return {
        id: fragmentId,
        path: imgPath,  // Relative path from data/
      };
    });

    insertBatch(batch);
    
    const progress = Math.min(i + BATCH_SIZE, images.length);
    console.log(`Processed ${progress}/${images.length} images...`);
  }

  // Close database
  db.close();

  // Report results
  console.log('\n=== Seeding Complete ===');
  console.log(`Total images found: ${images.length}`);
  console.log(`Inserted: ${inserted}`);
  console.log(`Skipped (duplicates): ${skipped}`);
  console.log(`Database size: ${(fs.statSync(DB_PATH).size / 1024 / 1024).toFixed(2)} MB`);
  console.log(`Database location: ${DB_PATH}`);
}

// Run the seeding
seedDatabase().catch((error) => {
  console.error('Seeding failed:', error);
  process.exit(1);
});
