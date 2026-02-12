import Database from 'better-sqlite3';
import { app } from 'electron';
import path from 'node:path';
import fs from 'node:fs';
import SCHEMA_SQL from './schema';

let db: Database.Database | null = null;

/**
 * Run database migrations to update schema for existing databases.
 * This ensures backward compatibility when adding new columns.
 */
function runMigrations(database: Database.Database): void {
  // Check if show_segmented column exists in project_fragments table
  const tableInfo = database.prepare("PRAGMA table_info(project_fragments)").all() as Array<{ name: string }>;
  const hasShowSegmentedColumn = tableInfo.some(col => col.name === 'show_segmented');

  if (!hasShowSegmentedColumn) {
    console.log('Running migration: Adding show_segmented column to project_fragments');
    try {
      // Add show_segmented column with default value of 1 (true)
      // This preserves existing behavior where segmented images were shown by default
      database.exec('ALTER TABLE project_fragments ADD COLUMN show_segmented BOOLEAN DEFAULT 1');
      console.log('Migration completed: show_segmented column added');
    } catch (error) {
      console.error('Migration failed:', error);
    }
  }

  // Ensure required fragment columns exist for newer features
  const fragmentInfo = database.prepare("PRAGMA table_info(fragments)").all() as Array<{ name: string }>;
  const fragmentColumns = new Set(fragmentInfo.map(col => col.name));

  const ensureFragmentColumn = (name: string, type: string) => {
    if (fragmentColumns.has(name)) {
      return;
    }
    try {
      database.exec(`ALTER TABLE fragments ADD COLUMN ${name} ${type}`);
      fragmentColumns.add(name);
    } catch (error) {
      console.error(`Migration failed: adding fragments.${name}`, error);
    }
  };

  ensureFragmentColumn('has_left_edge', 'BOOLEAN DEFAULT NULL');
  ensureFragmentColumn('has_right_edge', 'BOOLEAN DEFAULT NULL');
  ensureFragmentColumn('has_circle', 'BOOLEAN DEFAULT NULL');
  ensureFragmentColumn('scale_unit', 'TEXT');
  ensureFragmentColumn('pixels_per_unit', 'REAL');
  ensureFragmentColumn('scale_detection_status', 'TEXT');
  ensureFragmentColumn('scale_model_version', 'TEXT');

  // Create common indexes if columns exist
  const ensureIndex = (indexName: string, table: string, column: string) => {
    if (!fragmentColumns.has(column) && table === 'fragments') {
      return;
    }
    try {
      database.exec(`CREATE INDEX IF NOT EXISTS ${indexName} ON ${table}(${column})`);
    } catch (error) {
      console.error(`Migration failed: creating index ${indexName}`, error);
    }
  };

  ensureIndex('idx_fragment_id', 'fragments', 'fragment_id');
  ensureIndex('idx_line_count', 'fragments', 'line_count');
  ensureIndex('idx_script_type', 'fragments', 'script_type');
  ensureIndex('idx_edge_piece', 'fragments', 'edge_piece');
  ensureIndex('idx_scale_detection', 'fragments', 'scale_detection_status');
  ensureIndex('idx_has_circle', 'fragments', 'has_circle');
  ensureIndex('idx_project_fragments_project', 'project_fragments', 'project_id');
  ensureIndex('idx_custom_filters_key', 'custom_filters', 'filter_key');

  // Ensure custom filter columns exist on fragments table
  const customFilterRows = database.prepare("SELECT filter_key FROM custom_filters").all() as Array<{ filter_key: string }>;

  for (const row of customFilterRows) {
    const key = row.filter_key;
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) {
      continue;
    }
    if (!fragmentColumns.has(key)) {
      try {
        database.exec(`ALTER TABLE fragments ADD COLUMN ${key} TEXT`);
        database.exec(`CREATE INDEX IF NOT EXISTS idx_fragments_${key} ON fragments(${key})`);
        fragmentColumns.add(key);
      } catch (error) {
        console.error(`Migration failed: adding custom filter column ${key}`, error);
      }
    }
  }
}

/**
 * Get the path to the database file.
 * - In development: uses a local db in electron/resources/database/
 * - In production: copies bundled db to userData on first launch
 */
function getDatabasePath(): string {
  const isDev = !app.isPackaged;
  
  if (isDev) {
    // Development: use local database in project
    return path.join(process.cwd(), 'electron', 'resources', 'database', 'fragments.db');
  } else {
    // Production: use database in user data directory
    return path.join(app.getPath('userData'), 'fragments.db');
  }
}

/**
 * Initialize the database connection.
 * Creates tables if they don't exist.
 */
export function initDatabase(): Database.Database {
  if (db) {
    return db;
  }

  const dbPath = getDatabasePath();
  const dbDir = path.dirname(dbPath);

  // Ensure directory exists
  if (!fs.existsSync(dbDir)) {
    fs.mkdirSync(dbDir, { recursive: true });
  }

  // In production, copy bundled database if user database doesn't exist
  if (app.isPackaged && !fs.existsSync(dbPath)) {
    // extraResource copies ./electron/resources as "resources" subfolder inside Contents/Resources/
    const bundledDbPath = path.join(process.resourcesPath, 'resources', 'database', 'fragments.db');
    if (fs.existsSync(bundledDbPath)) {
      fs.copyFileSync(bundledDbPath, dbPath);
      console.log('Copied bundled database to user data directory');
    }
  }

  // Open database connection
  db = new Database(dbPath);

  // Enable foreign keys
  db.pragma('foreign_keys = ON');

  // Run schema (creates tables if they don't exist)
  db.exec(SCHEMA_SQL);

  // Run migrations for existing databases
  runMigrations(db);

  console.log(`Database initialized at: ${dbPath}`);

  return db;
}

/**
 * Get the database instance.
 * Throws if database hasn't been initialized.
 */
export function getDatabase(): Database.Database {
  if (!db) {
    throw new Error('Database not initialized. Call initDatabase() first.');
  }
  return db;
}

/**
 * Close the database connection.
 */
export function closeDatabase(): void {
  if (db) {
    db.close();
    db = null;
    console.log('Database connection closed');
  }
}

export default { initDatabase, getDatabase, closeDatabase };
