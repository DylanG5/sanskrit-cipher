import Database from 'better-sqlite3';
import { app } from 'electron';
import path from 'node:path';
import fs from 'node:fs';
import SCHEMA_SQL from './schema';

let db: Database.Database | null = null;

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
    const bundledDbPath = path.join(process.resourcesPath, 'database', 'fragments.db');
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
