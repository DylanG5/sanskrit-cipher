/**
 * IPC Handlers for Electron main process
 *
 * These handlers bridge the React frontend with the SQLite database.
 */

import { ipcMain, app } from 'electron';
import path from 'node:path';
import { getDatabase } from '../database/connection';

// Types for filter parameters
interface FragmentFilters {
  lineCountMin?: number;
  lineCountMax?: number;
  scripts?: string[];
  isEdgePiece?: boolean | null;
  search?: string;
  limit?: number;
  offset?: number;
}

interface CanvasFragment {
  fragmentId: string;
  x: number;
  y: number;
  width?: number;
  height?: number;
  rotation: number;
  scaleX: number;
  scaleY: number;
  isLocked: boolean;
  zIndex?: number;
}

interface CanvasState {
  fragments: CanvasFragment[];
}

/**
 * Register all IPC handlers
 */
export function registerIpcHandlers(): void {
  const db = getDatabase();

  // ============================================
  // Fragment Handlers
  // ============================================

  /**
   * Get all fragments with optional filtering
   */
  ipcMain.handle('fragments:getAll', async (_event, filters?: FragmentFilters) => {
    let query = 'SELECT * FROM fragments WHERE 1=1';
    const params: (string | number)[] = [];

    if (filters?.lineCountMin !== undefined && filters.lineCountMin !== null) {
      query += ' AND line_count >= ?';
      params.push(filters.lineCountMin);
    }

    if (filters?.lineCountMax !== undefined && filters.lineCountMax !== null) {
      query += ' AND line_count <= ?';
      params.push(filters.lineCountMax);
    }

    if (filters?.scripts && filters.scripts.length > 0) {
      const placeholders = filters.scripts.map(() => '?').join(',');
      query += ` AND script_type IN (${placeholders})`;
      params.push(...filters.scripts);
    }

    if (filters?.isEdgePiece !== undefined && filters.isEdgePiece !== null) {
      query += ' AND edge_piece = ?';
      params.push(filters.isEdgePiece ? 1 : 0);
    }

    if (filters?.search) {
      query += ' AND fragment_id LIKE ?';
      params.push(`%${filters.search}%`);
    }

    // Add ordering
    query += ' ORDER BY fragment_id ASC';

    // Add pagination
    if (filters?.limit) {
      query += ' LIMIT ?';
      params.push(filters.limit);

      if (filters?.offset) {
        query += ' OFFSET ?';
        params.push(filters.offset);
      }
    }

    try {
      const rows = db.prepare(query).all(...params);
      return { success: true, data: rows };
    } catch (error) {
      console.error('Error fetching fragments:', error);
      return { success: false, error: String(error) };
    }
  });

  /**
   * Get total count of fragments (for pagination)
   */
  ipcMain.handle('fragments:getCount', async (_event, filters?: FragmentFilters) => {
    let query = 'SELECT COUNT(*) as count FROM fragments WHERE 1=1';
    const params: (string | number)[] = [];

    if (filters?.lineCountMin !== undefined && filters.lineCountMin !== null) {
      query += ' AND line_count >= ?';
      params.push(filters.lineCountMin);
    }

    if (filters?.lineCountMax !== undefined && filters.lineCountMax !== null) {
      query += ' AND line_count <= ?';
      params.push(filters.lineCountMax);
    }

    if (filters?.scripts && filters.scripts.length > 0) {
      const placeholders = filters.scripts.map(() => '?').join(',');
      query += ` AND script_type IN (${placeholders})`;
      params.push(...filters.scripts);
    }

    if (filters?.isEdgePiece !== undefined && filters.isEdgePiece !== null) {
      query += ' AND edge_piece = ?';
      params.push(filters.isEdgePiece ? 1 : 0);
    }

    if (filters?.search) {
      query += ' AND fragment_id LIKE ?';
      params.push(`%${filters.search}%`);
    }

    try {
      const row = db.prepare(query).get(...params) as { count: number };
      return { success: true, count: row.count };
    } catch (error) {
      console.error('Error counting fragments:', error);
      return { success: false, error: String(error) };
    }
  });

  /**
   * Get a single fragment by ID
   */
  ipcMain.handle('fragments:getById', async (_event, fragmentId: string) => {
    try {
      const row = db.prepare('SELECT * FROM fragments WHERE fragment_id = ?').get(fragmentId);
      return { success: true, data: row || null };
    } catch (error) {
      console.error('Error fetching fragment:', error);
      return { success: false, error: String(error) };
    }
  });

  /**
   * Update fragment metadata (notes, etc.)
   */
  ipcMain.handle('fragments:updateMetadata', async (_event, fragmentId: string, metadata: Record<string, unknown>) => {
    const allowedFields = ['notes', 'edge_piece', 'has_top_edge', 'has_bottom_edge', 'line_count', 'script_type'];
    const updates: string[] = [];
    const params: unknown[] = [];

    for (const [key, value] of Object.entries(metadata)) {
      if (allowedFields.includes(key)) {
        updates.push(`${key} = ?`);
        params.push(value);
      }
    }

    if (updates.length === 0) {
      return { success: false, error: 'No valid fields to update' };
    }

    updates.push('updated_at = CURRENT_TIMESTAMP');
    params.push(fragmentId);

    const query = `UPDATE fragments SET ${updates.join(', ')} WHERE fragment_id = ?`;

    try {
      const result = db.prepare(query).run(...params);
      return { success: true, changes: result.changes };
    } catch (error) {
      console.error('Error updating fragment:', error);
      return { success: false, error: String(error) };
    }
  });

  // ============================================
  // Image Path Handler
  // ============================================

  /**
   * Resolve image path for loading
   */
  ipcMain.handle('images:getPath', async (_event, relativePath: string) => {
    const isDev = !app.isPackaged;
    const basePath = isDev
      ? path.join(process.cwd(), 'data')
      : path.join(process.resourcesPath, 'data');

    return path.join(basePath, relativePath);
  });

  // ============================================
  // Project Handlers
  // ============================================

  /**
   * List all projects
   */
  ipcMain.handle('projects:list', async () => {
    try {
      const rows = db.prepare('SELECT * FROM projects ORDER BY updated_at DESC').all();
      return { success: true, data: rows };
    } catch (error) {
      console.error('Error listing projects:', error);
      return { success: false, error: String(error) };
    }
  });

  /**
   * Create a new project
   */
  ipcMain.handle('projects:create', async (_event, name: string, description?: string) => {
    try {
      const result = db.prepare(
        'INSERT INTO projects (project_name, description) VALUES (?, ?)'
      ).run(name, description || '');
      return { success: true, projectId: result.lastInsertRowid };
    } catch (error) {
      console.error('Error creating project:', error);
      return { success: false, error: String(error) };
    }
  });

  /**
   * Save project canvas state and notes
   */
  ipcMain.handle('projects:save', async (_event, projectId: number, canvasState: CanvasState, notes: string) => {
    try {
      // Start transaction
      const saveTransaction = db.transaction(() => {
        // Update project timestamp
        db.prepare('UPDATE projects SET updated_at = CURRENT_TIMESTAMP WHERE id = ?').run(projectId);

        // Delete existing fragments for this project
        db.prepare('DELETE FROM project_fragments WHERE project_id = ?').run(projectId);

        // Insert canvas fragments
        const insertFragment = db.prepare(`
          INSERT INTO project_fragments (
            project_id, fragment_id, x, y, width, height, rotation, scale_x, scale_y, is_locked, z_index
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `);

        for (const frag of canvasState.fragments) {
          insertFragment.run(
            projectId,
            frag.fragmentId,
            frag.x,
            frag.y,
            frag.width || null,
            frag.height || null,
            frag.rotation || 0,
            frag.scaleX || 1,
            frag.scaleY || 1,
            frag.isLocked ? 1 : 0,
            frag.zIndex || 0
          );
        }

        // Upsert project notes
        db.prepare(`
          INSERT INTO project_notes (project_id, content, updated_at)
          VALUES (?, ?, CURRENT_TIMESTAMP)
          ON CONFLICT(project_id) DO UPDATE SET content = ?, updated_at = CURRENT_TIMESTAMP
        `).run(projectId, notes, notes);
      });

      saveTransaction();
      return { success: true };
    } catch (error) {
      console.error('Error saving project:', error);
      return { success: false, error: String(error) };
    }
  });

  /**
   * Load a project's canvas state and notes
   */
  ipcMain.handle('projects:load', async (_event, projectId: number) => {
    try {
      // Get project info
      const project = db.prepare('SELECT * FROM projects WHERE id = ?').get(projectId);
      if (!project) {
        return { success: false, error: 'Project not found' };
      }

      // Get canvas fragments
      const fragments = db.prepare(`
        SELECT
          fragment_id as fragmentId,
          x, y, width, height, rotation,
          scale_x as scaleX, scale_y as scaleY,
          is_locked as isLocked, z_index as zIndex
        FROM project_fragments
        WHERE project_id = ?
        ORDER BY z_index ASC
      `).all(projectId);

      // Get notes
      const notesRow = db.prepare(
        'SELECT content FROM project_notes WHERE project_id = ?'
      ).get(projectId) as { content: string } | undefined;

      return {
        success: true,
        data: {
          project,
          canvasState: { fragments },
          notes: notesRow?.content || '',
        },
      };
    } catch (error) {
      console.error('Error loading project:', error);
      return { success: false, error: String(error) };
    }
  });

  /**
   * Delete a project
   */
  ipcMain.handle('projects:delete', async (_event, projectId: number) => {
    try {
      const result = db.prepare('DELETE FROM projects WHERE id = ?').run(projectId);
      return { success: true, deleted: result.changes > 0 };
    } catch (error) {
      console.error('Error deleting project:', error);
      return { success: false, error: String(error) };
    }
  });

  console.log('IPC handlers registered');
}

export default { registerIpcHandlers };
