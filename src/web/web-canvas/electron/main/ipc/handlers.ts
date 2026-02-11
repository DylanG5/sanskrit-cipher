/**
 * IPC Handlers for Electron main process
 *
 * These handlers bridge the React frontend with the SQLite database.
 */

import { ipcMain, app, dialog } from 'electron';
import path from 'node:path';
import fs from 'node:fs';
import { getDatabase } from '../database/connection';

// Types for filter parameters
interface FragmentFilters {
  lineCountMin?: number;
  lineCountMax?: number;
  scripts?: string[];
  isEdgePiece?: boolean | null;
  hasTopEdge?: boolean | null;
  hasBottomEdge?: boolean | null;
  hasLeftEdge?: boolean | null;
  hasRightEdge?: boolean | null;
  hasCircle?: boolean | null;
  search?: string;
  custom?: Record<string, string | string[] | null | undefined>;
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
  showSegmented?: boolean;
}

interface CanvasState {
  fragments: CanvasFragment[];
}

interface CustomFilterDefinition {
  id: number;
  filterKey: string;
  label: string;
  type: 'multiselect' | 'text';
  options?: string[];
}

const VALID_IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*$/;

function normalizeFilterKey(label: string): string {
  const base = label
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
  if (!base) {
    return 'custom';
  }
  return /^[A-Za-z_]/.test(base) ? base : `f_${base}`;
}

function parseOptions(input: unknown): string[] {
  if (!Array.isArray(input)) {
    return [];
  }
  const seen = new Set<string>();
  const options: string[] = [];
  for (const item of input) {
    const value = String(item).trim();
    if (!value || seen.has(value)) {
      continue;
    }
    seen.add(value);
    options.push(value);
  }
  return options;
}

function parseOptionsJson(value: unknown): string[] {
  if (!value) {
    return [];
  }
  try {
    const parsed = JSON.parse(String(value));
    return parseOptions(parsed);
  } catch {
    return [];
  }
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

    if (filters?.hasTopEdge !== undefined && filters.hasTopEdge !== null) {
      query += ' AND has_top_edge = ?';
      params.push(filters.hasTopEdge ? 1 : 0);
    }

    if (filters?.hasBottomEdge !== undefined && filters.hasBottomEdge !== null) {
      query += ' AND has_bottom_edge = ?';
      params.push(filters.hasBottomEdge ? 1 : 0);
    }

    if (filters?.hasLeftEdge !== undefined && filters.hasLeftEdge !== null) {
      query += ' AND has_left_edge = ?';
      params.push(filters.hasLeftEdge ? 1 : 0);
    }

    if (filters?.hasRightEdge !== undefined && filters.hasRightEdge !== null) {
      query += ' AND has_right_edge = ?';
      params.push(filters.hasRightEdge ? 1 : 0);
    }

    if (filters?.hasCircle !== undefined && filters.hasCircle !== null) {
      query += ' AND has_circle = ?';
      params.push(filters.hasCircle ? 1 : 0);
    }

    if (filters?.search) {
      query += ' AND fragment_id LIKE ?';
      params.push(`%${filters.search}%`);
      console.log('Search query:', filters.search);
      console.log('SQL LIKE pattern:', `%${filters.search}%`);
      console.log('Full SQL query:', query);
    }

    if (filters?.custom) {
      const customFilterDefs = db.prepare('SELECT filter_key, type FROM custom_filters').all() as Array<{ filter_key: string; type: string }>;
      const allowedKeys = new Map(customFilterDefs.map((row) => [row.filter_key, row.type]));

      for (const [key, value] of Object.entries(filters.custom)) {
        if (!allowedKeys.has(key) || !VALID_IDENTIFIER.test(key)) {
          continue;
        }

        const filterType = allowedKeys.get(key);

        // Handle array values for multiselect filters (OR condition)
        if (Array.isArray(value)) {
          if (value.length === 0) continue;

          // For multiselect filters, match if the fragment value equals ANY of the selected options
          const conditions = value.map(() => `${key} = ?`).join(' OR ');
          query += ` AND (${conditions})`;
          params.push(...value);
        } else if (typeof value === 'string' && value !== '') {
          // Handle single string values for text filters
          query += ` AND ${key} = ?`;
          params.push(value);
        }
      }
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

    if (filters?.hasTopEdge !== undefined && filters.hasTopEdge !== null) {
      query += ' AND has_top_edge = ?';
      params.push(filters.hasTopEdge ? 1 : 0);
    }

    if (filters?.hasBottomEdge !== undefined && filters.hasBottomEdge !== null) {
      query += ' AND has_bottom_edge = ?';
      params.push(filters.hasBottomEdge ? 1 : 0);
    }

    if (filters?.hasLeftEdge !== undefined && filters.hasLeftEdge !== null) {
      query += ' AND has_left_edge = ?';
      params.push(filters.hasLeftEdge ? 1 : 0);
    }

    if (filters?.hasRightEdge !== undefined && filters.hasRightEdge !== null) {
      query += ' AND has_right_edge = ?';
      params.push(filters.hasRightEdge ? 1 : 0);
    }

    if (filters?.hasCircle !== undefined && filters.hasCircle !== null) {
      query += ' AND has_circle = ?';
      params.push(filters.hasCircle ? 1 : 0);
    }

    if (filters?.search) {
      query += ' AND fragment_id LIKE ?';
      params.push(`%${filters.search}%`);
    }

    if (filters?.custom) {
      const customFilterDefs = db.prepare('SELECT filter_key, type FROM custom_filters').all() as Array<{ filter_key: string; type: string }>;
      const allowedKeys = new Map(customFilterDefs.map((row) => [row.filter_key, row.type]));

      for (const [key, value] of Object.entries(filters.custom)) {
        if (!allowedKeys.has(key) || !VALID_IDENTIFIER.test(key)) {
          continue;
        }

        const filterType = allowedKeys.get(key);

        // Handle array values for multiselect filters (OR condition)
        if (Array.isArray(value)) {
          if (value.length === 0) continue;

          // For multiselect filters, match if the fragment value equals ANY of the selected options
          const conditions = value.map(() => `${key} = ?`).join(' OR ');
          query += ` AND (${conditions})`;
          params.push(...value);
        } else if (typeof value === 'string' && value !== '') {
          // Handle single string values for text filters
          query += ` AND ${key} = ?`;
          params.push(value);
        }
      }
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
   * Update fragment metadata
   */
  ipcMain.handle('fragments:updateMetadata', async (_event, fragmentId: string, metadata: Record<string, unknown>) => {
    const staticFields = [
      'edge_piece',
      'has_top_edge',
      'has_bottom_edge',
      'has_left_edge',
      'has_right_edge',
      'has_circle',
      'line_count',
      'script_type',
      'scale_unit',
      'pixels_per_unit',
      'scale_detection_status'
    ];
    const customKeys = db.prepare('SELECT filter_key FROM custom_filters').all() as Array<{ filter_key: string }>;
    const allowedFields = new Set<string>([
      ...staticFields,
      ...customKeys.map((row) => row.filter_key),
    ]);
    const updates: string[] = [];
    const params: unknown[] = [];

    for (const [key, value] of Object.entries(metadata)) {
      if (allowedFields.has(key)) {
        updates.push(`${key} = ?`);
        // Store values as strings (custom filters store single values per fragment)
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
  // Custom Filter Handlers
  // ============================================

  ipcMain.handle('customFilters:list', async () => {
    try {
      const rows = db.prepare(
        'SELECT id, filter_key, label, type, options FROM custom_filters ORDER BY id ASC'
      ).all() as Array<{ id: number; filter_key: string; label: string; type: 'multiselect' | 'text'; options: string | null }>;

      const data: CustomFilterDefinition[] = rows.map((row) => ({
        id: row.id,
        filterKey: row.filter_key,
        label: row.label,
        type: row.type,
        options: row.type === 'multiselect' ? parseOptionsJson(row.options) : undefined,
      }));

      return { success: true, data };
    } catch (error) {
      console.error('Error listing custom filters:', error);
      return { success: false, error: String(error) };
    }
  });

  ipcMain.handle('customFilters:create', async (_event, payload: {
    label?: string;
    type?: 'multiselect' | 'text';
    options?: string[];
  }) => {
    const label = typeof payload?.label === 'string' ? payload.label.trim() : '';
    const type = payload?.type;

    if (!label) {
      return { success: false, error: 'Filter name is required' };
    }
    if (type !== 'multiselect' && type !== 'text') {
      return { success: false, error: 'Invalid filter type' };
    }

    let options: string[] | undefined;
    if (type === 'multiselect') {
      options = parseOptions(payload?.options);
      if (!options.length) {
        return { success: false, error: 'Dropdown filters require at least one option' };
      }
    }

    const fragmentColumns = db.prepare("PRAGMA table_info(fragments)").all() as Array<{ name: string }>;
    const fragmentColumnSet = new Set(fragmentColumns.map((col) => col.name));
    const existingCustom = db.prepare('SELECT filter_key FROM custom_filters').all() as Array<{ filter_key: string }>;
    const existingCustomSet = new Set(existingCustom.map((row) => row.filter_key));

    const baseKey = normalizeFilterKey(label);
    let filterKey = baseKey;
    let suffix = 2;
    while (fragmentColumnSet.has(filterKey) || existingCustomSet.has(filterKey)) {
      filterKey = `${baseKey}_${suffix}`;
      suffix += 1;
    }

    if (!VALID_IDENTIFIER.test(filterKey)) {
      return { success: false, error: 'Generated filter key is invalid' };
    }

    try {
      const insert = db.prepare(
        'INSERT INTO custom_filters (filter_key, label, type, options) VALUES (?, ?, ?, ?)'
      );

      const transaction = db.transaction(() => {
        db.exec(`ALTER TABLE fragments ADD COLUMN ${filterKey} TEXT`);
        db.exec(`CREATE INDEX IF NOT EXISTS idx_fragments_${filterKey} ON fragments(${filterKey})`);
        const result = insert.run(
          filterKey,
          label,
          type,
          options ? JSON.stringify(options) : null
        );
        return result.lastInsertRowid;
      });

      const id = transaction() as number;
      const data: CustomFilterDefinition = {
        id: Number(id),
        filterKey,
        label,
        type,
        options: type === 'multiselect' ? options : undefined,
      };

      return { success: true, data };
    } catch (error) {
      console.error('Error creating custom filter:', error);
      return { success: false, error: String(error) };
    }
  });

  ipcMain.handle('customFilters:delete', async (_event, id: number) => {
    if (typeof id !== 'number') {
      return { success: false, error: 'Invalid filter id' };
    }

    try {
      const row = db.prepare(
        'SELECT filter_key FROM custom_filters WHERE id = ?'
      ).get(id) as { filter_key?: string } | undefined;

      if (!row?.filter_key) {
        return { success: false, error: 'Filter not found' };
      }

      const filterKey = row.filter_key;
      if (!VALID_IDENTIFIER.test(filterKey)) {
        return { success: false, error: 'Invalid filter key' };
      }

      const transaction = db.transaction(() => {
        db.prepare('DELETE FROM custom_filters WHERE id = ?').run(id);
        db.exec(`UPDATE fragments SET ${filterKey} = NULL`);
        db.exec(`DROP INDEX IF EXISTS idx_fragments_${filterKey}`);
      });

      transaction();
      return { success: true };
    } catch (error) {
      console.error('Error deleting custom filter:', error);
      return { success: false, error: String(error) };
    }
  });

  ipcMain.handle('customFilters:updateOptions', async (_event, id: number, optionsInput: string[]) => {
    if (typeof id !== 'number') {
      return { success: false, error: 'Invalid filter id' };
    }

    try {
      const row = db.prepare(
        'SELECT id, filter_key, label, type FROM custom_filters WHERE id = ?'
      ).get(id) as { id: number; filter_key: string; label: string; type: 'multiselect' | 'text' } | undefined;

      if (!row) {
        return { success: false, error: 'Filter not found' };
      }
      if (row.type !== 'multiselect') {
        return { success: false, error: 'Only dropdown filters have options' };
      }
      if (!VALID_IDENTIFIER.test(row.filter_key)) {
        return { success: false, error: 'Invalid filter key' };
      }

      const options = parseOptions(optionsInput);
      if (!options.length) {
        return { success: false, error: 'At least one option is required' };
      }

      const filterKey = row.filter_key;

      const transaction = db.transaction(() => {
        db.prepare('UPDATE custom_filters SET options = ? WHERE id = ?')
          .run(JSON.stringify(options), id);

        const placeholders = options.map(() => '?').join(',');
        db.prepare(
          `UPDATE fragments SET ${filterKey} = NULL WHERE ${filterKey} IS NOT NULL AND ${filterKey} NOT IN (${placeholders})`
        ).run(...options);
      });

      transaction();

      const data: CustomFilterDefinition = {
        id: row.id,
        filterKey,
        label: row.label,
        type: row.type,
        options,
      };

      return { success: true, data };
    } catch (error) {
      console.error('Error updating custom filter options:', error);
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

  /**
   * Check if segmented image exists for a fragment
   */
  ipcMain.handle('images:hasSegmented', async (_event, fragmentId: string) => {
    const isDev = !app.isPackaged;
    const basePath = isDev
      ? path.join(process.cwd(), 'electron/resources/cache/segmented')
      : path.join(process.resourcesPath, 'cache/segmented');

    const segmentedPath = path.join(basePath, `${fragmentId}_segmented.png`);

    try {
      await fs.promises.access(segmentedPath, fs.constants.F_OK);
      return { success: true, exists: true };
    } catch {
      return { success: true, exists: false };
    }
  });

  /**
   * Batch check segmented images for multiple fragments
   */
  ipcMain.handle('images:batchHasSegmented', async (_event, fragmentIds: string[]) => {
    const isDev = !app.isPackaged;
    const basePath = isDev
      ? path.join(process.cwd(), 'electron/resources/cache/segmented')
      : path.join(process.resourcesPath, 'cache/segmented');

    const results: Record<string, boolean> = {};

    await Promise.all(fragmentIds.map(async (id) => {
      const segmentedPath = path.join(basePath, `${id}_segmented.png`);
      try {
        await fs.promises.access(segmentedPath, fs.constants.F_OK);
        results[id] = true;
      } catch {
        results[id] = false;
      }
    }));

    return { success: true, data: results };
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
   * Save project canvas state
   */
  ipcMain.handle('projects:save', async (_event, projectId: number, canvasState: CanvasState) => {
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
            project_id, fragment_id, x, y, width, height, rotation, scale_x, scale_y, is_locked, z_index, show_segmented
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            frag.zIndex || 0,
            frag.showSegmented !== undefined ? (frag.showSegmented ? 1 : 0) : 1
          );
        }

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
          is_locked as isLocked, z_index as zIndex,
          show_segmented as showSegmented
        FROM project_fragments
        WHERE project_id = ?
        ORDER BY z_index ASC
      `).all(projectId);

      return {
        success: true,
        data: {
          project,
          canvasState: { fragments },
          notes: '',
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


  /**
   * Rename a project
   */
  ipcMain.handle('projects:rename', async (_event, projectId: number, newName: string) => {
    try {
      const result = db.prepare(
        'UPDATE projects SET project_name = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?'
      ).run(newName, projectId);
      return { success: true, changes: result.changes };
    } catch (error) {
      console.error('Error renaming project:', error);
      return { success: false, error: String(error) };
    }
  });

  // ============================================
  // File Upload Handlers
  // ============================================

  /**
   * Open file selection dialog
   */
  ipcMain.handle('files:selectImages', async () => {
    try {
      const result = await dialog.showOpenDialog({
        title: 'Select Fragment Images',
        properties: ['openFile', 'multiSelections'],
        filters: [
          { name: 'Images', extensions: ['jpg', 'jpeg', 'png'] }
        ]
      });

      if (result.canceled) {
        return { success: false, canceled: true };
      }

      return {
        success: true,
        filePaths: result.filePaths
      };
    } catch (error) {
      console.error('Error opening file dialog:', error);
      return { success: false, error: String(error) };
    }
  });

  /**
   * Upload files to the uploads directory and add to database
   */
  ipcMain.handle('fragments:uploadFiles', async (_event, filePaths: string[]) => {
    const isDev = !app.isPackaged;
    const dataDir = isDev
      ? path.join(process.cwd(), 'data')
      : path.join(process.resourcesPath, 'data');
    const uploadDir = path.join(dataDir, 'uploads');

    // Ensure uploads directory exists
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }

    const results = [];

    for (const filePath of filePaths) {
      try {
        // 1. Generate unique filename
        const filename = path.basename(filePath);
        const uniqueFilename = generateUniqueFilename(uploadDir, filename);

        // 2. Copy file to uploads directory
        const destPath = path.join(uploadDir, uniqueFilename);
        fs.copyFileSync(filePath, destPath);

        // 3. Insert into database
        const fragmentId = uniqueFilename.replace(/\.(jpg|jpeg|png)$/i, '');
        const imagePath = `uploads/${uniqueFilename}`;

        const insert = db.prepare(`
          INSERT INTO fragments (fragment_id, image_path)
          VALUES (?, ?)
        `);

        insert.run(fragmentId, imagePath);

        results.push({
          success: true,
          fragmentId,
          filename: uniqueFilename
        });

      } catch (error) {
        results.push({
          success: false,
          filename: path.basename(filePath),
          error: String(error)
        });
      }
    }

    return { success: true, results };
  });

  console.log('IPC handlers registered');
}

/**
 * Generate a unique filename to avoid collisions
 */
function generateUniqueFilename(dir: string, filename: string): string {
  const ext = path.extname(filename);
  const base = path.basename(filename, ext);

  // Check if file exists
  let targetPath = path.join(dir, filename);
  if (!fs.existsSync(targetPath)) {
    return filename;
  }

  // Generate timestamp suffix (YYYYMMDDHHmmss)
  const now = new Date();
  const timestamp = now.toISOString()
    .replace(/[-:T]/g, '')
    .replace(/\.\d{3}Z$/, '')
    .slice(0, 14);

  let counter = 0;
  let newFilename = `${base}_${timestamp}${ext}`;
  targetPath = path.join(dir, newFilename);

  // Handle edge case: multiple uploads in same second
  while (fs.existsSync(targetPath)) {
    counter++;
    newFilename = `${base}_${timestamp}_${counter}${ext}`;
    targetPath = path.join(dir, newFilename);
  }

  return newFilename;
}

export default { registerIpcHandlers };
