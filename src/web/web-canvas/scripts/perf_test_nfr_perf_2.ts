/**
 * Performance test: scripted single-user workflow (test-nfr-perf-2).
 *
 * Runs DB-backed operations for a fixed duration and records response times.
 * Outputs JSON summary and per-operation JSONL logs.
 */

import Database from 'better-sqlite3';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

type OperationType = 'search' | 'filter' | 'canvas' | 'session';

type OperationResult = {
  ts: string;
  type: OperationType;
  ok: boolean;
  durationMs: number;
  details?: string;
  error?: string;
};

type Summary = {
  test_id: string;
  started_at: string;
  ended_at: string;
  duration_seconds: number;
  totals: {
    operations: number;
    errors: number;
    error_rate: number;
  };
  search: {
    count: number;
    mean_ms: number;
    p95_ms: number;
    max_ms: number;
  };
  canvas: {
    count: number;
    p95_ms: number;
    max_ms: number;
    under_500ms_ratio: number;
  };
  checks: {
    mean_search_ok: boolean;
    p95_search_ok: boolean;
    canvas_500ms_ok: boolean;
    error_rate_ok: boolean;
  };
  thresholds: {
    mean_search_ms: number;
    p95_search_ms: number;
    canvas_ms: number;
    error_rate: number;
  };
  resource: {
    max_rss_mb: number;
    rss_start_mb: number;
    rss_end_mb: number;
  };
};

type Options = {
  durationMinutes: number;
  searchPct: number;
  filterPct: number;
  canvasPct: number;
  sessionPct: number;
  dbPath: string;
  logDir: string;
  reportPath: string;
  minFragments: number;
  maxFragments: number;
};

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..', '..');

function parseArgs(argv: string[]): Options {
  const options: Options = {
    durationMinutes: 20,
    searchPct: 10,
    filterPct: 40,
    canvasPct: 30,
    sessionPct: 20,
    dbPath: path.join(process.cwd(), 'electron', 'resources', 'database', 'fragments.db'),
    logDir: path.join(process.cwd(), 'logs'),
    reportPath: path.join(repoRoot, 'reports', 'perf_test_nfr_perf_2.json'),
    minFragments: 5,
    maxFragments: 10,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith('--')) continue;
    const key = arg.slice(2);
    const next = argv[i + 1];
    const value = next && !next.startsWith('--') ? next : undefined;

    switch (key) {
      case 'duration-minutes':
        if (value) options.durationMinutes = Number(value);
        i += 1;
        break;
      case 'search-pct':
        if (value) options.searchPct = Number(value);
        i += 1;
        break;
      case 'filter-pct':
        if (value) options.filterPct = Number(value);
        i += 1;
        break;
      case 'canvas-pct':
        if (value) options.canvasPct = Number(value);
        i += 1;
        break;
      case 'session-pct':
        if (value) options.sessionPct = Number(value);
        i += 1;
        break;
      case 'db-path':
        if (value) options.dbPath = value;
        i += 1;
        break;
      case 'log-dir':
        if (value) options.logDir = value;
        i += 1;
        break;
      case 'report':
        if (value) options.reportPath = value;
        i += 1;
        break;
      case 'min-fragments':
        if (value) options.minFragments = Number(value);
        i += 1;
        break;
      case 'max-fragments':
        if (value) options.maxFragments = Number(value);
        i += 1;
        break;
      default:
        break;
    }
  }

  return options;
}

function ensureDir(dirPath: string): void {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

function nowIso(): string {
  return new Date().toISOString();
}

function logLine(logFile: string, message: string): void {
  fs.appendFileSync(logFile, `${message}\n`, 'utf-8');
}

function logJsonLine(jsonlFile: string, payload: unknown): void {
  fs.appendFileSync(jsonlFile, `${JSON.stringify(payload)}\n`, 'utf-8');
}

function percentile(values: number[], p: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, Math.min(idx, sorted.length - 1))];
}

function mean(values: number[]): number {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function pickOperation(options: Options): OperationType {
  const roll = Math.random() * 100;
  if (roll < options.searchPct) return 'search';
  if (roll < options.searchPct + options.filterPct) return 'filter';
  if (roll < options.searchPct + options.filterPct + options.canvasPct) return 'canvas';
  return 'session';
}

function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function randomChoice<T>(items: T[]): T {
  return items[Math.floor(Math.random() * items.length)];
}

function runSearch(db: Database.Database, fragmentIds: string[]): OperationResult {
  const start = performance.now();
  try {
    const fragmentId = randomChoice(fragmentIds);
    const term = fragmentId.slice(0, Math.min(4, fragmentId.length));
    const rows = db.prepare('SELECT fragment_id FROM fragments WHERE fragment_id LIKE ? LIMIT 25').all(`%${term}%`);
    const durationMs = performance.now() - start;
    return {
      ts: nowIso(),
      type: 'search',
      ok: rows.length > 0,
      durationMs,
      details: `term=${term} count=${rows.length}`,
    };
  } catch (error) {
    return {
      ts: nowIso(),
      type: 'search',
      ok: false,
      durationMs: performance.now() - start,
      error: String(error),
    };
  }
}

function runFilter(db: Database.Database, scriptTypes: string[]): OperationResult {
  const start = performance.now();
  try {
    const filterType = randomInt(1, 3);
    let rows: Array<{ fragment_id: string }>;
    let details = '';
    if (filterType === 1) {
      rows = db.prepare('SELECT fragment_id FROM fragments WHERE edge_piece = ? LIMIT 25').all(0);
      details = 'edge_piece=0';
    } else if (filterType === 2 && scriptTypes.length > 0) {
      const script = randomChoice(scriptTypes);
      rows = db.prepare('SELECT fragment_id FROM fragments WHERE script_type = ? LIMIT 25').all(script);
      details = `script_type=${script}`;
    } else {
      rows = db.prepare('SELECT fragment_id FROM fragments WHERE has_circle = ? LIMIT 25').all(1);
      details = 'has_circle=1';
    }
    const durationMs = performance.now() - start;
    return {
      ts: nowIso(),
      type: 'filter',
      ok: rows.length > 0,
      durationMs,
      details,
    };
  } catch (error) {
    return {
      ts: nowIso(),
      type: 'filter',
      ok: false,
      durationMs: performance.now() - start,
      error: String(error),
    };
  }
}

function runCanvas(db: Database.Database, fragmentIds: string[], minFragments: number, maxFragments: number): OperationResult {
  const start = performance.now();
  const projectName = `perf-canvas-${Date.now()}`;
  try {
    const projectResult = db
      .prepare('INSERT INTO projects (project_name, description) VALUES (?, ?)')
      .run(projectName, 'perf workflow');
    const projectId = Number(projectResult.lastInsertRowid);

    const fragmentCount = randomInt(minFragments, maxFragments);
    const insertFragment = db.prepare(`
      INSERT INTO project_fragments (
        project_id, fragment_id, x, y, width, height, rotation, scale_x, scale_y, is_locked, z_index, show_segmented
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    const transaction = db.transaction(() => {
      for (let i = 0; i < fragmentCount; i += 1) {
        insertFragment.run(
          projectId,
          randomChoice(fragmentIds),
          100 + i * 10,
          100 + i * 10,
          null,
          null,
          0,
          1,
          1,
          0,
          i,
          1
        );
      }
    });
    transaction();

    const loaded = db
      .prepare('SELECT fragment_id FROM project_fragments WHERE project_id = ?')
      .all(projectId);

    db.prepare('DELETE FROM project_fragments WHERE project_id = ?').run(projectId);
    db.prepare('DELETE FROM projects WHERE id = ?').run(projectId);

    const durationMs = performance.now() - start;
    return {
      ts: nowIso(),
      type: 'canvas',
      ok: loaded.length === fragmentCount,
      durationMs,
      details: `fragments=${fragmentCount} loaded=${loaded.length}`,
    };
  } catch (error) {
    return {
      ts: nowIso(),
      type: 'canvas',
      ok: false,
      durationMs: performance.now() - start,
      error: String(error),
    };
  }
}

function runSession(db: Database.Database, fragmentIds: string[], minFragments: number, maxFragments: number): OperationResult {
  const start = performance.now();
  const projectName = `perf-session-${Date.now()}`;
  try {
    const projectResult = db
      .prepare('INSERT INTO projects (project_name, description) VALUES (?, ?)')
      .run(projectName, 'perf session');
    const projectId = Number(projectResult.lastInsertRowid);

    const fragmentCount = randomInt(minFragments, maxFragments);
    const insertFragment = db.prepare(`
      INSERT INTO project_fragments (
        project_id, fragment_id, x, y, width, height, rotation, scale_x, scale_y, is_locked, z_index, show_segmented
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    const transaction = db.transaction(() => {
      for (let i = 0; i < fragmentCount; i += 1) {
        insertFragment.run(
          projectId,
          randomChoice(fragmentIds),
          200 + i * 12,
          200 + i * 12,
          null,
          null,
          0,
          1,
          1,
          0,
          i,
          1
        );
      }
    });
    transaction();

    const loaded = db
      .prepare('SELECT fragment_id FROM project_fragments WHERE project_id = ? ORDER BY z_index ASC')
      .all(projectId);

    db.prepare('UPDATE projects SET updated_at = CURRENT_TIMESTAMP WHERE id = ?').run(projectId);
    db.prepare('DELETE FROM project_fragments WHERE project_id = ?').run(projectId);
    db.prepare('DELETE FROM projects WHERE id = ?').run(projectId);

    const durationMs = performance.now() - start;
    return {
      ts: nowIso(),
      type: 'session',
      ok: loaded.length === fragmentCount,
      durationMs,
      details: `fragments=${fragmentCount} loaded=${loaded.length}`,
    };
  } catch (error) {
    return {
      ts: nowIso(),
      type: 'session',
      ok: false,
      durationMs: performance.now() - start,
      error: String(error),
    };
  }
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  ensureDir(options.logDir);
  ensureDir(path.dirname(options.reportPath));

  const logFile = path.join(options.logDir, 'perf-workflow.log');
  const jsonlFile = path.join(options.logDir, 'perf-workflow.jsonl');

  logLine(logFile, `--- perf-workflow started ${nowIso()} ---`);
  console.log(`[perf-test-2] Starting. duration=${options.durationMinutes}min`);

  const db = new Database(options.dbPath);
  db.pragma('foreign_keys = ON');

  const fragments = db.prepare('SELECT fragment_id, script_type FROM fragments ORDER BY id LIMIT 1000').all() as Array<{ fragment_id: string; script_type: string | null }>;
  const fragmentIds = fragments.map((row) => row.fragment_id);
  const scriptTypes = Array.from(new Set(fragments.map((row) => row.script_type).filter((value): value is string => !!value)));

  if (!fragmentIds.length) {
    throw new Error('No fragments found in database.');
  }

  const startTime = Date.now();
  const endTime = startTime + options.durationMinutes * 60 * 1000;
  const rssStart = process.memoryUsage().rss / (1024 * 1024);
  let rssMax = rssStart;

  const results: OperationResult[] = [];

  while (Date.now() < endTime) {
    const opType = pickOperation(options);
    let result: OperationResult;

    switch (opType) {
      case 'search':
        result = runSearch(db, fragmentIds);
        break;
      case 'filter':
        result = runFilter(db, scriptTypes);
        break;
      case 'canvas':
        result = runCanvas(db, fragmentIds, options.minFragments, options.maxFragments);
        break;
      case 'session':
        result = runSession(db, fragmentIds, options.minFragments, options.maxFragments);
        break;
      default:
        result = runSearch(db, fragmentIds);
        break;
    }

    results.push(result);
    logJsonLine(jsonlFile, result);

    const rssNow = process.memoryUsage().rss / (1024 * 1024);
    rssMax = Math.max(rssMax, rssNow);
  }

  db.close();

  const searchDurations = results.filter((r) => r.type === 'search').map((r) => r.durationMs);
  const canvasDurations = results.filter((r) => r.type === 'canvas').map((r) => r.durationMs);
  const errors = results.filter((r) => !r.ok).length;

  const meanSearch = mean(searchDurations);
  const p95Search = percentile(searchDurations, 95);
  const p95Canvas = percentile(canvasDurations, 95);

  const under500 = canvasDurations.filter((d) => d <= 500).length;
  const under500Ratio = canvasDurations.length ? under500 / canvasDurations.length : 0;

  const durationSeconds = (Date.now() - startTime) / 1000;
  const errorRate = results.length ? errors / results.length : 0;

  const summary: Summary = {
    test_id: 'test-nfr-perf-2',
    started_at: new Date(startTime).toISOString(),
    ended_at: nowIso(),
    duration_seconds: Math.round(durationSeconds),
    totals: {
      operations: results.length,
      errors,
      error_rate: Number(errorRate.toFixed(4)),
    },
    search: {
      count: searchDurations.length,
      mean_ms: Number(meanSearch.toFixed(2)),
      p95_ms: Number(p95Search.toFixed(2)),
      max_ms: Number((searchDurations.length ? Math.max(...searchDurations) : 0).toFixed(2)),
    },
    canvas: {
      count: canvasDurations.length,
      p95_ms: Number(p95Canvas.toFixed(2)),
      max_ms: Number((canvasDurations.length ? Math.max(...canvasDurations) : 0).toFixed(2)),
      under_500ms_ratio: Number(under500Ratio.toFixed(4)),
    },
    checks: {
      mean_search_ok: meanSearch <= 2000,
      p95_search_ok: p95Search <= 3000,
      canvas_500ms_ok: under500Ratio >= 0.95,
      error_rate_ok: errorRate <= 0.02,
    },
    thresholds: {
      mean_search_ms: 2000,
      p95_search_ms: 3000,
      canvas_ms: 500,
      error_rate: 0.02,
    },
    resource: {
      max_rss_mb: Number(rssMax.toFixed(2)),
      rss_start_mb: Number(rssStart.toFixed(2)),
      rss_end_mb: Number((process.memoryUsage().rss / (1024 * 1024)).toFixed(2)),
    },
  };

  fs.writeFileSync(options.reportPath, JSON.stringify(summary, null, 2), 'utf-8');
  logLine(logFile, `Summary: ${JSON.stringify(summary)}`);

  console.log(`[perf-test-2] Summary written to ${options.reportPath}`);
}

main().catch((error) => {
  console.error('perf-test-2 failed:', error);
  process.exit(1);
});
