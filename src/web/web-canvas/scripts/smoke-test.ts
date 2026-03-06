/**
 * Reliability smoke test for the desktop app.
 *
 * Steps per run:
 *  - (Optional) Launch app command
 *  - Run search + filter queries
 *  - Create project, save fragments, reload project
 *  - Run DB integrity checks
 *
 * Logs are written to logs/smoke-test.log and logs/smoke-test.jsonl
 */

import Database from 'better-sqlite3';
import fs from 'node:fs';
import path from 'node:path';
import { spawn, ChildProcess } from 'node:child_process';

type StepResult = {
  name: string;
  ok: boolean;
  details?: string;
};

type RunResult = {
  runId: string;
  startedAt: string;
  endedAt: string;
  ok: boolean;
  steps: StepResult[];
  error?: string;
};

type Options = {
  runs: number;
  intervalSeconds: number;
  appCmd?: string;
  appCwd: string;
  appStartupWaitMs: number;
  dbPath: string;
  dataRoot: string;
  logDir: string;
  searchTerm?: string;
  projectFragments: number;
};

const PROJECT_ROOT = process.cwd();

function parseArgs(argv: string[]): Options {
  const defaultDbPath = path.join(
    PROJECT_ROOT,
    'electron',
    'resources',
    'database',
    'fragments.db'
  );
  const defaultDataRoot = path.join(PROJECT_ROOT, 'data');

  const options: Options = {
    runs: 7,
    intervalSeconds: 86400,
    appCmd: undefined,
    appCwd: PROJECT_ROOT,
    appStartupWaitMs: 10000,
    dbPath: defaultDbPath,
    dataRoot: defaultDataRoot,
    logDir: path.join(PROJECT_ROOT, 'logs'),
    searchTerm: undefined,
    projectFragments: 3,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith('--')) {
      continue;
    }
    const key = arg.slice(2);
    const next = argv[i + 1];
    const value = next && !next.startsWith('--') ? next : undefined;

    switch (key) {
      case 'runs':
        if (value) {
          options.runs = Number(value);
          i += 1;
        }
        break;
      case 'interval-seconds':
        if (value) {
          options.intervalSeconds = Number(value);
          i += 1;
        }
        break;
      case 'app-cmd':
        if (value) {
          options.appCmd = value;
          i += 1;
        }
        break;
      case 'app-cwd':
        if (value) {
          options.appCwd = value;
          i += 1;
        }
        break;
      case 'app-startup-wait-ms':
        if (value) {
          options.appStartupWaitMs = Number(value);
          i += 1;
        }
        break;
      case 'db-path':
        if (value) {
          options.dbPath = value;
          i += 1;
        }
        break;
      case 'data-root':
        if (value) {
          options.dataRoot = value;
          i += 1;
        }
        break;
      case 'log-dir':
        if (value) {
          options.logDir = value;
          i += 1;
        }
        break;
      case 'search-term':
        if (value) {
          options.searchTerm = value;
          i += 1;
        }
        break;
      case 'project-fragments':
        if (value) {
          options.projectFragments = Number(value);
          i += 1;
        }
        break;
      default:
        break;
    }
  }

  return options;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function ensureDir(dirPath: string): void {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

function formatRunId(index: number): string {
  const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\..+/, '');
  return `${stamp}-run${index + 1}`;
}

function resolveImagePath(dataRoot: string, imagePath: string): string {
  if (path.isAbsolute(imagePath)) {
    return imagePath;
  }
  return path.join(dataRoot, imagePath);
}

function logLine(logFile: string, message: string): void {
  fs.appendFileSync(logFile, `${message}\n`, 'utf-8');
}

function logJsonLine(jsonlFile: string, payload: unknown): void {
  fs.appendFileSync(jsonlFile, `${JSON.stringify(payload)}\n`, 'utf-8');
}

function launchApp(appCmd: string, appCwd: string): ChildProcess {
  return spawn(appCmd, {
    cwd: appCwd,
    shell: true,
    stdio: 'ignore',
  });
}

function closeApp(proc: ChildProcess): void {
  try {
    proc.kill();
  } catch {
    // Best-effort cleanup
  }
}

function checkIntegrity(db: Database.Database, steps: StepResult[]): void {
  const integrity = db.pragma('integrity_check');
  const integrityOk = Array.isArray(integrity)
    ? integrity.every((row) => row.integrity_check === 'ok')
    : integrity === 'ok';
  steps.push({
    name: 'db.integrity_check',
    ok: integrityOk,
    details: integrityOk ? 'ok' : JSON.stringify(integrity),
  });

  const fkCheck = db.pragma('foreign_key_check');
  const fkOk = Array.isArray(fkCheck) ? fkCheck.length === 0 : fkCheck === 'ok';
  steps.push({
    name: 'db.foreign_key_check',
    ok: fkOk,
    details: fkOk ? 'ok' : JSON.stringify(fkCheck),
  });
}

function runDbSteps(options: Options, steps: StepResult[]): void {
  const db = new Database(options.dbPath);
  db.pragma('foreign_keys = ON');

  const countRow = db.prepare('SELECT COUNT(*) as count FROM fragments').get() as { count: number };
  steps.push({
    name: 'db.count_fragments',
    ok: countRow.count > 0,
    details: `count=${countRow.count}`,
  });

  if (countRow.count === 0) {
    db.close();
    return;
  }

  const firstRow = db
    .prepare('SELECT fragment_id FROM fragments ORDER BY fragment_id ASC LIMIT 1')
    .get() as { fragment_id: string };
  const fallbackSearch = firstRow.fragment_id.slice(0, Math.min(3, firstRow.fragment_id.length));
  const searchTerm = options.searchTerm ?? fallbackSearch;

  const searchRow = db
    .prepare('SELECT COUNT(*) as count FROM fragments WHERE fragment_id LIKE ?')
    .get(`%${searchTerm}%`) as { count: number };
  steps.push({
    name: 'db.search',
    ok: searchRow.count > 0,
    details: `term=${searchTerm} count=${searchRow.count}`,
  });

  const filterRow = db
    .prepare('SELECT COUNT(*) as count FROM fragments WHERE edge_piece = ?')
    .get(0) as { count: number };
  steps.push({
    name: 'db.filter',
    ok: filterRow.count > 0,
    details: `edge_piece=0 count=${filterRow.count}`,
  });

  const fragments = db
    .prepare('SELECT fragment_id, image_path FROM fragments ORDER BY fragment_id ASC LIMIT ?')
    .all(options.projectFragments) as Array<{ fragment_id: string; image_path: string }>;

  steps.push({
    name: 'db.load_canvas_fragments',
    ok: fragments.length > 0,
    details: `count=${fragments.length}`,
  });

  const missingImages: string[] = [];
  for (const frag of fragments) {
    const resolved = resolveImagePath(options.dataRoot, frag.image_path);
    if (!fs.existsSync(resolved)) {
      missingImages.push(`${frag.fragment_id}:${resolved}`);
    }
  }
  steps.push({
    name: 'fs.image_paths',
    ok: missingImages.length === 0,
    details: missingImages.length ? missingImages.join(', ') : 'ok',
  });

  const projectName = `smoke-test-${new Date().toISOString()}`;
  const projectInsert = db.prepare(
    'INSERT INTO projects (project_name, description) VALUES (?, ?)'
  );
  const projectResult = projectInsert.run(projectName, 'Automated smoke test project');
  const projectId = Number(projectResult.lastInsertRowid);

  const insertProjectFragment = db.prepare(`
    INSERT INTO project_fragments (
      project_id, fragment_id, x, y, width, height, rotation, scale_x, scale_y, is_locked, z_index, show_segmented
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);

  const saveTransaction = db.transaction(() => {
    let index = 0;
    for (const frag of fragments) {
      insertProjectFragment.run(
        projectId,
        frag.fragment_id,
        100 + index * 20,
        100 + index * 20,
        null,
        null,
        0,
        1,
        1,
        0,
        index,
        1
      );
      index += 1;
    }
  });
  saveTransaction();

  const savedCountRow = db
    .prepare('SELECT COUNT(*) as count FROM project_fragments WHERE project_id = ?')
    .get(projectId) as { count: number };
  steps.push({
    name: 'db.project_save',
    ok: savedCountRow.count === fragments.length,
    details: `saved=${savedCountRow.count}`,
  });

  const loadRows = db
    .prepare('SELECT fragment_id FROM project_fragments WHERE project_id = ? ORDER BY z_index ASC')
    .all(projectId) as Array<{ fragment_id: string }>;
  steps.push({
    name: 'db.project_load',
    ok: loadRows.length === fragments.length,
    details: `loaded=${loadRows.length}`,
  });

  db.prepare('DELETE FROM project_fragments WHERE project_id = ?').run(projectId);
  db.prepare('DELETE FROM projects WHERE id = ?').run(projectId);
  steps.push({
    name: 'db.project_cleanup',
    ok: true,
    details: `project_id=${projectId}`,
  });

  checkIntegrity(db, steps);
  db.close();
}

async function runOnce(index: number, options: Options, logFile: string, jsonlFile: string): Promise<RunResult> {
  const runId = formatRunId(index);
  const startedAt = new Date().toISOString();
  const steps: StepResult[] = [];
  let appProcess: ChildProcess | undefined;
  let error: string | undefined;

  try {
    console.log(`[smoke-test] Starting ${runId}`);
    if (options.appCmd) {
      appProcess = launchApp(options.appCmd, options.appCwd);
      steps.push({ name: 'app.launch', ok: true, details: `pid=${appProcess.pid}` });
      if (appProcess.pid) {
        console.log(`[smoke-test] App launched (pid=${appProcess.pid}), waiting ${options.appStartupWaitMs}ms`);
      }
      appProcess.on('error', (err) => {
        steps.push({ name: 'app.launch_error', ok: false, details: String(err) });
      });
      await sleep(options.appStartupWaitMs);
    } else {
      steps.push({ name: 'app.launch', ok: true, details: 'skipped (no app-cmd)' });
    }

    if (!fs.existsSync(options.dbPath)) {
      steps.push({ name: 'db.path', ok: false, details: `missing ${options.dbPath}` });
    } else {
      steps.push({ name: 'db.path', ok: true, details: options.dbPath });
      runDbSteps(options, steps);
    }
  } catch (err) {
    error = err instanceof Error ? err.message : String(err);
    steps.push({ name: 'run.error', ok: false, details: error });
  } finally {
    if (appProcess) {
      closeApp(appProcess);
      steps.push({ name: 'app.exit', ok: true, details: 'terminated' });
    }
  }

  const ok = steps.every((step) => step.ok);
  const endedAt = new Date().toISOString();

  const result: RunResult = {
    runId,
    startedAt,
    endedAt,
    ok,
    steps,
    error,
  };

  console.log(`[smoke-test] Finished ${runId}: ${ok ? 'PASS' : 'FAIL'}`);
  logLine(logFile, `[${endedAt}] ${runId} ${ok ? 'PASS' : 'FAIL'}`);
  logJsonLine(jsonlFile, result);

  return result;
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  ensureDir(options.logDir);

  const logFile = path.join(options.logDir, 'smoke-test.log');
  const jsonlFile = path.join(options.logDir, 'smoke-test.jsonl');

  console.log('[smoke-test] Starting');
  console.log(`[smoke-test] runs=${options.runs} intervalSeconds=${options.intervalSeconds}`);
  console.log(`[smoke-test] db=${options.dbPath}`);
  console.log(`[smoke-test] dataRoot=${options.dataRoot}`);
  console.log(`[smoke-test] logs=${options.logDir}`);
  logLine(logFile, `--- Smoke test started ${new Date().toISOString()} ---`);

  const results: RunResult[] = [];

  for (let i = 0; i < options.runs; i += 1) {
    const result = await runOnce(i, options, logFile, jsonlFile);
    results.push(result);

    if (i < options.runs - 1) {
      await sleep(options.intervalSeconds * 1000);
    }
  }

  const passed = results.filter((r) => r.ok).length;
  const passRate = results.length ? passed / results.length : 0;
  const summary = {
    totalRuns: results.length,
    passed,
    failed: results.length - passed,
    passRate,
    threshold: 0.95,
  };

  logLine(logFile, `Summary: ${JSON.stringify(summary)}`);
  logJsonLine(jsonlFile, { type: 'summary', ...summary });

  if (passRate < 0.95) {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error('Smoke test failed:', error);
  process.exit(1);
});
