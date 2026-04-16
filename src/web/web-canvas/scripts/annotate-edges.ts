#!/usr/bin/env tsx

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync, spawn } from 'node:child_process';
import { stdin as input, stdout as output } from 'node:process';
import readline from 'node:readline/promises';
import { fileURLToPath } from 'node:url';

type EdgeValue = boolean | null;
type EdgeKey =
  | 'has_top_edge'
  | 'has_bottom_edge'
  | 'has_left_edge'
  | 'has_right_edge';
type OrderMode = 'balanced' | 'ordered' | 'targeted';
type TargetLabel = 'top' | 'bottom' | 'left' | 'right' | 'corner';

interface AnnotationRecord {
  has_top_edge: EdgeValue;
  has_bottom_edge: EdgeValue;
  has_left_edge: EdgeValue;
  has_right_edge: EdgeValue;
  reviewed: boolean;
  skipped: boolean;
  notes: string;
  updated_at: string;
}

interface AnnotationStore {
  version: number;
  data_root: string;
  created_at: string;
  updated_at: string;
  annotations: Record<string, AnnotationRecord>;
}

interface Options {
  dataRoot: string;
  annotationsFile: string;
  fragmentsDb?: string;
  collection?: string;
  match?: string;
  autoOpen: boolean;
  statsOnly: boolean;
  exportCsv?: string;
  start?: string;
  orderMode: OrderMode;
  seed: string;
  dataRootExplicit: boolean;
  annotationsFileExplicit: boolean;
  fragmentsDbExplicit: boolean;
  targetLabels: TargetLabel[];
}

interface ImageItem {
  absolutePath: string;
  relativePath: string;
  collection: string;
  filename: string;
  targetScore?: number;
  targetTags?: string[];
}

interface DbEdgePrediction {
  has_top_edge: boolean | null;
  has_bottom_edge: boolean | null;
  has_left_edge: boolean | null;
  has_right_edge: boolean | null;
  edge_piece: boolean | null;
}

interface TargetSummary {
  label: TargetLabel;
  reviewedPositives: number;
  predictedPositives: number;
}

interface TargetingContext {
  fragmentsDb: string;
  predictions: Map<string, DbEdgePrediction>;
  targetLabels: TargetLabel[];
}

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const APP_ROOT = path.resolve(SCRIPT_DIR, '..');
const REPO_ROOT = path.resolve(APP_ROOT, '..', '..', '..');
const DEFAULT_DATA_ROOT = path.join(APP_ROOT, 'data');
const DEFAULT_ANNOTATIONS_FILE = path.join(DEFAULT_DATA_ROOT, 'edge-annotations.json');
const DEFAULT_RANDOM_SEED = 'edge-annotator-balanced-v1';
const DEFAULT_FRAGMENTS_DB = path.join(APP_ROOT, 'electron', 'resources', 'database', 'fragments.db');
const IMAGE_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp']);
const EDGE_KEYS: EdgeKey[] = [
  'has_top_edge',
  'has_bottom_edge',
  'has_left_edge',
  'has_right_edge',
];

function nowIso(): string {
  return new Date().toISOString();
}

function toStoragePath(value: string): string {
  return value.split(path.sep).join('/');
}

function parseArgs(argv: string[]): Options {
  const options: Options = {
    dataRoot: DEFAULT_DATA_ROOT,
    annotationsFile: DEFAULT_ANNOTATIONS_FILE,
    fragmentsDb: DEFAULT_FRAGMENTS_DB,
    autoOpen: false,
    statsOnly: false,
    orderMode: 'balanced',
    seed: DEFAULT_RANDOM_SEED,
    dataRootExplicit: false,
    annotationsFileExplicit: false,
    fragmentsDbExplicit: false,
    targetLabels: [],
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = argv[i + 1];

    switch (arg) {
      case '--data-root':
        if (!next) {
          throw new Error('--data-root requires a value');
        }
        options.dataRoot = path.resolve(next);
        options.dataRootExplicit = true;
        i += 1;
        break;
      case '--annotations-file':
        if (!next) {
          throw new Error('--annotations-file requires a value');
        }
        options.annotationsFile = path.resolve(next);
        options.annotationsFileExplicit = true;
        i += 1;
        break;
      case '--fragments-db':
        if (!next) {
          throw new Error('--fragments-db requires a value');
        }
        options.fragmentsDb = path.resolve(next);
        options.fragmentsDbExplicit = true;
        i += 1;
        break;
      case '--collection':
        if (!next) {
          throw new Error('--collection requires a value');
        }
        options.collection = next.trim();
        i += 1;
        break;
      case '--match':
        if (!next) {
          throw new Error('--match requires a value');
        }
        options.match = next.trim();
        i += 1;
        break;
      case '--start':
        if (!next) {
          throw new Error('--start requires a value');
        }
        options.start = next.trim();
        i += 1;
        break;
      case '--seed':
        if (!next) {
          throw new Error('--seed requires a value');
        }
        options.seed = next.trim();
        i += 1;
        break;
      case '--targeted':
        options.orderMode = 'targeted';
        break;
      case '--targets':
        if (!next) {
          throw new Error('--targets requires a value');
        }
        options.targetLabels = parseTargetLabels(next);
        options.orderMode = 'targeted';
        i += 1;
        break;
      case '--ordered':
        options.orderMode = 'ordered';
        break;
      case '--open':
        options.autoOpen = true;
        break;
      case '--stats':
        options.statsOnly = true;
        break;
      case '--export-csv':
        if (!next) {
          throw new Error('--export-csv requires a value');
        }
        options.exportCsv = path.resolve(next);
        i += 1;
        break;
      case '--help':
      case '-h':
        printHelp();
        process.exit(0);
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (!options.annotationsFileExplicit) {
    options.annotationsFile = path.join(options.dataRoot, 'edge-annotations.json');
  }

  return options;
}

function printHelp(): void {
  const help = `
Edge annotation CLI for fragment images.

Usage:
  node scripts/annotate-edges.ts [options]
  npm run annotate:edges -- [options]

Options:
  --data-root <path>          Override the image root. Default: src/web/web-canvas/data
  --annotations-file <path>   Override the JSON annotations file
  --fragments-db <path>       Override fragments.db used for targeted ordering
  --collection <name>         Restrict to one collection folder, e.g. BLL42
  --match <text>              Restrict to paths containing text
  --start <value>             Start at 1-based index, exact path, or "first-unreviewed"
  --seed <value>              Seed for balanced random ordering. Default: ${DEFAULT_RANDOM_SEED}
  --targeted                  Prioritize likely underrepresented edge cases using fragments.db
  --targets <csv>             Focus targeted mode on: right,left,corner,top,bottom (default: auto)
  --ordered                   Disable balanced random ordering and use sorted paths
  --open                      Open each image in the OS viewer when selected
  --stats                     Print summary stats and exit
  --export-csv <path>         Write current annotations to CSV and exit
  --help, -h                  Show this help

Interactive commands:
  t | b | l | r              Cycle top / bottom / left / right edge: ? -> yes -> no -> ?
  d                          Mark reviewed and move to next image
  x                          Toggle skipped, save, and move to next image when skipping
  n | p                      Next / previous image
  u                          Jump to next unreviewed image
  o                          Open current image in the OS viewer
  note <text>                Set note text
  clear-note                 Remove note text
  reset                      Clear the current annotation record
  goto <index|path text>     Jump by 1-based index or first path match
  stats                      Print progress summary
  target-stats               Print live target coverage and remaining candidate counts
  save                       Force-save annotations JSON
  help                       Show commands
  q                          Quit

Notes:
  - Annotations autosave on every change.
  - Records are stored in data/edge-annotations.json by relative image path.
  - By default, images are shown in a balanced random order across collections.
  - Targeted mode uses fragments.db edge priors and your reviewed-label counts to bias toward underrepresented cases.
  - If the local data folder is empty, the tool will try nearby sibling repos before failing.
`;
  console.log(help.trim());
}

function parseTargetLabels(raw: string): TargetLabel[] {
  const normalized = raw
    .split(',')
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean);

  if (normalized.length === 0 || normalized.includes('auto')) {
    return [];
  }

  const labels: TargetLabel[] = [];
  const seen = new Set<TargetLabel>();
  const aliases: Record<string, TargetLabel> = {
    t: 'top',
    top: 'top',
    b: 'bottom',
    bottom: 'bottom',
    l: 'left',
    left: 'left',
    r: 'right',
    right: 'right',
    c: 'corner',
    corner: 'corner',
  };

  for (const value of normalized) {
    const label = aliases[value];
    if (!label) {
      throw new Error(`Unknown target label: ${value}`);
    }
    if (!seen.has(label)) {
      labels.push(label);
      seen.add(label);
    }
  }

  return labels;
}

function listImages(dataRoot: string): ImageItem[] {
  const items: ImageItem[] = [];
  const stack = [dataRoot];

  while (stack.length > 0) {
    const current = stack.pop();
    if (!current) {
      continue;
    }

    const entries = fs.readdirSync(current, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(fullPath);
        continue;
      }

      if (!entry.isFile()) {
        continue;
      }

      const extension = path.extname(entry.name).toLowerCase();
      if (!IMAGE_EXTENSIONS.has(extension)) {
        continue;
      }

      const relativePath = toStoragePath(path.relative(dataRoot, fullPath));
      const [collection = '', filename = entry.name] = relativePath.split(/\/(.+)/, 2);
      items.push({
        absolutePath: fullPath,
        relativePath,
        collection,
        filename,
      });
    }
  }

  return items.sort((left, right) => left.relativePath.localeCompare(right.relativePath));
}

function findNearbyDataRoots(): string[] {
  const candidates = new Set<string>();
  const repoParent = path.dirname(REPO_ROOT);
  const capstoneData = path.join(os.homedir(), 'capstone_data');

  if (fs.existsSync(capstoneData)) {
    candidates.add(path.resolve(capstoneData));
  }

  if (fs.existsSync(repoParent)) {
    const entries = fs.readdirSync(repoParent, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) {
        continue;
      }
      const candidate = path.join(repoParent, entry.name, 'src', 'web', 'web-canvas', 'data');
      if (fs.existsSync(candidate)) {
        candidates.add(path.resolve(candidate));
      }
    }
  }

  return [...candidates];
}

function findNearbyFragmentsDbs(): string[] {
  const candidates = new Set<string>();
  const repoParent = path.dirname(REPO_ROOT);

  const localDefault = path.resolve(DEFAULT_FRAGMENTS_DB);
  if (fs.existsSync(localDefault)) {
    candidates.add(localDefault);
  }

  if (fs.existsSync(repoParent)) {
    const entries = fs.readdirSync(repoParent, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory()) {
        continue;
      }
      const candidate = path.join(
        repoParent,
        entry.name,
        'src',
        'web',
        'web-canvas',
        'electron',
        'resources',
        'database',
        'fragments.db'
      );
      if (fs.existsSync(candidate)) {
        candidates.add(path.resolve(candidate));
      }
    }
  }

  return [...candidates];
}

function formatPathList(paths: string[]): string {
  return paths.map((candidate) => `  - ${candidate}`).join('\n');
}

function resolveDataInputs(options: Options): {
  dataRoot: string;
  annotationsFile: string;
  allImages: ImageItem[];
  autoDetected: boolean;
} {
  if (!fs.existsSync(options.dataRoot)) {
    throw new Error(`Data root not found: ${options.dataRoot}`);
  }

  const currentRoot = path.resolve(options.dataRoot);
  let allImages = listImages(currentRoot);
  if (allImages.length > 0) {
    return {
      dataRoot: currentRoot,
      annotationsFile: options.annotationsFile,
      allImages,
      autoDetected: false,
    };
  }

  if (options.dataRootExplicit) {
    throw new Error(
      `Data root exists but contains no supported images: ${currentRoot}\n` +
      'Use --data-root to point at a directory containing collection folders such as BLL10.'
    );
  }

  const alternates = findNearbyDataRoots().filter((candidate) => candidate !== currentRoot);
  const populatedAlternates = alternates.filter((candidate) => listImages(candidate).length > 0);

  if (populatedAlternates.length === 1) {
    const dataRoot = populatedAlternates[0];
    allImages = listImages(dataRoot);
    return {
      dataRoot,
      annotationsFile: options.annotationsFileExplicit
        ? options.annotationsFile
        : path.join(dataRoot, 'edge-annotations.json'),
      allImages,
      autoDetected: true,
    };
  }

  if (populatedAlternates.length > 1) {
    throw new Error(
      `Default data root contains no supported images: ${currentRoot}\n` +
      'Multiple nearby image roots were found. Re-run with --data-root using one of:\n' +
      formatPathList(populatedAlternates)
    );
  }

  throw new Error(
    `Default data root contains no supported images: ${currentRoot}\n` +
    'No nearby populated image roots were found. Re-run with --data-root pointing at your extracted corpus.'
  );
}

function resolveFragmentsDb(options: Options, dataRoot: string): string | undefined {
  const explicitDb = options.fragmentsDb ? path.resolve(options.fragmentsDb) : undefined;

  if (options.fragmentsDbExplicit) {
    if (!explicitDb || !fs.existsSync(explicitDb)) {
      throw new Error(`fragments.db not found: ${options.fragmentsDb}`);
    }
    return explicitDb;
  }

  const siblingDb = path.join(path.dirname(dataRoot), 'electron', 'resources', 'database', 'fragments.db');
  if (fs.existsSync(siblingDb)) {
    return path.resolve(siblingDb);
  }

  const alternates = findNearbyFragmentsDbs();
  if (alternates.length === 1) {
    return alternates[0];
  }

  if (alternates.length > 1) {
    const matchingPrefix = alternates.find((candidate) =>
      candidate.startsWith(path.dirname(path.dirname(path.dirname(dataRoot))))
    );
    if (matchingPrefix) {
      return matchingPrefix;
    }
  }

  return undefined;
}

function createEmptyStore(dataRoot: string): AnnotationStore {
  const stamp = nowIso();
  return {
    version: 1,
    data_root: toStoragePath(path.resolve(dataRoot)),
    created_at: stamp,
    updated_at: stamp,
    annotations: {},
  };
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function toEdgeValue(value: unknown): EdgeValue {
  if (value === true || value === false) {
    return value;
  }
  return null;
}

function normalizeRecord(value: unknown): AnnotationRecord {
  if (!isObject(value)) {
    return {
      has_top_edge: null,
      has_bottom_edge: null,
      has_left_edge: null,
      has_right_edge: null,
      reviewed: false,
      skipped: false,
      notes: '',
      updated_at: nowIso(),
    };
  }

  return {
    has_top_edge: toEdgeValue(value.has_top_edge),
    has_bottom_edge: toEdgeValue(value.has_bottom_edge),
    has_left_edge: toEdgeValue(value.has_left_edge),
    has_right_edge: toEdgeValue(value.has_right_edge),
    reviewed: Boolean(value.reviewed),
    skipped: Boolean(value.skipped),
    notes: typeof value.notes === 'string' ? value.notes : '',
    updated_at: typeof value.updated_at === 'string' ? value.updated_at : nowIso(),
  };
}

function loadStore(annotationsFile: string, dataRoot: string): AnnotationStore {
  if (!fs.existsSync(annotationsFile)) {
    return createEmptyStore(dataRoot);
  }

  const raw = fs.readFileSync(annotationsFile, 'utf-8');
  const parsed = JSON.parse(raw) as unknown;
  if (!isObject(parsed)) {
    return createEmptyStore(dataRoot);
  }

  const annotations: Record<string, AnnotationRecord> = {};
  const parsedAnnotations = isObject(parsed.annotations) ? parsed.annotations : {};
  for (const [relativePath, record] of Object.entries(parsedAnnotations)) {
    annotations[toStoragePath(relativePath)] = normalizeRecord(record);
  }

  return {
    version: typeof parsed.version === 'number' ? parsed.version : 1,
    data_root: typeof parsed.data_root === 'string'
      ? parsed.data_root
      : toStoragePath(path.resolve(dataRoot)),
    created_at: typeof parsed.created_at === 'string' ? parsed.created_at : nowIso(),
    updated_at: typeof parsed.updated_at === 'string' ? parsed.updated_at : nowIso(),
    annotations,
  };
}

function saveStore(annotationsFile: string, store: AnnotationStore): void {
  store.updated_at = nowIso();
  fs.mkdirSync(path.dirname(annotationsFile), { recursive: true });
  const tempFile = `${annotationsFile}.tmp`;
  fs.writeFileSync(tempFile, `${JSON.stringify(store, null, 2)}\n`, 'utf-8');
  fs.renameSync(tempFile, annotationsFile);
}

function getRecord(store: AnnotationStore, relativePath: string): AnnotationRecord {
  if (!store.annotations[relativePath]) {
    store.annotations[relativePath] = normalizeRecord(undefined);
  }
  return store.annotations[relativePath];
}

function cycleValue(value: EdgeValue): EdgeValue {
  if (value === null) {
    return true;
  }
  if (value === true) {
    return false;
  }
  return null;
}

function edgeValueLabel(value: EdgeValue): string {
  if (value === true) {
    return 'Y';
  }
  if (value === false) {
    return 'N';
  }
  return '?';
}

function deriveEdgePiece(record: AnnotationRecord): EdgeValue {
  const values = EDGE_KEYS.map((key) => record[key]);
  if (values.some((value) => value === true)) {
    return true;
  }
  if (values.every((value) => value === false)) {
    return false;
  }
  return null;
}

function filterImages(images: ImageItem[], options: Options): ImageItem[] {
  const normalizedCollection = options.collection?.toLowerCase();
  const normalizedMatch = options.match?.toLowerCase();

  return images.filter((image) => {
    if (normalizedCollection && image.collection.toLowerCase() !== normalizedCollection) {
      return false;
    }
    if (normalizedMatch && !image.relativePath.toLowerCase().includes(normalizedMatch)) {
      return false;
    }
    return true;
  });
}

function collectionFromRelativePath(relativePath: string): string {
  return relativePath.split(/\/(.+)/, 2)[0] || relativePath;
}

function createSeededRandom(seed: string): () => number {
  let h = 1779033703 ^ seed.length;
  for (let i = 0; i < seed.length; i += 1) {
    h = Math.imul(h ^ seed.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }

  return () => {
    h = Math.imul(h ^ (h >>> 16), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    const value = (h ^= h >>> 16) >>> 0;
    h = (h + 0x6D2B79F5) >>> 0;
    return value / 4294967296;
  };
}

function shuffleInPlace<T>(items: T[], random: () => number): void {
  for (let index = items.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    const current = items[index];
    items[index] = items[swapIndex];
    items[swapIndex] = current;
  }
}

function getReviewedCountsByCollection(store: AnnotationStore): Map<string, number> {
  const counts = new Map<string, number>();
  for (const [relativePath, record] of Object.entries(store.annotations)) {
    if (!record.reviewed || record.skipped) {
      continue;
    }
    const collection = collectionFromRelativePath(relativePath);
    counts.set(collection, (counts.get(collection) ?? 0) + 1);
  }
  return counts;
}

function toDbBoolean(value: unknown): boolean | null {
  if (value === 0 || value === false) {
    return false;
  }
  if (value === 1 || value === true) {
    return true;
  }
  return null;
}

function loadDbEdgePredictions(
  fragmentsDb: string,
  images: ImageItem[]
): Map<string, DbEdgePrediction> {
  const wantedPaths = new Set(images.map((image) => image.relativePath));
  const predictions = new Map<string, DbEdgePrediction>();
  const queryScript = `
import json
import sqlite3
import sys

conn = sqlite3.connect(sys.argv[1])
conn.row_factory = sqlite3.Row
rows = conn.execute(
    """
    SELECT
      REPLACE(image_path, '\\\\', '/') AS image_path,
      has_top_edge,
      has_bottom_edge,
      has_left_edge,
      has_right_edge,
      edge_piece
    FROM fragments
    """
).fetchall()
json.dump([dict(row) for row in rows], sys.stdout)
`;
  let raw = '';
  try {
    raw = execFileSync('python3', ['-c', queryScript, fragmentsDb], {
      encoding: 'utf-8',
      maxBuffer: 64 * 1024 * 1024,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Failed to read edge priors from fragments.db via python3: ${message}`);
  }
  const rows = JSON.parse(raw) as Array<Record<string, unknown>>;

  for (const row of rows) {
    const imagePath = String(row.image_path ?? '');
    if (!wantedPaths.has(imagePath)) {
      continue;
    }

    predictions.set(imagePath, {
      has_top_edge: toDbBoolean(row.has_top_edge),
      has_bottom_edge: toDbBoolean(row.has_bottom_edge),
      has_left_edge: toDbBoolean(row.has_left_edge),
      has_right_edge: toDbBoolean(row.has_right_edge),
      edge_piece: toDbBoolean(row.edge_piece),
    });
  }

  return predictions;
}

function countTrueEdges(prediction: DbEdgePrediction): number {
  return EDGE_KEYS.reduce((count, key) => count + (prediction[key] === true ? 1 : 0), 0);
}

function predictionHasLabel(prediction: DbEdgePrediction, label: TargetLabel): boolean {
  if (label === 'corner') {
    return countTrueEdges(prediction) >= 2;
  }

  const keyByLabel: Record<Exclude<TargetLabel, 'corner'>, EdgeKey> = {
    top: 'has_top_edge',
    bottom: 'has_bottom_edge',
    left: 'has_left_edge',
    right: 'has_right_edge',
  };

  return prediction[keyByLabel[label]] === true;
}

function getReviewedPositiveCounts(store: AnnotationStore): Map<TargetLabel, number> {
  const counts = new Map<TargetLabel, number>([
    ['top', 0],
    ['bottom', 0],
    ['left', 0],
    ['right', 0],
    ['corner', 0],
  ]);

  for (const record of Object.values(store.annotations)) {
    if (!record.reviewed || record.skipped) {
      continue;
    }

    if (record.has_top_edge === true) {
      counts.set('top', (counts.get('top') ?? 0) + 1);
    }
    if (record.has_bottom_edge === true) {
      counts.set('bottom', (counts.get('bottom') ?? 0) + 1);
    }
    if (record.has_left_edge === true) {
      counts.set('left', (counts.get('left') ?? 0) + 1);
    }
    if (record.has_right_edge === true) {
      counts.set('right', (counts.get('right') ?? 0) + 1);
    }

    const positiveEdges = EDGE_KEYS.filter((key) => record[key] === true).length;
    if (positiveEdges >= 2) {
      counts.set('corner', (counts.get('corner') ?? 0) + 1);
    }
  }

  return counts;
}

function formatReviewedPositiveCounts(store: AnnotationStore): string {
  const counts = getReviewedPositiveCounts(store);
  return ['top', 'bottom', 'left', 'right', 'corner']
    .map((label) => `${label}=${counts.get(label as TargetLabel) ?? 0}`)
    .join(' ');
}

function summarizeTargets(
  labels: TargetLabel[],
  images: ImageItem[],
  predictions: Map<string, DbEdgePrediction>,
  store: AnnotationStore
): TargetSummary[] {
  const reviewedCounts = getReviewedPositiveCounts(store);
  return labels.map((label) => {
    let predictedPositives = 0;
    for (const image of images) {
      const prediction = predictions.get(image.relativePath);
      if (prediction && predictionHasLabel(prediction, label)) {
        predictedPositives += 1;
      }
    }

    return {
      label,
      reviewedPositives: reviewedCounts.get(label) ?? 0,
      predictedPositives,
    };
  });
}

function getUnreviewedImages(images: ImageItem[], store: AnnotationStore): ImageItem[] {
  return images.filter((image) => !getRecord(store, image.relativePath).reviewed);
}

function getCurrentTargetSummaries(
  images: ImageItem[],
  store: AnnotationStore,
  targeting?: TargetingContext
): TargetSummary[] {
  if (!targeting) {
    return [];
  }

  return summarizeTargets(
    targeting.targetLabels,
    getUnreviewedImages(images, store),
    targeting.predictions,
    store
  );
}

function formatTargetSummary(summaries: TargetSummary[]): string {
  return summaries
    .map((summary) => `${summary.label} reviewed=${summary.reviewedPositives} candidates=${summary.predictedPositives}`)
    .join(' | ');
}

function printTargetStats(
  images: ImageItem[],
  store: AnnotationStore,
  targeting?: TargetingContext
): void {
  console.log(`Reviewed positives: ${formatReviewedPositiveCounts(store)}`);
  if (!targeting) {
    console.log('Targeted mode is not active. Launch with --targeted to see candidate counts.');
    return;
  }

  const summaries = getCurrentTargetSummaries(images, store, targeting);
  console.log(`Target focus: ${formatTargetSummary(summaries) || '(no positive target candidates found)'}`);
  console.log(`fragments.db: ${targeting.fragmentsDb}`);
}

function chooseAutoTargetLabels(
  images: ImageItem[],
  predictions: Map<string, DbEdgePrediction>,
  store: AnnotationStore
): TargetLabel[] {
  const summaries = summarizeTargets(['right', 'left', 'corner', 'top', 'bottom'], images, predictions, store)
    .filter((summary) => summary.predictedPositives > 0)
    .sort((left, right) => {
      const reviewedDiff = left.reviewedPositives - right.reviewedPositives;
      if (reviewedDiff !== 0) {
        return reviewedDiff;
      }
      const predictedDiff = right.predictedPositives - left.predictedPositives;
      if (predictedDiff !== 0) {
        return predictedDiff;
      }
      return left.label.localeCompare(right.label);
    });

  return summaries.slice(0, 3).map((summary) => summary.label);
}

function rankImagesForTargets(
  images: ImageItem[],
  predictions: Map<string, DbEdgePrediction>,
  targetSummaries: TargetSummary[]
): { hits: ImageItem[]; misses: ImageItem[] } {
  if (images.length === 0 || targetSummaries.length === 0) {
    return { hits: [], misses: [...images] };
  }

  const maxReviewed = Math.max(...targetSummaries.map((summary) => summary.reviewedPositives), 1);
  const scoreForLabel = new Map<TargetLabel, number>();
  for (const summary of targetSummaries) {
    scoreForLabel.set(summary.label, Math.max(1, maxReviewed - summary.reviewedPositives));
  }

  const hits: ImageItem[] = [];
  const misses: ImageItem[] = [];

  for (const image of images) {
    const prediction = predictions.get(image.relativePath);
    if (!prediction) {
      misses.push(image);
      continue;
    }

    let score = 0;
    const tags: string[] = [];
    for (const summary of targetSummaries) {
      if (!predictionHasLabel(prediction, summary.label)) {
        continue;
      }
      const weight = scoreForLabel.get(summary.label) ?? 1;
      score += summary.label === 'corner' ? weight * 1.15 : weight;
      tags.push(summary.label);
    }

    if (score > 0) {
      hits.push({ ...image, targetScore: score, targetTags: tags });
    } else {
      misses.push(image);
    }
  }

  return { hits, misses };
}

function buildTargetedCollectionOrder(
  images: ImageItem[],
  store: AnnotationStore,
  seed: string
): ImageItem[] {
  if (images.length <= 1) {
    return [...images];
  }

  const byCollection = new Map<string, ImageItem[]>();
  for (const image of images) {
    const items = byCollection.get(image.collection) ?? [];
    items.push(image);
    byCollection.set(image.collection, items);
  }

  for (const [collection, items] of byCollection.entries()) {
    const random = createSeededRandom(`${seed}::items::${collection}`);
    const tieBreaker = new Map(
      items.map((image) => [image.relativePath, random()])
    );
    items.sort((left, right) => {
      const scoreDiff = (right.targetScore ?? 0) - (left.targetScore ?? 0);
      if (scoreDiff !== 0) {
        return scoreDiff;
      }
      const tieDiff = (tieBreaker.get(left.relativePath) ?? 0) - (tieBreaker.get(right.relativePath) ?? 0);
      if (tieDiff !== 0) {
        return tieDiff;
      }
      return left.relativePath.localeCompare(right.relativePath);
    });
  }

  const reviewedCounts = getReviewedCountsByCollection(store);
  const collectionOrder = [...byCollection.keys()];
  const tieBreakers = new Map(
    collectionOrder.map((collection) => [
      collection,
      createSeededRandom(`${seed}::collection-rank::${collection}`)(),
    ])
  );

  collectionOrder.sort((left, right) => {
    const reviewedDiff = (reviewedCounts.get(left) ?? 0) - (reviewedCounts.get(right) ?? 0);
    if (reviewedDiff !== 0) {
      return reviewedDiff;
    }

    const bestLeft = byCollection.get(left)?.[0]?.targetScore ?? 0;
    const bestRight = byCollection.get(right)?.[0]?.targetScore ?? 0;
    const scoreDiff = bestRight - bestLeft;
    if (scoreDiff !== 0) {
      return scoreDiff;
    }

    const tieDiff = (tieBreakers.get(left) ?? 0) - (tieBreakers.get(right) ?? 0);
    if (tieDiff !== 0) {
      return tieDiff;
    }

    return left.localeCompare(right);
  });

  const ordered: ImageItem[] = [];
  let madeProgress = true;

  while (madeProgress) {
    madeProgress = false;
    for (const collection of collectionOrder) {
      const items = byCollection.get(collection);
      if (!items || items.length === 0) {
        continue;
      }
      const next = items.shift();
      if (!next) {
        continue;
      }
      ordered.push(next);
      madeProgress = true;
    }
  }

  return ordered;
}

function isReviewed(store: AnnotationStore, relativePath: string): boolean {
  return Boolean(store.annotations[relativePath]?.reviewed);
}

function buildBalancedRandomOrder(
  images: ImageItem[],
  store: AnnotationStore,
  seed: string
): ImageItem[] {
  if (images.length <= 1) {
    return [...images];
  }

  const byCollection = new Map<string, ImageItem[]>();
  for (const image of images) {
    const items = byCollection.get(image.collection) ?? [];
    items.push(image);
    byCollection.set(image.collection, items);
  }

  for (const [collection, items] of byCollection.entries()) {
    shuffleInPlace(items, createSeededRandom(`${seed}::items::${collection}`));
  }

  const reviewedCounts = getReviewedCountsByCollection(store);
  const collectionOrder = [...byCollection.keys()];
  const tieBreakers = new Map(
    collectionOrder.map((collection) => [
      collection,
      createSeededRandom(`${seed}::collection-rank::${collection}`)(),
    ])
  );

  collectionOrder.sort((left, right) => {
    const countDiff = (reviewedCounts.get(left) ?? 0) - (reviewedCounts.get(right) ?? 0);
    if (countDiff !== 0) {
      return countDiff;
    }

    const tieDiff = (tieBreakers.get(left) ?? 0) - (tieBreakers.get(right) ?? 0);
    if (tieDiff !== 0) {
      return tieDiff;
    }

    return left.localeCompare(right);
  });

  const ordered: ImageItem[] = [];
  let madeProgress = true;

  while (madeProgress) {
    madeProgress = false;
    for (const collection of collectionOrder) {
      const items = byCollection.get(collection);
      if (!items || items.length === 0) {
        continue;
      }

      const image = items.shift();
      if (!image) {
        continue;
      }

      ordered.push(image);
      madeProgress = true;
    }
  }

  return ordered;
}

function orderImages(
  images: ImageItem[],
  store: AnnotationStore,
  options: Options,
  targeting?: TargetingContext
): ImageItem[] {
  if (options.orderMode === 'ordered') {
    return images;
  }

  const unreviewed: ImageItem[] = [];
  const reviewed: ImageItem[] = [];

  for (const image of images) {
    if (isReviewed(store, image.relativePath)) {
      reviewed.push(image);
    } else {
      unreviewed.push(image);
    }
  }

  if (options.orderMode === 'targeted' && targeting) {
    const ranked = rankImagesForTargets(
      unreviewed,
      targeting.predictions,
      summarizeTargets(targeting.targetLabels, unreviewed, targeting.predictions, store)
    );

    return [
      ...buildTargetedCollectionOrder(ranked.hits, store, `${options.seed}::targeted-hits`),
      ...buildBalancedRandomOrder(ranked.misses, store, `${options.seed}::targeted-misses`),
      ...buildBalancedRandomOrder(reviewed, store, `${options.seed}::reviewed`),
    ];
  }

  return [
    ...buildBalancedRandomOrder(unreviewed, store, `${options.seed}::unreviewed`),
    ...buildBalancedRandomOrder(reviewed, store, `${options.seed}::reviewed`),
  ];
}

function getStats(images: ImageItem[], store: AnnotationStore): {
  total: number;
  reviewed: number;
  skipped: number;
  remaining: number;
} {
  let reviewed = 0;
  let skipped = 0;

  for (const image of images) {
    const record = store.annotations[image.relativePath];
    if (!record) {
      continue;
    }
    if (record.reviewed) {
      reviewed += 1;
    }
    if (record.skipped) {
      skipped += 1;
    }
  }

  return {
    total: images.length,
    reviewed,
    skipped,
    remaining: images.length - reviewed,
  };
}

function printStats(images: ImageItem[], store: AnnotationStore): void {
  const stats = getStats(images, store);
  console.log(`Images:    ${stats.total}`);
  console.log(`Reviewed:  ${stats.reviewed}`);
  console.log(`Skipped:   ${stats.skipped}`);
  console.log(`Remaining: ${stats.remaining}`);
}

function csvEscape(value: string): string {
  if (/[",\n]/.test(value)) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

function toCsvValue(value: EdgeValue | boolean | string): string {
  if (typeof value === 'boolean') {
    return value ? 'true' : 'false';
  }
  if (value === null) {
    return '';
  }
  return String(value);
}

function exportCsv(csvPath: string, images: ImageItem[], store: AnnotationStore): void {
  const lines = [
    [
      'relative_path',
      'collection',
      'filename',
      'has_top_edge',
      'has_bottom_edge',
      'has_left_edge',
      'has_right_edge',
      'edge_piece',
      'reviewed',
      'skipped',
      'notes',
      'updated_at',
    ].join(','),
  ];

  for (const image of images) {
    const record = getRecord(store, image.relativePath);
    const row = [
      image.relativePath,
      image.collection,
      image.filename,
      toCsvValue(record.has_top_edge),
      toCsvValue(record.has_bottom_edge),
      toCsvValue(record.has_left_edge),
      toCsvValue(record.has_right_edge),
      toCsvValue(deriveEdgePiece(record)),
      toCsvValue(record.reviewed),
      toCsvValue(record.skipped),
      record.notes,
      record.updated_at,
    ].map(csvEscape);
    lines.push(row.join(','));
  }

  fs.mkdirSync(path.dirname(csvPath), { recursive: true });
  fs.writeFileSync(csvPath, `${lines.join('\n')}\n`, 'utf-8');
}

function openImage(imagePath: string): void {
  let command = '';
  let args: string[] = [];

  if (process.platform === 'darwin') {
    command = 'open';
    args = [imagePath];
  } else if (process.platform === 'win32') {
    command = 'cmd';
    args = ['/c', 'start', '', imagePath];
  } else {
    command = 'xdg-open';
    args = [imagePath];
  }

  const child = spawn(command, args, {
    detached: true,
    stdio: 'ignore',
  });
  child.unref();
}

function findStartIndex(images: ImageItem[], store: AnnotationStore, start?: string): number {
  if (images.length === 0) {
    return 0;
  }

  if (!start || start === 'first-unreviewed') {
    const index = images.findIndex((image) => !getRecord(store, image.relativePath).reviewed);
    return index >= 0 ? index : 0;
  }

  const numeric = Number(start);
  if (Number.isInteger(numeric) && numeric >= 1 && numeric <= images.length) {
    return numeric - 1;
  }

  const exactIndex = images.findIndex((image) => image.relativePath === start);
  if (exactIndex >= 0) {
    return exactIndex;
  }

  const fuzzy = start.toLowerCase();
  const fuzzyIndex = images.findIndex((image) => image.relativePath.toLowerCase().includes(fuzzy));
  return fuzzyIndex >= 0 ? fuzzyIndex : 0;
}

function clampIndex(index: number, images: ImageItem[]): number {
  if (images.length === 0) {
    return 0;
  }
  if (index < 0) {
    return 0;
  }
  if (index >= images.length) {
    return images.length - 1;
  }
  return index;
}

function printCurrent(images: ImageItem[], index: number, store: AnnotationStore): void {
  const image = images[index];
  const record = getRecord(store, image.relativePath);
  const edgePiece = deriveEdgePiece(record);
  console.log('');
  console.log(`Image ${index + 1}/${images.length}`);
  console.log(`Path:      ${image.relativePath}`);
  console.log(`Collection:${image.collection}`);
  console.log(`Absolute:  ${image.absolutePath}`);
  console.log(
    `Edges:     T=${edgeValueLabel(record.has_top_edge)} ` +
    `B=${edgeValueLabel(record.has_bottom_edge)} ` +
    `L=${edgeValueLabel(record.has_left_edge)} ` +
    `R=${edgeValueLabel(record.has_right_edge)} ` +
    `| edge_piece=${edgeValueLabel(edgePiece)}`
  );
  if (image.targetTags && image.targetTags.length > 0) {
    console.log(`Target:    ${image.targetTags.join(', ')}${image.targetScore ? ` | score=${image.targetScore.toFixed(1)}` : ''}`);
  }
  console.log(`Status:    reviewed=${record.reviewed ? 'yes' : 'no'} skipped=${record.skipped ? 'yes' : 'no'}`);
  console.log(`Notes:     ${record.notes || '(none)'}`);
}

function printInteractiveHelp(): void {
  console.log('Commands: t b l r | d | x | n | p | u | o | auto-open | note <text> | clear-note | reset | goto <index|text> | stats | target-stats | save | help | q');
}

async function runInteractive(
  images: ImageItem[],
  store: AnnotationStore,
  options: Options,
  targeting?: TargetingContext
): Promise<void> {
  const rl = readline.createInterface({ input, output });
  let index = findStartIndex(images, store, options.start);
  let autoOpenImage = options.autoOpen;

  const showCurrent = (shouldOpen: boolean): void => {
    index = clampIndex(index, images);
    printCurrent(images, index, store);
    if (autoOpenImage && shouldOpen) {
      openImage(images[index].absolutePath);
    }
    printInteractiveHelp();
  };

  showCurrent(true);

  try {
    while (true) {
      const answer = await rl.question('annotate> ');
      const trimmed = answer.trim();
      const currentImage = images[index];
      const record = getRecord(store, currentImage.relativePath);

      if (trimmed === '') {
        index = clampIndex(index + 1, images);
        showCurrent(true);
        continue;
      }

      const [command, ...rest] = trimmed.split(' ');
      const normalized = command.toLowerCase();
      const value = rest.join(' ').trim();

      if (normalized === 'q' || normalized === 'quit' || normalized === 'exit') {
        break;
      }

      if (normalized === 'help' || normalized === 'h') {
        printInteractiveHelp();
        continue;
      }

      if (normalized === 'stats') {
        printStats(images, store);
        continue;
      }

      if (normalized === 'target-stats' || normalized === 'ts') {
        printTargetStats(images, store, targeting);
        continue;
      }

      if (normalized === 'save') {
        saveStore(options.annotationsFile, store);
        console.log(`Saved ${options.annotationsFile}`);
        continue;
      }

      if (normalized === 'o' || normalized === 'open') {
        openImage(currentImage.absolutePath);
        continue;
      }

      if (normalized === 'n' || normalized === 'next') {
        index = clampIndex(index + 1, images);
        showCurrent(true);
        continue;
      }

      if (normalized === 'p' || normalized === 'prev' || normalized === 'previous') {
        index = clampIndex(index - 1, images);
        showCurrent(true);
        continue;
      }

      if (normalized === 'u' || normalized === 'unreviewed') {
        const nextIndex = images.findIndex(
          (image, candidateIndex) =>
            candidateIndex > index && !getRecord(store, image.relativePath).reviewed
        );
        if (nextIndex >= 0) {
          index = nextIndex;
          showCurrent(true);
        } else {
          console.log('No later unreviewed image found.');
        }
        continue;
      }

      if (normalized === 'goto') {
        if (!value) {
          console.log('Usage: goto <index|path text>');
          continue;
        }

        const numeric = Number(value);
        if (Number.isInteger(numeric) && numeric >= 1 && numeric <= images.length) {
          index = numeric - 1;
          showCurrent(true);
          continue;
        }

        const matchValue = value.toLowerCase();
        const foundIndex = images.findIndex((image) =>
          image.relativePath.toLowerCase().includes(matchValue)
        );
        if (foundIndex >= 0) {
          index = foundIndex;
          showCurrent(true);
        } else {
          console.log(`No image matched "${value}".`);
        }
        continue;
      }

      if (normalized === 'note') {
        record.notes = value;
        record.updated_at = nowIso();
        saveStore(options.annotationsFile, store);
        console.log('Note updated.');
        continue;
      }

      if (normalized === 'clear-note') {
        record.notes = '';
        record.updated_at = nowIso();
        saveStore(options.annotationsFile, store);
        console.log('Note cleared.');
        continue;
      }

      if (normalized === 'reset') {
        store.annotations[currentImage.relativePath] = normalizeRecord(undefined);
        saveStore(options.annotationsFile, store);
        console.log('Annotation reset.');
        showCurrent(false);
        continue;
      }

      if (normalized === 'd' || normalized === 'done') {
        record.reviewed = true;
        record.skipped = false;
        record.updated_at = nowIso();
        saveStore(options.annotationsFile, store);
        index = clampIndex(index + 1, images);
        showCurrent(true);
        continue;
      }

      if (normalized === 'x' || normalized === 'skip') {
        record.skipped = !record.skipped;
        record.reviewed = record.skipped;
        record.updated_at = nowIso();
        saveStore(options.annotationsFile, store);
        console.log(`Skipped: ${record.skipped ? 'yes' : 'no'}`);
        if (record.skipped) {
          index = clampIndex(index + 1, images);
          showCurrent(true);
        }
        continue;
      }

      const edgeMap: Record<string, EdgeKey> = {
        t: 'has_top_edge',
        top: 'has_top_edge',
        b: 'has_bottom_edge',
        bottom: 'has_bottom_edge',
        l: 'has_left_edge',
        left: 'has_left_edge',
        r: 'has_right_edge',
        right: 'has_right_edge',
      };
      const edgeKey = edgeMap[normalized];
      if (edgeKey) {
        record[edgeKey] = cycleValue(record[edgeKey]);
        record.updated_at = nowIso();
        saveStore(options.annotationsFile, store);
        showCurrent(false);
        continue;
      }

      if (normalized === 'auto-open') {
        autoOpenImage = !autoOpenImage;
        console.log(`Auto-open: ${autoOpenImage ? 'on' : 'off'}`);
        continue;
      }

      console.log(`Unknown command: ${trimmed}`);
    }
  } finally {
    rl.close();
  }
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const resolved = resolveDataInputs(options);
  options.dataRoot = resolved.dataRoot;
  options.annotationsFile = resolved.annotationsFile;
  if (options.orderMode === 'targeted') {
    options.fragmentsDb = resolveFragmentsDb(options, resolved.dataRoot);
    if (!options.fragmentsDb) {
      console.warn('Targeted mode requested but no fragments.db was resolved. Falling back to balanced order.');
      options.orderMode = 'balanced';
    }
  }
  const store = loadStore(resolved.annotationsFile, resolved.dataRoot);
  const allImages = resolved.allImages;
  const filteredImages = filterImages(allImages, options);
  let targeting: TargetingContext | undefined;
  if (options.orderMode === 'targeted' && options.fragmentsDb) {
    const predictions = loadDbEdgePredictions(options.fragmentsDb, filteredImages);
    const unreviewed = getUnreviewedImages(filteredImages, store);
    const targetLabels = options.targetLabels.length > 0
      ? options.targetLabels
      : chooseAutoTargetLabels(unreviewed, predictions, store);
    targeting = {
      fragmentsDb: options.fragmentsDb,
      predictions,
      targetLabels,
    };
  }
  const images = orderImages(filteredImages, store, options, targeting);

  if (images.length === 0) {
    throw new Error(
      `No images matched the requested filters in ${resolved.dataRoot}.\n` +
      `Collection filter: ${options.collection ?? '(none)'}\n` +
      `Match filter: ${options.match ?? '(none)'}`
    );
  }

  if (options.statsOnly) {
    printStats(images, store);
    if (options.orderMode === 'targeted') {
      printTargetStats(images, store, targeting);
    }
    return;
  }

  if (options.exportCsv) {
    exportCsv(options.exportCsv, images, store);
    console.log(`Wrote ${options.exportCsv}`);
    return;
  }

  if (resolved.autoDetected) {
    console.log(`Resolved image root automatically: ${resolved.dataRoot}`);
  }
  console.log(`Loaded ${images.length} images from ${resolved.dataRoot}`);
  console.log(`Annotations file: ${resolved.annotationsFile}`);
  if (options.orderMode === 'targeted') {
    console.log(`Order: targeted across ${new Set(images.map((image) => image.collection)).size} collections (seed=${options.seed})`);
    printTargetStats(images, store, targeting);
  } else if (options.orderMode === 'balanced') {
    const collectionCount = new Set(images.map((image) => image.collection)).size;
    console.log(`Order: balanced-random across ${collectionCount} collections (seed=${options.seed})`);
  } else {
    console.log('Order: sorted by relative path');
  }
  await runInteractive(images, store, options, targeting);
  saveStore(options.annotationsFile, store);
  console.log('Saved annotations and exited.');
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
