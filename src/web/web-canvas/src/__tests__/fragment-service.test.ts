import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { mapToManuscriptFragment } from '../services/fragment-service';
import type { FragmentRecord } from '../services/electron-api';
import type { CustomFilterDefinition } from '../types/customFilters';

function makeRecord(overrides: Partial<FragmentRecord> = {}): FragmentRecord {
  return {
    id: 1,
    fragment_id: 'F001',
    image_path: 'uploads/f001.png',
    edge_piece: 1,
    has_top_edge: 1,
    has_bottom_edge: 0,
    has_left_edge: null,
    has_right_edge: null,
    line_count: 5,
    script_type: 'Early South Turkestan Brahmi',
    segmentation_coords: null,
    notes: null,
    created_at: '2025-01-01',
    updated_at: '2025-01-01',
    scale_unit: null,
    pixels_per_unit: null,
    scale_detection_status: null,
    scale_model_version: null,
    has_circle: null,
    ...overrides,
  };
}

// -----------------------------------------------------------------------
// mapToManuscriptFragment  (pure mapping – no IPC)
// -----------------------------------------------------------------------

describe('mapToManuscriptFragment', () => {
  it('maps basic fields', () => {
    const f = mapToManuscriptFragment(makeRecord());
    expect(f.id).toBe('F001');
    expect(f.name).toBe('F001');
    expect(f.imagePath).toContain('electron-image://');
  });

  it('maps edge fields as booleans', () => {
    const f = mapToManuscriptFragment(makeRecord());
    expect(f.metadata?.isEdgePiece).toBe(true);
    expect(f.metadata?.hasTopEdge).toBe(true);
    expect(f.metadata?.hasBottomEdge).toBe(false);
    expect(f.metadata?.hasLeftEdge).toBeUndefined();
  });

  it('maps line_count', () => {
    const f = mapToManuscriptFragment(makeRecord({ line_count: 10 }));
    expect(f.metadata?.lineCount).toBe(10);
  });

  it('maps script_type through display mapping', () => {
    const f = mapToManuscriptFragment(makeRecord());
    expect(f.metadata?.script).toBe('Early South Turkestan Brāhmī');
  });

  it('maps null script_type to undefined', () => {
    const f = mapToManuscriptFragment(makeRecord({ script_type: null }));
    expect(f.metadata?.script).toBeUndefined();
  });

  it('maps segmentation_coords', () => {
    const coords = '{"contours":[]}';
    const f = mapToManuscriptFragment(makeRecord({ segmentation_coords: coords }));
    expect(f.segmentationCoords).toBe(coords);
  });

  it('maps null segmentation_coords to undefined', () => {
    const f = mapToManuscriptFragment(makeRecord({ segmentation_coords: null }));
    expect(f.segmentationCoords).toBeUndefined();
  });

  it('maps scale data when present', () => {
    const f = mapToManuscriptFragment(
      makeRecord({ scale_unit: 'cm', pixels_per_unit: 120, scale_detection_status: 'success' }),
    );
    expect(f.metadata?.scale).toEqual({
      unit: 'cm',
      pixelsPerUnit: 120,
      detectionStatus: 'success',
    });
  });

  it('maps scale as undefined when unit missing', () => {
    const f = mapToManuscriptFragment(makeRecord());
    expect(f.metadata?.scale).toBeUndefined();
  });

  it('maps has_circle', () => {
    const f = mapToManuscriptFragment(makeRecord({ has_circle: 1 }));
    expect(f.metadata?.hasCircle).toBe(true);
  });

  it('maps has_circle 0 to false', () => {
    const f = mapToManuscriptFragment(makeRecord({ has_circle: 0 }));
    expect(f.metadata?.hasCircle).toBe(false);
  });

  it('maps has_circle null to undefined', () => {
    const f = mapToManuscriptFragment(makeRecord({ has_circle: null }));
    expect(f.metadata?.hasCircle).toBeUndefined();
  });

  it('maps custom filter columns', () => {
    const cfs: CustomFilterDefinition[] = [
      { id: 1, filterKey: 'col_site', label: 'Site', type: 'multiselect', options: ['A', 'B'] },
    ];
    const rec = makeRecord();
    (rec as Record<string, unknown>)['col_site'] = 'A';
    const f = mapToManuscriptFragment(rec, cfs);
    expect(f.metadata?.custom?.col_site).toBe('A');
  });

  it('maps missing custom filter key to undefined', () => {
    const cfs: CustomFilterDefinition[] = [
      { id: 1, filterKey: 'col_missing', label: 'Missing', type: 'text' },
    ];
    const f = mapToManuscriptFragment(makeRecord(), cfs);
    expect(f.metadata?.custom?.col_missing).toBeUndefined();
  });

  it('maps null custom filter value to null', () => {
    const cfs: CustomFilterDefinition[] = [
      { id: 1, filterKey: 'col_null', label: 'Null', type: 'text' },
    ];
    const rec = makeRecord();
    (rec as Record<string, unknown>)['col_null'] = null;
    const f = mapToManuscriptFragment(rec, cfs);
    expect(f.metadata?.custom?.col_null).toBeNull();
  });

  it('skips custom mapping when no custom filters provided', () => {
    const f = mapToManuscriptFragment(makeRecord());
    expect(f.metadata?.custom).toBeUndefined();
  });

  it('maps scale_detection_status error', () => {
    const f = mapToManuscriptFragment(
      makeRecord({ scale_unit: 'mm', pixels_per_unit: 50, scale_detection_status: 'error' }),
    );
    expect(f.metadata?.scale?.detectionStatus).toBe('error');
  });
});

// -----------------------------------------------------------------------
// Async functions that depend on electronAPI  (non-Electron env → fallback)
// -----------------------------------------------------------------------

describe('service functions outside Electron', () => {
  beforeEach(() => {
    (window as Record<string, unknown>).electronAPI = undefined;
  });

  it('getAllFragments returns [] outside Electron', async () => {
    const { getAllFragments } = await import('../services/fragment-service');
    expect(await getAllFragments()).toEqual([]);
  });

  it('getFragmentCount returns 0 outside Electron', async () => {
    const { getFragmentCount } = await import('../services/fragment-service');
    expect(await getFragmentCount()).toBe(0);
  });

  it('getFragmentById returns null outside Electron', async () => {
    const { getFragmentById } = await import('../services/fragment-service');
    expect(await getFragmentById('F001')).toBeNull();
  });

  it('updateFragmentMetadata returns error outside Electron', async () => {
    const { updateFragmentMetadata } = await import('../services/fragment-service');
    const result = await updateFragmentMetadata('F001', {});
    expect(result.success).toBe(false);
  });

  it('enrichWithSegmentationStatus returns fragments as-is outside Electron', async () => {
    const { enrichWithSegmentationStatus } = await import('../services/fragment-service');
    const frags = [
      { id: 'a', name: 'a', imagePath: '', thumbnailPath: '' },
    ];
    const result = await enrichWithSegmentationStatus(frags);
    expect(result).toEqual(frags);
  });

  it('enrichWithSegmentationStatus returns [] for empty array outside Electron', async () => {
    const { enrichWithSegmentationStatus } = await import('../services/fragment-service');
    expect(await enrichWithSegmentationStatus([])).toEqual([]);
  });

  it('getAvailableScripts returns []', async () => {
    const { getAvailableScripts } = await import('../services/fragment-service');
    expect(await getAvailableScripts()).toEqual([]);
  });
});

// -----------------------------------------------------------------------
// Async functions with mocked electronAPI (Electron paths)
// -----------------------------------------------------------------------

function installFragmentAPI(overrides: Record<string, unknown> = {}) {
  (window as Record<string, unknown>).electronAPI = {
    fragments: {
      getAll: async () => ({ success: true, data: [] }),
      getCount: async () => ({ success: true, count: 0 }),
      getById: async () => ({ success: true, data: null }),
      updateMetadata: async () => ({ success: true }),
      ...overrides,
    },
  };
}

describe('service functions inside Electron', () => {
  afterEach(() => {
    (window as Record<string, unknown>).electronAPI = undefined;
  });

  it('getAllFragments returns mapped fragments on success', async () => {
    installFragmentAPI({
      getAll: async () => ({
        success: true,
        data: [makeRecord({ fragment_id: 'X1' })],
      }),
    });
    const { getAllFragments } = await import('../services/fragment-service');
    const result = await getAllFragments();
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('X1');
  });

  it('getAllFragments returns [] on failure response', async () => {
    installFragmentAPI({
      getAll: async () => ({ success: false, error: 'db error' }),
    });
    const { getAllFragments } = await import('../services/fragment-service');
    expect(await getAllFragments()).toEqual([]);
  });

  it('getFragmentCount returns count on success', async () => {
    installFragmentAPI({
      getCount: async () => ({ success: true, count: 42 }),
    });
    const { getFragmentCount } = await import('../services/fragment-service');
    expect(await getFragmentCount()).toBe(42);
  });

  it('getFragmentCount returns 0 on failure', async () => {
    installFragmentAPI({
      getCount: async () => ({ success: false, error: 'err' }),
    });
    const { getFragmentCount } = await import('../services/fragment-service');
    expect(await getFragmentCount()).toBe(0);
  });

  it('getFragmentById returns fragment on success', async () => {
    installFragmentAPI({
      getById: async () => ({ success: true, data: makeRecord({ fragment_id: 'Z1' }) }),
    });
    const { getFragmentById } = await import('../services/fragment-service');
    const result = await getFragmentById('Z1');
    expect(result?.id).toBe('Z1');
  });

  it('getFragmentById returns null on failure', async () => {
    installFragmentAPI({
      getById: async () => ({ success: false }),
    });
    const { getFragmentById } = await import('../services/fragment-service');
    expect(await getFragmentById('Z1')).toBeNull();
  });

  it('updateFragmentMetadata returns response', async () => {
    installFragmentAPI({
      updateMetadata: async () => ({ success: true }),
    });
    const { updateFragmentMetadata } = await import('../services/fragment-service');
    const result = await updateFragmentMetadata('Z1', { notes: 'hi' });
    expect(result.success).toBe(true);
  });

  it('enrichWithSegmentationStatus adds hasSegmentation flag', async () => {
    installFragmentAPI();
    const { enrichWithSegmentationStatus } = await import('../services/fragment-service');
    const frags = [
      { id: 'a', name: 'a', imagePath: '', thumbnailPath: '', segmentationCoords: '{"contours":[]}' },
      { id: 'b', name: 'b', imagePath: '', thumbnailPath: '' },
    ];
    const result = await enrichWithSegmentationStatus(frags);
    expect(result[0].hasSegmentation).toBe(true);
    expect(result[1].hasSegmentation).toBe(false);
  });

  it('enrichWithSegmentationStatus returns empty for empty input', async () => {
    installFragmentAPI();
    const { enrichWithSegmentationStatus } = await import('../services/fragment-service');
    expect(await enrichWithSegmentationStatus([])).toEqual([]);
  });
});
