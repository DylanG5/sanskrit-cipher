import { describe, it, expect } from 'vitest';
import { filterFragments, getAvailableScripts } from '../utils/filterFragments';
import { ManuscriptFragment } from '../types/fragment';
import { FragmentFilters, DEFAULT_FILTERS } from '../types/filters';

function frag(overrides: Partial<ManuscriptFragment> = {}): ManuscriptFragment {
  return {
    id: 'f1',
    name: 'F001',
    imagePath: '/img/f1.png',
    thumbnailPath: '/img/f1.png',
    metadata: { lineCount: 5, script: 'Brahmi', isEdgePiece: false },
    ...overrides,
  };
}

const base: FragmentFilters = { ...DEFAULT_FILTERS };

// -----------------------------------------------------------------------
// filterFragments
// -----------------------------------------------------------------------

describe('filterFragments', () => {
  it('returns all when no filters active', () => {
    const frags = [frag(), frag({ id: 'f2' })];
    expect(filterFragments(frags, base)).toHaveLength(2);
  });

  it('filters by lineCountMin', () => {
    const frags = [
      frag({ id: 'a', metadata: { lineCount: 2 } }),
      frag({ id: 'b', metadata: { lineCount: 5 } }),
    ];
    const result = filterFragments(frags, { ...base, lineCountMin: 4 });
    expect(result.map((f) => f.id)).toEqual(['b']);
  });

  it('filters by lineCountMax', () => {
    const frags = [
      frag({ id: 'a', metadata: { lineCount: 2 } }),
      frag({ id: 'b', metadata: { lineCount: 5 } }),
    ];
    const result = filterFragments(frags, { ...base, lineCountMax: 3 });
    expect(result.map((f) => f.id)).toEqual(['a']);
  });

  it('filters by script type', () => {
    const frags = [
      frag({ id: 'a', metadata: { script: 'Brahmi' } }),
      frag({ id: 'b', metadata: { script: 'Kharosthi' } }),
    ];
    const result = filterFragments(frags, { ...base, scripts: ['Kharosthi'] });
    expect(result.map((f) => f.id)).toEqual(['b']);
  });

  it('filters by isEdgePiece', () => {
    const frags = [
      frag({ id: 'a', metadata: { isEdgePiece: true } }),
      frag({ id: 'b', metadata: { isEdgePiece: false } }),
    ];
    const result = filterFragments(frags, { ...base, isEdgePiece: true });
    expect(result.map((f) => f.id)).toEqual(['a']);
  });

  it('excludes fragments without metadata when filters active', () => {
    const frags = [frag({ id: 'a', metadata: undefined })];
    expect(filterFragments(frags, { ...base, lineCountMin: 1 })).toHaveLength(0);
  });

  it('includes fragments without metadata when no filters active', () => {
    const frags = [frag({ id: 'a', metadata: undefined })];
    expect(filterFragments(frags, base)).toHaveLength(1);
  });

  it('handles combined min+max range', () => {
    const frags = [
      frag({ id: 'a', metadata: { lineCount: 1 } }),
      frag({ id: 'b', metadata: { lineCount: 5 } }),
      frag({ id: 'c', metadata: { lineCount: 10 } }),
    ];
    const result = filterFragments(frags, { ...base, lineCountMin: 3, lineCountMax: 7 });
    expect(result.map((f) => f.id)).toEqual(['b']);
  });
});

// -----------------------------------------------------------------------
// getAvailableScripts
// -----------------------------------------------------------------------

describe('getAvailableScripts', () => {
  it('returns sorted unique scripts', () => {
    const frags = [
      frag({ id: 'a', metadata: { script: 'Kharosthi' } }),
      frag({ id: 'b', metadata: { script: 'Brahmi' } }),
      frag({ id: 'c', metadata: { script: 'Kharosthi' } }),
    ];
    expect(getAvailableScripts(frags)).toEqual(['Brahmi', 'Kharosthi']);
  });

  it('ignores fragments without script', () => {
    const frags = [frag({ metadata: {} }), frag({ metadata: { script: 'Brahmi' } })];
    expect(getAvailableScripts(frags)).toEqual(['Brahmi']);
  });

  it('returns empty for no fragments', () => {
    expect(getAvailableScripts([])).toEqual([]);
  });
});
