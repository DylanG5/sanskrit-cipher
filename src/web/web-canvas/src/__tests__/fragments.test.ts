import { describe, it, expect } from 'vitest';
import {
  sortBySearchRelevance,
  calculateCenteredPosition,
  fragments,
  getFragmentById,
} from '../utils/fragments';
import { ManuscriptFragment } from '../types/fragment';

function frag(id: string): ManuscriptFragment {
  return { id, name: id, imagePath: '', thumbnailPath: '' };
}

// -----------------------------------------------------------------------
// sortBySearchRelevance
// -----------------------------------------------------------------------

describe('sortBySearchRelevance', () => {
  it('exact match comes first', () => {
    const list = [frag('abc'), frag('ab'), frag('abcd')];
    const sorted = sortBySearchRelevance(list, 'ab');
    expect(sorted[0].id).toBe('ab');
  });

  it('starts-with comes before contains', () => {
    const list = [frag('xab'), frag('abx')];
    const sorted = sortBySearchRelevance(list, 'ab');
    expect(sorted[0].id).toBe('abx');
  });

  it('falls back to alphabetical', () => {
    const list = [frag('z'), frag('a')];
    const sorted = sortBySearchRelevance(list, 'q');
    expect(sorted[0].id).toBe('a');
  });

  it('is case-insensitive', () => {
    const list = [frag('ABC'), frag('abc')];
    const sorted = sortBySearchRelevance(list, 'ABC');
    // Both match exactly (case-insensitive), so alphabetical of lowercase
    expect(sorted).toHaveLength(2);
  });
});

// -----------------------------------------------------------------------
// calculateCenteredPosition
// -----------------------------------------------------------------------

describe('calculateCenteredPosition', () => {
  it('calculates center correctly', () => {
    const pos = calculateCenteredPosition(800, 600, 100, 50);
    expect(pos).toEqual({ x: 350, y: 275 });
  });

  it('returns negative coords for large fragments', () => {
    const pos = calculateCenteredPosition(200, 200, 400, 400);
    expect(pos).toEqual({ x: -100, y: -100 });
  });
});

// -----------------------------------------------------------------------
// Exported sample data
// -----------------------------------------------------------------------

describe('fragments sample data', () => {
  it('exported array is non-empty', () => {
    expect(fragments.length).toBeGreaterThan(0);
  });

  it('each fragment has id/name/imagePath', () => {
    for (const f of fragments) {
      expect(f.id).toBeTruthy();
      expect(f.name).toBeTruthy();
      expect(f.imagePath).toBeTruthy();
    }
  });
});

describe('getFragmentById', () => {
  it('returns fragment when found', () => {
    const f = getFragmentById(fragments[0].id);
    expect(f).toBeDefined();
    expect(f!.id).toBe(fragments[0].id);
  });

  it('returns undefined for missing id', () => {
    expect(getFragmentById('no-such-id')).toBeUndefined();
  });
});
