import { describe, it, expect } from 'vitest';
import {
  getScriptTypeDisplay,
  getScriptTypeDB,
  SCRIPT_TYPES,
  MODEL_SCRIPT_TYPES,
  SCRIPT_TYPE_DISPLAY_MAP,
  SCRIPT_TYPE_DB_MAP,
} from '../types/constants';

describe('getScriptTypeDisplay', () => {
  it('maps known db value to display value', () => {
    expect(getScriptTypeDisplay('Early South Turkestan Brahmi')).toBe(
      'Early South Turkestan Brāhmī',
    );
  });

  it('maps South Turkestan Brahmi to main type', () => {
    expect(getScriptTypeDisplay('South Turkestan Brahmi')).toBe(
      'South Turkestan Brāhmī (main type)',
    );
  });

  it('returns the value itself for unknown entries', () => {
    expect(getScriptTypeDisplay('Unknown Script')).toBe('Unknown Script');
  });

  it('returns undefined for null/undefined', () => {
    expect(getScriptTypeDisplay(null)).toBeUndefined();
    expect(getScriptTypeDisplay(undefined)).toBeUndefined();
  });

  it('returns undefined for empty string', () => {
    expect(getScriptTypeDisplay('')).toBeUndefined();
  });
});

describe('getScriptTypeDB', () => {
  it('maps display value to db value', () => {
    expect(getScriptTypeDB('Early South Turkestan Brāhmī')).toBe(
      'Early South Turkestan Brahmi',
    );
  });

  it('maps main type to South Turkestan Brahmi', () => {
    expect(getScriptTypeDB('South Turkestan Brāhmī (main type)')).toBe(
      'South Turkestan Brahmi',
    );
  });

  it('returns the value itself for unknown entries', () => {
    expect(getScriptTypeDB('Unknown Script')).toBe('Unknown Script');
  });

  it('returns undefined for null/undefined', () => {
    expect(getScriptTypeDB(null)).toBeUndefined();
    expect(getScriptTypeDB(undefined)).toBeUndefined();
  });
});

describe('SCRIPT_TYPES constant', () => {
  it('has expected length', () => {
    expect(SCRIPT_TYPES.length).toBeGreaterThan(0);
  });

  it('contains Early South Turkestan Brāhmī', () => {
    expect(SCRIPT_TYPES).toContain('Early South Turkestan Brāhmī');
  });
});

describe('MODEL_SCRIPT_TYPES constant', () => {
  it('has expected length', () => {
    expect(MODEL_SCRIPT_TYPES.length).toBeGreaterThan(0);
  });

  it('contains db-format values', () => {
    expect(MODEL_SCRIPT_TYPES).toContain('Early South Turkestan Brahmi');
  });
});

describe('bidirectional mappings consistency', () => {
  it('every MODEL_SCRIPT_TYPE has a display mapping', () => {
    for (const dbVal of MODEL_SCRIPT_TYPES) {
      expect(SCRIPT_TYPE_DISPLAY_MAP).toHaveProperty(dbVal);
    }
  });

  it('every SCRIPT_TYPE display value has a reverse mapping', () => {
    for (const displayVal of SCRIPT_TYPES) {
      expect(SCRIPT_TYPE_DB_MAP).toHaveProperty(displayVal);
    }
  });
});
