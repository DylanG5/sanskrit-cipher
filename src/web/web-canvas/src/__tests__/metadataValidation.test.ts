import { describe, it, expect } from 'vitest';
import {
  validateLineCount,
  validatePixelsPerUnit,
  validateScriptType,
  validateScaleUnit,
} from '../utils/metadataValidation';

describe('validateLineCount', () => {
  it('accepts valid integer', () => {
    expect(validateLineCount(5)).toEqual({ valid: true });
  });

  it('accepts zero', () => {
    expect(validateLineCount(0)).toEqual({ valid: true });
  });

  it('accepts 100', () => {
    expect(validateLineCount(100)).toEqual({ valid: true });
  });

  it('rejects negative', () => {
    const r = validateLineCount(-1);
    expect(r.valid).toBe(false);
    expect(r.error).toMatch(/negative/i);
  });

  it('rejects > 100', () => {
    const r = validateLineCount(101);
    expect(r.valid).toBe(false);
    expect(r.error).toMatch(/100/);
  });

  it('rejects non-integer', () => {
    const r = validateLineCount(2.5);
    expect(r.valid).toBe(false);
    expect(r.error).toMatch(/whole number/i);
  });

  it('rejects NaN', () => {
    const r = validateLineCount(NaN);
    expect(r.valid).toBe(false);
    expect(r.error).toMatch(/number/i);
  });

  it('accepts null/undefined (clear field)', () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(validateLineCount(null as any).valid).toBe(true);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(validateLineCount(undefined as any).valid).toBe(true);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(validateLineCount('' as any).valid).toBe(true);
  });
});

describe('validatePixelsPerUnit', () => {
  it('accepts positive value', () => {
    expect(validatePixelsPerUnit(120.5)).toEqual({ valid: true });
  });

  it('rejects zero', () => {
    expect(validatePixelsPerUnit(0).valid).toBe(false);
  });

  it('rejects negative', () => {
    expect(validatePixelsPerUnit(-5).valid).toBe(false);
  });

  it('rejects > 10000', () => {
    expect(validatePixelsPerUnit(10001).valid).toBe(false);
  });

  it('rejects NaN', () => {
    expect(validatePixelsPerUnit(NaN).valid).toBe(false);
  });
});

describe('validateScriptType', () => {
  it('accepts valid script type', () => {
    expect(validateScriptType('Early South Turkestan Brāhmī')).toEqual({ valid: true });
  });

  it('rejects unknown script type', () => {
    const r = validateScriptType('FakeScript');
    expect(r.valid).toBe(false);
  });

  it('accepts empty/null/undefined (clear field)', () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(validateScriptType(null as any).valid).toBe(true);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(validateScriptType(undefined as any).valid).toBe(true);
    expect(validateScriptType('').valid).toBe(true);
    expect(validateScriptType('  ').valid).toBe(true);
  });
});

describe('validateScaleUnit', () => {
  it('accepts cm', () => {
    expect(validateScaleUnit('cm')).toEqual({ valid: true });
  });

  it('accepts mm', () => {
    expect(validateScaleUnit('mm')).toEqual({ valid: true });
  });

  it('rejects other', () => {
    expect(validateScaleUnit('in').valid).toBe(false);
  });
});
