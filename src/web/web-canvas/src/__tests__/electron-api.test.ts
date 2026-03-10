import { describe, it, expect, beforeEach } from 'vitest';
import { isElectron, getElectronAPI, getElectronAPISafe } from '../services/electron-api';

describe('isElectron', () => {
  beforeEach(() => {
    (window as Record<string, unknown>).electronAPI = undefined;
  });

  it('returns false when electronAPI is undefined', () => {
    expect(isElectron()).toBe(false);
  });

  it('returns true when electronAPI is defined', () => {
    (window as Record<string, unknown>).electronAPI = {} as unknown;
    expect(isElectron()).toBe(true);
  });
});

describe('getElectronAPI', () => {
  beforeEach(() => {
    (window as Record<string, unknown>).electronAPI = undefined;
  });

  it('throws when electronAPI is not available', () => {
    expect(() => getElectronAPI()).toThrow('Electron API not available');
  });

  it('returns the API object when available', () => {
    const api = { fragments: {} } as unknown;
    (window as Record<string, unknown>).electronAPI = api;
    expect(getElectronAPI()).toBe(api);
  });
});

describe('getElectronAPISafe', () => {
  beforeEach(() => {
    (window as Record<string, unknown>).electronAPI = undefined;
  });

  it('returns null when electronAPI is not available', () => {
    expect(getElectronAPISafe()).toBeNull();
  });

  it('returns the API object when available', () => {
    const api = { fragments: {} } as unknown;
    (window as Record<string, unknown>).electronAPI = api;
    expect(getElectronAPISafe()).toBe(api);
  });
});
