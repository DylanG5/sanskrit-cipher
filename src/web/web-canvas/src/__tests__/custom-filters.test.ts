import { describe, it, expect, beforeEach } from 'vitest';
import {
  getCustomFilters,
  createCustomFilter,
  deleteCustomFilter,
  updateCustomFilterOptions,
} from '../services/custom-filters';

// Helper to install a mock electronAPI on window
function installElectronAPI(customFiltersOverride: Record<string, unknown> = {}) {
  (window as Record<string, unknown>).electronAPI = {
    customFilters: {
      list: async () => ({ success: true, data: [] }),
      create: async () => ({ success: true, data: null }),
      delete: async () => ({ success: true }),
      updateOptions: async () => ({ success: true, data: null }),
      ...customFiltersOverride,
    },
  };
}

describe('custom-filters service (non-Electron)', () => {
  beforeEach(() => {
    (window as Record<string, unknown>).electronAPI = undefined;
  });

  it('getCustomFilters returns []', async () => {
    expect(await getCustomFilters()).toEqual([]);
  });

  it('createCustomFilter returns null', async () => {
    expect(await createCustomFilter({ label: 'x', type: 'text' })).toBeNull();
  });

  it('deleteCustomFilter returns false', async () => {
    expect(await deleteCustomFilter(1)).toBe(false);
  });

  it('updateCustomFilterOptions returns null', async () => {
    expect(await updateCustomFilterOptions(1, ['a'])).toBeNull();
  });
});

describe('custom-filters service (Electron)', () => {
  beforeEach(() => {
    (window as Record<string, unknown>).electronAPI = undefined;
  });

  it('getCustomFilters returns data on success', async () => {
    const data = [{ id: 1, filterKey: 'col_x', label: 'X', type: 'text' as const }];
    installElectronAPI({ list: async () => ({ success: true, data }) });
    expect(await getCustomFilters()).toEqual(data);
  });

  it('getCustomFilters returns [] on failure response', async () => {
    installElectronAPI({ list: async () => ({ success: false }) });
    expect(await getCustomFilters()).toEqual([]);
  });

  it('createCustomFilter returns created filter', async () => {
    const cf = { id: 2, filterKey: 'col_y', label: 'Y', type: 'dropdown' as const, options: ['a'] };
    installElectronAPI({ create: async () => ({ success: true, data: cf }) });
    expect(await createCustomFilter({ label: 'Y', type: 'dropdown', options: ['a'] })).toEqual(cf);
  });

  it('createCustomFilter returns null on failure', async () => {
    installElectronAPI({ create: async () => ({ success: false }) });
    expect(await createCustomFilter({ label: 'Y', type: 'dropdown' })).toBeNull();
  });

  it('deleteCustomFilter returns true on success', async () => {
    installElectronAPI({ delete: async () => ({ success: true }) });
    expect(await deleteCustomFilter(1)).toBe(true);
  });

  it('deleteCustomFilter returns false on failure', async () => {
    installElectronAPI({ delete: async () => ({ success: false }) });
    expect(await deleteCustomFilter(1)).toBe(false);
  });

  it('updateCustomFilterOptions returns updated filter', async () => {
    const cf = { id: 3, filterKey: 'col_z', label: 'Z', type: 'text' as const };
    installElectronAPI({ updateOptions: async () => ({ success: true, data: cf }) });
    expect(await updateCustomFilterOptions(3, ['x'])).toEqual(cf);
  });

  it('updateCustomFilterOptions returns null on failure', async () => {
    installElectronAPI({ updateOptions: async () => ({ success: false }) });
    expect(await updateCustomFilterOptions(3, ['x'])).toBeNull();
  });
});
