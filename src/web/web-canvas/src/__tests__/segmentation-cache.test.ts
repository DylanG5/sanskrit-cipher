import { describe, it, expect, vi, beforeEach } from 'vitest';

// -----------------------------------------------------------------------
// Minimal in-memory IndexedDB mock
// -----------------------------------------------------------------------

function createMockIDB() {
  const store = new Map<string, unknown>();

  const makeRequest = (result: unknown, err: unknown = null) => {
    const req: Record<string, unknown> = {
      result,
      error: err,
      onsuccess: null as (() => void) | null,
      onerror: null as (() => void) | null,
    };
    // fire on next microtask
    Promise.resolve().then(() => {
      if (err && typeof req.onerror === 'function') (req.onerror as () => void)();
      else if (typeof req.onsuccess === 'function') (req.onsuccess as () => void)();
    });
    return req;
  };

  const objectStore = () => ({
    get: (key: string) => makeRequest(store.get(key)),
    put: (value: { fragmentId: string }) => {
      store.set(value.fragmentId, value);
      return makeRequest(undefined);
    },
    delete: (key: string) => {
      store.delete(key);
      return makeRequest(undefined);
    },
    clear: () => {
      store.clear();
      return makeRequest(undefined);
    },
    getAll: () => makeRequest(Array.from(store.values())),
    createIndex: () => {},
  });

  const mockDB = {
    transaction: () => ({
      objectStore: objectStore,
    }),
    objectStoreNames: { contains: () => false },
    createObjectStore: () => objectStore(),
  };

  const openReq: Record<string, unknown> = {
    result: mockDB,
    error: null,
    onsuccess: null,
    onerror: null,
    onupgradeneeded: null,
  };

  // Simulate IDB open flow (upgradeneeded then success)
  vi.stubGlobal('indexedDB', {
    open: () => {
      Promise.resolve().then(() => {
        if (typeof openReq.onupgradeneeded === 'function') {
          (openReq.onupgradeneeded as (e: unknown) => void)({ target: openReq });
        }
        if (typeof openReq.onsuccess === 'function') {
          (openReq.onsuccess as () => void)();
        }
      });
      return openReq;
    },
  });

  return { store };
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

describe('segmentation-cache', () => {
  let cache: typeof import('../utils/segmentation-cache');

  beforeEach(async () => {
    createMockIDB();
    // Dynamic import so the module picks up the stubGlobal indexedDB
    cache = await import('../utils/segmentation-cache');
  });

  it('getOrCreateSegmentedImage returns null for invalid segmentation', async () => {
    expect(await cache.getOrCreateSegmentedImage('f1', 'img.png', null)).toBeNull();
  });

  it('getOrCreateSegmentedImage returns null for bad JSON', async () => {
    expect(await cache.getOrCreateSegmentedImage('f1', 'img.png', 'not-json')).toBeNull();
  });

  it('getOrCreateSegmentedImage returns null for missing contours', async () => {
    expect(await cache.getOrCreateSegmentedImage('f1', 'img.png', '{}')).toBeNull();
  });

  it('clearCache does not throw', async () => {
    await expect(cache.clearCache()).resolves.toBeUndefined();
  });

  it('removeCached does not throw', async () => {
    await expect(cache.removeCached('f1')).resolves.toBeUndefined();
  });

  it('getCacheStats returns zeroes on empty store', async () => {
    const stats = await cache.getCacheStats();
    expect(stats).toEqual({ count: 0, estimatedSize: 0 });
  });
});
