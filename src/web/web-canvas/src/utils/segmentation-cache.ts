/**
 * Segmentation Cache Utility
 *
 * Provides IndexedDB-based caching for on-demand generated segmented images
 * to avoid regenerating them on repeat views.
 */

import { createSegmentedImage, hasValidSegmentation } from './segmentation-renderer';

const DB_NAME = 'segmentation-cache';
const STORE_NAME = 'segmented-images';
const DB_VERSION = 1;

interface CacheEntry {
  fragmentId: string;
  dataUrl: string;
  timestamp: number;
  modelVersion: string;
}

/**
 * Initialize IndexedDB database
 */
function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;

      // Create object store if it doesn't exist
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'fragmentId' });
        store.createIndex('timestamp', 'timestamp', { unique: false });
      }
    };
  });
}

/**
 * Get cached segmented image from IndexedDB
 */
async function getCached(fragmentId: string): Promise<string | null> {
  try {
    const db = await openDatabase();
    const transaction = db.transaction(STORE_NAME, 'readonly');
    const store = transaction.objectStore(STORE_NAME);

    return new Promise((resolve, reject) => {
      const request = store.get(fragmentId);

      request.onsuccess = () => {
        const entry = request.result as CacheEntry | undefined;
        resolve(entry?.dataUrl || null);
      };

      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.warn('Failed to get cached segmented image:', error);
    return null;
  }
}

/**
 * Store segmented image in IndexedDB cache
 */
async function setCached(
  fragmentId: string,
  dataUrl: string,
  modelVersion: string
): Promise<void> {
  try {
    const db = await openDatabase();
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);

    const entry: CacheEntry = {
      fragmentId,
      dataUrl,
      timestamp: Date.now(),
      modelVersion,
    };

    return new Promise((resolve, reject) => {
      const request = store.put(entry);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.warn('Failed to cache segmented image:', error);
  }
}

/**
 * Get or create a segmented image with caching.
 *
 * First checks the cache, then generates on-demand if not found.
 *
 * @param fragmentId - Unique fragment identifier (for cache key)
 * @param originalImageSrc - Source URL of the original fragment image
 * @param segmentationCoordsJson - JSON string containing contour coordinates
 * @returns Promise resolving to data URL of the segmented image
 */
export async function getOrCreateSegmentedImage(
  fragmentId: string,
  originalImageSrc: string,
  segmentationCoordsJson: string | null
): Promise<string | null> {
  // Validate segmentation data
  if (!hasValidSegmentation(segmentationCoordsJson)) {
    return null;
  }

  // Check cache first
  const cached = await getCached(fragmentId);
  if (cached) {
    return cached;
  }

  // Generate segmented image on-demand
  try {
    const dataUrl = await createSegmentedImage(originalImageSrc, segmentationCoordsJson!);

    // Extract model version from segmentation coords
    let modelVersion = 'unknown';
    try {
      const coordsData = JSON.parse(segmentationCoordsJson!);
      modelVersion = coordsData.model_version || 'unknown';
    } catch {
      // Ignore parsing errors for model version
    }

    // Cache the result for future use
    await setCached(fragmentId, dataUrl, modelVersion);

    return dataUrl;
  } catch (error) {
    console.error('Failed to create segmented image:', error);
    return null;
  }
}

/**
 * Clear all cached segmented images
 */
export async function clearCache(): Promise<void> {
  try {
    const db = await openDatabase();
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);

    return new Promise((resolve, reject) => {
      const request = store.clear();
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.warn('Failed to clear cache:', error);
  }
}

/**
 * Remove a specific fragment from cache
 */
export async function removeCached(fragmentId: string): Promise<void> {
  try {
    const db = await openDatabase();
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);

    return new Promise((resolve, reject) => {
      const request = store.delete(fragmentId);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.warn('Failed to remove cached image:', error);
  }
}

/**
 * Get cache statistics
 */
export async function getCacheStats(): Promise<{
  count: number;
  estimatedSize: number;
}> {
  try {
    const db = await openDatabase();
    const transaction = db.transaction(STORE_NAME, 'readonly');
    const store = transaction.objectStore(STORE_NAME);

    return new Promise((resolve, reject) => {
      const request = store.getAll();

      request.onsuccess = () => {
        const entries = request.result as CacheEntry[];
        const count = entries.length;

        // Estimate size based on data URL lengths
        const estimatedSize = entries.reduce((total, entry) => {
          return total + (entry.dataUrl?.length || 0);
        }, 0);

        resolve({ count, estimatedSize });
      };

      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.warn('Failed to get cache stats:', error);
    return { count: 0, estimatedSize: 0 };
  }
}
