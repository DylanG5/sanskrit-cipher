/**
 * Fragment Service
 *
 * Maps database records to UI types and provides fragment data access.
 */

import { ManuscriptFragment } from '../types/fragment';
import {
  getElectronAPI,
  isElectron,
  FragmentRecord,
  FragmentFilters as ApiFilters,
} from './electron-api';

// Re-export for convenience
export type { ApiFilters as FragmentApiFilters };

/**
 * Convert database fragment record to UI ManuscriptFragment type
 */
export function mapToManuscriptFragment(record: FragmentRecord): ManuscriptFragment {
  return {
    id: record.fragment_id,
    name: record.fragment_id,
    // Use electron-image protocol for loading images from data folder
    imagePath: `electron-image://${record.image_path}`,
    thumbnailPath: `electron-image://${record.image_path}`,
    metadata: {
      lineCount: record.line_count ?? undefined,
      script: record.script_type ?? undefined,
      isEdgePiece: record.edge_piece === 1,
      // Map scale data if available
      scale: record.scale_unit && record.pixels_per_unit ? {
        unit: record.scale_unit as 'cm' | 'mm',
        pixelsPerUnit: record.pixels_per_unit,
        detectionStatus: record.scale_detection_status === 'success' ? 'success' : 'error',
      } : undefined,
    },
  };
}

/**
 * Get all fragments from the database with optional filtering
 */
export async function getAllFragments(
  filters?: ApiFilters
): Promise<ManuscriptFragment[]> {
  if (!isElectron()) {
    console.warn('Not running in Electron, returning empty fragments');
    return [];
  }

  const api = getElectronAPI();
  const response = await api.fragments.getAll(filters);

  if (!response.success || !response.data) {
    console.error('Failed to fetch fragments:', response.error);
    return [];
  }

  return response.data.map(mapToManuscriptFragment);
}

/**
 * Get total count of fragments (for pagination)
 */
export async function getFragmentCount(filters?: ApiFilters): Promise<number> {
  if (!isElectron()) {
    return 0;
  }

  const api = getElectronAPI();
  const response = await api.fragments.getCount(filters);

  if (!response.success) {
    console.error('Failed to get fragment count:', response.error);
    return 0;
  }

  return response.count ?? 0;
}

/**
 * Get a single fragment by ID
 */
export async function getFragmentById(
  fragmentId: string
): Promise<ManuscriptFragment | null> {
  if (!isElectron()) {
    return null;
  }

  const api = getElectronAPI();
  const response = await api.fragments.getById(fragmentId);

  if (!response.success || !response.data) {
    return null;
  }

  return mapToManuscriptFragment(response.data);
}

/**
 * Update fragment notes
 */
export async function updateFragmentNotes(
  fragmentId: string,
  notes: string
): Promise<boolean> {
  if (!isElectron()) {
    return false;
  }

  const api = getElectronAPI();
  const response = await api.fragments.updateMetadata(fragmentId, { notes });

  return response.success;
}

/**
 * Get unique script types from database
 * For now returns empty array - would need a dedicated IPC handler
 * or fetch all and dedupe client-side
 */
export async function getAvailableScripts(): Promise<string[]> {
  // Since we don't have ML-populated script_type yet, return empty
  // In the future, add an IPC handler to get distinct script types
  return [];
}
