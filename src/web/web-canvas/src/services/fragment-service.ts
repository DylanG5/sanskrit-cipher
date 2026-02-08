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
import { getScriptTypeDisplay } from '../types/constants';
import { CustomFilterDefinition } from '../types/customFilters';

// Re-export for convenience
export type { ApiFilters as FragmentApiFilters };

/**
 * Convert database fragment record to UI ManuscriptFragment type
 */
export function mapToManuscriptFragment(
  record: FragmentRecord,
  customFilters: CustomFilterDefinition[] = []
): ManuscriptFragment {
  const custom: Record<string, string | null | undefined> = {};
  for (const filter of customFilters) {
    const rawValue = record[filter.filterKey];
    if (rawValue === undefined) {
      custom[filter.filterKey] = undefined;
    } else if (rawValue === null) {
      custom[filter.filterKey] = null;
    } else {
      custom[filter.filterKey] = String(rawValue);
    }
  }

  return {
    id: record.fragment_id,
    name: record.fragment_id,
    // Use electron-image protocol for loading images from data folder
    imagePath: `electron-image://${record.image_path}`,
    thumbnailPath: `electron-image://${record.image_path}`,
    // Include segmentation coordinates for on-demand rendering
    segmentationCoords: record.segmentation_coords ?? undefined,
    metadata: {
      lineCount: record.line_count ?? undefined,
      script: getScriptTypeDisplay(record.script_type),
      isEdgePiece: record.edge_piece === 1 ? true : record.edge_piece === 0 ? false : undefined,
      hasTopEdge: record.has_top_edge === 1 ? true : record.has_top_edge === 0 ? false : undefined,
      hasBottomEdge: record.has_bottom_edge === 1 ? true : record.has_bottom_edge === 0 ? false : undefined,
      hasLeftEdge: record.has_left_edge === 1 ? true : record.has_left_edge === 0 ? false : undefined,
      hasRightEdge: record.has_right_edge === 1 ? true : record.has_right_edge === 0 ? false : undefined,
      hasCircle: record.has_circle === 1 ? true : record.has_circle === 0 ? false : undefined,
      // Map scale data if available
      scale: record.scale_unit && record.pixels_per_unit ? {
        unit: record.scale_unit as 'cm' | 'mm',
        pixelsPerUnit: record.pixels_per_unit,
        detectionStatus: record.scale_detection_status === 'success' ? 'success' : 'error',
      } : undefined,
      custom: customFilters.length > 0 ? custom : undefined,
    },
  };
}

/**
 * Get all fragments from the database with optional filtering
 */
export async function getAllFragments(
  filters?: ApiFilters,
  customFilters: CustomFilterDefinition[] = []
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

  return response.data.map((record) => mapToManuscriptFragment(record, customFilters));
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
  fragmentId: string,
  customFilters: CustomFilterDefinition[] = []
): Promise<ManuscriptFragment | null> {
  if (!isElectron()) {
    return null;
  }

  const api = getElectronAPI();
  const response = await api.fragments.getById(fragmentId);

  if (!response.success || !response.data) {
    return null;
  }

  return mapToManuscriptFragment(response.data, customFilters);
}


/**
 * Update fragment metadata
 */
export async function updateFragmentMetadata(
  fragmentId: string,
  metadata: Record<string, unknown>
): Promise<{ success: boolean; error?: string }> {
  if (!isElectron()) {
    return { success: false, error: 'Not in Electron environment' };
  }

  const api = getElectronAPI();
  const response = await api.fragments.updateMetadata(fragmentId, metadata);
  return response;
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

/**
 * Enrich fragments with segmentation availability info
 *
 * Now checks for segmentationCoords in the database rather than cached files.
 * The hasSegmentation flag is set based on whether valid segmentation coordinates exist.
 */
export async function enrichWithSegmentationStatus(
  fragments: ManuscriptFragment[]
): Promise<ManuscriptFragment[]> {
  if (!isElectron() || fragments.length === 0) {
    return fragments;
  }

  // Check if each fragment has valid segmentation coordinates
  return fragments.map(fragment => ({
    ...fragment,
    hasSegmentation: !!(fragment.segmentationCoords && fragment.segmentationCoords.length > 0),
  }));
}
