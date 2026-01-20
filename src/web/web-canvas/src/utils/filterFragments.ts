import { ManuscriptFragment } from '../types/fragment';
import { FragmentFilters } from '../types/filters';

/**
 * Filter fragments based on metadata criteria
 */
export function filterFragments(
  fragments: ManuscriptFragment[],
  filters: FragmentFilters
): ManuscriptFragment[] {
  return fragments.filter((fragment) => {
    const { metadata } = fragment;

    // If no metadata and filters are applied, exclude this fragment
    if (!metadata) {
      // Only include if no filters are active
      const hasActiveFilters =
        filters.lineCountMin !== undefined ||
        filters.lineCountMax !== undefined ||
        filters.scripts.length > 0 ||
        filters.isEdgePiece !== null;

      return !hasActiveFilters;
    }

    // Filter by line count
    if (filters.lineCountMin !== undefined && metadata.lineCount !== undefined) {
      if (metadata.lineCount < filters.lineCountMin) {
        return false;
      }
    }

    if (filters.lineCountMax !== undefined && metadata.lineCount !== undefined) {
      if (metadata.lineCount > filters.lineCountMax) {
        return false;
      }
    }

    // Filter by script type
    if (filters.scripts.length > 0) {
      if (!metadata.script || !filters.scripts.includes(metadata.script)) {
        return false;
      }
    }

    // Filter by edge piece
    if (filters.isEdgePiece !== null) {
      if (metadata.isEdgePiece !== filters.isEdgePiece) {
        return false;
      }
    }

    return true;
  });
}

/**
 * Get all unique script types from fragments
 */
export function getAvailableScripts(fragments: ManuscriptFragment[]): string[] {
  const scripts = new Set<string>();

  fragments.forEach((fragment) => {
    if (fragment.metadata?.script) {
      scripts.add(fragment.metadata.script);
    }
  });

  return Array.from(scripts).sort();
}
