export interface FragmentFilters {
  lineCountMin?: number;
  lineCountMax?: number;
  scripts: string[]; // Empty array = all scripts
  isEdgePiece?: boolean | null; // null = don't care
  hasTopEdge?: boolean | null; // null = don't care
  hasBottomEdge?: boolean | null; // null = don't care
  hasLeftEdge?: boolean | null; // null = don't care
  hasRightEdge?: boolean | null; // null = don't care
  hasCircle?: boolean | null; // null = don't care
  search?: string; // Fragment ID search query
  custom?: Record<string, string | string[] | null | undefined>; // Support both single values and arrays
}

export const DEFAULT_FILTERS: FragmentFilters = {
  lineCountMin: undefined,
  lineCountMax: undefined,
  scripts: [],
  isEdgePiece: null,
  hasTopEdge: null,
  hasBottomEdge: null,
  hasLeftEdge: null,
  hasRightEdge: null,
  hasCircle: null,
  search: undefined,
  custom: {},
};
