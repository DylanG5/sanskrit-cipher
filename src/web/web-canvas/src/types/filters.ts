export interface FragmentFilters {
  lineCountMin?: number;
  lineCountMax?: number;
  scripts: string[]; // Empty array = all scripts
  isEdgePiece?: boolean | null; // null = don't care
  search?: string; // Fragment ID search query
}

export const DEFAULT_FILTERS: FragmentFilters = {
  lineCountMin: undefined,
  lineCountMax: undefined,
  scripts: [],
  isEdgePiece: null,
  search: undefined,
};
