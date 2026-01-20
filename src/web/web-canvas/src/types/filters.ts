export interface FragmentFilters {
  lineCountMin?: number;
  lineCountMax?: number;
  scripts: string[]; // Empty array = all scripts
  isEdgePiece?: boolean | null; // null = don't care
}

export const DEFAULT_FILTERS: FragmentFilters = {
  lineCountMin: undefined,
  lineCountMax: undefined,
  scripts: [],
  isEdgePiece: null,
};
