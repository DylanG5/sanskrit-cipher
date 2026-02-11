export type CustomFilterType = 'multiselect' | 'text';

export interface CustomFilterDefinition {
  id: number;
  filterKey: string; // Column name in fragments table
  label: string;
  type: CustomFilterType;
  options?: string[]; // Only for multiselect filters
}
