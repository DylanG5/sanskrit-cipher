
import { ManuscriptFragment } from './fragment';

export interface SearchQuery {
  searchText?: string;
  scripts: string[];
  lineCountMin?: number;
  lineCountMax?: number;
  isEdgePiece?: boolean | null;
}


export const DEFAULT_SEARCH_QUERY: SearchQuery = {
  searchText: undefined,
  scripts: [],
  lineCountMin: undefined,
  lineCountMax: undefined,
  isEdgePiece: null,
};


export const RESULTS_PAGE_SIZE = 25;

export interface SearchResultPage {
  items: ManuscriptFragment[];
  totalCount: number;
  currentPage: number;
  totalPages: number;
  hasNextPage: boolean;
  hasPreviousPage: boolean;
}


export const EMPTY_SEARCH_RESULT_PAGE: SearchResultPage = {
  items: [],
  totalCount: 0,
  currentPage: 0,
  totalPages: 0,
  hasNextPage: false,
  hasPreviousPage: false,
};


export type ExportFormat = 'csv' | 'json';


export const EXPORT_FORMATS: { value: ExportFormat; label: string }[] = [
  { value: 'csv', label: 'CSV (.csv)' },
  { value: 'json', label: 'JSON (.json)' },
];

// Exception Types (per MIS specification)

export class SearchModuleError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SearchModuleError';
  }
}

export class UIRenderError extends SearchModuleError {
  constructor(message: string = 'Failed to render search panel') {
    super(message);
    this.name = 'UIRenderError';
  }
}

export class ValidationError extends SearchModuleError {
  public readonly field?: string;

  constructor(message: string, field?: string) {
    super(message);
    this.name = 'ValidationError';
    this.field = field;
  }
}

export class NetworkError extends SearchModuleError {
  constructor(message: string = 'Backend service is unreachable') {
    super(message);
    this.name = 'NetworkError';
  }
}

export class ExportError extends SearchModuleError {
  public readonly format: ExportFormat;

  constructor(message: string, format: ExportFormat) {
    super(message);
    this.name = 'ExportError';
    this.format = format;
  }
}
