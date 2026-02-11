export interface FragmentFilters {
  lineCountMin?: number;
  lineCountMax?: number;
  scripts?: string[];
  isEdgePiece?: boolean | null;
  hasTopEdge?: boolean | null;
  hasBottomEdge?: boolean | null;
  hasLeftEdge?: boolean | null;
  hasRightEdge?: boolean | null;
  hasCircle?: boolean | null;
  search?: string;
  custom?: Record<string, string | null | undefined>;
  limit?: number;
  offset?: number;
}

export interface ElectronAPI {
  fragments: {
    getAll: (filters?: FragmentFilters) => Promise<any[]>;
    getById: (id: string) => Promise<any>;
    updateMetadata: (id: string, metadata: any) => Promise<void>;
  };
  customFilters: {
    list: () => Promise<any[]>;
    create: (payload: { label: string; type: 'dropdown' | 'text'; options?: string[] }) => Promise<any>;
    delete: (id: number) => Promise<any>;
    updateOptions: (id: number, options: string[]) => Promise<any>;
  };
  projects: {
    create: (name: string, description?: string) => Promise<number>;
    getAll: () => Promise<any[]>;
    save: (projectId: number, canvasState: any) => Promise<void>;
    load: (projectId: number) => Promise<{ canvasState: any; notes: string }>;
    delete: (projectId: number) => Promise<void>;
  };
}
