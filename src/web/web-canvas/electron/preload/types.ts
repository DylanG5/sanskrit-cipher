export interface FragmentFilters {
  lineCountMin?: number;
  lineCountMax?: number;
  scripts?: string[];
  isEdgePiece?: boolean | null;
}

export interface ElectronAPI {
  fragments: {
    getAll: (filters?: FragmentFilters) => Promise<any[]>;
    getById: (id: string) => Promise<any>;
    updateMetadata: (id: string, metadata: any) => Promise<void>;
  };
  projects: {
    create: (name: string, description?: string) => Promise<number>;
    getAll: () => Promise<any[]>;
    save: (projectId: number, canvasState: any) => Promise<void>;
    load: (projectId: number) => Promise<{ canvasState: any; notes: string }>;
    delete: (projectId: number) => Promise<void>;
  };
}
