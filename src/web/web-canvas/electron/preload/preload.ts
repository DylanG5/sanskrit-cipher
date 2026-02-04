import { contextBridge, ipcRenderer } from 'electron';

// Types for the API
export interface FragmentFilters {
  lineCountMin?: number;
  lineCountMax?: number;
  scripts?: string[];
  isEdgePiece?: boolean | null;
  search?: string;
  limit?: number;
  offset?: number;
}

export interface FragmentRecord {
  id: number;
  fragment_id: string;
  image_path: string;
  edge_piece: number;
  has_top_edge: number;
  has_bottom_edge: number;
  line_count: number | null;
  script_type: string | null;
  segmentation_coords: string | null;
  notes: string | null;
  created_at: string;
  updated_at: string;
}

export interface CanvasFragment {
  fragmentId: string;
  x: number;
  y: number;
  width?: number;
  height?: number;
  rotation: number;
  scaleX: number;
  scaleY: number;
  isLocked: boolean;
  zIndex?: number;
}

export interface CanvasState {
  fragments: CanvasFragment[];
}

export interface Project {
  id: number;
  project_name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  count?: number;
  changes?: number;
  projectId?: number;
  deleted?: boolean;
}

// Additional types for file upload
export interface FileSelectionResult {
  success: boolean;
  canceled?: boolean;
  filePaths?: string[];
  error?: string;
}

export interface UploadResult {
  success: boolean;
  fragmentId?: string;
  filename?: string;
  error?: string;
}

export interface UploadResponse {
  success: boolean;
  results: UploadResult[];
}

// Expose a safe API to the renderer process
const electronAPI = {
  files: {
    selectImages: (): Promise<FileSelectionResult> =>
      ipcRenderer.invoke('files:selectImages'),
  },
  fragments: {
    getAll: (filters?: FragmentFilters): Promise<ApiResponse<FragmentRecord[]>> =>
      ipcRenderer.invoke('fragments:getAll', filters),
    getCount: (filters?: FragmentFilters): Promise<ApiResponse<null> & { count?: number }> =>
      ipcRenderer.invoke('fragments:getCount', filters),
    getById: (id: string): Promise<ApiResponse<FragmentRecord | null>> =>
      ipcRenderer.invoke('fragments:getById', id),
    updateMetadata: (id: string, metadata: Record<string, unknown>): Promise<ApiResponse<null>> =>
      ipcRenderer.invoke('fragments:updateMetadata', id, metadata),
    uploadFiles: (filePaths: string[]): Promise<UploadResponse> =>
      ipcRenderer.invoke('fragments:uploadFiles', filePaths),
  },
  images: {
    getPath: (relativePath: string): Promise<string> =>
      ipcRenderer.invoke('images:getPath', relativePath),
    hasSegmented: (fragmentId: string): Promise<{ success: boolean; exists: boolean }> =>
      ipcRenderer.invoke('images:hasSegmented', fragmentId),
    batchHasSegmented: (fragmentIds: string[]): Promise<ApiResponse<Record<string, boolean>>> =>
      ipcRenderer.invoke('images:batchHasSegmented', fragmentIds),
  },
  projects: {
    list: (): Promise<ApiResponse<Project[]>> =>
      ipcRenderer.invoke('projects:list'),
    create: (name: string, description?: string): Promise<ApiResponse<null> & { projectId?: number }> =>
      ipcRenderer.invoke('projects:create', name, description),
    save: (projectId: number, canvasState: CanvasState): Promise<ApiResponse<null>> =>
      ipcRenderer.invoke('projects:save', projectId, canvasState),
    load: (projectId: number): Promise<ApiResponse<{ project: Project; canvasState: CanvasState; notes: string }>> =>
      ipcRenderer.invoke('projects:load', projectId),
    delete: (projectId: number): Promise<ApiResponse<null> & { deleted?: boolean }> =>
      ipcRenderer.invoke('projects:delete', projectId),
    rename: (projectId: number, newName: string): Promise<ApiResponse<null>> =>
      ipcRenderer.invoke('projects:rename', projectId, newName),
  },
};

contextBridge.exposeInMainWorld('electronAPI', electronAPI);

// Export types for use in renderer
export type ElectronAPI = typeof electronAPI;
