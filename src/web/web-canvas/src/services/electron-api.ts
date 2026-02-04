/**
 * Electron API service layer
 *
 * Provides typed access to the Electron IPC API from React components.
 */

// Types matching the preload script
export interface FragmentFilters {
  lineCountMin?: number;
  lineCountMax?: number;
  scripts?: string[];
  isEdgePiece?: boolean | null;
  hasCircle?: boolean | null;
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
  // Scale detection fields
  scale_unit: string | null;
  pixels_per_unit: number | null;
  scale_detection_status: string | null;
  scale_model_version: string | null;
  // Circle detection field
  has_circle: number | null;
}

export interface CanvasFragmentData {
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
  showSegmented?: boolean;
}

export interface CanvasStateData {
  fragments: CanvasFragmentData[];
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

// Declare the global electronAPI type
declare global {
  interface Window {
    electronAPI?: {
      fragments: {
        getAll: (filters?: FragmentFilters) => Promise<ApiResponse<FragmentRecord[]>>;
        getCount: (filters?: FragmentFilters) => Promise<ApiResponse<null> & { count?: number }>;
        getById: (id: string) => Promise<ApiResponse<FragmentRecord | null>>;
        updateMetadata: (id: string, metadata: Record<string, unknown>) => Promise<ApiResponse<null>>;
      };
      images: {
        getPath: (relativePath: string) => Promise<string>;
        hasSegmented: (fragmentId: string) => Promise<{ success: boolean; exists: boolean }>;
        batchHasSegmented: (fragmentIds: string[]) => Promise<ApiResponse<Record<string, boolean>>>;
      };
      projects: {
        list: () => Promise<ApiResponse<Project[]>>;
        create: (name: string, description?: string) => Promise<ApiResponse<null> & { projectId?: number }>;
        save: (projectId: number, canvasState: CanvasStateData, notes: string) => Promise<ApiResponse<null>>;
        load: (projectId: number) => Promise<ApiResponse<{ project: Project; canvasState: CanvasStateData; notes: string }>>;
        delete: (projectId: number) => Promise<ApiResponse<null> & { deleted?: boolean }>;
        rename: (projectId: number, newName: string) => Promise<ApiResponse<null>>;
      };
    };
  }
}

/**
 * Check if running in Electron environment
 */
export function isElectron(): boolean {
  return window.electronAPI !== undefined;
}

/**
 * Get the Electron API, throwing if not available
 */
export function getElectronAPI() {
  if (!window.electronAPI) {
    throw new Error('Electron API not available. Are you running in Electron?');
  }
  return window.electronAPI;
}

/**
 * Safely get the Electron API, returning null if not available
 */
export function getElectronAPISafe() {
  return window.electronAPI ?? null;
}
