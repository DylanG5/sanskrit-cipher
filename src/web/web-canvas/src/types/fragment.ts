export interface ManuscriptFragment {
  id: string;
  name: string;
  imagePath: string;
  thumbnailPath: string;
  // Metadata fields (optional, populated by ML models)
  metadata?: {
    lineCount?: number;
    script?: string;
    isEdgePiece?: boolean;
  };
}

export interface CanvasFragment {
  id: string;
  fragmentId: string;
  name: string;
  imagePath: string;
  x: number;
  y: number;
  width: number;
  height: number;
  rotation: number;
  scaleX: number;
  scaleY: number;
  isLocked: boolean;
  isSelected: boolean;
}

export interface CanvasState {
  fragments: CanvasFragment[];
  selectedFragmentIds: string[];
  stageScale: number;
  stageX: number;
  stageY: number;
}
