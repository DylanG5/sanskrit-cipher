export interface ManuscriptFragment {
  id: string;
  name: string;
  imagePath: string;
  thumbnailPath: string;
  // Segmentation info
  segmentedImagePath?: string;  // Path to segmented image if available
  hasSegmentation?: boolean;     // Quick flag for UI
  // Metadata fields (optional, populated by ML models)
  metadata?: {
    lineCount?: number;
    script?: string;
    isEdgePiece?: boolean;
    // Scale information from ruler detection
    scale?: {
      unit: 'cm' | 'mm';  // Physical unit
      pixelsPerUnit: number;  // Pixels per unit in original image
      detectionStatus: 'success' | 'error';  // Detection result
    };
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
