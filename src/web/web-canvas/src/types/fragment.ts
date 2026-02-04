export interface ManuscriptFragment {
  id: string;
  name: string;
  imagePath: string;
  thumbnailPath: string;
  // Segmentation info
  segmentedImagePath?: string;  // Path to segmented image if available
  hasSegmentation?: boolean;     // Quick flag for UI
  segmentationCoords?: string;   // JSON string with polygon contour coordinates
  // Metadata fields (optional, populated by ML models)
  metadata?: {
    lineCount?: number;
    script?: string;
    isEdgePiece?: boolean;
    hasTopEdge?: boolean;
    hasBottomEdge?: boolean;
    hasLeftEdge?: boolean;
    hasRightEdge?: boolean;
    hasCircle?: boolean;  // Circle classification from ML model
    // Scale information from ruler detection
    scale?: {
      unit: 'cm' | 'mm';  // Physical unit
      pixelsPerUnit: number;  // Pixels per unit in original image
      detectionStatus: 'success' | 'error';  // Detection result
    };
    segmentationStatus?: string; // For displaying segmentation processing status
  };
}

export interface CanvasFragment {
  id: string;
  fragmentId: string;
  name: string;
  imagePath: string;
  segmentationCoords?: string; // JSON string with polygon contour coordinates
  x: number;
  y: number;
  width: number;
  height: number;
  rotation: number;
  scaleX: number;
  scaleY: number;
  isLocked: boolean;
  isSelected: boolean;
  showSegmented: boolean; // Per-fragment toggle for showing segmented version
  originalWidth?: number; // Original image width in pixels
  originalHeight?: number; // Original image height in pixels
  hasBeenResized?: boolean; // Flag to indicate manual resize
}

export interface CanvasState {
  fragments: CanvasFragment[];
  selectedFragmentIds: string[];
  stageScale: number;
  stageX: number;
  stageY: number;
}
