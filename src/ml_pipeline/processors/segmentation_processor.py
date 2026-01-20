"""
Segmentation processor using YOLO model.

Processes fragments to extract contours and generate transparent PNGs
with background removed.
"""

import os
import json
from pathlib import Path
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from ml_pipeline.core.processor import (
    BaseProcessor,
    ProcessorMetadata,
    FragmentRecord,
    ProcessingResult
)


class SegmentationProcessor(BaseProcessor):
    """
    Processes fragments using YOLO segmentation model.

    - Extracts contour coordinates from masks
    - Generates transparent PNG with background removed
    - Stores contours in database and PNG in cache
    """

    def _setup(self) -> None:
        """Load YOLO model"""
        if YOLO is None:
            raise ImportError(
                "ultralytics package not installed. "
                "Install with: pip install ultralytics"
            )

        model_path = self.config.get('model_path')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.logger.info(f"Loading segmentation model: {model_path}")
        self.model = YOLO(model_path)

        # Get config parameters
        self.confidence = self.config['config']['confidence_threshold']
        self.iou = self.config['config']['iou_threshold']
        self.version = self.config['config']['model_version']

        self.logger.info(
            f"Segmentation model loaded (confidence={self.confidence}, iou={self.iou})"
        )

    def get_metadata(self) -> ProcessorMetadata:
        return ProcessorMetadata(
            name="segmentation",
            version=self.version,
            description="YOLO-based fragment segmentation",
            model_path=self.config.get('model_path'),
            requires_gpu=True,
            batch_size=1
        )

    def should_process(self, fragment: FragmentRecord) -> bool:
        """Skip if already segmented with this model version"""
        skip_processed = self.config.get('skip_processed', True)

        if not skip_processed:
            return True

        # Process if no segmentation data OR different model version
        return (
            fragment.segmentation_coords is None or
            fragment.segmentation_model_version != self.version
        )

    def process(self, fragment: FragmentRecord, data_dir: str) -> ProcessingResult:
        """Run segmentation on a fragment"""
        try:
            # Construct full image path
            img_path = os.path.join(data_dir, fragment.image_path)

            if not os.path.exists(img_path):
                return ProcessingResult(
                    success=False,
                    updates={},
                    error=f"Image not found: {img_path}"
                )

            # Load image
            img = cv2.imread(img_path)
            if img is None:
                return ProcessingResult(
                    success=False,
                    updates={},
                    error=f"Failed to read image: {img_path}"
                )

            # Run YOLO inference
            results = self.model.predict(
                source=img_path,
                conf=self.confidence,
                iou=self.iou,
                verbose=False
            )

            result = results[0]

            # Check if any masks were detected
            if result.masks is None or len(result.masks) == 0:
                return ProcessingResult(
                    success=False,
                    updates={'processing_error': 'No masks detected'},
                    error="No masks detected"
                )

            # Extract first mask (assuming single fragment per image)
            mask = result.masks.data[0].cpu().numpy()
            boxes = result.boxes.data[0].cpu().numpy()
            confidence = float(boxes[4])

            # Resize mask to image dimensions
            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)

            # Extract contours
            contours, _ = cv2.findContours(
                mask_binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # Convert contours to JSON-serializable format
            contours_list = []
            for contour in contours:
                # Convert numpy array to list, handling different shapes
                contour_squeezed = contour.squeeze()
                if contour_squeezed.ndim == 1:
                    # Single point
                    contours_list.append(contour_squeezed.tolist())
                else:
                    # Multiple points
                    contours_list.append(contour_squeezed.tolist())

            contours_data = {
                "contours": contours_list,
                "confidence": confidence,
                "model_version": self.version
            }

            # Generate transparent PNG
            transparent_png = self._create_transparent_fragment(img, mask_binary, boxes)

            # Prepare cache file path
            cache_filename = f"{fragment.fragment_id}_segmented.png"

            return ProcessingResult(
                success=True,
                updates={
                    'segmentation_coords': json.dumps(contours_data),
                    'segmentation_model_version': self.version,
                    'processing_status': 'completed',
                    'last_processed_at': 'CURRENT_TIMESTAMP',
                    'processing_error': None  # Clear any previous errors
                },
                cache_files={
                    cache_filename: transparent_png
                },
                metadata={
                    'confidence': confidence,
                    'num_contours': len(contours)
                }
            )

        except Exception as e:
            self.logger.error(f"Segmentation failed for {fragment.fragment_id}: {e}")
            return ProcessingResult(
                success=False,
                updates={'processing_error': str(e)},
                error=str(e)
            )

    def _create_transparent_fragment(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        box: np.ndarray
    ) -> np.ndarray:
        """
        Create transparent PNG from mask.

        Args:
            img: Original image (BGR)
            mask: Binary mask (0 or 1)
            box: Bounding box [x1, y1, x2, y2, confidence, ...]

        Returns:
            RGBA image with transparent background
        """
        # Create RGBA image
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        # Apply mask to alpha channel
        img_rgba[:, :, 3] = mask * 255

        # Crop to bounding box with padding
        x1, y1, x2, y2 = box[:4].astype(int)
        padding = 10

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img.shape[1], x2 + padding)
        y2 = min(img.shape[0], y2 + padding)

        # Crop the image
        cropped = img_rgba[y1:y2, x1:x2]

        return cropped

    def cleanup(self) -> None:
        """Release model resources"""
        self.model = None
        self.logger.info("Segmentation processor cleaned up")
