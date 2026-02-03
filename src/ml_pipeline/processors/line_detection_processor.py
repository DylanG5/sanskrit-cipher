"""
Line detection processor using YOLOv8 model.

Detects individual text lines in Sanskrit manuscript fragments using object detection.
Provides line count and bounding box coordinates for each detected line.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
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


class LineDetectionProcessor(BaseProcessor):
    """
    Detects text lines in fragments using YOLOv8 object detection.

    - Detects individual line instances
    - Counts total number of lines
    - Provides bounding box coordinates for each line
    - Stores line count and detection data in database
    """

    def _setup(self) -> None:
        """Load YOLOv8 line detection model"""
        if YOLO is None:
            raise ImportError(
                "ultralytics package not found. Please install it: pip install ultralytics"
            )

        model_path = self.config.get('model_path')
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Line detection model not found at: {model_path}"
            )

        self.model = YOLO(model_path)
        self.confidence = self.config.get('config', {}).get('confidence_threshold', 0.25)
        self.iou = self.config.get('config', {}).get('iou_threshold', 0.45)
        self.version = self.config.get('config', {}).get('model_version', '1.0')

        self.logger.info(f"Loaded YOLOv8 line detection model from {model_path}")
        self.logger.info(f"Confidence threshold: {self.confidence}, IOU threshold: {self.iou}")

    def get_metadata(self) -> ProcessorMetadata:
        return ProcessorMetadata(
            name="linedetection",
            version=self.version,
            description="YOLOv8-based text line detection and counting",
            model_path=self.config.get('model_path'),
            requires_gpu=True,
            batch_size=1
        )

    def should_process(self, fragment: FragmentRecord) -> bool:
        """Skip if already processed with this model version"""
        skip_processed = self.config.get('config', {}).get('skip_processed', True)

        if not skip_processed:
            return True

        # Process if no line detection data OR different model version
        # Assuming we'll add line_detection_model_version field to database
        return (
            not hasattr(fragment, 'line_detection_model_version') or
            fragment.line_detection_model_version != self.version
        )

    def process(self, fragment: FragmentRecord, data_dir: str) -> ProcessingResult:
        """Detect lines in fragment image"""
        try:
            # Construct full image path
            img_path = os.path.join(data_dir, fragment.image_path)

            if not os.path.exists(img_path):
                return ProcessingResult(
                    success=False,
                    updates={},
                    error=f"Image not found: {img_path}"
                )

            # Run YOLO inference
            results = self.model.predict(
                source=img_path,
                conf=self.confidence,
                iou=self.iou,
                verbose=False
            )

            result = results[0]

            # Extract detection results
            if result.boxes is None or len(result.boxes) == 0:
                # No lines detected
                line_count = 0
                detection_data = {
                    "num_lines": 0,
                    "lines": [],
                    "model_version": self.version
                }
                mean_confidence = 0.0
            else:
                # Extract bounding boxes and confidences
                boxes = result.boxes.data.cpu().numpy()
                line_count = len(boxes)

                # Parse each detection
                lines = []
                confidences = []
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2, conf, cls = box
                    lines.append({
                        "line_id": i + 1,
                        "bbox": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2)
                        },
                        "confidence": float(conf)
                    })
                    confidences.append(float(conf))

                mean_confidence = float(np.mean(confidences))

                # Sort lines by y-coordinate (top to bottom)
                lines.sort(key=lambda l: l['bbox']['y1'])
                
                # Reassign line IDs after sorting
                for i, line in enumerate(lines):
                    line['line_id'] = i + 1

                detection_data = {
                    "num_lines": line_count,
                    "lines": lines,
                    "mean_confidence": mean_confidence,
                    "model_version": self.version
                }

            # Prepare database updates
            updates = {
                'line_count': line_count,
                'line_detection_data': json.dumps(detection_data),
                'line_detection_model_version': self.version,
                'line_detection_confidence': mean_confidence,
                'processing_status': 'completed',
                'last_processed_at': 'CURRENT_TIMESTAMP',
                'processing_error': None
            }

            return ProcessingResult(
                success=True,
                updates=updates,
                metadata={
                    'line_count': line_count,
                    'mean_confidence': mean_confidence
                }
            )

        except Exception as e:
            self.logger.error(f"Line detection failed for {fragment.fragment_id}: {e}")
            return ProcessingResult(
                success=False,
                updates={'processing_error': str(e)},
                error=str(e)
            )

    def cleanup(self) -> None:
        """Release model resources"""
        self.model = None
        self.logger.info("Line detection processor cleaned up")
