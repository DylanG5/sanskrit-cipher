"""
Scale Detection Processor for ML Pipeline

Processes fragments to extract scale information from rulers using OCR and computer vision.
Stores pixels-per-unit ratio and unit type (cm/mm) in the database.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Import scale detection module from same directory
from .scale_detection_easyocr import detect_scale_ratio

# Import ML pipeline base classes
from ml_pipeline.core.processor import (
    BaseProcessor,
    ProcessorMetadata,
    FragmentRecord,
    ProcessingResult
)


class ScaleDetectionProcessor(BaseProcessor):
    """
    Processes fragments to extract scale information from rulers.

    - Uses EasyOCR to detect unit text (cm/mm)
    - Uses computer vision to detect ruler tick marks
    - Calculates pixels per unit from tick spacing
    - Stores results in database
    - Optionally saves debug visualizations
    """

    def _setup(self) -> None:
        """Initialize the scale detection processor (no model loading needed)"""
        # Get config parameters
        self.version = self.config['config']['model_version']
        self.visualize = self.config['config'].get('visualize', False)
        self.visualize_output_dir = self.config['config'].get(
            'visualize_output_dir',
            '../scale_detection_feature/debug_visualizations'
        )
        self.skip_processed = self.config['config'].get('skip_processed', True)
        self.ocr_enabled = self.config['config'].get('ocr_enabled', True)

        # Resolve absolute path for visualizations
        if self.visualize:
            self.visualize_output_dir = str(Path(self.visualize_output_dir).resolve())
            Path(self.visualize_output_dir).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Visualizations will be saved to: {self.visualize_output_dir}")

        self.logger.info(
            f"Scale detection processor initialized (version={self.version}, "
            f"visualize={self.visualize}, ocr_enabled={self.ocr_enabled})"
        )

    def get_metadata(self) -> ProcessorMetadata:
        return ProcessorMetadata(
            name="scale_detection",
            version=self.version,
            description="Ruler-based scale detection using EasyOCR + CV",
            model_path=None,  # No model file needed
            requires_gpu=False,  # EasyOCR can use GPU but not required
            batch_size=1  # Process one image at a time
        )

    def should_process(self, fragment: FragmentRecord) -> bool:
        """
        Skip if already processed with this model version.

        Args:
            fragment: Fragment record from database

        Returns:
            True if fragment should be processed, False to skip
        """
        if not self.skip_processed:
            return True

        # Check if fragment has scale detection fields
        scale_unit = getattr(fragment, 'scale_unit', None)
        scale_model_version = getattr(fragment, 'scale_model_version', None)

        # Process if:
        # - No scale data exists (scale_unit is NULL), OR
        # - Scale data exists but was processed with different version
        return (
            scale_unit is None or
            scale_model_version != self.version
        )

    def process(self, fragment: FragmentRecord, data_dir: str) -> ProcessingResult:
        """
        Process a single fragment to detect scale.

        Args:
            fragment: Fragment record from database
            data_dir: Base directory containing fragment images

        Returns:
            ProcessingResult with database updates
        """
        try:
            # Construct full image path
            image_path = os.path.join(data_dir, fragment.image_path)

            # Verify image file exists
            if not os.path.exists(image_path):
                return ProcessingResult(
                    success=False,
                    error=f"Image file not found: {image_path}"
                )

            # Run scale detection
            result = detect_scale_ratio(
                image_path,
                visualize=self.visualize,
                output_dir=self.visualize_output_dir if self.visualize else None
            )

            # Check if detection was successful
            if result['status'] != 'success':
                # Failed detection - store NULL values with error message
                return ProcessingResult(
                    success=True,  # Still successful (no crash), just no data
                    updates={
                        'scale_unit': None,
                        'pixels_per_unit': None,
                        'scale_detection_status': f"error: {result.get('message', 'Unknown error')}",
                        'scale_model_version': self.version
                    },
                    metadata={
                        'detection_failed': True,
                        'reason': result.get('message', 'Unknown error')
                    }
                )

            # Successful detection - store scale data
            return ProcessingResult(
                success=True,
                updates={
                    'scale_unit': result['unit'],
                    'pixels_per_unit': result['pixels_per_unit'],
                    'scale_detection_status': 'success',
                    'scale_model_version': self.version
                },
                metadata={
                    'num_ticks': result['num_ticks'],
                    'tick_positions': result.get('tick_positions', [])
                }
            )

        except Exception as e:
            # Unexpected error during processing
            self.logger.error(f"Error processing fragment {fragment.fragment_id}: {e}", exc_info=True)
            return ProcessingResult(
                success=False,
                error=str(e)
            )

    def cleanup(self) -> None:
        """Cleanup resources (nothing to cleanup for this processor)"""
        self.logger.info("Scale detection processor cleanup complete")
