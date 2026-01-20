"""
Classification processor using CNN for line count prediction.

Predicts the number of text lines (0-15) in manuscript fragments.
"""

import os
import json
import sys
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T

# Add classification directory to path to import the model
classification_dir = Path(__file__).parent.parent.parent / 'classification'
sys.path.insert(0, str(classification_dir))

try:
    from classification_POC import LineCountCNN
except ImportError:
    LineCountCNN = None

from ml_pipeline.core.processor import (
    BaseProcessor,
    ProcessorMetadata,
    FragmentRecord,
    ProcessingResult
)


class ClassificationProcessor(BaseProcessor):
    """
    Classifies fragments by line count using CNN.

    - Predicts line count (0-15)
    - Stores prediction and confidence
    """

    def _setup(self) -> None:
        """Load CNN model"""
        if LineCountCNN is None:
            raise ImportError(
                "Could not import LineCountCNN from classification_POC.py. "
                "Make sure the classification module is available."
            )

        model_path = self.config['model_path']
        meta_path = self.config['meta_path']

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        # Load metadata
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)

        self.max_lines = self.meta['max_lines']
        self.img_size = self.meta['img_size']
        self.version = self.config['config']['model_version']

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = LineCountCNN(num_classes=self.max_lines + 1)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        # Define transforms (match training transforms without augmentation)
        self.transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.logger.info(
            f"Classification model loaded (max_lines={self.max_lines}, "
            f"img_size={self.img_size}, device={self.device})"
        )

    def get_metadata(self) -> ProcessorMetadata:
        return ProcessorMetadata(
            name="classification",
            version=self.version,
            description="CNN line count classifier",
            model_path=self.config['model_path'],
            requires_gpu=True,
            batch_size=self.config['config']['batch_size']
        )

    def should_process(self, fragment: FragmentRecord) -> bool:
        """Skip if already classified with this model version"""
        skip_processed = self.config.get('skip_processed', True)

        if not skip_processed:
            return True

        # Process if no line count OR different model version
        return (
            fragment.line_count is None or
            fragment.classification_model_version != self.version
        )

    def process(self, fragment: FragmentRecord, data_dir: str) -> ProcessingResult:
        """Classify line count"""
        try:
            # Construct full image path
            img_path = os.path.join(data_dir, fragment.image_path)

            if not os.path.exists(img_path):
                return ProcessingResult(
                    success=False,
                    updates={},
                    error=f"Image not found: {img_path}"
                )

            # Load and preprocess image
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_class = probs.argmax(dim=1).item()
                confidence = probs.max().item()

            return ProcessingResult(
                success=True,
                updates={
                    'line_count': pred_class,
                    'classification_model_version': self.version,
                    'processing_status': 'completed',
                    'last_processed_at': 'CURRENT_TIMESTAMP',
                    'processing_error': None  # Clear any previous errors
                },
                metadata={
                    'confidence': confidence,
                    'predicted_lines': pred_class
                }
            )

        except Exception as e:
            self.logger.error(f"Classification failed for {fragment.fragment_id}: {e}")
            return ProcessingResult(
                success=False,
                updates={'processing_error': str(e)},
                error=str(e)
            )

    def cleanup(self) -> None:
        """Release model resources"""
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Classification processor cleaned up")
