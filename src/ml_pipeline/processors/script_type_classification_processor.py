

import os
import json
import sys
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
import torchvision.transforms as T

# Add classification directory to path to import the model
classification_dir = Path(__file__).parent.parent.parent / 'classification'
sys.path.insert(0, str(classification_dir))

try:
    from classify_script_types import ScriptTypeCNN
except ImportError:
    ScriptTypeCNN = None

from ml_pipeline.core.processor import (
    BaseProcessor,
    ProcessorMetadata,
    FragmentRecord,
    ProcessingResult
)


class ScriptTypeClassificationProcessor(BaseProcessor):
    """
    Classifies fragments by Sanskrit script type using CNN.

    - Predicts script type (e.g., South Turkestan Brahmi, North Turkestan Brahmi, etc.)
    - Stores prediction confidence and all class probabilities
    - Supports configurable confidence thresholds for uncertain predictions
    """

    def _setup(self) -> None:
        """Load script type CNN model"""
        if ScriptTypeCNN is None:
            raise ImportError(
                "Could not import ScriptTypeCNN from classify_script_types.py. "
                "Make sure the classification module is available."
            )

        model_path = self.config.get('model_path')
        meta_path = self.config.get('meta_path')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Script type model not found: {model_path}")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Model metadata not found: {meta_path}")

        # Load metadata
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)

        self.class_names = self.meta.get('class_names', [])
        self.img_size = self.meta.get('img_size', 224)
        self.num_classes = len(self.class_names)
        self.version = self.config.get('config', {}).get('model_version', '1.0')
        self.confidence_threshold = self.config.get('config', {}).get('confidence_threshold', 0.0)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = ScriptTypeCNN(num_classes=self.num_classes, dropout=0.2)
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
            f"Script type classification model loaded (num_classes={self.num_classes}, "
            f"img_size={self.img_size}, device={self.device}, "
            f"confidence_threshold={self.confidence_threshold})"
        )
        self.logger.info(f"Script types: {', '.join(self.class_names)}")

    def get_metadata(self) -> ProcessorMetadata:
        return ProcessorMetadata(
            name="script_type_classification",
            version=self.version,
            description="CNN script type classifier for Sanskrit manuscripts",
            model_path=self.config.get('model_path'),
            requires_gpu=True,
            batch_size=self.config.get('config', {}).get('batch_size', 16)
        )

    def should_process(self, fragment: FragmentRecord) -> bool:
        """Skip if already classified with this model version"""
        skip_processed = self.config.get('config', {}).get('skip_processed', True)

        if not skip_processed:
            return True

        # Process if no script type OR different model version
        return (
            fragment.script_type is None or
            fragment.script_type_classification_model_version != self.version
        )

    def process(self, fragment: FragmentRecord, data_dir: str) -> ProcessingResult:
        """Classify script type"""
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
                pred_class_idx = probs.argmax(dim=1).item()
                pred_class_name = self.class_names[pred_class_idx]
                confidence = probs.max().item()

            # Build probability distribution for all classes
            class_probabilities = {
                self.class_names[i]: float(probs[0][i].item())
                for i in range(self.num_classes)
            }

            # Check if confidence meets threshold
            below_threshold = confidence < self.confidence_threshold

            # Prepare detailed classification data for database storage
            classification_data = {
                'predicted_class': pred_class_name,
                'confidence': confidence,
                'class_probabilities': class_probabilities,
                'model_version': self.version
            }

            # Prepare database updates
            updates = {
                'script_type': pred_class_name,
                'script_type_classification_model_version': self.version,
                'script_type_confidence': confidence,
                'script_type_classification_data': json.dumps(classification_data),
                'processing_status': 'completed' if not below_threshold else 'completed_low_confidence',
                'last_processed_at': 'CURRENT_TIMESTAMP',
                'processing_error': None  # Clear any previous errors
            }

            # Metadata for logging/debugging
            metadata = {
                'script_type': pred_class_name,
                'confidence': confidence,
                'below_threshold': below_threshold,
                'class_probabilities': class_probabilities
            }

            return ProcessingResult(
                success=True,
                updates=updates,
                metadata=metadata
            )

        except Exception as e:
            self.logger.error(f"Script type classification failed for {fragment.fragment_id}: {e}")
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
        self.logger.info("Script type classification processor cleaned up")
