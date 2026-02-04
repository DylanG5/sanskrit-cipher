

import os
import json
import sys
import yaml
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
import torchvision.transforms as T

# Add yolo_script_type_classification directory to path to import the model
yolo_script_type_dir = Path(__file__).parent.parent.parent / 'yolo_script_type_classification'
sys.path.insert(0, str(yolo_script_type_dir))

try:
    from model import load_model, EfficientScriptTypeClassifier
except ImportError:
    load_model = None
    EfficientScriptTypeClassifier = None

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
        """Load script type efficient model (MobileNetV2-based)"""
        if load_model is None or EfficientScriptTypeClassifier is None:
            raise ImportError(
                "Could not import EfficientScriptTypeClassifier from model.py. "
                "Make sure the classification module is available."
            )

        model_path = self.config.get('model_path')
        meta_path = self.config.get('meta_path')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Script type model not found: {model_path}")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Model metadata not found: {meta_path}")

        # Load metadata (YAML format for efficient model)
        with open(meta_path, 'r') as f:
            if meta_path.endswith('.yaml'):
                self.meta = yaml.safe_load(f)
            else:
                self.meta = json.load(f)

        # Extract class names and configuration
        self.class_names = self.meta.get('class_names', [])
        self.img_size = self.meta.get('img_size', 224)
        self.num_classes = len(self.class_names)
        
        self.logger.info(f"Loaded metadata: {len(self.class_names)} classes")
        self.logger.info(f"Class names: {self.class_names}")
        
        self.version = self.config.get('config', {}).get('model_version', '2.0')
        self.confidence_threshold = self.config.get('config', {}).get('confidence_threshold', 0.0)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load efficient model using model.py's load_model utility
        self.model = load_model(
            model_path=model_path,
            num_classes=self.num_classes,
            device=str(self.device)
        )
        self.model.eval()

        # Define transforms (match training transforms without augmentation)
        self.transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.logger.info(
            f"Script type classification model loaded (MobileNetV2, num_classes={self.num_classes}, "
            f"img_size={self.img_size}, device={self.device}, "
            f"confidence_threshold={self.confidence_threshold})"
        )
        self.logger.info(f"Script types: {', '.join(self.class_names)}")

    def get_metadata(self) -> ProcessorMetadata:
        return ProcessorMetadata(
            name="script_type_classification",
            version=self.version,
            description="Efficient MobileNetV2-based script type classifier for Sanskrit manuscripts",
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
            fragment.classification_model_version != self.version
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

            # Prepare database updates
            updates = {
                'script_type': pred_class_name,
                'classification_model_version': self.version,
                'script_type_confidence': confidence,
                'processing_status': 'completed' if not below_threshold else 'completed_low_confidence',
                'last_processed_at': 'CURRENT_TIMESTAMP',
                'processing_error': None  # Clear any previous errors
            }

            # Optionally store detailed class probabilities as JSON
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
