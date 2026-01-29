from pathlib import Path
from ultralytics import YOLO
from ml_pipeline.core.processor import BaseProcessor, ProcessorMetadata, FragmentRecord, ProcessingResult
import os

CIRCLE_CLASS_ID = 0

class CircleDetectionProcessor(BaseProcessor):
    def _setup(self) -> None:
        model_path = self.config['model_path']
        self.conf = self.config['config']['confidence_threshold']
        self.version = self.config['config']['model_version']
        self.model = YOLO(model_path)

    def get_metadata(self) -> ProcessorMetadata:
        return ProcessorMetadata(
            name="circledetection",
            version=self.version,
            description="Detects presence of circle",
            model_path=self.config['model_path'],
            requires_gpu=True,
            batch_size=1,
        )

    def should_process(self, fragment: FragmentRecord) -> bool:
        # optional: only run if has_circle is None
        return True

    def process(self, fragment: FragmentRecord, data_dir: str) -> ProcessingResult:
        img_path = os.path.join(data_dir, fragment.image_path)
        if not os.path.exists(img_path):
            return ProcessingResult(success=False, updates={}, error=f"Image not found: {img_path}")

        results = self.model.predict(
            source=img_path,
            conf=self.conf,
            save=False,
            save_txt=False,
            save_conf=False,
            stream=False,  # single image
            task="segment",
            verbose=False,
        )

        res = results[0]
        circle_present = False

        if res.boxes is not None and len(res.boxes) > 0:
            for cls, conf in zip(res.boxes.cls, res.boxes.conf):
                if int(cls) == CIRCLE_CLASS_ID and float(conf) >= self.conf:
                    circle_present = True
                    break

        updates = {
            'has_circle': 1 if circle_present else 0,
            'processing_status': 'completed',
            'last_processed_at': 'CURRENT_TIMESTAMP',
            'processing_error': None,
        }

        return ProcessingResult(
            success=True,
            updates=updates,
            metadata={'circle_present': circle_present},
        )

    def cleanup(self) -> None:
        self.model = None
