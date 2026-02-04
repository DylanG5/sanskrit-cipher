"""
Inference module for script type prediction
"""

import json
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from model import load_model


class ScriptTypePredictor:
    """Wrapper for script type prediction"""
    
    def __init__(self, model_path, meta_path=None, device='cpu'):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model weights
            meta_path: Path to metadata JSON file
            device: torch device
        """
        self.device = torch.device(device)
        
        # Load metadata
        if meta_path and Path(meta_path).exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
        else:
            self.meta = {}
        
        num_classes = self.meta.get("num_classes", 5)
        self.class_names = self.meta.get("class_names", [f"Class_{i}" for i in range(num_classes)])
        self.img_size = self.meta.get("img_size", 224)
        
        # Load model
        self.model = load_model(model_path, num_classes=len(self.class_names), device=str(self.device))
    
    def preprocess(self, image_path):
        """Preprocess image for inference"""
        image = Image.open(image_path).convert("RGB")
        
        transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image = transform(image)
        return image.unsqueeze(0).to(self.device)
    
    def predict(self, image_path, return_probabilities=False):
        """
        Predict script type for an image
        
        Args:
            image_path: Path to image
            return_probabilities: Whether to return class probabilities
        
        Returns:
            Dict with prediction results
        """
        with torch.no_grad():
            image = self.preprocess(image_path)
            logits = self.model(image)
            probabilities = F.softmax(logits, dim=1)[0]
            predicted_idx = probabilities.argmax().item()
            confidence = probabilities[predicted_idx].item()
        
        result = {
            "image": str(image_path),
            "predicted_class": self.class_names[predicted_idx],
            "confidence": float(confidence),
            "predicted_idx": int(predicted_idx)
        }
        
        if return_probabilities:
            result["probabilities"] = {
                name: float(prob) 
                for name, prob in zip(self.class_names, probabilities)
            }
        
        return result
    
    def predict_batch(self, image_paths, return_probabilities=False):
        """Predict for multiple images"""
        results = []
        for img_path in image_paths:
            result = self.predict(img_path, return_probabilities)
            results.append(result)
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict script type")
    parser.add_argument("image", help="Image path to classify")
    parser.add_argument(
        "--model",
        type=str,
        default="./output_efficient/weights/best.pt",
        help="Path to model weights"
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="./output_efficient/metadata.json",
        help="Path to metadata file"
    )
    parser.add_argument(
        "--probs",
        action="store_true",
        help="Show all class probabilities"
    )
    
    args = parser.parse_args()
    
    predictor = ScriptTypePredictor(args.model, args.meta)
    result = predictor.predict(args.image, return_probabilities=args.probs)
    
    print(f"Predicted: {result['predicted_class']} (confidence: {result['confidence']:.3f})")
    
    if args.probs and "probabilities" in result:
        print("\nAll class probabilities:")
        for cls_name, prob in result['probabilities'].items():
            print(f"  {cls_name}: {prob:.3f}")
