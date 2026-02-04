"""
Evaluation module for script type classification
"""

import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from dataset import ScriptTypeDataset
from model import load_model


def evaluate(model_path, images_dir, class_names, img_size=224, batch_size=32, device='cpu'):
    """
    Evaluate model on a dataset
    
    Args:
        model_path: Path to trained model
        images_dir: Path to images directory
        class_names: List of class names
        img_size: Input image size
        batch_size: Batch size for evaluation
        device: Device to use
    
    Returns:
        Dict with evaluation metrics
    """
    device = torch.device(device)
    
    # Load model
    model = load_model(model_path, num_classes=len(class_names), device=str(device))
    
    # Dataset and loader
    dataset = ScriptTypeDataset(images_dir, img_size=img_size, augment=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, filenames in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
    
    # Note: This assumes filenames follow a pattern or labels are available separately
    # For full evaluation, you would need corresponding labels
    
    return {
        "num_samples": len(dataset),
        "predictions": all_preds,
    }


def evaluate_with_labels(model_path, images_dir, labels_dict, class_names, img_size=224, batch_size=32, device='cpu'):
    """
    Evaluate model with known labels
    
    Args:
        model_path: Path to trained model
        images_dir: Path to images directory
        labels_dict: Dict mapping filenames to class indices
        class_names: List of class names
        img_size: Input image size
        batch_size: Batch size for evaluation
        device: Device to use
    
    Returns:
        Dict with comprehensive evaluation metrics
    """
    device = torch.device(device)
    
    # Load model
    model = load_model(model_path, num_classes=len(class_names), device=str(device))
    
    # Dataset and loader
    dataset = ScriptTypeDataset(images_dir, img_size=img_size, augment=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, filenames in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            for filename, pred in zip(filenames, preds):
                all_preds.append(pred)
                if filename in labels_dict:
                    all_labels.append(labels_dict[filename])
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": {
            class_names[i]: {
                "precision": float(per_class_precision[i]),
                "recall": float(per_class_recall[i]),
                "f1": float(per_class_f1[i])
            }
            for i in range(len(class_names))
        }
    }
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate script type classifier")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--images", type=str, required=True, help="Path to images directory")
    parser.add_argument("--classes", type=str, help="Comma-separated class names")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    class_names = args.classes.split(",") if args.classes else [f"Class_{i}" for i in range(5)]
    
    metrics = evaluate(args.model, args.images, class_names, batch_size=args.batch_size)
    
    print(json.dumps(metrics, indent=2))
