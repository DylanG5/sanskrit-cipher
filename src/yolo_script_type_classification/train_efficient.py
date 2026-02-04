"""
Script Type Classification Training - Modular YOLO-style training pipeline
"""

import os
import json
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml

from prepare_data import prepare_data_split
from dataset import ScriptTypeDataset, LabeledScriptTypeDataset
from model import EfficientScriptTypeClassifier


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
    
    return running_loss / total, correct / total


def eval_model(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return running_loss / total, correct / total


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data split
    print("Preparing data split...")
    config, meta = prepare_data_split(
        args.csv,
        args.images,
        args.output,
        args.val_split,
        args.test_split
    )
    
    class_names = meta["class_names"]
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    
    # Create datasets
    print("Loading datasets...")
    train_ds = LabeledScriptTypeDataset(
        config["train_csv"],
        config["train"],
        img_size=args.img_size,
        augment=True
    )
    val_ds = LabeledScriptTypeDataset(
        config["val_csv"],
        config["val"],
        img_size=args.img_size,
        augment=False
    )
    test_ds = LabeledScriptTypeDataset(
        config["test_csv"],
        config["test"],
        img_size=args.img_size,
        augment=False
    )
    
    # Weighted sampler for class imbalance
    train_labels = [label for _, label in train_ds.samples]
    class_counts = {}
    for label in train_labels:
        class_counts[label] = class_counts.get(label, 0) + 1

    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights.get(label, 1.0) for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Model
    print(f"Loading model with {num_classes} classes...")
    model = EfficientScriptTypeClassifier(
        num_classes=num_classes,
        pretrained=True
    )
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )
    
    # Training loop
    best_val_acc = 0.0
    output_dir = Path(args.output) / "weights"
    output_dir.mkdir(exist_ok=True)
    
    best_model_path = output_dir / "best.pt"
    final_model_path = output_dir / "last.pt"
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = eval_model(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch}/{args.epochs}: "
              f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.3f}, "
              f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.3f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved! (Val Acc: {val_acc:.3f})")
    
    # Save final model
    torch.save(model.state_dict(), final_model_path)
    
    # Test evaluation
    test_loss, test_acc = eval_model(
        model, test_loader, criterion, device
    )
    
    print(f"\n{'='*50}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.3f}")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"{'='*50}")
    
    # Save results
    results = {
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_classes": num_classes,
        "class_names": class_names
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Training complete! Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train script type classifier"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="./script_count.csv",
        help="Path to CSV with labels"
    )
    parser.add_argument(
        "--images",
        type=str,
        default="../classification_script_images",
        help="Path to images directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output_efficient",
        help="Output directory for models and results"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Image size for model input"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Test split ratio"
    )
    
    args = parser.parse_args()
    main(args)
