# train_lines_cnn_pytorch.py
# CNN to predict number of lines in manuscript fragment images using PyTorch.
# Requirements: torch, torchvision, pandas, scikit-learn, pillow

import os
import math
import random
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ------------------------------
# Dataset definition
# ------------------------------
class FragmentDataset(Dataset):
    def __init__(self, dataframe, img_dir, img_size=224, max_lines=15, augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.img_size = img_size
        self.max_lines = max_lines
        self.augment = augment

        self.transform = self._build_transforms()

    def _build_transforms(self):
        base_transforms = []
        if self.augment:
            base_transforms += [
                T.RandomHorizontalFlip(p=0.5),
                # More aggressive color augmentation to handle varying backgrounds and aging
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                T.RandomRotation(10),
                # Random affine to handle different text sizes and orientations
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                # Random perspective for torn/warped fragments
                T.RandomPerspective(distortion_scale=0.2, p=0.3),
                # Simulate lighting variations
                T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
                T.RandomAutocontrast(p=0.3),
            ]
        base_transforms += [
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            # Use ImageNet normalization for better transfer learning potential
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        return T.Compose(base_transforms)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["File Name"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = int(min(row["Line Count"], self.max_lines))
        return img, label


# ------------------------------
# CNN model definition
# ------------------------------
class LineCountCNN(nn.Module):
    def __init__(self, num_classes):
        super(LineCountCNN, self).__init__()
        # Enhanced feature extraction with more depth
        self.features = nn.Sequential(
            # Block 1 - Initial feature extraction
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            # Block 2 - Intermediate features
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            # Block 3 - High-level features
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Block 4 - Deep features for line counting
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Enhanced classifier with residual-like connection
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ------------------------------
# Training and evaluation
# ------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
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


# ------------------------------
# Main training script
# ------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    parser = argparse.ArgumentParser(description="Train CNN to predict line counts in fragment images (PyTorch)")
    parser.add_argument("--images_dir", type=str, default="./classification_training_images")
    parser.add_argument("--labels_csv", type=str, default="./image_line_count.csv")
    parser.add_argument("--out_dir", type=str, default="./output")
    parser.add_argument("--img_size", type=int, default=256)  # Increased for better detail
    parser.add_argument("--batch_size", type=int, default=16)  # Reduced for larger model
    parser.add_argument("--epochs", type=int, default=50)  # More epochs for better convergence
    parser.add_argument("--lr", type=float, default=5e-4)  # Slightly higher initial LR
    parser.add_argument("--max_lines", type=int, default=15)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # -------------------- Data --------------------
    df = pd.read_csv(args.labels_csv)
    assert "File Name" in df.columns and "Line Count" in df.columns

    # Clip line counts to max_lines
    df["Line Count"] = df["Line Count"].clip(0, args.max_lines).astype(int)
    
    # Check class distribution
    class_counts = df["Line Count"].value_counts()
    print(f"Class distribution:\n{class_counts.sort_index()}")
    
    # Filter out classes with only 1 sample for stratified split
    valid_classes = class_counts[class_counts >= 2].index
    df_filtered = df[df["Line Count"].isin(valid_classes)].copy()
    
    if len(df_filtered) < len(df):
        print(f"Warning: Removed {len(df) - len(df_filtered)} samples with classes having only 1 member")
        print(f"Remaining samples: {len(df_filtered)}")
    
    # Stratified split
    strat = df_filtered["Line Count"].astype(int)
    train_val_df, test_df = train_test_split(df_filtered, test_size=args.test_split, random_state=args.seed, stratify=strat)
    val_size_rel = args.val_split / (1 - args.test_split)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size_rel, random_state=args.seed, stratify=train_val_df["Line Count"].astype(int))

    print(f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    train_ds = FragmentDataset(train_df, args.images_dir, args.img_size, args.max_lines, augment=True)
    val_ds = FragmentDataset(val_df, args.images_dir, args.img_size, args.max_lines, augment=False)
    test_ds = FragmentDataset(test_df, args.images_dir, args.img_size, args.max_lines, augment=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # -------------------- Model --------------------
    num_classes = args.max_lines + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = LineCountCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # -------------------- Training --------------------
    best_val_acc = 0
    patience_counter = 0
    early_stop_patience = 15
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc = eval_model(model, val_dl, criterion, device)
        
        # Step the scheduler
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.3f}, "
              f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
            print(f"  -> New best model saved! (Val Acc: {val_acc:.3f})")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print("Training complete. Best validation accuracy:", best_val_acc)

    # -------------------- Testing --------------------
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "best_model.pt")))
    test_loss, test_acc = eval_model(model, test_dl, criterion, device)
    print(f"Test Loss={test_loss:.4f}, Test Accuracy={test_acc:.3f}")

    # Save final model and metadata
    torch.save(model.state_dict(), os.path.join(args.out_dir, "final_model.pt"))
    meta = {"max_lines": args.max_lines, "img_size": args.img_size}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"Saved model and meta info to {args.out_dir}")


# ------------------------------
# Inference helper
# ------------------------------

def predict_image(model_path, image_path, meta_path="./output/meta.json", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load metadata
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    max_lines = meta['max_lines']
    img_size = meta['img_size']
    num_classes = max_lines + 1
    
    # Load model
    model = LineCountCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_t = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    # Predict
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs.max().item()
        
        # Get all probabilities as a dictionary
        all_probs = {i: probs[0][i].item() for i in range(num_classes)}
    
    return pred_class, confidence, all_probs


if __name__ == "__main__":
    main()
