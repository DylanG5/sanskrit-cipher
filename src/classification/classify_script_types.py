import os
import random
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T


# ------------------------------
# Dataset definition
# ------------------------------
class ScriptTypeDataset(Dataset):
    def __init__(self, dataframe, img_dir, img_size=224, augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.img_size = img_size
        self.augment = augment

        self.transform = self._build_transforms()

    def _build_transforms(self):
        base_transforms = []
        if self.augment:
            base_transforms += [
                T.RandomHorizontalFlip(p=0.3),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
                T.RandomRotation(8),
                T.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
            ]
        base_transforms += [
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        return T.Compose(base_transforms)

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, file_name: str) -> str:
        if file_name.lower().endswith(".jpg"):
            return os.path.join(self.img_dir, file_name)
        return os.path.join(self.img_dir, f"{file_name}.jpg")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(row["File Name"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = int(row["Label"])
        return img, label


# ------------------------------
# CNN model definition
# ------------------------------
class ScriptTypeCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 3
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            # Block 4
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(384 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ------------------------------
# Training and evaluation
# ------------------------------
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_one_epoch(model, dataloader, criterion, optimizer, device, use_mixup=False, mixup_alpha=0.2):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
            optimizer.zero_grad()
            outputs = model(images)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            loss.backward()
            optimizer.step()
            preds = outputs.argmax(dim=1)
            correct += (lam * (preds == labels_a).float() + (1 - lam) * (preds == labels_b).float()).sum().item()
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def _load_class_weights(class_counts_csv: Path, class_names: list[str]) -> torch.Tensor:
    if class_counts_csv.exists():
        counts_df = pd.read_csv(class_counts_csv)
        if "Script Type" in counts_df.columns and "Count" in counts_df.columns:
            counts = dict(zip(counts_df["Script Type"], counts_df["Count"]))
            weights = [counts.get(name, 1) for name in class_names]
            weights = np.array(weights, dtype=np.float32)
            weights = weights.sum() / (len(weights) * weights)
            return torch.tensor(weights, dtype=torch.float32)
    return torch.ones(len(class_names), dtype=torch.float32)


def _build_balanced_sampler(labels: pd.Series) -> WeightedRandomSampler:
    class_counts = labels.value_counts().to_dict()
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = labels.map(class_weights).astype(float).to_numpy()
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# ------------------------------
# Main training script
# ------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN to classify script types")
    parser.add_argument("--images_dir", type=str, default="../classification_script_images")
    parser.add_argument("--labels_csv", type=str, default="./script_count.csv")
    parser.add_argument("--class_counts_csv", type=str, default="./script_type_classification.csv")
    parser.add_argument("--out_dir", type=str, default="./output")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--use_mixup", action="store_true")
    parser.add_argument("--mixup_alpha", type=float, default=0.3)
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--balanced_sampler", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=1.5)
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
    if "File Name" not in df.columns or "Script Type" not in df.columns:
        raise ValueError("labels_csv must contain 'File Name' and 'Script Type' columns")

    class_names = sorted(df["Script Type"].unique())
    label_map = {name: idx for idx, name in enumerate(class_names)}
    df["Label"] = df["Script Type"].map(label_map)

    # Stratified split
    strat = df["Label"].astype(int)
    train_val_df, test_df = train_test_split(
        df, test_size=args.test_split, random_state=args.seed, stratify=strat
    )
    val_size_rel = args.val_split / (1 - args.test_split)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_rel,
        random_state=args.seed,
        stratify=train_val_df["Label"].astype(int),
    )

    print(f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    train_ds = ScriptTypeDataset(train_df, args.images_dir, args.img_size, augment=True)
    val_ds = ScriptTypeDataset(val_df, args.images_dir, args.img_size, augment=False)
    test_ds = ScriptTypeDataset(test_df, args.images_dir, args.img_size, augment=False)

    if args.balanced_sampler:
        sampler = _build_balanced_sampler(train_df["Label"])
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    else:
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # -------------------- Model --------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ScriptTypeCNN(num_classes=len(class_names), dropout=args.dropout).to(device)
    class_weights = _load_class_weights(Path(args.class_counts_csv), class_names).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    # -------------------- Training --------------------
    best_val_acc = 0
    patience_counter = 0
    early_stop_patience = 10

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_dl, criterion, optimizer, device, 
            use_mixup=args.use_mixup, mixup_alpha=args.mixup_alpha
        )
        val_loss, val_acc = eval_model(model, val_dl, criterion, device)
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.3f}, "
            f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
            print(f"  -> New best model saved! (Val Acc: {val_acc:.3f})")
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print("Training complete. Best validation accuracy:", best_val_acc)

    # -------------------- Testing --------------------
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "best_model.pt"), map_location=device))
    test_loss, test_acc = eval_model(model, test_dl, criterion, device)
    print(f"Test Loss={test_loss:.4f}, Test Accuracy={test_acc:.3f}")

    # Save final model and metadata
    torch.save(model.state_dict(), os.path.join(args.out_dir, "final_model.pt"))
    meta = {"img_size": args.img_size, "class_names": class_names}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"Saved model and meta info to {args.out_dir}")


# ------------------------------
# Inference helper
# ------------------------------
def predict_image(model_path, image_path, meta_path="./output/meta.json", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    img_size = meta["img_size"]
    class_names = meta["class_names"]
    num_classes = len(class_names)

    model = ScriptTypeCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs.max().item()
        all_probs = {class_names[i]: probs[0][i].item() for i in range(num_classes)}

    return class_names[pred_idx], confidence, all_probs


if __name__ == "__main__":
    main()
