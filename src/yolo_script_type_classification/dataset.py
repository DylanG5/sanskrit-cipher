"""
Dataset module for script type classification
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class ScriptTypeDataset(Dataset):
    """Dataset for script type classification images"""
    
    def __init__(self, images_dir, img_size=224, augment=False):
        self.images_dir = Path(images_dir)
        self.img_size = img_size
        self.augment = augment
        
        # Get all image files
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        self.transform = self._build_transforms()
    
    def _build_transforms(self):
        base_transforms = []
        
        if self.augment:
            base_transforms += [
                T.RandomHorizontalFlip(p=0.3),
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.15
                ),
                T.RandomRotation(8),
                T.RandomAffine(
                    degrees=0,
                    translate=(0.08, 0.08),
                    scale=(0.95, 1.05)
                ),
            ]
        
        base_transforms += [
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]
        
        return T.Compose(base_transforms)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = self.images_dir / img_file
        
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        # Return image and filename (for inference)
        return image, img_file


class LabeledScriptTypeDataset(Dataset):
    """Dataset with labels loaded from a split CSV"""

    def __init__(self, csv_path, images_dir, img_size=224, augment=False):
        import pandas as pd

        self.images_dir = Path(images_dir)
        self.img_size = img_size
        self.augment = augment
        self.transform = self._build_transforms()

        df = pd.read_csv(csv_path)
        if "Label" not in df.columns:
            raise ValueError("Split CSV must contain a 'Label' column")

        image_col = "Image File" if "Image File" in df.columns else "File Name"
        self.samples = [
            (row[image_col], int(row["Label"]))
            for _, row in df.iterrows()
        ]

    def _build_transforms(self):
        base_transforms = []

        if self.augment:
            base_transforms += [
                T.RandomHorizontalFlip(p=0.3),
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.15
                ),
                T.RandomRotation(8),
                T.RandomAffine(
                    degrees=0,
                    translate=(0.08, 0.08),
                    scale=(0.95, 1.05)
                ),
            ]

        base_transforms += [
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]

        return T.Compose(base_transforms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_file, label = self.samples[idx]
        img_path = self.images_dir / img_file

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label
