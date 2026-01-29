"""
Quick dataset inspection script - visualize a few samples with labels
"""

import cv2
import numpy as np
from pathlib import Path
import random
import os

def visualize_sample_labels(images_dir='images', labels_dir='labels', num_samples=5, output_dir='sample_visualizations'):
    """
    Visualize random samples from the dataset with their label annotations.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO labels
        num_samples: Number of random samples to visualize
        output_dir: Directory to save visualizations
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get all images with labels
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))
    
    valid_images = []
    for img_file in image_files:
        label_file = labels_dir / (img_file.stem + '.txt')
        if label_file.exists():
            valid_images.append(img_file)
    
    if len(valid_images) == 0:
        print("❌ No valid image-label pairs found!")
        return
    
    # Select random samples
    num_samples = min(num_samples, len(valid_images))
    samples = random.sample(valid_images, num_samples)
    
    print(f"Visualizing {num_samples} random samples...")
    print("=" * 70)
    
    for idx, img_path in enumerate(samples, 1):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️  Could not read image: {img_path.name}")
            continue
        
        h, w = img.shape[:2]
        
        # Read labels
        label_file = labels_dir / (img_path.stem + '.txt')
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        num_lines = 0
        for line in lines:
            if not line.strip():
                continue
            
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            class_id, cx, cy, bw, bh = map(float, parts)
            
            # Convert YOLO format to pixel coordinates
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add line number
            num_lines += 1
            label_text = f"L{num_lines}"
            cv2.putText(img, label_text, (x1 + 5, y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add total count
        count_text = f"Total Lines: {num_lines}"
        cv2.putText(img, count_text, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Save
        output_path = output_dir / f"sample_{idx}_{img_path.name}"
        cv2.imwrite(str(output_path), img)
        
        print(f"✅ Sample {idx}: {img_path.name} - {num_lines} lines")
        print(f"   Saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print(f"Visualizations saved to: {output_dir}/")
    print("Check these images to verify your labels are correct!")

if __name__ == '__main__':
    import sys
    
    # Allow command line argument for number of samples
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    
    base_dir = Path(__file__).parent
    
    print("=" * 70)
    print("DATASET SAMPLE VISUALIZATION")
    print("=" * 70)
    print()
    
    visualize_sample_labels(
        images_dir=base_dir / 'images',
        labels_dir=base_dir / 'labels',
        num_samples=num_samples,
        output_dir=base_dir / 'sample_visualizations'
    )
