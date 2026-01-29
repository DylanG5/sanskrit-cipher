"""
Utility functions for YOLO line detection project
"""

import os
from pathlib import Path
import yaml
import json

def count_images_and_labels(images_dir, labels_dir):
    """
    Count and validate images and their corresponding labels.
    
    Returns:
        dict with counts and validation info
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))
    label_files = list(labels_dir.glob('*.txt'))
    
    # Find images with labels
    images_with_labels = []
    images_without_labels = []
    
    for img_file in image_files:
        label_name = img_file.stem + '.txt'
        if (labels_dir / label_name).exists():
            images_with_labels.append(img_file.name)
        else:
            images_without_labels.append(img_file.name)
    
    # Find orphaned labels (labels without images)
    orphaned_labels = []
    for lbl_file in label_files:
        img_candidates = [
            images_dir / (lbl_file.stem + '.jpg'),
            images_dir / (lbl_file.stem + '.jpeg'),
            images_dir / (lbl_file.stem + '.png'),
        ]
        if not any(img.exists() for img in img_candidates):
            orphaned_labels.append(lbl_file.name)
    
    return {
        'total_images': len(image_files),
        'total_labels': len(label_files),
        'images_with_labels': len(images_with_labels),
        'images_without_labels': images_without_labels,
        'orphaned_labels': orphaned_labels
    }

def analyze_dataset_statistics(labels_dir):
    """
    Analyze statistics of the labeled dataset.
    
    Returns:
        dict with dataset statistics
    """
    labels_dir = Path(labels_dir)
    label_files = list(labels_dir.glob('*.txt'))
    
    if not label_files:
        return {'error': 'No label files found'}
    
    line_counts = []
    total_lines = 0
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            count = len([l for l in lines if l.strip()])
            line_counts.append(count)
            total_lines += count
    
    from collections import Counter
    import numpy as np
    
    distribution = Counter(line_counts)
    
    return {
        'total_images': len(label_files),
        'total_lines': total_lines,
        'avg_lines_per_image': np.mean(line_counts),
        'median_lines_per_image': np.median(line_counts),
        'min_lines': min(line_counts),
        'max_lines': max(line_counts),
        'std_lines': np.std(line_counts),
        'distribution': dict(sorted(distribution.items()))
    }

def print_dataset_info():
    """Print comprehensive dataset information"""
    base_dir = Path(__file__).parent
    images_dir = base_dir / 'images'
    labels_dir = base_dir / 'labels'
    
    print("=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)
    
    # Validate files
    validation = count_images_and_labels(images_dir, labels_dir)
    
    print(f"\n📁 File Counts:")
    print(f"  Total images: {validation['total_images']}")
    print(f"  Total labels: {validation['total_labels']}")
    print(f"  Images with labels: {validation['images_with_labels']}")
    
    if validation['images_without_labels']:
        print(f"\n⚠️  Warning: {len(validation['images_without_labels'])} images without labels")
        if len(validation['images_without_labels']) <= 5:
            for img in validation['images_without_labels']:
                print(f"    - {img}")
    
    if validation['orphaned_labels']:
        print(f"\n⚠️  Warning: {len(validation['orphaned_labels'])} labels without images")
    
    # Analyze statistics
    if validation['images_with_labels'] > 0:
        stats = analyze_dataset_statistics(labels_dir)
        
        print(f"\n📊 Line Detection Statistics:")
        print(f"  Total lines annotated: {stats['total_lines']}")
        print(f"  Average lines per image: {stats['avg_lines_per_image']:.2f}")
        print(f"  Median lines per image: {stats['median_lines_per_image']:.0f}")
        print(f"  Range: {stats['min_lines']} - {stats['max_lines']} lines")
        print(f"  Standard deviation: {stats['std_lines']:.2f}")
        
        print(f"\n📈 Line Count Distribution:")
        for count, freq in stats['distribution'].items():
            bar = '█' * int(freq / max(stats['distribution'].values()) * 40)
            print(f"  {count:2d} lines: {bar} ({freq} images)")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    print_dataset_info()
