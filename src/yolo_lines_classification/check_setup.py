"""
Setup Verification Script

Run this script to verify your environment is ready for YOLO training.
"""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    required = [
        ('torch', 'PyTorch'),
        ('ultralytics', 'YOLOv8'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'tqdm'),
    ]
    
    all_ok = True
    for module, name in required:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Not installed")
            all_ok = False
    
    return all_ok

def check_cuda():
    """Check if CUDA (GPU) is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available - GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA not available - Will use CPU (slower)")
            return False
    except:
        print("⚠️  Cannot check CUDA")
        return False

def check_data():
    """Check if dataset exists and is valid"""
    base_dir = Path(__file__).parent
    images_dir = base_dir / 'images'
    labels_dir = base_dir / 'labels'
    
    if not images_dir.exists():
        print("❌ images/ directory not found")
        return False
    
    if not labels_dir.exists():
        print("❌ labels/ directory not found")
        return False
    
    # Count files
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))
    label_files = list(labels_dir.glob('*.txt'))
    
    # Find matching pairs
    valid_pairs = 0
    for img_file in image_files:
        label_file = labels_dir / (img_file.stem + '.txt')
        if label_file.exists():
            valid_pairs += 1
    
    print(f"✅ Dataset found:")
    print(f"   Images: {len(image_files)}")
    print(f"   Labels: {len(label_files)}")
    print(f"   Valid pairs: {valid_pairs}")
    
    if valid_pairs < 50:
        print("⚠️  Warning: Less than 50 image-label pairs (need more data for good training)")
        return False
    
    return valid_pairs > 0

def check_disk_space():
    """Check available disk space"""
    import shutil
    base_dir = Path(__file__).parent
    total, used, free = shutil.disk_usage(base_dir)
    
    free_gb = free / (1024**3)
    if free_gb > 5:
        print(f"✅ Disk space: {free_gb:.1f} GB available")
        return True
    else:
        print(f"⚠️  Low disk space: {free_gb:.1f} GB available (need at least 5 GB)")
        return False

def main():
    """Run all checks"""
    print("=" * 70)
    print("YOLO LINE DETECTION - SETUP VERIFICATION")
    print("=" * 70)
    print()
    
    checks = []
    
    print("📋 Checking Python Version...")
    checks.append(check_python_version())
    print()
    
    print("📦 Checking Dependencies...")
    checks.append(check_dependencies())
    print()
    
    print("🎮 Checking GPU/CUDA...")
    check_cuda()  # Not critical, so don't add to checks
    print()
    
    print("📊 Checking Dataset...")
    checks.append(check_data())
    print()
    
    print("💾 Checking Disk Space...")
    checks.append(check_disk_space())
    print()
    
    # Summary
    print("=" * 70)
    if all(checks):
        print("✅ ALL CHECKS PASSED - Ready to train!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Run: python utils.py          (view dataset statistics)")
        print("  2. Run: python visualize_samples.py  (verify labels)")
        print("  3. Run: python train.py          (start training)")
        print()
    else:
        print("❌ SOME CHECKS FAILED")
        print("=" * 70)
        print()
        print("Fix the issues above, then:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Verify dataset exists in images/ and labels/ directories")
        print("  - Ensure image-label pairs match (same filename)")
        print()

if __name__ == '__main__':
    main()
