import os
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml
from sklearn.model_selection import train_test_split
import shutil

def prepare_data_split(images_dir, labels_dir, val_split=0.2, test_split=0.1):

    # Get all image files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    # Check corresponding labels exist
    valid_images = []
    for img_file in image_files:
        label_name = os.path.splitext(img_file)[0] + '.txt'
        if os.path.exists(os.path.join(labels_dir, label_name)):
            valid_images.append(img_file)
    
    print(f"Total images with labels: {len(valid_images)}")
    
    # Split data
    train_imgs, temp_imgs = train_test_split(valid_images, test_size=(val_split + test_split), random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=test_split/(val_split + test_split), random_state=42)
    
    print(f"Train: {len(train_imgs)}, Validation: {len(val_imgs)}, Test: {len(test_imgs)}")
    
    # Create split directories
    base_dir = Path(images_dir).parent
    train_img_dir = base_dir / 'images_train'
    val_img_dir = base_dir / 'images_val'
    test_img_dir = base_dir / 'images_test'
    train_lbl_dir = base_dir / 'labels_train'
    val_lbl_dir = base_dir / 'labels_val'
    test_lbl_dir = base_dir / 'labels_test'
    
    # Create directories
    for dir_path in [train_img_dir, val_img_dir, test_img_dir, train_lbl_dir, val_lbl_dir, test_lbl_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Copy files to respective directories
    def copy_files(file_list, img_dest, lbl_dest):
        for img_file in file_list:
            # Copy image
            src_img = os.path.join(images_dir, img_file)
            dst_img = img_dest / img_file
            shutil.copy2(src_img, dst_img)
            
            # Copy label
            label_name = os.path.splitext(img_file)[0] + '.txt'
            src_lbl = os.path.join(labels_dir, label_name)
            dst_lbl = lbl_dest / label_name
            shutil.copy2(src_lbl, dst_lbl)
    
    print("Copying files...")
    copy_files(train_imgs, train_img_dir, train_lbl_dir)
    copy_files(val_imgs, val_img_dir, val_lbl_dir)
    copy_files(test_imgs, test_img_dir, test_lbl_dir)
    
    print("Data split completed!")
    return train_img_dir, val_img_dir, test_img_dir

def update_yaml_config(base_dir, train_dir='images', val_dir='images'):
    """Update data.yaml to use images and labels folders directly"""
    yaml_path = base_dir / 'data.yaml'
    
    config = {
        'path': str(base_dir),
        'train': train_dir,
        'val': val_dir,
        'names': {0: 'line'},
        'nc': 1
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Updated {yaml_path}")

def train_yolo(
    model_size='s',  # Options: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra-large)
    epochs=100,
    img_size=640,
    batch_size=8,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train YOLOv8 model for line detection.
    
    Args:
        model_size: Model size ('n', 's', 'm', 'l', 'x'). 'n' is fastest, 'x' is most accurate
        epochs: Number of training epochs
        img_size: Input image size (will be resized)
        batch_size: Training batch size
        device: 'cuda' for GPU or 'cpu'
    """
    print(f"Using device: {device}")
    print(f"Training YOLOv8{model_size} model...\n")
    
    # Initialize YOLO model
    model = YOLO(f'yolov8{model_size}.pt')  # Load pretrained model
    
    # Get data config path
    data_config = Path(__file__).parent / 'data.yaml'
    
    # Train the model with simplified output
    results = model.train(
        data=str(data_config),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=str(Path(__file__).parent / 'runs' / 'train'),
        name='sanskrit_line_detection',
        patience=30, 
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        verbose=False, 
        
        # Data augmentation - reduced for better learning
        hsv_h=0.01,  # HSV-Hue augmentation
        hsv_s=0.5,    # HSV-Saturation augmentation
        hsv_v=0.3,    # HSV-Value augmentation
        degrees=3,    # Rotation (+/- degrees)
        translate=0.05,  # Translation (+/- fraction)
        scale=0.3,    # Scaling (+/- gain)
        shear=1,      # Shear (+/- degrees)
        flipud=0.0,   # Flip up-down probability (disabled for text)
        fliplr=0.3,   # Flip left-right probability
        mosaic=0.8,   # Mosaic augmentation probability
        
        # Optimization
        optimizer='AdamW',
        lr0=0.01,    # Initial learning rate (increased)
        lrf=0.01,     # Final learning rate (as fraction of lr0)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,  # Warmup epochs
        warmup_momentum=0.8,
        
        # Loss weights
        box=7.5,      # Box loss gain
        cls=0.5,      # Class loss gain
        dfl=1.5,      # DFL loss gain
        
        # Validation
        val=True,
        plots=True,   # Create training plots
    )
    
    print("\nTraining completed!")
    print(f"Best model saved at: {model.trainer.best}")
    print(f"Results saved at: {model.trainer.save_dir}")
    
    return model, results

def main():
    """Main training pipeline"""
    # Get current directory
    base_dir = Path(__file__).parent
    images_dir = base_dir / 'images'
    labels_dir = base_dir / 'labels'
    
    print("=" * 60)
    print("YOLOv8 Sanskrit Line Detection Training")
    print("=" * 60)
    
    # Check and display device information
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🔧 Device Configuration:")
    print(f"   Using: {device.upper()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print(f"   Note: Training on CPU will be slower. Consider using a GPU for faster training.")
    print()
    
    # Configure to use images and labels folders directly
    print("\nConfiguring to use 'images' and 'labels' folders directly...")
    print(f"Images: {len(list(images_dir.glob('*.jpg')))} files")
    print(f"Labels: {len(list(labels_dir.glob('*.txt')))} files")
    update_yaml_config(base_dir, train_dir='images', val_dir='images')
    
    print("\n" + "=" * 60)
    print("Starting Model Training")
    print("=" * 60)
    
    # Train model
    model, results = train_yolo(
        model_size='s',    
        epochs=150,        
        img_size=640,      # Standard YOLO image size
        batch_size=8,     
    )
    
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Check the results in: runs/train/sanskrit_line_detection")
    print(f"Best weights: runs/train/sanskrit_line_detection/weights/best.pt")
    print("\nKey metrics to check:")
    print("  - mAP50: Mean Average Precision at IoU=0.5")
    print("  - mAP50-95: Mean Average Precision at IoU=0.5:0.95")
    print("  - Precision: Correctness of detections")
    print("  - Recall: Completeness of detections")

if __name__ == '__main__':
    main()
