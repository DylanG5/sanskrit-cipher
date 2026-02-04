"""
Script Type Classification - Similar structure to yolo_lines_classification
Uses YOLO-based approach with efficient training pipeline
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml

def prepare_data_split(csv_path, images_dir, output_base_dir="./script_type_split", val_split=0.15, test_split=0.1):
    """
    Prepare train/val/test split from CSV labels
    Creates directory structure compatible with YOLO format
    """
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Total images: {len(df)}")
    
    # Create directories
    Path(output_base_dir).mkdir(exist_ok=True)
    
    train_img_dir = Path(output_base_dir) / 'images_train'
    val_img_dir = Path(output_base_dir) / 'images_val'
    test_img_dir = Path(output_base_dir) / 'images_test'
    
    for dir_path in [train_img_dir, val_img_dir, test_img_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Create splits with stratification
    class_names = sorted(df["Script Type"].unique())
    df['label_idx'] = df["Script Type"].map({name: idx for idx, name in enumerate(class_names)})
    
    train_df, temp_df = train_test_split(
        df, test_size=(val_split + test_split), random_state=42, stratify=df['label_idx']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=test_split/(val_split + test_split), random_state=42, stratify=temp_df['label_idx']
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Copy images to respective folders and build labeled CSVs
    def copy_split(split_df, img_dest_dir, split_name):
        missing = []
        split_rows = []
        for _, row in split_df.iterrows():
            file_name = row["File Name"]
            
            # Find image file
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                candidate = Path(images_dir) / f"{file_name}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            
            if not img_path:
                # Search recursively
                for root, dirs, files in os.walk(images_dir):
                    for f in files:
                        if file_name in f:
                            img_path = Path(root) / f
                            break
            
            if img_path and img_path.exists():
                dst = img_dest_dir / img_path.name
                shutil.copy2(img_path, dst)
                split_rows.append({
                    "File Name": file_name,
                    "Image File": img_path.name,
                    "Script Type": row["Script Type"],
                    "Label": int(row["label_idx"]),
                })
            else:
                missing.append(file_name)
        
        if missing:
            print(f"Warning: {len(missing)} images not found in {split_name}")
        else:
            print(f"Copied {len(split_df) - len(missing)} images to {split_name}")

        split_out_df = pd.DataFrame(split_rows)
        split_csv_path = Path(output_base_dir) / f"{split_name}.csv"
        split_out_df.to_csv(split_csv_path, index=False)
        return split_out_df, split_csv_path
    
    train_out_df, train_csv_path = copy_split(train_df, train_img_dir, "train")
    val_out_df, val_csv_path = copy_split(val_df, val_img_dir, "val")
    test_out_df, test_csv_path = copy_split(test_df, test_img_dir, "test")
    
    # Save metadata
    meta = {
        "class_names": class_names,
        "num_classes": len(class_names),
        "train_size": len(train_out_df),
        "val_size": len(val_out_df),
        "test_size": len(test_out_df),
        "splits": {
            "train": train_out_df[["File Name", "Image File", "Script Type", "Label"]].to_dict(orient='records'),
            "val": val_out_df[["File Name", "Image File", "Script Type", "Label"]].to_dict(orient='records'),
            "test": test_out_df[["File Name", "Image File", "Script Type", "Label"]].to_dict(orient='records'),
        }
    }
    
    # Save as YAML for compatibility
    config = {
        "path": str(Path(output_base_dir).absolute()),
        "train": str(train_img_dir),
        "val": str(val_img_dir),
        "test": str(test_img_dir),
        "train_csv": str(train_csv_path),
        "val_csv": str(val_csv_path),
        "test_csv": str(test_csv_path),
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)}
    }
    
    with open(Path(output_base_dir) / "data.yaml", "w") as f:
        yaml.dump(config, f)
    
    with open(Path(output_base_dir) / "metadata.yaml", "w") as f:
        yaml.dump(meta, f)
    
    print(f"Data split complete! Configuration saved to {output_base_dir}/data.yaml")
    return config, meta


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare script type classification data split")
    parser.add_argument("--csv", type=str, default="./script_count.csv", help="Path to CSV with labels")
    parser.add_argument("--images", type=str, default="../classification_script_images", help="Path to images directory")
    parser.add_argument("--output", type=str, default="./script_type_split", help="Output directory for splits")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test split ratio")
    
    args = parser.parse_args()
    
    prepare_data_split(
        args.csv, 
        args.images, 
        args.output, 
        args.val_split, 
        args.test_split
    )
