#!/usr/bin/env python3
"""
Manuscript Fragment Segmentation Model Testing Script

This script loads a trained YOLO segmentation model and runs inference
on test images, visualizing the results with segmentation masks overlaid.
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np


def setup_directories():
    """Create output directories if they don't exist."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Directory for transparent fragments (for web UI)
    transparent_dir = Path("outputs/transparent_fragments")
    transparent_dir.mkdir(exist_ok=True, parents=True)

    return output_dir, transparent_dir


def load_model(model_path="best.pt"):
    """Load the trained YOLO segmentation model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    print("Model loaded successfully!")
    return model


def get_test_images(test_dir="test_images"):
    """Get list of test images."""
    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"Test images directory not found: {test_dir}")

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    for ext in image_extensions:
        images.extend(list(test_path.glob(f"*{ext}")))
        images.extend(list(test_path.glob(f"*{ext.upper()}")))

    return sorted(images)


def extract_transparent_fragments(image_path, results, transparent_dir):
    """
    Extract individual fragments with transparent backgrounds for web UI.

    Args:
        image_path: Path to original image
        results: YOLO prediction results
        transparent_dir: Directory to save transparent fragment PNGs

    Returns:
        Number of fragments extracted
    """
    # Read original image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return 0

    # Get the first result
    result = results[0]

    # Check if masks exist
    if result.masks is None:
        return 0

    # Get masks and boxes
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.data.cpu().numpy()

    # Extract each fragment
    for idx, (mask, box) in enumerate(zip(masks, boxes)):
        # Resize mask to image size
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))

        # Threshold the mask
        mask_binary = (mask_resized > 0.5).astype(np.uint8)

        # Create RGBA image (add alpha channel)
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        # Apply mask to alpha channel (make background transparent)
        img_rgba[:, :, 3] = mask_binary * 255

        # Get bounding box coordinates
        x1, y1, x2, y2 = box[:4].astype(int)

        # Add padding around the bounding box
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img.shape[1], x2 + padding)
        y2 = min(img.shape[0], y2 + padding)

        # Crop to bounding box
        fragment_cropped = img_rgba[y1:y2, x1:x2]

        # Generate output filename
        base_name = image_path.stem  # filename without extension
        conf = box[4]
        output_filename = f"{base_name}_fragment_{idx+1}_conf{conf:.2f}.png"
        output_path = transparent_dir / output_filename

        # Save as PNG (supports transparency)
        cv2.imwrite(str(output_path), fragment_cropped)

    return len(masks)


def visualize_results(image_path, results, output_dir):
    """
    Visualize segmentation results by overlaying masks on the original image.

    Args:
        image_path: Path to original image
        results: YOLO prediction results
        output_dir: Directory to save output images
    """
    # Read original image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return

    # Create a copy for visualization
    vis_img = img.copy()

    # Get the first result (since we're processing one image at a time)
    result = results[0]

    # Check if masks exist
    if result.masks is not None:
        # Get masks and boxes
        masks = result.masks.data.cpu().numpy()  # Get mask data as numpy array
        boxes = result.boxes.data.cpu().numpy()  # Get box data

        # Create colors for each mask (different color for each fragment)
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

        print(f"  Found {len(masks)} fragment(s)")

        # Overlay each mask
        for idx, (mask, box) in enumerate(zip(masks, boxes)):
            # Resize mask to image size
            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))

            # Threshold the mask
            mask_binary = (mask_resized > 0.5).astype(np.uint8)

            # Get color for this mask
            color = colors[idx % len(colors)]

            # Create colored mask
            colored_mask = np.zeros_like(img)
            colored_mask[mask_binary == 1] = color

            # Blend with original image
            alpha = 0.4  # Transparency factor
            vis_img = cv2.addWeighted(vis_img, 1, colored_mask, alpha, 0)

            # Draw bounding box
            x1, y1, x2, y2 = box[:4].astype(int)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

            # Add confidence score
            conf = box[4]
            label = f"Fragment {idx+1}: {conf:.2f}"
            cv2.putText(vis_img, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        print(f"  No fragments detected")

    # Save the visualization
    output_path = output_dir / f"result_{image_path.name}"
    cv2.imwrite(str(output_path), vis_img)
    print(f"  Saved result to {output_path}")


def print_statistics(all_results, total_extracted):
    """Print overall statistics from all predictions."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    total_fragments = 0
    images_with_detections = 0

    for img_path, results in all_results:
        result = results[0]
        if result.masks is not None:
            num_fragments = len(result.masks)
            total_fragments += num_fragments
            if num_fragments > 0:
                images_with_detections += 1

    print(f"Total images processed: {len(all_results)}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total fragments detected: {total_fragments}")
    print(f"Transparent fragments extracted: {total_extracted}")
    if len(all_results) > 0:
        print(f"Average fragments per image: {total_fragments / len(all_results):.2f}")
    print("="*60)


def main():
    """Main testing workflow."""
    print("="*60)
    print("Manuscript Fragment Segmentation Model Testing")
    print("="*60)

    try:
        # Setup
        output_dir, transparent_dir = setup_directories()
        print(f"\nOutput directory: {output_dir.absolute()}")
        print(f"Transparent fragments directory: {transparent_dir.absolute()}")

        # Load model
        model = load_model()

        # Get test images
        test_images = get_test_images()
        print(f"\nFound {len(test_images)} test images")

        if len(test_images) == 0:
            print("No test images found. Please add images to the 'test_images' directory.")
            return

        # Run inference on each image
        print("\nRunning inference...")
        print("-" * 60)

        all_results = []
        total_extracted = 0

        for img_path in test_images:
            print(f"\nProcessing: {img_path.name}")

            # Run inference
            results = model.predict(
                source=str(img_path),
                conf=0.25,  # Confidence threshold
                iou=0.45,   # NMS IoU threshold
                verbose=False
            )

            # Visualize and save results
            visualize_results(img_path, results, output_dir)

            # Extract transparent fragments for web UI
            num_extracted = extract_transparent_fragments(img_path, results, transparent_dir)
            total_extracted += num_extracted

            if num_extracted > 0:
                print(f"  Extracted {num_extracted} transparent fragment(s)")

            all_results.append((img_path, results))

        # Print statistics
        print_statistics(all_results, total_extracted)

        print(f"\n✅ Testing complete!")
        print(f"  - Visualizations: '{output_dir}' directory")
        print(f"  - Transparent fragments: '{transparent_dir}' directory")

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
