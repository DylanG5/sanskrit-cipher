#!/usr/bin/env python3
"""
Batch Manuscript Fragment Segmentation Script
==============================================

Processes a directory of manuscript images using a YOLO segmentation model
and extracts transparent fragments with alpha channel for edge matching.

Usage:
    python batch_segment.py --input <input_dir> --output <output_dir> [options]

Example:
    python batch_segment.py \
        --input /path/to/BLL238 \
        --output datasets/expanded_fragments \
        --model ../segmentation/best.pt \
        --confidence 0.25
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


class BatchSegmenter:
    """Batch processor for manuscript fragment segmentation."""

    def __init__(
        self,
        model_path: str = "../segmentation/best.pt",
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        """
        Initialize batch segmenter.

        Args:
            model_path: Path to YOLO segmentation model
            confidence: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> YOLO:
        """Load the trained YOLO segmentation model."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"Loading model from {model_path}...")
        model = YOLO(str(model_path))
        print("Model loaded successfully!")
        return model

    def find_images(self, input_dir: Path) -> List[Path]:
        """
        Find all image files in input directory.

        Args:
            input_dir: Directory to search

        Returns:
            List of image file paths
        """
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        if not input_dir.is_dir():
            raise ValueError(f"Input path is not a directory: {input_dir}")

        # Supported image extensions
        extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files = []

        for ext in extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))

        return sorted(image_files)

    def extract_transparent_fragments(
        self,
        image_path: Path,
        results,
        output_dir: Path,
    ) -> Tuple[int, List[str]]:
        """
        Extract individual fragments with transparent backgrounds.

        Args:
            image_path: Path to original image
            results: YOLO prediction results
            output_dir: Directory to save transparent fragment PNGs

        Returns:
            Tuple of (number of fragments extracted, list of output filenames)
        """
        # Read original image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return 0, []

        # Get the first result
        result = results[0]

        # Check if masks exist
        if result.masks is None:
            return 0, []

        # Get masks and boxes
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.data.cpu().numpy()

        # Extract each fragment
        output_files = []
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
            output_path = output_dir / output_filename

            # Save as PNG (supports transparency)
            cv2.imwrite(str(output_path), fragment_cropped)
            output_files.append(output_filename)

        return len(masks), output_files

    def process_batch(
        self,
        input_dir: Path,
        output_dir: Path,
        show_progress: bool = True,
    ) -> dict:
        """
        Process a batch of images and extract fragments.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save extracted fragments
            show_progress: Whether to show progress information

        Returns:
            Dictionary with processing statistics
        """
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find images
        image_files = self.find_images(input_dir)

        if not image_files:
            print(f"No images found in {input_dir}")
            return {
                "total_images": 0,
                "successful": 0,
                "failed": 0,
                "total_fragments": 0,
                "time_elapsed": 0.0,
            }

        print(f"\nFound {len(image_files)} images in {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Confidence threshold: {self.confidence}")
        print(f"IoU threshold: {self.iou_threshold}")
        print("=" * 60)

        # Process each image
        stats = {
            "total_images": len(image_files),
            "successful": 0,
            "failed": 0,
            "total_fragments": 0,
            "images_with_detections": 0,
            "images_without_detections": 0,
            "fragments_per_image": [],
            "failed_images": [],
        }

        start_time = time.time()

        for i, image_path in enumerate(image_files, 1):
            if show_progress:
                print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")

            try:
                # Run inference
                results = self.model.predict(
                    source=str(image_path),
                    conf=self.confidence,
                    iou=self.iou_threshold,
                    verbose=False,
                )

                # Extract transparent fragments
                num_fragments, output_files = self.extract_transparent_fragments(
                    image_path, results, output_dir
                )

                if num_fragments > 0:
                    stats["successful"] += 1
                    stats["images_with_detections"] += 1
                    stats["total_fragments"] += num_fragments
                    stats["fragments_per_image"].append(num_fragments)

                    if show_progress:
                        print(f"  ✓ Extracted {num_fragments} fragment(s)")
                        for filename in output_files:
                            print(f"    - {filename}")
                else:
                    stats["successful"] += 1
                    stats["images_without_detections"] += 1
                    if show_progress:
                        print(f"  ⚠ No fragments detected")

            except Exception as e:
                stats["failed"] += 1
                stats["failed_images"].append(image_path.name)
                if show_progress:
                    print(f"  ✗ Error: {e}")

        elapsed = time.time() - start_time
        stats["time_elapsed"] = elapsed

        # Calculate average
        if stats["fragments_per_image"]:
            stats["avg_fragments_per_image"] = (
                sum(stats["fragments_per_image"]) / len(stats["fragments_per_image"])
            )
        else:
            stats["avg_fragments_per_image"] = 0.0

        return stats

    def print_statistics(self, stats: dict):
        """Print processing statistics."""
        print("\n" + "=" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total images processed: {stats['total_images']}")
        print(f"  Successful: {stats['successful']}")
        print(f"  Failed: {stats['failed']}")
        print()
        print(f"Detection results:")
        print(f"  Images with detections: {stats['images_with_detections']}")
        print(f"  Images without detections: {stats['images_without_detections']}")
        print()
        print(f"Fragment extraction:")
        print(f"  Total fragments extracted: {stats['total_fragments']}")
        if stats['avg_fragments_per_image'] > 0:
            print(
                f"  Average fragments per image: {stats['avg_fragments_per_image']:.2f}"
            )
        print()
        print(f"Performance:")
        print(f"  Time elapsed: {stats['time_elapsed']:.2f}s")
        if stats['total_images'] > 0:
            print(
                f"  Average time per image: {stats['time_elapsed'] / stats['total_images']:.2f}s"
            )

        if stats['failed_images']:
            print()
            print(f"Failed images:")
            for img_name in stats['failed_images']:
                print(f"  - {img_name}")

        print("=" * 60)


def main():
    """Main entry point for batch segmentation."""
    parser = argparse.ArgumentParser(
        description="Batch segmentation of manuscript fragments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single directory
  python batch_segment.py --input /path/to/BLL238 --output datasets/fragments

  # Process with custom model and confidence
  python batch_segment.py --input /path/to/images --output output \\
      --model path/to/model.pt --confidence 0.3

  # Quiet mode (less output)
  python batch_segment.py --input /path/to/images --output output --quiet
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing manuscript images",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for extracted fragments",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="../segmentation/best.pt",
        help="Path to YOLO segmentation model (default: ../segmentation/best.pt)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    # Convert to paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Print header
    print("=" * 60)
    print("BATCH MANUSCRIPT FRAGMENT SEGMENTATION")
    print("=" * 60)

    try:
        # Initialize segmenter
        segmenter = BatchSegmenter(
            model_path=args.model,
            confidence=args.confidence,
            iou_threshold=args.iou,
        )

        # Process batch
        stats = segmenter.process_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            show_progress=not args.quiet,
        )

        # Print statistics
        segmenter.print_statistics(stats)

        # Exit with appropriate code
        if stats["failed"] > 0:
            print(
                f"\n⚠ Warning: {stats['failed']} image(s) failed to process",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            print(f"\n✓ All images processed successfully!")
            sys.exit(0)

    except Exception as e:
        print(f"\n✗ Fatal error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
