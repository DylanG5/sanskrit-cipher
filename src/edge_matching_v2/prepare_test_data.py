"""
Prepare test data by running segmentation on sample images
and copying transparent fragments to test directory.
"""

import sys
import shutil
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'segmentation'))

from test_model import load_model, extract_transparent_fragments

def main():
    print("=" * 60)
    print("PREPARING TEST DATA FOR EDGE MATCHING")
    print("=" * 60)

    # Setup paths
    segmentation_dir = Path(__file__).parent.parent / 'segmentation'
    test_images_dir = segmentation_dir / 'test_images'
    model_path = segmentation_dir / 'best.pt'

    output_dir = Path(__file__).parent / 'datasets' / 'test_fragments'
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    if not test_images_dir.exists():
        print(f"Error: Test images directory not found at {test_images_dir}")
        return

    # Load segmentation model
    print(f"\nLoading segmentation model from {model_path}...")
    model = load_model(str(model_path))

    # Get test images (limit to 30 for testing)
    test_images = sorted(test_images_dir.glob('*.jpg'))[:30]
    print(f"Found {len(test_images)} test images")

    if not test_images:
        print("No test images found!")
        return

    print(f"\nProcessing images and extracting transparent fragments...")
    print("-" * 60)

    total_fragments = 0

    for i, img_path in enumerate(test_images, 1):
        print(f"[{i}/{len(test_images)}] Processing {img_path.name}")

        try:
            # Run segmentation
            results = model.predict(
                source=str(img_path),
                conf=0.25,
                iou=0.45,
                verbose=False
            )

            # Extract transparent fragments
            num_extracted = extract_transparent_fragments(
                img_path,
                results,
                output_dir
            )

            total_fragments += num_extracted
            print(f"  → Extracted {num_extracted} fragment(s)")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "=" * 60)
    print(f"EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Processed: {len(test_images)} images")
    print(f"Extracted: {total_fragments} fragments")
    print(f"Output: {output_dir}")
    print("\nNext steps:")
    print("1. Extract edge descriptors:")
    print(f"   ./venv/bin/python enhanced_edge_matching.py extract \\")
    print(f"     --input datasets/test_fragments \\")
    print(f"     --output output/edge_descriptors.json")
    print("\n2. Find matches for a fragment:")
    print(f"   ./venv/bin/python enhanced_edge_matching.py match \\")
    print(f"     --descriptors output/edge_descriptors.json \\")
    print(f"     --fragment <FRAGMENT_ID>")
    print("\n3. Run comprehensive evaluation:")
    print(f"   ./venv/bin/python evaluate_matching.py \\")
    print(f"     --descriptors output/edge_descriptors.json \\")
    print(f"     --fragments-dir datasets/test_fragments \\")
    print(f"     --num-test 10 --visualize")


if __name__ == '__main__':
    main()
