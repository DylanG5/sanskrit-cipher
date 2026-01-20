#!/usr/bin/env python3
"""
Enhanced scale detection using EasyOCR instead of pytesseract.
EasyOCR provides more consistent and reliable text detection.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import easyocr
    OCR_AVAILABLE = True
    # Initialize reader once (singleton pattern for efficiency)
    _OCR_READER = None
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: easyocr not available. Will use fallback method.")


def get_ocr_reader():
    """Get or create EasyOCR reader (singleton pattern)."""
    global _OCR_READER
    if _OCR_READER is None and OCR_AVAILABLE:
        # Initialize with English only for speed
        _OCR_READER = easyocr.Reader(['en'], gpu=False)
    return _OCR_READER


def find_scale_text_region(image: np.ndarray, bottom_fraction: float = 0.15) -> Tuple[Optional[str], int]:
    """
    Find the region containing 'cm' or 'mm' text using EasyOCR.

    Args:
        image: Input image (BGR or grayscale)
        bottom_fraction: Fraction of image height to search (from bottom)

    Returns:
        Tuple of (unit_type, text_x_end) where text_x_end is the right edge of the text
    """
    h, w = image.shape[:2]

    # Try EasyOCR first if available
    if OCR_AVAILABLE:
        reader = get_ocr_reader()
        if reader is not None:
            # Extract bottom region for OCR
            bottom_region = image[int(h * (1 - bottom_fraction)):, :]

            try:
                # EasyOCR returns: [([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], text, confidence), ...]
                results = reader.readtext(bottom_region)

                for bbox, text, confidence in results:
                    text_lower = text.lower().strip()
                    # Check for cm or mm (be flexible with spacing)
                    if 'cm' in text_lower or 'mm' in text_lower:
                        # Extract bounding box coordinates
                        x_coords = [point[0] for point in bbox]
                        x_max = max(x_coords)
                        text_x_end = int(x_max) + 10  # Add small margin

                        # Determine unit type
                        if 'mm' in text_lower:
                            return 'mm', text_x_end
                        else:
                            return 'cm', text_x_end

            except Exception as e:
                print(f"  EasyOCR error: {e}")

    # Fallback: estimate text region from bottom-left corner
    # The text is typically in the left 15% of the image width
    bottom_region = image[int(h * (1 - bottom_fraction)):, :int(w * 0.15)]

    if len(bottom_region.shape) == 3:
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = bottom_region

    # Look for white text on black background
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find where text might be (columns with white pixels)
    col_sums = np.sum(thresh, axis=0)
    white_cols = np.where(col_sums > 100)[0]

    if len(white_cols) > 0:
        # Text ends at the rightmost white pixel, plus margin
        text_x_end = white_cols[-1] + 20
        return 'cm', text_x_end  # Default to cm

    # If no text found, assume text region is first 10% of width
    return 'cm', int(w * 0.10)


def detect_ruler_ticks_fixed_region(image: np.ndarray, text_x_end: int = 0) -> List[int]:
    """
    Simple approach: look at the very bottom of the image for ticks.

    Args:
        image: Input image
        text_x_end: X-coordinate where text region ends (to exclude from detection)

    Returns:
        List of x-coordinates of detected ticks
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    h, w = image.shape[:2]

    # Fixed region: bottom 60 pixels (typical ruler bar height)
    ruler_region = image[max(0, h-60):, :]

    if len(ruler_region.shape) == 3:
        gray = cv2.cvtColor(ruler_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = ruler_region

    # Get intensity profile from middle of ruler
    mid_row = len(gray) // 2
    intensity_profile = np.mean(gray[max(0, mid_row-3):min(len(gray), mid_row+4), :], axis=0)

    smoothed = gaussian_filter1d(intensity_profile, sigma=1.5)

    # Simple peak detection
    height = np.mean(smoothed) + 0.4 * np.std(smoothed)
    peaks, _ = find_peaks(smoothed, height=height, distance=10, prominence=3)

    # Filter out peaks that are in the text region (left of text_x_end)
    filtered_peaks = [p for p in peaks if p > text_x_end]

    return filtered_peaks


def detect_ruler_ticks_adaptive(image: np.ndarray, text_x_end: int = 0, search_height_fraction: float = 0.15) -> List[int]:
    """
    Detect vertical tick marks in the bottom ruler area adaptively.

    Args:
        image: Input image
        text_x_end: X-coordinate where text region ends
        search_height_fraction: Fraction of image height to search

    Returns:
        List of x-coordinates of detected ticks
    """
    h, w = image.shape[:2]

    # Focus on bottom portion where ruler typically is
    bottom_region = image[int(h * (1 - search_height_fraction)):, :]

    # Convert to grayscale if needed
    if len(bottom_region.shape) == 3:
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = bottom_region

    # Find the black bar region (ruler background) - it should be very dark
    row_means = np.mean(gray, axis=1)
    dark_threshold = np.percentile(row_means, 20)  # Darkest 20% of rows

    # Find continuous dark region (the black bar)
    dark_rows = np.where(row_means < dark_threshold)[0]

    if len(dark_rows) < 3:
        # No clear black bar found, use entire bottom region
        ruler_region = gray
    else:
        # Use the dark region
        start_row = max(0, dark_rows[0] - 2)
        end_row = min(len(gray), dark_rows[-1] + 3)
        ruler_region = gray[start_row:end_row, :]

    # Get intensity profile across the middle of the ruler bar
    mid_row = len(ruler_region) // 2
    row_start = max(0, mid_row - 3)
    row_end = min(len(ruler_region), mid_row + 4)
    intensity_profile = np.mean(ruler_region[row_start:row_end, :], axis=0)

    # Find peaks in intensity (white tick marks on black background)
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    smoothed = gaussian_filter1d(intensity_profile, sigma=1.5)

    # Calculate adaptive thresholds
    mean_intensity = np.mean(smoothed)
    std_intensity = np.std(smoothed)

    # Try multiple parameter sets to be more robust
    all_peaks = []

    # Parameter set 1: Standard detection
    height1 = mean_intensity + 0.4 * std_intensity
    peaks1, _ = find_peaks(smoothed, height=height1, distance=10, prominence=3)
    all_peaks.append(peaks1)

    # Parameter set 2: More sensitive
    height2 = mean_intensity + 0.25 * std_intensity
    peaks2, _ = find_peaks(smoothed, height=height2, distance=8, prominence=2)
    all_peaks.append(peaks2)

    # Parameter set 3: Very sensitive for faint ticks
    height3 = np.percentile(smoothed, 70)
    peaks3, _ = find_peaks(smoothed, height=height3, distance=6, prominence=1)
    all_peaks.append(peaks3)

    # Select the best result: prefer one with 3+ peaks
    best_peaks = []
    for peaks in all_peaks:
        if len(peaks) >= 3:
            best_peaks = peaks
            break

    if len(best_peaks) == 0:
        # Take the one with most peaks (even if less than 3)
        best_peaks = max(all_peaks, key=len) if all_peaks else np.array([])

    tick_positions = best_peaks.tolist()

    # Filter out ticks in the text region (left of text_x_end)
    filtered_positions = [p for p in tick_positions if p > text_x_end]

    return filtered_positions


def detect_ruler_ticks_simple(image: np.ndarray, text_x_end: int = 0) -> List[int]:
    """
    Hybrid approach: try both fixed and adaptive methods, pick the best result.

    Args:
        image: Input image
        text_x_end: X-coordinate where text region ends

    Returns:
        List of x-coordinates of detected ticks
    """
    # Try both methods
    fixed_ticks = detect_ruler_ticks_fixed_region(image, text_x_end)
    adaptive_ticks = detect_ruler_ticks_adaptive(image, text_x_end)

    # Prefer result with more ticks, but not too many (avoid noise)
    # Typical rulers have 3-15 ticks with spacing > 20 pixels
    def score_result(ticks):
        n = len(ticks)
        if n < 2:
            return 0

        # Check minimum spacing - ticks should be at least 20 pixels apart
        spacings = [ticks[i+1] - ticks[i] for i in range(len(ticks)-1)]
        min_spacing = min(spacings) if spacings else 0

        if min_spacing < 20:
            # Penalize results with very close ticks (likely noise/artifacts)
            return max(0, n - 5)

        if 3 <= n <= 15:
            return n + 10  # Bonus for reasonable range with good spacing
        return n

    fixed_score = score_result(fixed_ticks)
    adaptive_score = score_result(adaptive_ticks)

    if fixed_score >= adaptive_score:
        return fixed_ticks
    else:
        return adaptive_ticks


def calculate_pixels_per_unit(tick_positions: List[int]) -> Optional[float]:
    """
    Calculate pixels per unit (cm or mm) from tick positions.
    Each gap between consecutive ticks = 1 unit.

    Args:
        tick_positions: List of x-coordinates of ticks

    Returns:
        Median gap size as pixels per unit, or None if insufficient data
    """
    if len(tick_positions) < 2:
        return None

    # Calculate gaps between consecutive ticks
    gaps = []
    for i in range(len(tick_positions) - 1):
        gap = tick_positions[i + 1] - tick_positions[i]
        gaps.append(gap)

    if not gaps:
        return None

    # Use median gap to be robust to outliers
    median_gap = np.median(gaps)

    return float(median_gap)


def detect_scale_ratio(image_path: str, visualize: bool = False, output_dir: Optional[str] = None) -> Dict:
    """
    Main function to detect scale ratio in an image.

    Args:
        image_path: Path to the image file
        visualize: If True, save visualization of detected ticks
        output_dir: Directory to save visualizations (if visualize=True)

    Returns:
        Dict with keys: 'status', 'unit', 'pixels_per_unit', 'num_ticks', 'tick_positions'
    """
    image = cv2.imread(image_path)
    if image is None:
        return {'status': 'error', 'message': 'Could not load image'}

    h, w = image.shape[:2]

    # Try to find scale text to exclude it from tick detection
    unit_type, text_x_end = find_scale_text_region(image)

    if unit_type is None:
        # Fallback: assume 'cm' if detection fails
        unit_type = 'cm'
        text_x_end = int(w * 0.10)  # Assume text is in left 10%
        print(f"  Could not find unit text, assuming '{unit_type}'")
    else:
        print(f"  Found unit: {unit_type}")

    # Detect ruler tick marks (excluding text region)
    tick_positions = detect_ruler_ticks_simple(image, text_x_end)

    if len(tick_positions) < 2:
        return {
            'status': 'error',
            'message': f'Not enough ticks detected ({len(tick_positions)})',
            'unit': unit_type
        }

    # Calculate pixels per unit
    pixels_per_unit = calculate_pixels_per_unit(tick_positions)

    if pixels_per_unit is None:
        return {
            'status': 'error',
            'message': 'Could not calculate pixel ratio',
            'unit': unit_type
        }

    # Visualization (optional)
    if visualize and output_dir:
        vis_image = image.copy()

        # Draw lines at detected tick positions
        ruler_start_y = h - 60  # Bottom 60 pixels where ruler typically is
        ruler_mid_y = h - 30     # Middle of ruler bar

        for x in tick_positions:
            # Draw vertical green lines aligned with actual tick marks
            cv2.line(vis_image, (x, ruler_start_y), (x, h), (0, 255, 0), 2)

        # Add text annotation
        text = f"{unit_type}: {pixels_per_unit:.1f}px/unit ({len(tick_positions)} ticks)"
        cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Save visualization
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{Path(image_path).stem}_detection.jpg"
        cv2.imwrite(str(output_file), vis_image)

    return {
        'status': 'success',
        'unit': unit_type,
        'pixels_per_unit': float(pixels_per_unit),
        'num_ticks': len(tick_positions),
        'tick_positions': tick_positions
    }


def main():
    """Process test images for validation"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python scale_detection_easyocr.py <image_path> [output_dir]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./debug_visualizations"

    print(f"Processing: {image_path}")
    result = detect_scale_ratio(image_path, visualize=True, output_dir=output_dir)

    if result['status'] == 'success':
        print(f"✓ Unit: {result['unit']}")
        print(f"✓ Pixels per {result['unit']}: {result['pixels_per_unit']:.2f}")
        print(f"✓ Ticks detected: {result['num_ticks']}")
    else:
        print(f"✗ Error: {result.get('message', 'Unknown error')}")


if __name__ == "__main__":
    main()
