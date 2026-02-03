"""
YOLOv8 Inference Script for Sanskrit Line Detection

This script uses a trained YOLOv8 model to detect lines in Sanskrit manuscript images
and count the number of lines per fragment.
"""

import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

class LineDetector:
    
    def __init__(self, model_path='runs/train/sanskrit_line_detection/weights/best.pt'):
        self.model = YOLO(model_path)
        print(f"Model loaded from: {model_path}")
    
    def detect_lines(self, image_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Detect lines in a single image.
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections (0-1)
            iou_threshold: IoU threshold for NMS (Non-Maximum Suppression)
        
        Returns:
            dict with detection results
        """
        # Run inference
        results = self.model.predict(
            image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )[0]
        
        # Extract detections
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else np.array([])
        confidences = results.boxes.conf.cpu().numpy() if results.boxes is not None else np.array([])
        
        # Count lines
        num_lines = len(boxes)
        
        return {
            'image_path': image_path,
            'num_lines': num_lines,
            'boxes': boxes,
            'confidences': confidences,
            'mean_confidence': float(np.mean(confidences)) if len(confidences) > 0 else 0.0
        }
    
    def detect_batch(self, image_dir, output_csv='line_count_results.csv', 
                     conf_threshold=0.25, visualize=True, output_dir='detections_output'):
        """
        Detect lines in a batch of images.
        
        Args:
            image_dir: Directory containing images
            output_csv: Path to save CSV results
            conf_threshold: Confidence threshold for detections
            visualize: Whether to save visualization images
            output_dir: Directory to save visualizations
        
        Returns:
            DataFrame with results
        """
        image_dir = Path(image_dir)
        image_files = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.jpeg')) + list(image_dir.glob('*.png')))
        
        if len(image_files) == 0:
            print(f"No images found in {image_dir}")
            return None
        
        print(f"Processing {len(image_files)} images...")
        
        # Create output directory if visualizing
        if visualize:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
        
        results_list = []
        
        for img_path in tqdm(image_files, desc="Detecting lines"):
            # Detect lines
            result = self.detect_lines(str(img_path), conf_threshold=conf_threshold)
            
            # Store results
            results_list.append({
                'image_name': img_path.name,
                'num_lines': result['num_lines'],
                'mean_confidence': result['mean_confidence']
            })
            
            # Visualize if requested
            if visualize and result['num_lines'] > 0:
                self.visualize_detections(str(img_path), result, 
                                         output_path / f"detected_{img_path.name}")
        
        # Create DataFrame
        df = pd.DataFrame(results_list)
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")
        
        # Print statistics
        print("\n" + "=" * 60)
        print("Detection Statistics")
        print("=" * 60)
        print(f"Total images processed: {len(df)}")
        print(f"Total lines detected: {df['num_lines'].sum()}")
        print(f"Average lines per image: {df['num_lines'].mean():.2f}")
        print(f"Min lines: {df['num_lines'].min()}")
        print(f"Max lines: {df['num_lines'].max()}")
        print(f"Average confidence: {df['mean_confidence'].mean():.3f}")
        
        # Line count distribution
        print("\nLine count distribution:")
        print(df['num_lines'].value_counts().sort_index())
        
        return df
    
    def visualize_detections(self, image_path, detection_result, output_path):
        """
        Visualize detection results on the image.
        
        Args:
            image_path: Path to original image
            detection_result: Detection results from detect_lines()
            output_path: Path to save visualization
        """
        # Read image
        img = cv2.imread(str(image_path))
        
        # Draw bounding boxes
        for i, (box, conf) in enumerate(zip(detection_result['boxes'], detection_result['confidences'])):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label with line number and confidence
            label = f"Line {i+1}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw label background
            cv2.rectangle(img, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add total line count
        total_text = f"Total Lines: {detection_result['num_lines']}"
        cv2.putText(img, total_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Save image
        cv2.imwrite(str(output_path), img)
    
    def predict_single(self, image_path, visualize=True, save_path=None):
        """
        Predict and optionally visualize a single image.
        
        Args:
            image_path: Path to image
            visualize: Whether to display the result
            save_path: Path to save visualization (optional)
        
        Returns:
            Detection results
        """
        result = self.detect_lines(image_path)
        
        print(f"\nImage: {Path(image_path).name}")
        print(f"Lines detected: {result['num_lines']}")
        print(f"Mean confidence: {result['mean_confidence']:.3f}")
        
        if visualize:
            if save_path is None:
                save_path = f"detected_{Path(image_path).name}"
            self.visualize_detections(image_path, result, save_path)
            print(f"Visualization saved to: {save_path}")
        
        return result

def main():
    """Main inference function with CLI"""
    parser = argparse.ArgumentParser(description='Detect lines in Sanskrit manuscript images using YOLOv8')
    parser.add_argument('--model', type=str, 
                       default='runs/train/sanskrit_line_detection/weights/best.pt',
                       help='Path to trained model weights')
    parser.add_argument('--image', type=str, help='Path to single image for prediction')
    parser.add_argument('--dir', type=str, help='Directory containing images for batch processing')
    parser.add_argument('--output-csv', type=str, default='line_count_results.csv',
                       help='Output CSV file for batch results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization output')
    parser.add_argument('--output-dir', type=str, default='detections_output',
                       help='Directory for visualization outputs')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = LineDetector(model_path=args.model)
    
    # Single image prediction
    if args.image:
        detector.predict_single(args.image, visualize=not args.no_viz)
    
    # Batch prediction
    elif args.dir:
        detector.detect_batch(
            args.dir, 
            output_csv=args.output_csv,
            conf_threshold=args.conf,
            visualize=not args.no_viz,
            output_dir=args.output_dir
        )
    
    else:
        print("Please specify either --image for single prediction or --dir for batch processing")
        parser.print_help()

if __name__ == '__main__':
    main()
