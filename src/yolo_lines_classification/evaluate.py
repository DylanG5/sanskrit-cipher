"""
Model Evaluation Script for YOLO Line Detection

This script evaluates the trained model on test data and generates detailed performance metrics.
"""

import os
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Evaluate YOLO model performance"""
    
    def __init__(self, model_path='../../runs/detect/runs/train/sanskrit_line_detection7/weights/best.pt'):
        """Initialize evaluator with trained model"""
        self.model = YOLO(model_path)
        print(f"Model loaded from: {model_path}")
    
    def evaluate_on_validation(self, data_yaml='data.yaml'):
        """
        Run official YOLO validation metrics.
        
        Returns:
            Validation metrics from YOLO
        """
        print("\n" + "=" * 70)
        print("Running YOLO Validation")
        print("=" * 70)
        
        results = self.model.val(data=data_yaml, plots=True)
        
        print(f"\n📊 Validation Metrics:")
        print(f"  mAP50: {results.box.map50:.4f}")
        print(f"  mAP50-95: {results.box.map:.4f}")
        print(f"  Precision: {results.box.mp:.4f}")
        print(f"  Recall: {results.box.mr:.4f}")
        
        return results
    
    def compare_predictions_to_labels(self, test_images_dir, test_labels_dir, conf_threshold=0.25):
        """
        Compare predicted line counts with ground truth labels.
        
        Args:
            test_images_dir: Directory with test images
            test_labels_dir: Directory with ground truth labels
            conf_threshold: Confidence threshold for predictions
        
        Returns:
            DataFrame with comparison results
        """
        test_images_dir = Path(test_images_dir)
        test_labels_dir = Path(test_labels_dir)
        
        image_files = sorted(list(test_images_dir.glob('*.jpg')) + 
                           list(test_images_dir.glob('*.jpeg')) + 
                           list(test_images_dir.glob('*.png')))
        
        results = []
        
        print(f"\nEvaluating {len(image_files)} test images...")
        
        for img_path in image_files:
            # Get ground truth
            label_file = test_labels_dir / (img_path.stem + '.txt')
            
            if not label_file.exists():
                continue
            
            with open(label_file, 'r') as f:
                gt_lines = len([line for line in f.readlines() if line.strip()])
            
            # Get prediction
            pred_results = self.model.predict(str(img_path), conf=conf_threshold, verbose=False)[0]
            pred_lines = len(pred_results.boxes) if pred_results.boxes is not None else 0
            
            results.append({
                'image': img_path.name,
                'ground_truth': gt_lines,
                'predicted': pred_lines,
                'error': pred_lines - gt_lines,
                'abs_error': abs(pred_lines - gt_lines),
                'correct': gt_lines == pred_lines
            })
        
        df = pd.DataFrame(results)
        return df
    
    def calculate_line_count_accuracy(self, comparison_df):
        """
        Calculate accuracy metrics for line counting task.
        
        Args:
            comparison_df: DataFrame from compare_predictions_to_labels
        
        Returns:
            Dictionary of metrics
        """
        total = len(comparison_df)
        correct = comparison_df['correct'].sum()
        accuracy = correct / total if total > 0 else 0
        
        mae = comparison_df['abs_error'].mean()
        mse = (comparison_df['error'] ** 2).mean()
        rmse = np.sqrt(mse)
        
        # Off-by-one accuracy (prediction within 1 line of ground truth)
        off_by_one = (comparison_df['abs_error'] <= 1).sum()
        off_by_one_accuracy = off_by_one / total if total > 0 else 0
        
        metrics = {
            'total_images': total,
            'exact_matches': int(correct),
            'exact_accuracy': accuracy,
            'off_by_one_accuracy': off_by_one_accuracy,
            'mean_absolute_error': mae,
            'root_mean_squared_error': rmse
        }
        
        return metrics
    
    def print_evaluation_report(self, comparison_df, metrics):
        """Print comprehensive evaluation report"""
        print("\n" + "=" * 70)
        print("LINE COUNT ACCURACY EVALUATION")
        print("=" * 70)
        
        print(f"\n✅ Exact Match Accuracy: {metrics['exact_accuracy']*100:.2f}%")
        print(f"   ({metrics['exact_matches']}/{metrics['total_images']} images)")
        
        print(f"\n✨ Off-by-One Accuracy: {metrics['off_by_one_accuracy']*100:.2f}%")
        print(f"   (Prediction within ±1 line)")
        
        print(f"\n📏 Error Metrics:")
        print(f"   Mean Absolute Error (MAE): {metrics['mean_absolute_error']:.3f} lines")
        print(f"   Root Mean Squared Error (RMSE): {metrics['root_mean_squared_error']:.3f} lines")
        
        # Error distribution
        print(f"\n📊 Error Distribution:")
        error_counts = comparison_df['error'].value_counts().sort_index()
        for error, count in error_counts.items():
            symbol = "✓" if error == 0 else ("+" if error > 0 else "")
            print(f"   {symbol}{error:+2d} lines: {count:3d} images")
        
        # Show worst predictions
        worst = comparison_df.nlargest(5, 'abs_error')[['image', 'ground_truth', 'predicted', 'error']]
        if len(worst) > 0:
            print(f"\n⚠️  Largest Errors:")
            for idx, row in worst.iterrows():
                print(f"   {row['image']}: GT={row['ground_truth']}, "
                      f"Pred={row['predicted']} (error: {row['error']:+d})")
    
    def plot_confusion_matrix(self, comparison_df, save_path='confusion_matrix_lines.png'):
        """Plot confusion matrix for line counts"""
        y_true = comparison_df['ground_truth']
        y_pred = comparison_df['predicted']
        
        # Create confusion matrix
        unique_labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=unique_labels, yticklabels=unique_labels)
        plt.xlabel('Predicted Lines')
        plt.ylabel('Ground Truth Lines')
        plt.title('Line Count Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\nConfusion matrix saved to: {save_path}")
        plt.close()
    
    def plot_error_distribution(self, comparison_df, save_path='error_distribution.png'):
        """Plot error distribution histogram"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Error distribution
        axes[0, 0].hist(comparison_df['error'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Prediction Error (Predicted - Ground Truth)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Error Distribution')
        axes[0, 0].grid(alpha=0.3)
        
        # Absolute error distribution
        axes[0, 1].hist(comparison_df['abs_error'], bins=20, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Absolute Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Absolute Error Distribution')
        axes[0, 1].grid(alpha=0.3)
        
        # Scatter plot: Predicted vs Ground Truth
        axes[1, 0].scatter(comparison_df['ground_truth'], comparison_df['predicted'], alpha=0.5)
        max_val = max(comparison_df['ground_truth'].max(), comparison_df['predicted'].max())
        axes[1, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect prediction')
        axes[1, 0].set_xlabel('Ground Truth Lines')
        axes[1, 0].set_ylabel('Predicted Lines')
        axes[1, 0].set_title('Predicted vs Ground Truth')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Accuracy by ground truth line count
        accuracy_by_lines = comparison_df.groupby('ground_truth')['correct'].mean()
        axes[1, 1].bar(accuracy_by_lines.index, accuracy_by_lines.values, edgecolor='black', alpha=0.7, color='green')
        axes[1, 1].set_xlabel('Ground Truth Line Count')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Accuracy by Line Count')
        axes[1, 1].set_ylim([0, 1.1])
        axes[1, 1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Error distribution plots saved to: {save_path}")
        plt.close()

def main():
    """Main evaluation pipeline"""
    base_dir = Path(__file__).parent
    
    # Use images and labels folders directly
    test_images = base_dir / 'images'
    test_labels = base_dir / 'labels'
    
    if not test_images.exists() or not test_labels.exists():
        print("❌ Images or labels folder not found.")
        return
    
    print("=" * 70)
    print("YOLO LINE DETECTION MODEL EVALUATION")
    print("=" * 70)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Run YOLO validation
    evaluator.evaluate_on_validation(str(base_dir / 'data.yaml'))
    
    # Compare predictions with ground truth
    print("\n" + "=" * 70)
    print("Comparing Predictions to Ground Truth")
    print("=" * 70)
    
    comparison_df = evaluator.compare_predictions_to_labels(test_images, test_labels)
    
    # Calculate metrics
    metrics = evaluator.calculate_line_count_accuracy(comparison_df)
    
    # Print report
    evaluator.print_evaluation_report(comparison_df, metrics)
    
    # Save comparison to CSV
    output_csv = 'evaluation_results.csv'
    comparison_df.to_csv(output_csv, index=False)
    print(f"\n💾 Detailed results saved to: {output_csv}")
    
    # Generate plots
    print("\n" + "=" * 70)
    print("Generating Visualization Plots")
    print("=" * 70)
    
    evaluator.plot_confusion_matrix(comparison_df)
    evaluator.plot_error_distribution(comparison_df)
    
    print("\n✅ Evaluation completed!")
    print("=" * 70)

if __name__ == '__main__':
    main()
