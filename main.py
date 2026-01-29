#!/usr/bin/env python3
"""
PPE Helmet Detection - Main Pipeline
Complete training pipeline from data preparation to model export
"""
import argparse
import sys

# Import all modules
from data_preparation import prepare_dataset
from train_model import install_ultralytics, train_model
from evaluate_model import test_and_report
from visualize import visualize_dataset_samples, visualize_predictions, print_dataset_statistics
from export_model import export_and_optimize
from config import get_latest_model_path
from analytic import PPEDetectionAnalytics


def run_full_pipeline():
    """Run the complete training pipeline"""
    print("\n" + "="*70)
    print(" "*20 + "PPE HELMET DETECTION")
    print(" "*20 + "COMPLETE PIPELINE")
    print("="*70 + "\n")
    
    try:
        # Step 1: Prepare dataset
        print("STEP 1/6: Dataset Preparation")
        prepare_dataset()
        
        # Step 2: Visualize dataset
        print("\nSTEP 2/6: Dataset Visualization")
        print_dataset_statistics()
        visualize_dataset_samples(split='train', num_samples=4)
        
        # Step 3: Install dependencies
        print("\nSTEP 3/6: Installing Dependencies")
        install_ultralytics()
        
        # Step 4: Train model
        print("\nSTEP 4/6: Model Training")
        train_model()
        
        # Step 5: Evaluate model
        print("\nSTEP 5/6: Model Evaluation")
        test_and_report()
        
        # Step 6: Export and optimize
        print("\nSTEP 6/6: Model Export & Optimization")
        export_and_optimize()
        
        # Final visualization
        print("\nGenerating prediction visualizations...")
        best_model_path = get_latest_model_path('best')
        visualize_predictions(best_model_path, split='test', num_samples=8)
        
        best_path = get_latest_model_path('best')
        last_path = get_latest_model_path('last')
        
        # Get the run directory for reports
        run_dir = best_path.parent.parent
        
        print("\n" + "="*70)
        print(" "*15 + "PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nGenerated files:")
        print("Model:")
        print(f"     - {best_path}")
        print(f"     - {last_path}")
        print("Reports:")
        print(f"     - {run_dir}/performance_report.txt")
        print(f"     - {run_dir}/optimization_report.txt")
        print("Visualizations:")
        print(f"     - output/visualizations/dataset_samples_*.png")
        print(f"     - {run_dir}/test_predictions.png")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """Main entry point with command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description='PPE Helmet Detection Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py
  
  # Run specific steps
  python main.py --step prepare
  python main.py --step train
  python main.py --step evaluate
  python main.py --step export
  python main.py --step visualize
        """
    )
    
    parser.add_argument(
        '--step',
        choices=['prepare', 'train', 'evaluate', 'export', 'visualize', 'analytics', 'all'],
        default='all',
        help='Run specific step (default: all)'
    )
    
    # Analytics specific arguments
    parser.add_argument('--mode', choices=['images', 'videos', 'webcam', 'pipeline', 'all'], 
                        default='pipeline', help='Analytics mode (if step=analytics)')
    parser.add_argument('--cam', type=int, default=0, help='Camera index for webcam')
    parser.add_argument('--duration', type=int, default=30, help='Webcam capture duration (sec)')
    parser.add_argument('--input', help='Input path for analytics')
    
    args = parser.parse_args()
    
    # Run based on selected step
    if args.step == 'all':
        run_full_pipeline()
    
    elif args.step == 'prepare':
        print("Running: Dataset Preparation")
        prepare_dataset()
        print_dataset_statistics()
    
    elif args.step == 'train':
        print("Running: Model Training")
        install_ultralytics()
        train_model()
    
    elif args.step == 'evaluate':
        print("Running: Model Evaluation")
        test_and_report()
    
    elif args.step == 'export':
        print("Running: Model Export & Optimization")
        export_and_optimize()
    
    elif args.step == 'visualize':
        print("Running: Visualization")
        print_dataset_statistics()
        visualize_dataset_samples(split='train', num_samples=4)
        
        # If model exists, visualize predictions
        from pathlib import Path
        best_model_path = get_latest_model_path('best')
        if best_model_path.exists():
            visualize_predictions(best_model_path, split='test', num_samples=8)
        else:
            print("Note: No trained model found. Train first to visualize predictions.")
    
    elif args.step == 'analytics':
        print(f"Running: Analytics (Mode: {args.mode})")
        analytics = PPEDetectionAnalytics(conf_threshold=0.2)
        
        if args.mode == 'images':
            analytics.process_images(args.input)
        elif args.mode == 'videos':
            analytics.process_videos(args.input)
        elif args.mode == 'webcam':
            result = analytics.process_webcam(camera_index=args.cam, duration_seconds=args.duration)
            if not result:
                print("Error: Webcam could not be opened. Check if the camera is connected and Docker has permission.")
                return
        elif args.mode == 'all':
            analytics.process_images(args.input)
            analytics.process_videos(args.input)
            analytics.process_webcam(camera_index=args.cam, duration_seconds=args.duration)
        else: # pipeline
            analytics.run_complete_pipeline(args.input)
            return # run_complete_pipeline already saves and prints
            
        if analytics.results_log:
            analytics.save_report()
            analytics.save_text_report()
            analytics.print_summary()
        else:
            print("No data processed. No reports generated.")

if __name__ == "__main__":
    main()