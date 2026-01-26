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
        choices=['prepare', 'train', 'evaluate', 'export', 'visualize', 'all'],
        default='all',
        help='Run specific step (default: all)'
    )
    
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

if __name__ == "__main__":
    main()