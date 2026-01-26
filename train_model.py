"""
Model training module
Handles YOLO model training with configured hyperparameters
"""
import os
import subprocess
import sys
from pathlib import Path

# Avoid fragmentation in memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from config import DATA_YAML, TRAINING_CONFIG, get_latest_model_path, PROJECT_DIR
from resource_monitor import ResourceMonitor

def install_ultralytics():
    """Install ultralytics package if not already installed"""
    try:
        from ultralytics import YOLO
        print("Ultralytics already installed")
    except ImportError:
        print("Installing ultralytics...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        print("Ultralytics installed successfully")


def train_model():
    """Train YOLOv8 model with configured hyperparameters"""
    from ultralytics import YOLO
    
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    # Check if data.yaml exists
    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(
            f"{DATA_YAML} not found. Please run data_preparation.py first."
        )
    
    # Initialize Resource Monitor
    monitor = ResourceMonitor(interval=2.0)
    monitor.start()
    
    training_results = None
    
    try:
        # Check for checkpoint to resume
        last_ckpt = Path(PROJECT_DIR) / 'train' / 'weights' / 'last.pt'
        if last_ckpt.exists():
            print(f"\nCheckpoint found: {last_ckpt}")
            print("Resuming training from last checkpoint...")
            model = YOLO(last_ckpt)
            resume = True
        else:
            # Load pre-trained model
            print(f"\nLoading base model: {TRAINING_CONFIG['model']}")
            model = YOLO(TRAINING_CONFIG['model'])
            resume = False
        
        # Start training
        print("\nStarting training...")
        print(f"Epochs: {TRAINING_CONFIG['epochs']}")
        print(f"Batch size: {TRAINING_CONFIG['batch']}")
        print(f"Image size: {TRAINING_CONFIG['imgsz']}")
        print(f"Device: {TRAINING_CONFIG['device']}")
        print(f"Resume: {resume}")
        print("-"*60)
        
        # Remove 'model' from config when resuming as we already loaded it
        train_args = TRAINING_CONFIG.copy()
        if resume:
            train_args.pop('model', None)

        training_results = model.train(
            data=DATA_YAML,
            project=PROJECT_DIR,
            name='train',
            exist_ok=False,  # Force unique directories to prevent overwriting
            resume=resume,
            **train_args
        )
        
        best_path = get_latest_model_path('best')
        last_path = get_latest_model_path('last')
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Best model: {best_path}")
        print(f"Last model: {last_path}")
        print(f"Results: {PROJECT_DIR}/train/")
        print("="*60 + "\n")
        
        return training_results

    finally:
        # Stop monitoring and generate report
        monitor.stop()
        report = monitor.generate_report()
        
        # Append training metrics if available
        if training_results and hasattr(training_results, 'results_dict'):
            try:
                metrics = training_results.results_dict
                # Keys might vary slightly, usually: metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)
                map50 = metrics.get('metrics/mAP50(B)', 0)
                map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
                precision = metrics.get('metrics/precision(B)', 0)
                recall = metrics.get('metrics/recall(B)', 0)
                
                metrics_report = [
                    "",
                    "-" * 60,
                    "TRAINING FINAL ACCURACY METRICS (Validation Set)",
                    f"  mAP@50:       {map50:.4f} ({map50*100:.2f}%)",
                    f"  mAP@50-95:    {map50_95:.4f} ({map50_95*100:.2f}%)",
                    f"  Precision:    {precision:.4f}",
                    f"  Recall:       {recall:.4f}",
                    "=" * 60
                ]
                report += "\n" + "\n".join(metrics_report)
            except Exception as e:
                print(f"Could not extract metrics: {e}")

        print(report)
        
        # Save report to file (append to existing performance report or create new one)
        try:
            try:
                best_path = get_latest_model_path('best')
                run_dir = best_path.parent.parent
                report_path = run_dir / 'resource_report.txt'
                with open(report_path, 'w') as f:
                    f.write(report)
                print(f"\nResource report saved to: {report_path}")
                
            except Exception:
                # Fallback
                print("\nCould not determine run directory to save resource report.")
                
        except Exception as e:
            print(f"Error saving resource report: {e}")

if __name__ == "__main__":
    install_ultralytics()
    train_model()