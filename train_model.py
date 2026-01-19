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
        exist_ok=True,
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

if __name__ == "__main__":
    install_ultralytics()
    train_model()