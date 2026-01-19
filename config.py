"""
Configuration file for PPE Helmet Detection
Contains all paths, classes, and training hyperparameters
"""
from pathlib import Path

# Dataset paths
INPUT_DIR = Path('raw_data')
OUTPUT_DIR = Path('PPE_Dataset')
DATA_YAML = 'data.yaml'
PROJECT_DIR = 'output'

# Classes
CLASSES = ['helmet', 'head', 'person']

# Dataset splits
TRAIN_SPLIT = 0.7
VALID_SPLIT = 0.15
TEST_SPLIT = 0.15

# Training hyperparameters
TRAINING_CONFIG = {
    'model': 'yolov8l.pt',
    'epochs': 200,
    'imgsz': 640,
    'batch': 16,
    'device': 0,  # GPU device (0 for first GPU, 'cpu' for CPU)
    'workers': 0,
    'patience': 30,
    'save': True,
    'plots': True,
    
    # Data augmentation
    'hsv_h': 0.02,
    'hsv_s': 0.8,
    'hsv_v': 0.5,
    'degrees': 15,
    'mixup': 0.1,
    'copy_paste': 0.1,
    'translate': 0.15,
    'scale': 0.7,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,
    
    # Optimizer
    'optimizer': 'AdamW',
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'lr0': 0.01,
    'lrf': 0.01,
    'warmup_epochs': 3,
    'warmup_momentum': 0.8,
}

# Testing configuration
TEST_CONFIG = {
    'conf': 0.2,  # Confidence threshold
    'split': 'test',
}

# Visualization settings
VIZ_CONFIG = {
    'colors': [(255, 0, 0), (0, 255, 0), (0, 0, 255)],  # BGR colors for each class
}

def get_latest_model_path(model_type='best'):
    """
    Find the latest model weights in the project output directory.
    Checks both /workspace/output and output locally.
    
    Args:
        model_type: 'best' or 'last' weights
        
    Returns:
        Path object to model weights, or default fallback path
    """
    output_dirs = [Path('/workspace') / PROJECT_DIR, Path(PROJECT_DIR)]
    
    for base_dir in output_dirs:
        # Check train/weights first (main training output)
        weights_path = base_dir / 'train' / 'weights' / f'{model_type}.pt'
        if weights_path.exists():
            return weights_path
            
        # Fallback to checking any train* directory if legacy structure exists
        if base_dir.exists():
            train_dirs = [d for d in base_dir.glob('train*') if d.is_dir()]
            if train_dirs:
                latest_run = max(train_dirs, key=lambda p: p.stat().st_mtime)
                weights_path = latest_run / 'weights' / f'{model_type}.pt'
                if weights_path.exists():
                    return weights_path

    # Default fallback
    return Path(PROJECT_DIR) / 'train' / 'weights' / f'{model_type}.pt'