"""
Configuration file for PPE Helmet Detection
Contains all paths, classes, and training hyperparameters
"""


import torch
from pathlib import Path

# Dataset paths
INPUT_DIR = Path('raw_data')
OUTPUT_DIR = Path('PPE_Dataset')
DATA_YAML = 'data.yaml'
PROJECT_DIR = 'output'

# Configure Ultralytics Global Settings to prevent 'runs' folder creation
try:
    from ultralytics import settings
    # Update runs_dir to use our project directory
    settings.update({'runs_dir': PROJECT_DIR})
except ImportError:
    pass

# Classes
CLASSES = ['helmet', 'head']

# Dataset splits
TRAIN_SPLIT = 0.7
VALID_SPLIT = 0.15
TEST_SPLIT = 0.15

# Training hyperparameters
TRAINING_CONFIG = {
    'model': 'yolov8m.pt',
    'epochs': 130,
    'imgsz': 640,
    'batch': 8,
    'device': 0 if torch.cuda.is_available() else 'cpu',  # Auto-detect GPU/CPU
    'workers': 4,
    'patience': 25,
    'save': True,
    'plots': True,
    
    # Data augmentation
    'hsv_h': 0.015,  # Reduced from 0.02
    'hsv_s': 0.7,    # Increased from 0.6
    'hsv_v': 0.4,
    'degrees': 15,   # Keeps rotation
    'mixup': 0.1,
    'translate': 0.1,
    'scale': 0.7,
    'shear': 0.0,
    'perspective': 0.001, # Added perspective
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 0.9,

    # Optimizer
    'optimizer': 'AdamW',
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'lr0': 0.001,
    'lrf': 0.01,
    'warmup_epochs': 5,
    'warmup_momentum': 0.8,

    # Stability
    'label_smoothing': 0.05,
    'cache': False,
    'amp': True,
    'seed': 42,
}

# Inference Thresholds (Standardized)
INFERENCE_THRESHOLDS = {
    'image': 0.25,
    'video': 0.35,
    'webcam': 0.55
}

# Testing configuration
TEST_CONFIG = {
    'conf': INFERENCE_THRESHOLDS['image'],  # Confidence threshold from centralized config
    'split': 'test',
}

# Visualization settings
VIZ_CONFIG = {
    'colors': [(255, 0, 0), (0, 255, 0)],  # BGR colors for each class (Helmet, Head)
}

def get_latest_model_path(model_type='best'):
    """
    Find the latest model weights in the project output directory.
    Checks both /workspace/output and output locally.
    """
    # Direct check first
    weights_path = Path(PROJECT_DIR) / 'train' / 'weights' / f'{model_type}.pt'
    if weights_path.exists():
        return weights_path
        
    # Check /workspace if in Docker
    weights_path_docker = Path('/workspace') / PROJECT_DIR / 'train' / 'weights' / f'{model_type}.pt'
    if weights_path_docker.exists():
        return weights_path_docker

    # Local fallback search
    base_dir = Path(PROJECT_DIR)
    if base_dir.exists():
        train_dirs = [d for d in base_dir.glob('train*') if d.is_dir()]
        if train_dirs:
            latest_run = max(train_dirs, key=lambda p: p.stat().st_mtime)
            weights_path = latest_run / 'weights' / f'{model_type}.pt'
            if weights_path.exists():
                return weights_path

    # Default fallback
    return Path(PROJECT_DIR) / 'train' / 'weights' / f'{model_type}.pt'