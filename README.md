# PPE Helmet Detection

## Overview
A modular YOLOv8-based system for detecting PPE (Personal Protective Equipment) helmets, heads, and persons in images.

## Dataset Structure

The script expects/creates the following structure:
```
PPE-Helmet-Detection/
├── raw_data/
│   ├── images/
│   └── annotations/
├── PPE_Dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── output/
│   ├── train/                   
│   ├── weights/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── performance_report.txt
│   └── optimization_report.txt
└── visualizations/
│   └── dataset_samples_*.png
├── analytic.py
├── check_ckpt.py
├── config.py
├── data_preparation.py
├── evaluate_model.py
├── export_model.py
├── main.py
├── train_model.py
├── visualize.py
├── data.yaml
├── requirements.txt
├── run.sh
├── yolov8*.pt
├── train.sh
└── README.md
```

## Prerequisites

### Required Python Packages
```bash
pip install numpy pandas opencv-python matplotlib seaborn scikit-learn ultralytics kagglehub tensorrt onnx-graphsurgeon onnx-simplifier
```

## Run Full Pipeline (Recommended for First Time)
docker exec gpu-ml python3 /workspace/main.py

# Enter the container
docker exec -it gpu-ml bash

# If container is already used, to check:
docker ps -a | grep gpu-ml

# To use the container:
docker start -ai gpu-ml

# To stop the container:
docker stop gpu-ml

# To remove the container:
docker rm gpu-ml

# Create Virtual Environment
```bash
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

### Fastest Way to Run
# Option 1: Use the automated script (RECOMMENDED)
```bash
./docker_run.sh
```

# Option 2: Manual command
```bash
docker exec gpu-ml python3 /workspace/main.py
```

### Direct Module Execution
```bash
python3 data_preparation.py  # Just prepare data
python3 train_model.py       # Just train
python3 evaluate_model.py    # Just evaluate
python3 export_model.py      # Just export
python3 visualize.py         # Just visualize
```

### Run Complete Pipeline
```bash
python3 main.py
```

This will:
1. Download and prepare the dataset
2. Visualize dataset samples
3. Train the YOLOv8 model
4. Evaluate on test set
5. Export to ONNX/TorchScript/TensorRT
6. Generate all reports and visualizations

### Run Specific Steps

```bash
# Prepare dataset only
python3 main.py --step prepare

# Train model only (requires prepared dataset)
python3 main.py --step train

# Evaluate trained model
python3 main.py --step evaluate

# Export model to ONNX/TorchScript
python3 main.py --step export

# Visualize dataset and predictions
python3 main.py --step visualize
```

### Run Individual Modules

```bash
# Data preparation
python3 data_preparation.py

# Training
python3 train_model.py

# Evaluation
python3 evaluate_model.py

# Visualization
python3 visualize.py

# Export
python3 export_model.py

## Resume training
    python3 main.py --step prepare
    python3 main.py --step train
```

## Configuration

Edit `config.py` to customize:

- Dataset paths: `INPUT_DIR`, `OUTPUT_DIR`
- Training parameters: Epochs, batch size, learning rate, etc.
- Data augmentation: Rotation, flip, mixup, etc.
- Model selection: YOLOv8s/m/l/x variants

## Usage

### 1. Basic Training (Full Pipeline)
Run the entire training pipeline:
```bash
python3 main.py
```

### 2. Remote GPU Training
When connecting to a remote GPU server:

```bash
# SSH into your remote GPU server
ssh user@remote-gpu-server

# Clone/upload your project
# Then run the training script
python3 main.py
```

### 3. Using Screen/Tmux for Long Training Sessions
To keep training running even if you disconnect:

```bash
# Using screen
screen -S ppe_training
python3 main.py
# Press Ctrl+A then D to detach
# Reattach later with: screen -r ppe_training

# Using tmux
tmux new -s ppe_training
python3 main.py
# Press Ctrl+B then D to detach
# Reattach later with: tmux attach -t ppe_training
```

## Training Configuration

The model is configured with the following parameters:
- Model: YOLOv8*
- Epochs: 130
- Image Size: 768x768
- Batch Size: 16
- Device: GPU (device=0)
- Workers: 4
- Patience: 25 (early stopping)

### Data Augmentation
- HSV augmentation (Hue, Saturation, Value)
- Rotation: ±15 degrees
- Mixup: 0.1
- Copy-paste: 0.1
- Translation: 0.15
- Scale: 0.7
- Horizontal flip: 0.5
- Mosaic: 1.0

### Optimizer Settings
- Optimizer: AdamW
- Initial LR: 0.01
- Final LR: 0.01
- Weight Decay: 0.0005
- Warmup Epochs: 3

## Output Files

After training completes, you'll find:

### Model Files
- `output/train/weights/best.pt` - Best model checkpoint
- `output/train/weights/last.pt` - Last epoch checkpoint

### Reports
- `performance_report.txt` - Detailed performance metrics
- `optimization_report.txt` - Optimization and speed metrics

### Visualizations
- `test_predictions.png` - Sample predictions on test images

### Exported Models
- ONNX format (for deployment)
- TorchScript format (for production)
- TensorRT

# Output files are in: /workspace/output/
To copy to host: docker cp gpu-ml:/workspace/output ./output_backup

# Module Descriptions

### `config.py`
Central configuration file for all hyperparameters and paths.

### `data_preparation.py`
- Downloads dataset from Kaggle
- Converts XML annotations to YOLO format
- Splits into train/valid/test sets
- Creates data.yaml config

### `train_model.py`
- Loads YOLOv8 model
- Trains with configured hyperparameters
- Saves checkpoints and training plots

### `evaluate_model.py`
- Tests model on test set
- Generates performance metrics
- Creates detailed reports

### `visualize.py`
- Visualizes dataset samples
- Shows model predictions
- Prints dataset statistics

### `export_model.py`
- Exports to ONNX format
- Exports to TorchScript format
- Exports to TensorRT format
- Benchmarks inference speed
- Generates optimization report

### `main.py`
- Orchestrates complete pipeline
- CLI interface for running specific steps
- Error handling and logging

### Custom Dataset
1. Place images in `raw_data/images/`
2. Place XML annotations in `raw_data/annotations/`
3. Run `python data_preparation.py`

## Monitoring Training

### GPU Utilization
Monitor GPU usage during training:
```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

### Training Logs
The script prints progress to stdout. To save logs:
```bash
python3 main.py 2>&1 | tee training.log
```

## Troubleshooting

### Out of Memory (OOM)
If you encounter OOM errors, reduce the batch size:
- Edit line 203 in the script: `batch=16` → `batch=8` or `batch=4`

Check for other GPU processes
```bash
ps -ef | grep python3

If possible, kill processes that you don’t need:
kill -9 <pid>
```

### CUDA Not Available
If GPU is not detected:
- Check CUDA installation: `nvidia-smi`
- Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Change device to CPU: Edit line 204: `device=0` → `device='cpu'` (much slower)

### FIX (INSIDE THE CONTAINER)
```bash
apt update
apt install -y libgl1 libglib2.0-0
```

### Remove old output files
```bash
docker exec gpu-ml rm -rf /workspace/output /workspace/__pycache__
```

### Verify Dependencies
Check that all required packages are installed:
```bash
docker exec gpu-ml python3 -c "import ultralytics, cv2, pandas, matplotlib, numpy, sklearn; print('All dependencies installed')"
```
Expected output: All dependencies installed

If any packages are missing, install them:
```bash
docker exec gpu-ml pip install -r /workspace/requirements.txt
```

### Resume Training
model = YOLO("output/train/weights/last.pt")

training_results = model.train(

    data=DATA_YAML,
    
    epochs=200,
    
    resume=True,  # Add this line
    
    # ... rest of parameters

)
