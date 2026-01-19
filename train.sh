# PPE Helmet Detection Training Launcher
# This script helps run the training with proper logging and monitoring

echo " "
echo "PPE Helmet Detection - Training Launcher"
echo " "

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed!"
    exit 1
fi

# Check if NVIDIA GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "Warning: No NVIDIA GPU detected. Training will be very slow on CPU."
    echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
    sleep 5
fi

# Create logs directory
mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${TIMESTAMP}.log"

echo "Starting training..."

# Run the training script with logging
python3 PPE_Helmet_Detection.py 2>&1 | tee "$LOG_FILE"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
    echo ""
    echo "Results saved in:"
    echo "  - Model: output/train/weights/best.pt"
    echo "  - Reports: performance_report.txt, comparison_report.txt"
    echo "  - Log: $LOG_FILE"
else
    echo ""
    echo "Training failed! Check the log file:"
    echo "  $LOG_FILE"
    exit 1
fi