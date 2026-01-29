#!/bin/bash
# Docker Cleanup and Run Script for PPE Helmet Detection

set -e  # Exit on error

echo "PPE Helmet Detection - Docker Runner"
echo ""

# Check if container is running
if ! docker ps | grep -q "gpu-ml"; then
    echo "Container 'gpu-ml' is not running. Starting it..."
    docker start gpu-ml
    sleep 2
fi

echo "Container 'gpu-ml' is running"
echo ""

# Clean up old training state (not the entire folder)
echo "Cleaning up old training state..."
docker exec gpu-ml rm -rf /workspace/output/train 2>/dev/null || true
docker exec gpu-ml rm -rf /workspace/__pycache__ 2>/dev/null || true
echo "Cleanup complete"
echo ""

# Verify dependencies
echo "Verifying dependencies..."
if docker exec gpu-ml python3 -c "import ultralytics, cv2, pandas, matplotlib, numpy, sklearn" 2>/dev/null; then
    echo "All dependencies installed"
else
    echo "Some dependencies missing. Installing..."
    docker exec gpu-ml pip install -r /workspace/requirements.txt
fi
echo ""

# Check GPU
echo "Checking GPU availability..."
docker exec gpu-ml nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU check failed"
echo ""

# Run main.py using tmux for persistence
SESSION_NAME="ppe_training"
echo "Starting training pipeline via tmux..."
echo "Session Name: $SESSION_NAME"
echo ""

# Check if session exists inside container
if ! docker exec gpu-ml tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Creating new tmux session..."
    # Create detached session
    docker exec gpu-ml tmux new-session -d -s $SESSION_NAME
    
    # Send command (wrapped to keep window open after finish)
    CMD="python3 /workspace/main.py $*"
    docker exec gpu-ml tmux send-keys -t $SESSION_NAME "$CMD; echo ''; echo 'Process finished. Press Enter to close window.'; read" C-m
else
    echo "Session already exists. Attaching..."
fi

echo "Attaching to session (Ctrl+B, D to detach)..."
# Use -it to allow interaction
docker exec -it gpu-ml tmux attach -t $SESSION_NAME

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "Detached or Finished."
    echo ""
    echo "To reattach: ./docker_run.sh"
    echo "Output files are in: /workspace/output/"
else
    echo "Tmux attach failed with code: $exit_code"
fi

exit $exit_code
