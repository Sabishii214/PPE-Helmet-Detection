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

# Clean up user-created files
echo "Cleaning up old output files..."
docker exec gpu-ml rm -rf /workspace/output /workspace/__pycache__ 2>/dev/null || true
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

# Run main.py with any arguments passed to this script
echo "Starting training pipeline..."
echo ""

docker exec gpu-ml python3 /workspace/main.py "$@"

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "Pipeline completed successfully!"
    echo ""
    echo "Output files are in: /workspace/output/"
    echo "To copy to host: docker cp gpu-ml:/workspace/output ./output_backup"
else
    echo "Pipeline failed with exit code: $exit_code"
fi

exit $exit_code
