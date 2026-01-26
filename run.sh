#!/bin/bash
# Quick Start Script for PPE Helmet Detection
# This script runs the complete pipeline using your virtual environment

echo " "
echo "PPE HELMET DETECTION - QUICK START"
echo " "

# Check if virtual environment exists
if [ ! -d "gpuenv" ]; then
    echo "Error: Virtual environment 'gpuenv' not found!"
    echo "Please create it first: python3 -m venv gpuenv"
    exit 1
fi

# Use the virtual environment's Python
PYTHON="./gpuenv/bin/python3"

echo "Using Python from: $PYTHON"
echo ""

# Check if dependencies are installed
echo "Checking dependencies..."
$PYTHON -c "import sklearn, cv2, ultralytics" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Some dependencies are missing. Installing..."
    $PYTHON -m pip install -r requirements.txt
fi

echo "Dependencies OK"
echo ""

# Run the pipeline
echo " "
echo "Starting training pipeline..."
echo " "

$PYTHON main.py "$@"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo " "
    echo "Pipeline completed successfully!"
    echo " "
else
    echo " "
    echo "Pipeline failed with exit code: $exit_code"
    echo " "
fi

exit $exit_code