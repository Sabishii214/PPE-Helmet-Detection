# PPE Detection Dataset Preparation

This project prepares a PPE (Hard Hat) detection dataset for YOLO training.  
It downloads the dataset from Kaggle, converts Pascal VOC XML annotations to YOLO format, and splits the data into train, validation, and test sets.

## Features
- XML → YOLO annotation conversion
- Train / Val / Test split (70 / 15 / 15)
- Automatic `data.yaml` generation
- Bounding box visualization

## Classes
- helmet
- head
- person
