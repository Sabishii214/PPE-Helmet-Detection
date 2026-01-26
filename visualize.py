"""
Visualization utilities
Functions for visualizing dataset samples and model predictions
"""
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
from collections import Counter
from config import OUTPUT_DIR, CLASSES, VIZ_CONFIG, get_latest_model_path


def draw_boxes(img_path, label_path):
    """
    Draw bounding boxes on image
    
    Args:
        img_path: Path to image file
        label_path: Path to YOLO format label file
        
    Returns:
        Image with bounding boxes drawn
    """
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    colors = VIZ_CONFIG['colors']
    
    with open(label_path) as f:
        for line in f:
            class_id, x_center, y_center, box_width, box_height = map(float, line.split())
            
            # Convert from YOLO format to pixel coordinates
            x1 = int((x_center - box_width/2) * width)
            y1 = int((y_center - box_height/2) * height)
            x2 = int((x_center + box_width/2) * width)
            y2 = int((y_center + box_height/2) * height)
            
            # Draw rectangle and label
            color = colors[int(class_id)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, CLASSES[int(class_id)], (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img


def visualize_dataset_samples(split='train', num_samples=4, save_path=None):
    """
    Visualize random samples from dataset
    
    Args:
        split: Dataset split ('train', 'valid', or 'test')
        num_samples: Number of samples to visualize
        save_path: Path to save visualization (if None, saves to output/visualizations with timestamp)
    """
    from datetime import datetime
    
    # Determine output path if not specified
    if save_path is None:
        viz_dir = Path('output') / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = viz_dir / f'dataset_samples_{split}_{timestamp}.png'
    
    print(f"\nVisualizing {num_samples} samples from {split} set...")
    
    samples = list((OUTPUT_DIR / split / 'images').glob('*.png'))[:num_samples]
    
    rows = (num_samples + 1) // 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6*rows))
    
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.ravel()
    
    for idx, img_path in enumerate(samples):
        label_path = OUTPUT_DIR / split / 'labels' / f"{img_path.stem}.txt"
        
        if label_path.exists():
            img = draw_boxes(img_path, label_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f"Sample {idx+1}", fontsize=10)
        
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {save_path}")
    plt.close()


def visualize_predictions(model, split='test', num_samples=8, 
                         conf_threshold=0.3, save_path=None):
    """
    Visualize model predictions on test images
    
    Args:
        model: Trained YOLO model or path to model
        split: Dataset split to use
        num_samples: Number of samples to visualize
        conf_threshold: Confidence threshold for predictions
        save_path: Path to save visualization (if None, saves to model's run directory)
    """
    from ultralytics import YOLO
    
    if isinstance(model, (str, Path)) or model is None:
        if model is None:
            model_path = get_latest_model_path('best')
        else:
            model_path = model
        model = YOLO(model_path)
    else:
        # If model is already loaded, try to get path from it
        model_path = getattr(model, 'ckpt_path', None) or get_latest_model_path('best')
    
    # Determine output path if not specified
    if save_path is None:
        model_path = Path(model_path)
        # Get the run directory (e.g., output/train5)
        run_dir = model_path.parent.parent
        save_path = run_dir / f'{split}_predictions.png'
    
    print(f"\nVisualizing predictions on {num_samples} {split} images...")
    
    test_images = list((OUTPUT_DIR / split / 'images').glob('*'))[:num_samples]
    
    rows = (num_samples + 3) // 4
    cols = min(4, num_samples)
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.ravel()
    
    colors = VIZ_CONFIG['colors']
    
    for idx, img_path in enumerate(test_images):
        # Run prediction (specify project to avoid creating 'runs' folder)
        # Using a temporary project/name that we can ignore or clean up, or directing to output
        result = model(str(img_path), conf=conf_threshold, verbose=False, save=False, project='output/visualizations', name='temp')[0]
        
        # Load and convert image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw predictions
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            color = colors[class_id]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            label = f"{CLASSES[class_id]} {confidence:.2f}"
            cv2.putText(img, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        axes[idx].imshow(img)
        axes[idx].set_title(f"Test {idx+1}", fontsize=10)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved predictions: {save_path}")
    plt.close()


def count_objects(labels_dir):
    """
    Count objects per class in a dataset split
    
    Args:
        labels_dir: Path to labels directory
        
    Returns:
        Counter object with class counts
    """
    counts = Counter()
    for label_file in Path(labels_dir).glob('*.txt'):
        for line in label_file.read_text().split('\n'):
            if line.strip():
                class_id = int(line.split()[0])
                counts[class_id] += 1
    return counts


def print_dataset_statistics():
    """Print statistics about the dataset"""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    for split in ['train', 'valid', 'test']:
        img_count = len(list((OUTPUT_DIR / split / 'images').glob('*')))
        print(f"\n{split.upper()} SET:")
        print(f"  Images: {img_count}")
        
        if img_count > 0:
            counts = count_objects(OUTPUT_DIR / split / 'labels')
            print("  Objects per class:")
            for class_id, count in sorted(counts.items()):
                print(f"    {CLASSES[class_id]:10s}: {count}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Print statistics
    print_dataset_statistics()
    
    # Visualize training samples
    visualize_dataset_samples(split='train', num_samples=4)