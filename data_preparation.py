"""
Dataset preparation utilities
Handles data download, conversion, and splitting
"""


import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
import cv2

from config import INPUT_DIR, OUTPUT_DIR, CLASSES, TRAIN_SPLIT, TEST_SPLIT

def download_dataset():
    """Download dataset from Kaggle using kagglehub"""
    import kagglehub
    print("Downloading dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download('andrewmvd/hard-hat-detection')
    print(f"Dataset downloaded to: {dataset_path}")
    return Path(dataset_path)


def setup_directories():
    """Create directory structure for train/valid/test splits"""
    print(f"Setting up directory structure at {OUTPUT_DIR}...")
    
    # Clean old dataset to avoid mixed class labels
    if OUTPUT_DIR.exists():
        print(f"Cleaning existing directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
        
    for split in ['train', 'valid', 'test']:
        (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
    print("Directories created")


def copy_raw_data(source_path):
    """Copy raw data from download location to working directory"""
    print("Copying raw data to working directory...")
    
    # Check if already copied
    if (INPUT_DIR / 'images').exists() and (INPUT_DIR / 'annotations').exists():
        print("Raw data already exists, skipping copy")
        return
    
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Copy images and annotations
    for subdir in ['images', 'annotations']:
        src = source_path / subdir
        dst = INPUT_DIR / subdir
        
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"Copied {subdir}")


def xml_to_yolo(xml_path, img_width, img_height):
    """Convert XML annotation to YOLO format"""
    tree = ET.parse(xml_path)
    labels = []
    
    for obj in tree.findall('.//object'):
        cls = obj.find('name').text
        if cls not in CLASSES:
            continue
            
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # Convert to YOLO format (normalized center x, center y, width, height)
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        class_id = CLASSES.index(cls)
        labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return labels

def split_dataset():
    """Split dataset into train/valid/test sets"""
    print("\nSplitting dataset...")
    
    images = sorted(list((INPUT_DIR / 'images').glob('*.png')))
    print(f"Total images found: {len(images)}")
    
    # Split: 70% train, 15% valid, 15% test
    train_imgs, temp = train_test_split(images, test_size=(1 - TRAIN_SPLIT), random_state=42)
    valid_imgs, test_imgs = train_test_split(temp, test_size=0.5, random_state=42)
    
    splits = {
        'train': train_imgs,
        'valid': valid_imgs,
        'test': test_imgs
    }
    
    print(f"Train: {len(train_imgs)} images")
    print(f"Valid: {len(valid_imgs)} images")
    print(f"Test: {len(test_imgs)} images")
    
    return splits

def process_and_copy_files(splits):
    """Process images and annotations, convert to YOLO format, and copy to output directories"""
    print("\nProcessing and copying files...")
    
    stats = {'train': 0, 'valid': 0, 'test': 0}
    
    for split_name, image_paths in splits.items():
        print(f"\nProcessing {split_name} set...")
        
        for img_path in image_paths:
            xml_path = INPUT_DIR / 'annotations' / f"{img_path.stem}.xml"
            
            # Skip if annotation doesn't exist
            if not xml_path.exists():
                continue
            
            # Read image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            height, width = img.shape[:2]
            
            # Convert XML to YOLO format
            labels = xml_to_yolo(xml_path, width, height)
            
            # Skip if no valid labels
            if not labels:
                continue
            
            # Copy image
            shutil.copy(img_path, OUTPUT_DIR / split_name / 'images' / img_path.name)
            
            # Save labels
            label_file = OUTPUT_DIR / split_name / 'labels' / f"{img_path.stem}.txt"
            label_file.write_text('\n'.join(labels))
            
            stats[split_name] += 1
        
        print(f"Processed {stats[split_name]} images for {split_name}")
    
    return stats

def create_yaml_config():
    """Create YAML configuration file for YOLO training"""
    from config import DATA_YAML
    
    names_content = "\n".join([f"  {i}: {name}" for i, name in enumerate(CLASSES)])
    
    yaml_content = f"""path: {OUTPUT_DIR.absolute()}
train: train/images
val: valid/images
test: test/images

names:
{names_content}
"""
    
    Path(DATA_YAML).write_text(yaml_content)
    print(f"\nCreated config file: {DATA_YAML}")

def prepare_dataset():
    """Main function to prepare the entire dataset"""
    print("="*60)
    print("DATASET PREPARATION")
    print("="*60)
    
    # Download dataset
    source_path = download_dataset()
    
    # Setup directories
    setup_directories()
    
    # Copy raw data
    copy_raw_data(source_path)
    
    # Split dataset
    splits = split_dataset()
    
    # Process and copy files
    stats = process_and_copy_files(splits)
    
    # Create YAML config
    create_yaml_config()
    
    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETE")
    print("="*60)
    print(f"Dataset location: {OUTPUT_DIR.absolute()}")
    print(f"Train: {stats['train']} images")
    print(f"Valid: {stats['valid']} images")
    print(f"Test: {stats['test']} images")
    print("="*60 + "\n")

if __name__ == "__main__":
    prepare_dataset()