"""
Model evaluation and testing module
Tests trained model and generates performance reports
"""
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

from config import DATA_YAML, TEST_CONFIG, CLASSES, get_latest_model_path, PROJECT_DIR

def evaluate_model(model_path=None):
    """
    Evaluate trained model on test set
    
    Args:
        model_path: Path to trained model weights
        
    Returns:
        Dictionary containing test results and metrics
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Resolve model path if not provided
    if model_path is None:
        model_path = get_latest_model_path('best')
    
    # Load best model
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)
    
    # Run evaluation on test set
    # Note: Using conf=0.001 for standard mAP calculation (standard practice)
    print(f"\nEvaluating performance on test set...")
    test_results = model.val(
        data=DATA_YAML,
        split=TEST_CONFIG['split'],
        conf=0.001,  # Standard low threshold for PR curve
        verbose=False,
        project=PROJECT_DIR,
        name='val',
        exist_ok=True,
        workers=0
    )
    
    # Extract metrics
    metrics = {
        'precision': test_results.results_dict['metrics/precision(B)'],
        'recall': test_results.results_dict['metrics/recall(B)'],
        'map50': test_results.results_dict['metrics/mAP50(B)'],
        'map50_95': test_results.results_dict['metrics/mAP50-95(B)'],
    }
    
    # Print results
    print("\n" + "-"*60)
    print("TEST SET PERFORMANCE")
    print("-"*60)
    print(f"Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:      {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"mAP@50:      {metrics['map50']:.4f} ({metrics['map50']*100:.2f}%)")
    print(f"mAP@50-95:   {metrics['map50_95']:.4f} ({metrics['map50_95']*100:.2f}%)")
    
    # Per-class performance (using AP@50)
    print("\n" + "-"*60)
    print("PER-CLASS PERFORMANCE (AP@50)")
    print("-"*60)
    for i, cls in enumerate(CLASSES):
        ap50 = test_results.box.ap50[i]
        print(f"{cls:10s}: {ap50:.4f} ({ap50*100:.2f}%)")
    
    # Primary Classes Average (Helmet & Head)
    primary_ap50 = (test_results.box.ap50[0] + test_results.box.ap50[1]) / 2
    print(f"\nPrimary Classes (Helmet & Head) Avg AP@50: {primary_ap50:.4f} ({primary_ap50*100:.2f}%)")
    print("-"*60)
    
    return {
        'metrics': metrics,
        'test_results': test_results,
        'model': model,
        'model_path': model_path
    }

def generate_performance_report(results, output_file='performance_report.txt'):
    """
    Generate detailed performance report
    
    Args:
        results: Dictionary from evaluate_model()
        output_file: Output file path
    """
    from config import OUTPUT_DIR
    
    metrics = results['metrics']
    test_results = results['test_results']
    
    # Count images in each split
    train_count = len(list((OUTPUT_DIR / 'train' / 'images').glob('*')))
    valid_count = len(list((OUTPUT_DIR / 'valid' / 'images').glob('*')))
    test_count = len(list((OUTPUT_DIR / 'test' / 'images').glob('*')))
    
    report = f"""PPE DETECTION MODEL - PERFORMANCE REPORT
{'='*70}

MODEL CONFIGURATION:
- Architecture: YOLOv8l (Large)
- Image Size: 640x640
- Confidence Threshold: {TEST_CONFIG['conf']}

DATASET:
- Train: {train_count} images
- Valid: {valid_count} images
- Test: {test_count} images
- Classes: {', '.join(CLASSES)}

TEST SET METRICS:
- Precision:  {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)
- Recall:     {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)
- mAP@50:     {metrics['map50']:.4f} ({metrics['map50']*100:.2f}%)
- mAP@50-95:  {metrics['map50_95']:.4f} ({metrics['map50_95']*100:.2f}%)

CLASS-WISE PERFORMANCE (AP@50):
- Helmet:     {test_results.box.ap50[0]:.4f} ({test_results.box.ap50[0]*100:.2f}%)
- Head:       {test_results.box.ap50[1]:.4f} ({test_results.box.ap50[1]*100:.2f}%)
- Person:     {test_results.box.ap50[2]:.4f} ({test_results.box.ap50[2]*100:.2f}%)

PRIMARY CLASSES SUMMARY (Helmet & Head):
- Average AP@50: {((test_results.box.ap50[0] + test_results.box.ap50[1]) / 2):.4f} ({((test_results.box.ap50[0] + test_results.box.ap50[1]) / 2)*100:.2f}%)

MODEL LOCATION: {results['model_path']}
{'='*70}
"""
    Path(output_file).write_text(report)
    print(f"\nPerformance report saved: {output_file}")
    
    return report

def compare_with_benchmarks(results, output_file='comparison_report.txt'):
    """
    Compare model performance with benchmark models
    
    Args:
        results: Dictionary from evaluate_model()
        output_file: Output file path
    """
    map50 = results['metrics']['map50']
    
    # Benchmark comparison data (Architecture stats only)
    comparison_data = {
        'Model': [
            'Faster R-CNN ResNet50',
            'RetinaNet ResNet50',
            'EfficientDet-D0',
            'DETR ResNet50',
            'Your Model (YOLOv8l)'
        ],
        'Year': [2017, 2018, 2020, 2020, 2024],
        'Parameters (M)': [41.8, 36.3, 3.9, 41.3, 43.7],
        'Size (MB)': [167, 145, 15.6, 165, 87.6]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Generate report
    report = f"""MODEL COMPARISON REPORT
{'='*70}

YOUR MODEL:
- Architecture: YOLOv8l
- mAP50: {map50*100:.2f}%
- Parameters: 43.7M
- Model Size: 87.6 MB

BENCHMARK COMPARISON:
{df.to_string(index=False)}
{'='*70}
"""
    
    Path(output_file).write_text(report)
    print(f"Comparison report saved: {output_file}")
    
    return report

def test_and_report(model_path=None):
    """
    Main function to test model and generate all reports
    
    Args:
        model_path: Path to trained model weights
    """
    # Evaluate model
    results = evaluate_model(model_path)
    
    # Generate reports
    generate_performance_report(results)
    compare_with_benchmarks(results)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print("Reports generated:")
    print("  - performance_report.txt")
    print("  - comparison_report.txt")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_and_report()