"""
Model export and optimization utilities
Export trained model to different formats (ONNX, TorchScript)
"""
import time
import torch
from pathlib import Path
from ultralytics import YOLO

from config import OUTPUT_DIR, get_latest_model_path, TRAINING_CONFIG


def export_model(model_path=None, force=False):
    """
    Export model to multiple high-performance formats.
    Skips existing files unless force=True.
    """
    print("\n" + "="*60)
    print("MODEL EXPORT & OPTIMIZATION")
    print("="*60)
    
    if model_path is None:
        model_path = get_latest_model_path('best')
    
    weights_dir = Path(model_path).parent
    results = {}
    
    # Define expected paths
    expected_paths = {
        'engine': weights_dir / 'best.engine',
        'onnx': weights_dir / 'best.onnx',
        'torchscript': weights_dir / 'best.torchscript'
    }

    # Force GPU usage
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Target device: {device}")
    
    # Check for TensorRT installation
    trt_available = False
    try:
        import tensorrt
        trt_available = True
        print("TensorRT library detected.")
    except ImportError:
        print("TensorRT not detected. Skipping .engine export.")

    model = None

    def get_model():
        nonlocal model
        if model is None:
            model = YOLO(model_path).to(device)
        return model

    # 1. Export to TensorRT (FP16)
    if trt_available:
        if not force and expected_paths['engine'].exists():
            print(f"Skipping TensorRT export: {expected_paths['engine']} already exists.")
            results['engine'] = str(expected_paths['engine'])
        else:
            print("\nExporting to TensorRT (FP16)...")
            try:
                engine_path = get_model().export(
                    format='engine', 
                    dynamic=True, 
                    half=True, 
                    device=0, 
                    batch=TRAINING_CONFIG['batch']
                )
                results['engine'] = engine_path
            except Exception as e:
                print(f"Warning: TensorRT export failed: {e}")

    # 2. Export to ONNX (FP16)
    if not force and expected_paths['onnx'].exists():
        print(f"Skipping ONNX export: {expected_paths['onnx']} already exists.")
        results['onnx'] = str(expected_paths['onnx'])
    else:
        print("\nExporting to ONNX (FP16)...")
        try:
            onnx_path = get_model().export(format='onnx', dynamic=True, half=True, device=device)
            results['onnx'] = onnx_path
        except Exception as e:
            print(f"Warning: ONNX export failed: {e}")
    
    # 3. Export to TorchScript
    if not force and expected_paths['torchscript'].exists():
        print(f"Skipping TorchScript export: {expected_paths['torchscript']} already exists.")
        results['torchscript'] = str(expected_paths['torchscript'])
    else:
        print("\nExporting to TorchScript format...")
        try:
            torchscript_path = get_model().export(format='torchscript')
            results['torchscript'] = torchscript_path
        except Exception as e:
            print(f"Warning: TorchScript export failed: {e}")
    
    return results

def benchmark_inference_speed(model_path=None, onnx_path=None, engine_path=None, num_iterations=100):
    """
    Benchmark inference speed using GPU
    """
    print("\n" + "="*60)
    print("GPU INFERENCE SPEED BENCHMARK")
    print("="*60)
    
    # Ensure device is set
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Benchmarking on device: {device}")

    test_images = list((OUTPUT_DIR / 'test' / 'images').glob('*'))
    if not test_images:
        print("No test images found!")
        return None
    
    test_img = str(test_images[0])
    
    if model_path is None:
        model_path = get_latest_model_path('best')
    
    results = {}

    def run_benchmark(m, name):
        print(f"Benchmarking {name}...")
        # Warmup
        for _ in range(20):
            _ = m(test_img, verbose=False, save=False)
        
        # Synchronize for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        start = time.time()
        for _ in range(num_iterations):
            _ = m(test_img, verbose=False, save=False)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        avg_time = (time.time() - start) / num_iterations
        return avg_time * 1000, 1.0 / avg_time

    # Benchmark PyTorch (Base)
    pt_model = YOLO(model_path).to(device)
    results['pytorch_ms'], results['pytorch_fps'] = run_benchmark(pt_model, "PyTorch (FP32)")

    # Benchmark ONNX (Half)
    if onnx_path and Path(onnx_path).exists():
        try:
            onnx_model = YOLO(onnx_path)
            results['onnx_ms'], results['onnx_fps'] = run_benchmark(onnx_model, "ONNX (FP16)")
        except Exception as e:
            print(f"ONNX benchmark failed: {e}")

    # Benchmark TensorRT (Engine)
    if engine_path and Path(engine_path).exists():
        try:
            trt_model = YOLO(engine_path)
            results['engine_ms'], results['engine_fps'] = run_benchmark(trt_model, "TensorRT (FP16)")
        except Exception as e:
            print(f"TensorRT benchmark failed: {e}")
    
    # Print results
    print("\n" + "-"*60)
    print("BENCHMARK RESULTS")
    print("-"*60)
    print(f"PyTorch Model:")
    print(f"  Time per image: {results['pytorch_ms']:.2f} ms")
    print(f"  FPS: {results['pytorch_fps']:.1f}")
    
    if 'onnx_ms' in results:
        print(f"\nONNX Model:")
        print(f"  Time per image: {results['onnx_ms']:.2f} ms")
        print(f"  FPS: {results['onnx_fps']:.1f}")
    
    if 'engine_ms' in results:
        print(f"\nTensorRT Model:")
        print(f"  Time per image: {results['engine_ms']:.2f} ms")
        print(f"  FPS: {results['engine_fps']:.1f}")
    
    print("-"*60)
    
    return results

def generate_optimization_report(export_paths, benchmark_results, 
                                 output_file='optimization_report.txt'):
    """
    Generate detailed optimization report
    """
    # Build list of exported formats, excluding those not present
    exported_formats = []
    if export_paths.get('engine'):
        exported_formats.append(f"- TensorRT (FP16): {export_paths['engine']}")
    if export_paths.get('onnx'):
        exported_formats.append(f"- ONNX (FP16): {export_paths['onnx']}")
    if export_paths.get('torchscript'):
        exported_formats.append(f"- TorchScript: {export_paths['torchscript']}")

    # Build list of inference speeds, excluding N/A
    inference_speeds = []
    def format_line(name, results):
        if f'{name}_ms' in results:
            return f"- {name}: {results[f'{name}_ms']:.2f} ms/image ({results[f'{name}_fps']:.1f} FPS)"
        return None

    for name in ['pytorch', 'onnx', 'engine']:
        line = format_line(name, benchmark_results)
        if line:
            inference_speeds.append(line)

    # Construct the report
    report_lines = [
        "MODEL OPTIMIZATION REPORT",
        "="*70,
        "",
        "EXPORTED FORMATS:",
    ] + (exported_formats if exported_formats else ["- None"]) + [
        "",
        "INFERENCE SPEED (GPU):",
    ] + (inference_speeds if inference_speeds else ["- N/A"]) + [
        "",
        "="*70,
        ""
    ]
    
    report = "\n".join(report_lines)
    
    Path(output_file).write_text(report)
    print(f"\nOptimization report saved: {output_file}")


def export_and_optimize(model_path=None):
    """
    Main function to export and optimize model
    
    Args:
        model_path: Path to trained model weights
    """
    # Export model
    export_paths = export_model(model_path)
    
    # Benchmark
    benchmark_results = benchmark_inference_speed(
        model_path=model_path,
        onnx_path=export_paths.get('onnx'),
        engine_path=export_paths.get('engine')
    )
    
    # Generate report
    if benchmark_results:
        generate_optimization_report(export_paths, benchmark_results)
    
    print("\n" + "="*60)
    print("EXPORT AND OPTIMIZATION COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    export_and_optimize()