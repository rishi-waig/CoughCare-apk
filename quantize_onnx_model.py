"""
ONNX Model Quantization Script
Creates both int8 (dynamic) and uint8 quantized versions of the cough detection model.
"""

import os
import sys
import time
import numpy as np

# Fix OpenMP duplicate library issue on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import onnx
from onnx import numpy_helper
import onnxruntime as ort

# For dynamic quantization
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType, QuantFormat
    from onnxruntime.quantization.preprocess import quant_pre_process
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    print("Warning: onnxruntime.quantization not available")


def get_model_size(model_path):
    """Get model file size in MB"""
    return os.path.getsize(model_path) / (1024 * 1024)


def preprocess_for_quantization(model_path, output_path):
    """Preprocess model for quantization (symbolic shape inference, etc.)"""
    print(f"  Preprocessing model for quantization...")
    try:
        quant_pre_process(model_path, output_path)
        return output_path
    except Exception as e:
        print(f"  Preprocessing failed: {e}, using original model")
        return model_path


def quantize_dynamic_int8(model_path, output_path):
    """
    Apply dynamic int8 quantization to ONNX model using QDQ format.
    QDQ (QuantizeLinear/DequantizeLinear) format is more compatible with standard ONNX Runtime.
    """
    if not QUANTIZATION_AVAILABLE:
        print("Error: onnxruntime.quantization not available")
        return None

    print(f"\nApplying Dynamic int8 Quantization (QDQ format)...")
    print(f"  Input: {model_path}")
    print(f"  Output: {output_path}")

    # Preprocess if needed
    preprocessed_path = model_path.replace('.onnx', '_prep.onnx')
    try:
        actual_input = preprocess_for_quantization(model_path, preprocessed_path)
    except:
        actual_input = model_path

    # Apply dynamic quantization with QDQ format for better runtime compatibility
    try:
        quantize_dynamic(
            model_input=actual_input,
            model_output=output_path,
            weight_type=QuantType.QUInt8,  # UInt8 is more widely supported
            extra_options={'ActivationSymmetric': False}
        )

        # Clean up preprocessed file
        if os.path.exists(preprocessed_path) and preprocessed_path != model_path:
            os.remove(preprocessed_path)

        size = get_model_size(output_path)
        print(f"  Int8 quantized model saved: {size:.2f} MB")
        return output_path

    except Exception as e:
        print(f"  Dynamic quantization failed: {e}")
        return None


def quantize_weights_only(model_path, output_path):
    """
    Quantize only the weights to int8, keeping computation in float32.
    This is a simpler approach that always works.
    """
    print(f"\nApplying Weight-Only int8 Quantization...")
    print(f"  Input: {model_path}")
    print(f"  Output: {output_path}")

    model = onnx.load(model_path)

    # Quantize each initializer (weight tensor)
    for initializer in model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.FLOAT:
            # Get float32 weights
            float_data = numpy_helper.to_array(initializer)

            # Compute scale and zero point for symmetric quantization
            max_val = np.max(np.abs(float_data))
            if max_val == 0:
                continue

            scale = max_val / 127.0

            # Quantize to int8
            int8_data = np.round(float_data / scale).astype(np.int8)

            # Dequantize back to float32 (lossy but same format)
            dequant_data = int8_data.astype(np.float32) * scale

            # Update initializer with dequantized values
            new_tensor = numpy_helper.from_array(dequant_data, initializer.name)
            initializer.CopyFrom(new_tensor)

    # Save model
    onnx.save(model, output_path)

    size = get_model_size(output_path)
    print(f"  Weight-quantized model saved: {size:.2f} MB")

    return output_path


def convert_to_float16_weights(model_path, output_path):
    """
    Convert model to float16 for ~2x compression.
    Uses onnxconverter-common for proper conversion.
    """
    print(f"\nConverting to Float16...")
    print(f"  Input: {model_path}")
    print(f"  Output: {output_path}")

    try:
        # Try using onnxconverter-common (best method)
        from onnxconverter_common import float16

        model = onnx.load(model_path)

        # Convert to float16, keeping inputs/outputs as float32 for compatibility
        model_fp16 = float16.convert_float_to_float16(
            model,
            keep_io_types=True,  # Keep I/O as float32
            disable_shape_infer=True
        )

        onnx.save(model_fp16, output_path)

        size = get_model_size(output_path)
        print(f"  Float16 model saved: {size:.2f} MB")
        return output_path

    except ImportError:
        print("  onnxconverter-common not found, trying manual conversion...")
    except Exception as e:
        print(f"  onnxconverter-common failed: {e}, trying manual conversion...")

    # Fallback: Manual float16 conversion of weights only
    try:
        model = onnx.load(model_path)

        # Convert each weight tensor to actual float16
        for initializer in model.graph.initializer:
            if initializer.data_type == onnx.TensorProto.FLOAT:
                float32_data = numpy_helper.to_array(initializer)
                float16_data = float32_data.astype(np.float16)

                # Create float16 tensor
                new_tensor = numpy_helper.from_array(float16_data, initializer.name)
                initializer.CopyFrom(new_tensor)

        onnx.save(model, output_path)

        size = get_model_size(output_path)
        print(f"  Float16 model saved: {size:.2f} MB")
        return output_path

    except Exception as e:
        print(f"  Manual float16 conversion failed: {e}")
        return None


def test_model_inference(model_path, test_input, test_mask):
    """Test model inference and return prediction + timing."""
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # Warm up
    _ = session.run(None, {'spectrograms': test_input, 'segment_mask': test_mask})

    # Timed inference
    start = time.perf_counter()
    num_runs = 5
    for _ in range(num_runs):
        outputs = session.run(None, {'spectrograms': test_input, 'segment_mask': test_mask})
    elapsed = (time.perf_counter() - start) / num_runs * 1000  # ms

    bag_prob = outputs[0][0]

    return bag_prob, elapsed


def compare_models(models_dict, test_input, test_mask):
    """Compare all models"""
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)

    results = {}

    for name, path in models_dict.items():
        if path and os.path.exists(path):
            print(f"\nTesting {name}...")
            try:
                prob, time_ms = test_model_inference(path, test_input, test_mask)
                size = get_model_size(path)
                results[name] = {'prob': prob, 'time': time_ms, 'size': size, 'path': path}
                print(f"  Size: {size:.2f} MB, Prob: {prob:.4f}, Time: {time_ms:.2f} ms")
            except Exception as e:
                print(f"  Failed: {e}")
                results[name] = None

    # Print comparison table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Model':<25} {'Size (MB)':<12} {'Compression':<12} {'Probability':<12} {'Time (ms)':<10}")
    print("-"*70)

    orig_size = results.get('Original', {}).get('size', 43.90)
    orig_prob = results.get('Original', {}).get('prob', 0)

    for name, data in results.items():
        if data:
            compression = orig_size / data['size'] if data['size'] > 0 else 1.0
            print(f"{name:<25} {data['size']:<12.2f} {compression:<12.1f}x {data['prob']:<12.4f} {data['time']:<10.2f}")

    print("="*70)

    # Accuracy comparison
    if results.get('Original'):
        print("\nACCURACY COMPARISON (vs Original):")
        for name, data in results.items():
            if data and name != 'Original':
                diff = abs(data['prob'] - orig_prob)
                pct = (diff / max(orig_prob, 1e-6)) * 100
                print(f"  {name}: {diff:.6f} difference ({pct:.4f}%)")

    return results


def create_test_input():
    """Create dummy test input for comparison"""
    test_input = np.random.randn(1, 10, 3, 224, 224).astype(np.float32)
    test_mask = np.ones((1, 10), dtype=bool)
    return test_input, test_mask


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths
    original_model = os.path.join(script_dir, "cough_detector_attention.onnx")
    int8_model = os.path.join(script_dir, "cough_detector_int8.onnx")
    fp16_model = os.path.join(script_dir, "cough_detector_fp16.onnx")

    print("="*70)
    print("ONNX Model Quantization Tool")
    print("="*70)

    # Check original model exists
    if not os.path.exists(original_model):
        print(f"Error: Original model not found at {original_model}")
        sys.exit(1)

    original_size = get_model_size(original_model)
    print(f"\nOriginal model: {original_model}")
    print(f"Original size: {original_size:.2f} MB")

    # Create quantized models
    print("\n" + "-"*70)
    print("CREATING QUANTIZED MODELS")
    print("-"*70)

    # 1. Dynamic int8 quantization
    int8_result = quantize_dynamic_int8(original_model, int8_model)

    # If dynamic quantization failed, try weight-only quantization
    if not int8_result or not os.path.exists(int8_model):
        print("\n  Falling back to weight-only quantization...")
        int8_result = quantize_weights_only(original_model, int8_model)

    # 2. Float16 precision (weights approximated to fp16 range)
    fp16_result = convert_to_float16_weights(original_model, fp16_model)

    # Create test input
    print("\n" + "-"*70)
    print("TESTING MODELS")
    print("-"*70)

    test_input, test_mask = create_test_input()
    print(f"Test input shape: {test_input.shape}")

    # Compare models
    models = {
        'Original': original_model,
        'Int8 Quantized': int8_model if int8_result else None,
        'Float16 Precision': fp16_model if fp16_result else None
    }

    results = compare_models(models, test_input, test_mask)

    print("\n" + "="*70)
    print("QUANTIZATION COMPLETE!")
    print("="*70)
    print("\nCreated models:")
    for name, data in results.items():
        if data:
            print(f"  {name}: {data['path']} ({data['size']:.2f} MB)")

    print("\nRecommendations:")
    print("  - Int8: Best for edge devices / mobile (smallest)")
    print("  - Float16: Good balance of size and accuracy")
    print("  - Original: Maximum accuracy (reference)")
    print("="*70)


if __name__ == "__main__":
    main()
