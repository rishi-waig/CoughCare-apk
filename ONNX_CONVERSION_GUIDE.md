# ONNX Model Conversion Guide

This guide explains how to convert your PyTorch model checkpoint to ONNX format for compression and deployment.

## Overview

ONNX (Open Neural Network Exchange) is a format that allows you to:
- **Compress** your model (typically 2-5x smaller)
- **Deploy** on various platforms (mobile, edge devices, etc.)
- **Optimize** inference speed
- **Use** with different inference engines (ONNX Runtime, TensorRT, etc.)

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install onnx onnxruntime onnxsim
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Convert Model to ONNX

Basic conversion:

```bash
python convert_to_onnx.py --checkpoint backup_best_model_20251015_170801.pth --output cough_model.onnx
```

With custom settings:

```bash
python convert_to_onnx.py \
    --checkpoint backup_best_model_20251015_170801.pth \
    --output cough_model.onnx \
    --max-segments 32 \
    --image-size 224 \
    --opset 14
```

### 2. Test ONNX Model

Verify the conversion was successful:

```bash
python test_onnx_model.py --onnx cough_model.onnx
```

## Command Line Options

### convert_to_onnx.py

- `--checkpoint`: Path to PyTorch checkpoint file (default: `backup_best_model_20251015_170801.pth`)
- `--output`: Output path for ONNX model (default: `cough_model.onnx`)
- `--max-segments`: Maximum number of segments (default: 32)
- `--image-size`: Image size HxW (default: 224)
- `--opset`: ONNX opset version (default: 14)
- `--no-dynamic`: Disable dynamic axes (fixed batch/segment sizes)
- `--no-optimize`: Skip ONNX optimization

### test_onnx_model.py

- `--onnx`: Path to ONNX model file (default: `cough_model.onnx`)
- `--max-segments`: Maximum number of segments (default: 32)
- `--image-size`: Image size (default: 224)

## Model Inputs/Outputs

### Inputs:
1. **spectrograms**: `(batch_size, num_segments, 3, 224, 224)` - Float32 tensor
   - Batch of mel spectrogram segments
   - Already normalized with ImageNet stats
2. **segment_mask**: `(batch_size, num_segments)` - Boolean tensor
   - Mask indicating valid segments (True = valid, False = padding)

### Outputs:
1. **bag_prob**: `(batch_size,)` - Float32 tensor
   - Bag-level probability (sigmoid applied)
2. **segment_probs**: `(batch_size, num_segments)` - Float32 tensor
   - Per-segment probabilities
3. **segment_logits**: `(batch_size, num_segments)` - Float32 tensor
   - Per-segment logits (before sigmoid)
4. **bag_logit**: `(batch_size,)` - Float32 tensor
   - Bag-level logit (before sigmoid)

## Using ONNX Model in Production

### Python Example

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession('cough_model.onnx', providers=['CPUExecutionProvider'])

# Prepare inputs (assuming you have preprocessed audio)
spectrograms = np.random.randn(1, 10, 3, 224, 224).astype(np.float32)
segment_mask = np.ones((1, 10), dtype=np.bool_)

# Run inference
outputs = session.run(None, {
    'spectrograms': spectrograms,
    'segment_mask': segment_mask
})

bag_prob = outputs[0][0, 0]
print(f"Cough probability: {bag_prob:.4f}")
```

### Integration with Backend API

You can modify `backend_api_actual_model.py` to use ONNX Runtime instead of PyTorch:

```python
import onnxruntime as ort

class ONNXCoughModel:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        # ... rest of initialization
    
    def predict(self, audio_bytes, saved_path=None):
        # Preprocess audio (same as before)
        batch, mask = self.preprocess_audio(audio_bytes, saved_path)
        
        # Convert to numpy
        batch_np = batch.cpu().numpy()
        mask_np = mask.cpu().numpy()
        
        # Run ONNX inference
        outputs = self.session.run(None, {
            'spectrograms': batch_np,
            'segment_mask': mask_np
        })
        
        bag_prob = outputs[0][0, 0]
        # ... rest of processing
```

## File Size Comparison

Typical compression results:
- **PyTorch checkpoint**: ~50-100 MB
- **ONNX model**: ~20-40 MB (2-3x compression)
- **Optimized ONNX**: ~15-30 MB (3-5x compression)

## Troubleshooting

### Error: "ONNX export failed"
- Make sure all model operations are ONNX-compatible
- Try a different opset version (e.g., `--opset 13` or `--opset 15`)

### Error: "onnxsim not found"
- Install with: `pip install onnxsim`
- Or use `--no-optimize` to skip optimization

### Dynamic axes issues
- If you get shape errors, try `--no-dynamic` for fixed-size inputs
- Note: This limits flexibility but may improve compatibility

### Inference speed
- ONNX Runtime is typically faster than PyTorch for inference
- For even better performance, consider:
  - TensorRT (NVIDIA GPUs)
  - OpenVINO (Intel CPUs)
  - CoreML (Apple devices)

## Notes

- The ONNX model maintains the same accuracy as the PyTorch model
- Dynamic axes allow variable batch sizes and segment counts
- The optimized version removes redundant operations for smaller file size
- ONNX Runtime supports CPU, GPU, and specialized accelerators


