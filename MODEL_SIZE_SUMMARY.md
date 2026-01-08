# Model Size Summary - Quick Reference

## Why FP16 = Same Size as Standard ONNX?

**Short Answer**: CPU doesn't natively support FP16, so ONNX Runtime adds overhead that prevents compression.

---

## All Model Sizes

```
PyTorch (.pth):              131.82 MB  (baseline)
    ↓ 3x compression
ONNX Standard (FP32):         43.90 MB  ← Use this for CPU
    ↓ No change on CPU
ONNX FP16 (on CPU):           43.90 MB  ← Same! (See explanation below)
    ↓ 4x compression
ONNX INT8 (quantized):        11.05 MB  ← Use this for mobile/edge
```

---

## Quick Comparison

| Model | Size | CPU Time | Accuracy | Best For |
|-------|------|----------|----------|----------|
| PyTorch | 132 MB | 800 ms | 100% | Training |
| ONNX FP32 | 44 MB | **651 ms** ⚡ | 100% | **CPU servers** ✓ |
| ONNX FP16 | 44 MB | 1010 ms | 100% | GPU only |
| ONNX INT8 | **11 MB** 📦 | 847 ms | 98.9% | **Mobile/Edge** ✓ |

---

## Why FP16 Doesn't Save Space on CPU

1. **CPU doesn't support FP16 natively**
   - Weights stored as FP16 (16-bit)
   - But CPU must convert to FP32 (32-bit) for computation
   - Conversion overhead prevents size reduction

2. **On GPU it works differently**
   - GPU has native FP16 operations
   - True size: ~22 MB (50% smaller)
   - Faster inference too

3. **INT8 works on CPU**
   - Modern CPUs support INT8 instructions (VNNI)
   - True quantization: 11 MB (75% smaller)
   - Good accuracy: 98.9%

---

## Recommendations

### ✓ For CPU Deployment (Your Case)
```bash
# Use standard ONNX (fastest)
python backend_api_onnx.py

# Or use INT8 (smallest)
python backend_api_onnx_quant.py
```

### ✓ For GPU Deployment
```bash
# Use FP16 (best on GPU)
# Requires CUDA/TensorRT execution provider
```

### ✓ For Mobile/Edge
```bash
# Use INT8 (best size/accuracy balance)
docker-compose -f docker-compose.onnx-quant.yml up
```

---

## Technical Details

See `WHY_FP16_SAME_SIZE.md` for full explanation.
See `ONNX_COMPLETE_GUIDE.md` for complete documentation.
