# ONNX Conversion Guide for Cough Detector Model

## Problem Summary
Your system has a **torchvision import issue** causing segmentation faults. This is a known Windows compatibility problem with certain torchvision builds.

## Solution Options

### Option 1: Use the Fixed Conversion Script (Recommended)
I've created `convert_to_onnx_fixed.py` which implements ResNet18 manually without importing from torchvision.

**To run:**
1. Double-click `convert_model.bat` in Windows Explorer
2. Or open Command Prompt (not Git Bash) and run:
   ```cmd
   cd "D:\coughcare_waig_3\coughcare_waig 3"
   python convert_to_onnx_fixed.py
   ```

### Option 2: Fix Torchvision Installation
Try reinstalling PyTorch and torchvision:

```cmd
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Then use the original `convert_to_onnx.py` script.

### Option 3: Use Google Colab (If local doesn't work)
If local execution continues to fail, you can use Google Colab:

1. Upload these files to Google Drive:
   - `backup_best_model_20251015_170801.pth`
   - `convert_to_onnx_fixed.py`

2. Run in Colab:
   ```python
   !pip install onnx onnxruntime onnx-simplifier
   # Mount Google Drive
   from google.colab import drive
   drive.mount('/content/drive')
   # Run conversion
   !python "/content/drive/MyDrive/convert_to_onnx_fixed.py"
   ```

## Expected Output

When conversion succeeds, you should see:
```
Loading checkpoint from: ...
Loaded checkpoint from epoch 3
Validation AUC: 0.9904
[OK] Weights loaded successfully
Model loaded successfully. Total parameters: 11,xxx,xxx

Exporting to ONNX...
[OK] ONNX model saved to: cough_detector_attention.onnx

File sizes:
  - Original PyTorch (.pth): 132.00 MB
  - ONNX (.onnx): ~44 MB
  - Compression ratio: 3.0x
```

## What the Conversion Does

1. **Loads your trained model** from `backup_best_model_20251015_170801.pth`
2. **Converts to ONNX format** with:
   - Dynamic batch size and segment count
   - Opset version 14 (widely compatible)
   - Constant folding optimizations
3. **Creates outputs:**
   - `cough_detector_attention.onnx` - Main model (recommended)
   - `cough_detector_attention_simplified.onnx` - Further optimized (if onnx-simplifier is installed)

## Using the ONNX Model

### With ONNX Runtime (Python)
```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession("cough_detector_attention.onnx")

# Prepare inputs
spectrograms = np.random.randn(1, 32, 3, 224, 224).astype(np.float32)
segment_mask = np.ones((1, 32), dtype=bool)

# Run inference
outputs = session.run(
    None,
    {
        'spectrograms': spectrograms,
        'segment_mask': segment_mask
    }
)

bag_probability = outputs[0][0]  # Prediction for the audio file
print(f"Cough probability: {bag_probability:.4f}")
```

### With ONNX Runtime (C++/JavaScript/Java)
The ONNX format allows you to deploy on:
- Mobile devices (iOS/Android with ONNX Runtime Mobile)
- Web browsers (ONNX.js)
- Edge devices (TensorRT, OpenVINO)
- Any platform with ONNX Runtime support

## Troubleshooting

### Issue: Segmentation Fault
- **Cause**: Bash/Python interaction issue or torchvision compatibility
- **Solution**: Run directly from Command Prompt (cmd), not Git Bash

### Issue: "weights_only" parameter error
- **Cause**: Older PyTorch version
- **Solution**: The script handles this automatically

### Issue: Missing onnx or onnxruntime
```cmd
pip install onnx onnxruntime onnx-simplifier
```

## Files Created

- `convert_to_onnx_fixed.py` - Main conversion script (no torchvision dependency)
- `convert_model.bat` - Windows batch file to run conversion
- `test_load_checkpoint.py` - Test if checkpoint can be loaded
- `test_minimal.py` - Test basic PyTorch/ONNX functionality

## Next Steps After Conversion

1. **Test the ONNX model** with sample audio files
2. **Compare outputs** between PyTorch and ONNX versions
3. **Optimize further** if needed (quantization, pruning)
4. **Deploy** to your target platform

## Model Information

- **Architecture**: AttnMILResNet (Attention-based Multiple Instance Learning with ResNet18)
- **Input**: Spectrograms (batch, segments, 3, 224, 224) + Segment mask (batch, segments)
- **Output**:
  - Bag probability (file-level prediction)
  - Segment probabilities (segment-level predictions)
  - Logits for both
- **Parameters**: ~11M
- **Training Performance**: 99.04% AUC on validation set

## Contact/Support

If you continue to have issues:
1. Try running from Windows Command Prompt instead of Git Bash
2. Check Python and PyTorch versions
3. Try the Google Colab approach
4. Consider using Windows PowerShell instead of Git Bash
