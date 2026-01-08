"""
Simple ONNX conversion script with error handling.
"""
import sys
import os

print("=" * 70)
print("Starting ONNX Conversion")
print("=" * 70)

# Step 1: Check imports
print("\n1. Checking imports...")
try:
    import torch
    print("  [OK] torch imported")
except Exception as e:
    print(f"  [ERROR] Failed to import torch: {e}")
    sys.exit(1)

try:
    import torch.nn as nn
    print("  [OK] torch.nn imported")
except Exception as e:
    print(f"  [ERROR] Failed to import torch.nn: {e}")
    sys.exit(1)

# Step 2: Import model classes
print("\n2. Importing model classes...")
try:
    from train_cough_detector_attention import Config, AttnMILResNet, get_device
    print("  [OK] Model classes imported successfully")
except Exception as e:
    print(f"  [ERROR] Failed to import model classes: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Load checkpoint
checkpoint_path = "backup_best_model_20251015_170801.pth"
output_path = "cough_model.onnx"

print(f"\n3. Loading checkpoint: {checkpoint_path}")
if not os.path.exists(checkpoint_path):
    print(f"  [ERROR] Checkpoint not found: {checkpoint_path}")
    sys.exit(1)

try:
    config = Config()
    device = get_device("cpu")
    print(f"  [OK] Using device: {device}")
    
    model = AttnMILResNet(config).to(device)
    print(f"  [OK] Model created")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        val_auc = checkpoint.get('val_auc', 0.0)
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"  [OK] Model loaded successfully!")
        print(f"    - Validation AUC: {val_auc:.4f}")
        print(f"    - Trained epoch: {epoch}")
    else:
        print("  [ERROR] Invalid checkpoint format")
        sys.exit(1)
    
    model.eval()
    print(f"  [OK] Model set to eval mode")
    
except Exception as e:
    print(f"  [ERROR] Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Create dummy inputs
print("\n4. Creating dummy inputs...")
try:
    batch_size = 1
    max_segments = 32
    image_size = 224
    
    dummy_x = torch.randn(batch_size, max_segments, 3, image_size, image_size)
    dummy_mask = torch.ones(batch_size, max_segments, dtype=torch.bool)
    
    print(f"  [OK] Input shapes:")
    print(f"    - spectrograms: {dummy_x.shape}")
    print(f"    - segment_mask: {dummy_mask.shape}")
except Exception as e:
    print(f"  [ERROR] Failed to create inputs: {e}")
    sys.exit(1)

# Step 5: Test forward pass
print("\n5. Testing forward pass...")
try:
    with torch.no_grad():
        bag_prob, seg_probs, seg_logits, bag_logit = model(dummy_x, dummy_mask)
        print(f"  [OK] Forward pass successful!")
        print(f"    - bag_prob: {bag_prob.shape}")
        print(f"    - seg_probs: {seg_probs.shape}")
except Exception as e:
    print(f"  [ERROR] Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Export to ONNX
print(f"\n6. Exporting to ONNX: {output_path}")
try:
    input_names = ['spectrograms', 'segment_mask']
    output_names = ['bag_prob', 'segment_probs', 'segment_logits', 'bag_logit']
    
    dynamic_axes = {
        'spectrograms': {0: 'batch_size', 1: 'num_segments'},
        'segment_mask': {0: 'batch_size', 1: 'num_segments'},
        'bag_prob': {0: 'batch_size'},
        'segment_probs': {0: 'batch_size', 1: 'num_segments'},
        'segment_logits': {0: 'batch_size', 1: 'num_segments'},
        'bag_logit': {0: 'batch_size'}
    }
    
    torch.onnx.export(
        model,
        (dummy_x, dummy_mask),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )
    
    print(f"  [OK] ONNX export successful!")
    
    # Get file sizes
    checkpoint_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    onnx_size = os.path.getsize(output_path) / (1024 * 1024)
    compression_ratio = checkpoint_size / onnx_size
    
    print(f"\n7. File sizes:")
    print(f"  PyTorch checkpoint: {checkpoint_size:.2f} MB")
    print(f"  ONNX model: {onnx_size:.2f} MB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    
except Exception as e:
    print(f"  [ERROR] ONNX export failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 7: Optimize (optional)
print("\n8. Attempting optimization...")
try:
    import onnx
    from onnxsim import simplify
    
    onnx_model = onnx.load(output_path)
    simplified_model, check = simplify(onnx_model)
    
    if check:
        optimized_path = output_path.replace('.onnx', '_optimized.onnx')
        onnx.save(simplified_model, optimized_path)
        optimized_size = os.path.getsize(optimized_path) / (1024 * 1024)
        print(f"  [OK] Optimized model saved: {optimized_path}")
        print(f"  Optimized size: {optimized_size:.2f} MB")
        print(f"  Additional compression: {onnx_size / optimized_size:.2f}x")
    else:
        print(f"  [WARN] Optimization check failed")
except ImportError:
    print(f"  [WARN] onnxsim not installed, skipping optimization")
    print(f"  Install with: pip install onnxsim")
except Exception as e:
    print(f"  [WARN] Optimization failed: {e}")

print("\n" + "=" * 70)
print("[OK] Conversion complete!")
print(f"  ONNX model: {output_path}")
print("=" * 70)
