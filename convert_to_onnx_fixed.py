"""
Convert trained PyTorch cough detection model to ONNX format for compression and deployment.
Uses torchvision.models.resnet18 for compatibility with the training checkpoint.
"""

import os
import sys

# Fix OpenMP duplicate library issue on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple
from torchvision.models import resnet18

# Fix Windows console encoding
if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass


@dataclass
class Config:
    # Model configuration (must match training config)
    model_name: str = "resnet18"
    dropout: float = 0.3
    pretrained: bool = True
    attn_hidden: int = 128

    # Input specifications for ONNX export
    batch_size: int = 1
    max_segments: int = 32
    img_height: int = 224
    img_width: int = 224
    img_channels: int = 3


class AttnMILResNet(nn.Module):
    """
    ResNet18 backbone -> per-segment features -> attention weights over features
    -> bag feature (weighted sum) -> bag_logit.
    Also produces per-segment logits for inspection.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # Use torchvision's resnet18 (no pretrained weights - we'll load from checkpoint)
        self.backbone = resnet18(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.dropout = nn.Dropout(config.dropout)

        # Attention scorer (Ilse et al.)
        self.attn = nn.Sequential(
            nn.Linear(num_features, config.attn_hidden),
            nn.Tanh(),
            nn.Linear(config.attn_hidden, 1),  # score per segment
        )

        # Bag head (on pooled feature)
        self.bag_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(config.dropout),
            nn.Linear(512, 1),  # bag_logit
        )

        # Optional per-segment scores for diagnostics
        self.seg_head = nn.Linear(num_features, 1)

    def forward(self, x, seg_mask):
        """
        x: (B, S, 3, H, W)
        seg_mask: (B, S) bool
        """
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        feats = self.backbone(x)               # (B*S, F)
        feats = self.dropout(feats)
        Fdim = feats.shape[1]
        feats_bs = feats.view(B, S, Fdim)

        # attention scores with mask -> softmax over valid S
        attn_scores = self.attn(feats).view(B, S)  # (B,S)
        # masked softmax
        neg_inf = torch.finfo(attn_scores.dtype).min
        masked_scores = torch.where(seg_mask, attn_scores, torch.full_like(attn_scores, neg_inf))
        attn_weights = torch.softmax(masked_scores, dim=1)  # (B,S), sums to 1 over valid segs

        # pooled bag feature
        bag_feat = torch.sum(attn_weights.unsqueeze(-1) * feats_bs, dim=1)  # (B,F)

        # bag logit (NO sigmoid here)
        bag_logit = self.bag_head(bag_feat).squeeze(1)  # (B,)

        # per-segment logits for inspection
        seg_logits = self.seg_head(feats).view(B, S)    # (B,S)
        seg_probs = torch.sigmoid(seg_logits)

        bag_prob = torch.sigmoid(bag_logit)
        return bag_prob, seg_probs, seg_logits, bag_logit


def convert_to_onnx(
    checkpoint_path: str,
    output_path: str,
    config: Config = None,
    opset_version: int = 14,
    simplify: bool = True
):
    """
    Convert PyTorch checkpoint to ONNX format.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        output_path: Path to save .onnx file
        config: Model configuration (if None, uses default)
        opset_version: ONNX opset version (14 recommended for compatibility)
        simplify: Whether to simplify ONNX model (requires onnx-simplifier)
    """

    if config is None:
        config = Config()

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint with weights_only=False to handle older checkpoints
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        # Older PyTorch versions don't have weights_only parameter
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation AUC: {checkpoint.get('val_auc', 'unknown'):.4f}")
        print(f"Validation AP: {checkpoint.get('val_ap', 'unknown'):.4f}")
    else:
        state_dict = checkpoint

    # Create model and load weights
    print("Creating model...")
    model = AttnMILResNet(config)

    try:
        model.load_state_dict(state_dict)
        print("[OK] Weights loaded successfully")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Trying with strict=False...")
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    # Disable dropout and batch norm training mode
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d)):
            module.eval()

    print(f"Model loaded successfully. Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy inputs for tracing
    dummy_specs = torch.randn(
        config.batch_size,
        config.max_segments,
        config.img_channels,
        config.img_height,
        config.img_width
    )
    dummy_mask = torch.ones(config.batch_size, config.max_segments, dtype=torch.bool)

    print(f"\nExporting to ONNX with input shapes:")
    print(f"  - Spectrograms: {tuple(dummy_specs.shape)}")
    print(f"  - Mask: {tuple(dummy_mask.shape)}")

    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_specs, dummy_mask),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['spectrograms', 'segment_mask'],
            output_names=['bag_probability', 'segment_probabilities', 'segment_logits', 'bag_logit'],
            dynamic_axes={
                'spectrograms': {0: 'batch_size', 1: 'num_segments'},
                'segment_mask': {0: 'batch_size', 1: 'num_segments'},
                'bag_probability': {0: 'batch_size'},
                'segment_probabilities': {0: 'batch_size', 1: 'num_segments'},
                'segment_logits': {0: 'batch_size', 1: 'num_segments'},
                'bag_logit': {0: 'batch_size'}
            }
        )

    print(f"\n[OK] ONNX model saved to: {output_path}")

    # Get file sizes
    pth_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    onnx_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nFile sizes:")
    print(f"  - Original PyTorch (.pth): {pth_size:.2f} MB")
    print(f"  - ONNX (.onnx): {onnx_size:.2f} MB")
    print(f"  - Compression ratio: {pth_size/onnx_size:.2f}x")

    # Optionally simplify the ONNX model
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            print("\nSimplifying ONNX model...")
            onnx_model = onnx.load(output_path)
            simplified_model, check = onnx_simplify(onnx_model)

            if check:
                simplified_path = output_path.replace('.onnx', '_simplified.onnx')
                onnx.save(simplified_model, simplified_path)
                simplified_size = os.path.getsize(simplified_path) / (1024 * 1024)
                print(f"[OK] Simplified ONNX model saved to: {simplified_path}")
                print(f"Simplified size: {simplified_size:.2f} MB")
            else:
                print("Warning: ONNX simplification failed validation check")
        except ImportError:
            print("\nNote: Install onnx-simplifier for additional compression:")
            print("  pip install onnx-simplifier")
        except Exception as e:
            print(f"\nWarning: Could not simplify ONNX model: {e}")

    # Verify the ONNX model
    try:
        import onnx
        print("\nVerifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("[OK] ONNX model is valid!")
    except ImportError:
        print("\nNote: Install onnx package to verify the model:")
        print("  pip install onnx")
    except Exception as e:
        print(f"\nWarning: ONNX verification failed: {e}")

    print("\n[OK] Conversion complete!")
    return output_path


def test_onnx_inference(onnx_path: str, config: Config = None):
    """
    Test inference with the ONNX model using ONNX Runtime.
    """
    try:
        import onnxruntime as ort
        import numpy as np

        if config is None:
            config = Config()

        print(f"\n=== Testing ONNX inference with ONNX Runtime ===")
        print(f"Loading model from: {onnx_path}")

        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path)

        # Print model info
        print("\nModel inputs:")
        for inp in session.get_inputs():
            print(f"  - {inp.name}: {inp.shape} ({inp.type})")

        print("\nModel outputs:")
        for out in session.get_outputs():
            print(f"  - {out.name}: {out.shape} ({out.type})")

        # Create dummy inputs
        dummy_specs = np.random.randn(
            config.batch_size,
            config.max_segments,
            config.img_channels,
            config.img_height,
            config.img_width
        ).astype(np.float32)

        dummy_mask = np.ones((config.batch_size, config.max_segments), dtype=bool)

        # Run inference
        print("\nRunning inference...")
        outputs = session.run(
            None,
            {
                'spectrograms': dummy_specs,
                'segment_mask': dummy_mask
            }
        )

        print("\n[OK] Inference successful!")
        print(f"  - Bag probability shape: {outputs[0].shape}, value: {outputs[0][0]:.4f}")
        print(f"  - Segment probabilities shape: {outputs[1].shape}")
        print(f"  - Example segment probs (first 5): {outputs[1][0][:5]}")

        return True

    except ImportError:
        print("\nNote: Install onnxruntime to test inference:")
        print("  pip install onnxruntime")
        return False
    except Exception as e:
        print(f"\nError during ONNX inference test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Configuration
    checkpoint_path = r"D:\coughcare_waig_3\coughcare_waig 3\backup_best_model_20251015_170801.pth"
    output_path = r"D:\coughcare_waig_3\coughcare_waig 3\cough_detector_attention.onnx"

    # Create config matching your training setup
    config = Config(
        dropout=0.3,
        attn_hidden=128,
        batch_size=1,
        max_segments=32,
        img_height=224,
        img_width=224,
        img_channels=3
    )

    print("="*60)
    print("ONNX Model Converter for Cough Detection")
    print("="*60)

    # Convert to ONNX
    try:
        onnx_path = convert_to_onnx(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            config=config,
            opset_version=14,
            simplify=True
        )

        # Test the ONNX model
        test_onnx_inference(onnx_path, config)

        print("\n" + "="*60)
        print("SUCCESS! Your model has been converted to ONNX format.")
        print("="*60)
        print("\nYou can now use the ONNX model for:")
        print("  - Deployment on edge devices")
        print("  - Faster inference with ONNX Runtime")
        print("  - Cross-platform compatibility")
        print("  - Integration with various frameworks (TensorRT, OpenVINO, etc.)")
        print("="*60)

    except Exception as e:
        print(f"\n[ERROR] Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
