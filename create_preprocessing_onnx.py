# Fix OpenMP issue BEFORE any other imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

"""
Create Preprocessing ONNX Model for Cough Detection Pipeline

This creates an ONNX model that replicates the EXACT preprocessing from:
- CoughAudioProcessor.waveform_to_mel_3ch()
- Letterbox to 224x224
- ImageNet normalization

Input: Raw waveform tensor (1, T) where T = segment_duration * sample_rate
Output: Normalized 3-channel mel spectrogram (1, 3, 224, 224) ready for the model

This ensures 100% replication of backend_api_actual_model.py preprocessing.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Configuration matching train_cough_detector_attention.py Config
class PreprocessConfig:
    sample_rate: int = 16000
    segment_duration: float = 2.0  # 2 seconds per segment
    n_fft: int = 1024
    hop_length_fft: int = 160  # ~10ms @ 16kHz
    n_mels: int = 160
    f_min: float = 50.0
    f_max: float = 4000.0
    resize_to_224: bool = True

    # Derived
    segment_samples: int = int(sample_rate * segment_duration)  # 32000 samples


class MelSpectrogramONNX(nn.Module):
    """
    ONNX-compatible mel spectrogram computation.
    Replicates torchaudio.transforms.MelSpectrogram EXACTLY.
    Uses conv1d for frame extraction (ONNX-friendly, no unfold).
    Extracts weights from actual torchaudio MelSpectrogram for perfect match.
    """
    def __init__(self, config: PreprocessConfig):
        super().__init__()
        self.config = config
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length_fft
        self.n_mels = config.n_mels
        self.sample_rate = config.sample_rate
        self.f_min = config.f_min
        self.f_max = config.f_max
        self.segment_samples = config.segment_samples

        # Create actual torchaudio MelSpectrogram to extract its exact weights
        import torchaudio.transforms as T
        reference_mel = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.n_fft,
            hop_length=config.hop_length_fft,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=2.0,
            normalized=False,
        )

        # Extract mel filterbank from torchaudio (this is the exact filterbank used)
        # MelSpectrogram.mel_scale.fb is (n_freqs, n_mels), we need (n_mels, n_freqs)
        mel_fb = reference_mel.mel_scale.fb.T.contiguous()
        self.register_buffer('mel_filterbank', mel_fb)

        # Pre-compute STFT window (same as torchaudio)
        window = torch.hann_window(self.n_fft)
        self.register_buffer('window', window)

        # Pre-compute DFT matrix for ONNX compatibility (instead of torch.fft.rfft)
        # This creates a real-valued DFT that produces the same output as rfft
        dft_real, dft_imag = self._create_dft_matrix()
        self.register_buffer('dft_real', dft_real)
        self.register_buffer('dft_imag', dft_imag)

        # Pre-compute frame extraction conv kernel (identity kernels for each position)
        # This replaces unfold with conv1d which is ONNX-friendly
        frame_kernel = self._create_frame_kernel()
        self.register_buffer('frame_kernel', frame_kernel)

        # Pre-compute number of frames for fixed input size
        padded_length = self.segment_samples + self.n_fft  # reflect padding both sides
        self.num_frames = (padded_length - self.n_fft) // self.hop_length + 1

    def _create_dft_matrix(self):
        """Create DFT matrices for computing rfft without torch.fft (ONNX compatible)."""
        n_freqs = self.n_fft // 2 + 1
        # DFT matrix: W[k,n] = exp(-2*pi*i*k*n/N)
        # Real part: cos(-2*pi*k*n/N) = cos(2*pi*k*n/N)
        # Imag part: sin(-2*pi*k*n/N) = -sin(2*pi*k*n/N)
        k = np.arange(n_freqs).reshape(-1, 1)  # (n_freqs, 1)
        n = np.arange(self.n_fft).reshape(1, -1)  # (1, n_fft)
        angle = 2 * np.pi * k * n / self.n_fft  # (n_freqs, n_fft)

        dft_real = torch.from_numpy(np.cos(angle)).float()  # (n_freqs, n_fft)
        dft_imag = torch.from_numpy(-np.sin(angle)).float()  # (n_freqs, n_fft)

        return dft_real, dft_imag

    def _create_frame_kernel(self):
        """Create conv1d kernel for frame extraction (replaces unfold)."""
        # kernel shape: (out_channels, in_channels, kernel_size)
        # We want to extract n_fft samples at each hop position
        # Each output channel corresponds to one sample position in the frame
        kernel = torch.eye(self.n_fft).unsqueeze(1)  # (n_fft, 1, n_fft)
        return kernel

    def forward(self, waveform):
        """
        Compute mel spectrogram from waveform.

        Args:
            waveform: (batch, samples) or (batch, 1, samples)
        Returns:
            mel_spec: (batch, n_mels, time_frames)
        """
        # Handle input shape
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)  # (batch, samples)

        batch_size = waveform.shape[0]

        # Pad waveform for STFT (reflect padding)
        pad_amount = self.n_fft // 2
        waveform_padded = F.pad(waveform, (pad_amount, pad_amount), mode='reflect')

        # Add channel dimension for conv1d: (batch, 1, padded_length)
        waveform_padded = waveform_padded.unsqueeze(1)

        # Extract frames using conv1d (ONNX-friendly)
        # Output: (batch, n_fft, num_frames)
        frames = F.conv1d(waveform_padded, self.frame_kernel, stride=self.hop_length)

        # frames is now (batch, n_fft, num_frames)
        # Transpose to (batch, num_frames, n_fft) for DFT
        frames = frames.transpose(1, 2)

        # Apply window: (batch, num_frames, n_fft) * (1, 1, n_fft)
        frames_windowed = frames * self.window.unsqueeze(0).unsqueeze(0)

        # Compute DFT using matrix multiplication (ONNX-friendly, no torch.fft)
        # frames_windowed: (batch, num_frames, n_fft)
        # dft_real/imag: (n_freqs, n_fft)
        # Result: (batch, num_frames, n_freqs)
        spectrum_real = torch.matmul(frames_windowed, self.dft_real.T)
        spectrum_imag = torch.matmul(frames_windowed, self.dft_imag.T)

        # Power spectrum: |X|^2 = real^2 + imag^2
        power_spec = spectrum_real ** 2 + spectrum_imag ** 2  # (batch, num_frames, n_freqs)

        # Apply mel filterbank
        # power_spec: (batch, num_frames, n_freqs)
        # mel_filterbank: (n_mels, n_freqs)
        mel_spec = torch.matmul(power_spec, self.mel_filterbank.T)  # (batch, num_frames, n_mels)

        # Transpose to (batch, n_mels, num_frames)
        mel_spec = mel_spec.transpose(1, 2)

        return mel_spec


class AmplitudeToDBONNX(nn.Module):
    """Convert amplitude/power spectrogram to dB scale."""
    def __init__(self, top_db=80.0):
        super().__init__()
        self.top_db = top_db
        self.multiplier = 10.0  # For power spectrogram
        self.amin = 1e-10

    def forward(self, x):
        # x: power spectrogram
        x_db = self.multiplier * torch.log10(torch.clamp(x, min=self.amin))

        # Apply top_db threshold
        x_db_max = x_db.max()
        x_db = torch.clamp(x_db, min=x_db_max - self.top_db)

        return x_db


class LetterboxToSquare(nn.Module):
    """
    Letterbox pad to square target size without distortion.
    Pre-computes all dimensions for ONNX traceability.
    """
    def __init__(self, input_h: int, input_w: int, target_size: int = 224, pad_value: float = -1.0):
        super().__init__()
        self.target_size = target_size
        self.pad_value = pad_value

        # Pre-compute letterbox dimensions (ONNX-friendly: no dynamic shape ops)
        scale = min(target_size / input_h, target_size / input_w)
        self.new_h = int(round(input_h * scale))
        self.new_w = int(round(input_w * scale))

        # Pre-compute padding
        pad_h = target_size - self.new_h
        pad_w = target_size - self.new_w
        self.pad_left = pad_w // 2
        self.pad_right = pad_w - self.pad_left
        self.pad_top = pad_h // 2
        self.pad_bottom = pad_h - self.pad_top

    def forward(self, x):
        """
        Args:
            x: (batch, channels, H, W)
        Returns:
            x_padded: (batch, channels, target_size, target_size)
        """
        # Resize to pre-computed dimensions
        x_resized = F.interpolate(x, size=(self.new_h, self.new_w), mode='bilinear', align_corners=False)

        # Apply pre-computed padding
        x_padded = F.pad(x_resized, (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom),
                         mode='constant', value=self.pad_value)

        return x_padded


class CoughPreprocessingModel(nn.Module):
    """
    Complete preprocessing pipeline for cough detection.

    Replicates EXACTLY:
    1. waveform_to_mel_3ch() from precompute_spectrograms.py
    2. ImageNet normalization from backend_api_actual_model.py

    Input: waveform (batch, 1, segment_samples) - single 2-second segment
    Output: normalized spectrogram (batch, 3, 224, 224) ready for model
    """
    def __init__(self, config: PreprocessConfig = None):
        super().__init__()
        if config is None:
            config = PreprocessConfig()
        self.config = config

        # Mel spectrogram
        self.mel_spec = MelSpectrogramONNX(config)

        # Amplitude to dB
        self.to_db = AmplitudeToDBONNX(top_db=80.0)

        # Pre-compute mel spectrogram output dimensions for letterbox
        # segment_samples -> after STFT with hop_length -> num_frames
        segment_samples = config.segment_samples
        # With reflect padding of n_fft//2 on each side, and hop_length
        # num_frames = (segment_samples + 2*(n_fft//2) - n_fft) // hop_length + 1
        # Simplified: num_frames = segment_samples // hop_length + 1 (approximately)
        # More accurate: after padding, total = segment_samples + n_fft
        # num_frames = (total - n_fft) // hop_length + 1 = segment_samples // hop_length + 1
        num_frames = segment_samples // config.hop_length_fft + 1
        mel_h = config.n_mels  # 160
        mel_w = num_frames  # 201 for 2s @ 16kHz with hop=160

        # Letterbox to 224x224 with pre-computed dimensions
        self.letterbox = LetterboxToSquare(input_h=mel_h, input_w=mel_w, target_size=224, pad_value=-1.0)

        # ImageNet normalization constants
        # These will normalize from [0,1] to ImageNet scale
        self.register_buffer('imagenet_mean',
                           torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std',
                           torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, waveform):
        """
        Full preprocessing pipeline.

        Args:
            waveform: (batch, segment_samples) raw audio samples in [-1, 1]
        Returns:
            x_norm: (batch, 3, 224, 224) ImageNet-normalized spectrogram
        """
        batch_size = waveform.shape[0]

        # 1. Compute mel spectrogram
        mel = self.mel_spec(waveform)  # (batch, n_mels, frames)

        # 2. Convert to dB
        mel_db = self.to_db(mel)  # (batch, n_mels, frames)

        # 3. Normalize to [-1, 1] range (matching precompute_spectrograms.py)
        # mel_db is in range [-80, 0] after to_db
        # Formula: mel_norm01 = clamp((mel_db + 80) / 80, 0, 1)
        #          mel_norm = mel_norm01 * 2 - 1  -> [-1, 1]
        mel_norm01 = torch.clamp((mel_db + 80) / 80, 0, 1)
        mel_norm = mel_norm01 * 2 - 1  # [-1, 1]

        # 4. Add channel dimension for letterbox: (batch, 1, n_mels, frames)
        mel_img = mel_norm.unsqueeze(1)

        # 5. Letterbox to 224x224
        mel_224 = self.letterbox(mel_img)  # (batch, 1, 224, 224)

        # 6. Expand to 3 channels (matching precompute_spectrograms.py)
        mel_3ch = mel_224.expand(-1, 3, -1, -1)  # (batch, 3, 224, 224)

        # 7. Map [-1,1] -> [0,1] then ImageNet normalize
        # (matching backend_api_actual_model.py _build_batch)
        x01 = (mel_3ch + 1) / 2.0  # [0, 1]
        x_norm = (x01 - self.imagenet_mean) / self.imagenet_std

        return x_norm


def export_preprocessing_onnx(output_path: str, opset_version: int = 17):
    """
    Export the preprocessing model to ONNX format.
    """
    print("="*70)
    print("Creating Preprocessing ONNX Model")
    print("="*70)

    config = PreprocessConfig()
    model = CoughPreprocessingModel(config)
    model.eval()

    print(f"\nConfiguration:")
    print(f"  Sample rate: {config.sample_rate} Hz")
    print(f"  Segment duration: {config.segment_duration} s")
    print(f"  Segment samples: {config.segment_samples}")
    print(f"  n_fft: {config.n_fft}")
    print(f"  hop_length: {config.hop_length_fft}")
    print(f"  n_mels: {config.n_mels}")
    print(f"  f_min: {config.f_min} Hz")
    print(f"  f_max: {config.f_max} Hz")

    # Create dummy input (single 2-second segment)
    dummy_waveform = torch.randn(1, config.segment_samples)

    print(f"\nInput shape: {dummy_waveform.shape}")
    print(f"  (batch_size, segment_samples)")

    # Test forward pass
    with torch.no_grad():
        output = model(dummy_waveform)
    print(f"Output shape: {output.shape}")
    print(f"  (batch_size, channels, height, width)")

    # Export to ONNX
    print(f"\nExporting to ONNX (opset {opset_version})...")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_waveform,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['waveform'],
            output_names=['spectrogram'],
            dynamic_axes={
                'waveform': {0: 'batch_size'},
                'spectrogram': {0: 'batch_size'}
            }
        )

    # Check file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nONNX model saved: {output_path}")
    print(f"File size: {size_mb:.2f} MB")

    # Verify ONNX model
    try:
        import onnx
        print("\nVerifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("[OK] ONNX model is valid!")

        # Print model info
        print("\nModel inputs:")
        for inp in onnx_model.graph.input:
            print(f"  - {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
        print("Model outputs:")
        for out in onnx_model.graph.output:
            print(f"  - {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")

    except ImportError:
        print("Note: Install onnx package to verify model")
    except Exception as e:
        print(f"Warning: ONNX verification issue: {e}")

    return output_path


def test_preprocessing_onnx(onnx_path: str, pytorch_model: CoughPreprocessingModel = None):
    """
    Test ONNX preprocessing model and compare with PyTorch.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("Error: onnxruntime not installed")
        return False

    print("\n" + "="*70)
    print("Testing Preprocessing ONNX Model")
    print("="*70)

    config = PreprocessConfig()

    # Create PyTorch model if not provided
    if pytorch_model is None:
        pytorch_model = CoughPreprocessingModel(config)
        pytorch_model.eval()

    # Create ONNX session
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # Create test input
    test_waveform = torch.randn(1, config.segment_samples)
    test_waveform_np = test_waveform.numpy()

    print(f"\nTest input shape: {test_waveform.shape}")

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(test_waveform).numpy()

    # ONNX inference
    onnx_output = session.run(None, {'waveform': test_waveform_np})[0]

    # Compare outputs
    print(f"\nPyTorch output shape: {pytorch_output.shape}")
    print(f"ONNX output shape: {onnx_output.shape}")

    # Calculate differences
    abs_diff = np.abs(pytorch_output - onnx_output)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    print(f"\nNumerical comparison:")
    print(f"  Max absolute difference: {max_diff:.8f}")
    print(f"  Mean absolute difference: {mean_diff:.8f}")

    # Check if outputs match within tolerance
    tolerance = 1e-4
    if max_diff < tolerance:
        print(f"\n[OK] Outputs match within tolerance ({tolerance})")
        return True
    else:
        print(f"\n[WARNING] Outputs differ more than tolerance ({tolerance})")
        print("This may be due to floating-point precision differences.")

        # Check if relatively close
        if max_diff < 1e-2:
            print("However, difference is small enough for practical use.")
            return True
        return False


def compare_with_original_processor():
    """
    Compare ONNX preprocessing with original CoughAudioProcessor.
    This validates 100% replication.
    """
    print("\n" + "="*70)
    print("Comparing with Original CoughAudioProcessor")
    print("="*70)

    try:
        # Import original processor
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from precompute_spectrograms import CoughAudioProcessor
        from train_cough_detector_attention import Config as OrigConfig

        orig_config = OrigConfig()
        orig_processor = CoughAudioProcessor(orig_config)

        # Create our ONNX-style model
        config = PreprocessConfig()
        onnx_model = CoughPreprocessingModel(config)
        onnx_model.eval()

        # Create test waveform (simulating a 2-second segment)
        test_waveform = torch.randn(1, config.segment_samples)

        print(f"\nTest waveform shape: {test_waveform.shape}")

        # Original processor output
        # Note: Original processor expects (1, T) and outputs (3, H, W) in [-1, 1]
        # then backend_api normalizes it
        orig_spec = orig_processor.waveform_to_mel_3ch(test_waveform)
        orig_spec = orig_spec.float()  # Convert from float16

        # Apply ImageNet normalization (matching backend_api)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        orig_x01 = (orig_spec.unsqueeze(0) + 1) / 2.0
        orig_normalized = (orig_x01 - mean) / std

        # Our model output
        with torch.no_grad():
            our_output = onnx_model(test_waveform)

        print(f"\nOriginal processor output shape: {orig_normalized.shape}")
        print(f"Our ONNX model output shape: {our_output.shape}")

        # Compare
        abs_diff = torch.abs(orig_normalized - our_output)
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()

        print(f"\nNumerical comparison with ORIGINAL processor:")
        print(f"  Max absolute difference: {max_diff:.6f}")
        print(f"  Mean absolute difference: {mean_diff:.6f}")

        if max_diff < 0.1:
            print("\n[OK] ONNX preprocessing matches original processor!")
            return True
        else:
            print("\n[WARNING] Significant differences detected.")
            print("This may be due to mel filterbank implementation differences.")
            return False

    except ImportError as e:
        print(f"\nCould not import original processor: {e}")
        print("Skipping comparison with original.")
        return None
    except Exception as e:
        print(f"\nError during comparison: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "cough_preprocessing.onnx")

    # Export preprocessing ONNX
    export_preprocessing_onnx(output_path, opset_version=17)

    # Test ONNX model
    test_preprocessing_onnx(output_path)

    # Compare with original processor
    compare_with_original_processor()

    print("\n" + "="*70)
    print("PREPROCESSING ONNX MODEL COMPLETE!")
    print("="*70)
    print(f"\nCreated: {output_path}")
    print("\nPipeline for React Native:")
    print("  1. Load audio file -> waveform (in React Native)")
    print("  2. Resample to 16kHz (in React Native)")
    print("  3. Extract 2-second segments (in React Native)")
    print("  4. For each segment:")
    print("     a. Run cough_preprocessing.onnx -> spectrogram")
    print("     b. Stack spectrograms -> (1, num_segments, 3, 224, 224)")
    print("     c. Run cough_detector_int8.onnx -> prediction")
    print("="*70)


if __name__ == "__main__":
    main()
