"""
Test ONNX model inference with sample audio file
"""

import os
import sys

# Fix OpenMP duplicate library issue on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from pydub import AudioSegment


class AudioConfig:
    """Audio processing configuration matching training"""
    sample_rate: int = 16000
    segment_duration: float = 2.0
    hop_length: float = 0.5
    n_fft: int = 1024
    hop_length_fft: int = 160
    n_mels: int = 160
    f_min: float = 50.0
    f_max: float = 4000.0
    resize_to_224: bool = True
    max_segments_per_file: int = 32


def letterbox_to_square(x: torch.Tensor, target_hw: int = 224, pad_val: float = -1.0):
    """Letterbox-pad to square target size without distortion."""
    C, H, W = x.shape
    scale = min(target_hw / H, target_hw / W)
    new_h, new_w = int(round(H * scale)), int(round(W * scale))

    x_resized = torch.nn.functional.interpolate(
        x.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)

    pad_h = target_hw - new_h
    pad_w = target_hw - new_w
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    x_padded = F.pad(x_resized, (pad_left, pad_right, pad_top, pad_bottom), value=pad_val)
    return x_padded


class CoughAudioProcessor:
    """Processes audio exactly as in the training pipeline."""

    def __init__(self, config):
        self.config = config
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.n_fft,
            hop_length=config.hop_length_fft,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=2.0,
            normalized=False
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def load_audio(self, path: str) -> torch.Tensor:
        """Load audio file and return waveform tensor (1, T) at config.sample_rate."""
        try:
            audio = AudioSegment.from_file(path)
            if audio.channels > 1:
                audio = audio.set_channels(1)

            sample_rate = audio.frame_rate
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

            if audio.sample_width == 1:
                samples = samples / 128.0 - 1.0
            elif audio.sample_width == 2:
                samples = samples / 32768.0
            elif audio.sample_width == 4:
                samples = samples / 2147483648.0
            else:
                samples = samples / (2 ** (8 * audio.sample_width - 1))

            waveform = torch.from_numpy(samples).unsqueeze(0)

            if sample_rate != self.config.sample_rate:
                resampler = T.Resample(sample_rate, self.config.sample_rate)
                waveform = resampler(waveform)

            return waveform

        except Exception as e:
            waveform, sample_rate = torchaudio.load(path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if sample_rate != self.config.sample_rate:
                resampler = T.Resample(sample_rate, self.config.sample_rate)
                waveform = resampler(waveform)

            return waveform

    def extract_segments(self, waveform: torch.Tensor) -> list:
        """Extract overlapping segments from waveform."""
        segments = []
        segment_samples = int(self.config.segment_duration * self.config.sample_rate)
        hop_samples = int(self.config.hop_length * self.config.sample_rate)

        total_samples = waveform.shape[1]

        start = 0
        while start < total_samples:
            end = start + segment_samples
            seg = waveform[:, start:end]

            if seg.shape[1] < segment_samples:
                padding = segment_samples - seg.shape[1]
                seg = torch.nn.functional.pad(seg, (0, padding))

            segments.append(seg)

            if len(segments) >= self.config.max_segments_per_file:
                break

            start += hop_samples

            if start >= total_samples:
                break

        return segments

    def waveform_to_mel_3ch(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform segment to 3-channel mel spectrogram."""
        mel = self.mel_transform(waveform)
        mel_db = self.db_transform(mel)

        mel_norm01 = torch.clamp((mel_db + 80) / 80, 0, 1)
        mel_norm = mel_norm01 * 2 - 1

        mel_img = mel_norm.squeeze(0)

        if self.config.resize_to_224:
            mel_img = letterbox_to_square(mel_img.unsqueeze(0), 224).squeeze(0)

        mel_3ch = mel_img.repeat(3, 1, 1)

        return mel_3ch.float()


def test_onnx_model(onnx_path: str, audio_path: str):
    """Test ONNX model with audio file"""
    print("="*70)
    print("ONNX Model Inference Test")
    print("="*70)

    print(f"\nONNX Model: {onnx_path}")
    print(f"Audio File: {audio_path}")

    # Check files exist
    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX model not found at {onnx_path}")
        return False

    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found at {audio_path}")
        return False

    # Load ONNX model
    print("\n1. Loading ONNX model...")
    session = ort.InferenceSession(onnx_path)
    print(f"   Model loaded successfully")

    # Print model info
    print("\n   Model inputs:")
    for inp in session.get_inputs():
        print(f"     - {inp.name}: {inp.shape}")

    print("\n   Model outputs:")
    for out in session.get_outputs():
        print(f"     - {out.name}: {out.shape}")

    # Process audio
    print("\n2. Processing audio file...")
    config = AudioConfig()
    processor = CoughAudioProcessor(config)

    waveform = processor.load_audio(audio_path)
    print(f"   Loaded audio: {waveform.shape[1]} samples ({waveform.shape[1]/config.sample_rate:.2f} seconds)")

    segments = processor.extract_segments(waveform)
    print(f"   Extracted {len(segments)} segments")

    # Build batch
    print("\n3. Building input batch...")
    specs = []
    for seg in segments:
        spec3 = processor.waveform_to_mel_3ch(seg)
        specs.append(spec3.numpy())

    x = np.stack(specs)[np.newaxis, ...]  # (1, S, 3, H, W)

    # ImageNet normalize
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1)

    x01 = (x + 1) / 2.0
    x_norm = (x01 - mean) / std
    x_norm = x_norm.astype(np.float32)

    mask = np.ones((1, x.shape[1]), dtype=bool)

    print(f"   Input batch shape: {x_norm.shape}")
    print(f"   Mask shape: {mask.shape}")

    # Run inference
    print("\n4. Running ONNX inference...")
    outputs = session.run(
        None,
        {
            'spectrograms': x_norm,
            'segment_mask': mask
        }
    )

    bag_prob = outputs[0][0]
    seg_probs = outputs[1][0]

    print(f"\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"   Bag Probability (Cough Score): {bag_prob:.4f}")
    print(f"   Cough Detected (>0.61):        {'YES' if bag_prob > 0.61 else 'NO'}")
    print(f"   Number of segments:            {len(seg_probs)}")
    print(f"   Segment probabilities (first 5): {seg_probs[:5]}")
    print(f"   Max segment probability:       {max(seg_probs):.4f}")
    print(f"   Mean segment probability:      {np.mean(seg_probs):.4f}")
    print("="*70)

    return True


if __name__ == "__main__":
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(script_dir, "cough_detector_attention.onnx")
    audio_path = os.path.join(script_dir, "public", "samples", "20251104_150725_454926_cough.wav")

    success = test_onnx_model(onnx_path, audio_path)

    if success:
        print("\nTest PASSED!")
    else:
        print("\nTest FAILED!")
        sys.exit(1)
