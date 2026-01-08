"""
Comprehensive test of all ONNX models with the actual sample audio file.
Compares: Original ONNX, Int8 Quantized, Float16 Precision
Uses the exact same preprocessing as test_onnx_inference.py
"""

import os
import sys
import time
import numpy as np

# Fix OpenMP duplicate library issue on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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


def process_audio(audio_path, config):
    """Process audio file and return batch + mask"""
    processor = CoughAudioProcessor(config)
    waveform = processor.load_audio(audio_path)
    segments = processor.extract_segments(waveform)

    specs = []
    for seg in segments:
        spec3 = processor.waveform_to_mel_3ch(seg)
        specs.append(spec3.numpy())

    x = np.stack(specs)[np.newaxis, ...]  # (1, S, 3, H, W) in [-1, 1]

    # ImageNet normalize - exact same as test_onnx_inference.py
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1)

    x01 = (x + 1) / 2.0
    x_norm = (x01 - mean) / std
    x_norm = x_norm.astype(np.float32)

    mask = np.ones((1, x.shape[1]), dtype=bool)
    return x_norm, mask, len(segments), waveform.shape[1] / config.sample_rate


def test_model(model_path, batch, mask):
    """Test a single model and return results"""
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        # Warm up
        _ = session.run(None, {'spectrograms': batch, 'segment_mask': mask})

        # Timed inference
        start = time.perf_counter()
        num_runs = 3
        for _ in range(num_runs):
            outputs = session.run(None, {'spectrograms': batch, 'segment_mask': mask})
        elapsed = (time.perf_counter() - start) / num_runs * 1000

        bag_prob = float(outputs[0][0])
        seg_probs = outputs[1][0].tolist()

        return {
            'success': True,
            'probability': bag_prob,
            'cough_detected': bag_prob > 0.61,
            'segment_probs': seg_probs,
            'max_segment': max(seg_probs),
            'mean_segment': np.mean(seg_probs),
            'inference_time_ms': elapsed,
            'size_mb': os.path.getsize(model_path) / (1024 * 1024)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'size_mb': os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0
        }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths
    audio_path = os.path.join(script_dir, "public", "samples", "20251104_150725_454926_cough.wav")
    models = {
        'Original ONNX': os.path.join(script_dir, "cough_detector_attention.onnx"),
        'Int8 Quantized': os.path.join(script_dir, "cough_detector_int8.onnx"),
        'Float16 Precision': os.path.join(script_dir, "cough_detector_fp16.onnx"),
    }

    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON TEST")
    print("="*80)

    # Check audio file
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    print(f"\nTest Audio: {audio_path}")

    # Process audio
    print("\nProcessing audio file...")
    config = AudioConfig()
    batch, mask, num_segments, duration = process_audio(audio_path, config)
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Segments: {num_segments}")
    print(f"  Batch shape: {batch.shape}")
    print(f"  Batch stats: min={batch.min():.4f}, max={batch.max():.4f}, mean={batch.mean():.4f}")

    # Test each model
    print("\n" + "-"*80)
    print("TESTING MODELS")
    print("-"*80)

    results = {}
    for name, path in models.items():
        print(f"\n{name}:")
        if not os.path.exists(path):
            print(f"  SKIPPED - File not found: {path}")
            continue

        result = test_model(path, batch, mask)
        results[name] = result

        if result['success']:
            print(f"  Size:         {result['size_mb']:.2f} MB")
            print(f"  Probability:  {result['probability']:.4f}")
            print(f"  Cough:        {'YES' if result['cough_detected'] else 'NO'}")
            print(f"  Max Segment:  {result['max_segment']:.4f}")
            print(f"  Mean Segment: {result['mean_segment']:.4f}")
            print(f"  Inference:    {result['inference_time_ms']:.2f} ms")
        else:
            print(f"  FAILED: {result['error']}")
            print(f"  Size: {result['size_mb']:.2f} MB")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Model':<20} {'Size (MB)':<12} {'Compression':<12} {'Probability':<12} {'Cough?':<8} {'Time (ms)':<10}")
    print("-"*80)

    orig_size = results.get('Original ONNX', {}).get('size_mb', 43.90)
    orig_prob = results.get('Original ONNX', {}).get('probability', 0)

    for name, result in results.items():
        if result.get('success'):
            compression = orig_size / result['size_mb'] if result['size_mb'] > 0 else 1.0
            cough = 'YES' if result['cough_detected'] else 'NO'
            print(f"{name:<20} {result['size_mb']:<12.2f} {compression:<12.1f}x {result['probability']:<12.4f} {cough:<8} {result['inference_time_ms']:<10.2f}")
        else:
            print(f"{name:<20} {result.get('size_mb', 0):<12.2f} {'N/A':<12} {'FAILED':<12} {'N/A':<8} {'N/A':<10}")

    print("="*80)

    # Accuracy comparison
    if orig_prob > 0:
        print("\nACCURACY COMPARISON (vs Original ONNX):")
        for name, result in results.items():
            if result.get('success') and name != 'Original ONNX':
                diff = abs(result['probability'] - orig_prob)
                pct_diff = (diff / orig_prob) * 100
                print(f"  {name}: {diff:.6f} absolute ({pct_diff:.2f}% relative)")

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    print(f"\nOriginal PyTorch (.pth): 131.82 MB (reference)")

    if results.get('Original ONNX', {}).get('success'):
        orig = results['Original ONNX']
        print(f"Original ONNX:           {orig['size_mb']:.2f} MB ({131.82/orig['size_mb']:.1f}x compression)")

    if results.get('Int8 Quantized', {}).get('success'):
        int8 = results['Int8 Quantized']
        print(f"Int8 Quantized:          {int8['size_mb']:.2f} MB ({131.82/int8['size_mb']:.1f}x compression)")

    if results.get('Float16 Precision', {}).get('success'):
        fp16 = results['Float16 Precision']
        print(f"Float16 Precision:       {fp16['size_mb']:.2f} MB ({131.82/fp16['size_mb']:.1f}x compression)")

    print("\nCough Detection Results:")
    for name, result in results.items():
        if result.get('success'):
            status = 'COUGH DETECTED' if result['cough_detected'] else 'No cough'
            print(f"  {name}: {status} (prob={result['probability']:.4f})")

    print("="*80)


if __name__ == "__main__":
    main()
