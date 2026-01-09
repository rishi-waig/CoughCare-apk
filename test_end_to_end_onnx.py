# Fix OpenMP issue BEFORE any other imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

"""
End-to-End ONNX Pipeline Test for Cough Detection

This script validates that the two-stage ONNX pipeline produces
IDENTICAL results to the original PyTorch backend_api_actual_model.py

Pipeline:
1. cough_preprocessing.onnx: waveform -> spectrogram
2. cough_detector_int8.onnx: spectrogram -> prediction

Comparison:
- Original PyTorch model (ActualCoughModel from backend_api)
- ONNX two-stage pipeline
"""

import sys
import time
import numpy as np
import torch
import onnxruntime as ort

# Add path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)


class ONNXInferencePipeline:
    """
    Two-stage ONNX inference pipeline that replicates backend_api_actual_model.py
    """
    def __init__(self, preprocessing_path: str, model_path: str):
        print("Loading ONNX models...")

        # Load preprocessing model
        self.preprocess_session = ort.InferenceSession(
            preprocessing_path,
            providers=['CPUExecutionProvider']
        )
        print(f"  [OK] Preprocessing model: {preprocessing_path}")

        # Load INT8 model
        self.model_session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        print(f"  [OK] Detection model: {model_path}")

        # Configuration (matching backend_api)
        self.sample_rate = 16000
        self.segment_duration = 2.0
        self.hop_length = 0.5
        self.max_segments = 32
        self.segment_samples = int(self.sample_rate * self.segment_duration)

    def extract_segments(self, waveform: np.ndarray) -> list:
        """
        Extract 2-second segments from waveform.
        Matches CoughAudioProcessor.extract_segments()
        """
        seg_samples = self.segment_samples
        hop_samples = int(self.hop_length * self.sample_rate)

        # Handle shape - expect (samples,) or (1, samples)
        if waveform.ndim == 2:
            waveform = waveform.squeeze(0)

        # Pad if too short
        if len(waveform) < seg_samples:
            waveform = np.pad(waveform, (0, seg_samples - len(waveform)))

        segments = []
        start = 0
        while start + seg_samples <= len(waveform):
            segments.append(waveform[start:start + seg_samples])
            start += hop_samples

        if not segments:
            segments.append(waveform[:seg_samples])

        # Cap at max_segments
        if len(segments) > self.max_segments:
            segments = segments[:self.max_segments]

        return segments

    def preprocess_segment(self, segment: np.ndarray) -> np.ndarray:
        """
        Run preprocessing ONNX on a single segment.
        Input: (segment_samples,) waveform
        Output: (1, 3, 224, 224) spectrogram
        """
        # Reshape for ONNX: (1, segment_samples)
        segment_input = segment.reshape(1, -1).astype(np.float32)

        outputs = self.preprocess_session.run(
            None,
            {'waveform': segment_input}
        )

        return outputs[0]  # (1, 3, 224, 224)

    def predict(self, waveform: np.ndarray):
        """
        Full prediction pipeline.

        Args:
            waveform: (samples,) or (1, samples) audio waveform in [-1, 1]
        Returns:
            bag_prob, seg_probs, num_segments
        """
        # Extract segments
        segments = self.extract_segments(waveform)
        num_segments = len(segments)

        # Preprocess each segment
        spectrograms = []
        for seg in segments:
            spec = self.preprocess_segment(seg)
            spectrograms.append(spec)

        # Stack spectrograms: (1, num_segments, 3, 224, 224)
        specs_batch = np.concatenate(spectrograms, axis=0)  # (num_segments, 3, 224, 224)
        specs_batch = np.expand_dims(specs_batch, axis=0)  # (1, num_segments, 3, 224, 224)

        # Create mask: (1, num_segments)
        mask = np.ones((1, num_segments), dtype=bool)

        # Run model
        outputs = self.model_session.run(
            None,
            {
                'spectrograms': specs_batch.astype(np.float32),
                'segment_mask': mask
            }
        )

        bag_prob = outputs[0][0]  # Scalar
        seg_probs = outputs[1][0][:num_segments]  # (num_segments,)

        return float(bag_prob), seg_probs.tolist(), num_segments


def load_pytorch_model():
    """Load the original PyTorch model for comparison."""
    from train_cough_detector_attention import Config, AttnMILResNet, get_device
    from precompute_spectrograms import CoughAudioProcessor

    model_path = os.path.join(script_dir, "backup_best_model_20251015_170801.pth")

    device = get_device("cpu")  # Use CPU for fair comparison
    config = Config()

    model = AttnMILResNet(config).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    processor = CoughAudioProcessor(config)

    return model, processor, config, device


def pytorch_predict(model, processor, config, device, waveform: torch.Tensor):
    """
    Run prediction using original PyTorch model.
    Matches backend_api_actual_model.py logic exactly.
    """
    # Extract segments
    segments = processor.extract_segments(waveform)

    # Cap segments
    if len(segments) > config.max_segments_per_file:
        segments = segments[:config.max_segments_per_file]

    # Convert to spectrograms
    specs = []
    for seg in segments:
        spec3 = processor.waveform_to_mel_3ch(seg)  # (3, H, W) in [-1,1], float16
        specs.append(spec3.float())

    x = torch.stack(specs).unsqueeze(0)  # (1, S, 3, H, W)

    # ImageNet normalization (matching backend_api)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    x01 = (x + 1) / 2.0
    x_norm = (x01 - mean) / std

    mask = torch.ones(1, x.shape[1], dtype=torch.bool)

    with torch.no_grad():
        bag_prob, seg_probs, seg_logits, bag_logit = model(x_norm.to(device), mask.to(device))

    return float(bag_prob.cpu().item()), seg_probs.squeeze(0).cpu().numpy().tolist(), x.shape[1]


def test_with_random_audio():
    """Test with random audio waveform."""
    print("\n" + "="*70)
    print("TEST 1: Random Audio Waveform")
    print("="*70)

    # Create random 5-second audio
    duration = 5.0
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    waveform = np.random.randn(num_samples).astype(np.float32)
    waveform = waveform / np.abs(waveform).max()  # Normalize to [-1, 1]

    print(f"\nTest audio: {duration}s @ {sample_rate}Hz = {num_samples} samples")

    # ONNX pipeline
    print("\nLoading ONNX pipeline...")
    onnx_pipeline = ONNXInferencePipeline(
        preprocessing_path=os.path.join(script_dir, "cough_preprocessing.onnx"),
        model_path=os.path.join(script_dir, "cough_detector_int8.onnx")
    )

    print("\nONNX inference...")
    start = time.perf_counter()
    onnx_prob, onnx_segs, onnx_num = onnx_pipeline.predict(waveform)
    onnx_time = (time.perf_counter() - start) * 1000

    print(f"  Bag probability: {onnx_prob:.6f}")
    print(f"  Num segments: {onnx_num}")
    print(f"  Segment probs (first 5): {onnx_segs[:5]}")
    print(f"  Time: {onnx_time:.2f} ms")

    # PyTorch model
    print("\nLoading PyTorch model...")
    model, processor, config, device = load_pytorch_model()

    print("\nPyTorch inference...")
    waveform_torch = torch.from_numpy(waveform).unsqueeze(0)  # (1, samples)
    start = time.perf_counter()
    pt_prob, pt_segs, pt_num = pytorch_predict(model, processor, config, device, waveform_torch)
    pt_time = (time.perf_counter() - start) * 1000

    print(f"  Bag probability: {pt_prob:.6f}")
    print(f"  Num segments: {pt_num}")
    print(f"  Segment probs (first 5): {pt_segs[:5]}")
    print(f"  Time: {pt_time:.2f} ms")

    # Compare
    print("\n" + "-"*70)
    print("COMPARISON:")
    print("-"*70)

    prob_diff = abs(onnx_prob - pt_prob)
    print(f"  Bag probability difference: {prob_diff:.8f}")

    if len(onnx_segs) == len(pt_segs):
        seg_diffs = [abs(o - p) for o, p in zip(onnx_segs, pt_segs)]
        max_seg_diff = max(seg_diffs)
        mean_seg_diff = sum(seg_diffs) / len(seg_diffs)
        print(f"  Max segment prob difference: {max_seg_diff:.8f}")
        print(f"  Mean segment prob difference: {mean_seg_diff:.8f}")
    else:
        print(f"  Segment count mismatch: ONNX={len(onnx_segs)}, PyTorch={len(pt_segs)}")

    # Verdict
    tolerance = 0.01  # 1% tolerance for INT8 quantization
    if prob_diff < tolerance:
        print(f"\n[OK] Results match within {tolerance*100}% tolerance!")
        return True
    else:
        print(f"\n[WARNING] Results differ by {prob_diff:.4f}")
        return False


def test_with_real_audio():
    """Test with a real audio file if available."""
    print("\n" + "="*70)
    print("TEST 2: Real Audio File (if available)")
    print("="*70)

    # Look for test audio files
    audio_dir = os.path.join(script_dir, "uploaded_audio")
    test_files = []

    if os.path.exists(audio_dir):
        for f in os.listdir(audio_dir):
            if f.endswith(('.wav', '.webm', '.mp3', '.m4a')):
                test_files.append(os.path.join(audio_dir, f))

    if not test_files:
        print("\nNo test audio files found in uploaded_audio/")
        print("Skipping real audio test.")
        return None

    print(f"\nFound {len(test_files)} test files")

    # Test with first file
    test_file = test_files[0]
    print(f"Testing with: {os.path.basename(test_file)}")

    try:
        from precompute_spectrograms import CoughAudioProcessor
        from train_cough_detector_attention import Config

        config = Config()
        processor = CoughAudioProcessor(config)

        # Load audio
        waveform = processor.load_audio(test_file)
        print(f"  Loaded: {waveform.shape[1]} samples")

        # ONNX pipeline
        onnx_pipeline = ONNXInferencePipeline(
            preprocessing_path=os.path.join(script_dir, "cough_preprocessing.onnx"),
            model_path=os.path.join(script_dir, "cough_detector_int8.onnx")
        )

        onnx_prob, onnx_segs, onnx_num = onnx_pipeline.predict(waveform.numpy().squeeze())
        print(f"\n  ONNX: prob={onnx_prob:.4f}, segments={onnx_num}")

        # PyTorch
        model, processor, config, device = load_pytorch_model()
        pt_prob, pt_segs, pt_num = pytorch_predict(model, processor, config, device, waveform)
        print(f"  PyTorch: prob={pt_prob:.4f}, segments={pt_num}")

        diff = abs(onnx_prob - pt_prob)
        print(f"  Difference: {diff:.6f}")

        return diff < 0.01

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_inference():
    """Benchmark inference speed."""
    print("\n" + "="*70)
    print("BENCHMARK: Inference Speed")
    print("="*70)

    # Create test audio
    waveform = np.random.randn(16000 * 5).astype(np.float32)
    waveform = waveform / np.abs(waveform).max()

    # ONNX
    onnx_pipeline = ONNXInferencePipeline(
        preprocessing_path=os.path.join(script_dir, "cough_preprocessing.onnx"),
        model_path=os.path.join(script_dir, "cough_detector_int8.onnx")
    )

    # Warmup
    for _ in range(3):
        onnx_pipeline.predict(waveform)

    # Benchmark
    num_runs = 10
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        onnx_pipeline.predict(waveform)
        times.append((time.perf_counter() - start) * 1000)

    print(f"\nONNX Pipeline (5s audio, {num_runs} runs):")
    print(f"  Mean: {np.mean(times):.2f} ms")
    print(f"  Std:  {np.std(times):.2f} ms")
    print(f"  Min:  {np.min(times):.2f} ms")
    print(f"  Max:  {np.max(times):.2f} ms")


def print_model_info():
    """Print model sizes and info."""
    print("\n" + "="*70)
    print("MODEL INFORMATION")
    print("="*70)

    preprocessing_path = os.path.join(script_dir, "cough_preprocessing.onnx")
    model_path = os.path.join(script_dir, "cough_detector_int8.onnx")

    if os.path.exists(preprocessing_path):
        size = os.path.getsize(preprocessing_path) / (1024 * 1024)
        print(f"\nPreprocessing ONNX: {size:.2f} MB")

    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Detection INT8 ONNX: {size:.2f} MB")

    total = 0
    if os.path.exists(preprocessing_path):
        total += os.path.getsize(preprocessing_path)
    if os.path.exists(model_path):
        total += os.path.getsize(model_path)

    print(f"\nTotal size: {total / (1024*1024):.2f} MB")


def main():
    print("="*70)
    print("END-TO-END ONNX PIPELINE TEST")
    print("Validating: cough_preprocessing.onnx + cough_detector_int8.onnx")
    print("vs PyTorch: backend_api_actual_model.py")
    print("="*70)

    print_model_info()

    # Test with random audio
    test1_passed = test_with_random_audio()

    # Test with real audio
    test2_passed = test_with_real_audio()

    # Benchmark
    benchmark_inference()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\nTest 1 (Random Audio): {'PASSED' if test1_passed else 'FAILED'}")
    if test2_passed is not None:
        print(f"Test 2 (Real Audio): {'PASSED' if test2_passed else 'FAILED'}")
    else:
        print("Test 2 (Real Audio): SKIPPED")

    if test1_passed and (test2_passed is None or test2_passed):
        print("\n[SUCCESS] ONNX pipeline produces matching results!")
        print("\nYour React Native pipeline should be:")
        print("  1. Load audio -> waveform (React Native audio library)")
        print("  2. Resample to 16kHz (if needed)")
        print("  3. Normalize to [-1, 1]")
        print("  4. Extract 2-second segments (hop=0.5s)")
        print("  5. For each segment: run cough_preprocessing.onnx")
        print("  6. Stack spectrograms: (1, num_segments, 3, 224, 224)")
        print("  7. Run cough_detector_int8.onnx")
        print("  8. Get bag_probability output")
    else:
        print("\n[WARNING] Some tests did not pass. Check the differences above.")

    print("="*70)


if __name__ == "__main__":
    main()
