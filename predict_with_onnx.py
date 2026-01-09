# Fix OpenMP issue BEFORE any other imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from typing import List, Tuple, Optional
from pathlib import Path

# Configuration matching train_cough_detector_attention.py
class Config:
    sample_rate: int = 16000
    segment_duration: float = 2.0  # 2 seconds per segment
    hop_length: float = 0.5  # 0.5 seconds hop (50% overlap)
    n_fft: int = 1024
    hop_length_fft: int = 160
    n_mels: int = 160
    f_min: float = 50.0
    f_max: float = 4000.0
    max_segments_per_file: int = 32
    resize_to_224: bool = True


class ONNXCoughPredictor:
    """
    Complete ONNX-based cough detection pipeline.
    
    Replicates backend_api_actual_model.py using ONNX models.
    """
    def __init__(
        self,
        preprocessing_onnx_path: str,
        detector_onnx_path: str,
        config: Config = None
    ):
        """
        Initialize ONNX models.
        
        Args:
            preprocessing_onnx_path: Path to cough_preprocessing.onnx
            detector_onnx_path: Path to cough_detector_int8.onnx
            config: Configuration object (uses default if None)
        """
        if config is None:
            config = Config()
        self.config = config
        
        # Load ONNX models
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed. Install with: pip install onnxruntime"
            )
        
        print("="*70)
        print("Loading ONNX Models")
        print("="*70)
        
        # Load preprocessing model
        print(f"\n[1/2] Loading preprocessing model: {preprocessing_onnx_path}")
        if not os.path.exists(preprocessing_onnx_path):
            raise FileNotFoundError(f"Preprocessing ONNX not found: {preprocessing_onnx_path}")
        
        self.preprocessing_session = ort.InferenceSession(
            preprocessing_onnx_path,
            providers=['CPUExecutionProvider']
        )
        
        # Get preprocessing input/output info
        prep_inputs = self.preprocessing_session.get_inputs()
        prep_outputs = self.preprocessing_session.get_outputs()
        print(f"  Input: {prep_inputs[0].name} {prep_inputs[0].shape}")
        print(f"  Output: {prep_outputs[0].name} {prep_outputs[0].shape}")
        
        # Load detector model
        print(f"\n[2/2] Loading detector model: {detector_onnx_path}")
        if not os.path.exists(detector_onnx_path):
            raise FileNotFoundError(f"Detector ONNX not found: {detector_onnx_path}")
        
        self.detector_session = ort.InferenceSession(
            detector_onnx_path,
            providers=['CPUExecutionProvider']
        )
        
        # Get detector input/output info
        det_inputs = self.detector_session.get_inputs()
        det_outputs = self.detector_session.get_outputs()
        print(f"  Inputs:")
        for inp in det_inputs:
            print(f"    - {inp.name} {inp.shape}")
        print(f"  Outputs:")
        for out in det_outputs:
            print(f"    - {out.name} {out.shape}")
        
        print("\n[OK] Both ONNX models loaded successfully!")
        print("="*70)
    
    def load_audio(self, file_path: str) -> torch.Tensor:
        """
        Load and preprocess audio file.
        
        Replicates CoughAudioProcessor.load_audio() from precompute_spectrograms.py
        
        Args:
            file_path: Path to audio file
            
        Returns:
            waveform: (1, T) tensor in [-1, 1] range, 16kHz mono
        """
        waveform = None
        sr = None
        
        # Try torchaudio first
        try:
            waveform, sr = torchaudio.load(file_path, normalize=False)
            waveform = waveform.float()
            if waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()
        except Exception as e1:
            # Try with different backends if available
            try:
                # Try ffmpeg backend if available
                waveform, sr = torchaudio.load(file_path, backend="ffmpeg", normalize=False)
                waveform = waveform.float()
                if waveform.abs().max() > 0:
                    waveform = waveform / waveform.abs().max()
            except Exception as e2:
                # Fallback to librosa
                try:
                    import librosa
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        y, sr = librosa.load(file_path, sr=None, mono=False, dtype=np.float32)
                    if y.ndim == 1:
                        y = y[None, :]
                    waveform = torch.from_numpy(y).float()
                    if waveform.abs().max() > 0:
                        waveform = waveform / waveform.abs().max()
                except Exception as e3:
                    raise RuntimeError(
                        f"Failed to load audio with torchaudio ({e1}), "
                        f"ffmpeg backend ({e2}), and librosa ({e3})"
                    )
        
        # Ensure we have valid audio
        if waveform is None or waveform.numel() == 0:
            raise RuntimeError("Empty or invalid audio file")
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz
        if sr != self.config.sample_rate:
            resampler = T.Resample(sr, self.config.sample_rate)
            waveform = resampler(waveform)
        
        # Apply high-pass filter (50Hz cutoff)
        try:
            hpf = T.HighpassBiquad(sample_rate=self.config.sample_rate, cutoff_freq=50.0)
            waveform = hpf(waveform)
        except (AttributeError, RuntimeError):
            # High-pass filter not available, continue without it
            pass
        
        return waveform  # (1, T)
    
    def extract_segments(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract overlapping segments from waveform.
        
        Replicates CoughAudioProcessor.extract_segments() from precompute_spectrograms.py
        
        Args:
            waveform: (1, T) audio waveform
            
        Returns:
            segments: List of (1, segment_samples) tensors
        """
        seg_samples = int(self.config.segment_duration * self.config.sample_rate)
        hop_samples = int(self.config.hop_length * self.config.sample_rate)
        
        # Pad if too short
        if waveform.shape[1] < seg_samples:
            waveform = torch.nn.functional.pad(waveform, (0, seg_samples - waveform.shape[1]))
        
        segments = []
        start = 0
        while start + seg_samples <= waveform.shape[1]:
            segments.append(waveform[:, start:start + seg_samples])
            start += hop_samples
        
        # Ensure at least one segment
        if not segments:
            segments.append(waveform[:, :seg_samples])
        
        # Cap to max_segments_per_file
        segments = segments[:self.config.max_segments_per_file]
        
        return segments
    
    def preprocess_segment(self, segment: torch.Tensor) -> np.ndarray:
        """
        Convert waveform segment to spectrogram using preprocessing ONNX.
        
        Args:
            segment: (1, segment_samples) waveform tensor
            
        Returns:
            spectrogram: (1, 3, 224, 224) normalized spectrogram
        """
        # Convert to numpy and ensure correct shape
        segment_np = segment.squeeze(0).numpy().astype(np.float32)  # (segment_samples,)
        
        # Run preprocessing ONNX
        # Input: (batch, segment_samples)
        # Output: (batch, 3, 224, 224)
        input_name = self.preprocessing_session.get_inputs()[0].name
        output = self.preprocessing_session.run(None, {input_name: segment_np[np.newaxis, :]})[0]
        
        return output  # (1, 3, 224, 224)
    
    def predict(self, audio_path: str, threshold: float = 0.61) -> dict:
        """
        Run complete prediction pipeline on audio file.
        
        Args:
            audio_path: Path to audio file
            threshold: Decision threshold for cough detection (default: 0.61)
            
        Returns:
            Dictionary with prediction results:
            {
                'coughDetected': bool,
                'confidence': float,
                'file_probability': float,
                'segment_probabilities': List[float],
                'segment_timestamps': List[Tuple[float, float]],
                'num_segments': int,
                'threshold_used': float,
                'max_segment_probability': float,
                'mean_segment_probability': float
            }
        """
        print("\n" + "="*70)
        print(f"Processing Audio: {audio_path}")
        print("="*70)
        
        # 1. Load audio
        print("\n[1/4] Loading audio...")
        waveform = self.load_audio(audio_path)
        duration = waveform.shape[1] / self.config.sample_rate
        print(f"  Loaded: {waveform.shape[1]} samples ({duration:.2f} seconds)")
        
        # 2. Extract segments
        print("\n[2/4] Extracting segments...")
        segments = self.extract_segments(waveform)
        num_segments = len(segments)
        print(f"  Extracted {num_segments} segments")
        
        if num_segments == 0:
            return {
                'coughDetected': False,
                'error': 'No segments extracted',
                'confidence': 0.0,
                'file_probability': 0.0,
                'segment_probabilities': [],
                'segment_timestamps': [],
                'num_segments': 0,
                'threshold_used': threshold
            }
        
        # 3. Preprocess segments using ONNX
        print("\n[3/4] Preprocessing segments with ONNX...")
        spectrograms = []
        for i, seg in enumerate(segments):
            spec = self.preprocess_segment(seg)  # (1, 3, 224, 224)
            # Remove batch dimension: (1, 3, 224, 224) -> (3, 224, 224)
            spec = spec.squeeze(0)  # (3, 224, 224)
            spectrograms.append(spec)
        
        # Stack spectrograms: (num_segments, 3, 224, 224) -> (1, num_segments, 3, 224, 224)
        batch_spectrograms = np.stack(spectrograms, axis=0)  # (num_segments, 3, 224, 224)
        batch_spectrograms = batch_spectrograms[np.newaxis, ...]  # (1, num_segments, 3, 224, 224)
        
        # Create mask for valid segments
        mask = np.ones((1, num_segments), dtype=bool)
        
        # Pad to max_segments if needed
        if num_segments < self.config.max_segments_per_file:
            padding = self.config.max_segments_per_file - num_segments
            # Pad spectrograms with zeros
            pad_shape = (1, padding, 3, 224, 224)
            batch_spectrograms = np.concatenate([
                batch_spectrograms,
                np.zeros(pad_shape, dtype=batch_spectrograms.dtype)
            ], axis=1)
            # Pad mask with False
            mask_pad = np.zeros((1, padding), dtype=bool)
            mask = np.concatenate([mask, mask_pad], axis=1)
        
        print(f"  Batch shape: {batch_spectrograms.shape}")
        print(f"  Mask shape: {mask.shape}")
        
        # 4. Run detector ONNX
        print("\n[4/4] Running detector ONNX model...")
        det_inputs = self.detector_session.get_inputs()
        input_dict = {}
        for inp in det_inputs:
            if 'spectrogram' in inp.name.lower() or 'x' in inp.name.lower():
                input_dict[inp.name] = batch_spectrograms.astype(np.float32)
            elif 'mask' in inp.name.lower():
                input_dict[inp.name] = mask
        
        outputs = self.detector_session.run(None, input_dict)
        
        # Parse outputs
        # Expected outputs: bag_probability, segment_probabilities, segment_logits, bag_logit
        output_names = [out.name for out in self.detector_session.get_outputs()]
        
        # Find bag probability (usually first output or contains 'bag'/'prob')
        bag_prob = None
        seg_probs = None
        
        for i, name in enumerate(output_names):
            if 'bag' in name.lower() and 'prob' in name.lower():
                bag_prob = outputs[i]
            elif 'segment' in name.lower() and 'prob' in name.lower():
                seg_probs = outputs[i]
        
        # Fallback: use first output as bag_prob
        if bag_prob is None:
            bag_prob = outputs[0]
        if seg_probs is None and len(outputs) > 1:
            seg_probs = outputs[1]
        
        # Extract values
        if isinstance(bag_prob, np.ndarray):
            bag_prob = float(bag_prob.item() if bag_prob.size == 1 else bag_prob[0])
        else:
            bag_prob = float(bag_prob)
        
        if seg_probs is not None:
            if isinstance(seg_probs, np.ndarray):
                seg_probs_list = seg_probs[0, :num_segments].tolist()  # Only valid segments
            else:
                seg_probs_list = seg_probs[:num_segments] if isinstance(seg_probs, list) else []
        else:
            seg_probs_list = []
        
        # Build timestamps
        timestamps = []
        for i in range(num_segments):
            start = i * self.config.hop_length
            end = start + self.config.segment_duration
            timestamps.append((float(start), float(end)))
        
        # Calculate statistics
        max_seg_prob = float(max(seg_probs_list)) if seg_probs_list else 0.0
        mean_seg_prob = float(sum(seg_probs_list) / len(seg_probs_list)) if seg_probs_list else 0.0
        
        # Decision
        is_cough = bag_prob > threshold
        
        print(f"\n  Bag probability: {bag_prob:.4f}")
        print(f"  Threshold: {threshold}")
        print(f"  Cough detected: {is_cough}")
        print(f"  Max segment prob: {max_seg_prob:.4f}")
        print(f"  Mean segment prob: {mean_seg_prob:.4f}")
        
        return {
            'coughDetected': is_cough,
            'confidence': float(bag_prob),
            'file_probability': float(bag_prob),
            'segment_probabilities': seg_probs_list,
            'segment_timestamps': timestamps,
            'num_segments': num_segments,
            'threshold_used': float(threshold),
            'max_segment_probability': max_seg_prob,
            'mean_segment_probability': mean_seg_prob
        }


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Cough detection using ONNX models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_with_onnx.py audio.wav
  python predict_with_onnx.py audio.wav --threshold 0.7
  python predict_with_onnx.py audio.wav --preprocessing model1.onnx --detector model2.onnx
        """
    )
    
    parser.add_argument(
        'audio_path',
        type=str,
        help='Path to audio file'
    )
    
    parser.add_argument(
        '--preprocessing',
        type=str,
        default='cough_preprocessing.onnx',
        help='Path to preprocessing ONNX model (default: cough_preprocessing.onnx)'
    )
    
    parser.add_argument(
        '--detector',
        type=str,
        default='cough_detector_int8.onnx',
        help='Path to detector ONNX model (default: cough_detector_int8.onnx)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.61,
        help='Decision threshold for cough detection (default: 0.61)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional: Save results to JSON file'
    )
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_path):
        print(f"ERROR: Audio file not found: {args.audio_path}")
        sys.exit(1)
    
    # Initialize predictor
    try:
        predictor = ONNXCoughPredictor(
            preprocessing_onnx_path=args.preprocessing,
            detector_onnx_path=args.detector
        )
    except Exception as e:
        print(f"ERROR: Failed to load ONNX models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run prediction
    try:
        results = predictor.predict(args.audio_path, threshold=args.threshold)
    except Exception as e:
        print(f"ERROR: Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"\nCough Detected: {results['coughDetected']}")
    print(f"Confidence: {results['confidence']:.4f}")
    print(f"File Probability: {results['file_probability']:.4f}")
    print(f"Number of Segments: {results['num_segments']}")
    print(f"Threshold Used: {results['threshold_used']:.4f}")
    print(f"Max Segment Probability: {results['max_segment_probability']:.4f}")
    print(f"Mean Segment Probability: {results['mean_segment_probability']:.4f}")
    
    if results.get('segment_probabilities'):
        print(f"\nSegment Probabilities:")
        for i, (prob, (start, end)) in enumerate(zip(
            results['segment_probabilities'],
            results['segment_timestamps']
        )):
            print(f"  Segment {i+1} [{start:.1f}s - {end:.1f}s]: {prob:.4f}")
    
    # Save to JSON if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print("="*70)


if __name__ == "__main__":
    main()

