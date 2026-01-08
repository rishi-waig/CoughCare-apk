"""
Example: Using ONNX model for inference (alternative to PyTorch).
This shows how to use the converted ONNX model instead of the PyTorch checkpoint.
"""

import onnxruntime as ort
import numpy as np
from precompute_spectrograms import CoughAudioProcessor, Config
import torch


class ONNXCoughModel:
    """
    ONNX-based cough detection model.
    Uses ONNX Runtime instead of PyTorch for inference.
    """
    
    def __init__(self, onnx_path: str):
        """
        Initialize ONNX model.
        
        Args:
            onnx_path: Path to ONNX model file
        """
        print(f"Loading ONNX model from: {onnx_path}")
        
        # Create ONNX Runtime session
        # Use CPU provider (can also use CUDAExecutionProvider for GPU)
        providers = ['CPUExecutionProvider']
        if ort.get_device() == 'GPU':
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"  Inputs: {self.input_names}")
        print(f"  Outputs: {self.output_names}")
        
        # Initialize audio processor
        self.config = Config()
        self.processor = CoughAudioProcessor(self.config)
        
        # ImageNet normalization (matching training)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        
        print("✓ ONNX model loaded successfully!")
    
    def _build_batch(self, segments):
        """
        Build batch from waveform segments.
        segments: list of (1, T) waveforms
        Returns: batch (1, S, 3, H, W), mask (1, S)
        """
        # Cap segments like training
        if len(segments) > self.config.max_segments_per_file:
            segments = segments[:self.config.max_segments_per_file]
        
        specs = []
        for seg in segments:
            # Convert each waveform segment to 3-channel mel spectrogram
            spec3 = self.processor.waveform_to_mel_3ch(seg)  # (3, H, W) in [-1,1]
            specs.append(spec3)
        
        x = torch.stack(specs).unsqueeze(0)  # (1, S, 3, H, W)
        
        # Map [-1,1] → [0,1] then ImageNet normalize
        x01 = (x + 1) / 2.0
        x_norm = (x01 - self.mean) / self.std
        
        mask = torch.ones(1, x.shape[1], dtype=torch.bool)
        
        return x_norm, mask
    
    def preprocess_audio(self, audio_path: str):
        """
        Preprocess audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            batch: (1, S, 3, H, W) numpy array
            mask: (1, S) boolean numpy array
        """
        # Load audio
        wav = self.processor.load_audio(audio_path)
        
        # Extract segments
        segments = self.processor.extract_segments(wav)
        
        if len(segments) == 0:
            return None, None
        
        # Build batch
        batch, mask = self._build_batch(segments)
        
        # Convert to numpy for ONNX Runtime
        batch_np = batch.cpu().numpy().astype(np.float32)
        mask_np = mask.cpu().numpy().astype(np.bool_)
        
        return batch_np, mask_np
    
    def predict(self, audio_path: str):
        """
        Run inference on audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict with prediction results
        """
        # Preprocess
        batch, mask = self.preprocess_audio(audio_path)
        
        if batch is None:
            return {
                'error': 'Failed to process audio',
                'probability': 0.0
            }
        
        # Run ONNX inference
        outputs = self.session.run(
            self.output_names,
            {
                self.input_names[0]: batch,  # spectrograms
                self.input_names[1]: mask     # segment_mask
            }
        )
        
        # Extract outputs
        bag_prob = outputs[0][0, 0]  # (batch_size,) -> scalar
        segment_probs = outputs[1][0]  # (batch_size, num_segments) -> (num_segments,)
        segment_logits = outputs[2][0]
        bag_logit = outputs[3][0, 0]
        
        # Build timestamps
        num_segments = int(batch.shape[1])
        timestamps = []
        for i in range(num_segments):
            start = i * self.config.hop_length
            end = start + self.config.segment_duration
            timestamps.append((start, end))
        
        return {
            'probability': float(bag_prob),
            'is_cough': bag_prob > 0.61,
            'tb_detected': bag_prob > 1.0,
            'segment_probabilities': segment_probs.tolist(),
            'segment_timestamps': timestamps,
            'num_segments': num_segments
        }


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python example_onnx_inference.py <onnx_model_path> <audio_file_path>")
        sys.exit(1)
    
    onnx_path = sys.argv[1]
    audio_path = sys.argv[2]
    
    # Load ONNX model
    model = ONNXCoughModel(onnx_path)
    
    # Run prediction
    print(f"\nRunning inference on: {audio_path}")
    result = model.predict(audio_path)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"\nResults:")
        print(f"  Probability: {result['probability']:.4f}")
        print(f"  Cough detected: {result['is_cough']}")
        print(f"  TB detected: {result['tb_detected']}")
        print(f"  Number of segments: {result['num_segments']}")
        print(f"  Segment probabilities: {result['segment_probabilities'][:5]}...")


if __name__ == '__main__':
    main()


