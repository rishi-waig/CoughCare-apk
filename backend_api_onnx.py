"""
Backend API using ONNX Model for Cough Detection
Lightweight, fast inference using ONNX Runtime - no PyTorch required at runtime
"""

import os
import sys

# Fix OpenMP duplicate library issue on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import onnxruntime as ort
import io
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Audio processing imports
import torchaudio
import torchaudio.transforms as T
import torch
import torch.nn.functional as F
from pydub import AudioSegment

# Configure ffmpeg/ffprobe paths for pydub if bundled locally
FFMPEG_DIR = os.path.join(os.path.dirname(__file__), 'ffmpeg_portable', 'ffmpeg-master-latest-win64-gpl', 'bin')
FFMPEG_PATH = os.path.join(FFMPEG_DIR, 'ffmpeg.exe')
FFPROBE_PATH = os.path.join(FFMPEG_DIR, 'ffprobe.exe')

if os.path.exists(FFMPEG_DIR):
    os.environ['PATH'] = FFMPEG_DIR + os.pathsep + os.environ.get('PATH', '')

if os.path.exists(FFMPEG_PATH):
    AudioSegment.converter = FFMPEG_PATH
    AudioSegment.ffmpeg = FFMPEG_PATH
    os.environ.setdefault('FFMPEG_BINARY', FFMPEG_PATH)
    print(f"Using bundled ffmpeg at: {FFMPEG_PATH}")

if os.path.exists(FFPROBE_PATH):
    AudioSegment.ffprobe = FFPROBE_PATH
    os.environ.setdefault('FFPROBE_BINARY', FFPROBE_PATH)
    print(f"Using bundled ffprobe at: {FFPROBE_PATH}")

app = Flask(__name__)
CORS(app)

# ONNX Model path
ONNX_MODEL_PATH = os.environ.get('ONNX_MODEL_PATH', 'cough_detector_attention.onnx')

# Directory to save uploaded audio files
AUDIO_SAVE_DIR = os.environ.get('AUDIO_SAVE_DIR', '/app/uploaded_audio')
AUDIO_SAVE_DIR = os.path.abspath(AUDIO_SAVE_DIR)
os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)
print(f"AUDIO_SAVE_DIR set to: {AUDIO_SAVE_DIR}")


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
        """Load audio file and return waveform tensor (1, T) at config.sample_rate.
        Matches training pipeline exactly: torchaudio primary, normalize to [-1,1], high-pass filter.
        """
        waveform = None
        sample_rate = None

        # Try torchaudio first (matches training)
        try:
            waveform, sample_rate = torchaudio.load(path, normalize=False)
            waveform = waveform.float()
            # Normalize to [-1, 1] by dividing by max (matches training exactly)
            if waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()
        except Exception as e1:
            # Try pydub as fallback
            try:
                audio = AudioSegment.from_file(path)
                if audio.channels > 1:
                    audio = audio.set_channels(1)

                sample_rate = audio.frame_rate
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

                # Normalize based on bit depth
                if audio.sample_width == 1:
                    samples = samples / 128.0 - 1.0
                elif audio.sample_width == 2:
                    samples = samples / 32768.0
                elif audio.sample_width == 4:
                    samples = samples / 2147483648.0
                else:
                    samples = samples / (2 ** (8 * audio.sample_width - 1))

                waveform = torch.from_numpy(samples).unsqueeze(0)

                # Normalize to [-1, 1] by dividing by max (matches training)
                if waveform.abs().max() > 0:
                    waveform = waveform / waveform.abs().max()

            except Exception as e2:
                raise ValueError(f"Could not load audio from {path}: {e1}, {e2}")

        # Ensure we have valid audio
        if waveform is None or waveform.numel() == 0:
            raise ValueError("Empty or invalid audio file")

        # Convert to mono (matches training)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed (matches training)
        if sample_rate != self.config.sample_rate:
            resampler = T.Resample(sample_rate, self.config.sample_rate)
            waveform = resampler(waveform)

        # Apply high-pass filter at 50Hz (matches training exactly!)
        try:
            hpf = T.HighpassBiquad(sample_rate=self.config.sample_rate, cutoff_freq=50.0)
            waveform = hpf(waveform)
        except (AttributeError, RuntimeError):
            # High-pass filter not available or failed, continue without it
            pass

        return waveform  # (1, T)

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


class ONNXCoughModel:
    """ONNX-based cough detection model for fast inference"""

    def __init__(self, onnx_path):
        print("Initializing ONNX cough detection model...")

        self.config = AudioConfig()

        print(f"  Loading ONNX model from: {onnx_path}")

        # Create ONNX Runtime session with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Use available providers (CPU, CUDA if available)
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
            print("  Using CUDA for inference")
        else:
            print("  Using CPU for inference")

        self.session = ort.InferenceSession(onnx_path, sess_options, providers=providers)

        # Get model metadata
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        print(f"  Model inputs: {self.input_names}")
        print(f"  Model outputs: {self.output_names}")

        # Initialize audio processor
        self.processor = CoughAudioProcessor(self.config)

        # ImageNet normalization constants
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1)

        # Store validation metrics (hardcoded from training)
        self.val_auc = 0.9904
        self.epoch = 3

        print("ONNX model ready for inference!")

    def _build_batch(self, segments):
        """Build batch from waveform segments."""
        if len(segments) > self.config.max_segments_per_file:
            segments = segments[:self.config.max_segments_per_file]

        specs = []
        for seg in segments:
            spec3 = self.processor.waveform_to_mel_3ch(seg)
            specs.append(spec3.numpy())

        x = np.stack(specs)[np.newaxis, ...]  # (1, S, 3, H, W)

        # Map [-1,1] -> [0,1] then ImageNet normalize
        x01 = (x + 1) / 2.0
        x_norm = (x01 - self.mean) / self.std

        mask = np.ones((1, x.shape[1]), dtype=bool)

        return x_norm.astype(np.float32), mask

    def preprocess_audio(self, audio_bytes, temp_path=None):
        """Preprocess audio for ONNX inference."""
        try:
            if temp_path is None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
                    tmp_path = tmp_file.name
                    tmp_file.write(audio_bytes)
            else:
                tmp_path = temp_path
                with open(tmp_path, 'wb') as f:
                    f.write(audio_bytes)

            try:
                wav = self.processor.load_audio(tmp_path)
                print(f"  Loaded audio: {wav.shape[1]} samples")

                segments = self.processor.extract_segments(wav)
                print(f"  Extracted {len(segments)} waveform segments")

                if len(segments) == 0:
                    print("No segments extracted from audio")
                    return None, None

                batch, mask = self._build_batch(segments)
                print(f"  Built batch: {batch.shape}, mask: {mask.shape}")

                return batch, mask

            finally:
                pass

        except Exception as e:
            print(f"Preprocessing error: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def predict(self, audio_bytes, saved_path=None):
        """Run ONNX model inference."""
        try:
            if saved_path:
                batch, mask = self.preprocess_audio(audio_bytes, temp_path=saved_path)
            else:
                batch, mask = self.preprocess_audio(audio_bytes)

            if batch is None:
                return None, 0.0, [], 0, []

            # Run ONNX inference
            outputs = self.session.run(
                self.output_names,
                {
                    'spectrograms': batch,
                    'segment_mask': mask
                }
            )

            # Parse outputs
            bag_prob = outputs[0]  # bag_probability
            seg_probs = outputs[1]  # segment_probabilities

            probability = float(bag_prob[0])

            try:
                seg_list = seg_probs[0].tolist()
            except Exception:
                seg_list = []

            is_cough = probability > 0.61
            tb_detected = probability > 1

            num_segments = int(batch.shape[1])
            timestamps = []
            for i in range(num_segments):
                start = i * self.config.hop_length
                end = start + self.config.segment_duration
                timestamps.append((start, end))

            print(f"  ONNX output: prob={probability:.4f}, cough={is_cough}, TB={tb_detected}")

            return is_cough, probability, seg_list, num_segments, timestamps

        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0, [], 0, []


# Initialize ONNX Model
print("\n" + "="*70)
print("CoughCare Backend - ONNX Model (Lightweight)")
print("="*70)

try:
    model = ONNXCoughModel(ONNX_MODEL_PATH)
    model_loaded = True
    print("\nONNX MODEL LOADED AND READY")
except Exception as e:
    print(f"\nFAILED TO LOAD ONNX MODEL")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    model_loaded = False
    model = None

print("="*70 + "\n")


@app.route('/api/detect-cough', methods=['POST'])
def detect_cough_endpoint():
    """Detect cough using ONNX model"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503

    saved_audio_path = None
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        audio_bytes = audio_file.read()

        print(f"\nRunning ONNX inference on {len(audio_bytes)} bytes")

        # Save audio file locally
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = audio_file.filename or 'cough_audio'
        file_ext = os.path.splitext(filename)[1] or '.webm'
        saved_audio_path = os.path.join(AUDIO_SAVE_DIR, f"{timestamp}_{filename}")
        with open(saved_audio_path, 'wb') as f:
            f.write(audio_bytes)
        print(f"  Saved audio to: {saved_audio_path}")

        # Get threshold
        threshold = None
        try:
            th_s = request.form.get('threshold', None)
            if th_s is None:
                th_s = request.args.get('threshold', None)
            if th_s is not None:
                threshold = float(th_s)
        except Exception:
            threshold = None
        if threshold is None or not (0.0 <= threshold <= 1.0):
            threshold = 0.61

        # Run ONNX inference
        is_cough_model, probability, seg_list, num_segments, timestamps = model.predict(audio_bytes, saved_path=saved_audio_path)

        if is_cough_model is None:
            return jsonify({'error': 'Failed to process audio'}), 500

        is_cough = probability > threshold
        tb_detected = probability > 1

        max_segment = float(max(seg_list)) if seg_list else 0.0
        mean_segment = float(sum(seg_list) / len(seg_list)) if seg_list else 0.0

        if not is_cough:
            return jsonify({
                'coughDetected': False,
                'error': 'No cough pattern detected by model',
                'confidence': probability,
                'file_probability': float(probability),
                'segment_probabilities': seg_list,
                'segment_timestamps': timestamps,
                'num_segments': num_segments,
                'threshold_used': float(threshold),
                'max_segment_probability': max_segment,
                'mean_segment_probability': mean_segment,
            })

        print(f"ONNX result: Cough={is_cough}, TB={tb_detected}, Prob={probability:.4f}")

        return jsonify({
            'coughDetected': True,
            'tbDetected': tb_detected,
            'confidence': float(probability),
            'file_probability': float(probability),
            'segment_probabilities': seg_list,
            'segment_timestamps': timestamps,
            'num_segments': num_segments,
            'threshold_used': float(threshold),
            'max_segment_probability': max_segment,
            'mean_segment_probability': mean_segment,
            'message': f'Analyzed by ONNX model (Val AUC: {model.val_auc:.4f})',
            'model_info': {
                'validation_auc': float(model.val_auc),
                'epoch': model.epoch,
                'runtime': 'ONNX'
            }
        })

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze-tb', methods=['POST'])
def analyze_tb_endpoint():
    """Analyze TB using ONNX model"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        audio_bytes = audio_file.read()

        tb_threshold = None
        try:
            th_s = request.form.get('tb_threshold', None)
            if th_s is None:
                th_s = request.args.get('tb_threshold', None)
            if th_s is not None:
                tb_threshold = float(th_s)
        except Exception:
            tb_threshold = None
        if tb_threshold is None or not (0.0 <= tb_threshold <= 1.0):
            tb_threshold = 1

        is_cough, probability, seg_list, num_segments, timestamps = model.predict(audio_bytes)

        if is_cough is None:
            return jsonify({'error': 'Failed to analyze'}), 500

        tb_detected = probability > tb_threshold

        return jsonify({
            'tbDetected': tb_detected,
            'prediction': 'TB' if tb_detected else 'No TB',
            'file_probability': float(probability),
            'segment_probabilities': seg_list,
            'segment_timestamps': timestamps,
            'num_segments': num_segments,
            'tb_threshold_used': float(tb_threshold),
            'message': f'ONNX model prediction (Val AUC: {model.val_auc:.4f})'
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy' if model_loaded else 'error',
        'model_loaded': model_loaded,
        'model_type': 'ONNX',
        'validation_auc': float(model.val_auc) if model_loaded else None,
        'runtime': 'onnxruntime',
        'mode': 'ONNX_INFERENCE'
    })


@app.route('/samples/<filename>', methods=['GET'])
def serve_sample(filename):
    """Serve sample audio files"""
    try:
        sample_path = os.path.join(AUDIO_SAVE_DIR, filename)
        if not os.path.exists(sample_path):
            return jsonify({'error': f'Sample file not found'}), 404

        from flask import send_file
        return send_file(sample_path, mimetype='audio/wav')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/samples', methods=['GET'])
def list_samples():
    """List available sample files"""
    try:
        samples = []
        if os.path.exists(AUDIO_SAVE_DIR):
            for filename in os.listdir(AUDIO_SAVE_DIR):
                if filename.endswith('.wav'):
                    filepath = os.path.join(AUDIO_SAVE_DIR, filename)
                    size = os.path.getsize(filepath)
                    samples.append({
                        'filename': filename,
                        'size': size,
                        'url': f'/samples/{filename}'
                    })
        return jsonify({'samples': samples})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def home():
    """Home"""
    return jsonify({
        'service': 'CoughCare Backend API',
        'version': '5.0.0-onnx',
        'model': 'AttnMILResNet - ONNX Optimized',
        'model_loaded': model_loaded,
        'validation_auc': float(model.val_auc) if model_loaded else None,
        'mode': 'ONNX_INFERENCE',
        'runtime': 'onnxruntime'
    })


if __name__ == '__main__':
    if model_loaded:
        print("="*70)
        print("SERVER READY WITH ONNX MODEL")
        print(f"  Model: AttnMILResNet (ONNX)")
        print(f"  Validation AUC: {model.val_auc:.4f}")
        print(f"  Runtime: ONNX Runtime")
        print(f"  Mode: LIGHTWEIGHT INFERENCE")
        print("="*70)
    else:
        print("Server starting WITHOUT model - all requests will fail")

    PORT = int(os.environ.get('PORT', 5000))
    print(f"\nServer: http://localhost:{PORT}")
    print("Press Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=PORT, debug=True)
