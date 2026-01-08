"""
Backend API using Quantized ONNX Models for Cough Detection
Supports both int8 and float16 quantized models for maximum efficiency
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

# Model paths - try int8 first, then fp16, then original
ONNX_MODEL_INT8 = os.environ.get('ONNX_MODEL_INT8', 'cough_detector_int8.onnx')
ONNX_MODEL_FP16 = os.environ.get('ONNX_MODEL_FP16', 'cough_detector_fp16.onnx')
ONNX_MODEL_ORIGINAL = os.environ.get('ONNX_MODEL_PATH', 'cough_detector_attention.onnx')

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
            try:
                waveform, sample_rate = torchaudio.load(path)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                if sample_rate != self.config.sample_rate:
                    resampler = T.Resample(sample_rate, self.config.sample_rate)
                    waveform = resampler(waveform)

                return waveform
            except Exception as e2:
                raise ValueError(f"Could not load audio from {path}: {e}, {e2}")

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


class QuantizedONNXCoughModel:
    """Quantized ONNX-based cough detection model"""

    def __init__(self):
        print("Initializing Quantized ONNX cough detection model...")

        self.config = AudioConfig()
        self.model_type = None
        self.model_path = None
        self.model_size_mb = 0

        # Try loading models in order of preference: int8 > fp16 > original
        script_dir = os.path.dirname(os.path.abspath(__file__))

        models_to_try = [
            ('int8', os.path.join(script_dir, ONNX_MODEL_INT8)),
            ('fp16', os.path.join(script_dir, ONNX_MODEL_FP16)),
            ('original', os.path.join(script_dir, ONNX_MODEL_ORIGINAL)),
        ]

        for model_type, model_path in models_to_try:
            if os.path.exists(model_path):
                try:
                    print(f"  Trying {model_type} model: {model_path}")
                    self._load_model(model_path)
                    self.model_type = model_type
                    self.model_path = model_path
                    self.model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
                    print(f"  Loaded {model_type} model successfully ({self.model_size_mb:.2f} MB)")
                    break
                except Exception as e:
                    print(f"  Failed to load {model_type}: {e}")
                    continue

        if self.session is None:
            raise RuntimeError("No ONNX model could be loaded")

        # Initialize audio processor
        self.processor = CoughAudioProcessor(self.config)

        # ImageNet normalization constants
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1)

        # Store validation metrics
        self.val_auc = 0.9904
        self.epoch = 3

        print(f"Model ready: {self.model_type} ({self.model_size_mb:.2f} MB)")

    def _load_model(self, model_path):
        """Load ONNX model with optimizations"""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)

    def _build_batch(self, segments):
        """Build batch from waveform segments."""
        if len(segments) > self.config.max_segments_per_file:
            segments = segments[:self.config.max_segments_per_file]

        specs = []
        for seg in segments:
            spec3 = self.processor.waveform_to_mel_3ch(seg)
            specs.append(spec3.numpy())

        x = np.stack(specs)[np.newaxis, ...]

        # ImageNet normalize
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

            wav = self.processor.load_audio(tmp_path)
            print(f"  Loaded audio: {wav.shape[1]} samples")

            segments = self.processor.extract_segments(wav)
            print(f"  Extracted {len(segments)} segments")

            if len(segments) == 0:
                return None, None

            batch, mask = self._build_batch(segments)
            print(f"  Built batch: {batch.shape}")

            return batch, mask

        except Exception as e:
            print(f"Preprocessing error: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def predict(self, audio_bytes, saved_path=None):
        """Run quantized ONNX model inference."""
        try:
            if saved_path:
                batch, mask = self.preprocess_audio(audio_bytes, temp_path=saved_path)
            else:
                batch, mask = self.preprocess_audio(audio_bytes)

            if batch is None:
                return None, 0.0, [], 0, []

            # Run inference
            outputs = self.session.run(
                None,
                {
                    'spectrograms': batch,
                    'segment_mask': mask
                }
            )

            bag_prob = outputs[0]
            seg_probs = outputs[1]

            probability = float(bag_prob[0])

            try:
                seg_list = seg_probs[0].tolist()
            except:
                seg_list = []

            is_cough = probability > 0.61
            tb_detected = probability > 1

            num_segments = int(batch.shape[1])
            timestamps = []
            for i in range(num_segments):
                start = i * self.config.hop_length
                end = start + self.config.segment_duration
                timestamps.append((start, end))

            print(f"  {self.model_type.upper()} output: prob={probability:.4f}, cough={is_cough}")

            return is_cough, probability, seg_list, num_segments, timestamps

        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0, [], 0, []


# Initialize Model
print("\n" + "="*70)
print("CoughCare Backend - Quantized ONNX Model")
print("="*70)

try:
    model = QuantizedONNXCoughModel()
    model_loaded = True
    print(f"\nQUANTIZED MODEL LOADED: {model.model_type.upper()}")
    print(f"Model size: {model.model_size_mb:.2f} MB")
except Exception as e:
    print(f"\nFAILED TO LOAD MODEL")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    model_loaded = False
    model = None

print("="*70 + "\n")


@app.route('/api/detect-cough', methods=['POST'])
def detect_cough_endpoint():
    """Detect cough using quantized ONNX model"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503

    saved_audio_path = None
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        audio_bytes = audio_file.read()

        print(f"\nRunning {model.model_type.upper()} inference on {len(audio_bytes)} bytes")

        # Save audio
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = audio_file.filename or 'cough_audio'
        saved_audio_path = os.path.join(AUDIO_SAVE_DIR, f"{timestamp}_{filename}")
        with open(saved_audio_path, 'wb') as f:
            f.write(audio_bytes)

        # Get threshold
        threshold = 0.61
        try:
            th_s = request.form.get('threshold') or request.args.get('threshold')
            if th_s:
                threshold = float(th_s)
        except:
            pass

        # Run inference
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
                'error': 'No cough pattern detected',
                'confidence': probability,
                'file_probability': float(probability),
                'segment_probabilities': seg_list,
                'segment_timestamps': timestamps,
                'num_segments': num_segments,
                'threshold_used': float(threshold),
                'max_segment_probability': max_segment,
                'mean_segment_probability': mean_segment,
            })

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
            'message': f'Analyzed by {model.model_type.upper()} ONNX model',
            'model_info': {
                'validation_auc': float(model.val_auc),
                'model_type': model.model_type,
                'model_size_mb': model.model_size_mb,
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
    """Analyze TB using quantized model"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        audio_bytes = audio_file.read()

        tb_threshold = 1.0
        try:
            th_s = request.form.get('tb_threshold') or request.args.get('tb_threshold')
            if th_s:
                tb_threshold = float(th_s)
        except:
            pass

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
            'model_type': model.model_type
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
        'model_type': model.model_type if model_loaded else None,
        'model_size_mb': model.model_size_mb if model_loaded else None,
        'validation_auc': float(model.val_auc) if model_loaded else None,
        'runtime': 'onnxruntime',
        'mode': 'QUANTIZED_ONNX_INFERENCE'
    })


@app.route('/samples/<filename>', methods=['GET'])
def serve_sample(filename):
    """Serve sample audio files"""
    try:
        sample_path = os.path.join(AUDIO_SAVE_DIR, filename)
        if not os.path.exists(sample_path):
            return jsonify({'error': 'File not found'}), 404
        from flask import send_file
        return send_file(sample_path, mimetype='audio/wav')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/samples', methods=['GET'])
def list_samples():
    """List sample files"""
    try:
        samples = []
        if os.path.exists(AUDIO_SAVE_DIR):
            for f in os.listdir(AUDIO_SAVE_DIR):
                if f.endswith('.wav'):
                    fp = os.path.join(AUDIO_SAVE_DIR, f)
                    samples.append({'filename': f, 'size': os.path.getsize(fp)})
        return jsonify({'samples': samples})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def home():
    """Home"""
    return jsonify({
        'service': 'CoughCare Backend API',
        'version': '6.0.0-quantized',
        'model': f'AttnMILResNet - {model.model_type.upper() if model_loaded else "N/A"} Quantized',
        'model_loaded': model_loaded,
        'model_size_mb': model.model_size_mb if model_loaded else None,
        'mode': 'QUANTIZED_ONNX_INFERENCE',
        'runtime': 'onnxruntime'
    })


if __name__ == '__main__':
    if model_loaded:
        print("="*70)
        print(f"SERVER READY WITH {model.model_type.upper()} QUANTIZED MODEL")
        print(f"  Model size: {model.model_size_mb:.2f} MB")
        print(f"  Validation AUC: {model.val_auc:.4f}")
        print("="*70)
    else:
        print("Server starting WITHOUT model")

    PORT = int(os.environ.get('PORT', 5000))
    print(f"\nServer: http://localhost:{PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=True)
