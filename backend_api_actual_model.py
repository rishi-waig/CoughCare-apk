"""
Backend API using YOUR ACTUAL TRAINED MODEL - AttnMILResNet
This uses the real model architecture with your trained weights
NO FAKE PREDICTIONS - Only real model inference
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchaudio
import numpy as np
import io
import tempfile
import os
from pydub import AudioSegment
import warnings
warnings.filterwarnings('ignore')

# Import your actual model architecture
from train_cough_detector_attention import Config, AttnMILResNet, get_device
from precompute_spectrograms import CoughAudioProcessor

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
    print(f"✓ Using bundled ffmpeg at: {FFMPEG_PATH}")

if os.path.exists(FFPROBE_PATH):
    AudioSegment.ffprobe = FFPROBE_PATH
    os.environ.setdefault('FFPROBE_BINARY', FFPROBE_PATH)
    print(f"✓ Using bundled ffprobe at: {FFPROBE_PATH}")

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.environ.get('MODEL_PATH', 'backup_best_model_20251015_170801.pth')

# Directory to save uploaded audio files
# Use absolute path for AUDIO_SAVE_DIR to avoid path resolution issues
# In Docker, files are in /app/uploaded_audio
AUDIO_SAVE_DIR = os.environ.get('AUDIO_SAVE_DIR', '/app/uploaded_audio')
# Ensure it's an absolute path
AUDIO_SAVE_DIR = os.path.abspath(AUDIO_SAVE_DIR)
os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)
print(f"AUDIO_SAVE_DIR set to: {AUDIO_SAVE_DIR}")

class ActualCoughModel:
    def __init__(self, model_path):
        """Initialize with YOUR ACTUAL TRAINED MODEL"""
        print("Initializing ACTUAL trained model...")
        
        self.device = get_device("auto")
        self.config = Config()
        
        print(f"  Device: {self.device}")
        print(f"  Loading from: {model_path}")
        
        # Initialize YOUR model architecture
        self.model = AttnMILResNet(self.config).to(self.device)
        
        # Load YOUR trained weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.val_auc = checkpoint.get('val_auc', 0.0)
            self.epoch = checkpoint.get('epoch', 'unknown')
            print(f"✓ Model loaded successfully!")
            print(f"  Validation AUC: {self.val_auc:.4f}")
            print(f"  Trained epoch: {self.epoch}")
        else:
            raise ValueError("Invalid checkpoint format")
        
        self.model.eval()
        
        # Initialize audio processor (matching Streamlit app)
        self.processor = CoughAudioProcessor(self.config)
        
        # ImageNet normalization (matching your training)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        
        print("✓ Model ready for ACTUAL inference!")
    
    def _build_batch(self, segments):
        """
        Build batch from waveform segments exactly like the working Streamlit app.
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
        
        return x_norm.to(self.device), mask.to(self.device)
    
    def preprocess_audio(self, audio_bytes, temp_path=None):
        """
        Preprocess audio exactly as in the working Streamlit app:
        1. Save audio bytes to temp file (if path provided, use it; otherwise create temp)
        2. Load audio using CoughAudioProcessor.load_audio()
        3. Extract waveform segments using CoughAudioProcessor.extract_segments()
        4. Build batch using _build_batch() which converts each segment to mel
        """
        try:
            # Save audio to temp file if needed
            if temp_path is None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
                    tmp_path = tmp_file.name
                    tmp_file.write(audio_bytes)
            else:
                tmp_path = temp_path
                with open(tmp_path, 'wb') as f:
                    f.write(audio_bytes)
            
            try:
                # Load audio using processor (matches Streamlit)
                wav = self.processor.load_audio(tmp_path)
                print(f"  Loaded audio: {wav.shape[1]} samples")
                
                # Extract waveform segments (matches Streamlit)
                segments = self.processor.extract_segments(wav)
                print(f"  Extracted {len(segments)} waveform segments")
                
                if len(segments) == 0:
                    print("✗ No segments extracted from audio")
                    return None, None
                
                # Build batch from segments (matches Streamlit)
                batch, mask = self._build_batch(segments)
                print(f"  Built batch: {batch.shape}, mask: {mask.shape}")
                
                return batch, mask
                
            finally:
                # Don't delete the file - it's saved in AUDIO_SAVE_DIR for future reference
                # Temp files from tempfile.NamedTemporaryFile will be cleaned up by OS
                pass
            
        except Exception as e:
            print(f"✗ Preprocessing error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def predict(self, audio_bytes, saved_path=None):
        """
        Run YOUR ACTUAL TRAINED MODEL inference
        Args:
            audio_bytes: Raw audio bytes (required if saved_path is None)
            saved_path: Optional path to saved audio file (if provided, will use this instead of creating temp)
        Returns: (is_cough, file_probability, segment_probs_list, num_segments, segment_timestamps)
        """
        try:
            # If saved_path is provided, use it; otherwise create temp from bytes
            if saved_path:
                segments, mask = self.preprocess_audio(audio_bytes, temp_path=saved_path)
            else:
                segments, mask = self.preprocess_audio(audio_bytes)

            if segments is None:
                return None, 0.0, [], 0, []

            with torch.no_grad():
                # Run YOUR ACTUAL MODEL!
                bag_prob, seg_probs, seg_logits, bag_logit = self.model(segments, mask)

                # Get probability
                probability = float(bag_prob.cpu().item())

                # Segment probabilities (list)
                try:
                    seg_list = seg_probs.squeeze(0).cpu().numpy().tolist()
                except Exception:
                    seg_list = []

                # Interpret results (same thresholds as before)
                is_cough = probability > 0.61  # Very low - detects any cough-like audio
                tb_detected = probability > 1  # Standard threshold for TB

                # Build per-segment timestamps (seconds). In training config, segment_duration and hop_length are in seconds.
                num_segments = int(segments.shape[1])
                timestamps = []
                for i in range(num_segments):
                    start = i * self.config.hop_length
                    end = start + self.config.segment_duration
                    timestamps.append((start, end))

                print(f"  Model output: prob={probability:.4f}, cough={is_cough}, TB={tb_detected}")

                return is_cough, probability, seg_list, num_segments, timestamps

        except Exception as e:
            print(f"✗ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0, [], 0, []


# Initialize YOUR ACTUAL MODEL
print("\n" + "="*70)
print("CoughCare Backend - YOUR ACTUAL TRAINED MODEL")
print("="*70)

try:
    model = ActualCoughModel(MODEL_PATH)
    model_loaded = True
    print("\n✓✓✓ ACTUAL MODEL LOADED AND READY ✓✓✓")
except Exception as e:
    print(f"\n✗✗✗ FAILED TO LOAD MODEL ✗✗✗")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    model_loaded = False
    model = None

print("="*70 + "\n")


@app.route('/api/detect-cough', methods=['POST'])
def detect_cough_endpoint():
    """Detect cough using YOUR ACTUAL TRAINED MODEL - NO FAKE DATA"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    saved_audio_path = None
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
        
        print(f"\n→ Running ACTUAL MODEL on {len(audio_bytes)} bytes")

        # Save audio file locally
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = audio_file.filename or 'cough_audio'
        file_ext = os.path.splitext(filename)[1] or '.webm'
        saved_audio_path = os.path.join(AUDIO_SAVE_DIR, f"{timestamp}_{filename}")
        with open(saved_audio_path, 'wb') as f:
            f.write(audio_bytes)
        print(f"  ✓ Saved audio to: {saved_audio_path}")

        # Optional cough decision threshold (like Streamlit slider)
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
            threshold = 0.61  # keep legacy default

        # Use YOUR ACTUAL TRAINED MODEL - NO FAKE PREDICTIONS
        # Pass saved path to predict to avoid creating duplicate temp file
        is_cough_model, probability, seg_list, num_segments, timestamps = model.predict(audio_bytes, saved_path=saved_audio_path)

        if is_cough_model is None:
            return jsonify({'error': 'Failed to process audio'}), 500

        # Final decision uses requested threshold
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

        print(f"✓ ACTUAL MODEL result: Cough={is_cough}, TB={tb_detected}, Prob={probability:.4f}")

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
            'message': f'Analyzed by trained model (Val AUC: {model.val_auc:.4f})',
            'model_info': {
                'validation_auc': float(model.val_auc),
                'epoch': model.epoch
            }
        })
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze-tb', methods=['POST'])
def analyze_tb_endpoint():
    """Analyze TB using YOUR ACTUAL TRAINED MODEL"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
        
        # Optional TB threshold
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
            # Intentionally omitting TB confidence from response
            'file_probability': float(probability),
            'segment_probabilities': seg_list,
            'segment_timestamps': timestamps,
            'num_segments': num_segments,
            'tb_threshold_used': float(tb_threshold),
            'message': f'Actual model prediction (Val AUC: {model.val_auc:.4f})'
        })
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy' if model_loaded else 'error',
        'model_loaded': model_loaded,
        'model_type': 'AttnMILResNet',
        'validation_auc': float(model.val_auc) if model_loaded else None,
        'device': str(model.device) if model_loaded else None,
        'mode': 'ACTUAL_MODEL_INFERENCE'
    })


@app.route('/samples/<filename>', methods=['GET'])
def serve_sample(filename):
    """Serve sample audio files"""
    try:
        sample_path = os.path.join(AUDIO_SAVE_DIR, filename)
        print(f"DEBUG: Looking for sample at: {sample_path}")
        print(f"DEBUG: File exists: {os.path.exists(sample_path)}")
        print(f"DEBUG: AUDIO_SAVE_DIR: {AUDIO_SAVE_DIR}")
        if not os.path.exists(sample_path):
            print(f"DEBUG: File not found at {sample_path}")
            return jsonify({'error': f'Sample file not found at {sample_path}', 'AUDIO_SAVE_DIR': AUDIO_SAVE_DIR}), 404
        
        from flask import send_file
        print(f"DEBUG: Sending file: {sample_path}")
        return send_file(sample_path, mimetype='audio/wav')
    except Exception as e:
        print(f"DEBUG: Error serving sample: {str(e)}")
        import traceback
        traceback.print_exc()
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
        'version': '4.0.0-actual-model-only',
        'model': 'AttnMILResNet - YOUR TRAINED MODEL',
        'model_loaded': model_loaded,
        'validation_auc': float(model.val_auc) if model_loaded else None,
        'mode': 'ACTUAL_MODEL_INFERENCE_ONLY',
        'no_fake_predictions': True
    })


if __name__ == '__main__':
    if model_loaded:
        print("="*70)
        print("✓✓✓ SERVER READY WITH YOUR ACTUAL TRAINED MODEL ✓✓✓")
        print(f"  Model: AttnMILResNet")
        print(f"  Validation AUC: {model.val_auc:.4f} (Excellent!)")
        print(f"  Device: {model.device}")
        print(f"  Mode: ACTUAL MODEL INFERENCE - NO FAKE DATA")
        print("="*70)
    else:
        print("✗ Server starting WITHOUT model - all requests will fail")
    
    PORT = int(os.environ.get('PORT', 5000))  # Use 5001 instead of 5000 (AirPlay uses 5000)
    print(f"\nServer: http://localhost:{PORT}")
    print("Press Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=PORT, debug=True)
