"""
Audio processing utilities matching the working Streamlit app.
This ensures parity with the training/preprocessing pipeline.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from pydub import AudioSegment
import tempfile
import os


def letterbox_to_square(x: torch.Tensor, target_hw: int = 224, pad_val: float = -1.0):
    """
    Letterbox-pad to square target size without distortion.
    Matches Streamlit app exactly.
    x: (C, H, W) in [-1, 1]
    """
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
    """
    Processes audio exactly as in the training pipeline.
    Matches the working Streamlit app implementation.
    """
    
    def __init__(self, config):
        self.config = config
        # Mel spectrogram transform - matching Streamlit app exactly
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
        # Amplitude to dB - matching Streamlit app exactly
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    
    def load_audio(self, path: str) -> torch.Tensor:
        """
        Load audio file and return waveform tensor (1, T) at config.sample_rate.
        Supports multiple formats via pydub/ffmpeg.
        """
        # Try loading with pydub first (handles more formats)
        try:
            audio = AudioSegment.from_file(path)
            if audio.channels > 1:
                audio = audio.set_channels(1)  # Convert to mono
            
            sample_rate = audio.frame_rate
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            
            # Normalize based on bit depth
            if audio.sample_width == 1:  # 8-bit
                samples = samples / 128.0 - 1.0
            elif audio.sample_width == 2:  # 16-bit
                samples = samples / 32768.0
            elif audio.sample_width == 4:  # 32-bit
                samples = samples / 2147483648.0
            else:
                samples = samples / (2 ** (8 * audio.sample_width - 1))
            
            waveform = torch.from_numpy(samples).unsqueeze(0)  # (1, T)
            
            # Resample if needed
            if sample_rate != self.config.sample_rate:
                resampler = T.Resample(sample_rate, self.config.sample_rate)
                waveform = resampler(waveform)
            
            return waveform
            
        except Exception as e:
            # Fallback to torchaudio (works for WAV files)
            try:
                waveform, sample_rate = torchaudio.load(path)
                # Convert to mono if needed
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Resample if needed
                if sample_rate != self.config.sample_rate:
                    resampler = T.Resample(sample_rate, self.config.sample_rate)
                    waveform = resampler(waveform)
                
                return waveform
            except Exception as e2:
                raise ValueError(f"Could not load audio from {path}: {e}, {e2}")
    
    def extract_segments(self, waveform: torch.Tensor) -> list:
        """
        Extract overlapping segments from waveform.
        Returns list of waveform segments: [(1, T), (1, T), ...]
        Each segment is segment_duration seconds with hop_length overlap.
        """
        segments = []
        segment_samples = int(self.config.segment_duration * self.config.sample_rate)
        hop_samples = int(self.config.hop_length * self.config.sample_rate)
        
        total_samples = waveform.shape[1]
        
        start = 0
        while start < total_samples:
            end = start + segment_samples
            seg = waveform[:, start:end]
            
            # Pad if segment is shorter than expected (last segment)
            if seg.shape[1] < segment_samples:
                padding = segment_samples - seg.shape[1]
                seg = torch.nn.functional.pad(seg, (0, padding))
            
            segments.append(seg)
            
            # Stop if we have enough segments
            if len(segments) >= self.config.max_segments_per_file:
                break
            
            start += hop_samples
            
            # Don't create segments that are all padding
            if start >= total_samples:
                break
        
        return segments
    
    def waveform_to_mel_3ch(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform segment to 3-channel mel spectrogram.
        Matches Streamlit app exactly.
        Returns (3, H, W) tensor in range [-1, 1].
        
        Process:
        1. Create mel spectrogram (1, n_mels, time)
        2. Convert to dB
        3. Normalize to [-1, 1] using fixed formula: clamp((mel_db + 80) / 80, 0, 1) * 2 - 1
        4. Resize to 224x224 using letterbox if config.resize_to_224
        5. Convert to 3 channels (repeat)
        """
        # Create mel spectrogram: (1, n_mels, time_frames)
        mel = self.mel_transform(waveform)  # (1, n_mels, time)
        
        # Convert to dB
        mel_db = self.db_transform(mel)  # (1, n_mels, time) in dB
        
        # Normalize to [-1, 1] - EXACTLY like Streamlit app
        mel_norm01 = torch.clamp((mel_db + 80) / 80, 0, 1)
        mel_norm = mel_norm01 * 2 - 1  # [-1, 1]
        
        # Remove batch dimension: (n_mels, time)
        mel_img = mel_norm.squeeze(0)  # (n_mels, time)
        
        # Optional letterbox to 224x224 (matches Streamlit app)
        if self.config.resize_to_224:
            # letterbox_to_square expects (C, H, W), so add channel dimension
            mel_img = letterbox_to_square(mel_img.unsqueeze(0), 224).squeeze(0)
        
        # Convert to 3 channels by repeating: (3, H, W)
        mel_3ch = mel_img.repeat(3, 1, 1)
        
        return mel_3ch.float()  # (3, H, W) in float32 (matching Streamlit)

