# precompute_spectrograms.py
import os
import pickle
import warnings
from typing import List
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
from tqdm import tqdm

# Suppress known warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# Use the same config as training
from train_cough_detector_attention import Config


def robust_label_to_int(x):
    """Hardened label parsing."""
    if isinstance(x, (bool, int, np.integer)):
        return int(bool(x))
    s = str(x).strip().lower()
    return 1 if s in {"true", "1", "yes", "y"} else 0


def letterbox_to_square(x: torch.Tensor, target_hw: int = 224, pad_val: float = -1.0):
    """
    Letterbox-pad to square target size without distortion.
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
    """Shared audio processor for precompute/train/app."""
    def __init__(self, config: Config):
        self.config = config
        self.mel_transform = T.MelSpectrogram(
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
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

    def load_audio(self, file_path: str) -> torch.Tensor:
        """Unify I/O: torchaudio primary, librosa fallback. Consistent normalization and processing."""
        waveform = None
        sr = None
        
        # Try torchaudio first with consistent settings
        try:
            # Load without normalization first to ensure consistency
            waveform, sr = torchaudio.load(file_path, normalize=False)  # (C, T)
            # Apply consistent normalization
            waveform = waveform.float()
            if waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()  # Normalize to [-1, 1]
        except Exception as e1:
            # Try with different backends
            try:
                waveform, sr = torchaudio.load(file_path, backend="ffmpeg", normalize=False)
                waveform = waveform.float()
                if waveform.abs().max() > 0:
                    waveform = waveform / waveform.abs().max()
            except Exception as e2:
                # Fallback to librosa with consistent normalization
                try:
                    import librosa
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Load with consistent settings
                        y, sr = librosa.load(file_path, sr=None, mono=False, dtype=np.float32)
                    if y.ndim == 1:
                        y = y[None, :]
                    waveform = torch.from_numpy(y).float()
                    # Apply same normalization as torchaudio
                    if waveform.abs().max() > 0:
                        waveform = waveform / waveform.abs().max()
                except Exception as e3:
                    raise RuntimeError(f"Failed to load audio with torchaudio ({e1}) and librosa ({e3})")

        # Ensure we have valid audio
        if waveform is None or waveform.numel() == 0:
            raise RuntimeError("Empty or invalid audio file")

        # Convert to mono (consistent for both paths)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed (consistent for both paths)
        if sr != self.config.sample_rate:
            resampler = T.Resample(sr, self.config.sample_rate)
            waveform = resampler(waveform)

        # Apply high-pass filter (consistent for both paths)
        try:
            hpf = T.HighpassBiquad(sample_rate=self.config.sample_rate, cutoff_freq=50.0)
            waveform = hpf(waveform)
        except (AttributeError, RuntimeError):
            # High-pass filter not available or failed, continue without it
            pass

        return waveform  # (1, T)

    def extract_segments(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        seg_samples = int(self.config.segment_duration * self.config.sample_rate)
        hop_samples = int(self.config.hop_length * self.config.sample_rate)

        # pad short
        if waveform.shape[1] < seg_samples:
            waveform = F.pad(waveform, (0, seg_samples - waveform.shape[1]))

        segments = []
        start = 0
        while start + seg_samples <= waveform.shape[1]:
            segments.append(waveform[:, start:start + seg_samples])
            start += hop_samples

        if not segments:
            segments.append(waveform[:, :seg_samples])

        return segments

    def waveform_to_mel_3ch(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        (1, T) -> mel in dB -> normalize to [-1,1] -> optional letterbox(224x224) -> 3ch -> float16
        Returns: (3, H, W) as float16 for 50% size reduction
        """
        mel = self.mel_transform(waveform)         # (1, n_mels, frames)
        mel_db = self.amplitude_to_db(mel)         # dB
        mel_norm01 = torch.clamp((mel_db + 80) / 80, 0, 1)
        mel_norm = mel_norm01 * 2 - 1              # [-1,1]
        mel_img = mel_norm.squeeze(0)              # (n_mels, frames)

        # Optional letterbox to 224x224
        if self.config.resize_to_224:
            mel_img = letterbox_to_square(mel_img.unsqueeze(0), 224).squeeze(0)

        # 3-ch
        mel_3ch = mel_img.repeat(3, 1, 1)
        
        # Convert to float16 for 50% size reduction
        # This maintains the [-1, 1] range with reduced precision
        return mel_3ch.half()  # (3, H, W) as float16


def precompute_spectrograms(csv_path: str, config: Config, cache_dir: str, split_name: str, batch_size: int = 100):
    """Process files in batches to prevent memory issues with float16 quantization."""
    os.makedirs(cache_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df['cough_label'] = df['cough_label'].apply(robust_label_to_int)

    processor = CoughAudioProcessor(config)
    cache_file = os.path.join(cache_dir, f"{split_name}_spectrograms.pkl")

    if os.path.exists(cache_file):
        print(f"[{split_name}] Using existing cache at {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # Process in batches to avoid memory issues
    processed = []
    total_files = len(df)
    
    for batch_start in tqdm(range(0, total_files, batch_size), desc=f"Processing {split_name} batches"):
        batch_end = min(batch_start + batch_size, total_files)
        batch_df = df.iloc[batch_start:batch_end]
        
        batch_processed = []
        for _, row in batch_df.iterrows():
            file_rel = row['file_path']
            label = int(row['cough_label'])
            file_path = os.path.join(config.base_dir, file_rel)
            
            try:
                wav = processor.load_audio(file_path)
                segments = processor.extract_segments(wav)

                # enforce cap for memory symmetry (train/infer)
                if len(segments) > config.max_segments_per_file:
                    segments = segments[:config.max_segments_per_file]

                specs = []
                for seg in segments:
                    spec = processor.waveform_to_mel_3ch(seg)  # (3, H, W) as float16
                    specs.append(spec)

                if specs:
                    specs_tensor = torch.stack(specs)  # (S, 3, H, W) as float16
                else:
                    # Fallback dummy with correct shape
                    H = 224 if config.resize_to_224 else config.n_mels
                    W = 224 if config.resize_to_224 else 64
                    specs_tensor = torch.zeros(1, 3, H, W, dtype=torch.float16)

                batch_processed.append({
                    "file_path": file_rel,
                    "label": label,
                    "spectrograms": specs_tensor,   # float16 for 50% size reduction
                    "num_segments": specs_tensor.shape[0],
                })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                H = 224 if config.resize_to_224 else config.n_mels
                W = 224 if config.resize_to_224 else 64
                batch_processed.append({
                    "file_path": file_rel,
                    "label": label,
                    "spectrograms": torch.zeros(1, 3, H, W, dtype=torch.float16),
                    "num_segments": 1,
                })
        
        # Add batch to main list
        processed.extend(batch_processed)
        
        # Clear batch from memory
        del batch_processed
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Save intermediate results every 500 files
        if (batch_start + batch_size) % 500 == 0 or batch_end == total_files:
            temp_cache_file = os.path.join(cache_dir, f"{split_name}_spectrograms_temp.pkl")
            with open(temp_cache_file, "wb") as f:
                pickle.dump(processed, f)
            print(f"[{split_name}] Saved intermediate results: {len(processed)} items")

    # Final save
    with open(cache_file, "wb") as f:
        pickle.dump(processed, f)
    
    # Clean up temp file
    temp_cache_file = os.path.join(cache_dir, f"{split_name}_spectrograms_temp.pkl")
    if os.path.exists(temp_cache_file):
        os.remove(temp_cache_file)

    print(f"[{split_name}] Saved {len(processed)} items → {cache_file}")
    return processed


def main():
    config = Config()
    cache_dir = os.path.join(config.output_dir, "spectrogram_cache")
    print("Precomputing spectrograms with settings:")
    print(f"  n_mels={config.n_mels}, n_fft={config.n_fft}, hop={config.hop_length_fft}, resize_to_224={config.resize_to_224}")

    splits = {"train": config.train_csv, "val": config.val_csv, "test": config.test_csv}
    for name, csvp in splits.items():
        data = precompute_spectrograms(csvp, config, cache_dir, name)
        tot_segs = sum(it["num_segments"] for it in data)
        pos = sum(1 for it in data if it["label"] == 1)
        print(f"[{name}] files={len(data)}, segments={tot_segs}, pos={pos}, neg={len(data)-pos}, avg_segs={tot_segs/len(data):.2f}")

    print("Done.")


if __name__ == "__main__":
    main()