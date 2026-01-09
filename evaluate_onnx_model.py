# evaluate_onnx_model.py
"""
Evaluate INT8 quantized ONNX model on the same test split used for PyTorch model.
Ensures NO discrepancy between test sets - uses exact same samples from splits/test.csv.
"""
import os
import json
import csv
import numpy as np
import onnxruntime as ort
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, f1_score, accuracy_score, classification_report, log_loss
)
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from typing import Tuple

# Audio processing
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from pydub import AudioSegment
import warnings
warnings.filterwarnings('ignore')

# Fix OpenMP on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


@dataclass
class Config:
    """Configuration matching training exactly."""
    # Paths - update for Windows
    base_dir: str = r"d:\coughcare_waig_3\Dataset"
    test_csv: str = r"d:\coughcare_waig_3\Cough\splits\test.csv"
    output_dir: str = r"d:\coughcare_waig_3\Cough\outputs"
    onnx_model_path: str = r"d:\coughcare_waig_3\coughcare_waig 3\cough_detector_int8.onnx"

    # Audio/mel (must match training)
    sample_rate: int = 16000
    segment_duration: float = 2.0
    hop_length: float = 0.5

    n_fft: int = 1024
    hop_length_fft: int = 160   # ~10 ms @ 16kHz
    n_mels: int = 160
    f_min: float = 50.0
    f_max: float = 4000.0

    # Resize to 224x224 (letterbox)
    resize_to_224: bool = True

    # MIL
    max_segments_per_file: int = 32


def letterbox_to_square(x: torch.Tensor, target_hw: int = 224, pad_val: float = -1.0):
    """Letterbox-pad to square target size without distortion (matches training)."""
    C, H, W = x.shape
    scale = min(target_hw / H, target_hw / W)
    new_h, new_w = int(round(H * scale)), int(round(W * scale))

    x_resized = F.interpolate(
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


class AudioProcessor:
    """Processes audio exactly as in training pipeline."""

    def __init__(self, config: Config):
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

        # ImageNet stats for normalization
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1)

    def load_audio(self, path: str) -> torch.Tensor:
        """Load audio file and return waveform tensor (1, T) at config.sample_rate."""
        try:
            # Try pydub first (handles webm, ogg, etc.)
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

            if sample_rate != self.config.sample_rate:
                resampler = T.Resample(sample_rate, self.config.sample_rate)
                waveform = resampler(waveform)

            return waveform

        except Exception as e:
            # Fallback to torchaudio
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
        """Extract overlapping segments from waveform (matches training)."""
        segments = []
        segment_samples = int(self.config.segment_duration * self.config.sample_rate)
        hop_samples = int(self.config.hop_length * self.config.sample_rate)

        total_samples = waveform.shape[1]

        start = 0
        while start < total_samples:
            end = start + segment_samples
            seg = waveform[:, start:end]

            # Pad if needed
            if seg.shape[1] < segment_samples:
                padding = segment_samples - seg.shape[1]
                seg = F.pad(seg, (0, padding))

            segments.append(seg)

            if len(segments) >= self.config.max_segments_per_file:
                break

            start += hop_samples
            if start >= total_samples:
                break

        return segments

    def waveform_to_mel_3ch(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform segment to 3-channel mel spectrogram (matches training)."""
        mel = self.mel_transform(waveform)
        mel_db = self.db_transform(mel)

        # Normalize to [-1, 1] (matches precompute_spectrograms.py)
        mel_norm01 = torch.clamp((mel_db + 80) / 80, 0, 1)
        mel_norm = mel_norm01 * 2 - 1

        mel_img = mel_norm.squeeze(0)  # (1, H, W) -> (H, W)

        if self.config.resize_to_224:
            mel_img = letterbox_to_square(mel_img.unsqueeze(0), 224).squeeze(0)

        # Expand to 3 channels
        mel_3ch = mel_img.repeat(3, 1, 1)

        return mel_3ch.float()

    def process_file(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """Process single audio file and return batch for ONNX inference."""
        waveform = self.load_audio(audio_path)
        segments = self.extract_segments(waveform)

        if len(segments) == 0:
            return None, None, 0

        # Convert to spectrograms
        specs = []
        for seg in segments:
            spec3 = self.waveform_to_mel_3ch(seg)
            specs.append(spec3.numpy())

        # Stack: (S, 3, 224, 224)
        x = np.stack(specs)

        # Add batch dimension: (1, S, 3, 224, 224)
        x = x[np.newaxis, ...]

        # Apply ImageNet normalization (matches training)
        # First convert from [-1, 1] to [0, 1]
        x01 = (x + 1) / 2.0
        # Then ImageNet normalize
        x_norm = (x01 - self.mean) / self.std

        # Create mask
        mask = np.ones((1, x.shape[1]), dtype=bool)

        return x_norm.astype(np.float32), mask, len(segments)


class ONNXModelEvaluator:
    """Evaluates ONNX model on test set."""

    def __init__(self, config: Config):
        self.config = config
        self.processor = AudioProcessor(config)

        # Load ONNX model
        print(f"Loading ONNX model from: {config.onnx_model_path}")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(
            config.onnx_model_path,
            sess_options,
            providers=providers
        )

        # Get model info
        model_size = os.path.getsize(config.onnx_model_path) / (1024 * 1024)
        print(f"Model size: {model_size:.2f} MB")
        print(f"Providers: {self.session.get_providers()}")

    def predict(self, batch: np.ndarray, mask: np.ndarray) -> Tuple[float, np.ndarray]:
        """Run ONNX inference."""
        outputs = self.session.run(
            None,
            {
                'spectrograms': batch,
                'segment_mask': mask
            }
        )

        bag_prob = float(outputs[0][0])
        seg_probs = outputs[1][0]

        return bag_prob, seg_probs

    def load_test_data(self) -> list:
        """Load test samples from CSV (same as training used)."""
        test_data = []

        with open(self.config.test_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_path = row['file_path']
                # Convert cough_label from string to int
                label = 1 if row['cough_label'].lower() == 'true' else 0

                # Build full path
                audio_path = os.path.join(self.config.base_dir, file_path)

                test_data.append({
                    'file_path': file_path,
                    'audio_path': audio_path,
                    'label': label
                })

        return test_data

    def evaluate(self, save_plots: bool = True):
        """Run full evaluation on test set."""
        print("\n" + "="*60)
        print("ONNX MODEL EVALUATION (INT8 Quantized)")
        print("="*60)

        # Load test data
        test_data = self.load_test_data()
        print(f"\nTest set size: {len(test_data)} samples")

        pos_count = sum(1 for d in test_data if d['label'] == 1)
        neg_count = len(test_data) - pos_count
        print(f"Positive samples: {pos_count}")
        print(f"Negative samples: {neg_count}")
        print(f"Class balance: {pos_count/len(test_data):.3f}")

        # Evaluate
        all_probs = []
        all_labels = []
        all_seg_probs = []
        failed_files = []

        print("\nEvaluating...")
        for item in tqdm(test_data, desc="Processing"):
            try:
                batch, mask, num_segs = self.processor.process_file(item['audio_path'])

                if batch is None:
                    failed_files.append(item['file_path'])
                    continue

                bag_prob, seg_probs = self.predict(batch, mask)

                all_probs.append(bag_prob)
                all_labels.append(item['label'])

                # Store segment probs for analysis
                valid_seg_probs = seg_probs[:num_segs]
                all_seg_probs.extend(valid_seg_probs.tolist())

            except Exception as e:
                print(f"\nError processing {item['file_path']}: {e}")
                failed_files.append(item['file_path'])
                continue

        if failed_files:
            print(f"\nWarning: {len(failed_files)} files failed to process")

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_seg_probs = np.array(all_seg_probs)

        # Calculate metrics
        print("\nCalculating metrics...")

        # Basic metrics
        auc = roc_auc_score(all_labels, all_probs)
        ap = average_precision_score(all_labels, all_probs)

        # Compute test loss (Binary Cross Entropy)
        # Clip probabilities to avoid log(0)
        eps = 1e-7
        clipped_probs = np.clip(all_probs, eps, 1 - eps)
        test_loss = log_loss(all_labels, clipped_probs)

        # Threshold-based metrics
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        threshold_results = {}

        for thresh in thresholds:
            preds = (all_probs > thresh).astype(int)
            acc = accuracy_score(all_labels, preds)
            f1 = f1_score(all_labels, preds)

            # Confusion matrix
            tp = ((preds == 1) & (all_labels == 1)).sum()
            fp = ((preds == 1) & (all_labels == 0)).sum()
            fn = ((preds == 0) & (all_labels == 1)).sum()
            tn = ((preds == 0) & (all_labels == 0)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            threshold_results[f"threshold_{thresh}"] = {
                "accuracy": float(acc),
                "f1_score": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn)
            }

        # Find optimal threshold
        f1_scores = []
        test_thresholds = np.linspace(0.01, 0.99, 99)
        for thresh in test_thresholds:
            preds = (all_probs > thresh).astype(int)
            f1 = f1_score(all_labels, preds)
            f1_scores.append(f1)

        optimal_f1_idx = np.argmax(f1_scores)
        optimal_thresh = test_thresholds[optimal_f1_idx]
        optimal_f1 = f1_scores[optimal_f1_idx]

        # Metrics at optimal threshold
        optimal_preds = (all_probs > optimal_thresh).astype(int)
        optimal_acc = accuracy_score(all_labels, optimal_preds)

        # Print results
        print("\n" + "="*60)
        print("TEST SET EVALUATION RESULTS (ONNX INT8)")
        print("="*60)
        print(f"Test set size: {len(all_labels)} samples")
        print(f"Positive samples: {all_labels.sum():.0f}")
        print(f"Negative samples: {len(all_labels) - all_labels.sum():.0f}")
        print(f"Class balance: {all_labels.mean():.3f}")
        print()

        print("OVERALL METRICS:")
        print(f"  Test Loss (BCE): {test_loss:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Average Precision: {ap:.4f}")
        print()

        print("THRESHOLD-BASED METRICS:")
        for thresh in thresholds:
            results = threshold_results[f"threshold_{thresh}"]
            print(f"  Threshold {thresh}: Acc={results['accuracy']:.3f}, F1={results['f1_score']:.3f}, "
                  f"Prec={results['precision']:.3f}, Rec={results['recall']:.3f}, Spec={results['specificity']:.3f}")

        print(f"\nOPTIMAL THRESHOLD: {optimal_thresh:.3f} (F1={optimal_f1:.3f}, Acc={optimal_acc:.3f})")

        # Detailed classification report
        print(f"\nDETAILED CLASSIFICATION REPORT (threshold={optimal_thresh:.3f}):")
        print(classification_report(all_labels, optimal_preds, target_names=['No Cough', 'Cough']))

        # Segment-level analysis
        if len(all_seg_probs) > 0:
            print(f"\nSEGMENT-LEVEL ANALYSIS:")
            print(f"  Total segments analyzed: {len(all_seg_probs)}")
            print(f"  Mean segment probability: {all_seg_probs.mean():.3f}")
            print(f"  Std segment probability: {all_seg_probs.std():.3f}")
            print(f"  Min segment probability: {all_seg_probs.min():.3f}")
            print(f"  Max segment probability: {all_seg_probs.max():.3f}")

        # Save test_evaluation_results.json
        eval_results = {
            "model_type": "ONNX_INT8",
            "model_path": self.config.onnx_model_path,
            "test_set_size": int(len(all_labels)),
            "positive_samples": int(all_labels.sum()),
            "negative_samples": int(len(all_labels) - all_labels.sum()),
            "class_balance": float(all_labels.mean()),
            "overall_metrics": {
                "test_loss": float(test_loss),
                "auc": float(auc),
                "average_precision": float(ap)
            },
            "threshold_results": threshold_results,
            "optimal_threshold": {
                "threshold": float(optimal_thresh),
                "f1_score": float(optimal_f1),
                "accuracy": float(optimal_acc)
            },
            "segment_analysis": {
                "total_segments": int(len(all_seg_probs)),
                "mean_probability": float(all_seg_probs.mean()) if len(all_seg_probs) > 0 else 0.0,
                "std_probability": float(all_seg_probs.std()) if len(all_seg_probs) > 0 else 0.0,
                "min_probability": float(all_seg_probs.min()) if len(all_seg_probs) > 0 else 0.0,
                "max_probability": float(all_seg_probs.max()) if len(all_seg_probs) > 0 else 0.0
            },
            "failed_files": failed_files
        }

        eval_results_path = os.path.join(self.config.output_dir, "test_evaluation_results_onnx.json")
        with open(eval_results_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nEvaluation results saved to: {eval_results_path}")

        # Save test_results.json (compact format matching training)
        test_results = {
            "model_type": "ONNX_INT8",
            "test_loss": float(test_loss),
            "test_accuracy": float(optimal_acc),
            "test_auc": float(auc),
            "test_ap": float(ap),
            "optimal_threshold": float(optimal_thresh),
            "optimal_f1": float(optimal_f1)
        }

        test_results_path = os.path.join(self.config.output_dir, "test_results_onnx.json")
        with open(test_results_path, "w") as f:
            json.dump(test_results, f, indent=2)
        print(f"Test results saved to: {test_results_path}")

        # Create plots
        if save_plots:
            print("\nCreating plots...")
            self._create_plots(all_labels, all_probs, all_seg_probs, optimal_thresh)

        return eval_results, test_results

    def _create_plots(self, y_true, y_prob, seg_probs, optimal_thresh):
        """Create evaluation plots."""
        # ROC and PR curves
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)

        # Threshold analysis
        thresholds = np.linspace(0.01, 0.99, 99)
        f1_scores = []
        acc_scores = []

        for thresh in thresholds:
            preds = (y_prob > thresh).astype(int)
            f1_scores.append(f1_score(y_true, preds))
            acc_scores.append(accuracy_score(y_true, preds))

        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ONNX INT8 Model - Test Set Evaluation', fontsize=14, fontweight='bold')

        # ROC Curve
        axes[0,0].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.3f}')
        axes[0,0].plot([0,1], [0,1], 'k--', alpha=0.5)
        axes[0,0].set_xlabel('False Positive Rate')
        axes[0,0].set_ylabel('True Positive Rate')
        axes[0,0].set_title('ROC Curve')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()

        # Precision-Recall Curve
        axes[0,1].plot(recall, precision, 'g-', linewidth=2, label=f'AP = {ap:.3f}')
        axes[0,1].axhline(y_true.mean(), color='k', linestyle='--', alpha=0.5,
                         label=f'Baseline = {y_true.mean():.3f}')
        axes[0,1].set_xlabel('Recall')
        axes[0,1].set_ylabel('Precision')
        axes[0,1].set_title('Precision-Recall Curve')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()

        # F1 vs Threshold
        axes[0,2].plot(thresholds, f1_scores, 'r-', linewidth=2)
        axes[0,2].axvline(optimal_thresh, color='k', linestyle='--',
                         label=f'Optimal: {optimal_thresh:.3f}')
        axes[0,2].set_xlabel('Threshold')
        axes[0,2].set_ylabel('F1-Score')
        axes[0,2].set_title('F1-Score vs Threshold')
        axes[0,2].grid(True, alpha=0.3)
        axes[0,2].legend()

        # Accuracy vs Threshold
        axes[1,0].plot(thresholds, acc_scores, 'purple', linewidth=2)
        axes[1,0].axvline(optimal_thresh, color='k', linestyle='--',
                         label=f'Optimal: {optimal_thresh:.3f}')
        axes[1,0].set_xlabel('Threshold')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].set_title('Accuracy vs Threshold')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()

        # Probability distributions
        axes[1,1].hist(y_prob[y_true == 0], bins=50, alpha=0.7, label='No Cough', color='blue')
        axes[1,1].hist(y_prob[y_true == 1], bins=50, alpha=0.7, label='Cough', color='red')
        axes[1,1].axvline(optimal_thresh, color='k', linestyle='--', label=f'Optimal: {optimal_thresh:.3f}')
        axes[1,1].set_xlabel('Predicted Probability')
        axes[1,1].set_ylabel('Count')
        axes[1,1].set_title('Probability Distributions')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        # Segment probability distribution
        if len(seg_probs) > 0:
            axes[1,2].hist(seg_probs, bins=50, alpha=0.7, color='green')
            axes[1,2].axvline(optimal_thresh, color='k', linestyle='--', label=f'Optimal: {optimal_thresh:.3f}')
            axes[1,2].set_xlabel('Segment Probability')
            axes[1,2].set_ylabel('Count')
            axes[1,2].set_title('Segment Probability Distribution')
            axes[1,2].legend()
            axes[1,2].grid(True, alpha=0.3)
        else:
            axes[1,2].text(0.5, 0.5, 'No segment data', ha='center', va='center')
            axes[1,2].set_title('Segment Probability Distribution')

        plt.tight_layout()
        plot_file = os.path.join(self.config.output_dir, 'test_evaluation_plots_onnx.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plots saved to: {plot_file}")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate ONNX INT8 model on test set')
    parser.add_argument('--model_path', type=str,
                       default=r"d:\coughcare_waig_3\coughcare_waig 3\cough_detector_int8.onnx",
                       help='Path to ONNX model')
    parser.add_argument('--data_dir', type=str,
                       default=r"d:\coughcare_waig_3\Dataset",
                       help='Path to dataset directory')
    parser.add_argument('--test_csv', type=str,
                       default=r"d:\coughcare_waig_3\Cough\splits\test.csv",
                       help='Path to test.csv')
    parser.add_argument('--output_dir', type=str,
                       default=r"d:\coughcare_waig_3\Cough\outputs",
                       help='Output directory')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating plots')

    args = parser.parse_args()

    config = Config()
    config.onnx_model_path = args.model_path
    config.base_dir = args.data_dir
    config.test_csv = args.test_csv
    config.output_dir = args.output_dir

    os.makedirs(config.output_dir, exist_ok=True)

    try:
        evaluator = ONNXModelEvaluator(config)
        eval_results, test_results = evaluator.evaluate(save_plots=not args.no_plots)

        print("\n" + "="*60)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"AUC: {test_results['test_auc']:.4f}")
        print(f"AP: {test_results['test_ap']:.4f}")
        print(f"Accuracy @ optimal threshold: {test_results['test_accuracy']:.4f}")
        print(f"Optimal threshold: {test_results['optimal_threshold']:.3f}")
        print(f"F1 @ optimal threshold: {test_results['optimal_f1']:.4f}")
        print("="*60)

    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
