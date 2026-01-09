# evaluate_full_pipeline.py
"""
Evaluate the FULL ONNX Pipeline (Preprocessing + Detection) on the test set.
This script validates the exact client-side architecture:
Audio -> Preprocessing ONNX -> Spectrogram -> Detection ONNX -> Prediction
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
from typing import Tuple, List, Optional
import torch
import torchaudio
import torchaudio.transforms as T
from pydub import AudioSegment
import warnings

warnings.filterwarnings('ignore')

# Fix OpenMP on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


@dataclass
class Config:
    """Configuration for full pipeline evaluation."""
    # Paths
    base_dir: str = r"d:\coughcare_waig_3\Dataset"
    test_csv: str = r"d:\coughcare_waig_3\Cough\splits\test.csv"
    output_dir: str = r"d:\coughcare_waig_3\Cough\outputs"
    
    # Models
    preprocessing_model_path: str = r"d:\coughcare_waig_3\coughcare_waig 3\cough_preprocessing.onnx"
    detection_model_path: str = r"d:\coughcare_waig_3\coughcare_waig 3\cough_detector_int8.onnx"

    # Audio
    sample_rate: int = 16000
    segment_duration: float = 2.0
    hop_length: float = 0.5
    max_segments_per_file: int = 32


class AudioLoader:
    """Helper to load audio files consistent with training."""
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def load_audio(self, path: str) -> np.ndarray:
        """Load audio and return numpy array (samples,)."""
        try:
            # Try pydub first
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

            # Resample if needed
            if sample_rate != self.sample_rate:
                # Use torchaudio for high-quality resampling
                waveform = torch.from_numpy(samples).unsqueeze(0)
                resampler = T.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
                samples = waveform.squeeze(0).numpy()

            return samples

        except Exception as e:
            # Fallback to torchaudio
            try:
                waveform, sample_rate = torchaudio.load(path)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                if sample_rate != self.sample_rate:
                    resampler = T.Resample(sample_rate, self.sample_rate)
                    waveform = resampler(waveform)

                return waveform.squeeze(0).numpy()
            except Exception as e2:
                raise ValueError(f"Could not load audio from {path}: {e}, {e2}")


class ONNXInferencePipeline:
    """
    Two-stage ONNX inference pipeline.
    Replicates the client-side logic exactly.
    """
    def __init__(self, preprocessing_path: str, model_path: str):
        # Load preprocessing model
        self.preprocess_session = ort.InferenceSession(
            preprocessing_path,
            providers=['CPUExecutionProvider']
        )
        
        # Load detection model
        self.model_session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )

        self.sample_rate = 16000
        self.segment_duration = 2.0
        self.hop_length = 0.5
        self.max_segments = 32
        self.segment_samples = int(self.sample_rate * self.segment_duration)

    def extract_segments(self, waveform: np.ndarray) -> list:
        """Extract overlapping segments."""
        seg_samples = self.segment_samples
        hop_samples = int(self.hop_length * self.sample_rate)

        # Handle shape
        if waveform.ndim == 2:
            waveform = waveform.squeeze()

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
        """Run preprocessing ONNX on a single segment."""
        # Reshape for ONNX: (1, segment_samples)
        segment_input = segment.reshape(1, -1).astype(np.float32)

        outputs = self.preprocess_session.run(
            None,
            {'waveform': segment_input}
        )

        return outputs[0]  # (1, 3, 224, 224)

    def predict(self, waveform: np.ndarray) -> Tuple[float, List[float], int]:
        """Full prediction pipeline."""
        # 1. Extract segments
        segments = self.extract_segments(waveform)
        num_segments = len(segments)

        if num_segments == 0:
            return 0.0, [], 0

        # 2. Preprocess each segment (ONNX 1)
        spectrograms = []
        for seg in segments:
            spec = self.preprocess_segment(seg)
            spectrograms.append(spec)

        # 3. Stack spectrograms
        specs_batch = np.concatenate(spectrograms, axis=0)  # (num_segments, 3, 224, 224)
        specs_batch = np.expand_dims(specs_batch, axis=0)   # (1, num_segments, 3, 224, 224)

        # 4. Create mask
        mask = np.ones((1, num_segments), dtype=bool)

        # 5. Run detection (ONNX 2)
        outputs = self.model_session.run(
            None,
            {
                'spectrograms': specs_batch.astype(np.float32),
                'segment_mask': mask
            }
        )

        bag_prob = float(outputs[0][0])
        seg_probs = outputs[1][0][:num_segments].tolist()

        return bag_prob, seg_probs, num_segments


class FullPipelineEvaluator:
    """Evaluates the full pipeline on the test set."""

    def __init__(self, config: Config):
        self.config = config
        self.loader = AudioLoader(config.sample_rate)
        
        print(f"Initializing Full ONNX Pipeline...")
        print(f"  Preprocessing: {config.preprocessing_model_path}")
        print(f"  Detection: {config.detection_model_path}")
        
        self.pipeline = ONNXInferencePipeline(
            config.preprocessing_model_path,
            config.detection_model_path
        )

    def load_test_data(self) -> list:
        """Load test samples from CSV."""
        test_data = []
        with open(self.config.test_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_path = row['file_path']
                label = 1 if row['cough_label'].lower() == 'true' else 0
                audio_path = os.path.join(self.config.base_dir, file_path)
                
                test_data.append({
                    'file_path': file_path,
                    'audio_path': audio_path,
                    'label': label
                })
        return test_data

    def evaluate(self, save_plots: bool = True):
        """Run full evaluation."""
        print("\n" + "="*60)
        print("FULL ONNX PIPELINE EVALUATION")
        print("="*60)

        test_data = self.load_test_data()
        print(f"\nTest set size: {len(test_data)} samples")

        all_probs = []
        all_labels = []
        all_seg_probs = []
        failed_files = []

        print("\nRunning Inference...")
        for item in tqdm(test_data, desc="Processing"):
            try:
                # Load audio
                waveform = self.loader.load_audio(item['audio_path'])
                
                # Run pipeline
                bag_prob, seg_probs, _ = self.pipeline.predict(waveform)

                all_probs.append(bag_prob)
                all_labels.append(item['label'])
                all_seg_probs.extend(seg_probs)

            except Exception as e:
                print(f"\nError processing {item['file_path']}: {e}")
                failed_files.append(item['file_path'])
                continue

        if failed_files:
            print(f"\nWarning: {len(failed_files)} files failed to process")

        # Convert to numpy
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_seg_probs = np.array(all_seg_probs)

        # Calculate metrics
        self._calculate_and_print_metrics(all_labels, all_probs, all_seg_probs, save_plots)

    def _calculate_and_print_metrics(self, y_true, y_prob, seg_probs, save_plots):
        """Calculate metrics, print them, and save JSON results."""
        print("\nCalculating metrics...")
        
        # Basic metrics
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        
        # Test Loss (BCE)
        eps = 1e-7
        clipped_probs = np.clip(y_prob, eps, 1 - eps)
        test_loss = log_loss(y_true, clipped_probs)

        # Threshold analysis
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        threshold_results = {}

        for thresh in thresholds:
            preds = (y_prob > thresh).astype(int)
            acc = accuracy_score(y_true, preds)
            f1 = f1_score(y_true, preds)
            
            # Confusion matrix
            tp = ((preds == 1) & (y_true == 1)).sum()
            fp = ((preds == 1) & (y_true == 0)).sum()
            fn = ((preds == 0) & (y_true == 1)).sum()
            tn = ((preds == 0) & (y_true == 0)).sum()

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
        test_thresholds = np.linspace(0.01, 0.99, 99)
        f1_scores = [f1_score(y_true, (y_prob > t).astype(int)) for t in test_thresholds]
        optimal_idx = np.argmax(f1_scores)
        optimal_thresh = test_thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        
        optimal_preds = (y_prob > optimal_thresh).astype(int)
        optimal_acc = accuracy_score(y_true, optimal_preds)

        print("\n" + "="*60)
        print("FULL PIPELINE RESULTS")
        print("="*60)
        print(f"AUC: {auc:.4f}")
        print(f"AP: {ap:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Optimal Threshold: {optimal_thresh:.3f}")
        print(f"F1 Score: {optimal_f1:.4f}")
        print(f"Accuracy: {optimal_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, optimal_preds, target_names=['No Cough', 'Cough']))

        # Save full evaluation results
        eval_results = {
            "model_type": "FULL_ONNX_PIPELINE",
            "preprocessing_model": self.config.preprocessing_model_path,
            "detection_model": self.config.detection_model_path,
            "test_set_size": int(len(y_true)),
            "positive_samples": int(y_true.sum()),
            "negative_samples": int(len(y_true) - y_true.sum()),
            "class_balance": float(y_true.mean()),
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
                "total_segments": int(len(seg_probs)),
                "mean_probability": float(seg_probs.mean()) if len(seg_probs) > 0 else 0.0,
                "std_probability": float(seg_probs.std()) if len(seg_probs) > 0 else 0.0,
                "min_probability": float(seg_probs.min()) if len(seg_probs) > 0 else 0.0,
                "max_probability": float(seg_probs.max()) if len(seg_probs) > 0 else 0.0
            }
        }

        eval_results_path = os.path.join(self.config.output_dir, "full_pipeline_evaluation_results.json")
        with open(eval_results_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nDetailed results saved to: {eval_results_path}")

        # Save compact test results
        test_results = {
            "model_type": "FULL_ONNX_PIPELINE",
            "test_loss": float(test_loss),
            "test_accuracy": float(optimal_acc),
            "test_auc": float(auc),
            "test_ap": float(ap),
            "optimal_threshold": float(optimal_thresh),
            "optimal_f1": float(optimal_f1)
        }

        test_results_path = os.path.join(self.config.output_dir, "full_pipeline_test_results.json")
        with open(test_results_path, "w") as f:
            json.dump(test_results, f, indent=2)
        print(f"Compact results saved to: {test_results_path}")

        if save_plots:
            self._create_plots(y_true, y_prob, seg_probs, optimal_thresh, auc, ap)

    def _create_plots(self, y_true, y_prob, seg_probs, optimal_thresh, auc, ap):
        """Create evaluation plots."""
        print("\nCreating plots...")
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Full ONNX Pipeline Evaluation', fontsize=14, fontweight='bold')

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        axes[0,0].plot(fpr, tpr, 'b-', label=f'AUC = {auc:.3f}')
        axes[0,0].plot([0,1], [0,1], 'k--')
        axes[0,0].set_title('ROC Curve')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        axes[0,1].plot(recall, precision, 'g-', label=f'AP = {ap:.3f}')
        axes[0,1].set_title('Precision-Recall Curve')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # Probability Histograms
        axes[1,0].hist(y_prob[y_true==0], bins=30, alpha=0.5, label='No Cough', color='blue')
        axes[1,0].hist(y_prob[y_true==1], bins=30, alpha=0.5, label='Cough', color='red')
        axes[1,0].axvline(optimal_thresh, color='k', linestyle='--', label='Threshold')
        axes[1,0].set_title('Prediction Probabilities')
        axes[1,0].legend()

        # Segment Probabilities
        if len(seg_probs) > 0:
            axes[1,1].hist(seg_probs, bins=50, color='purple', alpha=0.7)
            axes[1,1].set_title('Segment Probabilities (All segments)')
            axes[1,1].set_xlabel('Probability')

        plt.tight_layout()
        plot_path = os.path.join(self.config.output_dir, 'full_pipeline_plots.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Plots saved to: {plot_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Limit number of files for quick testing')
    args = parser.parse_args()

    config = Config()
    evaluator = FullPipelineEvaluator(config)
    
    # Optional: limit dataset for testing
    if args.limit:
        print(f"Limiting to first {args.limit} samples...")
        # Monkey patch load_test_data to return subset
        original_load = evaluator.load_test_data
        evaluator.load_test_data = lambda: original_load()[:args.limit]

    evaluator.evaluate()


if __name__ == "__main__":
    main()
