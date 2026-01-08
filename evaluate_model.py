# evaluate_model.py
import os
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, f1_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

from train_cough_detector_attention import Config, AttnMILResNet, OptimizedCoughDataset, collate_fn, get_device
from torch.utils.data import DataLoader

def evaluate_model(model_path=None, output_dir=None, save_plots=True):
    """
    Evaluate the trained model on test dataset.
    
    Args:
        model_path: Path to model checkpoint. If None, uses best_model.pth from output_dir
        output_dir: Output directory. If None, uses config.output_dir
        save_plots: Whether to save ROC/PR plots
    """
    config = Config()
    device = get_device(config.device)
    
    # Set paths
    if output_dir is None:
        output_dir = config.output_dir
    if model_path is None:
        model_path = os.path.join(output_dir, "best_model.pth")
    
    print(f"Loading model from: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    
    # Load test data
    print("\nLoading test dataset...")
    test_ds = OptimizedCoughDataset("test", config)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn, pin_memory=False)
    
    print(f"Test set: {len(test_ds)} files")
    pos_samples = sum(1 for item in test_ds.data if item["label"] == 1)
    neg_samples = len(test_ds.data) - pos_samples
    print(f"  Positive: {pos_samples}, Negative: {neg_samples}")
    print(f"  Class balance: {pos_samples/len(test_ds.data):.3f}")
    
    # Load model
    print("\nLoading model...")
    model = AttnMILResNet(config).to(device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if "val_auc" in checkpoint:
            print(f"Validation AUC: {checkpoint['val_auc']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict")
    
    model.eval()
    
    # Evaluate
    print("\nEvaluating on test set...")
    all_probs, all_labels, all_bag_logits = [], [], []
    all_seg_probs = []  # Store segment-level predictions for analysis
    
    with torch.no_grad():
        for specs, labels, mask in tqdm(test_loader, desc="Evaluating"):
            specs, labels, mask = specs.to(device), labels.to(device), mask.to(device)
            bag_prob, seg_probs, seg_logits, bag_logit = model(specs, mask)
            
            all_probs.extend(bag_prob.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_bag_logits.extend(bag_logit.cpu().numpy().tolist())
            
            # Store segment probabilities for analysis
            for i in range(specs.shape[0]):
                valid_segs = mask[i].sum().item()
                seg_probs_i = seg_probs[i][:valid_segs].cpu().numpy()
                all_seg_probs.extend(seg_probs_i.tolist())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_bag_logits = np.array(all_bag_logits)
    all_seg_probs = np.array(all_seg_probs)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    # Basic metrics
    auc = roc_auc_score(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    
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
    
    # Calculate metrics at optimal threshold
    optimal_preds = (all_probs > optimal_thresh).astype(int)
    optimal_acc = accuracy_score(all_labels, optimal_preds)
    optimal_precision = precision_score(all_labels, optimal_preds) if hasattr(__import__('sklearn.metrics'), 'precision_score') else None
    
    # Print results
    print("\n" + "="*60)
    print("TEST SET EVALUATION RESULTS")
    print("="*60)
    print(f"Test set size: {len(all_labels)} samples")
    print(f"Positive samples: {all_labels.sum()}")
    print(f"Negative samples: {len(all_labels) - all_labels.sum()}")
    print(f"Class balance: {all_labels.mean():.3f}")
    print()
    
    print("OVERALL METRICS:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Average Precision: {ap:.4f}")
    print()
    
    print("THRESHOLD-BASED METRICS:")
    for thresh in thresholds:
        results = threshold_results[f"threshold_{thresh}"]
        print(f"  Threshold {thresh}: Acc={results['accuracy']:.3f}, F1={results['f1_score']:.3f}, "
              f"Prec={results['precision']:.3f}, Rec={results['recall']:.3f}, Spec={results['specificity']:.3f}")
    
    print(f"\nOPTIMAL THRESHOLD: {optimal_thresh:.3f} (F1={optimal_f1:.3f}, Acc={optimal_acc:.3f})")
    
    # Detailed classification report at optimal threshold
    print(f"\nDETAILED CLASSIFICATION REPORT (threshold={optimal_thresh:.3f}):")
    print(classification_report(all_labels, optimal_preds, target_names=['No Cough', 'Cough']))
    
    # Segment-level analysis
    print(f"\nSEGMENT-LEVEL ANALYSIS:")
    print(f"  Total segments analyzed: {len(all_seg_probs)}")
    print(f"  Mean segment probability: {all_seg_probs.mean():.3f}")
    print(f"  Std segment probability: {all_seg_probs.std():.3f}")
    print(f"  Min segment probability: {all_seg_probs.min():.3f}")
    print(f"  Max segment probability: {all_seg_probs.max():.3f}")
    
    # Save results
    results = {
        "test_set_size": int(len(all_labels)),
        "positive_samples": int(all_labels.sum()),
        "negative_samples": int(len(all_labels) - all_labels.sum()),
        "class_balance": float(all_labels.mean()),
        "overall_metrics": {
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
            "mean_probability": float(all_seg_probs.mean()),
            "std_probability": float(all_seg_probs.std()),
            "min_probability": float(all_seg_probs.min()),
            "max_probability": float(all_seg_probs.max())
        }
    }
    
    results_file = os.path.join(output_dir, "test_evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Create plots
    if save_plots:
        print("\nCreating plots...")
        create_evaluation_plots(all_labels, all_probs, all_seg_probs, optimal_thresh, output_dir)
    
    return results

def create_evaluation_plots(y_true, y_prob, seg_probs, optimal_thresh, output_dir):
    """Create comprehensive evaluation plots."""
    
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
    axes[1,2].hist(seg_probs, bins=50, alpha=0.7, color='green')
    axes[1,2].axvline(optimal_thresh, color='k', linestyle='--', label=f'Optimal: {optimal_thresh:.3f}')
    axes[1,2].set_xlabel('Segment Probability')
    axes[1,2].set_ylabel('Count')
    axes[1,2].set_title('Segment Probability Distribution')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'test_evaluation_plots.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to: {plot_file}")

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained cough detection model')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model checkpoint (default: outputs/best_model.pth)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: config.output_dir)')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    try:
        results = evaluate_model(
            model_path=args.model_path,
            output_dir=args.output_dir,
            save_plots=not args.no_plots
        )
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()