# train_cough_detector_attention.py
import os
import json
import pickle
import random
from dataclasses import dataclass, asdict
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm


@dataclass
class Config:
    # Paths
    base_dir: str = "/Users/aakashpant-waig/Desktop/Cough/data"
    train_csv: str = "/Users/aakashpant-waig/Desktop/Cough/splits/train.csv"
    val_csv: str = "/Users/aakashpant-waig/Desktop/Cough/splits/val.csv"
    test_csv: str = "/Users/aakashpant-waig/Desktop/Cough/splits/test.csv"
    output_dir: str = "/Users/aakashpant-waig/Desktop/Cough/outputs"
    cache_dir: str = "/Users/aakashpant-waig/Desktop/Cough/outputs/spectrogram_cache"

    # Audio/mel (updated per request)
    sample_rate: int = 16000
    segment_duration: float = 2.0
    hop_length: float = 0.5

    n_fft: int = 1024
    hop_length_fft: int = 160   # ~10 ms @ 16kHz
    n_mels: int = 160
    f_min: float = 50.0
    f_max: float = 4000.0

    # Optional resize
    resize_to_224: bool = True  # letterbox to 224x224 after mel

    # Model/train
    model_name: str = "resnet18"
    dropout: float = 0.3
    pretrained: bool = True

    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 100
    early_stopping_patience: int = 5

    # MIL
    max_segments_per_file: int = 32
    attn_hidden: int = 128

    # Aug
    enable_augmentation: bool = True
    spec_noise_std: float = 0.01
    spec_gain_range: Tuple[float, float] = (0.9, 1.1)
    spec_time_mask: int = 16
    spec_freq_mask: int = 12

    # System
    device: str = "auto"  # auto, cpu, cuda, mps
    num_workers: int = 0
    seed: int = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


class OptimizedCoughDataset(Dataset):
    """Loads precomputed spectrograms; applies ImageNet normalization just-in-time."""
    def __init__(self, split: str, config: Config):
        self.config = config
        cache_file = os.path.join(config.cache_dir, f"{split}_spectrograms.pkl")
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Missing precomputed cache: {cache_file}")

        with open(cache_file, "rb") as f:
            self.data = pickle.load(f)

        self.split = split
        print(f"[{split}] {len(self.data)} files loaded.")

        # stats
        pos = sum(1 for it in self.data if it["label"] == 1)
        segs = sum(it["num_segments"] for it in self.data)
        print(f"  pos={pos}, neg={len(self.data)-pos}, avg_segs={segs/len(self.data):.2f}")

        # ImageNet stats
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.data)

    def _augment_spec(self, x):
        # x: (S, 3, H, W) in [-1,1]
        if random.random() < 0.10:
            x = x + torch.randn_like(x) * self.config.spec_noise_std
        if random.random() < 0.10:
            gain = random.uniform(*self.config.spec_gain_range)
            x = torch.clamp(x * gain, -1, 1)
        # SpecAugment-like (time/freq mask) on one channel set; apply to all ch
        if random.random() < 0.10:
            _, _, H, W = x.shape
            # freq mask
            f = min(self.config.spec_freq_mask, H)
            f0 = random.randint(0, H - f)
            x[:, :, f0:f0 + f, :] = 0.0
        if random.random() < 0.10:
            _, _, H, W = x.shape
            t = min(self.config.spec_time_mask, W)
            t0 = random.randint(0, W - t)
            x[:, :, :, t0:t0 + t] = 0.0
        return x

    def __getitem__(self, idx):
        item = self.data[idx]
        specs = item["spectrograms"].float()        # (S, 3, H, W) in [-1,1]
        label = torch.tensor(item["label"], dtype=torch.float32)

        if self.split == "train" and self.config.enable_augmentation:
            specs = self._augment_spec(specs)

        # map [-1,1] -> [0,1] then ImageNet normalize
        specs = (specs + 1) / 2.0
        specs = (specs - self.mean) / self.std

        S = specs.shape[0]
        return specs, label, S


def collate_fn(batch: List[tuple]):
    specs, labels, nsegs = zip(*batch)
    B = len(specs)
    maxS = max(nsegs)
    _, C, H, W = specs[0].shape
    out = torch.zeros(B, maxS, C, H, W, dtype=specs[0].dtype)
    mask = torch.zeros(B, maxS, dtype=torch.bool)
    for i, (x, s) in enumerate(zip(specs, nsegs)):
        out[i, :s] = x
        mask[i, :s] = True
    labels = torch.stack(labels)  # (B,)
    return out, labels, mask


class AttnMILResNet(nn.Module):
    """
    ResNet18 backbone → per-segment features → attention weights over features
    → bag feature (weighted sum) → bag_logit.
    Also produces per-segment logits for inspection.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        weights = ResNet18_Weights.DEFAULT if config.pretrained else None
        self.backbone = resnet18(weights=weights)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.dropout = nn.Dropout(config.dropout)

        # Attention scorer (Ilse et al.)
        self.attn = nn.Sequential(
            nn.Linear(num_features, config.attn_hidden),
            nn.Tanh(),
            nn.Linear(config.attn_hidden, 1),  # score per segment
        )

        # Bag head (on pooled feature)
        self.bag_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(config.dropout),
            nn.Linear(512, 1),  # bag_logit
        )

        # Optional per-segment scores for diagnostics
        self.seg_head = nn.Linear(num_features, 1)

    def forward(self, x, seg_mask):
        """
        x: (B, S, 3, H, W)
        seg_mask: (B, S) bool
        """
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        feats = self.backbone(x)               # (B*S, F)
        feats = self.dropout(feats)
        Fdim = feats.shape[1]
        feats_bs = feats.view(B, S, Fdim)

        # attention scores with mask -> softmax over valid S
        attn_scores = self.attn(feats).view(B, S)  # (B,S)
        # masked softmax
        neg_inf = torch.finfo(attn_scores.dtype).min
        masked_scores = torch.where(seg_mask, attn_scores, torch.tensor(neg_inf, device=attn_scores.device))
        attn_weights = torch.softmax(masked_scores, dim=1)  # (B,S), sums to 1 over valid segs

        # pooled bag feature
        bag_feat = torch.sum(attn_weights.unsqueeze(-1) * feats_bs, dim=1)  # (B,F)

        # bag logit (NO sigmoid here)
        bag_logit = self.bag_head(bag_feat).squeeze(1)  # (B,)

        # per-segment logits for inspection
        seg_logits = self.seg_head(feats).view(B, S)    # (B,S)
        seg_probs = torch.sigmoid(seg_logits)

        bag_prob = torch.sigmoid(bag_logit)
        return bag_prob, seg_probs, seg_logits, bag_logit


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0, restore_best_weights=True, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode  # "max" for AUC/AP, "min" for loss
        self.best = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, value, model):
        if self.best is None:
            self.best = value
            self.best_weights = model.state_dict().copy()
            return False
        improved = (value > self.best + self.min_delta) if self.mode == "max" else (value < self.best - self.min_delta)
        if improved:
            self.best = value
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    correct = 0
    total = 0
    for specs, labels, mask in tqdm(loader, desc="Train"):
        specs, labels, mask = specs.to(device), labels.to(device), mask.to(device)
        optimizer.zero_grad()
        bag_prob, seg_prob, seg_logit, bag_logit = model(specs, mask)
        loss = criterion(bag_logit, labels)
        loss.backward()
        optimizer.step()
        running += loss.item()
        pred = (bag_prob > 0.5).float()
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return running / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running = 0.0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for specs, labels, mask in tqdm(loader, desc="Eval"):
            specs, labels, mask = specs.to(device), labels.to(device), mask.to(device)
            bag_prob, _, _, bag_logit = model(specs, mask)
            loss = criterion(bag_logit, labels)
            running += loss.item()
            all_probs.extend(bag_prob.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    acc = ((all_probs > 0.5) == all_labels).mean()
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
        ap = average_precision_score(all_labels, all_probs)
    else:
        auc = 0.0
        ap = 0.0
    return running / len(loader), acc, auc, ap, all_probs, all_labels


def plot_metrics(train_losses, val_losses, val_aucs, val_aps, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0,0].plot(train_losses, label="Train")
    axes[0,0].plot(val_losses, label="Val")
    axes[0,0].set_title("Loss"); axes[0,0].legend(); axes[0,0].grid(True, alpha=.3)

    axes[0,1].plot(val_aucs, label="Val AUC")
    axes[0,1].set_title("AUC"); axes[0,1].legend(); axes[0,1].grid(True, alpha=.3)

    axes[1,0].plot(val_aps, label="Val AP")
    axes[1,0].set_title("AP"); axes[1,0].legend(); axes[1,0].grid(True, alpha=.3)

    axes[1,1].axis("off"); axes[1,1].text(.5,.5,"ROC saved separately",ha="center",va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "training_metrics.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(y_true, y_prob, outdir, name="Test"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig = plt.figure(figsize=(12,5))
    ax1 = plt.subplot(1,2,1)
    ax1.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax1.plot([0,1],[0,1],'k--',alpha=.5)
    ax1.set_title(f"{name} ROC"); ax1.legend(); ax1.grid(True, alpha=.3)

    ax2 = plt.subplot(1,2,2)
    ax2.plot(rec, prec, label=f"AP={ap:.3f}")
    ax2.axhline(y=(np.mean(y_true)), linestyle="--", alpha=.5, label=f"Baseline={np.mean(y_true):.3f}")
    ax2.set_title(f"{name} PR"); ax2.legend(); ax2.grid(True, alpha=.3)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name.lower()}_roc_pr.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = Config()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Data
    train_ds = OptimizedCoughDataset("train", cfg)
    val_ds   = OptimizedCoughDataset("val", cfg)
    test_ds  = OptimizedCoughDataset("test", cfg)

    pin = (device.type == "cuda")  # MPS/CPU don't benefit
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=pin)

    # Model
    model = AttnMILResNet(cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: total={total:,}, trainable={trainable:,}")

    # Imbalance handling: compute pos_weight from train set
    y_train = np.array([it["label"] for it in train_ds.data])
    pos_weight = None
    if y_train.mean() not in (0,1):
        pw = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-8)
        pos_weight = torch.tensor([pw], device=device, dtype=torch.float32)
        print(f"Using pos_weight={pw:.3f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-6)
    stopper = EarlyStopping(patience=cfg.early_stopping_patience, mode="max")  # monitor AUC

    train_losses, val_losses, val_aucs, val_aps = [], [], [], []
    best_auc = -1.0
    best_state = None

    for epoch in range(cfg.num_epochs):
        print(f"\nEpoch {epoch+1}/{cfg.num_epochs}")
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, vl_auc, vl_ap, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        val_aucs.append(vl_auc); val_aps.append(vl_ap)

        print(f"Train loss {tr_loss:.4f} acc {tr_acc:.4f}")
        print(f"Val   loss {vl_loss:.4f} acc {vl_acc:.4f} AUC {vl_auc:.4f} AP {vl_ap:.4f}")

        # Save best by AUC
        if vl_auc > best_auc:
            best_auc = vl_auc
            best_state = model.state_dict().copy()
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": vl_auc,
                "val_ap": vl_ap,
                "config": asdict(cfg),
            }, os.path.join(cfg.output_dir, "best_model.pth"))
            print(f"  → Saved best (AUC={vl_auc:.4f})")

        # Early stopping on AUC
        if stopper(vl_auc, model):
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    print("\nTesting...")
    te_loss, te_acc, te_auc, te_ap, te_prob, te_label = evaluate(model, test_loader, criterion, device)
    print(f"Test: loss {te_loss:.4f} acc {te_acc:.4f} AUC {te_auc:.4f} AP {te_ap:.4f}")

    with open(os.path.join(cfg.output_dir, "test_results.json"), "w") as f:
        json.dump({
            "test_loss": te_loss, "test_acc": te_acc, "test_auc": te_auc, "test_ap": te_ap,
            "best_val_auc": best_auc
        }, f, indent=2)

    plot_metrics(train_losses, val_losses, val_aucs, val_aps, cfg.output_dir)
    plot_roc_pr(te_label, te_prob, cfg.output_dir, name="Test")
    print(f"\nDone. Outputs → {cfg.output_dir}")


if __name__ == "__main__":
    main()