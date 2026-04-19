"""
STEP 3: TRAINING — v3 (Dual-head: segmentation + year regression)
==================================================================
WHY THIS VERSION:
  Year=0% on leaderboard → we now train a SECOND output head to predict
  the year of deforestation at pixel level.

  FPR=87% → we now use:
    1. filter_empty in dataset (no empty patches)
    2. Higher pos_weight for actual deforestation detection
    3. Tversky loss (alpha=0.3, beta=0.7) which penalizes FP more than FN

ARCHITECTURE: UNet++ with TWO decoder heads:
    Head 1 (seg_head):  1 channel → binary deforestation probability
    Head 2 (year_head): 1 channel → normalized year [0,1], only where deforested

LOSS:
    L_seg  = Dice + Tversky(FP-heavy) + BCE(pos_weight)  → binary mask
    L_year = masked MSE (only on deforested pixels)        → year regression

CONFIG (kept small for time):
    PATCH=256, BATCH=4, EPOCHS=20, ENCODER=resnet34
"""

import torch, torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import segmentation_models_pytorch as smp

import sys
sys.path.insert(0, str(Path(__file__).parent))
from step2_dataset import build_index, ForestDataset, norm_to_year

DATA_ROOT = Path("data/makeathon-challenge")
OUT_DIR   = Path("training_output")
OUT_DIR.mkdir(exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────────────────────
PATCH_SIZE   = 256
BATCH_SIZE   = 4
EPOCHS       = 20
ENCODER      = "resnet34"
LR           = 1e-4
IN_CHANNELS  = 76
NUM_WORKERS  = 4
POS_WEIGHT   = 12.0
YEAR_LOSS_W  = 0.5   # weight of year regression vs segmentation loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Training on: {DEVICE}")

# ── Dual-head model ───────────────────────────────────────────────────────────

class DualHeadUNet(nn.Module):
    """
    UNet++ encoder-decoder with two separate 1×1 conv output heads:
      seg_head  → binary deforestation logit
      year_head → year regression (sigmoid output in [0,1])

    We share the full encoder+decoder and only split at the very last layer.
    This forces the shared features to be useful for BOTH tasks.
    """
    def __init__(self, encoder_name=ENCODER, in_channels=IN_CHANNELS):
        super().__init__()
        # Build UNet++ but don't apply final activation (we'll add our own heads)
        self.backbone = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=32,           # 32 intermediate feature channels
            activation=None,
        )
        # Two heads on top of 32 intermediate channels
        self.seg_head  = nn.Conv2d(32, 1, kernel_size=1)
        self.year_head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()   # output in [0,1] range for normalized year
        )

    def forward(self, x):
        feats      = self.backbone(x)         # (B, 32, H, W)
        seg_logit  = self.seg_head(feats)     # (B,  1, H, W) — raw logit for BCE
        year_pred  = self.year_head(feats)    # (B,  1, H, W) — [0,1] sigmoid
        return seg_logit, year_pred


def build_model():
    return DualHeadUNet(ENCODER, IN_CHANNELS).to(DEVICE)

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(seg_logit, year_pred, targets, threshold=0.5):
    mask_gt  = targets[:, 0:1]   # binary mask
    year_gt  = targets[:, 1:2]   # normalized year

    preds = (torch.sigmoid(seg_logit) > threshold).float()
    tp = (preds * mask_gt).sum()
    fp = (preds * (1 - mask_gt)).sum()
    fn = ((1 - preds) * mask_gt).sum()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)
    fpr       = fp / (fp + (1 - mask_gt).sum() + 1e-8)

    # Year MAE on deforested pixels only
    defor_mask = (mask_gt > 0.5) & (year_gt > 0)
    if defor_mask.sum() > 0:
        year_mae = (year_pred[defor_mask] - year_gt[defor_mask]).abs().mean().item()
        # convert to actual years
        year_mae_years = year_mae * 9.0   # (YEAR_MAX - YEAR_MIN) = 9
    else:
        year_mae_years = 0.0

    return {
        "precision": precision.item(), "recall": recall.item(),
        "f1": f1.item(), "iou": iou.item(), "fpr": fpr.item(),
        "year_mae": year_mae_years,
    }

# ── Optimal threshold finder ──────────────────────────────────────────────────

def find_optimal_threshold(model, val_loader):
    print("\n🔍 Finding optimal threshold …")
    model.eval()
    all_probs, all_masks = [], []
    with torch.no_grad():
        for X, y in tqdm(val_loader, desc="Sweep", leave=False):
            X = X.to(DEVICE)
            seg_logit, _ = model(X)
            probs = torch.sigmoid(seg_logit).cpu().numpy().flatten()
            masks = y[:, 0].numpy().flatten()
            all_probs.extend(probs.tolist())
            all_masks.extend(masks.tolist())

    all_probs = np.array(all_probs)
    all_masks = np.array(all_masks)
    best_f1, best_t = 0.0, 0.5

    print(f"\n{'Threshold':>10} {'F1':>8} {'Precision':>11} {'Recall':>8} {'FPR':>8}")
    print("-" * 50)
    for t in np.arange(0.3, 0.85, 0.05):
        preds     = (all_probs > t).astype(int)
        tp        = ((preds == 1) & (all_masks == 1)).sum()
        fp        = ((preds == 1) & (all_masks == 0)).sum()
        fn        = ((preds == 0) & (all_masks == 1)).sum()
        tn        = ((preds == 0) & (all_masks == 0)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        fpr       = fp / (fp + tn + 1e-8)
        marker    = " ← BEST" if f1 > best_f1 else ""
        print(f"{t:>10.2f} {f1:>8.4f} {precision:>11.4f} {recall:>8.4f} {fpr:>8.4f}{marker}")
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"\n✅ Best threshold: {best_t:.2f}  (F1={best_f1:.4f})")
    return best_t, best_f1

# ── Training ──────────────────────────────────────────────────────────────────

def train():
    index        = build_index("train")
    full_dataset = ForestDataset(index, patch_size=PATCH_SIZE, augment=True,
                                 filter_empty=True, repeats_per_tile=20)

    if len(full_dataset) == 0:
        print("❌ No samples."); return

    val_size   = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    # Val set: don't filter empty (we want full coverage for threshold sweep)
    val_ds.dataset.filter_empty = False

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model     = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Segmentation losses
    dice_loss    = smp.losses.DiceLoss(mode="binary", from_logits=True)
    # Tversky: alpha=FP weight, beta=FN weight.  alpha>beta → penalise FP more → lower FPR
    tversky_loss = smp.losses.TverskyLoss(mode="binary", from_logits=True,
                                           alpha=0.4, beta=0.6)
    bce_loss     = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT]).to(DEVICE)
    )
    # Year regression loss — only computed on pixels that ARE deforested
    mse_loss = nn.MSELoss(reduction="none")

    history = {k: [] for k in ["train_loss", "val_loss", "val_f1",
                                "val_iou", "val_fpr", "val_year_mae",
                                "val_precision", "val_recall"]}
    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False)
        for X, y in pbar:
            X, y      = X.to(DEVICE), y.to(DEVICE)
            mask_gt   = y[:, 0:1]
            year_gt   = y[:, 1:2]

            optimizer.zero_grad()
            seg_logit, year_pred = model(X)

            # Segmentation loss
            l_seg = (dice_loss(seg_logit, mask_gt)
                     + tversky_loss(seg_logit, mask_gt)
                     + 0.5 * bce_loss(seg_logit, mask_gt))

            # Year regression — MSE only on pixels labelled as deforested
            defor_mask = (mask_gt > 0.5) & (year_gt > 0)
            if defor_mask.sum() > 0:
                l_year = mse_loss(year_pred, year_gt)[defor_mask].mean()
            else:
                l_year = torch.tensor(0.0, device=DEVICE)

            loss = l_seg + YEAR_LOSS_W * l_year

            if torch.isnan(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        val_losses, all_metrics = [], []
        with torch.no_grad():
            for X, y in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", leave=False):
                X, y      = X.to(DEVICE), y.to(DEVICE)
                mask_gt   = y[:, 0:1]
                year_gt   = y[:, 1:2]

                seg_logit, year_pred = model(X)

                l_seg = (dice_loss(seg_logit, mask_gt)
                         + tversky_loss(seg_logit, mask_gt)
                         + 0.5 * bce_loss(seg_logit, mask_gt))
                defor_mask = (mask_gt > 0.5) & (year_gt > 0)
                l_year = mse_loss(year_pred, year_gt)[defor_mask].mean() \
                         if defor_mask.sum() > 0 else torch.tensor(0.0)

                val_losses.append((l_seg + YEAR_LOSS_W * l_year).item())
                all_metrics.append(compute_metrics(seg_logit, year_pred, y))

        scheduler.step()

        def avg(key): return np.mean([m[key] for m in all_metrics]) if all_metrics else 0

        for k in ["val_f1", "val_iou", "val_fpr", "val_year_mae",
                  "val_precision", "val_recall"]:
            metric_key = k.replace("val_", "")
            history[k].append(avg(metric_key))

        avg_train = np.mean(train_losses) if train_losses else 0
        avg_val   = np.mean(val_losses) if val_losses else 0
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        f1  = history["val_f1"][-1]
        fpr = history["val_fpr"][-1]
        yr  = history["val_year_mae"][-1]
        print(f"Epoch {epoch:02d} | Loss {avg_train:.3f}/{avg_val:.3f} | "
              f"F1={f1:.4f} IoU={avg('iou'):.4f} FPR={fpr:.3f} YearMAE={yr:.2f}yr")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_f1": best_f1, "history": history,
                "config": {
                    "encoder": ENCODER, "patch_size": PATCH_SIZE,
                    "in_channels": IN_CHANNELS, "threshold": 0.5,
                    "arch": "DualHeadUNet",
                }
            }, OUT_DIR / "best_model.pth")
            print(f"   💾 Saved (F1={best_f1:.4f})")

        plot_training_curves(history, epoch)

    # Threshold sweep
    best_t, t_f1 = find_optimal_threshold(model, val_loader)
    ckpt = torch.load(OUT_DIR / "best_model.pth", map_location="cpu", weights_only=False)
    ckpt["config"]["threshold"] = float(best_t)
    ckpt["threshold_f1"]        = float(t_f1)
    ckpt["history"]             = history
    torch.save(ckpt, OUT_DIR / "best_model.pth")

    print(f"\n✅ Done! Best F1={best_f1:.4f}  threshold={best_t:.2f}")
    print("   Run: python step4_predict.py")
    return model, history


def plot_training_curves(history, epoch):
    fig, axes = plt.subplots(1, 5, figsize=(25, 4))
    ep = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(ep, history["train_loss"], "b-o", markersize=3, label="Train")
    axes[0].plot(ep, history["val_loss"],   "r-o", markersize=3, label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep, history["val_f1"], "g-o", markersize=3)
    if history["val_f1"]:
        axes[1].axhline(max(history["val_f1"]), color="g", linestyle="--", alpha=0.5,
                        label=f"Best={max(history['val_f1']):.4f}")
    axes[1].set_title("Val F1"); axes[1].set_ylim(0, 1)
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(ep, history["val_fpr"], "r-o", markersize=3, label="FPR")
    axes[2].set_title("Val FPR (lower=better)"); axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(ep, history["val_precision"], "b-o", markersize=3, label="P")
    axes[3].plot(ep, history["val_recall"],    "r-o", markersize=3, label="R")
    axes[3].set_title("Precision / Recall"); axes[3].set_ylim(0, 1)
    axes[3].legend(); axes[3].grid(True, alpha=0.3)

    axes[4].plot(ep, history["val_year_mae"], "purple", marker="o", markersize=3)
    axes[4].set_title("Year MAE (years)"); axes[4].grid(True, alpha=0.3)

    plt.suptitle(f"Epoch {epoch}/{EPOCHS} | {ENCODER} | 76ch | DualHead", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "10_training_curves.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 3: DUAL-HEAD TRAINING (seg + year regression)")
    print(f"  patch={PATCH_SIZE} batch={BATCH_SIZE} epochs={EPOCHS} encoder={ENCODER}")
    print(f"  pos_weight={POS_WEIGHT} year_loss_w={YEAR_LOSS_W}")
    print("=" * 60)
    train()
