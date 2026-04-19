"""
STEP 5: RESULTS DASHBOARD
Creates a comprehensive visualization dashboard for your hackathon presentation.
Produces: final_output/dashboard.png — a single figure summarizing everything.

Run last, after predictions are generated.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import rasterio
import json

OUT_DIR = Path("final_output")
OUT_DIR.mkdir(exist_ok=True)

PRED_DIR = Path("predictions")
TRAIN_LOG = Path("training_output/best_model.pth")

# ── Load training history from checkpoint ────────────────────────────────────

def load_history():
    """Returns dummy history if no real one is saved — replace with yours."""
    import torch
    if TRAIN_LOG.exists():
        ckpt = torch.load(TRAIN_LOG, map_location="cpu", weights_only=False)
        # If you saved history in the checkpoint, extract it here
        # For now return what we have
        return ckpt
    return None

# ── Full Dashboard ────────────────────────────────────────────────────────────

def make_dashboard():
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0D1117")  # dark background
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)

    title_color = "white"
    accent = "#00E676"   # green
    warn   = "#FF5252"   # red

    def dark_ax(ax):
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")
        ax.xaxis.label.set_color("gray")
        ax.yaxis.label.set_color("gray")
        ax.title.set_color(title_color)
        return ax

    # ── Row 0: Title ─────────────────────────────────────────────────────────
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(0.5, 0.7, "🌲 Deforestation Detection — Makeathon 2026",
                  ha="center", va="center", fontsize=22, color=accent,
                  fontweight="bold", transform=ax_title.transAxes)
    ax_title.text(0.5, 0.2,
                  "Multimodal U-Net++ | Sentinel-1 + Sentinel-2 + AEF Embeddings (70 channels)",
                  ha="center", va="center", fontsize=13, color="gray",
                  transform=ax_title.transAxes)

    # ── Row 1: Architecture diagram (text-based) ──────────────────────────────
    ax_arch = fig.add_subplot(gs[1, :2])
    dark_ax(ax_arch)
    ax_arch.axis("off")
    ax_arch.set_title("Model Architecture", fontsize=11, pad=8)

    arch_text = (
        "Input Stack (70 channels)\n"
        "  ├─ S2 bands R,G,B,NIR  (4ch)\n"
        "  ├─ NDVI               (1ch)\n"
        "  ├─ S1 SAR              (1ch)\n"
        "  └─ AEF embeddings     (64ch)\n"
        "        │\n"
        "   ┌────▼─────────────────────┐\n"
        "   │   UNet++ / ResNet18      │\n"
        "   │   Encoder → Bridge →     │\n"
        "   │   Decoder with skip conn │\n"
        "   └────────────┬─────────────┘\n"
        "                │\n"
        "   Binary mask (1 ch)  [0=Forest, 1=Deforested]\n"
        "\n"
        "Loss = DiceLoss + FocalLoss\n"
        "Optimizer = AdamW + CosineAnnealing"
    )
    ax_arch.text(0.05, 0.95, arch_text, transform=ax_arch.transAxes,
                 fontsize=9, color=accent, va="top", fontfamily="monospace")

    # ── Row 1: Key metrics ────────────────────────────────────────────────────
    ax_metrics = fig.add_subplot(gs[1, 2:])
    dark_ax(ax_metrics)
    ax_metrics.axis("off")
    ax_metrics.set_title("Key Results", fontsize=11, pad=8)

    # Try to load real metrics
    ckpt = load_history()
    f1_val = ckpt.get("best_f1", 0.0) if ckpt else 0.0
    epoch_val = ckpt.get("epoch", "?") if ckpt else "?"

    metrics_text = (
        f"Best Validation F1:  {f1_val:.4f}\n"
        f"Best Epoch:          {epoch_val}\n\n"
        f"Channel Count:       70\n"
        f"Label Sources:       GLAD-L + GLAD-S2 + RADD\n"
        f"Patch Size:          128 (fast) → 256 (final)\n"
        f"Test Tiles:          5\n"
        f"Train Tiles:         16\n"
    )
    ax_metrics.text(0.05, 0.9, metrics_text, transform=ax_metrics.transAxes,
                    fontsize=11, color="white", va="top", fontfamily="monospace",
                    linespacing=1.8)

    # ── Row 2: Channel importance schematic ──────────────────────────────────
    ax_channels = fig.add_subplot(gs[2, :2])
    dark_ax(ax_channels)
    ax_channels.set_title("Input Channel Breakdown", fontsize=11)

    channels = ["S2\n(R,G,B,NIR)", "NDVI", "S1\n(SAR)", "AEF\n(64 bands)"]
    widths = [4, 1, 1, 64]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    bars = ax_channels.barh(channels, widths, color=colors)
    for bar, w in zip(bars, widths):
        ax_channels.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                         f"{w}ch", va="center", color="white", fontsize=10)
    ax_channels.set_xlabel("Number of channels")
    ax_channels.set_xlim(0, 75)

    # ── Row 2: Prediction map (if available) ─────────────────────────────────
    ax_pred = fig.add_subplot(gs[2, 2:])
    dark_ax(ax_pred)

    pred_files = list(PRED_DIR.glob("*.tif"))
    if pred_files:
        with rasterio.open(pred_files[0]) as src:
            prob = src.read(2)  # probability band
        im = ax_pred.imshow(prob, cmap="RdYlGn_r", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax_pred, fraction=0.046)
        ax_pred.set_title(f"Sample Prediction: {pred_files[0].stem}", fontsize=9)
        ax_pred.axis("off")
    else:
        ax_pred.text(0.5, 0.5, "Run step4_predict.py\nto generate predictions",
                     ha="center", va="center", color="gray", fontsize=12,
                     transform=ax_pred.transAxes)
        ax_pred.set_title("Prediction Map", fontsize=11)

    plt.savefig(OUT_DIR / "dashboard.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("✅ Saved: final_output/dashboard.png")


# ── Confusion Matrix from Train Val ──────────────────────────────────────────

def plot_confusion_matrix_style():
    """Show a simple precision/recall/F1 summary if metrics are available."""
    ckpt = load_history()
    if not ckpt:
        return

    f1 = ckpt.get("best_f1", 0.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_facecolor("#161B22")
    fig.patch.set_facecolor("#0D1117")

    metrics = {"F1 Score": f1, "IoU (est)": f1 / (2 - f1)}
    names = list(metrics.keys())
    vals  = list(metrics.values())
    colors = ["#4CAF50" if v > 0.5 else "#FF9800" if v > 0.3 else "#F44336" for v in vals]

    bars = ax.bar(names, vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.4f}", ha="center", color="white", fontsize=12)

    ax.set_ylim(0, 1.1)
    ax.set_title("Validation Metrics", color="white", fontsize=13)
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")
    ax.set_facecolor("#161B22")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "13_metrics_summary.png", dpi=120,
                facecolor=fig.get_facecolor())
    plt.close()
    print("✅ Saved: 13_metrics_summary.png")


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*50)
    print("STEP 5: RESULTS DASHBOARD")
    print("="*50)
    make_dashboard()
    plot_confusion_matrix_style()
    print("\n✅ Dashboard ready in final_output/")
    print("   Use these images in your hackathon presentation!")
