"""
STEP 4: PREDICT — v3 (Dual-head: binary mask + year map)
=========================================================
WHY THIS VERSION:
  Now the model outputs TWO things per pixel:
    1. Deforestation probability (→ binary mask after threshold)
    2. Year of deforestation (normalized, → decoded to actual year)

  The submission format requires a time_step column = year of deforestation.
  step6_submit.py reads band3 (year map) from the GeoTIFF to populate this.

OUTPUT GeoTIFF bands:
  Band 1: binary mask (cleaned, 0/1)
  Band 2: deforestation probability [0,1]
  Band 3: predicted year (integer, e.g. 2020), 0 = no deforestation
"""

import torch, torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from tqdm import tqdm
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import binary_opening, binary_closing
import segmentation_models_pytorch as smp

import sys
sys.path.insert(0, str(Path(__file__).parent))
from step2_dataset import (build_index, read_warped,
                           norm_to_year, YEAR_MIN, YEAR_MAX, IN_CHANNELS)

DATA_ROOT  = Path("data/makeathon-challenge")
MODEL_PATH = Path("training_output/best_model.pth")
PRED_DIR   = Path("predictions")
OUT_DIR    = Path("prediction_output")
PRED_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 256
OVERLAP    = 64


# ── Rebuild the model class ───────────────────────────────────────────────────
# (must match step3 exactly)

class DualHeadUNet(nn.Module):
    def __init__(self, encoder_name="resnet34", in_channels=IN_CHANNELS):
        super().__init__()
        self.backbone = smp.UnetPlusPlus(
            encoder_name=encoder_name, encoder_weights=None,
            in_channels=in_channels, classes=32, activation=None,
        )
        self.seg_head  = nn.Conv2d(32, 1, kernel_size=1)
        self.year_head = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        feats = self.backbone(x)
        return self.seg_head(feats), self.year_head(feats)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_model(path):
    ckpt      = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg       = ckpt.get("config", {})
    encoder   = cfg.get("encoder", "resnet34")
    in_ch     = cfg.get("in_channels", IN_CHANNELS)
    threshold = cfg.get("threshold", 0.5)

    model = DualHeadUNet(encoder, in_ch).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"✅ Model: encoder={encoder} epoch={ckpt.get('epoch','?')} "
          f"F1={ckpt.get('best_f1',0):.4f} threshold={threshold:.2f}")
    return model, threshold


# ── Sliding window ────────────────────────────────────────────────────────────

def sliding_window_predict(model, tile_data):
    C, H, W   = tile_data.shape
    step      = PATCH_SIZE - OVERLAP
    prob_map  = np.zeros((H, W), dtype=np.float32)
    year_map  = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H, step):
        for x in range(0, W, step):
            y_end = min(y + PATCH_SIZE, H); x_end = min(x + PATCH_SIZE, W)
            y0    = y_end - PATCH_SIZE;     x0    = x_end - PATCH_SIZE

            patch = tile_data[:, y0:y_end, x0:x_end]
            if patch.shape[1] < 16 or patch.shape[2] < 16: continue

            t = torch.from_numpy(patch[np.newaxis]).float().to(DEVICE)
            with torch.no_grad():
                seg_logit, yr_pred = model(t)
                probs = torch.sigmoid(seg_logit)[0, 0].cpu().numpy()
                yrs   = yr_pred[0, 0].cpu().numpy()

            prob_map[y0:y_end, x0:x_end]  += probs
            year_map[y0:y_end, x0:x_end]  += yrs
            count_map[y0:y_end, x0:x_end] += 1

    count_map = np.maximum(count_map, 1)
    return prob_map / count_map, year_map / count_map


# ── Build tile stack ──────────────────────────────────────────────────────────

def load_tile_stack(tile_data, H, W, ref_crs, ref_transform):
    """Builds 76-channel stack matching step2 ForestDataset."""

    def read_bands(path, bands, out_h, out_w, resample=Resampling.bilinear):
        with rasterio.open(path) as src:
            data = src.read(bands, out_shape=(len(bands), out_h, out_w),
                            resampling=resample)
        return np.nan_to_num(data.astype(np.float32), nan=0.0)

    if not tile_data["s2"]:
        return None

    s2_sorted = sorted(tile_data["s2"], key=lambda x: x["days"])
    s2e_rec   = s2_sorted[0]
    s2l_rec   = s2_sorted[-1]

    # S2 late
    s2l = read_bands(s2l_rec["path"], [4, 3, 2, 8], H, W) / 10000.0
    ndvi_late  = (s2l[3] - s2l[0]) / (s2l[3] + s2l[0] + 1e-7)

    # S2 early (warped)
    s2e = read_warped(s2e_rec["path"], [4, 3, 2, 8],
                      ref_crs, ref_transform, H, W,
                      rasterio.windows.Window(0, 0, W, H), H, W) / 10000.0
    ndvi_early = (s2e[3] - s2e[0]) / (s2e[3] + s2e[0] + 1e-7)
    delta_ndvi = np.clip(ndvi_late - ndvi_early, -1.0, 1.0)

    # S1
    if tile_data["s1"]:
        s1_rec = min(tile_data["s1"], key=lambda x: abs(x["days"] - s2l_rec["days"]))
        s1_raw = read_bands(s1_rec["path"], [1], H, W)
        s1 = (np.clip(10 * np.log10(np.clip(s1_raw, 1e-4, 1.0)), -20, 0) + 20) / 20.0
    else:
        s1 = np.zeros((1, H, W), dtype=np.float32)

    # AEF
    if tile_data["aef"]:
        yr       = s2l_rec["y"]
        aef_path = tile_data["aef"].get(yr, next(iter(tile_data["aef"].values())))
        aef = read_bands(aef_path, list(range(1, 65)), H, W)
    else:
        aef = np.zeros((64, H, W), dtype=np.float32)

    return np.concatenate([
        s2e, s2l,
        ndvi_early[np.newaxis], ndvi_late[np.newaxis], delta_ndvi[np.newaxis],
        s1, aef,
    ], axis=0)   # (76, H, W)


# ── Predict all tiles ─────────────────────────────────────────────────────────

def predict_test_tiles(model, threshold):
    test_index = build_index("test")
    if not test_index:
        print("⚠️  No test tiles found. Using train tiles.")
        test_index = build_index("train")

    print(f"\nInference on {len(test_index)} tiles (threshold={threshold:.2f}) …")
    pred_paths = []

    for tile_id, tile_data in tqdm(test_index.items(), desc="Tiles"):
        if not tile_data["s2"]: continue
        ref_path = sorted(tile_data["s2"], key=lambda x: x["days"])[-1]["path"]

        with rasterio.open(ref_path) as src:
            H, W       = src.height, src.width
            transform  = src.transform
            crs        = src.crs
            ref_crs    = src.crs
            ref_transform = src.transform

        stack = load_tile_stack(tile_data, H, W, ref_crs, ref_transform)
        if stack is None: continue

        prob_map, year_norm_map = sliding_window_predict(model, stack)
        pred_mask_raw = (prob_map > threshold).astype(np.uint8)

        # Morphological cleanup (reduce FPR)
        pred_mask = binary_opening(pred_mask_raw, iterations=2)
        pred_mask = binary_closing(pred_mask,     iterations=2)
        pred_mask = pred_mask.astype(np.uint8)

        # Decode year: only where mask=1
        year_int_map = np.zeros((H, W), dtype=np.float32)
        defor_pixels = pred_mask > 0
        if defor_pixels.any():
            raw_years = year_norm_map[defor_pixels] * (YEAR_MAX - YEAR_MIN) + YEAR_MIN
            year_int_map[defor_pixels] = np.round(raw_years).astype(np.float32)

        print(f"  {tile_id}: {pred_mask.mean()*100:.1f}% deforested  "
              f"years: {sorted(set(year_int_map[defor_pixels].astype(int).tolist())) if defor_pixels.any() else '—'}")

        out_path = PRED_DIR / f"{tile_id}_prediction.tif"
        with rasterio.open(out_path, "w", driver="GTiff",
                           height=H, width=W, count=3, dtype="float32",
                           crs=crs, transform=transform) as dst:
            dst.write(pred_mask.astype(np.float32), 1)   # binary mask
            dst.write(prob_map,                     2)   # probability
            dst.write(year_int_map,                 3)   # year (integer, 0=no event)

        pred_paths.append((tile_id, out_path, stack, prob_map, pred_mask, year_int_map))

    return pred_paths


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_predictions(pred_paths, threshold, n=4):
    n = min(n, len(pred_paths))
    if n == 0: return

    fig, axes = plt.subplots(n, 5, figsize=(25, 5 * n))
    if n == 1: axes = [axes]
    cmap_prob = plt.cm.RdYlGn_r
    cmap_year = plt.cm.plasma

    for i, (tid, _, stack, prob, mask, yr_map) in enumerate(pred_paths[:n]):
        rgb = np.clip(np.stack([stack[4], stack[5], stack[6]], axis=-1) * 3, 0, 1)
        delta = stack[10]

        axes[i][0].imshow(rgb);                  axes[i][0].set_title(f"S2 late RGB\n{tid}", fontsize=8)
        im1 = axes[i][1].imshow(delta, cmap="RdBu", vmin=-0.5, vmax=0.5)
        axes[i][1].set_title("delta_NDVI\n(red=loss)", fontsize=8)
        plt.colorbar(im1, ax=axes[i][1], fraction=0.046)

        axes[i][2].imshow(prob, cmap=cmap_prob, vmin=0, vmax=1)
        axes[i][2].set_title(f"Prob (thr={threshold:.2f})", fontsize=8)
        plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_prob, norm=mcolors.Normalize(0,1)),
                     ax=axes[i][2], fraction=0.046)

        axes[i][3].imshow(rgb)
        ov = np.zeros((*mask.shape, 4)); ov[mask > 0] = [1,0,0,0.6]
        axes[i][3].imshow(ov)
        axes[i][3].set_title(f"Mask {mask.mean()*100:.1f}%", fontsize=8)

        yr_display = np.where(yr_map > 0, yr_map, np.nan)
        im4 = axes[i][4].imshow(yr_display, cmap=cmap_year, vmin=YEAR_MIN, vmax=YEAR_MAX)
        axes[i][4].set_title("Year of deforestation", fontsize=8)
        plt.colorbar(im4, ax=axes[i][4], fraction=0.046)

        for ax in axes[i]: ax.axis("off")

    plt.suptitle("Dual-Head Predictions (mask + year)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "11_predictions.png", dpi=120)
    plt.close()
    print("✅ 11_predictions.png")


def plot_prediction_summary(pred_paths, threshold):
    tile_ids   = [p[0] for p in pred_paths]
    defor_pcts = [p[4].mean() * 100 for p in pred_paths]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#F44336" if d > 5 else "#4CAF50" for d in defor_pcts]
    axes[0].barh(tile_ids, defor_pcts, color=colors)
    axes[0].set_xlabel("Deforested Area (%)")
    axes[0].set_title("Predicted Deforestation per Tile")
    axes[0].axvline(5, color="black", linestyle="--", alpha=0.5, label="5% threshold")
    axes[0].legend()

    all_probs = np.concatenate([p[3].flatten() for p in pred_paths[:4]])
    axes[1].hist(all_probs, bins=50, color="#2196F3", edgecolor="white")
    axes[1].axvline(threshold, color="red", linestyle="--", label=f"Threshold={threshold:.2f}")
    axes[1].set_title("Probability Distribution")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "12_prediction_summary.png", dpi=120)
    plt.close()
    print("✅ 12_prediction_summary.png")


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 4: DUAL-HEAD PREDICTION (mask + year)")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print(f"❌ No model at {MODEL_PATH}. Run step3_train.py first."); exit(1)

    model, threshold = load_model(MODEL_PATH)
    pred_paths = predict_test_tiles(model, threshold)

    if pred_paths:
        plot_predictions(pred_paths, threshold, n=min(4, len(pred_paths)))
        plot_prediction_summary(pred_paths, threshold)
        print(f"\n✅ {len(pred_paths)} tiles predicted.")
        print(f"   GeoTIFFs in {PRED_DIR}/  (band1=mask, band2=prob, band3=year)")
    else:
        print("❌ No predictions generated.")
