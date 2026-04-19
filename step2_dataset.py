"""
STEP 2: DATASET — v3 (Year prediction + temporal stack + FPR fix)
==================================================================
WHY THIS VERSION:
  Leaderboard showed: FPR=87%, Year=0.0%, IoU=10%

  Problems fixed:
  A. Year=0%  → Now output has 2 channels: [binary_mask, normalized_year]
               Year is decoded from GLAD-L (YYDOY format) and GLAD-S2/RADD (days format)
  B. FPR=87%  → filter_empty=True skips patches with <1% deforestation during training.
               This stops the model from learning "predict forest everywhere."
  C. IoU=10%  → Temporal stack: S2_early + S2_late + delta_NDVI gives explicit before/after.
               Model can now SEE the change, not just a single snapshot.

Input:  76 channels = S2e(4)+S2l(4)+NDVIe(1)+NDVIl(1)+dNDVI(1)+S1(1)+AEF(64)
Output: 2 channels  = binary_mask(1) + year_normalized(1)
"""

import re, torch, rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from torch.utils.data import Dataset
from pathlib import Path
from datetime import datetime

DATA_ROOT    = Path("data/makeathon-challenge")
OUT_DIR      = Path("dataset_output")
OUT_DIR.mkdir(exist_ok=True)

PATCH_SIZE    = 256
BATCH_SIZE    = 4
IN_CHANNELS   = 76    # S2e(4)+S2l(4)+NDVIe(1)+NDVIl(1)+dNDVI(1)+S1(1)+AEF(64)
MIN_POS_RATIO = 0.01  # skip patches with <1% deforestation (FPR fix)
YEAR_MIN, YEAR_MAX = 2015, 2024


def get_tile_id(path_str):
    m = re.search(r"([0-9]{2}[A-Z]{3}_\d_\d)", str(path_str))
    return m.group(1) if m else None

def get_date_values(year, month):
    base   = datetime(2014, 12, 31)
    cur    = datetime(year, month, 1)
    days   = (cur - base).days
    yydoy  = (year % 100) * 1000 + int(cur.strftime("%j"))
    return days, yydoy

def year_to_norm(year_arr):
    """Float/int array of years → normalized [0,1]."""
    return np.clip((year_arr - YEAR_MIN) / (YEAR_MAX - YEAR_MIN), 0.0, 1.0).astype(np.float32)

def norm_to_year(val):
    """Single normalized float → integer year (for display/submission)."""
    return int(round(float(val) * (YEAR_MAX - YEAR_MIN) + YEAR_MIN))


# ─────────────────────────────────────────────────────────────────────────────
# Index builder
# ─────────────────────────────────────────────────────────────────────────────

def build_index(split="train"):
    index = {}

    for tif in (DATA_ROOT / "sentinel-2" / split).rglob("*.tif"):
        tid = get_tile_id(tif.name)
        if not tid: continue
        index.setdefault(tid, {"s2": [], "s1": [], "labels": {}, "aef": {}})
        parts = tif.stem.split("_")
        try:
            y, m        = int(parts[-2]), int(parts[-1])
            days, yydoy = get_date_values(y, m)
            index[tid]["s2"].append(
                {"path": str(tif), "y": y, "m": m, "days": days, "yydoy": yydoy}
            )
        except (ValueError, IndexError):
            pass

    for tif in (DATA_ROOT / "sentinel-1" / split).rglob("*.tif"):
        tid = get_tile_id(tif.name)
        if not tid or tid not in index: continue
        parts = tif.stem.split("_")
        try:
            y, m    = int(parts[-3]), int(parts[-2])
            days, _ = get_date_values(y, m)
            index[tid]["s1"].append({"path": str(tif), "days": days, "y": y})
        except (ValueError, IndexError):
            pass

    for lt in ["gladl", "glads2", "radd"]:
        lroot = DATA_ROOT / "labels" / split / lt
        if not lroot.exists(): continue
        for tif in lroot.rglob("*.tif"):
            tid = get_tile_id(tif.name)
            if tid and tid in index:
                index[tid]["labels"][lt] = str(tif)

    aef_root = DATA_ROOT / "aef-embeddings" / split
    if aef_root.exists():
        for tif in aef_root.rglob("*.tiff"):
            tid = get_tile_id(tif.name)
            try:   year = int(tif.stem.split("_")[-1])
            except ValueError: continue
            if tid and tid in index:
                index[tid]["aef"][year] = str(tif)

    with_labels = sum(1 for v in index.values() if v["labels"])
    with_aef    = sum(1 for v in index.values() if v["aef"])
    print(f"Index [{split}]: {len(index)} tiles | {with_labels} w/ labels | {with_aef} w/ AEF")
    return index


# ─────────────────────────────────────────────────────────────────────────────
# I/O helper
# ─────────────────────────────────────────────────────────────────────────────

def read_warped(path, bands, ref_crs, ref_transform, ref_h, ref_w,
                win, out_h, out_w, resample=Resampling.bilinear):
    with rasterio.open(path) as src:
        vrt_options = {
            "crs": ref_crs, "transform": ref_transform,
            "width": ref_w, "height": ref_h, "resampling": resample,
        }
        with WarpedVRT(src, **vrt_options) as vrt:
            data = vrt.read(bands, window=win,
                            out_shape=(len(bands), out_h, out_w),
                            resampling=resample)
    return np.nan_to_num(data.astype(np.float32), nan=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Label decoder → binary mask + year map
# ─────────────────────────────────────────────────────────────────────────────

def decode_labels(label_dict, ref_crs, ref_transform, H, W, win, ps, s2_rec):
    """
    Returns:
        binary   (ps,ps) float32 : 1 = deforested at or before s2_rec date
        year_norm(ps,ps) float32 : normalized year [0,1] of event (0 = no event)

    gladl  pixel value = YYDOY (e.g. 20001 = day 1 of year 2020)
    glads2 pixel value = days since 2014-12-31
    radd   pixel value = days since 2014-12-31
    """
    binary   = np.zeros((ps, ps), dtype=np.float32)
    year_map = np.zeros((ps, ps), dtype=np.float32)   # raw year float, 0 = none

    for lt, lp in label_dict.items():
        raw = read_warped(lp, [1], ref_crs, ref_transform, H, W,
                          win, ps, ps, resample=Resampling.nearest)[0]
        raw_int = raw.astype(np.int32)

        if lt == "gladl":
            # YYDOY: year = (raw // 1000) + 2000
            event_year = (raw_int // 1000 + 2000).astype(np.float32)
            detected   = (raw_int > 0) & (raw_int <= s2_rec["yydoy"])
        else:
            # days since 2014-12-31: approximate year
            event_year = np.where(raw_int > 0,
                                  raw_int / 365.25 + 2015.0, 0.0).astype(np.float32)
            detected   = (raw_int > 0) & (raw_int <= s2_rec["days"])

        binary = np.maximum(binary, detected.astype(np.float32))

        # Year: keep earliest detection across sources
        new_det  = detected & (year_map == 0)
        both_det = detected & (year_map > 0)
        year_map[new_det]  = event_year[new_det]
        year_map[both_det] = np.minimum(year_map[both_det], event_year[both_det])

    year_norm = np.where(year_map > 0, year_to_norm(year_map), 0.0).astype(np.float32)
    return binary, year_norm


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ForestDataset(Dataset):
    """
    X shape: (76, ps, ps)
    y shape: ( 2, ps, ps)
      y[0] = binary deforestation mask
      y[1] = normalized year of deforestation [0,1], 0 = no event
    """

    def __init__(self, index, patch_size=PATCH_SIZE, augment=False,
                 filter_empty=True, repeats_per_tile=20):
        self.ps             = patch_size
        self.augment        = augment
        self.filter_empty   = filter_empty
        self.repeats        = repeats_per_tile
        self.samples        = []

        for tid, data in index.items():
            if not data["s2"] or not data["labels"]:
                continue
            s2_sorted = sorted(data["s2"], key=lambda x: x["days"])
            self.samples.append({
                "tid":      tid,
                "s2_early": s2_sorted[0],
                "s2_late":  s2_sorted[-1],
                "s1_list":  data["s1"],
                "labels":   data["labels"],
                "aef":      data["aef"],
            })

        print(f"Dataset: {len(self.samples)} tiles × {repeats_per_tile} crops = "
              f"{len(self.samples)*repeats_per_tile} samples  "
              f"(filter_empty={filter_empty}), {IN_CHANNELS}ch")

    def __len__(self):
        return len(self.samples) * self.repeats

    def __getitem__(self, idx):
        s  = self.samples[idx % len(self.samples)]
        ps = self.ps

        with rasterio.open(s["s2_late"]["path"]) as ref:
            H, W          = ref.height, ref.width
            ref_crs       = ref.crs
            ref_transform = ref.transform

        # Retry up to 20 crops to find a patch with enough positive pixels
        for attempt in range(20):
            y0  = np.random.randint(0, max(1, H - ps))
            x0  = np.random.randint(0, max(1, W - ps))
            win = rasterio.windows.Window(x0, y0, ps, ps)

            binary, year_norm = decode_labels(
                s["labels"], ref_crs, ref_transform, H, W, win, ps, s["s2_late"]
            )

            if self.filter_empty and binary.mean() < MIN_POS_RATIO and attempt < 19:
                continue
            break

        # ── S2 LATE ──────────────────────────────────────────────────────
        with rasterio.open(s["s2_late"]["path"]) as src:
            s2l_raw = src.read([4, 3, 2, 8], window=win,
                               out_shape=(4, ps, ps), resampling=Resampling.bilinear)
        s2l = np.nan_to_num(s2l_raw.astype(np.float32), nan=0.0) / 10000.0
        ndvi_late  = (s2l[3] - s2l[0]) / (s2l[3] + s2l[0] + 1e-7)

        # ── S2 EARLY (warped) ─────────────────────────────────────────────
        s2e = read_warped(s["s2_early"]["path"], [4, 3, 2, 8],
                          ref_crs, ref_transform, H, W, win, ps, ps) / 10000.0
        ndvi_early = (s2e[3] - s2e[0]) / (s2e[3] + s2e[0] + 1e-7)
        delta_ndvi = np.clip(ndvi_late - ndvi_early, -1.0, 1.0)

        # ── S1 ────────────────────────────────────────────────────────────
        if s["s1_list"]:
            s1_rec = min(s["s1_list"],
                         key=lambda x: abs(x["days"] - s["s2_late"]["days"]))
            s1_raw = read_warped(s1_rec["path"], [1],
                                 ref_crs, ref_transform, H, W, win, ps, ps)
            s1 = (np.clip(10 * np.log10(np.clip(s1_raw, 1e-4, 1.0)), -20, 0) + 20) / 20.0
        else:
            s1 = np.zeros((1, ps, ps), dtype=np.float32)

        # ── AEF ───────────────────────────────────────────────────────────
        if s["aef"]:
            yr       = s["s2_late"]["y"]
            aef_path = s["aef"].get(yr, next(iter(s["aef"].values())))
            aef = read_warped(aef_path, list(range(1, 65)),
                              ref_crs, ref_transform, H, W, win, ps, ps)
        else:
            aef = np.zeros((64, ps, ps), dtype=np.float32)

        # ── Augmentation ─────────────────────────────────────────────────
        if self.augment:
            for axis in [1, 2]:
                if np.random.rand() > 0.5:
                    s2l        = np.flip(s2l,        axis=axis).copy()
                    s2e        = np.flip(s2e,        axis=axis).copy()
                    s1         = np.flip(s1,         axis=axis).copy()
                    aef        = np.flip(aef,        axis=axis).copy()
                    ndvi_late  = np.flip(ndvi_late,  axis=axis-1).copy()
                    ndvi_early = np.flip(ndvi_early, axis=axis-1).copy()
                    delta_ndvi = np.flip(delta_ndvi, axis=axis-1).copy()
                    binary     = np.flip(binary,     axis=axis-1).copy()
                    year_norm  = np.flip(year_norm,  axis=axis-1).copy()

            k = np.random.randint(0, 4)
            if k > 0:
                s2l        = np.rot90(s2l,        k, axes=(1, 2)).copy()
                s2e        = np.rot90(s2e,        k, axes=(1, 2)).copy()
                s1         = np.rot90(s1,         k, axes=(1, 2)).copy()
                aef        = np.rot90(aef,        k, axes=(1, 2)).copy()
                ndvi_late  = np.rot90(ndvi_late,  k, axes=(0, 1)).copy()
                ndvi_early = np.rot90(ndvi_early, k, axes=(0, 1)).copy()
                delta_ndvi = np.rot90(delta_ndvi, k, axes=(0, 1)).copy()
                binary     = np.rot90(binary,     k, axes=(0, 1)).copy()
                year_norm  = np.rot90(year_norm,  k, axes=(0, 1)).copy()

            if np.random.rand() > 0.7:
                s2l = np.clip(s2l + np.random.normal(0, 0.01, s2l.shape).astype(np.float32), 0, 1)

        # ── Final stack: 76 channels ──────────────────────────────────────
        X = np.concatenate([
            s2e,                       # 0-3:  S2 early
            s2l,                       # 4-7:  S2 late
            ndvi_early[np.newaxis],    # 8:    NDVI early
            ndvi_late[np.newaxis],     # 9:    NDVI late
            delta_ndvi[np.newaxis],    # 10:   delta NDVI ← key signal
            s1,                        # 11:   S1 SAR
            aef,                       # 12-75: AEF
        ], axis=0)

        y = np.stack([binary, year_norm], axis=0)

        return torch.from_numpy(X).float(), torch.from_numpy(y).float()


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 2 — Temporal stack + Year labels (76ch, 2-output)")
    print("=" * 60)
    index   = build_index("train")
    dataset = ForestDataset(index, patch_size=PATCH_SIZE, filter_empty=True)
    X, y = dataset[0]
    print(f"  X : {X.shape}   expected [76, {PATCH_SIZE}, {PATCH_SIZE}]")
    print(f"  y : {y.shape}   expected [ 2, {PATCH_SIZE}, {PATCH_SIZE}]")
    print(f"  Binary positive: {y[0].mean()*100:.1f}%")
    print(f"  Year non-zero:   {(y[1]>0).float().mean()*100:.1f}%")
    active = y[1][y[1] > 0]
    if len(active):
        years = sorted(set([norm_to_year(float(v)) for v in active[:200]]))
        print(f"  Years detected: {years}")
    print(f"  NaN in X: {torch.isnan(X).any()}")
    print("\n✅ Run step3_train.py next.")
