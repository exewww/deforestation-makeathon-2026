"""
STEP 1: DATA EXPLORATION
Run this first! It will show you your data visually.
Produces: explore_output/ folder with PNG images of your tiles, labels, and sensor data.

pip install rasterio matplotlib numpy geopandas
"""

import os, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

DATA_ROOT = Path("data/makeathon-challenge")
OUT_DIR = Path("explore_output")
OUT_DIR.mkdir(exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def get_tile_id(path_str):
    match = re.search(r"([0-9]{2}[A-Z]{3}_\d_\d)", str(path_str))
    return match.group(1) if match else None

def norm(arr, p_low=2, p_high=98):
    lo, hi = np.percentile(arr, p_low), np.percentile(arr, p_high)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)

# ── 1. Show GeoJSON tile footprints ──────────────────────────────────────────

def plot_tile_footprints():
    try:
        import geopandas as gpd
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, split in zip(axes, ["train", "test"]):
            gdf_path = DATA_ROOT / "metadata" / f"{split}_tiles.geojson"
            if gdf_path.exists():
                gdf = gpd.read_file(gdf_path)
                gdf.plot(ax=ax, color="green", edgecolor="black", alpha=0.5)
                for idx, row in gdf.iterrows():
                    c = row.geometry.centroid
                    ax.annotate(row["name"], (c.x, c.y), fontsize=7, ha="center")
                ax.set_title(f"{split.upper()} tiles ({len(gdf)})")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "01_tile_footprints.png", dpi=120)
        plt.close()
        print("✅ Saved: 01_tile_footprints.png")
    except Exception as e:
        print(f"⚠️  Tile footprint plot skipped: {e}")

# ── 2. Show one S2 RGB image ──────────────────────────────────────────────────

def plot_s2_sample():
    import rasterio
    s2_files = sorted((DATA_ROOT / "sentinel-2" / "train").rglob("*.tif"))
    if not s2_files:
        print("⚠️  No S2 files found"); return

    fig, axes = plt.subplots(1, min(3, len(s2_files)), figsize=(15, 5))
    if len(s2_files) == 1:
        axes = [axes]

    for ax, f in zip(axes, s2_files[:3]):
        with rasterio.open(f) as src:
            # Bands 4=Red, 3=Green, 2=Blue (1-indexed), scale to 0-1
            if src.count >= 4:
                rgb = np.stack([src.read(4), src.read(3), src.read(2)], axis=-1) / 10000.0
            else:
                band = src.read(1)
                rgb = np.stack([band]*3, axis=-1) / band.max()
        rgb = norm(rgb)
        ax.imshow(rgb)
        ax.set_title(f.name[:30], fontsize=7)
        ax.axis("off")

    plt.suptitle("Sentinel-2 RGB Preview (Bands 4-3-2)", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_sentinel2_rgb.png", dpi=120)
    plt.close()
    print("✅ Saved: 02_sentinel2_rgb.png")

# ── 3. Show S1 SAR image ─────────────────────────────────────────────────────

def plot_s1_sample():
    import rasterio
    s1_files = sorted((DATA_ROOT / "sentinel-1" / "train").rglob("*.tif"))
    if not s1_files:
        print("⚠️  No S1 files found"); return

    fig, axes = plt.subplots(1, min(3, len(s1_files)), figsize=(15, 5))
    if len(s1_files) == 1:
        axes = [axes]

    for ax, f in zip(axes, s1_files[:3]):
        with rasterio.open(f) as src:
            band = src.read(1)
        band_db = 10 * np.log10(np.clip(band, 1e-4, None))
        ax.imshow(band_db, cmap="gray")
        ax.set_title(f.name[:30], fontsize=7)
        ax.axis("off")

    plt.suptitle("Sentinel-1 SAR (dB scale)", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_sentinel1_sar.png", dpi=120)
    plt.close()
    print("✅ Saved: 03_sentinel1_sar.png")

# ── 4. Show label masks overlaid on S2 ───────────────────────────────────────

def plot_labels_vs_s2():
    import rasterio
    from rasterio.enums import Resampling

    label_root = DATA_ROOT / "labels" / "train"
    label_types = ["gladl", "glads2", "radd"]
    colors = {"gladl": "red", "glads2": "orange", "radd": "yellow"}

    # Find a tile that has labels
    s2_files = sorted((DATA_ROOT / "sentinel-2" / "train").rglob("*.tif"))
    for s2f in s2_files:
        tid = get_tile_id(s2f)
        labels_found = {}
        for lt in label_types:
            lfiles = list((label_root / lt).rglob(f"*{tid}*.tif")) if (label_root / lt).exists() else []
            if lfiles:
                labels_found[lt] = lfiles[0]
        if labels_found:
            break
    else:
        print("⚠️  No tile with labels found"); return

    with rasterio.open(s2f) as src:
        if src.count >= 4:
            rgb = np.stack([src.read(4), src.read(3), src.read(2)], axis=-1) / 10000.0
        else:
            b = src.read(1); rgb = np.stack([b]*3, axis=-1) / (b.max()+1e-8)
        h, w = src.height, src.width
    rgb = norm(rgb)

    fig, axes = plt.subplots(1, len(labels_found) + 1, figsize=(5 * (len(labels_found)+1), 5))
    axes[0].imshow(rgb); axes[0].set_title(f"S2 RGB\n{tid}"); axes[0].axis("off")

    for ax, (lt, lp) in zip(axes[1:], labels_found.items()):
        with rasterio.open(lp) as src:
            mask = src.read(1, out_shape=(h, w), resampling=Resampling.nearest)
        ax.imshow(rgb)
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask > 0] = [1, 0, 0, 0.6]  # red where deforested
        ax.imshow(overlay)
        defor_pct = 100 * (mask > 0).mean()
        ax.set_title(f"{lt}\n{defor_pct:.1f}% deforested", fontsize=9)
        ax.axis("off")

    plt.suptitle(f"Label comparison for tile {tid}", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_labels_vs_s2.png", dpi=120)
    plt.close()
    print("✅ Saved: 04_labels_vs_s2.png")

# ── 5. AEF embedding PCA visualization ───────────────────────────────────────

def plot_aef_pca():
    import rasterio
    from sklearn.decomposition import PCA

    aef_files = sorted((DATA_ROOT / "aef-embeddings" / "train").rglob("*.tiff"))
    if not aef_files:
        print("⚠️  No AEF files found"); return

    f = aef_files[0]
    with rasterio.open(f) as src:
        data = src.read()  # (64, H, W)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    C, H, W = data.shape
    flat = data.reshape(C, -1).T  # (H*W, 64)
    # subsample for speed
    idx = np.random.choice(len(flat), min(50000, len(flat)), replace=False)
    pca = PCA(n_components=3)
    rgb_pca = pca.fit_transform(flat[idx])
    # Map back to image
    full_pca = pca.transform(flat)
    pca_img = full_pca.reshape(H, W, 3)
    pca_img = norm(pca_img)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(pca_img)
    axes[0].set_title(f"AEF Embeddings — PCA(3)\n{f.name}", fontsize=9)
    axes[0].axis("off")

    explained = pca.explained_variance_ratio_ * 100
    axes[1].bar(["PC1", "PC2", "PC3"], explained, color=["#2196F3","#4CAF50","#FF9800"])
    axes[1].set_ylabel("Variance Explained (%)")
    axes[1].set_title("PCA Explained Variance")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_aef_pca.png", dpi=120)
    plt.close()
    print("✅ Saved: 05_aef_pca.png")

# ── 6. Dataset statistics ─────────────────────────────────────────────────────

def plot_dataset_stats():
    s2_files = list((DATA_ROOT / "sentinel-2" / "train").rglob("*.tif"))
    s1_files = list((DATA_ROOT / "sentinel-1" / "train").rglob("*.tif"))
    aef_files = list((DATA_ROOT / "aef-embeddings" / "train").rglob("*.tiff"))

    # Parse dates from S2
    months = []
    for f in s2_files:
        parts = f.stem.split("_")
        try:
            months.append(int(parts[-1]))
        except:
            pass

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # File counts
    counts = [len(s2_files), len(s1_files), len(aef_files)]
    labels = [f"S2\n({counts[0]})", f"S1\n({counts[1]})", f"AEF\n({counts[2]})"]
    axes[0].bar(labels, counts, color=["#4CAF50","#2196F3","#FF9800"])
    axes[0].set_title("Training Files per Sensor")
    axes[0].set_ylabel("Count")

    # Month distribution
    if months:
        axes[1].hist(months, bins=12, range=(1,13), color="#4CAF50", edgecolor="white")
        axes[1].set_title("S2 Temporal Distribution")
        axes[1].set_xlabel("Month"); axes[1].set_ylabel("Count")
        axes[1].set_xticks(range(1, 13))
        axes[1].set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])

    # Label counts
    label_counts = {}
    for lt in ["gladl", "glads2", "radd"]:
        lroot = DATA_ROOT / "labels" / "train" / lt
        if lroot.exists():
            label_counts[lt] = len(list(lroot.rglob("*.tif")))
    if label_counts:
        axes[2].bar(list(label_counts.keys()), list(label_counts.values()),
                    color=["#F44336","#FF9800","#FFEB3B"])
        axes[2].set_title("Label Files per Source")
        axes[2].set_ylabel("Count")

    plt.suptitle("Dataset Statistics", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_dataset_stats.png", dpi=120)
    plt.close()
    print("✅ Saved: 06_dataset_stats.png")

# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*50)
    print("STEP 1: DATA EXPLORATION")
    print(f"Output directory: {OUT_DIR.resolve()}")
    print("="*50)

    plot_tile_footprints()
    plot_s2_sample()
    plot_s1_sample()
    plot_labels_vs_s2()
    try:
        plot_aef_pca()
    except ImportError:
        print("⚠️  sklearn not installed, skipping AEF PCA. pip install scikit-learn")
    plot_dataset_stats()

    print("\n✅ Done! Check explore_output/ for your images.")
    print("   If everything looks good, run: python step2_dataset.py")
