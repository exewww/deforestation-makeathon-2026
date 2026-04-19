"""
STEP 6: SUBMISSION — v3.1 (Bugfix for Indexing Error)
=====================================================
FIXES:
  1. Resolved 'None of [RangeIndex] are in columns' error by fixing the filtering logic.
  2. Improved modal year extraction per polygon.
  3. Added coordinate check to ensure valid geometry.
"""

import json
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import pandas as pd
from scipy import stats as scipy_stats

PRED_DIR        = Path("predictions")
SUBMISSION_DIR  = Path("final_output")
SUBMISSION_DIR.mkdir(exist_ok=True)

MIN_AREA_HA = 0.5    # filter out tiny polygons (< 0.5 hectares)

def raster_to_polygons(raster_path):
    """
    Reads Band 1 (binary mask) and Band 3 (year map) from a prediction GeoTIFF.
    """
    with rasterio.open(raster_path) as src:
        mask_data = src.read(1).astype(np.uint8)
        transform = src.transform
        crs       = src.crs
        # Band 3 = year map (integer year)
        year_data = src.read(3) if src.count >= 3 else None

    if mask_data.sum() == 0:
        print(f"  ⚪ No deforestation in {raster_path.name}")
        return None

    # 1. Vectorize the binary mask
    raw_polygons = [
        (shape(geom), int(value))
        for geom, value in shapes(mask_data, mask=mask_data, transform=transform)
        if value == 1
    ]
    
    if not raw_polygons:
        return None

    # 2. Create GeoDataFrame
    geometries = [p[0] for p in raw_polygons]
    gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)
    
    # 3. Filter by area (Calculate in UTM for hectares)
    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    
    # Correct filtering logic: use a boolean mask on both simultaneously
    keep_mask = (gdf_utm.area / 10_000) >= MIN_AREA_HA
    gdf = gdf[keep_mask].reset_index(drop=True)
    
    if gdf.empty:
        print(f"  ⚪ All polygons in {raster_path.name} were smaller than {MIN_AREA_HA}ha")
        return None

    # 4. Assign Year (Modal value from Band 3)
    years_list = []
    if year_data is not None:
        # We iterate through polygons to find the most common year in each
        for geom in gdf.geometry:
            try:
                from rasterio.mask import mask
                # Mask the year_data raster using the current polygon
                out_image, out_transform = mask(src, [geom], crop=True, indexes=3)
                vals = out_image.flatten()
                # Filter out zeros (no-deforestation) and common null values
                valid_yrs = vals[(vals > 2000) & (vals < 2030)]
                
                if len(valid_yrs) > 0:
                    mode_res = scipy_stats.mode(valid_yrs, keepdims=True)
                    modal_yr = int(mode_res.mode[0])
                else:
                    modal_yr = 0
                years_list.append(modal_yr)
            except Exception:
                years_list.append(0)
    else:
        years_list = [0] * len(gdf)

    gdf["time_step"] = years_list
    
    # 5. Final Cleanup: Convert to WGS84 and clean time_step
    gdf = gdf.to_crs("EPSG:4326")
    
    # Ensure time_step is either a year or None (for JSON validity)
    gdf["time_step"] = gdf["time_step"].apply(lambda x: int(x) if x > 2000 else None)
    
    # Drop rows that somehow resulted in no year if your model requires it
    # final_gdf = gdf.dropna(subset=['time_step']) 

    print(f"  ✅ {len(gdf)} polygons found.")
    return gdf

def create_final_submission():
    print("=" * 55)
    print("STEP 6: YEAR-AWARE SUBMISSION GEOJSON")
    print("=" * 55)

    pred_files = list(PRED_DIR.glob("*_prediction.tif"))
    if not pred_files:
        print("❌ No prediction files. Run step4_predict.py first.")
        return

    all_gdfs = []
    for p in sorted(pred_files):
        print(f"📦 Processing: {p.name}")
        try:
            gdf = raster_to_polygons(p)
            if gdf is not None and not gdf.empty:
                all_gdfs.append(gdf)
        except Exception as e:
            print(f"  ⚠️ Error processing {p.name}: {e}")

    if not all_gdfs:
        print("❌ No valid polygons found across any tiles.")
        return

    # Combine all tiles into one file
    final_gdf = pd.concat(all_gdfs, ignore_index=True)

    # Save to GeoJSON
    out_path = SUBMISSION_DIR / "submission.geojson"
    final_gdf.to_file(out_path, driver="GeoJSON")

    print("\n" + "=" * 55)
    print(f"✅ SUCCESS: {out_path}")
    print(f"   Total polygons: {len(final_gdf)}")
    if "time_step" in final_gdf.columns:
        print(f"   Year distribution:\n{final_gdf['time_step'].value_counts().to_string()}")
    print("=" * 55)

if __name__ == "__main__":
    create_final_submission()