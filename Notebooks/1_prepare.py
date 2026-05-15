"""
1_prepare.py
============
Aligns all inputs to the Sentinel-2 reference grid (EPSG:32630, 10 m)
and generates visualizations:
  - output/original/   → raw images as downloaded
  - output/aligned/    → inputs reprojected to the same grid
  - the target (DTM Lidar HD) is kept as-is for now

Inputs (in data/):
  sentinel2_caen.tif   — 5 bands (B02, B03, B04, B08, NDVI), EPSG:32630, 10 m
  copdem30_caen.tif    — 1 band, EPSG:4326, ~30 m
  tcd_caen.tif         — 4 bands RGBA (WMS render), EPSG:4326, ~10 m
  imd_caen.tif         — 4 bands RGBA (WMS render), EPSG:4326, ~10 m
  dtm_lidar_caen.tif   — 1 band, EPSG:4326, ~10 m

Outputs (in data/):
  aligned_sentinel2.tif  — already on the right grid, copied as-is
  aligned_copdem30.tif   — reprojected + resampled to 10 m
  aligned_tcd.tif        — reprojected, band 1 only (greyscale from render)
  aligned_imd.tif        — reprojected, band 1 only
  dtm_lidar_caen.tif     — unchanged (target)
"""

import os
import sys
import shutil
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.crs import CRS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# 1. Paths
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR   = BASE_DIR / "output"
ORIGINAL_DIR = OUTPUT_DIR / "original"
ALIGNED_DIR  = OUTPUT_DIR / "aligned"

ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ZONE = "caen"

FILES = {
    "sentinel2": DATA_DIR / f"sentinel2_{ZONE}.tif",
    "copdem30":  DATA_DIR / f"copdem30_{ZONE}.tif",
    "tcd":       DATA_DIR / f"tcd_{ZONE}.tif",
    "imd":       DATA_DIR / f"imd_{ZONE}.tif",
    "dtm_lidar": DATA_DIR / f"dtm_lidar_{ZONE}.tif",
}

# Check that all files exist
for name, path in FILES.items():
    if not path.exists():
        print(f"ERROR: missing file — {path}")
        sys.exit(1)

# =============================================================================
# 2. Reference grid (Sentinel-2)
# =============================================================================
print("=" * 60)
print("REFERENCE GRID (Sentinel-2)")
print("=" * 60)

with rasterio.open(FILES["sentinel2"]) as src_ref:
    REF_CRS = src_ref.crs
    REF_TRANSFORM = src_ref.transform
    REF_WIDTH = src_ref.width
    REF_HEIGHT = src_ref.height
    REF_BOUNDS = src_ref.bounds

print(f"  CRS:        {REF_CRS}")
print(f"  Size:       {REF_WIDTH} x {REF_HEIGHT}")
print(f"  Resolution: {REF_TRANSFORM.a:.1f} m")
print(f"  Bounds:     {REF_BOUNDS}")


# =============================================================================
# 3. Functions
# =============================================================================
def reproject_to_grid(input_path, output_path, band=None,
                      resampling=Resampling.bilinear):
    """Reproject a raster to the Sentinel-2 reference grid."""
    with rasterio.open(input_path) as src:
        n_bands = 1 if band else src.count
        band_indices = [band] if band else list(range(1, src.count + 1))

        dst_profile = src.profile.copy()
        dst_profile.update(
            crs=REF_CRS,
            transform=REF_TRANSFORM,
            width=REF_WIDTH,
            height=REF_HEIGHT,
            count=n_bands,
            dtype="float32",
            driver="GTiff",
            compress="deflate",
            nodata=np.nan,
        )

        with rasterio.open(output_path, "w", **dst_profile) as dst:
            for i, idx in enumerate(band_indices, start=1):
                src_data = src.read(idx).astype("float32")
                # Handle nodata
                if src.nodata is not None:
                    src_data[src_data == src.nodata] = np.nan

                dst_data = np.full((REF_HEIGHT, REF_WIDTH), np.nan, dtype="float32")

                reproject(
                    source=src_data,
                    destination=dst_data,
                    src_transform=src.transform,
                    src_crs=src.crs if src.crs else CRS.from_epsg(4326),
                    dst_transform=REF_TRANSFORM,
                    dst_crs=REF_CRS,
                    resampling=resampling,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                )
                dst.write(dst_data, i)

    with rasterio.open(output_path) as src:
        d = src.read(1)
        valid = np.isfinite(d)
        print(f"  → {output_path}")
        print(f"    Shape: {src.width}x{src.height}, {src.count} band(s)")
        print(f"    min={np.nanmin(d):.4f}, max={np.nanmax(d):.4f}, "
              f"NaN={np.sum(~valid)} ({100*np.sum(~valid)/d.size:.1f}%)")

    return output_path


def reproject_rgba_wms(input_path, output_path,
                       resampling=Resampling.nearest):
    """Reproject an RGBA raster from WMS.

    Converts the 3 RGB bands to greyscale (luminance),
    uses the alpha channel (band 4) as mask (alpha==0 → NaN).
    """
    with rasterio.open(input_path) as src:
        r = src.read(1).astype("float32")
        g = src.read(2).astype("float32")
        b = src.read(3).astype("float32")
        alpha = src.read(4) if src.count >= 4 else np.full_like(r, 255, dtype="uint8")

        # Greyscale via luminance
        grey = 0.299 * r + 0.587 * g + 0.114 * b

        # Mask transparent pixels
        grey[alpha == 0] = np.nan

        dst_profile = src.profile.copy()
        dst_profile.update(
            crs=REF_CRS,
            transform=REF_TRANSFORM,
            width=REF_WIDTH,
            height=REF_HEIGHT,
            count=1,
            dtype="float32",
            driver="GTiff",
            compress="deflate",
            nodata=np.nan,
        )

        dst_data = np.full((REF_HEIGHT, REF_WIDTH), np.nan, dtype="float32")

        reproject(
            source=grey,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs if src.crs else CRS.from_epsg(4326),
            dst_transform=REF_TRANSFORM,
            dst_crs=REF_CRS,
            resampling=resampling,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

        with rasterio.open(output_path, "w", **dst_profile) as dst:
            dst.write(dst_data, 1)

    with rasterio.open(output_path) as src:
        d = src.read(1)
        valid = np.isfinite(d)
        print(f"  → {output_path}")
        print(f"    Shape: {src.width}x{src.height}, {src.count} band(s)")
        print(f"    min={np.nanmin(d):.4f}, max={np.nanmax(d):.4f}, "
              f"NaN={np.sum(~valid)} ({100*np.sum(~valid)/d.size:.1f}%)")

    return output_path


def plot_raster(path, title, fig_path, band=1, cmap="terrain",
                vmin=None, vmax=None, nodata_val=None):
    """Generate a PNG figure for a single-band raster."""
    with rasterio.open(path) as src:
        data = src.read(band).astype("float32")
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        if nodata_val is not None:
            data[data == nodata_val] = np.nan

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Column (pixel)")
    ax.set_ylabel("Row (pixel)")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Value")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure: {fig_path}")


def plot_rgb(path, title, fig_path, bands=(3, 2, 1), stretch=3.0):
    """Generate an RGB composite figure for Sentinel-2 (R=B04, G=B03, B=B02)."""
    with rasterio.open(path) as src:
        rgb = np.stack([src.read(b).astype("float32") for b in bands], axis=-1)

    # Stretch for visualization
    rgb = np.clip(rgb * stretch, 0, 1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(rgb)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Column (pixel)")
    ax.set_ylabel("Row (pixel)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure: {fig_path}")


def plot_rgba_wms(path, title, fig_path):
    """Generate a figure for an RGBA raster from WMS."""
    with rasterio.open(path) as src:
        if src.count >= 3:
            rgb = np.stack([src.read(b) for b in [1, 2, 3]], axis=-1)
        else:
            rgb = src.read(1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(rgb)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Column (pixel)")
    ax.set_ylabel("Row (pixel)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure: {fig_path}")


# =============================================================================
# 4. Visualizations — ORIGINAL data
# =============================================================================
print("\n" + "=" * 60)
print("FIGURES — ORIGINAL DATA")
print("=" * 60)

# Sentinel-2 RGB
print("\n--- Sentinel-2 (RGB composite) ---")
plot_rgb(
    FILES["sentinel2"],
    "Sentinel-2 L2A — RGB (original)",
    os.path.join(ORIGINAL_DIR, "sentinel2_rgb.png"),
    bands=(3, 2, 1), stretch=3.0,
)

# Sentinel-2 NDVI
print("\n--- Sentinel-2 NDVI ---")
plot_raster(
    FILES["sentinel2"],
    "Sentinel-2 — NDVI (original)",
    os.path.join(ORIGINAL_DIR, "sentinel2_ndvi.png"),
    band=5, cmap="RdYlGn", vmin=-0.2, vmax=0.9,
)

# Copernicus DEM
print("\n--- Copernicus DEM 30 m ---")
plot_raster(
    FILES["copdem30"],
    "Copernicus DEM 30 m (original)",
    os.path.join(ORIGINAL_DIR, "copdem30.png"),
    cmap="terrain",
)

# TCD (WMS RGBA render)
print("\n--- TCD (WMS render) ---")
plot_rgba_wms(
    FILES["tcd"],
    "TCD — Tree Cover Density (WMS render, original)",
    os.path.join(ORIGINAL_DIR, "tcd.png"),
)

# IMD (WMS RGBA render)
print("\n--- IMD (WMS render) ---")
plot_rgba_wms(
    FILES["imd"],
    "IMD — Imperviousness (WMS render, original)",
    os.path.join(ORIGINAL_DIR, "imd.png"),
)

# DTM Lidar HD
print("\n--- DTM Lidar HD (target) ---")
plot_raster(
    FILES["dtm_lidar"],
    "DTM Lidar HD — high-resolution DTM (target)",
    os.path.join(ORIGINAL_DIR, "dtm_lidar.png"),
    cmap="terrain", nodata_val=-99999.0,
)


# =============================================================================
# 5. Alignment to Sentinel-2 grid
# =============================================================================
print("\n" + "=" * 60)
print("ALIGNMENT TO SENTINEL-2 GRID")
print("=" * 60)

# 5a. Sentinel-2 — already on the right grid, just copy
print("\n--- Sentinel-2 (already aligned) ---")
s2_aligned_path = os.path.join(DATA_DIR, "aligned_sentinel2.tif")
shutil.copy2(FILES["sentinel2"], s2_aligned_path)
print(f"  → {s2_aligned_path} (copy)")

# 5b. Copernicus DEM → reprojected to EPSG:32630, 10 m
print("\n--- Copernicus DEM 30 m → 10 m UTM ---")
dem_aligned_path = os.path.join(DATA_DIR, "aligned_copdem30.tif")
reproject_to_grid(
    FILES["copdem30"], dem_aligned_path,
    resampling=Resampling.bilinear,
)

# 5c. TCD — RGBA from WMS → greyscale with alpha mask
# Note: the WMS returns a color render (RGBA), not raw values.
# We convert RGB → greyscale and use the alpha channel as mask.
print("\n--- TCD → reprojected greyscale ---")
tcd_aligned_path = os.path.join(DATA_DIR, "aligned_tcd.tif")
reproject_rgba_wms(
    FILES["tcd"], tcd_aligned_path,
    resampling=Resampling.nearest,
)

# 5d. IMD — same approach
print("\n--- IMD → reprojected greyscale ---")
imd_aligned_path = os.path.join(DATA_DIR, "aligned_imd.tif")
reproject_rgba_wms(
    FILES["imd"], imd_aligned_path,
    resampling=Resampling.nearest,
)

# 5e. DTM Lidar — kept as-is (this is the target)
print("\n--- DTM Lidar HD — kept as-is (target) ---")
print(f"  → {FILES['dtm_lidar']} (unchanged)")


# =============================================================================
# 6. Visualizations — ALIGNED data
# =============================================================================
print("\n" + "=" * 60)
print("FIGURES — ALIGNED DATA")
print("=" * 60)

# Sentinel-2 RGB
print("\n--- Sentinel-2 aligned (RGB) ---")
plot_rgb(
    s2_aligned_path,
    "Sentinel-2 L2A — RGB (aligned, 10 m UTM)",
    os.path.join(ALIGNED_DIR, "sentinel2_rgb.png"),
    bands=(3, 2, 1), stretch=3.0,
)

# Sentinel-2 NDVI
print("\n--- Sentinel-2 aligned (NDVI) ---")
plot_raster(
    s2_aligned_path,
    "Sentinel-2 — NDVI (aligned, 10 m UTM)",
    os.path.join(ALIGNED_DIR, "sentinel2_ndvi.png"),
    band=5, cmap="RdYlGn", vmin=-0.2, vmax=0.9,
)

# COP-DEM
print("\n--- Copernicus DEM aligned ---")
plot_raster(
    dem_aligned_path,
    "Copernicus DEM 30 m → 10 m (aligned UTM)",
    os.path.join(ALIGNED_DIR, "copdem30.png"),
    cmap="terrain",
)

# TCD
print("\n--- TCD aligned ---")
plot_raster(
    tcd_aligned_path,
    "TCD — Tree Cover Density (aligned, 10 m UTM)",
    os.path.join(ALIGNED_DIR, "tcd.png"),
    band=1, cmap="Greens", vmin=0, vmax=255,
)

# IMD
print("\n--- IMD aligned ---")
plot_raster(
    imd_aligned_path,
    "IMD — Imperviousness (aligned, 10 m UTM)",
    os.path.join(ALIGNED_DIR, "imd.png"),
    band=1, cmap="Reds", vmin=0, vmax=255,
)

# DTM Lidar (original, not aligned)
print("\n--- DTM Lidar HD (target, not aligned) ---")
plot_raster(
    FILES["dtm_lidar"],
    "DTM Lidar HD — target (original, not reprojected)",
    os.path.join(ALIGNED_DIR, "dtm_lidar_target.png"),
    cmap="terrain", nodata_val=-99999.0,
)


# =============================================================================
# 7. Summary panel
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY PANEL")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Sentinel-2 RGB
with rasterio.open(s2_aligned_path) as src:
    rgb = np.stack([src.read(b).astype("float32") for b in [3, 2, 1]], axis=-1)
axes[0, 0].imshow(np.clip(rgb * 3, 0, 1))
axes[0, 0].set_title("Sentinel-2 RGB")

# NDVI
with rasterio.open(s2_aligned_path) as src:
    ndvi = src.read(5)
im1 = axes[0, 1].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.9)
axes[0, 1].set_title("NDVI")
plt.colorbar(im1, ax=axes[0, 1], shrink=0.7)

# COP-DEM
with rasterio.open(dem_aligned_path) as src:
    dem = src.read(1)
    dem[~np.isfinite(dem)] = np.nan
im2 = axes[0, 2].imshow(dem, cmap="terrain")
axes[0, 2].set_title("Copernicus DEM 30 m")
plt.colorbar(im2, ax=axes[0, 2], shrink=0.7, label="m")

# TCD
with rasterio.open(tcd_aligned_path) as src:
    tcd = src.read(1)
    tcd[~np.isfinite(tcd)] = np.nan
im3 = axes[1, 0].imshow(tcd, cmap="Greens", vmin=0, vmax=255)
axes[1, 0].set_title("TCD (tree cover)")
plt.colorbar(im3, ax=axes[1, 0], shrink=0.7)

# IMD
with rasterio.open(imd_aligned_path) as src:
    imd = src.read(1)
    imd[~np.isfinite(imd)] = np.nan
im4 = axes[1, 1].imshow(imd, cmap="Reds", vmin=0, vmax=255)
axes[1, 1].set_title("IMD (imperviousness)")
plt.colorbar(im4, ax=axes[1, 1], shrink=0.7)

# DTM Lidar
with rasterio.open(FILES["dtm_lidar"]) as src:
    dtm = src.read(1).astype("float32")
    dtm[dtm == -99999.0] = np.nan
im5 = axes[1, 2].imshow(dtm, cmap="terrain")
axes[1, 2].set_title("DTM Lidar HD (target)")
plt.colorbar(im5, ax=axes[1, 2], shrink=0.7, label="m")

for ax in axes.flat:
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

plt.suptitle(f"Data Overview — {ZONE.title()} — Inputs + Target", fontsize=16, y=1.01)
plt.tight_layout()
panel_path = os.path.join(OUTPUT_DIR, "data_overview.png")
plt.savefig(panel_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Panel: {panel_path}")


# =============================================================================
# 8. Comparison panel: all 6 layers side by side
# =============================================================================
print("\n" + "=" * 60)
print("COMPARISON PANEL — 6 ALIGNED LAYERS")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(20, 13))

# --- Row 1 ---

# 1) Sentinel-2 RGB
with rasterio.open(s2_aligned_path) as src:
    rgb = np.stack([src.read(b).astype("float32") for b in [3, 2, 1]], axis=-1)
axes[0, 0].imshow(np.clip(rgb * 3, 0, 1))
axes[0, 0].set_title("Sentinel-2 RGB", fontsize=13, fontweight="bold")

# 2) NDVI
with rasterio.open(s2_aligned_path) as src:
    ndvi = src.read(5)
im_ndvi = axes[0, 1].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.9)
axes[0, 1].set_title("NDVI (Sentinel-2)", fontsize=13, fontweight="bold")
plt.colorbar(im_ndvi, ax=axes[0, 1], shrink=0.7)

# 3) Copernicus DEM
with rasterio.open(dem_aligned_path) as src:
    dem = src.read(1)
    dem[~np.isfinite(dem)] = np.nan
im_dem = axes[0, 2].imshow(dem, cmap="terrain")
axes[0, 2].set_title("Copernicus DEM 30 m", fontsize=13, fontweight="bold")
plt.colorbar(im_dem, ax=axes[0, 2], shrink=0.7, label="m")

# --- Row 2 ---

# 4) TCD
with rasterio.open(tcd_aligned_path) as src:
    tcd = src.read(1)
    tcd[~np.isfinite(tcd)] = np.nan
im_tcd = axes[1, 0].imshow(tcd, cmap="Greens", vmin=0, vmax=255)
axes[1, 0].set_title("TCD (tree cover)", fontsize=13, fontweight="bold")
plt.colorbar(im_tcd, ax=axes[1, 0], shrink=0.7)

# 5) IMD
with rasterio.open(imd_aligned_path) as src:
    imd = src.read(1)
    imd[~np.isfinite(imd)] = np.nan
im_imd = axes[1, 1].imshow(imd, cmap="Reds", vmin=0, vmax=255)
axes[1, 1].set_title("IMD (imperviousness)", fontsize=13, fontweight="bold")
plt.colorbar(im_imd, ax=axes[1, 1], shrink=0.7)

# 6) DTM Lidar (target)
with rasterio.open(FILES["dtm_lidar"]) as src:
    dtm = src.read(1).astype("float32")
    dtm[dtm == -99999.0] = np.nan
im_dtm = axes[1, 2].imshow(dtm, cmap="terrain")
axes[1, 2].set_title("DTM Lidar HD (target)", fontsize=13, fontweight="bold")
plt.colorbar(im_dtm, ax=axes[1, 2], shrink=0.7, label="m")

for ax in axes.flat:
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

plt.suptitle(
    f"Overview — {ZONE.title()} — 4 inputs + NDVI + DTM target",
    fontsize=16, fontweight="bold", y=1.01,
)
plt.tight_layout()
comparison_path = os.path.join(OUTPUT_DIR, "comparison_6_layers.png")
plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Comparison figure: {comparison_path}")


# =============================================================================
# 9. Summary
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

aligned_files = {
    "Sentinel-2 (5 bands)": s2_aligned_path,
    "Copernicus DEM 30 m":  dem_aligned_path,
    "TCD (band 1)":         tcd_aligned_path,
    "IMD (band 1)":         imd_aligned_path,
}

print("\nAligned files (same grid: EPSG:32630, 10 m):")
for name, path in aligned_files.items():
    with rasterio.open(path) as src:
        size_kb = os.path.getsize(path) / 1024
        print(f"  {name:30s} → {src.width}x{src.height}, {src.count} band(s), {size_kb:.0f} KB")

print(f"\nTarget (not reprojected):")
with rasterio.open(FILES["dtm_lidar"]) as src:
    size_kb = os.path.getsize(FILES["dtm_lidar"]) / 1024
    print(f"  DTM Lidar HD               → {src.width}x{src.height}, {src.count} band(s), {size_kb:.0f} KB")

print(f"\nOriginal figures: {ORIGINAL_DIR}/")
for f in sorted(os.listdir(ORIGINAL_DIR)):
    print(f"  {f}")

print(f"\nAligned figures:  {ALIGNED_DIR}/")
for f in sorted(os.listdir(ALIGNED_DIR)):
    print(f"  {f}")

print(f"\nPanel:      {panel_path}")
print(f"Comparison: {comparison_path}")
print("\nDone!")
