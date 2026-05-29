"""
1_prepare.py
============
Reprojects and aligns all raw inputs to the Sentinel-2 10 m reference grid
for every zone, then generates diagnostic plots.

The Sentinel-2 raster defines the reference grid (UTM projected, ~10 m/px).
All other inputs are reprojected / resampled to match this grid exactly:

  sentinel2.tif  -- 5 bands (B02 B03 B04 B08 NDVI)  native 10 m  [reference]
  copdem30.tif   -- COP-DEM bilinearly upsampled from 30 m → 10 m
  slope.tif      -- slope in degrees derived from the 10 m COP-DEM
  tcd.tif        -- TCD [0–1]  (see format detection below)
  imd.tif        -- IMD [0–1]  (same logic)
  water_mask.tif -- COP-DEM WBM resampled with nearest neighbour (0=land, >0=water)
  dtm_lidar.tif  -- Lidar HD DTM resampled to 10 m  [TARGET]

  All float layers (copdem30, slope, tcd, imd, dtm_lidar, sentinel2) have
  their water pixels (water_mask > 0) set to NaN after alignment.
  A union NaN mask is then applied: any pixel that is NaN in any layer
  (inputs or target) is set to NaN in all layers.

TCD / IMD format detection (automatic, no --force needed)
----------------------------------------------------------
  The raw TCD/IMD file can be in one of two formats depending on whether
  the download succeeded via WCS (numeric) or fell back to WMS (RGBA):

  1-band numeric (WCS)   : raw values 0–100 (% cover)
                           → output = clip(value / 100, 0, 1)
  ≥3-band RGBA (WMS)     : colourised render — the WMS colour ramp encodes
                           low values as light/white and high values as dark.
                           Luminance L = 0.299·R + 0.587·G + 0.114·B (÷255)
                           → output = 1 − L  (invert so dark = high value)

  In both cases the output is a continuous float32 [0, 1] raster.
  The band-count check is done each run so re-preparing is always correct.

Slope correctness
-----------------
  Since Sentinel-2 UTM rasters have pixel sizes in metres, the slope
  computation receives  resolution_m = abs(transform.a) ≈ 10 m,  which
  gives accurate degree values via arctan(|∇z| / resolution).

Usage
-----
  python 1_prepare.py                         # prepare all zones
  python 1_prepare.py --zones caen grenoble   # specific zones only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


# ---------------------------------------------------------------------------
# 1.  Geometry helpers
# ---------------------------------------------------------------------------

def get_reference_grid(s2_path: Path) -> dict:
    """
    Extract CRS, transform, width, height from the Sentinel-2 raster.
    The S2 raster is stored in UTM (projected, metres), so the pixel size
    returned by abs(transform.a) is in real metres (~10 m).
    """
    with rasterio.open(s2_path) as src:
        return dict(
            crs       = src.crs,
            transform = src.transform,
            width     = src.width,
            height    = src.height,
        )


def _save_single_band(output_path: Path, arr: np.ndarray, grid: dict,
                      nodata: float = np.nan,
                      band_description: str = "") -> None:
    """Write a 2-D float32 array to a single-band GeoTIFF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver    = "GTiff",
        dtype     = "float32",
        count     = 1,
        crs       = grid["crs"],
        transform = grid["transform"],
        width     = grid["width"],
        height    = grid["height"],
        compress  = "deflate",
        nodata    = nodata,
    )
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)
        if band_description:
            dst.set_band_description(1, band_description)


def reproject_to_grid(input_path: Path, output_path: Path,
                      grid: dict, band: int = 1,
                      resampling: Resampling = Resampling.bilinear,
                      nodata: float = np.nan) -> None:
    """Reproject one band from input_path to the reference grid."""
    with rasterio.open(input_path) as src:
        dst_arr = np.full((grid["height"], grid["width"]), np.nan, dtype="float32")
        reproject(
            source        = rasterio.band(src, band),
            destination   = dst_arr,
            src_transform = src.transform,
            src_crs       = src.crs,
            dst_transform = grid["transform"],
            dst_crs       = grid["crs"],
            resampling    = resampling,
            src_nodata    = src.nodata,
            dst_nodata    = nodata,
        )
    _save_single_band(output_path, dst_arr, grid, nodata=nodata)


def reproject_multiband(input_path: Path, output_path: Path,
                        grid: dict,
                        resampling: Resampling = Resampling.bilinear) -> None:
    """Reproject all bands of a multi-band raster to the reference grid."""
    with rasterio.open(input_path) as src:
        n_bands = src.count
        dst_arr = np.zeros((n_bands, grid["height"], grid["width"]), dtype="float32")
        band_names = [src.descriptions[i] or f"band_{i+1}" for i in range(n_bands)]
        for b in range(1, n_bands + 1):
            reproject(
                source        = rasterio.band(src, b),
                destination   = dst_arr[b - 1],
                src_transform = src.transform,
                src_crs       = src.crs,
                dst_transform = grid["transform"],
                dst_crs       = grid["crs"],
                resampling    = resampling,
                src_nodata    = src.nodata,
                dst_nodata    = np.nan,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver    = "GTiff",
        dtype     = "float32",
        count     = n_bands,
        crs       = grid["crs"],
        transform = grid["transform"],
        width     = grid["width"],
        height    = grid["height"],
        compress  = "deflate",
    )
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(dst_arr)
        for i, name in enumerate(band_names, 1):
            dst.set_band_description(i, name)


def reproject_hrl(input_path: Path, output_path: Path, grid: dict) -> str:
    """
    Reproject a TCD or IMD raw file to the reference grid, auto-detecting format.

    Format detection is based on the band count of the raw file:

      1-band  → numeric WCS raster (values 0–100 % cover)
                output = clip(value / 100, 0, 1)

      ≥3-band → colourised WMS RGBA render (light=low, dark=high)
                output = 1 − luminance/255   (invert for correct ordering)

    Returns
    -------
    "wcs" or "wms" string indicating which path was used (for logging).
    """
    with rasterio.open(input_path) as src:
        n_bands = src.count

    if n_bands == 1:
        # ── WCS numeric path ─────────────────────────────────────────────
        arr = np.full((grid["height"], grid["width"]), np.nan, dtype="float32")
        with rasterio.open(input_path) as src:
            reproject(
                source        = rasterio.band(src, 1),
                destination   = arr,
                src_transform = src.transform,
                src_crs       = src.crs,
                dst_transform = grid["transform"],
                dst_crs       = grid["crs"],
                resampling    = Resampling.bilinear,
                src_nodata    = src.nodata,
                dst_nodata    = np.nan,
            )
        arr = np.clip(arr / 100.0, 0.0, 1.0)
        fmt = "wcs"

    else:
        # ── WMS RGBA path (inverted luminance) ───────────────────────────
        with rasterio.open(input_path) as src:
            rgba = np.zeros((src.count, grid["height"], grid["width"]), dtype="float32")
            for b in range(1, src.count + 1):
                reproject(
                    source        = rasterio.band(src, b),
                    destination   = rgba[b - 1],
                    src_transform = src.transform,
                    src_crs       = src.crs,
                    dst_transform = grid["transform"],
                    dst_crs       = grid["crs"],
                    resampling    = Resampling.bilinear,
                    src_nodata    = src.nodata,
                    dst_nodata    = 0.0,
                )
        r, g, b_ch = rgba[0], rgba[1], rgba[2]
        alpha = rgba[3] if rgba.shape[0] >= 4 else None
        # Luminance in [0, 1]; invert so dark (high value) → high output
        lum = (0.299 * r + 0.587 * g + 0.114 * b_ch) / 255.0
        arr = 1.0 - lum
        if alpha is not None:
            arr[alpha < 1] = np.nan
        fmt = "wms-inverted"

    _save_single_band(output_path, arr, grid, nodata=np.nan)
    return fmt


# ---------------------------------------------------------------------------
# 2.  Slope computation
# ---------------------------------------------------------------------------

def compute_slope(dem: np.ndarray, resolution_m: float = 10.0) -> np.ndarray:
    """
    Compute terrain slope in degrees from a 2-D DEM (float32, NaN = nodata).

    Uses numpy.gradient (central differences) with the cell size in metres.
    resolution_m should be abs(transform.a) of a UTM (projected) raster, which
    gives ~10 m for a Sentinel-2 grid.  Do NOT pass degree values here.
    """
    valid  = np.isfinite(dem)
    filled = dem.copy()
    if not valid.all():
        filled[~valid] = float(np.nanmean(filled))
    dy, dx = np.gradient(filled, resolution_m, resolution_m)
    slope  = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2))).astype("float32")
    slope[~valid] = np.nan
    return slope


# ---------------------------------------------------------------------------
# 3.  Visualisation helpers
# ---------------------------------------------------------------------------

def _plot_raster(ax: plt.Axes, data: np.ndarray, title: str,
                 cmap: str = "terrain", unit: str = "",
                 vmin: float | None = None, vmax: float | None = None) -> None:
    valid = data[np.isfinite(data)]
    if vmin is None:
        vmin = float(np.percentile(valid, 2))  if valid.size else 0
    if vmax is None:
        vmax = float(np.percentile(valid, 98)) if valid.size else 1
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=unit)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _plot_rgb(ax: plt.Axes, stack: np.ndarray, title: str,
              bands: tuple[int, int, int] = (2, 1, 0)) -> None:
    """Display a true-colour composite from a (C, H, W) Sentinel-2 stack."""
    rgb = stack[list(bands)].transpose(1, 2, 0)   # (H, W, 3)
    # S2 nodata = 0; mask pixels where any channel is 0
    valid = (rgb > 0).all(axis=2)
    out = np.full_like(rgb, 0.5)   # nodata pixels rendered as mid-grey
    for c in range(3):
        ch   = rgb[:, :, c]
        vals = ch[valid]
        if vals.size == 0:
            continue
        lo = float(np.percentile(vals, 2))
        hi = float(np.percentile(vals, 98))
        out[:, :, c] = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1)
    out[~valid] = 0.5
    ax.imshow(out, aspect="equal")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# ---------------------------------------------------------------------------
# 4.  Per-zone preparation pipeline
# ---------------------------------------------------------------------------

def prepare_zone(zone_id: str, data_root: Path, output_root: Path,
                 verbose: bool = True) -> bool:
    """
    Align all raw inputs for one zone to the Sentinel-2 10 m reference grid.

    Returns True on success, False if sentinel2.tif is missing.
    """
    raw_dir     = data_root  / "zones"   / zone_id
    aligned_dir = data_root  / "aligned" / zone_id
    fig_dir     = output_root / "zones"  / zone_id
    aligned_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Reference grid from Sentinel-2 ───────────────────────────────────
    s2_raw = raw_dir / "sentinel2.tif"
    if not s2_raw.exists():
        print(f"  ERROR: {s2_raw} not found -- run 0_download_data.py first")
        return False

    if verbose:
        print(f"\n[1] Reference grid from sentinel2.tif (Sentinel-2 UTM ~10 m)")
    grid = get_reference_grid(s2_raw)
    res  = abs(grid["transform"].a)   # pixel size in metres (UTM CRS)
    if verbose:
        print(f"  CRS: {grid['crs']}  size: {grid['width']}x{grid['height']}  "
              f"res: {res:.2f} m/px")

    # ── [2] Sentinel-2 (reference — reproject to canonical grid) ─────────
    s2_aligned = aligned_dir / "sentinel2.tif"
    if verbose:
        print("[2] Aligning Sentinel-2 (native 10 m, reference layer) ...")
    reproject_multiband(s2_raw, s2_aligned, grid)
    if verbose:
        print(f"  -> {s2_aligned.name}")

    # ── [3] COP-DEM (bilinear upsampling 30 m → 10 m) ────────────────────
    dem_raw     = raw_dir     / "copdem30.tif"
    dem_aligned = aligned_dir / "copdem30.tif"
    if dem_raw.exists():
        if verbose:
            print("[3] Reprojecting COP-DEM 30 m → 10 m (bilinear) ...")
        reproject_to_grid(dem_raw, dem_aligned, grid,
                          resampling=Resampling.bilinear)
        if verbose:
            with rasterio.open(dem_aligned) as src:
                v = src.read(1)
                v = v[np.isfinite(v)]
                print(f"  -> {dem_aligned.name}  "
                      f"range [{v.min():.1f}, {v.max():.1f}] m")
    else:
        print(f"  WARNING: {dem_raw} not found -- skipping COP-DEM")

    # ── [4] Slope (derived from 10 m COP-DEM) ────────────────────────────
    slope_aligned = aligned_dir / "slope.tif"
    if dem_aligned.exists():
        if verbose:
            print("[4] Computing slope from 10 m aligned COP-DEM ...")
        with rasterio.open(dem_aligned) as src:
            dem_arr = src.read(1).astype("float32")
            if src.nodata is not None:
                dem_arr[dem_arr == src.nodata] = np.nan
        slope_arr = compute_slope(dem_arr, resolution_m=res)  # res ≈ 10 m in metres
        _save_single_band(slope_aligned, slope_arr, grid,
                          band_description="slope_deg")
        if verbose:
            sv = slope_arr[np.isfinite(slope_arr)]
            print(f"  -> {slope_aligned.name}  "
                  f"range [{sv.min():.1f}, {sv.max():.1f}] deg  "
                  f"mean={sv.mean():.2f}")
    else:
        print("  WARNING: COP-DEM missing — skipping slope")

    # ── [5/6] TCD and IMD (auto-detect format, always continuous [0,1]) ──
    for step, layer_name in [("5", "tcd"), ("6", "imd")]:
        raw_path     = raw_dir     / f"{layer_name}.tif"
        aligned_path = aligned_dir / f"{layer_name}.tif"
        if not raw_path.exists():
            print(f"  WARNING: {raw_path} not found -- skipping {layer_name.upper()}")
            continue
        if verbose:
            with rasterio.open(raw_path) as src:
                fmt_str = ("1-band numeric WCS" if src.count == 1
                           else f"{src.count}-band RGBA WMS")
            print(f"[{step}] Aligning {layer_name.upper()} ({fmt_str}) ...")
        fmt = reproject_hrl(raw_path, aligned_path, grid)
        if verbose:
            with rasterio.open(aligned_path) as src:
                av = src.read(1)
                av = av[np.isfinite(av)]
                print(f"  -> {aligned_path.name}  [{av.min():.3f}, {av.max():.3f}]  "
                      f"mean={av.mean():.3f}  (format: {fmt})")

    # ── [7] DTM Lidar HD (TARGET) ─────────────────────────────────────────
    dtm_raw     = raw_dir     / "dtm_lidar.tif"
    dtm_aligned = aligned_dir / "dtm_lidar.tif"
    if dtm_raw.exists():
        if verbose:
            print("[7] Aligning Lidar HD DTM (target, → 10 m bilinear) ...")
        reproject_to_grid(dtm_raw, dtm_aligned, grid,
                          resampling=Resampling.bilinear, nodata=np.nan)
        if verbose:
            with rasterio.open(dtm_aligned) as src:
                v = src.read(1)
                v = v[np.isfinite(v)]
                print(f"  -> {dtm_aligned.name}  "
                      f"range [{v.min():.1f}, {v.max():.1f}] m")
    else:
        print(f"  WARNING: {dtm_raw} not found -- target missing")

    # ── [8] Water mask (COP-DEM WBM → 10 m nearest) ──────────────────────
    wbm_raw     = raw_dir     / "water_mask.tif"
    wbm_aligned = aligned_dir / "water_mask.tif"
    if wbm_raw.exists():
        if verbose:
            print("[8] Reprojecting water mask (WBM, nearest neighbour) ...")
        reproject_to_grid(wbm_raw, wbm_aligned, grid,
                          resampling=Resampling.nearest, nodata=np.nan)
        if verbose:
            with rasterio.open(wbm_aligned) as src:
                wm = src.read(1)
                n_water = int((wm > 0).sum())
                pct = 100.0 * n_water / max(wm.size, 1)
                print(f"  -> {wbm_aligned.name}  "
                      f"water pixels: {n_water} ({pct:.1f}%)")

        # Apply water mask: set water pixels to NaN in all float layers
        if verbose:
            print("[8] Applying water mask (NaN) to all aligned layers ...")
        with rasterio.open(wbm_aligned) as src:
            water_mask = (src.read(1) > 0)   # True where water

        n_masked = int(water_mask.sum())
        if n_masked > 0:
            # Single-band float layers
            single_band_layers = [
                "copdem30.tif", "slope.tif", "tcd.tif", "imd.tif", "dtm_lidar.tif",
            ]
            for fname in single_band_layers:
                p = aligned_dir / fname
                if not p.exists():
                    continue
                with rasterio.open(p) as src:
                    arr    = src.read(1).astype("float32")
                    prof   = src.profile.copy()
                    descr  = src.descriptions[0] or ""
                arr[water_mask] = np.nan
                with rasterio.open(p, "w", **prof) as dst:
                    dst.write(arr, 1)
                    if descr:
                        dst.set_band_description(1, descr)

            # Multi-band Sentinel-2
            s2_aligned = aligned_dir / "sentinel2.tif"
            if s2_aligned.exists():
                with rasterio.open(s2_aligned) as src:
                    stack  = src.read().astype("float32")
                    prof   = src.profile.copy()
                    descrs = [src.descriptions[i] or "" for i in range(src.count)]
                stack[:, water_mask] = np.nan
                with rasterio.open(s2_aligned, "w", **prof) as dst:
                    dst.write(stack)
                    for i, d in enumerate(descrs, 1):
                        if d:
                            dst.set_band_description(i, d)

            if verbose:
                print(f"  NaN applied to {n_masked} water pixels "
                      f"across all layers.")
    else:
        print(f"  WARNING: {wbm_raw} not found -- skipping water mask")

    # ── [8b] Union NaN mask across all layers ─────────────────────────────
    # Build the union of all NaN positions across every aligned layer so that
    # any pixel that is NaN in *any* input or in the target is NaN everywhere.
    if verbose:
        print("[8b] Building union NaN mask across all aligned layers ...")

    float_layers = [
        aligned_dir / "sentinel2.tif",   # multi-band (first band used for mask)
        aligned_dir / "copdem30.tif",
        aligned_dir / "slope.tif",
        aligned_dir / "tcd.tif",
        aligned_dir / "imd.tif",
        aligned_dir / "dtm_lidar.tif",
    ]
    nan_union: "np.ndarray | None" = None
    for p in float_layers:
        if not p.exists():
            continue
        with rasterio.open(p) as src:
            arr = src.read(1).astype("float32")
        nan_union = ~np.isfinite(arr) if nan_union is None else (nan_union | ~np.isfinite(arr))

    if nan_union is not None and nan_union.any():
        n_union = int(nan_union.sum())
        for p in float_layers:
            if not p.exists():
                continue
            with rasterio.open(p) as src:
                stack  = src.read().astype("float32")
                prof   = src.profile.copy()
                descrs = [src.descriptions[i] or "" for i in range(src.count)]
            stack[:, nan_union] = np.nan
            with rasterio.open(p, "w", **prof) as dst:
                dst.write(stack)
                for i, d in enumerate(descrs, 1):
                    if d:
                        dst.set_band_description(i, d)
        if verbose:
            print(f"  Union NaN mask: {n_union} pixels "
                  f"({100.0 * n_union / nan_union.size:.1f}%) masked in all layers.")
    elif verbose:
        print("  No additional pixels to union-mask.")

    # ── [9] Residual diagnostic ───────────────────────────────────────────
    if dem_aligned.exists() and dtm_aligned.exists():
        if verbose:
            with rasterio.open(dtm_aligned) as s1, rasterio.open(dem_aligned) as s2:
                dtm = s1.read(1).astype("float32")
                dem = s2.read(1).astype("float32")
                if s1.nodata: dtm[dtm == s1.nodata] = np.nan
                if s2.nodata: dem[dem == s2.nodata] = np.nan
            res_arr = dtm - dem
            vr = res_arr[np.isfinite(res_arr)]
            print(f"  Residual (DTM−COP-DEM):  "
                  f"mean={vr.mean():.2f} m  std={vr.std():.2f} m  "
                  f"p5={np.percentile(vr,5):.1f}  p95={np.percentile(vr,95):.1f}")

    # ── [9] Diagnostic figures ────────────────────────────────────────────
    if verbose:
        print("[9] Generating diagnostic plots ...")
    _make_plots(zone_id, aligned_dir, fig_dir)
    if verbose:
        print(f"  Figures saved to {fig_dir}/")

    return True


def _make_plots(zone_id: str, aligned_dir: Path, fig_dir: Path) -> None:
    """Generate overview + residual diagnostic figures for one zone."""
    # Compute shared elevation scale so COP-DEM and DTM use the same colorbar
    elev_vmin: float | None = None
    elev_vmax: float | None = None
    for fname in ("copdem30.tif", "dtm_lidar.tif"):
        p = aligned_dir / fname
        if p.exists():
            with rasterio.open(p) as src:
                arr = src.read(1).astype("float32")
                if src.nodata is not None:
                    arr[arr == src.nodata] = np.nan
            v = arr[np.isfinite(arr)]
            if v.size:
                lo = float(np.percentile(v, 2))
                hi = float(np.percentile(v, 98))
                elev_vmin = lo if elev_vmin is None else min(elev_vmin, lo)
                elev_vmax = hi if elev_vmax is None else max(elev_vmax, hi)

    panels = [
        ("sentinel2.tif",  "S2 true colour (B04/B03/B02)",     "viridis", "",      None,       None      ),
        ("copdem30.tif",   "COP-DEM → 10 m (m)",               "terrain", "m",     elev_vmin,  elev_vmax ),
        ("slope.tif",      "Slope from COP-DEM (°)",            "hot_r",   "deg",   None,       None      ),
        ("tcd.tif",        "TCD (tree cover density, 0–1)",     "Greens",  "0–1",   0.0,        1.0       ),
        ("imd.tif",        "IMD (imperviousness, 0–1)",         "Reds",    "0–1",   0.0,        1.0       ),
        ("water_mask.tif", "Water mask (0=land, 1-3=water)",   "Blues",   "class", 0.0,        3.0       ),
        ("dtm_lidar.tif",  "DTM Lidar HD 10 m [TARGET] (m)",   "terrain", "m",     elev_vmin,  elev_vmax ),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(f"Zone: {zone_id}  — aligned to Sentinel-2 10 m UTM grid",
                 fontsize=12)

    all_axes = axes.ravel()
    for ax, (fname, title, cmap, unit, vmin, vmax) in zip(all_axes, panels):
        p = aligned_dir / fname
        if not p.exists():
            ax.set_title(f"{title}\n(missing)", fontsize=9)
            ax.axis("off")
            continue
        with rasterio.open(p) as src:
            data = src.read(1).astype("float32")
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
        if fname == "sentinel2.tif":
            with rasterio.open(p) as src:
                stack = src.read().astype("float32")
            _plot_rgb(ax, stack, title, bands=(2, 1, 0))
            # Placeholder colorbar so this panel has the same geometry as the others
            sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_visible(False)
        else:
            _plot_raster(ax, data, title, cmap=cmap, unit=unit, vmin=vmin, vmax=vmax)

    # Hide the last empty subplot (7 panels in a 2×4 grid)
    for ax in all_axes[len(panels):]:
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(fig_dir / "overview_aligned.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Residual: DTM_lidar − COP-DEM
    dtm_p = aligned_dir / "dtm_lidar.tif"
    dem_p = aligned_dir / "copdem30.tif"
    if dtm_p.exists() and dem_p.exists():
        with rasterio.open(dtm_p) as s1, rasterio.open(dem_p) as s2:
            dtm = s1.read(1).astype("float32")
            dem = s2.read(1).astype("float32")
            if s1.nodata: dtm[dtm == s1.nodata] = np.nan
            if s2.nodata: dem[dem == s2.nodata] = np.nan
        res_arr = dtm - dem
        vr = res_arr[np.isfinite(res_arr)]

        fig2, ax2 = plt.subplots(figsize=(7, 6))
        lim = float(np.percentile(np.abs(vr), 98)) if vr.size else 10
        im = ax2.imshow(res_arr, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="equal")
        plt.colorbar(im, ax=ax2, label="m")
        ax2.set_title(
            f"{zone_id}: DTM_lidar − COP-DEM (m)\n"
            f"mean={vr.mean():.2f}  std={vr.std():.2f}  "
            f"p5={np.percentile(vr,5):.1f}  p95={np.percentile(vr,95):.1f}",
            fontsize=9
        )
        ax2.axis("off")
        fig2.tight_layout()
        fig2.savefig(fig_dir / "residual_dtm_copedem.png", dpi=150,
                     bbox_inches="tight")
        plt.close(fig2)


# ---------------------------------------------------------------------------
# 5.  Completeness check
# ---------------------------------------------------------------------------

REQUIRED_RAW_FILES = [
    "sentinel2.tif",
    "copdem30.tif",
    "tcd.tif",
    "imd.tif",
    "dtm_lidar.tif",
]


def check_zone_completeness(zone_id: str, data_root: Path) -> list[str]:
    """
    Return a list of required raw files that are missing for the given zone.
    An empty list means the zone is complete and ready to prepare.
    """
    raw_dir = data_root / "zones" / zone_id
    return [
        fname for fname in REQUIRED_RAW_FILES
        if not (raw_dir / fname).exists()
    ]


# ---------------------------------------------------------------------------
# 6.  Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Align and prepare DSM/DTM zone data for model training."
    )
    parser.add_argument(
        "--zones", nargs="*", metavar="ZONE_ID",
        help="Zone IDs to prepare (default: all zones in zones.py).",
    )
    parser.add_argument(
        "--include-incomplete", action="store_true",
        help="Process zones even if some raw files are missing (default: skip them).",
    )
    return parser.parse_args()


def main():
    from zones import ZONES, ZONES_BY_ID

    args = parse_args()
    if args.zones:
        unknown = set(args.zones) - set(ZONES_BY_ID.keys())
        if unknown:
            print(f"ERROR: unknown zone ID(s): {sorted(unknown)}")
            sys.exit(1)
        selected = [ZONES_BY_ID[zid] for zid in args.zones]
    else:
        selected = ZONES

    data_root   = Path(__file__).resolve().parent.parent / "data"
    output_root = Path(__file__).resolve().parent.parent / "output"

    print(f"\nPreparing {len(selected)} zone(s)")
    print(f"  Raw data   : {data_root}/zones/")
    print(f"  Aligned    : {data_root}/aligned/")
    print(f"  Figures    : {output_root}/zones/")

    # ── Completeness pre-check ────────────────────────────────────────────
    if not args.include_incomplete:
        complete, skipped = [], []
        for zone in selected:
            missing = check_zone_completeness(zone["id"], data_root)
            if missing:
                skipped.append((zone, missing))
            else:
                complete.append(zone)

        if skipped:
            print(f"\nSkipping {len(skipped)} incomplete zone(s) "
                  f"(use --include-incomplete to force):")
            for zone, missing in skipped:
                print(f"  [{zone['id']}]  missing: {', '.join(missing)}")

        if not complete:
            print("\nNo complete zones to process. "
                  "Run 0_download_data.py first.")
            sys.exit(0)

        selected = complete
        print(f"\nPreparing {len(selected)} complete zone(s)")
    else:
        print(f"\nPreparing {len(selected)} zone(s) "
              "(--include-incomplete: skipping completeness check)")

    ok_count = 0
    for zone in selected:
        print(f"\n{'=' * 64}")
        print(f"  ZONE: {zone['name']}  [{zone['id']}]")
        print(f"{'=' * 64}")
        if prepare_zone(zone["id"], data_root, output_root, verbose=True):
            ok_count += 1

    print(f"\nDone: {ok_count}/{len(selected)} zones prepared successfully.")
    if ok_count < len(selected):
        print("  Run 0_download_data.py first for missing zones.")


if __name__ == "__main__":
    main()
