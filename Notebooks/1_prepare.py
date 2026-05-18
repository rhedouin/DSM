"""
1_prepare.py
============
Reprojects and aligns all raw inputs to the Sentinel-2 reference grid for
every zone, then generates diagnostic plots.

For each zone the following aligned files are saved under
  data/aligned/{zone_id}/:
  sentinel2.tif   -- 5 bands (B02, B03, B04, B08, NDVI) -- reference grid
  copdem30.tif    -- COP-DEM 30 m bilinearly resampled to 10 m
  slope.tif       -- terrain slope (degrees) derived from aligned COP-DEM
  tcd.tif         -- TCD greyscale proxy (0-255, derived from RGBA WMS render)
  imd.tif         -- IMD greyscale proxy (0-255, derived from RGBA WMS render)
  dtm_lidar.tif   -- Lidar HD DTM bilinearly resampled to 10 m  [TARGET]

All outputs share the exact same CRS, transform, width, and height as the
corresponding zone's Sentinel-2 image (auto-detected from data/zones/{id}/).

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
from rasterio.warp import calculate_default_transform, reproject


# ---------------------------------------------------------------------------
# 1.  Geometry helpers
# ---------------------------------------------------------------------------

def get_reference_grid(s2_path: Path) -> dict:
    """
    Extract CRS, transform, width, height from the Sentinel-2 file.
    All other layers will be reprojected to match this grid.
    """
    with rasterio.open(s2_path) as src:
        return dict(
            crs       = src.crs,
            transform = src.transform,
            width     = src.width,
            height    = src.height,
        )


def reproject_to_grid(input_path: Path, output_path: Path,
                      grid: dict, band: int = 1,
                      resampling: Resampling = Resampling.bilinear,
                      nodata: float = np.nan) -> None:
    """
    Reproject a single band from input_path to match a reference grid.

    Parameters
    ----------
    input_path : source raster (any CRS / resolution)
    output_path: destination GeoTIFF
    grid       : dict with keys crs, transform, width, height
    band       : band index to read (1-based)
    resampling : rasterio Resampling method
    nodata     : nodata value for the output
    """
    with rasterio.open(input_path) as src:
        dst_arr = np.full((grid["height"], grid["width"]), np.nan, dtype="float32")
        reproject(
            source      = rasterio.band(src, band),
            destination = dst_arr,
            src_transform = src.transform,
            src_crs       = src.crs,
            dst_transform = grid["transform"],
            dst_crs       = grid["crs"],
            resampling    = resampling,
            src_nodata    = src.nodata,
            dst_nodata    = nodata,
        )
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
        dst.write(dst_arr, 1)


def reproject_rgba_wms(input_path: Path, output_path: Path,
                       grid: dict) -> None:
    """
    Reproject a 4-band RGBA WMS render to the reference grid.

    Converts the colourised RGBA image to a greyscale luminance proxy:
        L = 0.299*R + 0.587*G + 0.114*B,  scaled to [0, 1]
    Pixels where alpha=0 (no data) are set to NaN.

    Note: This is an approximation of the underlying TCD/IMD values since
    the WMS only exposes a visualisation, not raw numeric rasters.  The
    luminance preserves the relative ordering of values in the colour ramp.
    """
    with rasterio.open(input_path) as src:
        if src.count < 3:
            # Fallback: treat as single-band
            reproject_to_grid(input_path, output_path, grid)
            return

        # Read RGBA into destination CRS directly
        rgba_dst = np.zeros((src.count, grid["height"], grid["width"]), dtype="float32")
        for b in range(1, src.count + 1):
            reproject(
                source        = rasterio.band(src, b),
                destination   = rgba_dst[b - 1],
                src_transform = src.transform,
                src_crs       = src.crs,
                dst_transform = grid["transform"],
                dst_crs       = grid["crs"],
                resampling    = Resampling.bilinear,
                src_nodata    = src.nodata,
                dst_nodata    = 0.0,
            )

    r, g, b_ch = rgba_dst[0], rgba_dst[1], rgba_dst[2]
    alpha = rgba_dst[3] if rgba_dst.shape[0] >= 4 else None

    grey = (0.299 * r + 0.587 * g + 0.114 * b_ch) / 255.0   # [0, 1]
    if alpha is not None:
        grey[alpha < 1] = np.nan    # transparent = no data

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
        nodata    = np.nan,
    )
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(grey, 1)


def reproject_multiband(input_path: Path, output_path: Path,
                        grid: dict,
                        resampling: Resampling = Resampling.bilinear) -> None:
    """
    Reproject all bands of a multi-band raster to the reference grid.
    Used for Sentinel-2 (5 bands).
    """
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


# ---------------------------------------------------------------------------
# 2.  Slope computation
# ---------------------------------------------------------------------------

def compute_slope(dem: np.ndarray, resolution_m: float = 10.0) -> np.ndarray:
    """
    Compute terrain slope in degrees from a 2-D DEM (float32, NaN = nodata).

    Uses numpy gradient (central differences) with the cell size in metres.
    NaN cells in the input yield NaN in the output.
    """
    valid  = np.isfinite(dem)
    filled = dem.copy()
    if not valid.all():
        # Fill NaN with global mean for gradient computation (border artefacts
        # are masked back to NaN afterwards)
        filled[~valid] = float(np.nanmean(filled))

    dy, dx = np.gradient(filled, resolution_m, resolution_m)
    slope  = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))
    slope  = slope.astype("float32")
    slope[~valid] = np.nan      # restore nodata
    return slope


# ---------------------------------------------------------------------------
# 3.  Visualisation helpers
# ---------------------------------------------------------------------------

def _plot_raster(ax: plt.Axes, data: np.ndarray, title: str,
                 cmap: str = "terrain", unit: str = "") -> None:
    valid = data[np.isfinite(data)]
    vmin  = float(np.percentile(valid, 2))  if valid.size else 0
    vmax  = float(np.percentile(valid, 98)) if valid.size else 1
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=unit)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _plot_rgb(ax: plt.Axes, stack: np.ndarray, title: str,
              bands: tuple[int, int, int] = (2, 1, 0)) -> None:
    """Display a false/true colour composite from a (C, H, W) stack."""
    rgb = stack[list(bands)].transpose(1, 2, 0)
    # Robust stretch per channel
    out = np.zeros_like(rgb)
    for c in range(3):
        ch   = rgb[:, :, c]
        lo   = float(np.nanpercentile(ch, 2))
        hi   = float(np.nanpercentile(ch, 98))
        out[:, :, c] = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1)
    ax.imshow(out, aspect="equal")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# ---------------------------------------------------------------------------
# 4.  Per-zone preparation pipeline
# ---------------------------------------------------------------------------

def prepare_zone(zone_id: str, data_root: Path, output_root: Path,
                 verbose: bool = True) -> bool:
    """
    Align all raw inputs for one zone to the Sentinel-2 reference grid.

    Returns True on success, False if mandatory files are missing.
    """
    raw_dir     = data_root  / "zones"   / zone_id
    aligned_dir = data_root  / "aligned" / zone_id
    fig_dir     = output_root / "zones"  / zone_id
    aligned_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    s2_raw = raw_dir / "sentinel2.tif"
    if not s2_raw.exists():
        print(f"  ERROR: {s2_raw} not found -- run 0_download_data.py first")
        return False

    if verbose:
        print(f"\n[1] Reference grid from {s2_raw.name}")
    grid = get_reference_grid(s2_raw)
    res  = abs(grid["transform"].a)   # pixel size in metres (CRS units)
    if verbose:
        print(f"  CRS: {grid['crs']}  size: {grid['width']}x{grid['height']}  "
              f"res: {res:.1f} m/px")

    # ------------------------------------------------------------------ #
    # Sentinel-2 (reproject all 5 bands in one pass)                     #
    # ------------------------------------------------------------------ #
    s2_aligned = aligned_dir / "sentinel2.tif"
    if verbose:
        print(f"[2] Aligning Sentinel-2 ...")
    reproject_multiband(s2_raw, s2_aligned, grid)
    if verbose:
        print(f"  -> {s2_aligned.name}")

    # ------------------------------------------------------------------ #
    # COP-DEM 30 m                                                        #
    # ------------------------------------------------------------------ #
    dem_raw     = raw_dir     / "copdem30.tif"
    dem_aligned = aligned_dir / "copdem30.tif"
    if dem_raw.exists():
        if verbose:
            print("[3] Aligning COP-DEM ...")
        reproject_to_grid(dem_raw, dem_aligned, grid, resampling=Resampling.bilinear)
        if verbose:
            print(f"  -> {dem_aligned.name}")
    else:
        print(f"  WARNING: {dem_raw} not found -- skipping COP-DEM")

    # ------------------------------------------------------------------ #
    # Slope (derived from aligned COP-DEM)                               #
    # ------------------------------------------------------------------ #
    slope_aligned = aligned_dir / "slope.tif"
    if dem_aligned.exists():
        if verbose:
            print("[4] Computing slope ...")
        with rasterio.open(dem_aligned) as src:
            dem_arr = src.read(1).astype("float32")
            dem_arr[dem_arr == src.nodata] = np.nan
        slope_arr = compute_slope(dem_arr, resolution_m=res)
        profile   = dict(
            driver="GTiff", dtype="float32", count=1,
            crs=grid["crs"], transform=grid["transform"],
            width=grid["width"], height=grid["height"],
            compress="deflate", nodata=np.nan,
        )
        with rasterio.open(slope_aligned, "w", **profile) as dst:
            dst.write(slope_arr, 1)
            dst.set_band_description(1, "slope_deg")
        if verbose:
            valid_slope = slope_arr[np.isfinite(slope_arr)]
            print(f"  -> {slope_aligned.name}  "
                  f"range [{valid_slope.min():.1f}, {valid_slope.max():.1f}] deg")

    # ------------------------------------------------------------------ #
    # TCD and IMD  (RGBA WMS -> greyscale luminance)                     #
    # ------------------------------------------------------------------ #
    for layer_name in ("tcd", "imd"):
        raw_path     = raw_dir     / f"{layer_name}.tif"
        aligned_path = aligned_dir / f"{layer_name}.tif"
        if raw_path.exists():
            if verbose:
                print(f"[{'5' if layer_name=='tcd' else '6'}] Aligning {layer_name.upper()} ...")
            reproject_rgba_wms(raw_path, aligned_path, grid)
            if verbose:
                print(f"  -> {aligned_path.name}")
        else:
            print(f"  WARNING: {raw_path} not found -- skipping {layer_name.upper()}")

    # ------------------------------------------------------------------ #
    # DTM Lidar HD (TARGET)                                               #
    # ------------------------------------------------------------------ #
    dtm_raw     = raw_dir     / "dtm_lidar.tif"
    dtm_aligned = aligned_dir / "dtm_lidar.tif"
    if dtm_raw.exists():
        if verbose:
            print("[7] Aligning Lidar HD DTM (target) ...")
        with rasterio.open(dtm_raw) as src:
            n_bands = src.count
        # Single or multi-band: use band 1
        reproject_to_grid(dtm_raw, dtm_aligned, grid,
                          resampling=Resampling.bilinear, nodata=np.nan)
        if verbose:
            with rasterio.open(dtm_aligned) as src:
                arr = src.read(1)
                valid = arr[np.isfinite(arr)]
                print(f"  -> {dtm_aligned.name}  "
                      f"range [{valid.min():.1f}, {valid.max():.1f}] m")
    else:
        print(f"  WARNING: {dtm_raw} not found -- target missing")

    # ------------------------------------------------------------------ #
    # Diagnostic figures                                                  #
    # ------------------------------------------------------------------ #
    if verbose:
        print("[8] Generating plots ...")
    _make_plots(zone_id, aligned_dir, fig_dir)
    if verbose:
        print(f"  Figures saved to {fig_dir}/")

    return True


def _make_plots(zone_id: str, aligned_dir: Path, fig_dir: Path) -> None:
    """Generate a 2-row overview figure for one zone."""
    layers = [
        ("sentinel2.tif",  None,           "S2 RGB (B04/B03/B02)",  "viridis", ""),
        ("copdem30.tif",   None,           "COP-DEM 30m (m)",       "terrain", "m"),
        ("slope.tif",      None,           "Slope (deg)",            "hot_r",   "deg"),
        ("tcd.tif",        None,           "TCD proxy",              "Greens",  "luminance"),
        ("imd.tif",        None,           "IMD proxy",              "Reds",    "luminance"),
        ("dtm_lidar.tif",  None,           "DTM Lidar HD (m)",       "terrain", "m"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Zone: {zone_id}  — aligned to Sentinel-2 grid", fontsize=12)
    axes_flat = axes.ravel()

    for ax, (fname, _, title, cmap, unit) in zip(axes_flat, layers):
        p = aligned_dir / fname
        if not p.exists():
            ax.set_title(f"{title}\n(missing)", fontsize=9)
            ax.axis("off")
            continue
        with rasterio.open(p) as src:
            data = src.read(1).astype("float32")
            # Mark nodata
            if src.nodata is not None:
                data[data == src.nodata] = np.nan

        if fname == "sentinel2.tif":
            with rasterio.open(aligned_dir / fname) as src:
                stack = src.read().astype("float32")
            # Band order: B02=0, B03=1, B04=2, B08=3, NDVI=4
            _plot_rgb(ax, stack, title, bands=(2, 1, 0))   # RGB true colour
        else:
            _plot_raster(ax, data, title, cmap=cmap, unit=unit)

    plt.tight_layout()
    out_path = fig_dir / "overview_aligned.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Residual: DTM_lidar - COP-DEM
    dtm_p  = aligned_dir / "dtm_lidar.tif"
    dem_p  = aligned_dir / "copdem30.tif"
    if dtm_p.exists() and dem_p.exists():
        with rasterio.open(dtm_p) as s1, rasterio.open(dem_p) as s2:
            dtm = s1.read(1).astype("float32")
            dem = s2.read(1).astype("float32")
        if s1.nodata is not None:
            dtm[dtm == s1.nodata] = np.nan
        if s2.nodata is not None:
            dem[dem == s2.nodata] = np.nan
        residual = dtm - dem
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        _plot_raster(ax2, residual,
                     f"{zone_id}: DTM_lidar - COP-DEM 30m (m)",
                     cmap="RdBu_r", unit="m")
        fig2.savefig(fig_dir / "residual_dtm_vs_copedem.png", dpi=150,
                     bbox_inches="tight")
        plt.close(fig2)


# ---------------------------------------------------------------------------
# 5.  Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Align and prepare DSM/DTM zone data for model training."
    )
    parser.add_argument(
        "--zones", nargs="*", metavar="ZONE_ID",
        help="Zone IDs to prepare (default: all).",
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

    ok_count = 0
    for zone in selected:
        zone_id = zone["id"]
        print(f"\n{'=' * 64}")
        print(f"  ZONE: {zone['name']}  [{zone_id}]")
        print(f"{'=' * 64}")
        success = prepare_zone(zone_id, data_root, output_root, verbose=True)
        if success:
            ok_count += 1

    print(f"\nDone: {ok_count}/{len(selected)} zones prepared successfully.")
    if ok_count < len(selected):
        print("  Run 0_download_data.py first for missing zones.")


if __name__ == "__main__":
    main()
