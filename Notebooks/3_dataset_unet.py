"""
3_dataset_unet.py
=================
PyTorch Dataset and normalization utilities for the U-Net DTM model.

This module is imported by 4_train_unet.py via the dataset_unet.py shim.

Target formulation
------------------
The model predicts the **residual**  y = DTM_lidar − COP-DEM_10m  rather than
absolute elevation.  This has several advantages:

  • Near-zero mean across all zones (mean ≈ −2 m, std ≈ 3–7 m) — no
    zone-level elevation bias poisoning the loss.
  • Range ~±30 m instead of ±1000 m → loss converges much faster.
  • Trivial baseline (predict 0 everywhere) equals just using COP-DEM,
    so any positive R² means the model improves on COP-DEM.

At inference, the absolute DTM is recovered as:
    DTM_pred = COP-DEM_input + predicted_residual

Both the input channels and the residual target are normalised to zero mean /
unit standard deviation before being fed to the network.  The residual stats
(res_mean, res_std) are stored alongside the input channel stats in
data/norm_stats.json.

Input channel specification (9 channels)
-----------------------------------------
  0  cop_dem  -- COP-DEM bilinearly upsampled to 10 m (m)
  1  slope    -- slope in degrees derived from the 10 m COP-DEM
  2  B02      -- Sentinel-2 blue reflectance
  3  B03      -- Sentinel-2 green reflectance
  4  B04      -- Sentinel-2 red reflectance
  5  B08      -- Sentinel-2 NIR reflectance
  6  ndvi     -- NDVI = (NIR − Red) / (NIR + Red)
  7  tcd      -- TCD (tree cover density) continuous [0, 1]
  8  imd      -- IMD (imperviousness) continuous [0, 1]

Usage
-----
  from zones import TRAIN_ZONES
  from dataset_unet import DSMPatchDataset, compute_normalization_stats

  stats    = compute_normalization_stats(TRAIN_ZONES, data_root)
  train_ds = DSMPatchDataset(TRAIN_ZONES, data_root, stats,
                             patch_size=256, stride=128, augment=True)
  x, y = train_ds[0]
  # x: (9, 256, 256) normalised inputs
  # y: (1, 256, 256) normalised residual (DTM - COP_DEM)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import NamedTuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Input channel specification
# ---------------------------------------------------------------------------

# (aligned_filename, band_index_1based, channel_name)
CHANNEL_SPEC: list[tuple[str, int, str]] = [
    ("copdem30.tif",  1, "cop_dem"),
    ("slope.tif",     1, "slope"),
    ("sentinel2.tif", 1, "B02"),
    ("sentinel2.tif", 2, "B03"),
    ("sentinel2.tif", 3, "B04"),
    ("sentinel2.tif", 4, "B08"),
    ("sentinel2.tif", 5, "ndvi"),
    ("tcd.tif",       1, "tcd"),
    ("imd.tif",       1, "imd"),
]
CHANNEL_NAMES = [c[2] for c in CHANNEL_SPEC]
N_CHANNELS    = len(CHANNEL_SPEC)  # 9

# Maximum plausible magnitude for the DTM_lidar − COP-DEM residual.
# Values beyond this almost always indicate nodata fill values (e.g. −99999)
# or systematic CRS/alignment artefacts, NOT real terrain differences.
RESIDUAL_MAX_M: float = 200.0


# ---------------------------------------------------------------------------
# Zone data loader
# ---------------------------------------------------------------------------

class ZoneData(NamedTuple):
    """Holds in-memory arrays for one aligned zone."""
    zone_id   : str
    inputs    : np.ndarray    # (C, H, W) float32  raw (un-normalised)
    target    : np.ndarray    # (H, W)    float32  absolute DTM in metres
    residual  : np.ndarray    # (H, W)    float32  DTM_lidar − COP-DEM_10m (metres)
    valid_mask: np.ndarray    # (H, W)    bool     True where all channels + residual finite


def load_zone(zone_id: str, data_root: Path) -> "ZoneData | None":
    """
    Load all aligned channels and compute residual for one zone.

    Returns None if dtm_lidar.tif is missing.
    The residual = dtm_lidar − copdem30 is computed here so the Dataset
    can return it directly without re-reading files per patch.
    """
    aligned_dir = data_root / "aligned" / zone_id

    dtm_path = aligned_dir / "dtm_lidar.tif"
    if not dtm_path.exists():
        print(f"  WARNING [{zone_id}]: dtm_lidar.tif not found -- zone skipped")
        return None

    with rasterio.open(dtm_path) as src:
        target = src.read(1).astype("float32")
        nd = src.nodata
        if nd is not None and not (isinstance(nd, float) and np.isnan(nd)):
            target[target == nd] = np.nan
    # Mask implausible elevations (fill values such as −99999, +99999)
    target[(target < -500.0) | (target > 9000.0)] = np.nan

    H, W = target.shape
    inputs = np.zeros((N_CHANNELS, H, W), dtype="float32")

    open_files: dict[str, rasterio.DatasetReader] = {}
    try:
        for ci, (fname, band_idx, cname) in enumerate(CHANNEL_SPEC):
            fpath = aligned_dir / fname
            key   = str(fpath)
            if key not in open_files:
                if not fpath.exists():
                    print(f"  WARNING [{zone_id}]: {fname} not found -- "
                          f"channel {cname!r} set to 0")
                    inputs[ci] = 0.0
                    continue
                open_files[key] = rasterio.open(fpath)
            src = open_files[key]
            arr = src.read(band_idx).astype("float32")
            nd = src.nodata
            if nd is not None and not (isinstance(nd, float) and np.isnan(nd)):
                arr[arr == nd] = np.nan
            # For COP-DEM (channel 0) mask implausible elevations too
            if cname == "cop_dem":
                arr[(arr < -500.0) | (arr > 9000.0)] = np.nan
            if arr.shape != (H, W):
                print(f"  WARNING [{zone_id}]: {fname} band {band_idx} shape "
                      f"{arr.shape} != target {(H, W)} -- filling NaN")
                arr = np.full((H, W), np.nan, dtype="float32")
            inputs[ci] = arr
    finally:
        for f in open_files.values():
            f.close()

    # Residual: DTM_lidar − COP-DEM_10m (channel 0 = cop_dem)
    residual = target - inputs[0]   # (H, W), NaN where either is NaN

    # Mask residuals outside the physically plausible range.
    # This catches fill-value artefacts (e.g. DTM=-99999) and systematic
    # CRS/alignment issues that produce hundreds-of-metres residuals.
    residual[np.abs(residual) > RESIDUAL_MAX_M] = np.nan

    # Valid mask: finite in all channels AND in the (clipped) residual
    valid_mask = np.isfinite(residual)
    for ci in range(N_CHANNELS):
        valid_mask &= np.isfinite(inputs[ci])

    return ZoneData(
        zone_id    = zone_id,
        inputs     = inputs,
        target     = target,
        residual   = residual,
        valid_mask = valid_mask,
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

NormStats = dict  # {"channel_names", "means", "stds", "res_mean", "res_std"}


def compute_normalization_stats(zone_ids: list[str], data_root: Path,
                                max_pixels_per_zone: int = 200_000,
                                seed: int = 42) -> NormStats:
    """
    Compute per-channel (mean, std) for inputs AND (mean, std) for the
    residual from the specified training zones.  Results are saved to
    data/norm_stats.json.

    Parameters
    ----------
    zone_ids            : list of training zone IDs
    data_root           : path to data/ directory
    max_pixels_per_zone : max valid pixels sampled per zone
    seed                : RNG seed for reproducibility

    Returns
    -------
    NormStats dict with keys:
      "channel_names", "means", "stds"   -- per-channel input stats
      "res_mean", "res_std"              -- residual stats (metres)
    """
    rng = np.random.default_rng(seed)
    input_samples: list[np.ndarray] = []   # list of (N, C) arrays
    res_samples:   list[np.ndarray] = []   # list of (N,) arrays

    for zone_id in zone_ids:
        zd = load_zone(zone_id, data_root)
        if zd is None:
            continue

        # Sample from valid pixels only
        flat_valid = np.where(zd.valid_mask.ravel())[0]
        if len(flat_valid) == 0:
            print(f"  [{zone_id}] WARNING: no valid pixels — skipping")
            continue
        if len(flat_valid) > max_pixels_per_zone:
            flat_valid = rng.choice(flat_valid, size=max_pixels_per_zone, replace=False)

        sample_inputs = zd.inputs.reshape(N_CHANNELS, -1)[:, flat_valid].T  # (N, C)
        sample_res    = zd.residual.ravel()[flat_valid]                       # (N,)
        input_samples.append(sample_inputs)
        res_samples.append(sample_res)
        print(f"  [{zone_id}] {len(flat_valid):,} pixels  |  "
              f"residual  mean={sample_res.mean():.2f} m  "
              f"std={sample_res.std():.2f} m")

    if not input_samples:
        raise RuntimeError("No training data found to compute normalization stats.")

    combined_inputs = np.concatenate(input_samples, axis=0)  # (N_total, C)
    means = combined_inputs.mean(axis=0).tolist()
    stds  = [max(float(s), 1e-6) for s in combined_inputs.std(axis=0).tolist()]

    combined_res = np.concatenate(res_samples)
    res_mean = float(combined_res.mean())
    res_std  = float(max(combined_res.std(), 0.1))

    stats: NormStats = {
        "channel_names" : CHANNEL_NAMES,
        "means"         : means,
        "stds"          : stds,
        "res_mean"      : res_mean,
        "res_std"       : res_std,
    }
    stats_path = data_root / "norm_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nNorm stats saved to {stats_path}")
    print(f"  {'channel':12s}  {'mean':>10s}  {'std':>10s}")
    for name, m, s in zip(CHANNEL_NAMES, means, stds):
        print(f"  {name:12s}  {m:10.4f}  {s:10.4f}")
    print(f"  {'residual':12s}  {res_mean:10.4f}  {res_std:10.4f}  (metres)")
    return stats


def load_normalization_stats(data_root: Path) -> NormStats:
    """Load previously computed normalization stats from JSON."""
    stats_path = data_root / "norm_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"{stats_path} not found.  "
            "Run compute_normalization_stats() first "
            "(called automatically by 4_train_unet.py on first run)."
        )
    with open(stats_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DSMPatchDataset(Dataset):
    """
    Patch-based dataset for U-Net DTM residual prediction.

    Each item is:
      x : (9, patch_size, patch_size)  float32  normalised input channels
      y : (1, patch_size, patch_size)  float32  normalised residual target

    The residual target is  (DTM_lidar − COP-DEM_10m),  normalised by
    subtracting res_mean and dividing by res_std (stored in norm_stats).
    To reconstruct absolute DTM metres at inference:
        DTM_pred = (y_pred * res_std + res_mean) + COP-DEM_input_metres

    Parameters
    ----------
    zone_ids       : list of zone IDs to load
    data_root      : path to data/ directory
    norm_stats     : NormStats dict (from compute_normalization_stats)
    patch_size     : spatial patch size in pixels (default 256)
    stride         : stride between patch origins (default 128; use
                     patch_size for non-overlapping eval)
    use_tcd        : include TCD (tree cover density) channel (default True)
    use_imd        : include IMD (imperviousness) channel (default True)
    augment        : if True, random flips + 90° rotations
    min_valid_frac : minimum fraction of finite pixels in a patch to keep
    """

    def __init__(
        self,
        zone_ids      : list[str],
        data_root     : Path,
        norm_stats    : NormStats,
        patch_size    : int   = 256,
        stride        : int   = 128,
        augment       : bool  = False,
        min_valid_frac: float = 0.5,
        use_tcd       : bool  = True,
        use_imd       : bool  = True,
    ):
        self.patch_size     = patch_size
        self.stride         = stride
        self.augment        = augment
        self.min_valid_frac = min_valid_frac

        # Channel selection based on use_tcd / use_imd
        self._channel_indices = [
            i for i, (_, _, name) in enumerate(CHANNEL_SPEC)
            if not (name == "tcd" and not use_tcd)
            and not (name == "imd" and not use_imd)
        ]
        self.n_channels = len(self._channel_indices)

        # Input channel normalization (only for selected channels)
        means = np.array(norm_stats["means"], dtype="float32")
        stds  = np.array(norm_stats["stds"],  dtype="float32")
        self.means = means[self._channel_indices, None, None]   # (C_sel, 1, 1)
        self.stds  = stds[self._channel_indices, None, None]

        # Residual normalization
        self.res_mean = float(norm_stats.get("res_mean", 0.0))
        self.res_std  = float(norm_stats.get("res_std",  1.0))

        # Enumerate valid patches
        self._patches: list[tuple[ZoneData, int, int]] = []
        for zone_id in zone_ids:
            zd = load_zone(zone_id, data_root)
            if zd is None:
                continue
            _, H, W = zd.inputs.shape
            n_added = 0
            for row in range(0, H - patch_size + 1, stride):
                for col in range(0, W - patch_size + 1, stride):
                    frac = zd.valid_mask[row:row+patch_size,
                                         col:col+patch_size].mean()
                    if frac >= min_valid_frac:
                        self._patches.append((zd, row, col))
                        n_added += 1
            print(f"  [{zone_id}] {n_added} valid patches  "
                  f"(stride={stride}, min_valid={min_valid_frac:.0%})")

        print(f"  Total patches: {len(self._patches)}")

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, idx: int) -> "tuple[torch.Tensor, torch.Tensor]":
        zd, row, col = self._patches[idx]
        ps = self.patch_size

        x = zd.inputs[self._channel_indices, row:row+ps, col:col+ps].copy()  # (C_sel, H, W) raw
        y = zd.residual[row:row+ps, col:col+ps].copy()     # (H, W)    raw residual

        # Replace NaN in inputs with the channel mean (→ 0 after normalisation)
        for ci in range(x.shape[0]):
            mask = ~np.isfinite(x[ci])
            if mask.any():
                x[ci, mask] = float(self.means[ci, 0, 0])

        # Replace NaN in residual with res_mean (→ 0 after normalisation)
        y_nan = ~np.isfinite(y)
        if y_nan.any():
            y[y_nan] = self.res_mean

        # Normalise inputs
        x = (x - self.means) / self.stds

        # Normalise residual
        y = (y - self.res_mean) / self.res_std

        if self.augment:
            x, y = self._augment(x, y)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y[None].astype("float32")),   # (1, H, W)
        )

    @staticmethod
    def _augment(x: np.ndarray, y: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
        """Random horizontal/vertical flip and 90° rotation."""
        if random.random() > 0.5:
            x = x[:, :, ::-1].copy()
            y = y[:, ::-1].copy()
        if random.random() > 0.5:
            x = x[:, ::-1, :].copy()
            y = y[::-1, :].copy()
        k = random.randint(0, 3)
        if k:
            x = np.rot90(x, k, axes=(1, 2)).copy()
            y = np.rot90(y, k, axes=(0, 1)).copy()
        return x, y


# ---------------------------------------------------------------------------
# CLI: visualise n random patches
# ---------------------------------------------------------------------------

# Colormaps and display names for each channel
_CHANNEL_CMAPS = {
    "cop_dem": ("terrain", False),
    "slope":   ("magma",   False),
    "B02":     ("gray",    False),
    "B03":     ("gray",    False),
    "B04":     ("gray",    False),
    "B08":     ("gray",    False),
    "ndvi":    ("RdYlGn",  True),   # symmetric around 0
    "tcd":     ("Greens",  False),
    "imd":     ("hot_r",   False),
}


def visualise_patches(n: int, zone_ids: list[str], data_root: Path,
                      norm_stats: NormStats,
                      patch_size: int = 256, stride: int = 128,
                      out_dir: Path | None = None,
                      seed: int = 0,
                      use_tcd: bool = True,
                      use_imd: bool = True) -> None:
    """
    Draw n random patches from the dataset and save one figure per patch.

    Each figure has a grid of subplots:
      • one subplot per input channel (9 total, shown in physical units)
      • one subplot for the residual target (DTM_lidar − COP-DEM, in metres)

    Physical-unit reconstruction:
      channel_value = x_norm * std + mean
      residual_m    = y_norm * res_std + res_mean

    Parameters
    ----------
    n          : number of patches to visualise
    zone_ids   : zone list to build the dataset from
    data_root  : path to data/ directory
    norm_stats : NormStats dict
    patch_size : patch size in pixels
    stride     : stride used to enumerate patches
    out_dir    : directory to save figures (default: data_root/../output/dataset_viz)
    seed       : random seed for patch selection
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = DSMPatchDataset(
        zone_ids, data_root, norm_stats,
        patch_size=patch_size, stride=stride,
        augment=False,
        use_tcd=use_tcd, use_imd=use_imd,
    )
    selected_names = [CHANNEL_SPEC[i][2] for i in ds._channel_indices]
    total = len(ds)
    if total == 0:
        print("ERROR: no patches found — run 1_prepare.py first.")
        return

    n = min(n, total)
    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=n, replace=False).tolist()

    if out_dir is None:
        out_dir = data_root.parent / "output" / "dataset_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    means_1d = ds.means[:, 0, 0]   # (C,)
    stds_1d  = ds.stds[:, 0, 0]    # (C,)
    res_mean = ds.res_mean
    res_std  = ds.res_std

    n_cols   = 5
    n_panels = ds.n_channels + 1   # selected inputs + 1 target
    n_rows   = (n_panels + n_cols - 1) // n_cols

    for k, idx in enumerate(indices):
        x_norm, y_norm = ds[idx]
        x_norm = x_norm.numpy()            # (C, H, W)
        y_norm = y_norm.squeeze().numpy()  # (H, W)

        # Denormalise to physical units
        x_phys = x_norm * stds_1d[:, None, None] + means_1d[:, None, None]
        y_phys = y_norm * res_std + res_mean       # residual in metres

        # Zone / row / col metadata
        zd, row, col = ds._patches[idx]
        title_main = (f"Patch {k+1}/{n}  |  zone={zd.zone_id}  "
                      f"row={row} col={col}  ({patch_size}×{patch_size} px)")

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 3.2, n_rows * 3.2))
        axes = axes.ravel()

        for ci, cname in enumerate(selected_names):
            ax  = axes[ci]
            arr = x_phys[ci]
            cmap, symmetric = _CHANNEL_CMAPS.get(cname, ("viridis", False))
            v = arr[np.isfinite(arr)]
            if symmetric:
                lim = float(np.percentile(np.abs(v), 98)) if v.size else 1.0
                im = ax.imshow(arr, cmap=cmap, vmin=-lim, vmax=lim,
                               aspect="equal", interpolation="nearest")
            else:
                vmin = float(np.percentile(v, 2))  if v.size else 0
                vmax = float(np.percentile(v, 98)) if v.size else 1
                im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                               aspect="equal", interpolation="nearest")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"[{ci}] {cname}", fontsize=8)
            ax.axis("off")

        # Last panel: residual target
        ax = axes[ds.n_channels]
        v = y_phys[np.isfinite(y_phys)]
        lim = float(np.percentile(np.abs(v), 98)) if v.size else 1.0
        im = ax.imshow(y_phys, cmap="RdBu_r", vmin=-lim, vmax=lim,
                       aspect="equal", interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"[target] residual (m)\nmean={v.mean():.2f} std={v.std():.2f}",
                     fontsize=8)
        ax.axis("off")

        # Hide leftover axes
        for ax in axes[n_panels:]:  # noqa: F821
            ax.set_visible(False)

        fig.suptitle(title_main, fontsize=9, y=1.01)
        plt.tight_layout()
        out_path = out_dir / f"patch_{k+1:03d}_zone_{zd.zone_id}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{k+1}/{n}] {out_path.name}")

    print(f"\nSaved {n} figures to {out_dir}/")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Visualise random patches from the DSMPatchDataset."
    )
    parser.add_argument("n", type=int,
                        help="Number of random patches to visualise")
    parser.add_argument("--zones", nargs="+", default=None,
                        help="Zone IDs to use (default: all TRAIN+VAL+TEST zones)")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride",     type=int, default=128)
    parser.add_argument("--out-dir",    type=str, default=None,
                        help="Output directory (default: output/dataset_viz/)")
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--no-tcd",    action="store_true",
                        help="Exclude TCD (tree cover density) channel")
    parser.add_argument("--no-imd",    action="store_true",
                        help="Exclude IMD (imperviousness) channel")
    args = parser.parse_args()

    _root = Path(__file__).resolve().parent.parent
    _data_root = _root / "data"

    # Resolve zone list
    if args.zones:
        _zone_ids = args.zones
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from zones import ZONES
        _zone_ids = [z["id"] for z in ZONES]

    # Load or compute norm stats
    _stats_path = _data_root / "norm_stats.json"
    if _stats_path.exists():
        _norm_stats = load_normalization_stats(_data_root)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from zones import TRAIN_ZONES
        print("norm_stats.json not found — computing from train zones ...")
        _norm_stats = compute_normalization_stats(TRAIN_ZONES, _data_root)

    _out_dir = Path(args.out_dir) if args.out_dir else None
    visualise_patches(
        n          = args.n,
        zone_ids   = _zone_ids,
        data_root  = _data_root,
        norm_stats = _norm_stats,
        patch_size = args.patch_size,
        stride     = args.stride,
        out_dir    = _out_dir,
        seed       = args.seed,
        use_tcd    = not args.no_tcd,
        use_imd    = not args.no_imd,
    )
