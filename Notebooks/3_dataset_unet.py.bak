"""
3_dataset_unet.py
=================
PyTorch Dataset and normalization utilities for the U-Net DTM model.

This module is imported by 4_train_unet.py; it is not run directly.

Dataset
-------
  DSMPatchDataset
    Loads 256x256 patches on the fly from aligned zone data.
    Each patch has:
      - x: float32 tensor  (9, 256, 256)  -- input channels
      - y: float32 tensor  (1, 256, 256)  -- DTM target (metres)

    Input channel order (9 channels):
      0  cop_dem   -- COP-DEM 30m bilinear (m)
      1  slope     -- slope from COP-DEM (degrees)
      2  B02       -- Sentinel-2 blue
      3  B03       -- Sentinel-2 green
      4  B04       -- Sentinel-2 red
      5  B08       -- Sentinel-2 NIR
      6  ndvi      -- NDVI from Sentinel-2
      7  tcd       -- TCD greyscale proxy  [0, 1]
      8  imd       -- IMD greyscale proxy  [0, 1]

Normalization
-------------
  compute_normalization_stats(zone_ids, data_root)
    Computes per-channel (mean, std) from a subset of pixels drawn from the
    specified zones. Results saved to data/norm_stats.json.

  load_normalization_stats(data_root)
    Loads previously computed stats.

Usage
-----
  from zones import TRAIN_ZONES
  from dataset_unet import DSMPatchDataset, compute_normalization_stats

  stats = compute_normalization_stats(TRAIN_ZONES, data_root)
  train_ds = DSMPatchDataset(TRAIN_ZONES, data_root, stats,
                             patch_size=256, stride=128, augment=True)
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

# List of (aligned_filename, band_1based, channel_name)
CHANNEL_SPEC = [
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
N_CHANNELS    = len(CHANNEL_SPEC)


# ---------------------------------------------------------------------------
# Zone data loader
# ---------------------------------------------------------------------------

class ZoneData(NamedTuple):
    """Holds memory-mapped arrays for one zone."""
    zone_id  : str
    inputs   : np.ndarray    # (C, H, W) float32
    target   : np.ndarray    # (H, W)    float32
    valid_mask: np.ndarray   # (H, W)    bool  -- True where all channels + target are finite


def load_zone(zone_id: str, data_root: Path) -> "ZoneData | None":
    """
    Load all aligned channels and the DTM target for one zone.

    Returns None if mandatory files are missing.
    """
    aligned_dir = data_root / "aligned" / zone_id

    dtm_path = aligned_dir / "dtm_lidar.tif"
    if not dtm_path.exists():
        print(f"  WARNING [{zone_id}]: dtm_lidar.tif not found -- zone skipped")
        return None

    with rasterio.open(dtm_path) as src:
        target = src.read(1).astype("float32")
        if src.nodata is not None:
            target[target == src.nodata] = np.nan

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
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            # Resize if shape mismatch (should not happen after prepare_zone)
            if arr.shape != (H, W):
                print(f"  WARNING [{zone_id}]: {fname} band {band_idx} shape "
                      f"{arr.shape} != target {(H,W)} -- skipped")
                arr = np.full((H, W), np.nan, dtype="float32")
            inputs[ci] = arr
    finally:
        for f in open_files.values():
            f.close()

    valid_mask = np.isfinite(target)
    for ci in range(N_CHANNELS):
        valid_mask &= np.isfinite(inputs[ci])

    return ZoneData(
        zone_id   = zone_id,
        inputs    = inputs,
        target    = target,
        valid_mask = valid_mask,
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

NormStats = dict   # {"means": list[float], "stds": list[float]}


def compute_normalization_stats(zone_ids: list[str], data_root: Path,
                                max_pixels_per_zone: int = 100_000,
                                seed: int = 42) -> NormStats:
    """
    Compute per-channel mean and std from training zones and save to JSON.

    Parameters
    ----------
    zone_ids           : list of zone IDs to compute stats from (training zones only)
    data_root          : Path to data/ directory
    max_pixels_per_zone: max number of valid pixels sampled per zone
    seed               : random seed for reproducibility

    Returns
    -------
    dict with keys "means" and "stds" (lists of length N_CHANNELS)
    """
    rng = np.random.default_rng(seed)
    all_samples: list[np.ndarray] = []   # list of (N, C) arrays

    for zone_id in zone_ids:
        zd = load_zone(zone_id, data_root)
        if zd is None:
            continue
        flat_valid = np.where(zd.valid_mask.ravel())[0]
        if len(flat_valid) > max_pixels_per_zone:
            flat_valid = rng.choice(flat_valid, size=max_pixels_per_zone, replace=False)
        sample = zd.inputs.reshape(N_CHANNELS, -1)[:, flat_valid].T   # (N, C)
        all_samples.append(sample)
        print(f"  [{zone_id}] {len(flat_valid):,} pixels sampled")

    if not all_samples:
        raise RuntimeError("No training data found to compute normalization stats.")

    combined = np.concatenate(all_samples, axis=0)   # (N_total, C)
    means = combined.mean(axis=0).tolist()
    stds  = combined.std(axis=0).tolist()

    # Replace zero std with 1 to avoid division by zero
    stds = [s if s > 0 else 1.0 for s in stds]

    stats: NormStats = {
        "channel_names": CHANNEL_NAMES,
        "means": means,
        "stds": stds,
    }
    stats_path = data_root / "norm_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nNorm stats saved to {stats_path}")
    for i, (name, m, s) in enumerate(zip(CHANNEL_NAMES, means, stds)):
        print(f"  {name:12s}  mean={m:10.4f}  std={s:10.4f}")
    return stats


def load_normalization_stats(data_root: Path) -> NormStats:
    """Load previously computed normalization stats from JSON."""
    stats_path = data_root / "norm_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"{stats_path} not found.  "
            "Run compute_normalization_stats() first (called automatically by "
            "4_train_unet.py)."
        )
    with open(stats_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DSMPatchDataset(Dataset):
    """
    Patch-based dataset for U-Net DTM estimation.

    Samples 256x256 patches from any number of aligned zones.  Patches
    whose valid pixel fraction falls below `min_valid_frac` are discarded.

    Parameters
    ----------
    zone_ids      : list of zone IDs to include
    data_root     : Path to data/ directory
    norm_stats    : normalization stats (from compute_normalization_stats)
    patch_size    : spatial size of each patch (pixels)
    stride        : stride between patch origins (< patch_size -> overlap)
    augment       : if True, apply random flips + 90-degree rotations
    min_valid_frac: minimum fraction of finite pixels in a patch to keep it
    """

    def __init__(
        self,
        zone_ids     : list[str],
        data_root    : Path,
        norm_stats   : NormStats,
        patch_size   : int   = 256,
        stride       : int   = 128,
        augment      : bool  = False,
        min_valid_frac: float = 0.5,
    ):
        self.patch_size     = patch_size
        self.stride         = stride
        self.augment        = augment
        self.min_valid_frac = min_valid_frac

        means = np.array(norm_stats["means"], dtype="float32")
        stds  = np.array(norm_stats["stds"],  dtype="float32")
        self.means = means[:, None, None]   # (C, 1, 1) for broadcasting
        self.stds  = stds[:, None, None]

        # Load all zones and enumerate valid patches
        self._patches: list[tuple[ZoneData, int, int]] = []  # (zone_data, row, col)
        for zone_id in zone_ids:
            zd = load_zone(zone_id, data_root)
            if zd is None:
                continue
            _, H, W = zd.inputs.shape
            n_added = 0
            for row in range(0, H - patch_size + 1, stride):
                for col in range(0, W - patch_size + 1, stride):
                    patch_valid = zd.valid_mask[row:row+patch_size, col:col+patch_size]
                    frac = patch_valid.mean()
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

        x = zd.inputs[:, row:row+ps, col:col+ps].copy()   # (C, H, W)
        y = zd.target[row:row+ps, col:col+ps].copy()      # (H, W)

        # Replace NaN with channel mean (zero after normalization)
        for ci in range(x.shape[0]):
            nan_mask = ~np.isfinite(x[ci])
            if nan_mask.any():
                x[ci, nan_mask] = float(self.means[ci, 0, 0])

        y_nan = ~np.isfinite(y)
        if y_nan.any():
            y[y_nan] = float(np.nanmean(zd.target))

        # Normalize inputs
        x = (x - self.means) / self.stds

        # Augmentation
        if self.augment:
            x, y = self._augment(x, y)

        return (torch.from_numpy(x),
                torch.from_numpy(y[None].astype("float32")))   # (1, H, W) target

    @staticmethod
    def _augment(x: np.ndarray, y: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
        """Random horizontal/vertical flip and 90-degree rotation."""
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
