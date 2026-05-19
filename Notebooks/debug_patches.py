"""
debug_patches.py
================
Diagnostic script — run standalone, does NOT modify any existing scripts.

Produces in  output/debug/:
  1. patch_inputs_{zone}_{k}.png  —  all 9 input channels + DTM target + residual
  2. elevation_distributions.png  —  DTM and residual histograms per zone
  3. prediction_analysis.png      —  what the trained model actually predicts

Usage
-----
  python debug_patches.py
  python debug_patches.py --checkpoint output/unet_base_config_2000_epochs/best_checkpoint.pt
  python debug_patches.py --zones caen grenoble landes --n-patches 2
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
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Notebooks"))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _show(ax, arr, title, cmap="terrain", pct=(2, 98)):
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        ax.set_title(f"{title}\n(no data)", fontsize=7); ax.axis("off"); return
    vmin = float(np.percentile(valid, pct[0]))
    vmax = float(np.percentile(valid, pct[1]))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def _load_raw_patch(zone_id, row, col, ps, data_root):
    """Load one patch in PHYSICAL units (no normalisation)."""
    from dataset_unet import CHANNEL_SPEC
    adir = data_root / "aligned" / zone_id

    channels, names = [], []
    open_files = {}
    for fname, band_idx, cname in CHANNEL_SPEC:
        fpath = adir / fname
        key = str(fpath)
        if key not in open_files:
            if not fpath.exists():
                channels.append(np.full((ps, ps), np.nan, "float32"))
                names.append(cname)
                continue
            open_files[key] = rasterio.open(fpath)
        src = open_files[key]
        arr = src.read(band_idx, window=rasterio.windows.Window(col, row, ps, ps)).astype("float32")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        channels.append(arr)
        names.append(cname)
    for f in open_files.values():
        f.close()

    dtm_path = adir / "dtm_lidar.tif"
    with rasterio.open(dtm_path) as src:
        dtm = src.read(1, window=rasterio.windows.Window(col, row, ps, ps)).astype("float32")
        if src.nodata is not None:
            dtm[dtm == src.nodata] = np.nan

    return np.stack(channels, axis=0), names, dtm   # (C, ps, ps), list[str], (ps, ps)


# ---------------------------------------------------------------------------
# 1. Patch full-input visualisation
# ---------------------------------------------------------------------------

def plot_patch_inputs(zone_id, row, col, patch_size, data_root, out_dir, label=""):
    try:
        inputs, names, dtm = _load_raw_patch(zone_id, row, col, patch_size, data_root)
    except Exception as e:
        print(f"  [{zone_id}] patch load failed: {e}")
        return

    cop_dem = inputs[0]
    residual = dtm - cop_dem

    # 9 inputs + DTM + residual = 11 panels, layout 3 rows × 4 cols
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    fig.suptitle(f"Zone: {zone_id}  patch ({row},{col})  {label}", fontsize=10)
    axes_flat = axes.ravel()

    cmaps = ["terrain", "hot_r", "Blues", "Greens", "Reds", "inferno",
             "RdYlGn", "Greens", "Reds"]
    units = ["m", "°", "refl", "refl", "refl", "refl", "", "0–1", "0–1"]

    for i, (arr, name, cmap, unit) in enumerate(zip(inputs, names, cmaps, units)):
        _show(axes_flat[i], arr, f"ch{i}: {name}  [{unit}]", cmap=cmap)

    _show(axes_flat[9],  dtm,      "DTM Lidar HD (m)  [TARGET]",   "terrain")
    _show(axes_flat[10], residual, "Residual: DTM − COP-DEM (m)", "RdBu_r", pct=(1, 99))

    # Stats text
    stats_ax = axes_flat[11]
    stats_ax.axis("off")
    lines = [
        f"Zone: {zone_id}",
        "",
        f"COP-DEM  : [{np.nanmin(cop_dem):.0f}, {np.nanmax(cop_dem):.0f}] m",
        f"DTM      : [{np.nanmin(dtm):.0f}, {np.nanmax(dtm):.0f}] m",
        f"Residual : [{np.nanmin(residual):.1f}, {np.nanmax(residual):.1f}] m",
        f"  mean   : {np.nanmean(residual):.2f} m",
        f"  std    : {np.nanstd(residual):.2f} m",
        "",
        f"TCD raw  : [{np.nanmin(inputs[7]):.3f}, {np.nanmax(inputs[7]):.3f}]",
        f"IMD raw  : [{np.nanmin(inputs[8]):.3f}, {np.nanmax(inputs[8]):.3f}]",
        f"NDVI raw : [{np.nanmin(inputs[6]):.3f}, {np.nanmax(inputs[6]):.3f}]",
    ]
    stats_ax.text(0.05, 0.95, "\n".join(lines), transform=stats_ax.transAxes,
                  fontsize=8, va="top", family="monospace",
                  bbox=dict(fc="lightyellow", ec="gray", lw=0.5))

    plt.tight_layout()
    fname = out_dir / f"patch_inputs_{zone_id}_{label}.png"
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname.name}")


# ---------------------------------------------------------------------------
# 2. Elevation distributions
# ---------------------------------------------------------------------------

def plot_elevation_distributions(zone_ids, data_root, out_dir):
    fig, axes = plt.subplots(2, len(zone_ids), figsize=(4 * len(zone_ids), 7))
    fig.suptitle("Elevation distributions per zone", fontsize=11)

    for j, zone_id in enumerate(zone_ids):
        adir = data_root / "aligned" / zone_id
        dtm_p = adir / "dtm_lidar.tif"
        dem_p = adir / "copdem30.tif"
        if not dtm_p.exists():
            print(f"  [{zone_id}] dtm missing — skipped")
            continue

        with rasterio.open(dtm_p) as s:
            dtm = s.read(1).astype("float32"); dtm[dtm == s.nodata] = np.nan
        with rasterio.open(dem_p) as s:
            dem = s.read(1).astype("float32"); dem[dem == s.nodata] = np.nan

        res = dtm - dem
        dtm_flat = dtm[np.isfinite(dtm)].ravel()
        res_flat = res[np.isfinite(res)].ravel()

        # Subsample for speed
        if len(dtm_flat) > 200_000:
            idx = np.random.choice(len(dtm_flat), 200_000, replace=False)
            dtm_flat = dtm_flat[idx]
        if len(res_flat) > 200_000:
            idx = np.random.choice(len(res_flat), 200_000, replace=False)
            res_flat = res_flat[idx]

        ax_top = axes[0, j]
        ax_top.hist(dtm_flat, bins=80, color="steelblue", alpha=0.8, edgecolor="none")
        ax_top.axvline(np.mean(dtm_flat), color="red", lw=1.5, ls="--",
                       label=f"mean={np.mean(dtm_flat):.0f}m")
        ax_top.set_title(f"{zone_id}\nDTM absolute (m)", fontsize=8)
        ax_top.set_xlabel("m"); ax_top.legend(fontsize=7)

        ax_bot = axes[1, j]
        ax_bot.hist(res_flat, bins=80, color="darkorange", alpha=0.8, edgecolor="none")
        ax_bot.axvline(0, color="black", lw=1, ls=":")
        ax_bot.axvline(np.mean(res_flat), color="red", lw=1.5, ls="--",
                       label=f"mean={np.mean(res_flat):.2f}m\nstd={np.std(res_flat):.2f}m")
        ax_bot.set_title(f"Residual DTM−COP-DEM (m)", fontsize=8)
        ax_bot.set_xlabel("m"); ax_bot.legend(fontsize=7)

    plt.tight_layout()
    out = out_dir / "elevation_distributions.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# 3. Prediction analysis
# ---------------------------------------------------------------------------

def plot_prediction_analysis(checkpoint_path, data_root, out_dir, n_samples=4):
    """Show what the model actually predicts for a few val patches."""
    from dataset_unet import (
        DSMPatchDataset, load_normalization_stats, N_CHANNELS
    )
    from zones import VAL_ZONES

    # local import to avoid circular deps
    sys.path.insert(0, str(ROOT / "Notebooks"))
    # Need UNetDTM — import from 4_train_unet via importlib
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_train", ROOT / "Notebooks" / "4_train_unet.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    UNetDTM = m.UNetDTM

    norm_stats = load_normalization_stats(data_root)
    val_ds = DSMPatchDataset(VAL_ZONES, data_root, norm_stats,
                             patch_size=256, stride=256, augment=False)
    print(f"  Val dataset: {len(val_ds)} patches")

    device = torch.device("cpu")
    model  = UNetDTM(in_channels=N_CHANNELS, base_filters=32)
    state  = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    indices = np.linspace(0, len(val_ds) - 1, n_samples, dtype=int)
    pred_values, true_values = [], []

    fig, axes = plt.subplots(n_samples, 4, figsize=(14, 3.5 * n_samples))
    fig.suptitle("Model prediction analysis  (val zone)", fontsize=10)

    for i, idx in enumerate(indices):
        x, y = val_ds[idx]
        with torch.no_grad():
            pred_dtm, _ = model(x.unsqueeze(0))
        pred = pred_dtm.squeeze().numpy()
        true = y.squeeze().numpy()
        dem_norm = x[0].numpy()   # normalised COP-DEM

        # Denormalise COP-DEM for display
        dem_mean = norm_stats["means"][0]
        dem_std  = norm_stats["stds"][0]
        dem_raw  = dem_norm * dem_std + dem_mean

        pred_values.append(pred.ravel())
        true_values.append(true.ravel())

        row_axes = axes[i] if n_samples > 1 else axes
        mae_p = float(np.mean(np.abs(pred - true)))

        _show(row_axes[0], dem_raw,        f"COP-DEM (m)", "terrain")
        _show(row_axes[1], true,           f"DTM Lidar (m)", "terrain")
        _show(row_axes[2], pred,           f"Prediction (m)  MAE={mae_p:.1f}m", "terrain")
        _show(row_axes[3], pred - true,    f"Error pred−true (m)", "RdBu_r", pct=(1, 99))

    plt.tight_layout()
    out = out_dir / "prediction_analysis.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")

    # Scatter pred vs true
    all_pred = np.concatenate(pred_values)
    all_true = np.concatenate(true_values)
    print(f"\n  Prediction stats:")
    print(f"    pred range : [{all_pred.min():.1f}, {all_pred.max():.1f}] m")
    print(f"    true range : [{all_true.min():.1f}, {all_true.max():.1f}] m")
    print(f"    pred mean  : {all_pred.mean():.1f} m")
    print(f"    true mean  : {all_true.mean():.1f} m")
    print(f"    MAE        : {np.mean(np.abs(all_pred-all_true)):.2f} m")

    fig2, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(all_true[:5000], all_pred[:5000], s=1, alpha=0.3, color="steelblue")
    lim = [min(all_true.min(), all_pred.min()) - 5,
           max(all_true.max(), all_pred.max()) + 5]
    ax.plot(lim, lim, "r--", lw=1, label="perfect")
    ax.set_xlabel("True DTM (m)"); ax.set_ylabel("Predicted DTM (m)")
    ax.set_title("Scatter: pred vs true (first 5000 px)")
    ax.legend(); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.grid(True, alpha=0.3)
    out2 = out_dir / "scatter_pred_vs_true.png"
    fig2.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {out2.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zones",      nargs="*",
                   default=["caen", "grenoble", "landes"])
    p.add_argument("--n-patches",  type=int, default=2)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--checkpoint", type=str,
                   default=str(ROOT / "output/unet_base_config_2000_epochs/best_checkpoint.pt"))
    p.add_argument("--output-dir", type=str,
                   default=str(ROOT / "output/debug"))
    return p.parse_args()


def main():
    args      = parse_args()
    data_root = ROOT / "data"
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDebug output → {out_dir}/")

    # 1. Patch visualisation for each zone
    print("\n[1] Patch input visualisations ...")
    for zone_id in args.zones:
        adir = data_root / "aligned" / zone_id
        dtm_p = adir / "dtm_lidar.tif"
        if not dtm_p.exists():
            print(f"  [{zone_id}] missing — skipped")
            continue
        with rasterio.open(dtm_p) as src:
            H, W = src.height, src.width
        ps = args.patch_size
        # pick patches near centre and near a corner
        centres = [
            (max(0, H // 2 - ps // 2), max(0, W // 2 - ps // 2)),
            (max(0, H // 4),            max(0, W // 4)),
        ]
        for k, (row, col) in enumerate(centres[:args.n_patches]):
            if row + ps > H or col + ps > W:
                row = max(0, H - ps); col = max(0, W - ps)
            plot_patch_inputs(zone_id, row, col, ps, data_root,
                              out_dir, label=f"patch{k+1}")

    # 2. Elevation distributions (all zones incl. val)
    print("\n[2] Elevation distributions ...")
    all_zones = args.zones + ["paris_sud"]
    plot_elevation_distributions(list(dict.fromkeys(all_zones)), data_root, out_dir)

    # 3. Prediction analysis
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        print(f"\n[3] Prediction analysis from {ckpt.name} ...")
        plot_prediction_analysis(ckpt, data_root, out_dir, n_samples=4)
    else:
        print(f"\n[3] Checkpoint not found ({ckpt}) — skipping prediction analysis")

    print("\nDone.")


if __name__ == "__main__":
    main()
