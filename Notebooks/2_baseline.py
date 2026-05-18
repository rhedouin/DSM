"""
2_baseline.py
=============
XGBoost per-pixel baseline for DSM/DTM estimation.

For each pixel the feature vector consists of:
  - COP-DEM value and 3x3 neighbourhood statistics (mean, std, range)
  - Slope
  - TCD proxy
  - IMD proxy
  - NDVI
  - Sentinel-2 B02, B03, B04, B08

Target: Lidar HD DTM value (absolute elevation, metres).

The model is trained on TRAIN zones (caen, grenoble, landes) and evaluated
on VAL (paris_sud) and TEST (toulouse) zones.

Usage
-----
  python 2_baseline.py                  # train + evaluate
  python 2_baseline.py --max-pixels 100000   # override pixels per zone
  python 2_baseline.py --output-dir ./output/baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("WARNING: xgboost not installed -- falling back to sklearn RandomForest")
    from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Pixels sampled per zone for training/evaluation
DEFAULT_MAX_PIXELS = 50_000

# Random seed for reproducibility
RANDOM_SEED = 42

# Neighbourhood window half-size for DEM features
WINDOW_HALF = 1   # 1 -> 3x3 neighbourhood

# Input channels loaded from aligned/{zone_id}/ and the source file / band:
#   (aligned_filename, band_index_1based, feature_name)
CHANNEL_SPEC = [
    ("copdem30.tif",  1, "dem"),
    ("slope.tif",     1, "slope"),
    ("sentinel2.tif", 1, "B02"),
    ("sentinel2.tif", 2, "B03"),
    ("sentinel2.tif", 3, "B04"),
    ("sentinel2.tif", 4, "B08"),
    ("sentinel2.tif", 5, "ndvi"),
    ("tcd.tif",       1, "tcd"),
    ("imd.tif",       1, "imd"),
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def load_aligned_stack(zone_id: str, data_root: Path) -> "tuple[np.ndarray, np.ndarray]":
    """
    Load all input channels and the DTM target for one zone.

    Returns
    -------
    X_stack : float32 array of shape (n_channels, H, W)
    Y_dtm   : float32 array of shape (H, W)   -- NaN where no data
    """
    aligned_dir = data_root / "aligned" / zone_id

    # Check target first
    dtm_path = aligned_dir / "dtm_lidar.tif"
    if not dtm_path.exists():
        raise FileNotFoundError(f"Target DTM not found: {dtm_path}")

    with rasterio.open(dtm_path) as src:
        y = src.read(1).astype("float32")
        if src.nodata is not None:
            y[y == src.nodata] = np.nan

    # Load each input channel
    channels = []
    channel_names = []
    open_files: dict[str, rasterio.DatasetReader] = {}

    try:
        for fname, band_idx, cname in CHANNEL_SPEC:
            fpath = aligned_dir / fname
            if fpath not in open_files:
                if not fpath.exists():
                    print(f"  WARNING: {fpath.name} missing -- channel {cname!r} set to NaN")
                    # Use NaN array of same shape as target
                    arr = np.full(y.shape, np.nan, dtype="float32")
                else:
                    src = rasterio.open(fpath)
                    open_files[str(fpath)] = src
                    arr = src.read(band_idx).astype("float32")
                    if src.nodata is not None:
                        arr[arr == src.nodata] = np.nan
            else:
                src = open_files[str(fpath)]
                arr = src.read(band_idx).astype("float32")
                if src.nodata is not None:
                    arr[arr == src.nodata] = np.nan
            channels.append(arr)
            channel_names.append(cname)
    finally:
        for f in open_files.values():
            f.close()

    X_stack = np.stack(channels, axis=0)   # (C, H, W)
    return X_stack, y


def extract_windowed_features(X_stack: np.ndarray, y: np.ndarray,
                              max_pixels: int = DEFAULT_MAX_PIXELS,
                              seed: int = RANDOM_SEED,
                              w: int = WINDOW_HALF) -> "tuple[np.ndarray, np.ndarray]":
    """
    Extract per-pixel feature vectors with local neighbourhood statistics
    for the DEM channel.

    For the DEM channel (index 0) we add:  mean, std, range over a (2w+1)^2 window.
    All other channels are taken as point values.

    Returns
    -------
    features : float32 array of shape (n_valid_pixels, n_features)
    targets  : float32 array of shape (n_valid_pixels,)
    """
    from scipy.ndimage import uniform_filter, generic_filter

    n_ch, H, W = X_stack.shape
    dem = X_stack[0]   # first channel is always DEM

    # DEM neighbourhood stats
    pad = w
    dem_pad = np.pad(dem, pad, mode="edge")
    kernel_size = 2 * w + 1
    neigh_sum = uniform_filter(dem_pad, kernel_size)[pad:pad+H, pad:pad+W]
    neigh_sq  = uniform_filter(dem_pad ** 2, kernel_size)[pad:pad+H, pad:pad+W]
    neigh_mean = neigh_sum
    neigh_std  = np.sqrt(np.maximum(neigh_sq - neigh_mean ** 2, 0.0)).astype("float32")

    # Simple range via max-min in window (slightly expensive but one-time)
    def range_func(vals):
        return vals.max() - vals.min()
    dem_range = generic_filter(dem, range_func, size=kernel_size).astype("float32")

    extra_feats = [neigh_mean.astype("float32"),
                   neigh_std,
                   dem_range]

    # Build pixel mask: valid where ALL channels and target are finite
    mask = np.isfinite(y)
    for ch in range(n_ch):
        mask &= np.isfinite(X_stack[ch])
    valid_idx = np.where(mask.ravel())[0]

    if len(valid_idx) > max_pixels:
        rng = np.random.default_rng(seed)
        valid_idx = rng.choice(valid_idx, size=max_pixels, replace=False)
        valid_idx.sort()

    # Flatten channels
    flat_channels = [X_stack[c].ravel()[valid_idx] for c in range(n_ch)]
    flat_extra    = [ef.ravel()[valid_idx] for ef in extra_feats]

    features = np.stack(flat_channels + flat_extra, axis=1)
    targets  = y.ravel()[valid_idx]
    return features.astype("float32"), targets.astype("float32")


def feature_names() -> list[str]:
    """Return ordered feature names matching extract_windowed_features output."""
    base = [cname for _, _, cname in CHANNEL_SPEC]
    return base + ["dem_neigh_mean", "dem_neigh_std", "dem_neigh_range"]


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def build_model(n_trees: int = 300, max_depth: int = 8, seed: int = RANDOM_SEED):
    """Build XGBoost regressor (or RandomForest if XGBoost unavailable)."""
    if XGB_AVAILABLE:
        return xgb.XGBRegressor(
            n_estimators      = n_trees,
            max_depth         = max_depth,
            learning_rate     = 0.05,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            tree_method       = "hist",
            random_state      = seed,
            n_jobs            = -1,
            verbosity         = 0,
        )
    else:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators = n_trees,
            max_depth    = max_depth,
            random_state = seed,
            n_jobs       = -1,
        )


def evaluate_zone(model, zone_id: str, data_root: Path,
                  max_pixels: int = DEFAULT_MAX_PIXELS) -> dict:
    """Evaluate a trained model on one zone and return metrics dict."""
    X_stack, y = load_aligned_stack(zone_id, data_root)
    X, y_true  = extract_windowed_features(X_stack, y, max_pixels=max_pixels)

    y_pred = model.predict(X).astype("float32")
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    return dict(zone=zone_id, n=len(y_true), mae=mae, rmse=rmse, r2=r2, bias=bias)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_residuals(model, zone_id: str, data_root: Path, out_dir: Path,
                   max_pixels: int = 5000) -> None:
    """Scatter plot and residual histogram for one zone."""
    X_stack, y = load_aligned_stack(zone_id, data_root)
    X, y_true  = extract_windowed_features(X_stack, y, max_pixels=max_pixels)
    y_pred = model.predict(X).astype("float32")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Baseline — {zone_id}", fontsize=11)

    # Scatter
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=4, rasterized=True)
    axes[0].plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1)
    axes[0].set_xlabel("True DTM (m)")
    axes[0].set_ylabel("Predicted DTM (m)")
    axes[0].set_title("Predicted vs True")
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    axes[0].text(0.05, 0.95, f"MAE={mae:.2f} m\nRMSE={rmse:.2f} m",
                 transform=axes[0].transAxes, va="top", fontsize=9,
                 bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    # Residual histogram
    residuals = y_pred - y_true
    axes[1].hist(residuals, bins=50, color="steelblue", edgecolor="white", linewidth=0.3)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Residual (m)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual distribution")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"residuals_{zone_id}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(model, out_dir: Path) -> None:
    """Bar chart of feature importances."""
    names = feature_names()
    if XGB_AVAILABLE and hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        return

    order  = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([names[i] for i in order[::-1]], imp[order[::-1]], color="steelblue")
    ax.set_xlabel("Feature importance")
    ax.set_title("XGBoost feature importances")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="XGBoost baseline for DTM estimation from aligned multi-zone data."
    )
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS,
                        help="Max pixels sampled per zone (default: %(default)s)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: output/baseline/)")
    parser.add_argument("--trees", type=int, default=300,
                        help="Number of estimators (default: %(default)s)")
    return parser.parse_args()


def main():
    from zones import ZONES_BY_ID, TRAIN_ZONES, VAL_ZONES, TEST_ZONES

    args      = parse_args()
    data_root = Path(__file__).resolve().parent.parent / "data"
    out_dir   = (Path(args.output_dir) if args.output_dir
                 else Path(__file__).resolve().parent.parent / "output" / "baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBaseline XGBoost")
    print(f"  Train zones : {TRAIN_ZONES}")
    print(f"  Val zones   : {VAL_ZONES}")
    print(f"  Test zones  : {TEST_ZONES}")
    print(f"  Max pixels  : {args.max_pixels} per zone")
    print(f"  Trees       : {args.trees}")

    # ---- Collect training data ----
    print("\n[1] Collecting training features ...")
    X_train_parts, y_train_parts = [], []
    for zone_id in TRAIN_ZONES:
        print(f"  {zone_id} ...", end=" ", flush=True)
        try:
            X_stack, y = load_aligned_stack(zone_id, data_root)
            X, y_s     = extract_windowed_features(
                X_stack, y, max_pixels=args.max_pixels)
            X_train_parts.append(X)
            y_train_parts.append(y_s)
            print(f"{X.shape[0]} pixels, {X.shape[1]} features")
        except FileNotFoundError as e:
            print(f"SKIP ({e})")

    if not X_train_parts:
        print("ERROR: no training data found.  "
              "Run 0_download_data.py and 1_prepare.py first.")
        return

    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    print(f"\n  Total training pixels: {X_train.shape[0]}")

    # ---- Train ----
    print("\n[2] Training model ...")
    model = build_model(n_trees=args.trees)
    if XGB_AVAILABLE:
        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train)],
                  verbose=False)
    else:
        model.fit(X_train, y_train)
    print("  Done.")

    # ---- Evaluate ----
    print("\n[3] Evaluation")
    all_results = []
    for split_name, zone_list in [("TRAIN", TRAIN_ZONES),
                                   ("VAL",   VAL_ZONES),
                                   ("TEST",  TEST_ZONES)]:
        for zone_id in zone_list:
            try:
                metrics = evaluate_zone(model, zone_id, data_root, args.max_pixels)
                metrics["split"] = split_name
                all_results.append(metrics)
                print(f"  [{split_name}] {zone_id:15s}  "
                      f"MAE={metrics['mae']:.2f} m  RMSE={metrics['rmse']:.2f} m  "
                      f"R2={metrics['r2']:.3f}  bias={metrics['bias']:+.2f} m  "
                      f"n={metrics['n']:,}")
                plot_residuals(model, zone_id, data_root, out_dir)
            except FileNotFoundError as e:
                print(f"  [{split_name}] {zone_id:15s}  SKIP ({e})")

    # ---- Save metrics ----
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")

    # ---- Feature importance ----
    plot_feature_importance(model, out_dir)
    print(f"  Feature importance plot saved to {out_dir}/feature_importance.png")

    # ---- Save model ----
    try:
        import joblib
        model_path = out_dir / "baseline_model.pkl"
        joblib.dump(model, model_path)
        print(f"  Model saved to {model_path}")
    except Exception as e:
        print(f"  WARNING: could not save model ({e})")


if __name__ == "__main__":
    main()
