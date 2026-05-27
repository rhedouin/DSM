"""
4_train_unet.py
===============
U-Net training script for Lidar HD DTM super-resolution from multi-source inputs.

Model
-----
  UNetDTM: encoder-decoder with skip connections.
    Input  : 9 channels (cop_dem, slope, B02, B03, B04, B08, ndvi, tcd, imd)
    Output : 1 channel  — normalised residual  ŷ = (DTM − COP-DEM) / res_std

  At inference the absolute DTM is recovered as:
      DTM_pred = ŷ * res_std + res_mean + COP-DEM_input

Loss
----
  L = L1(ŷ, y) + beta * Sobel_edge_loss(ŷ, y)
  Default beta = 0.05

  The edge loss encourages sharp terrain boundaries (building edges,
  ridge lines) while the L1 term controls the bulk reconstruction error.

  Trivial baseline: predicting ŷ = 0 everywhere gives MAE = res_std ≈ 3–7 m
  (this equals simply using COP-DEM as the DTM).  Any positive R² means
  the model improves on the raw COP-DEM.

Geographic split
----------------
  TRAIN (14 zones), VAL (3 zones), TEST (3 zones)  — defined in zones.py

Usage
-----
  python 4_train_unet.py                    # 100 epochs, default settings
  python 4_train_unet.py --epochs 500 --batch-size 8
  python 4_train_unet.py --resume output/unet/checkpoint.pt
  python 4_train_unet.py --output-dir output/unet_exp1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# 1.  U-Net model
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two consecutive  Conv2d → BatchNorm → ReLU  blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """MaxPool 2×2  then  DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Bilinear upsampling  then  DoubleConv (with skip connection)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if spatial dims differ by 1 px (odd input sizes)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.pad(x, [0, skip.shape[-1] - x.shape[-1],
                          0, skip.shape[-2] - x.shape[-2]])
        return self.conv(torch.cat([x, skip], dim=1))


class UNetDTM(nn.Module):
    """
    Encoder-decoder U-Net for DTM residual regression.

    Predicts  ŷ = (DTM_lidar − COP-DEM_10m) / res_std  (normalised residual).
    At inference: DTM_pred = ŷ * res_std + res_mean + COP-DEM_input_metres.

    Parameters
    ----------
    in_channels  : number of input channels (9 by default)
    base_filters : filters in the first encoder block (doubles at each level)
    """

    def __init__(self, in_channels: int = 9, base_filters: int = 32):
        super().__init__()
        f = base_filters

        # Encoder
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = Down(f,     f * 2)
        self.enc3 = Down(f * 2, f * 4)
        self.enc4 = Down(f * 4, f * 8)

        # Bottleneck
        self.bottleneck = Down(f * 8, f * 16)

        # Decoder
        self.dec4 = Up(f * 16, f * 8,  f * 8)
        self.dec3 = Up(f * 8,  f * 4,  f * 4)
        self.dec2 = Up(f * 4,  f * 2,  f * 2)
        self.dec1 = Up(f * 2,  f,      f)

        # Single regression head: normalised residual
        self.head = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bottleneck(e4)
        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.head(d1)   # (B, 1, H, W)


# ---------------------------------------------------------------------------
# 2.  Loss functions
# ---------------------------------------------------------------------------

def _sobel_edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    L1 loss on Sobel gradient magnitudes.

    Penalises differences in local edge sharpness — helps preserve
    building edges and ridge lines.  Both tensors are (B, 1, H, W).
    """
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=pred.dtype, device=pred.device,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)

    def _grad_mag(t: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(t, kx, padding=1)
        gy = F.conv2d(t, ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    return F.l1_loss(_grad_mag(pred), _grad_mag(target))


def total_loss(pred: torch.Tensor, true: torch.Tensor,
               beta: float = 0.05) -> torch.Tensor:
    """
    L1 + beta * Sobel edge loss, operating on normalised residuals.

    Parameters
    ----------
    pred : (B, 1, H, W)  normalised residual prediction
    true : (B, 1, H, W)  normalised residual target
    beta : weight of the edge term (default 0.05)
    """
    return F.l1_loss(pred, true) + beta * _sobel_edge_loss(pred, true)


# ---------------------------------------------------------------------------
# 3.  Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    scaler=None) -> float:
    model.train()
    running_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                pred = model(x)
                loss = total_loss(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = total_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             res_mean: float, res_std: float) -> dict:
    """
    Evaluate the model on a dataloader.

    Metrics are reported in metres of residual error (denormalised), so
    they have a clear physical interpretation:
      • MAE = mean absolute error of (predicted_residual − true_residual)
      • R²  > 0 means the model improves on the trivial COP-DEM baseline
              (which has R² = 0 by definition since it predicts residual = 0)
    """
    model.eval()
    pred_list, true_list = [], []

    for x, y in loader:
        pred = model(x.to(device)).cpu().numpy()
        pred_list.append(pred.ravel())
        true_list.append(y.numpy().ravel())

    pred_norm = np.concatenate(pred_list)
    true_norm = np.concatenate(true_list)

    # Denormalise to metres
    pred_m = pred_norm * res_std + res_mean
    true_m = true_norm * res_std + res_mean

    mae  = float(np.mean(np.abs(pred_m - true_m)))
    rmse = float(np.sqrt(np.mean((pred_m - true_m) ** 2)))
    ss_res = np.sum((pred_m - true_m) ** 2)
    ss_tot = np.sum((true_m - true_m.mean()) ** 2) + 1e-8
    r2   = float(1 - ss_res / ss_tot)
    return dict(mae=mae, rmse=rmse, r2=r2)


# ---------------------------------------------------------------------------
# 4.  Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, model: nn.Module,
                    optimizer=None, scheduler=None) -> int:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    return state.get("epoch", 0)


# ---------------------------------------------------------------------------
# 5.  Visualisation helpers
# ---------------------------------------------------------------------------

def plot_training_curves(history: list[dict], out_dir: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Training history", fontsize=11)

    for key, ax, title in [
        ("train_loss", axes[0], "Loss (normalised residual)"),
        ("val_mae",    axes[1], "Val MAE (metres of residual)"),
    ]:
        vals = [h[key] for h in history if key in h]
        if vals:
            ax.plot(epochs[:len(vals)], vals, marker="o", markersize=2)
            ax.set_xlabel("Epoch")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def visualise_predictions(model: nn.Module, dataset, device: torch.device,
                          out_dir: Path, n_samples: int = 4) -> None:
    """
    Plot COP-DEM | Predicted DTM | True DTM | Error  for sample patches.

    All panels are shown in absolute metres, making the output directly
    interpretable.  Error is symmetric-clipped at the 95th percentile of
    its absolute value for visual clarity.
    """
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    res_mean = dataset.res_mean
    res_std  = dataset.res_std
    cop_mean = float(dataset.means[0, 0, 0])
    cop_std  = float(dataset.stds[0, 0, 0])

    n = len(dataset)
    if n == 0:
        print("  WARNING: dataset empty — skipping visualisations")
        return

    indices = np.linspace(0, n - 1, min(n_samples, n), dtype=int)
    # Remove duplicates (can happen when n < n_samples)
    indices = list(dict.fromkeys(indices.tolist()))

    for k, idx in enumerate(indices):
        x, y = dataset[idx]
        pred_norm = model(x.unsqueeze(0).to(device)).squeeze().cpu().numpy()

        # Denormalise COP-DEM (channel 0) back to metres
        cop_dem = x[0].numpy() * cop_std + cop_mean

        # Denormalise residuals to metres
        pred_res = pred_norm * res_std + res_mean
        true_res = y.squeeze().numpy() * res_std + res_mean

        # Reconstruct absolute DTMs
        pred_dtm  = cop_dem + pred_res
        true_dtm  = cop_dem + true_res
        error     = pred_dtm - true_dtm          # model error
        cop_error = cop_dem  - true_dtm          # COP-DEM baseline error (= -true_res)

        # Shared symmetric scale for both error panels (95th percentile of |error|)
        err_vals = np.concatenate([
            error[np.isfinite(error)],
            cop_error[np.isfinite(cop_error)],
        ])
        shared_err_lim = float(np.percentile(np.abs(err_vals), 95)) if err_vals.size else 1.0

        # Shared elevation scale for the three DTM panels
        elev_vals = np.concatenate([
            cop_dem[np.isfinite(cop_dem)],
            pred_dtm[np.isfinite(pred_dtm)],
            true_dtm[np.isfinite(true_dtm)],
        ])
        elev_vmin = float(np.percentile(elev_vals, 2))  if elev_vals.size else 0.0
        elev_vmax = float(np.percentile(elev_vals, 98)) if elev_vals.size else 1.0

        fig, axes = plt.subplots(1, 5, figsize=(22, 4))

        # sym_lim=None + elev=True → shared elev scale; sym_lim=float → symmetric ±lim
        for ax, arr, title, cmap, sym_lim, use_elev in [
            (axes[0], cop_dem,   "COP-DEM input (m)",          "terrain", None,             True),
            (axes[1], pred_dtm,  "Predicted DTM (m)",          "terrain", None,             True),
            (axes[2], true_dtm,  "True DTM Lidar (m)",         "terrain", None,             True),
            (axes[3], error,     "Error: pred − true (m)",     "RdBu_r",  shared_err_lim,  False),
            (axes[4], cop_error, "COP-DEM − true DTM (m)",     "RdBu_r",  shared_err_lim,  False),
        ]:
            if sym_lim is not None:
                im = ax.imshow(arr, cmap=cmap, vmin=-sym_lim, vmax=sym_lim, aspect="equal")
            elif use_elev:
                im = ax.imshow(arr, cmap=cmap, vmin=elev_vmin, vmax=elev_vmax, aspect="equal")
            else:
                v = arr[np.isfinite(arr)]
                vmin = float(np.percentile(v, 2))  if v.size else 0
                vmax = float(np.percentile(v, 98)) if v.size else 1
                im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        err_v    = error[np.isfinite(error)]
        mae_p    = float(np.mean(np.abs(err_v))) if err_v.size else float("nan")
        fig.suptitle(f"Sample {k+1}  |  patch MAE = {mae_p:.2f} m", fontsize=10)
        plt.tight_layout()
        fig.savefig(out_dir / f"pred_sample_{k+1}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# 6.  Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train U-Net for Lidar HD DTM residual estimation."
    )
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch-size",   type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--base-filters", type=int,   default=32)
    parser.add_argument("--patch-size",   type=int,   default=256)
    parser.add_argument("--stride",       type=int,   default=128)
    parser.add_argument("--beta",         type=float, default=0.05,
                        help="Weight of Sobel edge loss term (default 0.05)")
    parser.add_argument("--resume",       type=str,   default=None,
                        help="Path to checkpoint .pt file to resume training")
    parser.add_argument("--output-dir",   type=str,   default=None,
                        help="Output directory (default: output/unet/)")
    return parser.parse_args()


def main():
    from zones import TRAIN_ZONES, VAL_ZONES, TEST_ZONES
    from dataset_unet import (
        DSMPatchDataset, N_CHANNELS,
        compute_normalization_stats, load_normalization_stats,
    )

    args      = parse_args()
    data_root = Path(__file__).resolve().parent.parent / "data"
    out_dir   = (
        Path(args.output_dir) if args.output_dir
        else Path(__file__).resolve().parent.parent / "output" / "unet"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nU-Net DTM training  (residual formulation)")
    print(f"  Device       : {device}")
    print(f"  Train zones  : {TRAIN_ZONES}")
    print(f"  Val zones    : {VAL_ZONES}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Patch size   : {args.patch_size}")
    print(f"  Stride       : {args.stride}")

    # ── [1] Normalization stats ───────────────────────────────────────────
    stats_path = data_root / "norm_stats.json"
    if stats_path.exists():
        print("\n[1] Loading existing normalization stats ...")
        norm_stats = load_normalization_stats(data_root)
        print(f"  res_mean={norm_stats['res_mean']:.3f} m  "
              f"res_std={norm_stats['res_std']:.3f} m")
    else:
        print("\n[1] Computing normalization stats from train zones ...")
        norm_stats = compute_normalization_stats(TRAIN_ZONES, data_root)

    res_mean = norm_stats["res_mean"]
    res_std  = norm_stats["res_std"]

    # ── [2] Datasets ──────────────────────────────────────────────────────
    print("\n[2] Building datasets ...")
    print("  Train:")
    train_ds = DSMPatchDataset(
        TRAIN_ZONES, data_root, norm_stats,
        patch_size=args.patch_size, stride=args.stride,
        augment=True,
    )
    print("  Val:")
    val_ds = DSMPatchDataset(
        VAL_ZONES, data_root, norm_stats,
        patch_size=args.patch_size, stride=args.patch_size,   # no overlap for val
        augment=False,
    )

    if len(train_ds) == 0:
        print("ERROR: no training patches found — "
              "run 0_download_data.py and 1_prepare.py first.")
        sys.exit(1)

    n_workers = min(4, len(TRAIN_ZONES))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=n_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    # ── [3] Model ─────────────────────────────────────────────────────────
    print("\n[3] Building model ...")
    model = UNetDTM(in_channels=N_CHANNELS, base_filters=args.base_filters)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters : {n_params:,}")
    print(f"  Input channels       : {N_CHANNELS}")
    print(f"  Base filters         : {args.base_filters}")

    # ── [4] Optimiser and scheduler ───────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # AMP scaler (CUDA only)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            start_epoch = load_checkpoint(resume_path, model, optimizer, scheduler)
            print(f"  Resumed from epoch {start_epoch}: {resume_path}")
        else:
            print(f"  WARNING: {resume_path} not found — starting fresh")

    # Trivial baseline MAE = res_std (predicting 0 everywhere = just COP-DEM)
    print(f"\n  Trivial baseline MAE = {res_std:.3f} m  "
          f"(= res_std, equivalent to using raw COP-DEM)")

    # ── [5] Training loop ─────────────────────────────────────────────────
    print(f"\n[4] Training for {args.epochs} epochs ...")
    history:  list[dict] = []
    best_mae = float("inf")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, scaler
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, res_mean, res_std)
        val_mae  = val_metrics["mae"]
        val_rmse = val_metrics["rmse"]
        val_r2   = val_metrics["r2"]
        lr_now   = scheduler.get_last_lr()[0]

        print(f"  Epoch {epoch:4d}/{args.epochs}  "
              f"loss={train_loss:.4f}  "
              f"val_MAE={val_mae:.3f} m  RMSE={val_rmse:.3f}  R²={val_r2:.3f}  "
              f"lr={lr_now:.2e}")

        history.append(dict(
            epoch=epoch, train_loss=train_loss,
            val_mae=val_mae, val_rmse=val_rmse, val_r2=val_r2,
        ))

        # Save best checkpoint
        if val_mae < best_mae:
            best_mae = val_mae
            save_checkpoint(
                dict(epoch=epoch, model=model.state_dict(),
                     optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(),
                     val_mae=val_mae,
                     res_mean=res_mean, res_std=res_std),
                out_dir / "best_checkpoint.pt",
            )

        # Periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(
                dict(epoch=epoch, model=model.state_dict(),
                     optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(),
                     res_mean=res_mean, res_std=res_std),
                out_dir / "checkpoint.pt",
            )

    print(f"\n  Best val MAE : {best_mae:.3f} m  "
          f"(baseline = {res_std:.3f} m)")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, out_dir)

    # ── [6] Final evaluation on all splits ───────────────────────────────
    print("\n[5] Final evaluation (best checkpoint) ...")
    load_checkpoint(out_dir / "best_checkpoint.pt", model)
    model.to(device)

    for split_name, zone_list in [("TRAIN", TRAIN_ZONES),
                                   ("VAL",   VAL_ZONES),
                                   ("TEST",  TEST_ZONES)]:
        print(f"  {split_name}:")
        split_ds = DSMPatchDataset(
            zone_list, data_root, norm_stats,
            patch_size=args.patch_size, stride=args.patch_size,
            augment=False,
        )
        if len(split_ds) == 0:
            print(f"    No patches found")
            continue
        loader = DataLoader(
            split_ds, batch_size=args.batch_size,
            num_workers=2, pin_memory=(device.type == "cuda"),
        )
        m = evaluate(model, loader, device, res_mean, res_std)
        print(f"    MAE={m['mae']:.3f} m  RMSE={m['rmse']:.3f} m  R²={m['r2']:.3f}")

    # ── [7] Visual predictions ────────────────────────────────────────────
    print("\n[6] Saving prediction visualisations ...")
    for split_name, zone_list in [
        ("train", TRAIN_ZONES),
        ("val",   VAL_ZONES),
        ("test",  TEST_ZONES),
    ]:
        viz_ds = DSMPatchDataset(
            zone_list, data_root, norm_stats,
            patch_size=args.patch_size, stride=args.patch_size,
            augment=False,
        )
        if len(viz_ds) == 0:
            print(f"  {split_name}: no patches — skipping")
            continue
        sub_dir = out_dir / "predictions" / split_name
        visualise_predictions(model, viz_ds, device, sub_dir, n_samples=4)
        print(f"  {split_name}: saved to {sub_dir}/")
    print(f"\nDone.  Output dir: {out_dir}/")


if __name__ == "__main__":
    main()
