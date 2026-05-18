"""
4_train_unet.py
===============
U-Net training script for DSM/DTM super-resolution.

Architecture
------------
  UNetDTM (encoder-decoder with skip connections)
    in_channels  = 9  (cop_dem, slope, B02, B03, B04, B08, ndvi, tcd, imd)
    out_channels = 1  (DTM metres) + 2 auxiliary mask channels (built, veg)
    base_filters = 32

Loss
----
  L = L1_height + alpha * BCE_built_veg + beta * Sobel_edge_loss
  alpha = 0.1,  beta = 0.05

Geographic split (from zones.py)
---------------------------------
  TRAIN : caen, grenoble, landes
  VAL   : paris_sud
  TEST  : toulouse  (never seen during training)

Usage
-----
  python 4_train_unet.py                    # default 60 epochs
  python 4_train_unet.py --epochs 30 --batch-size 8
  python 4_train_unet.py --resume output/unet/checkpoint.pt
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
    """Two consecutive  Conv2d -> BatchNorm -> ReLU  blocks."""

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
    """MaxPool2x2 then DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Bilinear upsampling then DoubleConv (skip connection from encoder)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if spatial dims differ by 1 pixel (common with odd input sizes)
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[-1] - x.shape[-1],
                          0, skip.shape[-2] - x.shape[-2]])
        return self.conv(torch.cat([x, skip], dim=1))


class UNetDTM(nn.Module):
    """
    Encoder-decoder U-Net for DTM regression with an auxiliary mask head.

    Parameters
    ----------
    in_channels  : number of input channels (default 9)
    base_filters : number of filters in the first encoder block (doubles
                   at each level, max 8x)
    out_mask_ch  : number of auxiliary binary mask channels (built / veg)
    """

    def __init__(self, in_channels: int = 9,
                 base_filters: int = 32, out_mask_ch: int = 2):
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

        # Regression head: DTM height (one channel, no activation)
        self.head_reg  = nn.Conv2d(f, 1, kernel_size=1)

        # Auxiliary mask head: built / veg (two channels, sigmoid during inference)
        self.head_mask = nn.Conv2d(f, out_mask_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bottleneck(e4)
        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.head_reg(d1), self.head_mask(d1)


# ---------------------------------------------------------------------------
# 2.  Loss functions
# ---------------------------------------------------------------------------

def _sobel_edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Penalise differences in local gradients (sharpness / edge preservation).

    Both pred and target are (B, 1, H, W) tensors.
    """
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=pred.dtype, device=pred.device
    ).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)

    def _grad_mag(t):
        gx = F.conv2d(t, kx, padding=1)
        gy = F.conv2d(t, ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    return F.l1_loss(_grad_mag(pred), _grad_mag(target))


def total_loss(pred_dtm: torch.Tensor, pred_mask: torch.Tensor,
               true_dtm: torch.Tensor, true_mask: "torch.Tensor | None",
               alpha: float = 0.1, beta: float = 0.05) -> torch.Tensor:
    """
    Combined loss:  L1_DTM + alpha * BCE_mask + beta * edge_loss

    Parameters
    ----------
    pred_dtm  : (B, 1, H, W)
    pred_mask : (B, 2, H, W)  -- logits
    true_dtm  : (B, 1, H, W)
    true_mask : (B, 2, H, W) binary, or None if not available
    """
    l1   = F.l1_loss(pred_dtm, true_dtm)
    edge = _sobel_edge_loss(pred_dtm, true_dtm)

    if true_mask is not None:
        bce = F.binary_cross_entropy_with_logits(pred_mask, true_mask.float())
        return l1 + alpha * bce + beta * edge
    else:
        return l1 + beta * edge


# ---------------------------------------------------------------------------
# 3.  Training loop
# ---------------------------------------------------------------------------

def _build_mask_proxy(y_batch: torch.Tensor,
                      tcd: torch.Tensor,
                      imd: torch.Tensor) -> torch.Tensor:
    """
    Derive a crude built/veg binary mask from TCD/IMD channel values.

    Returns (B, 2, H, W) float32 tensor (not normalised -- channel indices
    from the normalised input tensor are passed in).
    The thresholds are approximate (TCD > 0.25 -> veg, IMD > 0.25 -> built).
    """
    veg   = (tcd > 0.0).float()   # TCD proxy > 0 after norm -> vegetated
    built = (imd > 0.0).float()   # IMD proxy > 0 after norm -> impervious
    return torch.stack([built, veg], dim=1)   # (B, 2, H, W)


def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        # Derive crude mask proxy from TCD (ch7) and IMD (ch8) inputs
        tcd_ch = x[:, 7:8, :, :]
        imd_ch = x[:, 8:9, :, :]
        true_mask = _build_mask_proxy(y, tcd_ch, imd_ch)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                pred_dtm, pred_mask = model(x)
                loss = total_loss(pred_dtm, pred_mask, y, true_mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_dtm, pred_mask = model(x)
            loss = total_loss(pred_dtm, pred_mask, y, true_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred_dtm, _ = model(x)
        preds.append(pred_dtm.cpu().numpy())
        trues.append(y.cpu().numpy())

    preds = np.concatenate([p.ravel() for p in preds])
    trues = np.concatenate([t.ravel() for t in trues])

    mae  = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    r2   = float(1 - np.sum((preds - trues) ** 2) / (np.sum((trues - trues.mean()) ** 2) + 1e-8))
    return dict(mae=mae, rmse=rmse, r2=r2)


# ---------------------------------------------------------------------------
# 4.  Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None) -> int:
    state  = torch.load(path, map_location="cpu")
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
    """Plot train/val loss and MAE curves."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Training history", fontsize=11)

    for key, ax, title in [
        ("train_loss", axes[0], "Loss"),
        ("val_mae",    axes[1], "Val MAE (m)"),
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
def visualise_predictions(model, dataset, device, out_dir: Path,
                          n_samples: int = 3) -> None:
    """
    Plot input / predicted DTM / true DTM side-by-side for a few patches.
    """
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)

    for k, idx in enumerate(indices):
        x, y = dataset[idx]
        x_in = x.unsqueeze(0).to(device)
        pred_dtm, _ = model(x_in)
        pred = pred_dtm.squeeze().cpu().numpy()
        true = y.squeeze().numpy()
        dem  = x[0].numpy()   # COP-DEM (normalised)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, arr, title, cmap in [
            (axes[0], dem,  "COP-DEM 30m input (norm)", "terrain"),
            (axes[1], pred, "U-Net prediction (m)",     "terrain"),
            (axes[2], true, "Lidar HD DTM (m)",          "terrain"),
        ]:
            valid = arr[np.isfinite(arr)]
            vmin  = float(np.percentile(valid, 2))  if valid.size else 0
            vmax  = float(np.percentile(valid, 98)) if valid.size else 1
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        mae_patch = float(np.mean(np.abs(pred - true)))
        fig.suptitle(f"Sample {k+1}  MAE = {mae_patch:.2f} m", fontsize=10)
        plt.tight_layout()
        fig.savefig(out_dir / f"pred_sample_{k+1}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# 6.  Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train U-Net for Lidar HD DTM estimation from multi-source inputs."
    )
    parser.add_argument("--epochs",      type=int, default=60)
    parser.add_argument("--batch-size",  type=int, default=4)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--base-filters",type=int, default=32)
    parser.add_argument("--patch-size",  type=int, default=256)
    parser.add_argument("--stride",      type=int, default=128)
    parser.add_argument("--resume",      type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--output-dir",  type=str, default=None)
    return parser.parse_args()


def main():
    from zones import TRAIN_ZONES, VAL_ZONES, TEST_ZONES
    from dataset_unet import (
        DSMPatchDataset, compute_normalization_stats, load_normalization_stats
    )

    args      = parse_args()
    data_root = Path(__file__).resolve().parent.parent / "data"
    out_dir   = (Path(args.output_dir) if args.output_dir
                 else Path(__file__).resolve().parent.parent / "output" / "unet")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nU-Net DTM training")
    print(f"  Device      : {device}")
    print(f"  Train zones : {TRAIN_ZONES}")
    print(f"  Val zones   : {VAL_ZONES}")
    print(f"  Test zones  : {TEST_ZONES}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Patch size  : {args.patch_size}")

    # ---- Normalization stats ----
    stats_path = data_root / "norm_stats.json"
    if stats_path.exists():
        print("\n[1] Loading existing normalization stats ...")
        norm_stats = load_normalization_stats(data_root)
    else:
        print("\n[1] Computing normalization stats from train zones ...")
        norm_stats = compute_normalization_stats(TRAIN_ZONES, data_root)

    # ---- Datasets ----
    print("\n[2] Building datasets ...")
    train_ds = DSMPatchDataset(
        TRAIN_ZONES, data_root, norm_stats,
        patch_size=args.patch_size, stride=args.stride,
        augment=True,
    )
    val_ds = DSMPatchDataset(
        VAL_ZONES, data_root, norm_stats,
        patch_size=args.patch_size, stride=args.patch_size,   # no overlap for val
        augment=False,
    )

    if len(train_ds) == 0:
        print("ERROR: no training patches found.  "
              "Run 0_download_data.py and 1_prepare.py first.")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    # ---- Model ----
    print("\n[3] Building model ...")
    from dataset_unet import N_CHANNELS
    model = UNetDTM(in_channels=N_CHANNELS, base_filters=args.base_filters)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # ---- Optimiser and scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    # Optional AMP (CUDA only)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            start_epoch = load_checkpoint(resume_path, model, optimizer, scheduler)
            print(f"  Resumed from epoch {start_epoch}: {resume_path}")
        else:
            print(f"  WARNING: checkpoint {resume_path} not found -- starting fresh")

    # ---- Training loop ----
    print(f"\n[4] Training for {args.epochs} epochs ...")
    history  = []
    best_mae = float("inf")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        val_mae  = val_metrics["mae"]
        val_rmse = val_metrics["rmse"]
        val_r2   = val_metrics["r2"]
        lr_now   = scheduler.get_last_lr()[0]

        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"loss={train_loss:.4f}  "
              f"val_MAE={val_mae:.3f} m  RMSE={val_rmse:.3f}  R2={val_r2:.3f}  "
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
                     val_mae=val_mae),
                out_dir / "best_checkpoint.pt",
            )

        # Regular checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(
                dict(epoch=epoch, model=model.state_dict(),
                     optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict()),
                out_dir / "checkpoint.pt",
            )

    print(f"\n  Best val MAE: {best_mae:.3f} m")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, out_dir)

    # ---- Final evaluation on all splits ----
    print("\n[5] Final evaluation ...")
    load_checkpoint(out_dir / "best_checkpoint.pt", model)
    model.to(device)

    for split_name, zone_list in [("TRAIN", TRAIN_ZONES),
                                   ("VAL",   VAL_ZONES),
                                   ("TEST",  TEST_ZONES)]:
        split_ds = DSMPatchDataset(
            zone_list, data_root, norm_stats,
            patch_size=args.patch_size, stride=args.patch_size,
            augment=False,
        )
        if len(split_ds) == 0:
            continue
        loader = DataLoader(split_ds, batch_size=args.batch_size,
                            num_workers=2, pin_memory=(device.type == "cuda"))
        m = evaluate(model, loader, device)
        print(f"  [{split_name}]  MAE={m['mae']:.3f} m  RMSE={m['rmse']:.3f}  R2={m['r2']:.3f}")

    # ---- Visual predictions ----
    print("\n[6] Saving prediction visualisations ...")
    visualise_predictions(model, val_ds, device, out_dir / "predictions", n_samples=4)
    print(f"  Saved to {out_dir}/predictions/")
    print("\nDone.")


if __name__ == "__main__":
    main()
