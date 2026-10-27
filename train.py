"""
Step 3: High-Capacity Training & Ablation Pipeline for ARCNet-v2.

Features:
---------
1. Zero-argument turnkey execution:
   Running `python3 train.py` automatically loads `data/arcnet_v2_palettes.npz`,
   trains `arcnet_v2_base` with `ARCNetV2Loss`, saves the best checkpoint to
   `checkpoints/arcnet_v2_best.pt`, and exports ONNX models for Unity Sentis/Barracuda.
2. Anti-Underfitting Optimization:
   - Parameter-group decoupled AdamW (no weight decay on LayerNorm, LayerScale, or
     learnable Fourier frequencies).
   - Linear Warmup + Cosine Annealing LR scheduler.
   - Exponential Moving Average (EMA) model weights for smoother generalization.
   - Support for both 'single_color' (3D) and 'multi_material' (10D PBR) supervision.
3. Built-in Baseline Comparison (`--compare-v1`):
   Includes the original ARCNet-v1 MLP (6 -> 64 -> 32 -> 3 from DL_based_harmonization.pdf)
   so you can benchmark ARCNet-v1 vs. ARCNet-v2 on the exact same large dataset.
"""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from arcnet_v2 import (
    ARCNetV2,
    arcnet_v2_base,
    arcnet_v2_large,
    arcnet_v2_mobile,
    arcnet_v2_xl,
    count_parameters,
)
from color_space import oklab_to_oklch, oklab_to_srgb, srgb_to_oklab
from dataset import create_dataloaders
from losses import ARCNetV2Loss


PRESETS = {
    "mobile": arcnet_v2_mobile,
    "base": arcnet_v2_base,
    "large": arcnet_v2_large,
    "xl": arcnet_v2_xl,
}


# =====================================================================
# Original ARCNet-v1 Baseline (for Direct Ablation Comparison)
# =====================================================================

class ARCNetV1Baseline(nn.Module):
    """
    Original ARCNet-v1 architecture from `DL_based_harmonization.pdf` (Section II-A, Table I M6):
        Input(6) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(3, ELU(alpha=0.1))
    Takes the first 2 dominant sRGB colors (6 floats) and predicts the 3rd sRGB color.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.ELU(alpha=0.1),
        )

    def forward(
        self,
        colors: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        # Take top 2 dominant background colors -> [B, 6]
        if colors.dim() == 3:
            x = colors[:, :2, :3].reshape(colors.shape[0], 6)
            bg_oklab = srgb_to_oklab(colors[..., :3])
        else:
            x = colors[:, :6]
            bg_oklab = srgb_to_oklab(colors[:, :6].reshape(-1, 2, 3))

        pred_srgb = torch.clamp(self.net(x), 0.0, 1.0)
        if not return_aux:
            return pred_srgb

        pred_oklab = srgb_to_oklab(pred_srgb)
        aux = {
            "pred_primary_oklab": pred_oklab,
            "bg_oklab": bg_oklab,
            "bg_weights": weights,
            "expert_weights": None,
        }
        return pred_srgb, aux


# =====================================================================
# Exponential Moving Average (EMA) Helper
# =====================================================================

class ModelEMA:
    """Maintains an exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema_model = deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, ema_v in self.ema_model.state_dict().items():
            model_v = msd[k].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)
            else:
                ema_v.copy_(model_v)


# =====================================================================
# Optimizer & Scheduler Construction
# =====================================================================

def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    epochs: int = 100,
    steps_per_epoch: int = 100,
    warmup_epochs: int = 5,
):
    """
    Separates parameters into decay (Linear weights) and no-decay (biases, LayerNorm,
    LayerScale, Fourier frequencies, attention queries) to prevent underfitting.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim < 2
            or "norm" in name
            or "bias" in name
            or "layer_scale" in name
            or "fourier_freqs" in name
            or "queries" in name
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.999))

    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


# =====================================================================
# Training & Validation Loops
# =====================================================================

@torch.no_grad()
def compute_harmonic_alignment_metrics(
    pred_oklab: torch.Tensor,
    bg_oklab: torch.Tensor,
    contrast_margin_gamma: float = 0.15,
) -> Tuple[float, float]:
    """
    Computes:
      1. Mean angular distance (in degrees) from predicted hue to nearest classical
         Itten harmonic angle [0°, 30°, 120°, 150°, 180°] relative to dominant bg hue.
      2. Distinguishability rate (% of samples with dE_Oklab >= gamma from dominant bg).
    """
    pred_lch = oklab_to_oklch(pred_oklab)
    dom_bg_lch = oklab_to_oklch(bg_oklab[:, 0, :])

    delta_hue_deg = torch.abs(pred_lch[..., 2] - dom_bg_lch[..., 2]) * (180.0 / math.pi)
    delta_hue_deg = torch.minimum(delta_hue_deg, 360.0 - delta_hue_deg)  # [B] in [0, 180]

    itten_sectors = torch.tensor(
        [0.0, 30.0, 120.0, 150.0, 180.0],
        device=pred_oklab.device,
        dtype=pred_oklab.dtype,
    )
    angular_diffs = torch.abs(delta_hue_deg.unsqueeze(-1) - itten_sectors.unsqueeze(0))
    min_deg_err, _ = torch.min(angular_diffs, dim=-1)
    mean_deg_err = float(min_deg_err.mean().item())

    # Contrast rate against top dominant background cluster
    bg_diff = pred_oklab - bg_oklab[:, 0, :]
    de_dom = torch.sqrt(torch.sum(bg_diff * bg_diff, dim=-1) + 1e-6)
    distinguishable_rate = float((de_dom >= contrast_margin_gamma).float().mean().item() * 100.0)

    return mean_deg_err, distinguishable_rate


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: ARCNetV2Loss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    output_mode: str = "single_color",
    ema: Optional[ModelEMA] = None,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    running = {
        "loss_total": 0.0,
        "loss_recon": 0.0,
        "loss_dist": 0.0,
        "loss_harm": 0.0,
        "delta_e_oklab": 0.0,
    }
    num_batches = 0

    for batch in loader:
        bg_srgb = batch["bg_srgb"].to(device, non_blocking=True)
        bg_weights = batch["bg_weights"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        target_srgb = batch["target_srgb"].to(device, non_blocking=True)
        pbr_target = batch["pbr_target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        pred, aux = model(bg_srgb, bg_weights, mask=mask, return_aux=True)

        loss, metrics = criterion(
            pred_primary_oklab=aux["pred_primary_oklab"],
            target_color=target_srgb,
            bg_oklab=aux["bg_oklab"],
            bg_weights=bg_weights,
            expert_weights=aux["expert_weights"],
        )

        # If training in multi_material PBR mode, add auxiliary supervision for
        # secondary color, specular shift, and roughness (cols 3:10)
        if output_mode == "multi_material" and pred.shape[-1] == 10:
            pbr_aux_loss = F.smooth_l1_loss(pred[:, 3:], pbr_target[:, 3:])
            loss = loss + 0.25 * pbr_aux_loss

        loss.backward()
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        if ema is not None:
            ema.update(model)

        for k in running:
            running[k] += float(metrics[k].item())
        num_batches += 1

    return {k: v / max(1, num_batches) for k, v in running.items()}


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: ARCNetV2Loss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    totals = {
        "loss_total": 0.0,
        "loss_recon": 0.0,
        "loss_dist": 0.0,
        "loss_harm": 0.0,
        "delta_e_oklab": 0.0,
        "mae_srgb": 0.0,
        "mse_srgb": 0.0,
        "psnr_db": 0.0,
        "harmonic_deg_err": 0.0,
        "distinguishable_pct": 0.0,
    }
    num_batches = 0

    for batch in loader:
        bg_srgb = batch["bg_srgb"].to(device, non_blocking=True)
        bg_weights = batch["bg_weights"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        target_srgb = batch["target_srgb"].to(device, non_blocking=True)

        pred, aux = model(bg_srgb, bg_weights, mask=mask, return_aux=True)
        pred_primary_srgb = pred[:, :3]

        _, metrics = criterion(
            pred_primary_oklab=aux["pred_primary_oklab"],
            target_color=target_srgb,
            bg_oklab=aux["bg_oklab"],
            bg_weights=bg_weights,
            expert_weights=aux["expert_weights"],
        )

        mae = F.l1_loss(pred_primary_srgb, target_srgb).item()
        mse = F.mse_loss(pred_primary_srgb, target_srgb).item()
        psnr = 10.0 * math.log10(1.0 / max(mse, 1e-10))

        deg_err, dist_pct = compute_harmonic_alignment_metrics(
            pred_oklab=aux["pred_primary_oklab"],
            bg_oklab=aux["bg_oklab"],
            contrast_margin_gamma=criterion.contrast_margin_gamma,
        )

        for k in ("loss_total", "loss_recon", "loss_dist", "loss_harm", "delta_e_oklab"):
            totals[k] += float(metrics[k].item())
        totals["mae_srgb"] += mae
        totals["mse_srgb"] += mse
        totals["psnr_db"] += psnr
        totals["harmonic_deg_err"] += deg_err
        totals["distinguishable_pct"] += dist_pct
        num_batches += 1

    return {k: v / max(1, num_batches) for k, v in totals.items()}


# =====================================================================
# Main Training Orchestrator
# =====================================================================

def train_arcnet_v2(
    data_path: str = "data/arcnet_v2_palettes.npz",
    preset: str = "base",
    output_mode: str = "single_color",
    epochs: int = 80,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    save_dir: str = "checkpoints",
    num_workers: int = 4,
    export_onnx_on_finish: bool = True,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"=== ARCNet-v2 Training Pipeline ===")
    print(f"  Device      : {device}")
    print(f"  Dataset     : {data_path}")
    print(f"  Preset      : {preset} (output_mode={output_mode})")

    train_loader, val_loader = create_dataloaders(
        npz_path=data_path,
        batch_size=batch_size,
        min_k=4,
        num_workers=num_workers,
    )
    print(f"  Train samples: {len(train_loader.dataset):,d} | Val samples: {len(val_loader.dataset):,d}")

    if preset == "v1_baseline":
        model: nn.Module = ARCNetV1Baseline().to(device)
    else:
        model = PRESETS[preset](output_mode=output_mode).to(device)

    num_params = count_parameters(model)
    print(f"  Parameters  : {num_params:,d}")

    criterion = ARCNetV2Loss(
        lambda_dist=0.15,
        lambda_harm=0.05,
        lambda_chroma=0.20,
        lambda_balance=0.02,
    ).to(device)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=max(2, epochs // 15),
    )
    ema = ModelEMA(model, decay=0.995)

    best_val_loss = float("inf")
    best_metrics: Dict[str, float] = {}
    history: List[Dict[str, float]] = []

    ckpt_best = save_path / f"arcnet_v2_{preset}_best.pt"
    ckpt_last = save_path / f"arcnet_v2_{preset}_latest.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_mode=output_mode,
            ema=ema,
        )
        val_metrics = evaluate_model(
            model=ema.ema_model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "lr": cur_lr,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)

        is_best = val_metrics["loss_total"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss_total"]
            best_metrics = val_metrics
            torch.save(
                {
                    "epoch": epoch,
                    "preset": preset,
                    "output_mode": output_mode,
                    "model_state_dict": ema.ema_model.state_dict(),
                    "val_metrics": val_metrics,
                },
                ckpt_best,
            )

        torch.save(
            {
                "epoch": epoch,
                "preset": preset,
                "output_mode": output_mode,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.ema_model.state_dict(),
                "val_metrics": val_metrics,
            },
            ckpt_last,
        )

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs or is_best:
            star = "*" if is_best else " "
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}]{star} ({elapsed:.1f}s) | "
                f"Train Loss: {train_metrics['loss_total']:.4f} | "
                f"Val Loss: {val_metrics['loss_total']:.4f} | "
                f"dE_Oklab: {val_metrics['delta_e_oklab']:.4f} | "
                f"MAE: {val_metrics['mae_srgb']:.4f} | "
                f"PSNR: {val_metrics['psnr_db']:.2f} dB | "
                f"HarmErr: {val_metrics['harmonic_deg_err']:.1f}° | "
                f"Contrast: {val_metrics['distinguishable_pct']:.1f}%"
            )

    history_file = save_path / f"history_{preset}.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\n=== Training Complete ({preset}) ===")
    print(f"  Best Checkpoint : {ckpt_best}")
    print(
        f"  Best Val Metrics: Loss={best_metrics['loss_total']:.4f}, "
        f"dE_Oklab={best_metrics['delta_e_oklab']:.4f}, "
        f"MAE={best_metrics['mae_srgb']:.4f}, "
        f"PSNR={best_metrics['psnr_db']:.2f} dB, "
        f"Harmonic Alignment Error={best_metrics['harmonic_deg_err']:.2f}°"
    )

    # Automatically export best trained model to ONNX for Unity Sentis / Barracuda
    if export_onnx_on_finish and preset != "v1_baseline":
        try:
            best_ckpt = torch.load(ckpt_best, map_location="cpu", weights_only=False)
            export_model = PRESETS[preset](output_mode=output_mode).eval()
            export_model.load_state_dict(best_ckpt["model_state_dict"])

            # 1. Dynamic Set ONNX [batch_size, num_clusters, 4] for Unity Sentis ARCNet-v2
            onnx_set_path = save_path / f"arcnet_v2_{preset}_set.onnx"
            dummy_set = torch.rand(1, 7, 4, dtype=torch.float32)
            torch.onnx.export(
                export_model,
                (dummy_set,),
                str(onnx_set_path),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size", 1: "num_clusters"},
                    "output": {0: "batch_size"},
                },
            )

            # 2. Legacy [batch_size, 6] ONNX for drop-in replacement in existing RunMLP.cs
            onnx_flat6_path = save_path / f"arcnet_v2_{preset}_flat6.onnx"
            dummy_flat6 = torch.rand(1, 6, dtype=torch.float32)
            torch.onnx.export(
                export_model,
                (dummy_flat6,),
                str(onnx_flat6_path),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
            print(f"  Exported ONNX (Set [B,K-1,4]) : {onnx_set_path}")
            print(f"  Exported ONNX (Flat [B,6])    : {onnx_flat6_path}")
        except Exception as e:
            print(f"  [ONNX Export Note] Skipped automatic ONNX export: {e}")

    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Train ARCNet-v2 on Preprocessed Palette Dataset")
    parser.add_argument(
        "--data",
        type=str,
        default="data/arcnet_v2_palettes.npz",
        help="Path to preprocessed .npz palette file (default: data/arcnet_v2_palettes.npz)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="base",
        choices=["mobile", "base", "large", "xl", "v1_baseline"],
        help="Model capacity preset (default: base)",
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        default="single_color",
        choices=["single_color", "multi_material"],
        help="Output head mode: 'single_color' (3D) or 'multi_material' (10D PBR)",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs (default: 80)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate (default: 3e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay (default: 1e-2)")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints & ONNX")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument(
        "--compare-v1",
        action="store_true",
        help="Run head-to-head ablation comparing ARCNet-v1 vs ARCNet-v2, saving tables and plots",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="When using --compare-v1, force retraining ARCNet-v2 even if its checkpoint already exists",
    )
    args = parser.parse_args()

    if args.compare_v1:
        from ablation_plot import plot_ablation_results, save_ablation_tables

        v2_preset = args.preset if args.preset != "v1_baseline" else "base"
        save_path = Path(args.save_dir)
        ckpt_v1_path = save_path / "arcnet_v2_v1_baseline_best.pt"
        ckpt_v2_path = save_path / f"arcnet_v2_{v2_preset}_best.pt"

        print("=== Running Head-to-Head Underfitting Ablation Study: ARCNet-v1 vs. ARCNet-v2 ===")

        # 1. Train or load ARCNet-v1 (Original 2,339-param MLP)
        if ckpt_v1_path.exists() and not args.force_retrain:
            print(f"[Reusing Existing ARCNet-v1 Checkpoint]: {ckpt_v1_path}")
            c1 = torch.load(ckpt_v1_path, map_location="cpu", weights_only=False)
            m_v1 = c1["val_metrics"]
        else:
            m_v1 = train_arcnet_v2(
                data_path=args.data,
                preset="v1_baseline",
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                save_dir=args.save_dir,
                num_workers=args.workers,
                export_onnx_on_finish=False,
            )

        # 2. Reuse already-trained ARCNet-v2 checkpoint if available, otherwise train it
        if ckpt_v2_path.exists() and not args.force_retrain:
            print(f"\n[Reusing Existing ARCNet-v2 ({v2_preset}) Checkpoint]: {ckpt_v2_path}")
            c2 = torch.load(ckpt_v2_path, map_location="cpu", weights_only=False)
            m_v2 = c2["val_metrics"]
        else:
            m_v2 = train_arcnet_v2(
                data_path=args.data,
                preset=v2_preset,
                output_mode=args.output_mode,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                save_dir=args.save_dir,
                num_workers=args.workers,
            )

        params_v1 = count_parameters(ARCNetV1Baseline())
        params_v2 = count_parameters(PRESETS[v2_preset](output_mode=args.output_mode))

        print("\n================ ABLATION SUMMARY ================")
        print(f"Metric                 | ARCNet-v1 ({params_v1:,d}) | ARCNet-v2 ({v2_preset}, {params_v2:,d})")
        print("--------------------------------------------------")
        print(f"Val Total Loss (↓)     | {m_v1['loss_total']:16.4f} | {m_v2['loss_total']:.4f}")
        print(f"Delta-E Oklab (↓)      | {m_v1['delta_e_oklab']:16.4f} | {m_v2['delta_e_oklab']:.4f}")
        print(f"sRGB MAE (↓)           | {m_v1['mae_srgb']:16.4f} | {m_v2['mae_srgb']:.4f}")
        print(f"sRGB MSE (↓)           | {m_v1['mse_srgb']:16.4f} | {m_v2['mse_srgb']:.4f}")
        print(f"PSNR dB (↑)            | {m_v1['psnr_db']:16.2f} | {m_v2['psnr_db']:.2f}")
        print(f"Harmonic Angle Err (↓) | {m_v1['harmonic_deg_err']:15.2f}° | {m_v2['harmonic_deg_err']:.2f}°")
        print(f"Contrast Rate (↑)      | {m_v1['distinguishable_pct']:15.1f}% | {m_v2['distinguishable_pct']:.1f}%")
        print("==================================================")

        hist_v1_file = save_path / "history_v1_baseline.json"
        hist_v2_file = save_path / f"history_{v2_preset}.json"
        hist_v1 = json.loads(hist_v1_file.read_text(encoding="utf-8")) if hist_v1_file.exists() else None
        hist_v2 = json.loads(hist_v2_file.read_text(encoding="utf-8")) if hist_v2_file.exists() else None

        save_ablation_tables(
            metrics_v1=m_v1,
            metrics_v2=m_v2,
            params_v1=params_v1,
            params_v2=params_v2,
            preset_v2=v2_preset,
            save_dir=args.save_dir,
        )
        plot_ablation_results(
            history_v1=hist_v1,
            history_v2=hist_v2,
            metrics_v1=m_v1,
            metrics_v2=m_v2,
            preset_v2=v2_preset,
            save_dir=args.save_dir,
        )
    else:
        train_arcnet_v2(
            data_path=args.data,
            preset=args.preset,
            output_mode=args.output_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            save_dir=args.save_dir,
            num_workers=args.workers,
        )

