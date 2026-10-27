"""
Step 4: Comprehensive Evaluation & Harmonic Verification Suite for ARCNet-v2.

Evaluates a trained ARCNet-v2 checkpoint (`checkpoints/arcnet_v2_base_best.pt`) on:
1. Quantitative Metrics: MAE, MSE, PSNR (dB), and Perceptual Delta-E (Oklab).
2. Harmonic Palette Verification:
   - Measures angular alignment on the 360° Oklch hue wheel against classical
     Itten harmonic sectors (Complementary 180°, Split-Complementary 150°,
     Triadic 120°, Analogous 30°, Monochromatic 0°).
   - Reports exact sector distribution (%) and mean angular deviation (°).
3. Distinguishability / Anti-Camouflage Verification:
   - Reports percentage of predictions satisfying perceptual contrast dE_Oklab >= gamma.
4. Real-Time AR Inference Latency Benchmark:
   - Measures single-frame latency (ms) to verify the <= 3 ms mobile AR target.
5. Temporal EMA Filter Verification (`ARCNet_V2_roadmap.pdf` Eq. 7):
   - Simulates camera sensor noise / frame-to-frame jitter and measures chromatic
     flicker reduction achieved by the EMA filter (alpha in [0.15, 0.25]).
"""

import argparse
import math
from pathlib import Path
import time
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F

from arcnet_v2 import PRESETS, count_parameters
from color_space import oklab_to_oklch
from dataset import create_dataloaders
from losses import ARCNetV2Loss
from train import ARCNetV1Baseline


SECTOR_NAMES = [
    "Monochromatic (0°)",
    "Analogous (30°)",
    "Triadic (120°)",
    "Split-Complementary (150°)",
    "Complementary (180°)",
]
SECTOR_DEGREES = [0.0, 30.0, 120.0, 150.0, 180.0]


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str = "checkpoints/arcnet_v2_base_best.pt",
    data_path: str = "data/arcnet_v2_palettes.npz",
    batch_size: int = 256,
    ema_alpha: float = 0.20,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        # Auto-discover most recent *_best.pt in checkpoints/
        candidates = sorted(
            Path("checkpoints").glob("arcnet_v2_*_best.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            ckpt_file = candidates[0]
            print(f"[Auto-Detected Checkpoint] Using: {ckpt_file}")
        else:
            raise FileNotFoundError(
                f"Checkpoint not found at {ckpt_file}. Run `python3 train.py` first."
            )

    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    preset = ckpt.get("preset", "base")
    output_mode = ckpt.get("output_mode", "single_color")

    if preset == "v1_baseline":
        model = ARCNetV1Baseline().to(device)
    else:
        model = PRESETS[preset](output_mode=output_mode).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, val_loader = create_dataloaders(
        npz_path=data_path, batch_size=batch_size, min_k=4, num_workers=2
    )
    criterion = ARCNetV2Loss().to(device)

    all_mae = []
    all_mse = []
    all_de = []
    all_deg_err = []
    all_contrast_ok = []
    sector_counts = np.zeros(len(SECTOR_DEGREES), dtype=np.int64)

    sectors_tensor = torch.tensor(SECTOR_DEGREES, device=device, dtype=torch.float32)

    for batch in val_loader:
        bg_srgb = batch["bg_srgb"].to(device)
        bg_weights = batch["bg_weights"].to(device)
        mask = batch["mask"].to(device)
        target_srgb = batch["target_srgb"].to(device)

        pred, aux = model(bg_srgb, bg_weights, mask=mask, return_aux=True)
        pred_srgb = pred[:, :3]
        pred_oklab = aux["pred_primary_oklab"]
        bg_oklab = aux["bg_oklab"]

        _, metrics = criterion(
            pred_primary_oklab=pred_oklab,
            target_color=target_srgb,
            bg_oklab=bg_oklab,
            bg_weights=bg_weights,
            expert_weights=aux["expert_weights"],
        )

        mae = torch.mean(torch.abs(pred_srgb - target_srgb), dim=-1)
        mse = torch.mean((pred_srgb - target_srgb) ** 2, dim=-1)
        target_oklab = batch["target_oklab"].to(device)
        de = torch.sqrt(torch.sum((pred_oklab - target_oklab) ** 2, dim=-1) + 1e-6)

        # Hue wheel harmonic verification
        pred_lch = oklab_to_oklch(pred_oklab)
        dom_bg_lch = oklab_to_oklch(bg_oklab[:, 0, :])
        delta_hue_deg = torch.abs(pred_lch[..., 2] - dom_bg_lch[..., 2]) * (180.0 / math.pi)
        delta_hue_deg = torch.minimum(delta_hue_deg, 360.0 - delta_hue_deg)

        diffs = torch.abs(delta_hue_deg.unsqueeze(-1) - sectors_tensor.unsqueeze(0))
        min_err, nearest_sector = torch.min(diffs, dim=-1)

        for s_idx in nearest_sector.cpu().numpy():
            sector_counts[int(s_idx)] += 1

        # Anti-camouflage contrast verification against dominant background
        bg_de = torch.sqrt(torch.sum((pred_oklab - bg_oklab[:, 0, :]) ** 2, dim=-1) + 1e-6)
        contrast_ok = (bg_de >= criterion.contrast_margin_gamma).float()

        all_mae.append(mae.cpu().numpy())
        all_mse.append(mse.cpu().numpy())
        all_de.append(de.cpu().numpy())
        all_deg_err.append(min_err.cpu().numpy())
        all_contrast_ok.append(contrast_ok.cpu().numpy())

    mae_val = float(np.concatenate(all_mae).mean())
    mse_val = float(np.concatenate(all_mse).mean())
    psnr_val = 10.0 * math.log10(1.0 / max(mse_val, 1e-10))
    de_val = float(np.concatenate(all_de).mean())
    deg_err_val = float(np.concatenate(all_deg_err).mean())
    contrast_pct = float(np.concatenate(all_contrast_ok).mean() * 100.0)

    # Measure single-frame AR inference latency (batch_size = 1, K-1 = 7 clusters)
    dummy_bg = torch.rand(1, 7, 3, device=device)
    dummy_w = torch.full((1, 7, 1), 1.0 / 7.0, device=device)
    for _ in range(20):
        _ = model(dummy_bg, dummy_w)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 200
    for _ in range(iters):
        _ = model(dummy_bg, dummy_w)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    # Measure Temporal Flicker Reduction under simulated camera jitter (Roadmap Eq. 7 & 8)
    base_palette = dummy_bg.repeat(60, 1, 1)
    sensor_noise = torch.randn_like(base_palette) * 0.03
    noisy_stream = torch.clamp(base_palette + sensor_noise, 0.0, 1.0)
    raw_preds = model(noisy_stream, dummy_w.repeat(60, 1, 1))[:, :3]

    ema_preds = torch.zeros_like(raw_preds)
    ema_preds[0] = raw_preds[0]
    for t in range(1, 60):
        ema_preds[t] = ema_alpha * raw_preds[t] + (1.0 - ema_alpha) * ema_preds[t - 1]

    flicker_raw = float(torch.mean(torch.abs(raw_preds[1:] - raw_preds[:-1])).item())
    flicker_ema = float(torch.mean(torch.abs(ema_preds[1:] - ema_preds[:-1])).item())
    flicker_reduction_pct = (1.0 - (flicker_ema / max(flicker_raw, 1e-8))) * 100.0

    total_samples = int(sector_counts.sum())
    print("\n============================================================")
    print(f" ARCNet-v2 Evaluation Report ({preset}, {count_parameters(model):,d} params)")
    print("============================================================")
    print(f"  Validation Samples          : {total_samples:,d}")
    print(f"  sRGB MAE (↓)                : {mae_val:.4f}")
    print(f"  sRGB MSE (↓)                : {mse_val:.4f}")
    print(f"  PSNR (↑)                    : {psnr_val:.2f} dB")
    print(f"  Perceptual dE_Oklab (↓)     : {de_val:.4f}")
    print(f"  Itten Harmonic Angle Err (↓): {deg_err_val:.2f}°")
    print(f"  Anti-Camouflage Contrast (↑): {contrast_pct:.1f}%")
    print(f"  Single-Frame Latency        : {latency_ms:.3f} ms/frame (Target <= 3.0 ms)")
    print(
        f"  Temporal Flicker (E_temp)   : Raw={flicker_raw:.4f} -> EMA={flicker_ema:.4f} "
        f"(-{flicker_reduction_pct:.1f}% flicker)"
    )
    print("\n--- Predicted Palette Distribution Across Itten Harmonic Sectors ---")
    for name, count in zip(SECTOR_NAMES, sector_counts):
        pct = (count / max(1, total_samples)) * 100.0
        print(f"  {name:28s}: {count:6,d} ({pct:5.1f}%)")
    print("============================================================\n")

    return {
        "mae": mae_val,
        "mse": mse_val,
        "psnr": psnr_val,
        "delta_e_oklab": de_val,
        "harmonic_deg_err": deg_err_val,
        "contrast_pct": contrast_pct,
        "latency_ms": latency_ms,
        "flicker_reduction_pct": flicker_reduction_pct,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: Evaluate Trained ARCNet-v2 Checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/arcnet_v2_base_best.pt",
        help="Path to trained checkpoint .pt file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/arcnet_v2_palettes.npz",
        help="Path to preprocessed .npz file",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ema-alpha", type=float, default=0.20, help="EMA smoothing factor alpha in [0.15, 0.25]")
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        batch_size=args.batch_size,
        ema_alpha=args.ema_alpha,
    )
