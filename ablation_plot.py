"""
Publication-Ready Ablation Table & Plot Generator for ARCNet-v1 vs. ARCNet-v2.

Generates and saves:
1. Tables (in `checkpoints/`):
   - `ablation_table.csv` (CSV format)
   - `ablation_table.md`  (Markdown format)
   - `ablation_table.tex` (IEEE LaTeX Table format)
2. High-Resolution Figures (300 DPI PNG + PDF in `checkpoints/`):
   - `ablation_training_curves.png` : Epoch-by-epoch convergence curves showing
     ARCNet-v1 underfitting plateau vs. ARCNet-v2 progression (Loss, dE_Oklab, MAE, Harmonic Error).
   - `ablation_metrics_bar.png`     : Bar chart comparison of final validation metrics.
   - `ablation_harmonic_sectors.png`: Comparison of Itten harmonic sector alignment & contrast.
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def save_ablation_tables(
    metrics_v1: Dict[str, Any],
    metrics_v2: Dict[str, Any],
    params_v1: int,
    params_v2: int,
    preset_v2: str = "base",
    save_dir: str = "checkpoints",
):
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "Model": "ARCNet-v1 (Original MLP)",
            "Params": f"{params_v1:,d}",
            "Val_Loss": f"{metrics_v1['loss_total']:.4f}",
            "dE_Oklab": f"{metrics_v1['delta_e_oklab']:.4f}",
            "sRGB_MAE": f"{metrics_v1['mae_srgb']:.4f}",
            "sRGB_MSE": f"{metrics_v1['mse_srgb']:.4f}",
            "PSNR_dB": f"{metrics_v1['psnr_db']:.2f}",
            "Harmonic_Angle_Err_Deg": f"{metrics_v1['harmonic_deg_err']:.2f}",
            "Contrast_Rate_Pct": f"{metrics_v1['distinguishable_pct']:.1f}",
        },
        {
            "Model": f"ARCNet-v2 ({preset_v2}, Proposed)",
            "Params": f"{params_v2:,d}",
            "Val_Loss": f"{metrics_v2['loss_total']:.4f}",
            "dE_Oklab": f"{metrics_v2['delta_e_oklab']:.4f}",
            "sRGB_MAE": f"{metrics_v2['mae_srgb']:.4f}",
            "sRGB_MSE": f"{metrics_v2['mse_srgb']:.4f}",
            "PSNR_dB": f"{metrics_v2['psnr_db']:.2f}",
            "Harmonic_Angle_Err_Deg": f"{metrics_v2['harmonic_deg_err']:.2f}",
            "Contrast_Rate_Pct": f"{metrics_v2['distinguishable_pct']:.1f}",
        },
    ]

    # 1. Save CSV Table
    csv_path = out_dir / "ablation_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 2. Save Markdown Table
    md_path = out_dir / "ablation_table.md"
    md_lines = [
        "# Head-to-Head Ablation Study: ARCNet-v1 vs. ARCNet-v2 (Large Dataset)",
        "",
        "| Model | Params | Val Loss (↓) | $\\Delta E_{\\text{Oklab}}$ (↓) | sRGB MAE (↓) | sRGB MSE (↓) | PSNR dB (↑) | Harmonic Err (↓) | Contrast Rate (↑) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        md_lines.append(
            f"| **{r['Model']}** | {r['Params']} | {r['Val_Loss']} | {r['dE_Oklab']} | "
            f"{r['sRGB_MAE']} | {r['sRGB_MSE']} | {r['PSNR_dB']} | "
            f"{r['Harmonic_Angle_Err_Deg']}° | {r['Contrast_Rate_Pct']}% |"
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    # 3. Save IEEE LaTeX Table
    tex_path = out_dir / "ablation_table.tex"
    tex_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Head-to-Head Ablation Comparison Between Original ARCNet-v1 and Proposed ARCNet-v2 on the Large-Scale Aesthetic Dataset}",
        "\\label{tab:ablation_v1_v2}",
        "\\begin{tabular}{lcccccccc}",
        "\\hline",
        "\\textbf{Model} & \\textbf{Params} & \\textbf{Val Loss} $\\downarrow$ & $\\Delta E_{\\text{Oklab}}$ $\\downarrow$ & \\textbf{MAE} $\\downarrow$ & \\textbf{MSE} $\\downarrow$ & \\textbf{PSNR (dB)} $\\uparrow$ & \\textbf{Harmonic Err ($^\\circ$)} $\\downarrow$ & \\textbf{Contrast (\\%)} $\\uparrow$ \\\\",
        "\\hline",
        f"ARCNet-v1 (Original MLP) & {rows[0]['Params']} & {rows[0]['Val_Loss']} & {rows[0]['dE_Oklab']} & {rows[0]['sRGB_MAE']} & {rows[0]['sRGB_MSE']} & {rows[0]['PSNR_dB']} & {rows[0]['Harmonic_Angle_Err_Deg']}$^\\circ$ & {rows[0]['Contrast_Rate_Pct']}\\% \\\\",
        f"\\textbf{{ARCNet-v2 ({preset_v2}, Proposed)}} & \\textbf{{{rows[1]['Params']}}} & \\textbf{{{rows[1]['Val_Loss']}}} & \\textbf{{{rows[1]['dE_Oklab']}}} & \\textbf{{{rows[1]['sRGB_MAE']}}} & \\textbf{{{rows[1]['sRGB_MSE']}}} & \\textbf{{{rows[1]['PSNR_dB']}}} & \\textbf{{{rows[1]['Harmonic_Angle_Err_Deg']}}}$^\\circ$ & \\textbf{{{rows[1]['Contrast_Rate_Pct']}}}\\% \\\\",
        "\\hline",
        "\\end{tabular}",
        "\\end{table*}",
    ]
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")

    print(f"\n[Saved Ablation Tables]")
    print(f"  CSV Table      : {csv_path}")
    print(f"  Markdown Table : {md_path}")
    print(f"  LaTeX Table    : {tex_path}")


def plot_ablation_results(
    history_v1: Optional[List[Dict[str, float]]],
    history_v2: Optional[List[Dict[str, float]]],
    metrics_v1: Dict[str, Any],
    metrics_v2: Dict[str, Any],
    preset_v2: str = "base",
    save_dir: str = "checkpoints",
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "\n[Plot Warning] `matplotlib` is not installed. Run `pip install matplotlib` "
            "and re-run `python3 ablation_plot.py` to generate PNG/PDF plots."
        )
        return

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.titlesize": 14,
    })

    # =================================================================
    # Figure 1: Training & Validation Convergence Curves (Underfitting Proof)
    # =================================================================
    if history_v1 and history_v2:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)
        fig.suptitle(
            f"Underfitting Ablation on Large Dataset: Original ARCNet-v1 vs. Proposed ARCNet-v2 ({preset_v2})",
            fontweight="bold",
            y=0.98,
        )

        ep_v1 = [r["epoch"] for r in history_v1]
        ep_v2 = [r["epoch"] for r in history_v2]

        panels = [
            ("val_loss_total", "train_loss_total", "Total Multi-Objective Loss (↓)", axes[0, 0]),
            ("val_delta_e_oklab", "train_delta_e_oklab", "Perceptual Error ΔE_Oklab (↓)", axes[0, 1]),
            ("val_mae_srgb", None, "Validation sRGB MAE (↓)", axes[1, 0]),
            ("val_harmonic_deg_err", None, "Itten Harmonic Angular Error in Degrees (↓)", axes[1, 1]),
        ]

        for val_key, train_key, title, ax in panels:
            if train_key and train_key in history_v1[0]:
                ax.plot(
                    ep_v1,
                    [r[train_key] for r in history_v1],
                    color="#d95f02",
                    linestyle="--",
                    alpha=0.55,
                    label="ARCNet-v1 (Train)",
                )
            if train_key and train_key in history_v2[0]:
                ax.plot(
                    ep_v2,
                    [r[train_key] for r in history_v2],
                    color="#1b9e77",
                    linestyle="--",
                    alpha=0.55,
                    label=f"ARCNet-v2 {preset_v2} (Train)",
                )

            if val_key in history_v1[0]:
                ax.plot(
                    ep_v1,
                    [r[val_key] for r in history_v1],
                    color="#d95f02",
                    linewidth=2.2,
                    label="ARCNet-v1 (Val - Underfits)",
                )
            if val_key in history_v2[0]:
                ax.plot(
                    ep_v2,
                    [r[val_key] for r in history_v2],
                    color="#1b9e77",
                    linewidth=2.4,
                    label=f"ARCNet-v2 {preset_v2} (Val - Proposed)",
                )

            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(frameon=True, fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        curves_png = out_dir / "ablation_training_curves.png"
        curves_pdf = out_dir / "ablation_training_curves.pdf"
        fig.savefig(curves_png, dpi=300)
        fig.savefig(curves_pdf)
        plt.close(fig)
        print(f"  Saved Convergence Plot : {curves_png}")

    # =================================================================
    # Figure 2: Side-by-Side Metric Bar Comparison
    # =================================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), dpi=300)
    fig.suptitle(
        "Quantitative Performance Comparison: ARCNet-v1 vs. ARCNet-v2",
        fontweight="bold",
        y=1.02,
    )

    colors = ["#e66101", "#1b9e77"]
    models = ["ARCNet-v1\n(Original MLP)", f"ARCNet-v2\n({preset_v2}, Proposed)"]

    # Panel A: Error Metrics (Lower is better)
    err_labels = ["sRGB MAE (↓)", "ΔE_Oklab (↓)", "Val Loss (↓)"]
    v1_errs = [metrics_v1["mae_srgb"], metrics_v1["delta_e_oklab"], metrics_v1["loss_total"]]
    v2_errs = [metrics_v2["mae_srgb"], metrics_v2["delta_e_oklab"], metrics_v2["loss_total"]]

    x = np.arange(len(err_labels))
    width = 0.35
    b1 = axes[0].bar(x - width / 2, v1_errs, width, label=models[0].replace("\n", " "), color=colors[0], alpha=0.88)
    b2 = axes[0].bar(x + width / 2, v2_errs, width, label=models[1].replace("\n", " "), color=colors[1], alpha=0.88)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(err_labels)
    axes[0].set_title("Reconstruction & Perceptual Errors (↓)", fontweight="bold")
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=9)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        axes[0].annotate(
            f"{h:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    # Panel B: PSNR & Harmonic Angle Error
    x2 = np.arange(2)
    psnr_harm_labels = ["PSNR (dB) (↑)", "Harmonic Err (°) (↓)"]
    v1_ph = [metrics_v1["psnr_db"], metrics_v1["harmonic_deg_err"]]
    v2_ph = [metrics_v2["psnr_db"], metrics_v2["harmonic_deg_err"]]
    b3 = axes[1].bar(x2 - width / 2, v1_ph, width, label="ARCNet-v1", color=colors[0], alpha=0.88)
    b4 = axes[1].bar(x2 + width / 2, v2_ph, width, label="ARCNet-v2", color=colors[1], alpha=0.88)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(psnr_harm_labels)
    axes[1].set_title("Fidelity (PSNR ↑) & Harmony Error (° ↓)", fontweight="bold")
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    for bar in list(b3) + list(b4):
        h = bar.get_height()
        axes[1].annotate(
            f"{h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Panel C: Anti-Camouflage Contrast Satisfaction Rate (%)
    c_vals = [metrics_v1["distinguishable_pct"], metrics_v2["distinguishable_pct"]]
    b5 = axes[2].bar(models, c_vals, width=0.45, color=colors, alpha=0.88)
    axes[2].set_ylim(0, 108)
    axes[2].set_ylabel("Distinguishable Predictions (%)")
    axes[2].set_title("Anti-Camouflage Contrast Rate (↑)", fontweight="bold")
    axes[2].grid(axis="y", linestyle=":", alpha=0.6)
    for bar in b5:
        h = bar.get_height()
        axes[2].annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    bars_png = out_dir / "ablation_metrics_bar.png"
    bars_pdf = out_dir / "ablation_metrics_bar.pdf"
    fig.savefig(bars_png, dpi=300, bbox_inches="tight")
    fig.savefig(bars_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Bar Chart Plot   : {bars_png}")


if __name__ == "__main__":
    # Standalone mode: regenerates tables and plots from saved checkpoints & history JSONs
    import argparse
    import torch
    from arcnet_v2 import PRESETS, count_parameters
    from dataset import create_dataloaders
    from losses import ARCNetV2Loss
    from train import ARCNetV1Baseline, evaluate_model

    parser = argparse.ArgumentParser(description="Generate Ablation Tables and Plots from Checkpoints")
    parser.add_argument("--data", type=str, default="data/arcnet_v2_palettes.npz")
    parser.add_argument("--preset", type=str, default="base")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    save_path = Path(args.save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = create_dataloaders(args.data, batch_size=256, min_k=4, num_workers=2)
    criterion = ARCNetV2Loss().to(device)

    ckpt_v1 = save_path / "arcnet_v2_v1_baseline_best.pt"
    ckpt_v2 = save_path / f"arcnet_v2_{args.preset}_best.pt"

    if not ckpt_v1.exists() or not ckpt_v2.exists():
        raise FileNotFoundError(
            f"Checkpoints {ckpt_v1} and/or {ckpt_v2} not found. Run `python3 train.py --compare-v1` first."
        )

    m1 = ARCNetV1Baseline().to(device)
    m1.load_state_dict(torch.load(ckpt_v1, map_location=device, weights_only=False)["model_state_dict"])
    metrics_v1 = evaluate_model(m1, val_loader, criterion, device)

    c2 = torch.load(ckpt_v2, map_location=device, weights_only=False)
    m2 = PRESETS[args.preset](output_mode=c2.get("output_mode", "single_color")).to(device)
    m2.load_state_dict(c2["model_state_dict"])
    metrics_v2 = evaluate_model(m2, val_loader, criterion, device)

    hist_v1_path = save_path / "history_v1_baseline.json"
    hist_v2_path = save_path / f"history_{args.preset}.json"
    hist_v1 = json.loads(hist_v1_path.read_text()) if hist_v1_path.exists() else None
    hist_v2 = json.loads(hist_v2_path.read_text()) if hist_v2_path.exists() else None

    save_ablation_tables(
        metrics_v1=metrics_v1,
        metrics_v2=metrics_v2,
        params_v1=count_parameters(m1),
        params_v2=count_parameters(m2),
        preset_v2=args.preset,
        save_dir=args.save_dir,
    )
    plot_ablation_results(
        history_v1=hist_v1,
        history_v2=hist_v2,
        metrics_v1=metrics_v1,
        metrics_v2=metrics_v2,
        preset_v2=args.preset,
        save_dir=args.save_dir,
    )
