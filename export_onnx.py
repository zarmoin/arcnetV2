"""
Verification and ONNX Export Utility for ARCNet-v2 (Unity Sentis / Barracuda).

Verifies:
1. Parameter counts across presets (mobile, base, large, xl).
2. Strict Permutation Invariance across K-1 background color clusters.
3. Backward compatibility with flat [B, 6] inputs (ARCNet-v1 style) as well as
   dynamic K in [4, 8] cluster sets [B, K-1, 4].
4. ONNX export (opset >= 14) for Unity Sentis deployment.
"""

import argparse
import torch

from arcnet_v2 import (
    arcnet_v2_base,
    arcnet_v2_large,
    arcnet_v2_mobile,
    arcnet_v2_xl,
    count_parameters,
)
from losses import ARCNetV2Loss


PRESETS = {
    "mobile": arcnet_v2_mobile,
    "base": arcnet_v2_base,
    "large": arcnet_v2_large,
    "xl": arcnet_v2_xl,
}


def verify_model(preset_name: str = "base", num_clusters: int = 5):
    print(f"=== Verifying ARCNet-v2 ({preset_name}) ===")
    for name, fn in PRESETS.items():
        m = fn()
        print(f"  Preset '{name:6s}': {count_parameters(m):,d} trainable parameters")

    model = PRESETS[preset_name](output_mode="single_color").eval()

    # 1. Test permutation invariance on K-1 background clusters
    B = 4
    colors = torch.rand(B, num_clusters, 3)
    weights = torch.softmax(torch.randn(B, num_clusters, 1), dim=1)

    perm = torch.randperm(num_clusters)
    colors_perm = colors[:, perm, :]
    weights_perm = weights[:, perm, :]

    with torch.no_grad():
        out1 = model(colors, weights)
        out2 = model(colors_perm, weights_perm)
        max_diff = torch.max(torch.abs(out1 - out2)).item()

    print(f"\n[Permutation Invariance Check (M={num_clusters} clusters)]")
    print(f"  Max absolute difference after shuffling cluster order: {max_diff:.2e}")
    assert max_diff < 1e-5, "Permutation invariance violated!"

    # 2. Test legacy [B, 6] flat input compatibility (RunMLP.cs / ARCNet-v1 drop-in)
    legacy_in = torch.rand(B, 6)
    with torch.no_grad():
        legacy_out = model(legacy_in)
    print(f"\n[Legacy [B, 6] Input Compatibility Check]")
    print(f"  Input shape: {tuple(legacy_in.shape)} -> Output shape: {tuple(legacy_out.shape)}")

    # 3. Test Multi-Material PBR Output Mode + Multi-Objective Loss
    pbr_model = PRESETS[preset_name](output_mode="multi_material").train()
    criterion = ARCNetV2Loss()
    target_rgb = torch.rand(B, 3)

    pred_pbr, aux = pbr_model(colors, weights, return_aux=True)
    loss, metrics = criterion(
        pred_primary_oklab=aux["pred_primary_oklab"],
        target_color=target_rgb,
        bg_oklab=aux["bg_oklab"],
        bg_weights=aux["bg_weights"],
        expert_weights=aux["expert_weights"],
    )
    loss.backward()
    print(f"\n[Multi-Material PBR & Loss Backward Check]")
    print(f"  PBR Output shape: {tuple(pred_pbr.shape)} (c_primary[3], c_secondary[3], dc_spec[3], roughness[1])")
    print(
        f"  Loss Total: {metrics['loss_total']:.4f} | "
        f"Recon: {metrics['loss_recon']:.4f} | "
        f"Dist: {metrics['loss_dist']:.4f} | "
        f"Harm: {metrics['loss_harm']:.4f} | "
        f"dE_Oklab: {metrics['delta_e_oklab']:.4f}"
    )


def export_to_onnx(
    preset_name: str = "base",
    output_path: str = "arcnet_v2.onnx",
    input_format: str = "set",
    num_clusters: int = 5,
    output_mode: str = "single_color",
    opset: int = 14,
):
    """
    Export ARCNet-v2 to ONNX for Unity Sentis / Barracuda.
      - input_format='set': expects [batch_size, num_clusters, 4] where each row is (R, G, B, weight)
      - input_format='flat6': expects [batch_size, 6] for direct drop-in replacement in RunMLP.cs
    """
    model = PRESETS[preset_name](output_mode=output_mode).eval()

    if input_format == "flat6":
        dummy_input = torch.rand(1, 6, dtype=torch.float32)
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    else:
        dummy_input = torch.rand(1, num_clusters, 4, dtype=torch.float32)
        dynamic_axes = {
            "input": {0: "batch_size", 1: "num_clusters"},
            "output": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    print(f"\nExported ARCNet-v2 ({preset_name}, {output_mode}) to {output_path} (opset {opset})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify and Export ARCNet-v2 Model")
    parser.add_argument("--preset", type=str, default="base", choices=list(PRESETS.keys()))
    parser.add_argument("--clusters", type=int, default=5, help="K-1 background clusters")
    parser.add_argument("--export", type=str, default="", help="Path to export ONNX file")
    parser.add_argument("--input-format", type=str, default="set", choices=["set", "flat6"])
    parser.add_argument("--output-mode", type=str, default="single_color", choices=["single_color", "multi_material"])
    args = parser.parse_args()

    verify_model(preset_name=args.preset, num_clusters=args.clusters)
    if args.export:
        export_to_onnx(
            preset_name=args.preset,
            output_path=args.export,
            input_format=args.input_format,
            num_clusters=args.clusters,
            output_mode=args.output_mode,
        )
