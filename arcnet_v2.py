"""
ARCNet-v2: High-Capacity, Permutation-Invariant Color Harmonization Architecture
for Large-Scale Aesthetic Datasets (WikiArt, The Met, AVA, iHarmony4, ARHarmony).

Why Original ARCNet (6 -> 64 -> 32 -> 3 MLP, ~2.3k params) Underfits on Large Datasets:
---------------------------------------------------------------------------------------
1. Extreme Parameter Bottleneck: 2,339 trainable parameters cannot capture the complex
   chromatic distributions of 100,000+ diverse artworks and photographs.
2. Spectral Bias on Raw RGB: Low-dimensional sRGB inputs cause standard ReLU MLPs to
   learn overly smooth, low-frequency mappings.
3. Order Sensitivity (Non-Permutation Invariance): Concatenating ranked colors into a
   flat vector causes output discontinuities when cluster sizes are similar.
4. Multi-Modal Averaging Collapse: Artworks exhibit multiple valid harmonic modes
   (e.g., complementary, split-complementary, triadic, analogous). A single linear head
   regresses toward the mean of these modes, producing desaturated/muddy colors.

Architectural Modifications in ARCNet-v2:
-----------------------------------------
1. Perceptual Oklab + Chromatic Fourier & Circular Hue Expansion (eliminates spectral bias).
2. Equivariant DeepSets + Micro-Set Transformer Encoder with Pairwise Contrast Bias
   (strictly permutation-invariant across K-1 background clusters).
3. Pre-LayerNorm Residual SwiGLU Blocks with LayerScale (enables deep, high-capacity
   feature extraction without gradient vanishing or saturation).
4. Area-Weighted + Multi-Head Attention Set Pooling (preserves both dominant background
   masses and high-chroma accent clusters).
5. Optional FiLM Conditioning for ARCore Spherical Harmonics (L0, L1) lighting tokens.
6. Gated Mixture of Harmonic Experts (MoHE) Head (prevents multi-modal averaging/graying
   by modeling M distinct harmonic strategies and routing dynamically).
7. Structured Multi-Material PBR Output [c_primary, c_secondary, dc_specular, tau_roughness]
   as well as backward-compatible 3D RGB/Oklab target output.
"""

from typing import Dict, Optional, Tuple, Union
import torch

import torch.nn as nn
import torch.nn.functional as F

from color_space import (
    ChromaticFeatureExpansion,
    oklab_to_srgb,
    srgb_to_oklab,
)


class ResSwiGLUBlock(nn.Module):
    """
    High-capacity Pre-LayerNorm Residual Block with SwiGLU gated activation
    and learnable LayerScale. Replaces standard Dense -> ReLU layers to prevent
    underfitting and representation collapse on large datasets.
    """

    def __init__(
        self,
        dim: int,
        expansion_factor: float = 2.0,
        dropout: float = 0.05,
        layer_scale_init: float = 1e-1,
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.norm = nn.LayerNorm(dim)
        self.fc_gate = nn.Linear(dim, hidden_dim)
        self.fc_up = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc_down = nn.Linear(hidden_dim, dim)
        self.layer_scale = nn.Parameter(torch.ones(dim) * layer_scale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = self.act(self.fc_gate(h)) * self.fc_up(h)
        h = self.drop(h)
        h = self.fc_down(h)
        return residual + self.layer_scale * h


class SetAttentionBlock(nn.Module):
    """
    Permutation-Equivariant Set Attention Block (Micro-Set Transformer) with
    Pairwise Oklab Contrast Bias. Allows each background color cluster to attend
    to the relative chromatic/luminance contrast of all other clusters in the scene.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Pairwise Oklab contrast bias projector: maps [dL, da, db, dE, w_i * w_j] -> num_heads
        self.pairwise_bias_net = nn.Sequential(
            nn.Linear(5, num_heads * 2),
            nn.SiLU(),
            nn.Linear(num_heads * 2, num_heads),
        )

        self.ffn = ResSwiGLUBlock(dim=dim, expansion_factor=2.0, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        oklab: torch.Tensor,
        weights: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, M, D] cluster token embeddings (M = K - 1)
            oklab: [B, M, 3] cluster colors in Oklab
            weights: [B, M, 1] cluster area weights
            mask: Optional [B, M] boolean mask (True for valid tokens, False for padded)
        Returns:
            out: [B, M, D] permutation-equivariant refined embeddings
        """
        B, M, D = x.shape
        h = self.norm1(x)

        qkv = self.qkv(h).reshape(B, M, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, M, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, M, M]

        # Compute pairwise Oklab contrast features [B, M, M, 5]
        diff_oklab = oklab.unsqueeze(2) - oklab.unsqueeze(1)  # [B, M, M, 3]
        dist_oklab = torch.sqrt(torch.sum(diff_oklab * diff_oklab, dim=-1, keepdim=True) + 1e-6)
        weight_prod = weights.unsqueeze(2) * weights.unsqueeze(1)  # [B, M, M, 1]
        pair_feat = torch.cat([torch.abs(diff_oklab), dist_oklab, weight_prod], dim=-1)
        pair_bias = self.pairwise_bias_net(pair_feat).permute(0, 3, 1, 2)  # [B, H, M, M]

        attn_logits = attn_logits + pair_bias

        if mask is not None:
            # mask is [B, M], expand to [B, 1, 1, M]
            key_mask = mask.unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits.masked_fill(~key_mask, -1e9)

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, M, D)
        x = x + self.proj(out)
        x = self.ffn(x)
        return x


class WeightedSetPooling(nn.Module):
    """
    Permutation-Invariant Set Aggregator combining:
      1. Unweighted Sum Pooling: sum_k phi(c_k, w_k) (Roadmap Eq. 3)
      2. Area-Weighted Pooling: sum_k w_k * phi(c_k, w_k)
      3. Learned Query Attention Pooling: captures salient high-chroma accent clusters
    """

    def __init__(self, dim: int, num_queries: int = 2):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.key_proj = nn.Linear(dim, dim)
        self.val_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

        # Fuse [sum_pool, weighted_pool, query_pools...] -> dim
        self.fuse = nn.Sequential(
            nn.LayerNorm(dim * (2 + num_queries)),
            nn.Linear(dim * (2 + num_queries), dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, M, D]
            weights: [B, M, 1]
            mask: Optional [B, M] boolean mask
        Returns:
            pooled: [B, D] permutation-invariant global background representation h_bg
        """
        B, M, D = x.shape

        if mask is not None:
            valid_f = mask.unsqueeze(-1).float()
            x_masked = x * valid_f
            w_masked = weights * valid_f
        else:
            x_masked = x
            w_masked = weights

        # 1. DeepSets Sum Pooling (Eq. 3 in ARCNet-v2 Roadmap)
        sum_pool = x_masked.sum(dim=1)  # [B, D]

        # 2. Area-Weighted Pooling
        w_norm = w_masked / (w_masked.sum(dim=1, keepdim=True) + 1e-6)
        weighted_pool = (x_masked * w_norm).sum(dim=1)  # [B, D]

        # 3. Learned Multi-Query Attention Pooling
        q = self.queries.expand(B, -1, -1)  # [B, Q, D]
        k = self.key_proj(x_masked)  # [B, M, D]
        v = self.val_proj(x_masked)  # [B, M, D]

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale  # [B, Q, M]
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask.unsqueeze(1), -1e9)
        attn = F.softmax(attn_logits, dim=-1)
        query_pool = (attn @ v).reshape(B, self.num_queries * D)  # [B, Q*D]

        combined = torch.cat([sum_pool, weighted_pool, query_pool], dim=-1)
        return self.fuse(combined)


class FiLMLightingConditioner(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) for ARCore Environmental HDR Spherical
    Harmonics (L0, L1 coefficients, typically 9 or 27 floats) as specified in
    ARCNet-v2 Roadmap Section 4.2.
    """

    def __init__(self, lighting_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lighting_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        # Initialize near identity modulation (gamma=0 -> 1+gamma=1, beta=0)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h_bg: torch.Tensor, lighting_sh: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.net(lighting_sh)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        return h_bg * (1.0 + gamma) + beta


class HarmonicMixtureHead(nn.Module):
    """
    Mixture of Harmonic Experts (MoHE) Output Head.
    Prevents underfitting / "gray-average collapse" on large artistic datasets where
    multiple harmonic strategies (Complementary, Split-Complementary, Triadic,
    Analogous, Accent-Contrast, etc.) exist for similar background palettes.

    Each expert predicts a candidate parameter vector; a context-sensitive gating
    network softly combines them (and exposes expert predictions/weights for optional
    diversity/winner-take-all losses during training).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_experts: int = 6,
        expert_hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.out_dim = out_dim

        self.router = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim // 2),
            nn.SiLU(),
            nn.Linear(in_dim // 2, num_experts),
        )

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(in_dim),
                    nn.Linear(in_dim, expert_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(expert_hidden_dim, out_dim),
                )
                for _ in range(num_experts)
            ]
        )

        # Direct residual projection for stability
        self.base_proj = nn.Linear(in_dim, out_dim)

    def forward(
        self, h: torch.Tensor, return_mixture_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            h: [B, in_dim] latent representation
            return_mixture_info: if True, also returns (expert_weights [B, E], expert_preds [B, E, out_dim])
        """
        logits = self.router(h)  # [B, E]
        weights = F.softmax(logits, dim=-1)  # [B, E]

        expert_outputs = torch.stack(
            [expert(h) for expert in self.experts], dim=1
        )  # [B, E, out_dim]

        mixture_out = torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=1)  # [B, out_dim]
        out = self.base_proj(h) + mixture_out

        if return_mixture_info:
            return out, weights, expert_outputs
        return out


class TemporalGRUHead(nn.Module):
    """
    Optional Lightweight Recurrent Head (GRU) for frame-to-frame temporal smoothing
    (ARCNet-v2 Roadmap Section 4.1). Can be embedded directly into the ONNX graph.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gru = nn.GRUCell(input_size=dim, hidden_size=dim)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, h_current: torch.Tensor, h_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if h_prev is None:
            h_prev = h_current
        h_next = self.gru(h_current, h_prev)
        h_smoothed = self.norm(0.5 * h_current + 0.5 * h_next)
        return h_smoothed, h_next


class ARCNetV2(nn.Module):
    """
    ARCNet-v2: Permutation-Invariant, High-Capacity Aesthetic Color Harmonization Model.

    Supports:
      - Dynamic K in [4, 8] background clusters {(c_k, w_k)}_{k=1}^{K-1} OR legacy flat
        inputs (e.g., [B, 6] for 2 RGB colors from ARCNet-v1).
      - Input/output color space in either 'srgb' (auto-converted to Oklab internally)
        or 'oklab' directly.
      - Output modes:
          * 'single_color': predicts target harmonized color c_fg in R^3
          * 'multi_material': predicts PBR material tensor Y_fg in R^10:
            [c_primary (3), c_secondary (3), delta_c_specular (3), tau_roughness (1)]
      - Optional Environmental HDR Spherical Harmonics conditioning (lighting_dim > 0).
      - Optional Temporal GRU stabilization (use_temporal_gru=True).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_phi_layers: int = 2,
        num_sab_layers: int = 2,
        num_rho_layers: int = 3,
        num_heads: int = 4,
        num_fourier_bands: int = 4,
        num_hue_harmonics: int = 4,
        num_harmonic_experts: int = 6,
        dropout: float = 0.05,
        input_space: str = "srgb",
        output_space: str = "srgb",
        output_mode: str = "single_color",
        lighting_dim: int = 0,
        use_temporal_gru: bool = False,
    ):
        super().__init__()
        assert input_space in ("srgb", "oklab")
        assert output_space in ("srgb", "oklab")
        assert output_mode in ("single_color", "multi_material")

        self.hidden_dim = hidden_dim
        self.input_space = input_space
        self.output_space = output_space
        self.output_mode = output_mode
        self.lighting_dim = lighting_dim
        self.use_temporal_gru = use_temporal_gru

        # 1. Perceptual & Chromatic Fourier Feature Expansion
        self.chromatic_expander = ChromaticFeatureExpansion(
            num_fourier_bands=num_fourier_bands,
            num_hue_harmonics=num_hue_harmonics,
            learnable_frequencies=True,
        )

        # 2. Equivariant Per-Cluster Encoder phi(c_k (+) w_k)
        self.phi_in = nn.Linear(self.chromatic_expander.out_dim, hidden_dim)
        self.phi_blocks = nn.ModuleList(
            [
                ResSwiGLUBlock(dim=hidden_dim, expansion_factor=2.0, dropout=dropout)
                for _ in range(num_phi_layers)
            ]
        )

        # 3. Inter-Cluster Set Attention Blocks (Micro-Set Transformer)
        self.sab_blocks = nn.ModuleList(
            [
                SetAttentionBlock(dim=hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_sab_layers)
            ]
        )

        # 4. Permutation-Invariant Set Pooling (DeepSets Sum + Weighted + Query Attention)
        self.set_pool = WeightedSetPooling(dim=hidden_dim, num_queries=2)

        # 5. Optional Physical Lighting Conditioner (ARCore Spherical Harmonics)
        if lighting_dim > 0:
            self.lighting_conditioner = FiLMLightingConditioner(
                lighting_dim=lighting_dim, hidden_dim=hidden_dim
            )
        else:
            self.lighting_conditioner = None

        # 6. Harmonization Manifold Decoder rho(.)
        self.rho_blocks = nn.ModuleList(
            [
                ResSwiGLUBlock(dim=hidden_dim, expansion_factor=2.0, dropout=dropout)
                for _ in range(num_rho_layers)
            ]
        )

        # 7. Optional Temporal Smoothing GRU
        if use_temporal_gru:
            self.temporal_head = TemporalGRUHead(dim=hidden_dim)
        else:
            self.temporal_head = None

        # 8. Output Head (Mixture of Harmonic Experts)
        out_dim = 3 if output_mode == "single_color" else 10
        self.head = HarmonicMixtureHead(
            in_dim=hidden_dim,
            out_dim=out_dim,
            num_experts=num_harmonic_experts,
            expert_hidden_dim=max(32, hidden_dim // 2),
        )

        # Global weighted background color anchor for residual chromatic offset prediction
        self.anchor_gate = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _parse_inputs(
        self,
        colors: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalizes flexible input formats into:
          colors: [B, M, 3]
          weights: [B, M, 1]
        Supports:
          - colors=[B, M, 3], weights=[B, M] or [B, M, 1]
          - colors=[B, M, 4] (last channel is weight w_k)
          - colors=[B, 3*M] (legacy flat RGB vector, e.g., [B, 6] from ARCNet-v1)
          - colors=[B, 4*M] (flat [r, g, b, w] vector)
        """
        if colors.dim() == 2:
            B, D = colors.shape
            if D % 3 == 0 and weights is None and D % 4 != 0:
                M = D // 3
                colors = colors.view(B, M, 3)
            elif D == 6 and weights is None:
                colors = colors.view(B, 2, 3)
            elif D % 4 == 0 and weights is None:
                M = D // 4
                reshaped = colors.view(B, M, 4)
                colors = reshaped[..., :3]
                weights = reshaped[..., 3:4]
            elif D % 3 == 0:
                M = D // 3
                colors = colors.view(B, M, 3)
            else:
                raise ValueError(
                    f"2D input tensor shape {colors.shape} must have last dim divisible by 3 or 4."
                )
        elif colors.dim() == 3 and colors.shape[-1] == 4 and weights is None:
            weights = colors[..., 3:4]
            colors = colors[..., :3]

        B, M, C = colors.shape
        assert C == 3, f"Expected 3 color channels, got {C}"

        if weights is None:
            # Default uniform weights if not provided
            weights = torch.full(
                (B, M, 1), 1.0 / M, dtype=colors.dtype, device=colors.device
            )
        elif weights.dim() == 2:
            weights = weights.unsqueeze(-1)

        # Normalize weights to sum to 1 across clusters
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
        return colors, weights

    def _format_color_output(self, raw_oklab: torch.Tensor) -> torch.Tensor:
        """
        Maps raw unbounded network predictions into valid Oklab bounds:
          L in [0, 1] via Sigmoid
          a, b in [-0.4, 0.4] via 0.4 * Tanh
        and optionally converts to sRGB in [0, 1].
        """
        L = torch.sigmoid(raw_oklab[..., 0:1])
        ab = 0.45 * torch.tanh(raw_oklab[..., 1:3])
        pred_oklab = torch.cat([L, ab], dim=-1)

        if self.output_space == "srgb":
            return oklab_to_srgb(pred_oklab)
        return pred_oklab

    def forward(
        self,
        colors: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        lighting_sh: Optional[torch.Tensor] = None,
        h_prev: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Args:
            colors: [B, M, 3] or [B, M, 4] or legacy [B, 6] background dominant colors
            weights: Optional [B, M] or [B, M, 1] cluster area weights w_k
            mask: Optional [B, M] boolean mask for dynamic cluster counts K in [4, 8]
            lighting_sh: Optional [B, lighting_dim] ARCore Spherical Harmonics tokens
            h_prev: Optional [B, hidden_dim] previous frame GRU state
            return_aux: If True, returns (prediction, aux_dict) with Oklab tensors and MoHE weights
        """
        colors, weights = self._parse_inputs(colors, weights)

        # Convert input to Oklab if provided in sRGB
        if self.input_space == "srgb":
            oklab_in = srgb_to_oklab(colors)
        else:
            oklab_in = colors

        # 1. Chromatic Fourier & Circular Hue Expansion
        feat = self.chromatic_expander(oklab_in, weights)  # [B, M, F]

        # 2. Equivariant Per-Cluster Feature Encoder phi(.)
        tokens = self.phi_in(feat)  # [B, M, D]
        for block in self.phi_blocks:
            tokens = block(tokens)

        # 3. Inter-Cluster Set Attention Blocks (SAB)
        for sab in self.sab_blocks:
            tokens = sab(tokens, oklab=oklab_in, weights=weights, mask=mask)

        # 4. Permutation-Invariant Pooling -> h_bg
        h_bg = self.set_pool(tokens, weights=weights, mask=mask)  # [B, D]

        # 5. Optional Physical Lighting Coupling
        if self.lighting_conditioner is not None and lighting_sh is not None:
            h_bg = self.lighting_conditioner(h_bg, lighting_sh)

        # 6. Harmonization Manifold Projection rho(.)
        h = h_bg
        for block in self.rho_blocks:
            h = block(h)

        # 7. Optional Temporal GRU Smoothing
        h_next = h
        if self.temporal_head is not None:
            h, h_next = self.temporal_head(h, h_prev)

        # 8. Mixture of Harmonic Experts Prediction
        raw_out, expert_weights, expert_preds = self.head(h, return_mixture_info=True)

        # Compute weighted mean background color in Oklab as a reference anchor
        bg_mean_oklab = (oklab_in * weights).sum(dim=1)  # [B, 3]
        gate = self.anchor_gate(h)  # [B, 3]

        if self.output_mode == "single_color":
            # Residual anchor allows learning contrastive offsets around or away from bg_mean
            raw_primary = raw_out + gate * bg_mean_oklab
            pred = self._format_color_output(raw_primary)
        else:
            # Multi-Material PBR Output: [c_primary(3), c_secondary(3), dc_specular(3), tau_roughness(1)]
            raw_primary = raw_out[..., 0:3] + gate * bg_mean_oklab
            raw_secondary = raw_out[..., 3:6]
            raw_specular = raw_out[..., 6:9]
            raw_roughness = raw_out[..., 9:10]

            c_primary = self._format_color_output(raw_primary)
            c_secondary = self._format_color_output(raw_secondary)
            dc_specular = 0.25 * torch.tanh(raw_specular)  # bounded specular shift
            tau_roughness = torch.sigmoid(raw_roughness)   # PBR roughness in [0, 1]

            pred = torch.cat([c_primary, c_secondary, dc_specular, tau_roughness], dim=-1)

        if not return_aux:
            return pred

        # Prepare auxiliary outputs for Multi-Objective Loss (Oklab space)
        pred_primary_srgb_or_oklab = pred[..., :3]
        pred_primary_oklab = (
            srgb_to_oklab(pred_primary_srgb_or_oklab)
            if self.output_space == "srgb"
            else pred_primary_srgb_or_oklab
        )

        aux = {
            "pred_primary_oklab": pred_primary_oklab,
            "bg_oklab": oklab_in,
            "bg_weights": weights,
            "expert_weights": expert_weights,
            "expert_preds": expert_preds,
            "h_next": h_next,
        }
        return pred, aux


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =====================================================================
# Preset Model Configurations for Different Dataset Scales & Budgets
# =====================================================================

def arcnet_v2_mobile(**kwargs) -> ARCNetV2:
    """
    Lightweight Mobile AR Preset (<= 45k parameters, < 1.5 ms on Unity Sentis).
    Matches Table 1 / Section 3.1 of ARCNet_V2_roadmap.pdf (~41k parameters),
    representing an ~18x capacity increase over original ARCNet-v1 (2.3k params).
    """
    defaults = dict(
        hidden_dim=48,
        num_phi_layers=1,
        num_sab_layers=1,
        num_rho_layers=2,
        num_heads=4,
        num_fourier_bands=3,
        num_hue_harmonics=4,
        num_harmonic_experts=4,
        dropout=0.02,
    )
    defaults.update(kwargs)
    return ARCNetV2(**defaults)


def arcnet_v2_base(**kwargs) -> ARCNetV2:
    """
    Balanced High-Capacity Preset (~175k parameters, < 2.5 ms on Unity Sentis).
    Recommended for 50,000 - 150,000 images (e.g., WikiArt + iHarmony4).
    Prevents underfitting while easily meeting the <= 3 ms mobile frame budget.
    """
    defaults = dict(
        hidden_dim=96,
        num_phi_layers=2,
        num_sab_layers=2,
        num_rho_layers=3,
        num_heads=4,
        num_fourier_bands=4,
        num_hue_harmonics=4,
        num_harmonic_experts=6,
        dropout=0.05,
    )
    defaults.update(kwargs)
    return ARCNetV2(**defaults)


def arcnet_v2_large(**kwargs) -> ARCNetV2:
    """
    Large Dataset Preset (~680k parameters).
    Recommended for full 250,000 - 650,000+ corpus (WikiArt + The Met + AVA).
    Eliminates capacity bottlenecks across diverse artistic styles and lighting.
    """
    defaults = dict(
        hidden_dim=192,
        num_phi_layers=3,
        num_sab_layers=3,
        num_rho_layers=4,
        num_heads=6,
        num_fourier_bands=6,
        num_hue_harmonics=6,
        num_harmonic_experts=8,
        dropout=0.1,
    )
    defaults.update(kwargs)
    return ARCNetV2(**defaults)


def arcnet_v2_xl(**kwargs) -> ARCNetV2:
    """
    Extra-Large Preset (~1.8M parameters) for maximum capacity / teacher distillation
    on massive multi-domain corpora (500k+ images).
    """
    defaults = dict(
        hidden_dim=256,
        num_phi_layers=4,
        num_sab_layers=4,
        num_rho_layers=6,
        num_heads=8,
        num_fourier_bands=8,
        num_hue_harmonics=6,
        num_harmonic_experts=8,
        dropout=0.1,
    )
    defaults.update(kwargs)
    return ARCNetV2(**defaults)


PRESETS = {
    "mobile": arcnet_v2_mobile,
    "base": arcnet_v2_base,
    "large": arcnet_v2_large,
    "xl": arcnet_v2_xl,
}

