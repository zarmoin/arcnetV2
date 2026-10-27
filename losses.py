"""
Multi-Objective Perceptual-Contrastive Loss Formulation for ARCNet-v2
(ARCNet_V2_roadmap.pdf Section 3.3, Equations 5 & 6).

Combines:
1. Perceptual Oklab Reconstruction Loss (L_recon):
   Charbonnier / Huber + Squared Euclidean distance in Oklab space, plus
   Chroma-preservation term to prevent gray/desaturated mode-averaging on large datasets.
2. Distinguishability Hinge Penalty (L_dist, Eq. 6):
   Prevents virtual objects from camouflaging into background clusters by enforcing
   a minimum perceptual contrast margin gamma in Oklab space.
3. Itten Harmonic Sector Regularizer (L_harm):
   Soft circular loss on the Oklch hue wheel encouraging classical harmonic angles
   (0° monochromatic, 30° analogous, 120° triadic, 150° split-complementary, 180° complementary)
   relative to the dominant background hue.
4. Mixture of Harmonic Experts Load-Balancing Regularizer (L_balance):
   Prevents router collapse so all harmonic experts remain active across the large dataset.
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from color_space import oklab_to_oklch, srgb_to_oklab


class ARCNetV2Loss(nn.Module):
    """
    Compound Objective:
        L_total = L_recon + lambda_dist * L_dist + lambda_harm * L_harm + lambda_bal * L_balance
    """

    def __init__(
        self,
        lambda_dist: float = 0.15,
        lambda_harm: float = 0.05,
        lambda_chroma: float = 0.20,
        lambda_balance: float = 0.02,
        contrast_margin_gamma: float = 0.15,  # ~15 dE in Oklab (where L in [0,1])
        target_space: str = "srgb",
    ):
        super().__init__()
        self.lambda_dist = lambda_dist
        self.lambda_harm = lambda_harm
        self.lambda_chroma = lambda_chroma
        self.lambda_balance = lambda_balance
        self.contrast_margin_gamma = contrast_margin_gamma
        self.target_space = target_space

        # Classical Itten harmonic hue offsets in radians:
        # [0 (monochromatic), pi/6 (30° analogous), 2*pi/3 (120° triadic),
        #  5*pi/6 (150° split-comp), pi (180° complementary)]
        harmonic_angles = torch.tensor(
            [0.0, math.pi / 6.0, 2.0 * math.pi / 3.0, 5.0 * math.pi / 6.0, math.pi],
            dtype=torch.float32,
        )
        self.register_buffer("harmonic_angles", harmonic_angles)

    def forward(
        self,
        pred_primary_oklab: torch.Tensor,
        target_color: torch.Tensor,
        bg_oklab: torch.Tensor,
        bg_weights: Optional[torch.Tensor] = None,
        expert_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            pred_primary_oklab: [B, 3] predicted foreground color in Oklab
            target_color: [B, 3] ground-truth target color (in sRGB or Oklab per target_space)
            bg_oklab: [B, K-1, 3] background cluster centroids in Oklab
            bg_weights: Optional [B, K-1, 1] cluster area weights
            expert_weights: Optional [B, E] Mixture-of-Harmonic-Experts routing probabilities
        Returns:
            total_loss: scalar Tensor
            metrics: dict of individual loss components
        """
        if self.target_space == "srgb":
            target_oklab = srgb_to_oklab(target_color)
        else:
            target_oklab = target_color

        # 1. Perceptual Reconstruction Loss in Oklab (Eq. 5.1) + Chroma Preservation
        diff = pred_primary_oklab - target_oklab
        sq_dist = torch.sum(diff * diff, dim=-1)  # ||c_hat - c_target||_Oklab^2
        charbonnier = torch.sqrt(sq_dist + 1e-6)
        l_recon_base = 0.5 * sq_dist.mean() + 0.5 * charbonnier.mean()

        # Chroma preservation term prevents regression to desaturated gray average on large datasets
        pred_lch = oklab_to_oklch(pred_primary_oklab)
        targ_lch = oklab_to_oklch(target_oklab)
        l_chroma = F.l1_loss(pred_lch[..., 1], targ_lch[..., 1])
        l_recon = l_recon_base + self.lambda_chroma * l_chroma

        # 2. Distinguishability Hinge Penalty (Roadmap Eq. 6)
        # L_dist = sum_{k=1}^{K-1} w_k * max(0, gamma - ||c_hat - c_k||_Oklab)
        bg_diff = pred_primary_oklab.unsqueeze(1) - bg_oklab  # [B, K-1, 3]
        bg_delta_e = torch.sqrt(torch.sum(bg_diff * bg_diff, dim=-1) + 1e-6)  # [B, K-1]
        hinge = F.relu(self.contrast_margin_gamma - bg_delta_e)  # [B, K-1]
        if bg_weights is not None:
            w = bg_weights.squeeze(-1)
            l_dist = torch.sum(w * hinge, dim=-1).mean()
        else:
            l_dist = hinge.mean()

        # 3. Itten Harmonic Sector Regularizer (Roadmap Sec. 3.3.3)
        # Compute circular angular distance between predicted hue and dominant background hue
        dom_bg_oklab = bg_oklab[:, 0, :]  # Top dominant background cluster [B, 3]
        dom_bg_lch = oklab_to_oklch(dom_bg_oklab)
        delta_hue = torch.abs(pred_lch[..., 2] - dom_bg_lch[..., 2])  # [B]
        # Fold to [0, pi] on the hue circle
        delta_hue = torch.minimum(delta_hue, 2.0 * math.pi - delta_hue)  # [B]

        # Distance to the nearest classic Itten harmonic sector
        # sector_diffs: [B, num_sectors]
        sector_diffs = 1.0 - torch.cos(
            delta_hue.unsqueeze(-1) - self.harmonic_angles.unsqueeze(0)
        )
        min_sector_loss, _ = torch.min(sector_diffs, dim=-1)
        l_harm = min_sector_loss.mean()

        # 4. Expert Load-Balancing Loss (prevents MoHE router collapse)
        if expert_weights is not None:
            # Encourage uniform average utilization across batch
            mean_prob = expert_weights.mean(dim=0)  # [E]
            num_experts = expert_weights.shape[-1]
            uniform = torch.full_like(mean_prob, 1.0 / num_experts)
            l_balance = F.mse_loss(mean_prob, uniform) * num_experts
        else:
            l_balance = torch.zeros_like(l_recon)

        total_loss = (
            l_recon
            + self.lambda_dist * l_dist
            + self.lambda_harm * l_harm
            + self.lambda_balance * l_balance
        )

        metrics = {
            "loss_total": total_loss.detach(),
            "loss_recon": l_recon.detach(),
            "loss_dist": l_dist.detach(),
            "loss_harm": l_harm.detach(),
            "loss_chroma": l_chroma.detach(),
            "loss_balance": l_balance.detach(),
            "delta_e_oklab": charbonnier.mean().detach(),
        }
        return total_loss, metrics
