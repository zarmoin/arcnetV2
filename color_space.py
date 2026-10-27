"""
Differentiable Perceptual Color Space Conversions (sRGB <-> Oklab <-> Oklch)
and Multi-Scale Chromatic Fourier Feature Expansion for ARCNet-v2.

Addresses underfitting caused by:
1. Perceptual non-uniformity of standard sRGB Euclidean metrics (Roadmap Sec. 2.2).
2. Spectral bias in low-dimensional color coordinates by expanding (L, a, b)
   into circular harmonic hue encodings (capturing Itten's 180° complementary,
   120° triadic, 90° tetradic relationships) and multi-frequency Fourier features.
"""

import math
import torch
import torch.nn as nn


def srgb_to_linear(srgb: torch.Tensor) -> torch.Tensor:
    """Convert standard sRGB in [0, 1] to linear RGB."""
    srgb = torch.clamp(srgb, 0.0, 1.0)
    return torch.where(
        srgb <= 0.04045,
        srgb / 12.92,
        torch.pow((srgb + 0.055) / 1.055, 2.4),
    )


def linear_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    """Convert linear RGB in [0, 1] to standard sRGB in [0, 1]."""
    linear = torch.clamp(linear, 0.0, 1.0)
    return torch.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * torch.pow(torch.clamp(linear, min=1e-8), 1.0 / 2.4) - 0.055,
    )


def srgb_to_oklab(srgb: torch.Tensor) -> torch.Tensor:
    """
    Differentiable conversion from sRGB [..., 3] in [0, 1] to Oklab (L, a, b) [..., 3].
    Reference: Björn Ottosson (2020) & ARCNet-v2 Roadmap Section 2.2.
    """
    lin = srgb_to_linear(srgb)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]

    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_ = torch.pow(torch.clamp(l, min=1e-8), 1.0 / 3.0)
    m_ = torch.pow(torch.clamp(m, min=1e-8), 1.0 / 3.0)
    s_ = torch.pow(torch.clamp(s, min=1e-8), 1.0 / 3.0)

    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_out = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return torch.stack([L, a, b_out], dim=-1)


def oklab_to_srgb(oklab: torch.Tensor) -> torch.Tensor:
    """
    Differentiable conversion from Oklab (L, a, b) [..., 3] to sRGB [..., 3] in [0, 1].
    """
    L, a, b = oklab[..., 0], oklab[..., 1], oklab[..., 2]

    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_

    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_lin = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    lin = torch.stack([r, g, b_lin], dim=-1)
    return linear_to_srgb(lin)


def oklab_to_oklch(oklab: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert Oklab (L, a, b) [..., 3] to cylindrical Oklch (L, C, h_rad) [..., 3].
    """
    L, a, b = oklab[..., 0], oklab[..., 1], oklab[..., 2]
    C = torch.sqrt(a * a + b * b + eps)
    h = torch.atan2(b + eps, a + eps)  # [-pi, pi]
    return torch.stack([L, C, h], dim=-1)


class ChromaticFeatureExpansion(nn.Module):
    """
    Mitigates MLP spectral bias on low-dimensional color inputs by expanding
    each cluster token (c_k, w_k) into a rich perceptual feature vector:
      1. Oklab coordinates (L, a, b)
      2. Chroma magnitude C = sqrt(a^2 + b^2)
      3. Circular Hue Harmonics: [sin(m*h), cos(m*h)] for m = 1..num_hue_harmonics
         (m=1: 360° fundamental, m=2: 180° complementary, m=3: 120° triadic,
          m=4: 90° tetradic/square harmony)
      4. Multi-scale Fourier encodings of (L, a, b, w_k) with learnable or fixed frequencies
      5. Relative cluster area weight w_k and log-weight log(w_k + eps)
    """

    def __init__(
        self,
        num_fourier_bands: int = 4,
        num_hue_harmonics: int = 4,
        learnable_frequencies: bool = True,
    ):
        super().__init__()
        self.num_fourier_bands = num_fourier_bands
        self.num_hue_harmonics = num_hue_harmonics

        # Frequencies: 2^0 * pi, 2^1 * pi, ..., 2^(B-1) * pi
        init_freqs = (2.0 ** torch.arange(num_fourier_bands, dtype=torch.float32)) * math.pi
        if learnable_frequencies:
            self.fourier_freqs = nn.Parameter(init_freqs)
        else:
            self.register_buffer("fourier_freqs", init_freqs)

        # Output feature dimension calculation:
        # 3 (Oklab) + 1 (Chroma) + 2 (w_k, log_w_k) + 2*num_hue_harmonics + 4*2*num_fourier_bands
        self.out_dim = 6 + (2 * num_hue_harmonics) + (8 * num_fourier_bands)

    def forward(self, oklab: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            oklab: [..., 3] in Oklab space (L, a, b)
            weights: [..., 1] cluster area weights w_k in [0, 1]
        Returns:
            expanded_features: [..., self.out_dim]
        """
        L = oklab[..., 0:1]
        a = oklab[..., 1:2]
        b = oklab[..., 2:3]

        chroma = torch.sqrt(a * a + b * b + 1e-6)
        hue = torch.atan2(b + 1e-6, a + 1e-6)  # [..., 1]

        # Circular Hue Harmonics for Itten color wheel sectors
        hue_harmonics = []
        for m in range(1, self.num_hue_harmonics + 1):
            hue_harmonics.append(torch.sin(m * hue))
            hue_harmonics.append(torch.cos(m * hue))
        hue_feat = torch.cat(hue_harmonics, dim=-1)  # [..., 2 * num_hue_harmonics]

        # Multi-frequency Fourier encoding on [L, a, b, w_k]
        base_4d = torch.cat([oklab, weights], dim=-1)  # [..., 4]
        # Shape: [..., 4, num_fourier_bands]
        scaled = base_4d.unsqueeze(-1) * self.fourier_freqs
        fourier_sin = torch.sin(scaled).flatten(start_dim=-2)
        fourier_cos = torch.cos(scaled).flatten(start_dim=-2)

        log_w = torch.log(torch.clamp(weights, min=1e-4))

        return torch.cat(
            [oklab, chroma, weights, log_w, hue_feat, fourier_sin, fourier_cos],
            dim=-1,
        )
