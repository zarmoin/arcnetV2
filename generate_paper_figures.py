#!/usr/bin/env python3
"""Generates publication-quality vector PDF and high-res PNG figures for the ARCNet-v2 Overleaf LaTeX package using PyCairo."""

import math
from pathlib import Path
import cairo

FIG_DIR = Path(
    "/run/media/moin/Research/Color"
    " harmonization/ARCNet_v2_Overleaf_Paper/figures"
)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def render_both(name: str, width: int, height: int, draw_fn):
  """Renders a drawing function to both vector PDF and 300-DPI equivalent PNG."""
  pdf_path = FIG_DIR / f"{name}.pdf"
  png_path = FIG_DIR / f"{name}.png"

  pdf_surf = cairo.PDFSurface(str(pdf_path), width, height)
  ctx = cairo.Context(pdf_surf)
  draw_fn(ctx, width, height)
  pdf_surf.finish()

  scale = 2.5
  img_surf = cairo.ImageSurface(
      cairo.FORMAT_ARGB32, int(width * scale), int(height * scale)
  )
  ctx2 = cairo.Context(img_surf)
  ctx2.scale(scale, scale)
  draw_fn(ctx2, width, height)
  img_surf.write_to_png(str(png_path))
  print(f"Generated {pdf_path.name} and {png_path.name}")


def draw_rounded_rect(
    ctx, x, y, w, h, r, fill_rgb, stroke_rgb=None, line_w=1.5
):
  ctx.new_sub_path()
  ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
  ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
  ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
  ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
  ctx.close_path()
  if fill_rgb:
    ctx.set_source_rgb(*fill_rgb)
    if stroke_rgb:
      ctx.fill_preserve()
    else:
      ctx.fill()
  if stroke_rgb:
    ctx.set_source_rgb(*stroke_rgb)
    ctx.set_line_width(line_w)
    ctx.stroke()


def draw_centered_text(
    ctx,
    text,
    cx,
    cy,
    size=10,
    bold=False,
    color=(0.1, 0.1, 0.15),
    family="Sans",
):
  ctx.select_font_face(
      family,
      cairo.FONT_SLANT_NORMAL,
      cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL,
  )
  ctx.set_font_size(size)
  ctx.set_source_rgb(*color)
  lines = text.split("\n")
  line_h = size * 1.28
  total_h = line_h * len(lines)
  start_y = cy - total_h / 2 + size * 0.85
  for i, line in enumerate(lines):
    ext = ctx.text_extents(line)
    ctx.move_to(cx - ext.width / 2 - ext.x_bearing, start_y + i * line_h)
    ctx.show_text(line)


def draw_arrow(ctx, x1, y1, x2, y2, color=(0.2, 0.25, 0.35), lw=1.8):
  ctx.set_source_rgb(*color)
  ctx.set_line_width(lw)
  ctx.move_to(x1, y1)
  ctx.line_to(x2, y2)
  ctx.stroke()

  angle = math.atan2(y2 - y1, x2 - x1)
  head_len = 7.0
  ctx.move_to(x2, y2)
  ctx.line_to(
      x2 - head_len * math.cos(angle - 0.4),
      y2 - head_len * math.sin(angle - 0.4),
  )
  ctx.line_to(
      x2 - head_len * math.cos(angle + 0.4),
      y2 - head_len * math.sin(angle + 0.4),
  )
  ctx.close_path()
  ctx.fill()


# ==============================================================================
# FIGURE 1: ARCNet-v2 System Architecture & Pipeline
# ==============================================================================
def fig1_architecture(ctx, W, H):
  ctx.set_source_rgb(1, 1, 1)
  ctx.paint()

  draw_centered_text(
      ctx,
      "ARCNet-v2: Permutation-Invariant Oklab Color Harmonization Architecture",
      W / 2,
      18,
      size=13,
      bold=True,
  )

  draw_rounded_rect(
      ctx,
      15,
      36,
      W - 30,
      125,
      8,
      (0.96, 0.98, 1.0),
      (0.25, 0.48, 0.75),
      line_w=1.4,
  )
  draw_centered_text(
      ctx,
      "Phase 1 & 2: Aesthetic Corpus Curation, Oklab Quantization & High-Capacity"
      " Set Architecture",
      W / 2,
      48,
      size=10.5,
      bold=True,
      color=(0.12, 0.30, 0.55),
  )

  boxes_top = [
      (
          28,
          64,
          122,
          82,
          "Curated Art Corpus\n86,994 Paintings\n(WikiArt + Best Art)\n360° Hue"
          " Rotation",
          (0.88, 0.93, 0.99),
          (0.2, 0.45, 0.75),
      ),
      (
          168,
          64,
          122,
          82,
          "Perceptual Oklab\nMedian-Cut (K=8)\nBg {(c_k, w_k)} (K-1)\nTarget"
          " c_target",
          (0.88, 0.95, 0.90),
          (0.2, 0.60, 0.35),
      ),
      (
          308,
          64,
          132,
          82,
          "Chromatic Fourier\nFeature Expansion\nOklab + Chroma C\n+ sin/cos(m"
          " theta) [R^40]",
          (0.99, 0.93, 0.85),
          (0.80, 0.45, 0.15),
      ),
      (
          458,
          64,
          140,
          82,
          "Equivariant Encoder\n& Set Transformer\nPre-LN SwiGLU +\nOklab"
          " Contrast Bias",
          (0.94, 0.89, 0.98),
          (0.50, 0.25, 0.70),
      ),
      (
          616,
          64,
          140,
          82,
          "Tri-Modal Pooling &\nMoHE (H=6 Heads)\nSum + Area + Q-Attn\nSoft"
          " Gating -> c_pred",
          (0.99, 0.88, 0.90),
          (0.75, 0.20, 0.30),
      ),
  ]
  for x, y, bw, bh, label, fc, sc in boxes_top:
    draw_rounded_rect(ctx, x, y, bw, bh, 6, fc, sc, line_w=1.4)
    draw_centered_text(ctx, label, x + bw / 2, y + bh / 2, size=8.5, bold=False)

  for i in range(len(boxes_top) - 1):
    x1 = boxes_top[i][0] + boxes_top[i][2]
    x2 = boxes_top[i + 1][0]
    ym = boxes_top[i][1] + boxes_top[i][3] / 2
    draw_arrow(ctx, x1 + 2, ym, x2 - 2, ym)

  draw_rounded_rect(
      ctx,
      15,
      174,
      W - 30,
      112,
      8,
      (0.98, 0.97, 0.94),
      (0.55, 0.45, 0.25),
      line_w=1.4,
  )
  draw_centered_text(
      ctx,
      "Phase 3 & 4: Compound Perceptual-Harmonic Optimization & Real-Time Unity"
      " AR Execution Pipeline",
      W / 2,
      186,
      size=10.5,
      bold=True,
      color=(0.45, 0.30, 0.10),
  )

  boxes_bot = [
      (
          28,
          200,
          170,
          74,
          "Compound Objective Loss\nL_recon (Oklab Huber) + L_chroma\n+ L_dist"
          " (Anti-Camouflage Hinge)\n+ L_harm (Itten Sector) + L_bal",
          (1.0, 0.94, 0.88),
          (0.75, 0.40, 0.15),
      ),
      (
          218,
          200,
          165,
          74,
          "ONNX Export (Opset 14)\narcnet_v2_base_set.onnx\n&"
          " arcnet_v2_base_flat6.onnx\n(573.5K Params, 4.40 ms)",
          (0.90, 0.95, 0.99),
          (0.20, 0.45, 0.75),
      ),
      (
          403,
          200,
          165,
          74,
          "Temporal Stabilization\nFrame-to-Frame EMA Filter\n+ Recurrent GRU"
          " State\n(-81.2% Flicker Reduction)",
          (0.90, 0.97, 0.91),
          (0.20, 0.60, 0.35),
      ),
      (
          588,
          200,
          168,
          74,
          "Unity AR Material Shader\nPreserves 3D Luminance\nShading Relief"
          " (dL, dab)\n& Anti-Camouflage (86.2%)",
          (0.95, 0.90, 0.98),
          (0.50, 0.25, 0.70),
      ),
  ]
  for x, y, bw, bh, label, fc, sc in boxes_bot:
    draw_rounded_rect(ctx, x, y, bw, bh, 6, fc, sc, line_w=1.4)
    draw_centered_text(ctx, label, x + bw / 2, y + bh / 2, size=8.5, bold=False)

  for i in range(len(boxes_bot) - 1):
    x1 = boxes_bot[i][0] + boxes_bot[i][2]
    x2 = boxes_bot[i + 1][0]
    ym = boxes_bot[i][1] + boxes_bot[i][3] / 2
    draw_arrow(ctx, x1 + 2, ym, x2 - 2, ym)


# ==============================================================================
# FIGURE 2: Head-to-Head Ablation Study (ARCNet-v1 vs ARCNet-v2)
# ==============================================================================
def fig2_ablation(ctx, W, H):
  ctx.set_source_rgb(1, 1, 1)
  ctx.paint()

  draw_centered_text(
      ctx,
      "Head-to-Head Ablation on 86,994-Artwork Corpus (13,049 Validation"
      " Samples): ARCNet-v1 vs. ARCNet-v2",
      W / 2,
      18,
      size=12,
      bold=True,
  )

  panels = [
      (
          "Error Metrics (Lower is Better)",
          ["Val Loss", "dE_Oklab", "sRGB MAE", "sRGB MSE"],
          [0.2031, 0.2622, 0.3351, 0.1638],
          [0.0293, 0.0411, 0.0479, 0.0077],
          0.38,
          "{:.4f}",
      ),
      ("PSNR Fidelity (Higher is Better)", ["PSNR (dB)"], [7.86], [21.28], 25.0, "{:.2f}"),
      (
          "Itten Harmonic Error (Lower)",
          ["Angle Err (°)"],
          [18.93],
          [10.40],
          22.5,
          "{:.2f}°",
      ),
      (
          "Temporal Flicker E_temp (Lower)",
          ["Raw vs EMA"],
          [0.0856],
          [0.0161],
          0.105,
          "{:.4f}",
      ),
  ]

  x_offsets = [38, 330, 485, 640]
  widths = [250, 115, 115, 115]
  y_top = 52
  plot_h = 190

  c_v1 = (0.85, 0.28, 0.22)
  c_v2 = (0.18, 0.48, 0.78)

  for p_idx, (title, cats, v1_vals, v2_vals, ymax, fmt) in enumerate(panels):
    px = x_offsets[p_idx]
    pw = widths[p_idx]

    draw_rounded_rect(
        ctx, px, y_top, pw, plot_h, 2, (0.99, 0.99, 1.0), (0.7, 0.7, 0.75), 1.0
    )
    draw_centered_text(ctx, title, px + pw / 2, y_top - 11, size=8.8, bold=True)

    for g in range(1, 5):
      gy = y_top + plot_h - (g / 4.0) * (plot_h - 24)
      ctx.set_source_rgb(0.88, 0.88, 0.90)
      ctx.set_line_width(0.8)
      ctx.move_to(px, gy)
      ctx.line_to(px + pw, gy)
      ctx.stroke()

    n_cats = len(cats)
    group_w = pw / n_cats
    bar_w = min(26.0, group_w * 0.34)

    for i in range(n_cats):
      cx = px + (i + 0.5) * group_w
      h1 = (v1_vals[i] / ymax) * (plot_h - 28)
      h2 = (v2_vals[i] / ymax) * (plot_h - 28)

      by1 = y_top + plot_h - h1
      by2 = y_top + plot_h - h2

      draw_rounded_rect(ctx, cx - bar_w - 2, by1, bar_w, h1, 2, c_v1, None)
      draw_centered_text(
          ctx,
          fmt.format(v1_vals[i]),
          cx - bar_w / 2 - 2,
          by1 - 8,
          size=7.5,
          bold=True,
          color=c_v1,
      )

      draw_rounded_rect(ctx, cx + 2, by2, bar_w, h2, 2, c_v2, None)
      draw_centered_text(
          ctx,
          fmt.format(v2_vals[i]),
          cx + bar_w / 2 + 2,
          by2 - 8,
          size=7.5,
          bold=True,
          color=c_v2,
      )

      draw_centered_text(ctx, cats[i], cx, y_top + plot_h + 13, size=8.5, bold=True)

  draw_rounded_rect(ctx, 210, 266, 14, 10, 2, c_v1)
  draw_centered_text(
      ctx,
      "ARCNet-v1 Baseline (2,627 params / Unsmoothed)",
      335,
      271,
      size=8.8,
      bold=True,
  )
  draw_rounded_rect(ctx, 470, 266, 14, 10, 2, c_v2)
  draw_centered_text(
      ctx,
      "ARCNet-v2 Proposed (573,562 params / EMA)",
      595,
      271,
      size=8.8,
      bold=True,
  )


# ==============================================================================
# FIGURE 3: Cross-Dataset Benchmark Comparison (iHarmony4 & ARHarmony)
# ==============================================================================
def fig3_cross_dataset(ctx, W, H):
  ctx.set_source_rgb(1, 1, 1)
  ctx.paint()

  draw_centered_text(
      ctx,
      "Cross-Dataset Performance Comparison on iHarmony4 and ARHarmony"
      " Benchmarks",
      W / 2,
      18,
      size=12.5,
      bold=True,
  )

  models = [
      "DoveNet",
      "iSSAM",
      "HDNet",
      "DuCoNet",
      "ARCNet-v1",
      "ARCNet-v2 (Ours)",
  ]
  data = {
      "SSIM": (
          [0.803, 0.886, 0.931, 0.930, 0.942, 0.966],
          [0.830, 0.890, 0.951, 0.932, 0.962, 0.977],
          0.75,
          1.00,
          "SSIM (Higher is Better)",
          "{:.3f}",
      ),
      "MAE": (
          [0.1280, 0.1050, 0.0818, 0.0760, 0.0761, 0.0504],
          [0.1262, 0.1015, 0.0752, 0.0772, 0.0750, 0.0493],
          0.0,
          0.145,
          "MAE (Lower is Better)",
          "{:.4f}",
      ),
      "MSE": (
          [52.33, 35.20, 19.32, 18.47, 18.50, 8.13],
          [50.85, 32.90, 18.12, 19.03, 17.97, 7.75],
          0.0,
          58.0,
          "MSE (Lower is Better)",
          "{:.1f}",
      ),
      "PSNR": (
          [30.943, 32.665, 35.272, 35.466, 35.459, 39.032],
          [31.068, 32.959, 35.549, 35.336, 35.585, 39.235],
          28.0,
          41.0,
          "PSNR dB (Higher is Better)",
          "{:.2f}",
      ),
  }

  coords = [
      ("SSIM", 42, 48, 335, 145),
      ("MAE", 418, 48, 335, 145),
      ("MSE", 42, 232, 335, 145),
      ("PSNR", 418, 232, 335, 145),
  ]

  c_ih = (0.28, 0.65, 0.32)
  c_ar = (0.20, 0.48, 0.78)

  for key, px, py, pw, ph in coords:
    ih_vals, ar_vals, ymin, ymax, title, fmt = data[key]
    draw_rounded_rect(
        ctx, px, py, pw, ph, 2, (0.99, 0.99, 1.0), (0.72, 0.72, 0.76), 1.0
    )
    draw_centered_text(ctx, title, px + pw / 2, py - 10, size=9.5, bold=True)

    for g in range(1, 5):
      gy = py + ph - (g / 4.0) * (ph - 22)
      ctx.set_source_rgb(0.88, 0.88, 0.90)
      ctx.set_line_width(0.7)
      ctx.move_to(px, gy)
      ctx.line_to(px + pw, gy)
      ctx.stroke()
      val_g = ymin + (g / 4.0) * (ymax - ymin)
      draw_centered_text(
          ctx, f"{val_g:.2g}", px - 16, gy, size=7.2, color=(0.35, 0.35, 0.4)
      )

    n = len(models)
    gw = pw / n
    bw = gw * 0.36

    for i in range(n):
      cx = px + (i + 0.5) * gw
      h_ih = ((ih_vals[i] - ymin) / (ymax - ymin)) * (ph - 22)
      h_ar = ((ar_vals[i] - ymin) / (ymax - ymin)) * (ph - 22)

      y_ih = py + ph - h_ih
      y_ar = py + ph - h_ar

      draw_rounded_rect(ctx, cx - bw - 1, y_ih, bw, h_ih, 1.5, c_ih)
      draw_rounded_rect(ctx, cx + 1, y_ar, bw, h_ar, 1.5, c_ar)

      draw_centered_text(
          ctx,
          fmt.format(ih_vals[i]),
          cx - bw / 2 - 1,
          y_ih - 6,
          size=6.3,
          bold=(i == n - 1),
          color=(0.15, 0.45, 0.18),
      )
      draw_centered_text(
          ctx,
          fmt.format(ar_vals[i]),
          cx + bw / 2 + 1,
          y_ar - 6,
          size=6.3,
          bold=(i == n - 1),
          color=(0.10, 0.32, 0.58),
      )

      draw_centered_text(
          ctx, models[i], cx, py + ph + 11, size=7.5, bold=(i == n - 1)
      )

  draw_rounded_rect(ctx, 250, 396, 14, 10, 2, c_ih)
  draw_centered_text(ctx, "iHarmony4 Dataset", 315, 401, size=9.0, bold=True)
  draw_rounded_rect(ctx, 430, 396, 14, 10, 2, c_ar)
  draw_centered_text(ctx, "ARHarmony Dataset", 495, 401, size=9.0, bold=True)


# ==============================================================================
# FIGURE 4: Itten Harmonic Sector Distribution (13,049 Validation Samples)
# ==============================================================================
def fig4_itten_distribution(ctx, W, H):
  ctx.set_source_rgb(1, 1, 1)
  ctx.paint()

  draw_centered_text(
      ctx,
      "Predicted Palette Distribution Across Itten Harmonic Sectors (13,049"
      " Validation Artworks)",
      W / 2,
      18,
      size=11.5,
      bold=True,
  )

  sectors = [
      ("Monochromatic\n(0°)", 5498, 42.1, (0.22, 0.49, 0.78)),
      ("Analogous\n(30°)", 5789, 44.4, (0.30, 0.68, 0.42)),
      ("Triadic\n(120°)", 1050, 8.0, (0.95, 0.60, 0.18)),
      ("Split-Compl.\n(150°)", 459, 3.5, (0.65, 0.38, 0.78)),
      ("Complementary\n(180°)", 253, 1.9, (0.86, 0.26, 0.25)),
  ]

  px, py, pw, ph = 55, 48, 420, 185
  draw_rounded_rect(
      ctx, px, py, pw, ph, 3, (0.99, 0.99, 1.0), (0.72, 0.72, 0.76), 1.0
  )

  ymax = 52.0
  for g in range(1, 6):
    gy = py + ph - (g / 5.0) * (ph - 24)
    ctx.set_source_rgb(0.88, 0.88, 0.90)
    ctx.set_line_width(0.8)
    ctx.move_to(px, gy)
    ctx.line_to(px + pw, gy)
    ctx.stroke()
    draw_centered_text(
        ctx, f"{g*10}%", px - 18, gy, size=8.0, color=(0.35, 0.35, 0.4)
    )

  gw = pw / len(sectors)
  bw = gw * 0.52
  for i, (name, count, pct, col) in enumerate(sectors):
    cx = px + (i + 0.5) * gw
    bh = (pct / ymax) * (ph - 24)
    by = py + ph - bh
    draw_rounded_rect(ctx, cx - bw / 2, by, bw, bh, 3, col)
    draw_centered_text(
        ctx,
        f"{pct:.1f}%\n({count:,})",
        cx,
        by - 13,
        size=8.0,
        bold=True,
        color=col,
    )
    draw_centered_text(ctx, name, cx, py + ph + 16, size=8.2, bold=True)

  cx_pie, cy_pie, r_out, r_in = 595, 132, 74, 34
  start_ang = -math.pi / 2
  for name, count, pct, col in sectors:
    sweep = (pct / 100.0) * 2 * math.pi
    ctx.set_source_rgb(*col)
    ctx.move_to(cx_pie, cy_pie)
    ctx.arc(cx_pie, cy_pie, r_out, start_ang, start_ang + sweep)
    ctx.close_path()
    ctx.fill()
    start_ang += sweep

  ctx.set_source_rgb(1, 1, 1)
  ctx.arc(cx_pie, cy_pie, r_in, 0, 2 * math.pi)
  ctx.fill()
  draw_centered_text(
      ctx,
      "Itten Err\n10.40°\n(-45.1%)",
      cx_pie,
      cy_pie,
      size=8.5,
      bold=True,
      color=(0.15, 0.2, 0.3),
  )
  draw_centered_text(
      ctx,
      "86.5% Cohesive (0°–30°)\n13.5% Contrastive (120°–180°)",
      cx_pie,
      235,
      size=8.5,
      bold=True,
      color=(0.2, 0.25, 0.35),
  )


# ==============================================================================
# FIGURE 5: Parameter Efficiency vs. PSNR Pareto Frontier
# ==============================================================================
def fig5_complexity_pareto(ctx, W, H):
  ctx.set_source_rgb(1, 1, 1)
  ctx.paint()

  draw_centered_text(
      ctx,
      "Model Complexity (Log10 Parameters) vs. ARHarmony PSNR (dB) Pareto"
      " Frontier",
      W / 2,
      18,
      size=11.5,
      bold=True,
  )

  px, py, pw, ph = 65, 45, W - 95, H - 85
  draw_rounded_rect(
      ctx, px, py, pw, ph, 3, (0.99, 0.99, 1.0), (0.72, 0.72, 0.76), 1.0
  )

  xmin, xmax = 3.0, 8.0
  ymin, ymax = 30.0, 40.5

  for gx in range(3, 9):
    xx = px + ((gx - xmin) / (xmax - xmin)) * pw
    ctx.set_source_rgb(0.88, 0.88, 0.90)
    ctx.set_line_width(0.8)
    ctx.move_to(xx, py)
    ctx.line_to(xx, py + ph)
    ctx.stroke()
    labels_x = {
        3: "1K",
        4: "10K",
        5: "100K",
        6: "1M",
        7: "10M",
        8: "100M",
    }
    draw_centered_text(
        ctx, labels_x[gx], xx, py + ph + 14, size=8.5, bold=True
    )

  for gy_val in [31, 33, 35, 37, 39]:
    yy = py + ph - ((gy_val - ymin) / (ymax - ymin)) * ph
    ctx.set_source_rgb(0.88, 0.88, 0.90)
    ctx.set_line_width(0.8)
    ctx.move_to(px, yy)
    ctx.line_to(px + pw, yy)
    ctx.stroke()
    draw_centered_text(
        ctx, f"{gy_val} dB", px - 24, yy, size=8.2, color=(0.3, 0.3, 0.35)
    )

  points = [
      (
          "DoveNet (54.8M)",
          math.log10(54.8e6),
          31.068,
          (0.75, 0.30, 0.30),
          6.5,
          0,
          -14,
      ),
      (
          "iSSAM (39.2M)",
          math.log10(39.2e6),
          32.959,
          (0.80, 0.50, 0.20),
          6.5,
          0,
          -14,
      ),
      (
          "DuCoNet (31.5M)",
          math.log10(31.5e6),
          35.336,
          (0.55, 0.40, 0.75),
          6.5,
          18,
          16,
      ),
      (
          "HDNet (14.7M)",
          math.log10(14.7e6),
          35.549,
          (0.20, 0.58, 0.62),
          6.5,
          -16,
          -14,
      ),
      (
          "ARCNet-v1 (2.6K)",
          math.log10(2627),
          35.585,
          (0.45, 0.45, 0.50),
          6.5,
          0,
          -14,
      ),
      (
          "ARCNet-v2-Mobile (41.2K)",
          math.log10(41200),
          38.150,
          (0.20, 0.68, 0.38),
          7.5,
          0,
          -15,
      ),
      (
          "ARCNet-v2-Base (573.6K, Ours)",
          math.log10(573562),
          39.235,
          (0.12, 0.42, 0.82),
          9.0,
          0,
          -16,
      ),
  ]

  for label, lx, ly, col, rad, dx, dy in points:
    xx = px + ((lx - xmin) / (xmax - xmin)) * pw
    yy = py + ph - ((ly - ymin) / (ymax - ymin)) * ph
    ctx.set_source_rgb(*col)
    ctx.arc(xx, yy, rad, 0, 2 * math.pi)
    ctx.fill()
    draw_centered_text(ctx, label, xx + dx, yy + dy, size=8.2, bold=True, color=col)

  draw_centered_text(
      ctx,
      "Model Parameter Count (Logarithmic Scale)",
      px + pw / 2,
      py + ph + 30,
      size=9.5,
      bold=True,
  )


if __name__ == "__main__":
  render_both("fig1_arcnet_v2_pipeline", 780, 300, fig1_architecture)
  render_both("ablation_v1_vs_v2", 780, 290, fig2_ablation)
  render_both("cross_dataset_comparison", 780, 420, fig3_cross_dataset)
  render_both("itten_harmony_distribution", 700, 260, fig4_itten_distribution)
  render_both("model_complexity_pareto", 680, 265, fig5_complexity_pareto)
