"""
Step 2: Large-Scale Aesthetic Data Curation & Dominant Palette Preprocessing
for ARCNet-v2 (ARCNet_V2_roadmap.pdf Section 2.1 - 2.3).

Features:
---------
1. Zero-Disk-Space Hugging Face Streaming (`--hf-stream`):
   Streams images directly in RAM from Hugging Face (e.g., `huggan/wikiart`)
   without downloading 35+ GB of raw JPEGs to disk, extracting K=8 Oklab/sRGB
   dominant color palettes on the fly into a compact ~15 MB `.npz` file.
2. Automatic Kaggle Dataset Download (`--kaggle-datasets`):
   Uses `kagglehub` to automatically fetch datasets like
   `ikarus777/best-artworks-of-all-time` and preprocess them.
3. Generalized Recursive Median-Cut Quantization (MCQ) in Oklab or sRGB (`K in [4, 8]`):
   - Splits color boxes along channel c* = argmax_{c in {L,a,b}} (max(c) - min(c))
   - Computes centroid c_k and relative population weight w_k = |B_k| / N.
   - Automatically filters out monochrome/grayscale sketches and blank studio backgrounds.
4. Multi-core parallel processing across local folders, Kaggle datasets, and HF streams.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError:
    Image = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


# =====================================================================
# Vectorized NumPy sRGB <-> Oklab Conversions (for CPU Multiprocessing)
# =====================================================================

def np_srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB in [0, 1] to linear RGB in [0, 1]."""
    srgb = np.clip(srgb, 0.0, 1.0)
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def np_linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB in [0, 1] to sRGB in [0, 1]."""
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(np.maximum(linear, 1e-8), 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def np_srgb_to_oklab(srgb: np.ndarray) -> np.ndarray:
    """
    Convert sRGB array [..., 3] in [0, 1] to Oklab (L, a, b) [..., 3].
    Matches ARCNet_V2_roadmap.pdf Section 2.2.
    """
    lin = np_srgb_to_linear(srgb)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]

    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_ = np.cbrt(np.maximum(l, 1e-8))
    m_ = np.cbrt(np.maximum(m, 1e-8))
    s_ = np.cbrt(np.maximum(s, 1e-8))

    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_out = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return np.stack([L, a, b_out], axis=-1).astype(np.float32)


def np_oklab_to_srgb(oklab: np.ndarray) -> np.ndarray:
    """Convert Oklab (L, a, b) [..., 3] to sRGB [..., 3] in [0, 1]."""
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

    lin = np.stack([r, g, b_lin], axis=-1)
    return np_linear_to_srgb(lin)


# =====================================================================
# Generalized Median-Cut Quantization (MCQ) in Oklab / sRGB Space
# =====================================================================

def median_cut_quantize(
    pixels_srgb: np.ndarray,
    num_clusters: int = 8,
    quantize_space: str = "oklab",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generalized Median-Cut Quantization (MCQ) into K representative color clusters
    (ARCNet_V2_roadmap.pdf Section 2.3, Equations 2-3).
    """
    N = len(pixels_srgb)
    if N == 0:
        raise ValueError("Empty pixel array passed to median_cut_quantize.")

    pixels_oklab = np_srgb_to_oklab(pixels_srgb)
    work_pixels = pixels_oklab if quantize_space == "oklab" else pixels_srgb

    boxes: List[np.ndarray] = [np.arange(N, dtype=np.int32)]

    while len(boxes) < num_clusters:
        best_box_idx = -1
        best_channel = 0
        best_score = -1.0

        for idx, box_indices in enumerate(boxes):
            if len(box_indices) < 2:
                continue
            pts = work_pixels[box_indices]
            ranges = pts.max(axis=0) - pts.min(axis=0)
            c_star = int(np.argmax(ranges))
            max_range = float(ranges[c_star])
            if max_range <= 1e-7:
                continue
            score = max_range * np.sqrt(len(box_indices) / float(N))
            if score > best_score:
                best_score = score
                best_box_idx = idx
                best_channel = c_star

        if best_box_idx == -1:
            boxes.append(boxes[0].copy())
            continue

        target_indices = boxes.pop(best_box_idx)
        channel_vals = work_pixels[target_indices, best_channel]
        order = np.argsort(channel_vals, kind="mergesort")
        sorted_indices = target_indices[order]

        mid = len(sorted_indices) // 2
        boxes.append(sorted_indices[:mid])
        boxes.append(sorted_indices[mid:])

    # Rank color boxes in descending order of population: |B_(1)| >= |B_(2)| >= ... >= |B_(K)|
    boxes.sort(key=lambda b: len(b), reverse=True)

    centroids_srgb = np.zeros((num_clusters, 3), dtype=np.float32)
    centroids_oklab = np.zeros((num_clusters, 3), dtype=np.float32)
    weights = np.zeros((num_clusters,), dtype=np.float32)

    for k, box_indices in enumerate(boxes):
        if len(box_indices) == 0:
            centroids_srgb[k] = centroids_srgb[max(0, k - 1)]
            centroids_oklab[k] = centroids_oklab[max(0, k - 1)]
            weights[k] = 0.0
        else:
            c_oklab = pixels_oklab[box_indices].mean(axis=0)
            c_srgb = pixels_srgb[box_indices].mean(axis=0)
            centroids_oklab[k] = c_oklab
            centroids_srgb[k] = np.clip(c_srgb, 0.0, 1.0)
            weights[k] = len(box_indices) / float(N)

    weight_sum = weights.sum()
    if weight_sum > 0:
        weights /= weight_sum
    else:
        weights[:] = 1.0 / num_clusters

    return centroids_srgb, centroids_oklab, weights


def extract_palette_from_pil_image(
    img: "Image.Image",
    num_clusters: int = 8,
    resize_dim: int = 128,
    quantize_space: str = "oklab",
    border_crop: float = 0.03,
    min_chroma: float = 0.025,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Core extractor: converts a PIL Image into K ranked Oklab/sRGB clusters and PBR targets.
    """
    img = img.convert("RGB")
    w, h = img.size
    if w < 16 or h < 16:
        return None

    # Optional border crop to remove scanned picture frames/margins
    if border_crop > 0.0:
        dx = int(w * border_crop)
        dy = int(h * border_crop)
        if w - 2 * dx >= 16 and h - 2 * dy >= 16:
            img = img.crop((dx, dy, w - dx, h - dy))

    img = img.resize((resize_dim, resize_dim), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    pixels = arr.reshape(-1, 3)

    centroids_srgb, centroids_oklab, weights = median_cut_quantize(
        pixels, num_clusters=num_clusters, quantize_space=quantize_space
    )

    # Filter out grayscale/monochrome sketches or plain studio backgrounds
    cluster_chroma = np.sqrt(centroids_oklab[:, 1] ** 2 + centroids_oklab[:, 2] ** 2)
    if float(cluster_chroma.max()) < min_chroma:
        return None

    bg_srgb = centroids_srgb[:-1]      # [K-1, 3]
    bg_oklab = centroids_oklab[:-1]    # [K-1, 3]
    bg_weights = weights[:-1]          # [K-1]
    bg_weights = bg_weights / np.maximum(bg_weights.sum(), 1e-6)

    target_srgb = centroids_srgb[-1]   # [3]
    target_oklab = centroids_oklab[-1] # [3]

    secondary_srgb = centroids_srgb[-2] if num_clusters >= 3 else target_srgb
    oklab_all = np_srgb_to_oklab(pixels)
    lum_std = float(np.std(oklab_all[:, 0]))
    tau_roughness = np.float32(np.clip(1.0 - (lum_std / 0.25), 0.05, 0.95))
    dc_specular = np.clip(centroids_srgb[0] - target_srgb, -0.25, 0.25) * 0.25

    pbr_target = np.concatenate(
        [target_srgb, secondary_srgb, dc_specular, [tau_roughness]], axis=0
    ).astype(np.float32)

    return {
        "all_srgb": centroids_srgb,
        "all_oklab": centroids_oklab,
        "all_weights": weights,
        "bg_srgb": bg_srgb,
        "bg_oklab": bg_oklab,
        "bg_weights": bg_weights,
        "target_srgb": target_srgb,
        "target_oklab": target_oklab,
        "pbr_target": pbr_target,
    }


def process_single_artwork_path(
    task: Tuple[str, int, int, str, float]
) -> Optional[Dict[str, np.ndarray]]:
    """Worker for local/Kaggle image file paths."""
    img_path, num_clusters, resize_dim, quantize_space, border_crop = task
    if Image is None:
        raise RuntimeError("Pillow (PIL) is required. Run: pip install pillow")
    try:
        with Image.open(img_path) as img:
            return extract_palette_from_pil_image(
                img=img,
                num_clusters=num_clusters,
                resize_dim=resize_dim,
                quantize_space=quantize_space,
                border_crop=border_crop,
            )
    except Exception:
        return None


def process_single_image_bytes(
    task: Tuple[bytes, int, int, str, float]
) -> Optional[Dict[str, np.ndarray]]:
    """Worker for streamed image bytes (from Hugging Face streaming datasets)."""
    img_bytes, num_clusters, resize_dim, quantize_space, border_crop = task
    if Image is None:
        raise RuntimeError("Pillow (PIL) is required. Run: pip install pillow")
    try:
        with Image.open(io.BytesIO(img_bytes)) as img:
            return extract_palette_from_pil_image(
                img=img,
                num_clusters=num_clusters,
                resize_dim=resize_dim,
                quantize_space=quantize_space,
                border_crop=border_crop,
            )
    except Exception:
        return None


# =====================================================================
# Automated Dataset Fetching: HuggingFace Streaming & KaggleHub
# =====================================================================

def download_kaggle_datasets(dataset_handles: List[str]) -> List[str]:
    """
    Automatically downloads Kaggle datasets using `kagglehub` and returns their
    local cache directories.
    Example handles:
      - 'ikarus777/best-artworks-of-all-time'
      - 'steubk/wikiart'
    """
    try:
        import kagglehub
    except ImportError:
        raise RuntimeError(
            "kagglehub is required for --kaggle-datasets. Install via: pip install kagglehub"
        )

    downloaded_dirs: List[str] = []
    for handle in dataset_handles:
        print(f"[KaggleHub] Downloading / verifying dataset '{handle}'...")
        local_dir = kagglehub.dataset_download(handle)
        print(f"[KaggleHub] Ready at: {local_dir}")
        downloaded_dirs.append(local_dir)
    return downloaded_dirs


def stream_huggingface_datasets(
    hf_repos: List[str],
    num_clusters: int = 8,
    resize_dim: int = 128,
    quantize_space: str = "oklab",
    border_crop: float = 0.03,
    max_images_per_repo: int = 50000,
    num_workers: int = 4,
) -> List[Dict[str, np.ndarray]]:
    """
    Streams images directly in RAM from Hugging Face `datasets` without saving
    raw images to disk, extracting K dominant Oklab clusters in parallel batches.

    Recommended repos:
      - 'huggan/wikiart' (~81,444 fine art paintings)
      - 'Artificio/WikiArt'
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "Hugging Face `datasets` is required for --hf-stream. Install via: pip install datasets"
        )

    all_results: List[Dict[str, np.ndarray]] = []
    batch_size = max(64, num_workers * 16)

    for repo_id in hf_repos:
        print(f"\n[HuggingFace Stream] Connecting to '{repo_id}' (streaming=True, zero local disk usage)...")
        ds = load_dataset(repo_id, split="train", streaming=True)

        count = 0
        pending_bytes: List[Tuple[bytes, int, int, str, float]] = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            pbar = tqdm(total=max_images_per_repo if max_images_per_repo > 0 else None, desc=f"Streaming {repo_id}")
            for sample in ds:
                # Find the image field in the HF sample dict
                img_obj: Any = sample.get("image") or sample.get("img") or sample.get("jpg")
                if img_obj is None:
                    for val in sample.values():
                        if isinstance(val, Image.Image) or (isinstance(val, dict) and "bytes" in val):
                            img_obj = val
                            break
                if img_obj is None:
                    continue

                try:
                    if isinstance(img_obj, dict) and "bytes" in img_obj and img_obj["bytes"]:
                        raw_bytes = img_obj["bytes"]
                    elif isinstance(img_obj, Image.Image):
                        # Downsample in main thread before serializing to worker for minimal IPC overhead
                        w, h = img_obj.size
                        if w < 16 or h < 16:
                            continue
                        buf = io.BytesIO()
                        img_obj.convert("RGB").resize((resize_dim, resize_dim), Image.Resampling.BILINEAR).save(
                            buf, format="JPEG", quality=95
                        )
                        raw_bytes = buf.getvalue()
                    else:
                        continue
                except Exception:
                    continue

                pending_bytes.append((raw_bytes, num_clusters, resize_dim, quantize_space, border_crop))
                count += 1
                pbar.update(1)

                if len(pending_bytes) >= batch_size:
                    futures = [executor.submit(process_single_image_bytes, t) for t in pending_bytes]
                    for fut in as_completed(futures):
                        res = fut.result()
                        if res is not None:
                            all_results.append(res)
                    pending_bytes.clear()

                if 0 < max_images_per_repo <= count:
                    break

            if pending_bytes:
                futures = [executor.submit(process_single_image_bytes, t) for t in pending_bytes]
                for fut in as_completed(futures):
                    res = fut.result()
                    if res is not None:
                        all_results.append(res)
                pending_bytes.clear()
            pbar.close()

        print(f"[HuggingFace Stream] Extracted {len(all_results):,d} valid harmonic palettes so far.")

    return all_results


# =====================================================================
# Dataset Discovery & Saving
# =====================================================================

def discover_images(data_dirs: List[str], max_images: int = 0) -> List[str]:
    """Recursively find all valid image files across one or more dataset directories."""
    image_paths: List[str] = []
    for d in data_dirs:
        root = Path(d)
        if not root.exists():
            raise FileNotFoundError(f"Input directory does not exist: {d}")
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(str(p))
                if 0 < max_images <= len(image_paths):
                    return image_paths
    image_paths.sort()
    return image_paths


def save_results_to_npz(
    results: List[Dict[str, np.ndarray]],
    output_path: str,
    val_split: float = 0.15,
    seed: int = 42,
):
    if not results:
        raise RuntimeError("No valid palettes were extracted.")

    all_srgb = np.stack([r["all_srgb"] for r in results], axis=0)
    all_oklab = np.stack([r["all_oklab"] for r in results], axis=0)
    all_weights = np.stack([r["all_weights"] for r in results], axis=0)
    bg_srgb = np.stack([r["bg_srgb"] for r in results], axis=0)
    bg_oklab = np.stack([r["bg_oklab"] for r in results], axis=0)
    bg_weights = np.stack([r["bg_weights"] for r in results], axis=0)
    target_srgb = np.stack([r["target_srgb"] for r in results], axis=0)
    target_oklab = np.stack([r["target_oklab"] for r in results], axis=0)
    pbr_target = np.stack([r["pbr_target"] for r in results], axis=0)

    _save_split_npz(
        output_path=output_path,
        all_srgb=all_srgb,
        all_oklab=all_oklab,
        all_weights=all_weights,
        bg_srgb=bg_srgb,
        bg_oklab=bg_oklab,
        bg_weights=bg_weights,
        target_srgb=target_srgb,
        target_oklab=target_oklab,
        pbr_target=pbr_target,
        val_split=val_split,
        seed=seed,
    )


def _save_split_npz(
    output_path: str,
    all_srgb: np.ndarray,
    all_oklab: np.ndarray,
    all_weights: np.ndarray,
    bg_srgb: np.ndarray,
    bg_oklab: np.ndarray,
    bg_weights: np.ndarray,
    target_srgb: np.ndarray,
    target_oklab: np.ndarray,
    pbr_target: np.ndarray,
    val_split: float = 0.15,
    seed: int = 42,
):
    N = len(all_srgb)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    val_size = max(1, int(N * val_split))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_file,
        train_all_srgb=all_srgb[train_idx],
        train_all_oklab=all_oklab[train_idx],
        train_all_weights=all_weights[train_idx],
        train_bg_srgb=bg_srgb[train_idx],
        train_bg_oklab=bg_oklab[train_idx],
        train_bg_weights=bg_weights[train_idx],
        train_target_srgb=target_srgb[train_idx],
        train_target_oklab=target_oklab[train_idx],
        train_pbr_target=pbr_target[train_idx],
        val_all_srgb=all_srgb[val_idx],
        val_all_oklab=all_oklab[val_idx],
        val_all_weights=all_weights[val_idx],
        val_bg_srgb=bg_srgb[val_idx],
        val_bg_oklab=bg_oklab[val_idx],
        val_bg_weights=bg_weights[val_idx],
        val_target_srgb=target_srgb[val_idx],
        val_target_oklab=target_oklab[val_idx],
        val_pbr_target=pbr_target[val_idx],
    )
    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"\nSaved preprocessed dataset to: {out_file} ({size_mb:.2f} MB)")
    print(f"  Total valid samples : {N:,d} (Train: {len(train_idx):,d}, Val: {len(val_idx):,d})")
    print(f"  Clusters per sample : K={all_srgb.shape[1]} (K-1={bg_srgb.shape[1]} background + 1 target)")


def generate_synthetic_demo_dataset(
    output_path: str,
    num_samples: int = 5000,
    num_clusters: int = 8,
    seed: int = 42,
):
    """Generates a synthetic Oklab harmonic palette dataset (.npz) for fast verification."""
    rng = np.random.default_rng(seed)
    K = num_clusters
    K_bg = K - 1

    L_bg = rng.uniform(0.25, 0.85, size=(num_samples, K_bg)).astype(np.float32)
    C_bg = rng.uniform(0.04, 0.22, size=(num_samples, K_bg)).astype(np.float32)
    base_hue = rng.uniform(-np.pi, np.pi, size=(num_samples, 1)).astype(np.float32)
    hue_offsets = rng.normal(0.0, 0.35, size=(num_samples, K_bg)).astype(np.float32)
    h_bg = base_hue + hue_offsets

    a_bg = C_bg * np.cos(h_bg)
    b_bg = C_bg * np.sin(h_bg)
    bg_oklab = np.stack([L_bg, a_bg, b_bg], axis=-1).astype(np.float32)
    bg_srgb = np_oklab_to_srgb(bg_oklab)

    raw_w = rng.exponential(scale=1.0, size=(num_samples, K)).astype(np.float32)
    raw_w.sort(axis=1)
    all_weights = raw_w[:, ::-1]
    all_weights /= all_weights.sum(axis=1, keepdims=True)
    bg_weights = all_weights[:, :K_bg]
    bg_weights = bg_weights / bg_weights.sum(axis=1, keepdims=True)

    harmonic_shifts = np.array([np.pi, 2.0 * np.pi / 3.0, 5.0 * np.pi / 6.0, np.pi / 6.0], dtype=np.float32)
    mode_idx = rng.integers(0, len(harmonic_shifts), size=(num_samples,))
    target_hue = base_hue[:, 0] + harmonic_shifts[mode_idx] + rng.normal(0.0, 0.05, size=(num_samples,)).astype(np.float32)

    mean_bg_L = (L_bg * bg_weights).sum(axis=1)
    target_L = np.where(mean_bg_L > 0.55, mean_bg_L - 0.25, mean_bg_L + 0.25).astype(np.float32)
    target_L = np.clip(target_L, 0.2, 0.9)
    target_C = np.clip(C_bg[:, 0] * 1.15, 0.08, 0.26).astype(np.float32)

    target_oklab = np.stack(
        [target_L, target_C * np.cos(target_hue), target_C * np.sin(target_hue)],
        axis=-1,
    ).astype(np.float32)
    target_srgb = np_oklab_to_srgb(target_oklab)

    all_srgb = np.concatenate([bg_srgb, target_srgb[:, None, :]], axis=1)
    all_oklab = np.concatenate([bg_oklab, target_oklab[:, None, :]], axis=1)

    secondary_srgb = bg_srgb[:, -1, :]
    dc_spec = np.clip(bg_srgb[:, 0, :] - target_srgb, -0.25, 0.25) * 0.25
    roughness = np.full((num_samples, 1), 0.5, dtype=np.float32)
    pbr_target = np.concatenate([target_srgb, secondary_srgb, dc_spec, roughness], axis=-1).astype(np.float32)

    _save_split_npz(
        output_path=output_path,
        all_srgb=all_srgb,
        all_oklab=all_oklab,
        all_weights=all_weights,
        bg_srgb=bg_srgb,
        bg_oklab=bg_oklab,
        bg_weights=bg_weights,
        target_srgb=target_srgb,
        target_oklab=target_oklab,
        pbr_target=pbr_target,
        val_split=0.15,
        seed=seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 2: Automated Dataset Fetcher & Oklab MCQ Preprocessor for ARCNet-v2"
    )
    parser.add_argument(
        "--auto-curate",
        type=str,
        default="",
        choices=["", "baseline", "wikiart-stream", "full"],
        help=(
            "One-command automated curation: "
            "'baseline' = Best Artworks (8.3k via KaggleHub), "
            "'wikiart-stream' = Stream WikiArt (81k via HuggingFace, zero disk space), "
            "'full' = Best Artworks + Streamed WikiArt (~90k images)"
        ),
    )
    parser.add_argument(
        "--hf-stream",
        nargs="+",
        default=[],
        help="Hugging Face dataset repo IDs to stream in RAM (e.g., huggan/wikiart)",
    )
    parser.add_argument(
        "--kaggle-datasets",
        nargs="+",
        default=[],
        help="Kaggle dataset handles to auto-download via kagglehub (e.g., ikarus777/best-artworks-of-all-time)",
    )
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        default=[],
        help="Local directories containing images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/arcnet_v2_palettes.npz",
        help="Output .npz file path for extracted dominant palettes",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=8,
        help="Number of dominant color clusters K in [4, 8] (default: 8)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=128,
        help="Downsample resolution for MCQ extraction (default: 128)",
    )
    parser.add_argument(
        "--space",
        type=str,
        default="oklab",
        choices=["oklab", "srgb"],
        help="Color space used for median-cut splitting (default: oklab)",
    )
    parser.add_argument(
        "--border-crop",
        type=float,
        default=0.03,
        help="Fraction of image border to crop out (default: 0.03)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation split fraction (default: 0.15)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel CPU processes (0 = auto)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional cap on number of images per source (0 = all)",
    )
    parser.add_argument(
        "--generate-synthetic-demo",
        action="store_true",
        help="Generate a synthetic Oklab harmonic palette dataset (.npz) immediately",
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=5000,
        help="Number of samples when using --generate-synthetic-demo",
    )
    args = parser.parse_args()

    num_workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 4) - 1)

    if args.generate_synthetic_demo:
        generate_synthetic_demo_dataset(
            output_path=args.output,
            num_samples=args.synthetic_samples,
            num_clusters=args.clusters,
        )
        raise SystemExit(0)

    # Apply --auto-curate presets
    hf_repos = list(args.hf_stream)
    kaggle_handles = list(args.kaggle_datasets)
    local_dirs = list(args.data_dirs)

    # If run with no dataset arguments at all, default to 'full' (Best Artworks + WikiArt stream)
    auto_mode = args.auto_curate
    if not auto_mode and not hf_repos and not kaggle_handles and not local_dirs:
        print("[Auto-Default] No arguments specified -> running 'full' pipeline (Best Artworks + WikiArt stream)...")
        auto_mode = "full"

    if auto_mode == "baseline":
        kaggle_handles.append("ikarus777/best-artworks-of-all-time")
    elif auto_mode == "wikiart-stream":
        hf_repos.append("huggan/wikiart")
    elif auto_mode == "full":
        kaggle_handles.append("ikarus777/best-artworks-of-all-time")
        hf_repos.append("huggan/wikiart")

    combined_results: List[Dict[str, np.ndarray]] = []

    # 1. Auto-download Kaggle datasets if requested
    if kaggle_handles:
        try:
            downloaded = download_kaggle_datasets(kaggle_handles)
            local_dirs.extend(downloaded)
        except Exception as e:
            print(f"[KaggleHub Warning] Skipping Kaggle download ({e}). Continuing with remaining sources...")

    # 2. Process local / Kaggle directories
    if local_dirs:
        image_paths = discover_images(local_dirs, max_images=args.max_images)
        print(f"Discovered {len(image_paths):,d} local/Kaggle images.")
        tasks = [
            (p, args.clusters, args.resize, args.space, args.border_crop)
            for p in image_paths
        ]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_single_artwork_path, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Local/Kaggle MCQ"):
                res = fut.result()
                if res is not None:
                    combined_results.append(res)

    # 3. Stream Hugging Face datasets in RAM (zero disk storage for raw images)
    if hf_repos:
        stream_res = stream_huggingface_datasets(
            hf_repos=hf_repos,
            num_clusters=args.clusters,
            resize_dim=args.resize,
            quantize_space=args.space,
            border_crop=args.border_crop,
            max_images_per_repo=args.max_images,
            num_workers=num_workers,
        )
        combined_results.extend(stream_res)

    if combined_results:
        save_results_to_npz(
            results=combined_results,
            output_path=args.output,
            val_split=args.val_split,
        )
    else:
        raise RuntimeError("No palettes were extracted. Check your network connection or dataset paths.")
