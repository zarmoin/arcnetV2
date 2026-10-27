# ARCNet-v2: High-Capacity Permutation-Invariant Model for Large-Scale Datasets

## 1. Inspection Summary of Existing Models in `color_recognition_unity`

1. **`color_recognition_unity/PythonServer/model.h5`**:
   - **What it is**: A legacy Keras `Sequential` CNN model (`Input(784) -> Reshape(28, 28, 1) -> Conv2D(32, 5x5, ReLU) -> MaxPooling2D(2x2) -> Flatten -> Dense(512, ReLU, MaxNorm=3) -> Dropout(0.5) -> Dense(62, Softmax)`) trained with `categorical_crossentropy` for 28×28 grayscale character recognition (EMNIST 62-class template used in the Unity ZMQ example).
2. **`color_recognition_unity/Assets/mlp_model.onnx` & `custom_model.onnx`**:
   - **What they are**: The PyTorch 2.5.1 exported ONNX models loaded by `Assets/RunMLP.cs` via Unity Barracuda (`Input(1, 6) -> fc1(6, 256) -> LeakyReLU -> Norm -> fc2(256, 256) -> LeakyReLU -> Norm -> fc3(256, 128) -> LeakyReLU -> fc4(128, 3)`).
3. **Original ARCNet Paper (`DL_based_harmonization.pdf`)**:
   - Uses a shallow 3-layer MLP (`6 -> 64 -> 32 -> 3` with ReLU + ELU($\alpha=0.1$), **2,339 parameters**) taking $n-1=2$ dominant sRGB colors extracted via Median-Cut Quantization (MCQ) on 8,355 paintings (*Best Artworks of All Time*).

---

## 2. Why Original ARCNet Faces Underfitting on Larger Datasets (10×–50× Scale)

When scaling training data from 8,355 artworks to 100,000–650,000+ images (WikiArt, The Met Open Access, AVA, iHarmony4) as planned in `ARCNet_V2_roadmap.pdf`:

1. **Severe Parameter & Bottleneck Under-Capacity**:
   - The 3-layer `6 -> 64 -> 32 -> 3` MLP has only ~2.3k parameters and no residual pathways, causing gradient saturation and inability to memorize or generalize across diverse artistic periods and real-world lighting conditions.
2. **Spectral Bias on Low-Dimensional Color Coordinates**:
   - Standard MLPs fed raw 3D $(R, G, B)$ scalars suffer from *spectral bias*, learning only overly smooth, low-frequency functions and failing to capture sharp angular harmonic boundaries on the color wheel.
3. **Input-Order Sensitivity & Fixed $n=3$ Palette Bottleneck**:
   - Flattening 2 ranked RGB vectors into $\mathbb{R}^6$ discards secondary background clusters and is sensitive to rank swaps when two clusters have similar pixel populations.
4. **Multi-Modal Target Averaging ("Gray-Mode Collapse")**:
   - Across 100k+ artworks, the same background palette can be harmonized via multiple valid strategies (Complementary $180^\circ$, Split-Complementary $150^\circ$, Triadic $120^\circ$, Analogous $30^\circ$). A single deterministic linear output head trained with MSE averages these distinct modes together, predicting desaturated, muddy colors.

---

## 3. Architectural Modifications in `ARCNet_v2_model`

| Component | Original ARCNet | Modified `ARCNetV2` (`ARCNet_v2_model/`) | How It Prevents Underfitting |
| :--- | :--- | :--- | :--- |
| **Color Space & Input Features** | Raw sRGB $\mathbb{R}^6$ | Differentiable **Oklab** + `ChromaticFeatureExpansion` (Chroma, Circular Hue Harmonics $\sin(m\theta), \cos(m\theta)$ for $m=1..6$, Multi-Band Learnable Fourier Features, Area Weights $w_k, \log w_k$) | Eliminates perceptual non-uniformity and overcomes MLP spectral bias on low-dimensional color inputs. |
| **Backbone Architecture** | Plain MLP (`Dense -> ReLU`) | **Equivariant DeepSets ($\phi$) + Micro-Set Transformer (`SetAttentionBlock`) + Pre-LN `ResSwiGLUBlock` ($\rho$) with LayerScale** | Strictly permutation-invariant across dynamic $K \in [4, 8]$ clusters; pairwise Oklab contrast bias models inter-cluster contrast; SwiGLU + Pre-LN scales cleanly to deep/wide networks. |
| **Set Aggregation** | Concatenation (order-sensitive) | `WeightedSetPooling` (DeepSets Sum + Area-Weighted Sum + Multi-Query Attention Pooling) | Preserves both large background color masses and small, high-chroma accent clusters. |
| **Output Head** | Single Linear + ELU (`32 -> 3`) | **`HarmonicMixtureHead` (Mixture of $M$ Harmonic Experts)** + Gated Background Anchor Residual | Prevents multi-modal averaging (muddy/gray predictions) by routing across $M$ specialized harmonic experts. |
| **Output Modes** | 1D Uniform Tint $[R, G, B]$ | Both `single_color` ($\mathbb{R}^3$) and `multi_material` PBR ($\mathbb{R}^{10}$: $[c_{\text{primary}}, c_{\text{secondary}}, \Delta c_{\text{specular}}, \tau_{\text{roughness}}]$) | Supports full PBR material adaptation in Unity (`MaterialPropertyBlock`) as well as backward compatibility with `RunMLP.cs`. |
| **Lighting & Temporal** | None | Optional `FiLMLightingConditioner` (ARCore Spherical Harmonics) + `TemporalGRUHead` | Adapts palettes to ambient HDR illumination and stabilizes frame-to-frame predictions. |

---

## 4. Model Capacity Presets (`arcnet_v2.py`)

Choose the preset matching your dataset size and runtime target:

- `arcnet_v2_mobile()`: **~42k parameters** (strictly within the $\le 45\text{k}$ mobile NPU budget in `ARCNet_V2_roadmap.pdf` Section 3.1; ~18× larger capacity than ARCNet-v1).
- `arcnet_v2_base()`: **~175k parameters** (recommended default for 50k–150k images; $< 2.5\text{ ms}$ on Unity Sentis).
- `arcnet_v2_large()`: **~680k parameters** (recommended for 150k–650k+ images across WikiArt + The Met + AVA).
- `arcnet_v2_xl()`: **~1.8M parameters** (high-capacity teacher model or desktop VR/AR).

## 5. Files in This Folder

- [`color_space.py`](./color_space.py): Differentiable `srgb_to_oklab`, `oklab_to_srgb`, `oklab_to_oklch`, and `ChromaticFeatureExpansion`.
- [`arcnet_v2.py`](./arcnet_v2.py): Core `ARCNetV2` architecture and capacity presets (`arcnet_v2_mobile`, `arcnet_v2_base`, `arcnet_v2_large`, `arcnet_v2_xl`).
- [`losses.py`](./losses.py): Multi-objective `ARCNetV2Loss` ($\mathcal{L}_{\text{recon}} + \lambda_1 \mathcal{L}_{\text{dist}} + \lambda_2 \mathcal{L}_{\text{harm}} + \lambda_{\text{chroma}} \mathcal{L}_{\text{chroma}} + \lambda_{\text{bal}} \mathcal{L}_{\text{balance}}$).
- [`preprocess_dataset.py`](./preprocess_dataset.py): Parallel multi-core preprocessor implementing Generalized Oklab Median-Cut Quantization (MCQ) for $K \in [4, 8]$ clusters across large artwork/photography folders (WikiArt, The Met, AVA) or composite mask pairs (iHarmony4), plus Hugging Face zero-disk streaming and KaggleHub auto-download.
- [`dataset.py`](./dataset.py): PyTorch `ARCNetPaletteDataset` and `create_dataloaders` with anti-underfitting color augmentations (Oklab $360^\circ$ hue-circle rotation, dynamic $K \in [4, 8]$ cluster masking, leave-one-out target sampling, and random cluster permutation).
- [`train.py`](./train.py): Turnkey training pipeline with decoupled `AdamW`, Cosine Annealing + Warmup, EMA model weights, automatic ONNX export, and `--compare-v1` ablation mode (benchmarking original ARCNet-v1 vs. ARCNet-v2).
- [`evaluate.py`](./evaluate.py): Evaluation & verification suite computing `MAE`, `MSE`, `PSNR`, $\Delta E_{\text{Oklab}}$, **Itten Harmonic Sector Alignment (°)**, **Anti-Camouflage Contrast Rate (%)**, inference latency (ms), and **Temporal EMA Flicker Reduction**.
- [`export_onnx.py`](./export_onnx.py): Verification script (parameter count, permutation invariance, backward pass) and ONNX exporter (`opset >= 14`) for Unity Sentis / Barracuda.


(Note: In the zip archive and repository, checkpoints/ and large dataset cache directories are ignored from git to keep file sizes manageable, and are created automatically when you run train.py or cross_dataset_eval.py) 
