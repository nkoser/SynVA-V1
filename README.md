# SynVA-V1: Topology-Conditioned Flow Matching for Healthy Vessel Synthesis (preliminary)

SynVA-V1 is a two-stage topology-conditioned generative model for
synthetic healthy vascular trees. The pipeline explicitly separates
**discrete tree topology** from **topology-conditioned continuous
geometry** so that anatomy and shape can be controlled independently:
a small GPT-2 decoder samples a rooted binary tree from a
run-length-compressed pre-order event sequence, and a
topology-conditioned typed message-passing GNN (TreeFlowNet) maps that
topology to per-node spline geometry under an OT conditional
flow-matching objective. Predicted spline trees are turned into
watertight meshes by a deterministic five-stage spline-station →
screened-Poisson → bifurcation-aware smoothing pipeline.

Each tree node carries a 39-D feature vector
`x_i = (p_i, c_i, s_i)` (3-D parent-relative position, 24 cross-section
control points, 12 B-spline knots), plus three discrete structure
tokens `(k_i, d_i, s_i)` (child count, depth, child slot).

## Paper review / preliminary repository

This repository is **only** the preliminary companion codebase for
**paper review**. It is **not** intended as the final, cleaned-up
release (e.g., stable APIs, packaging, full reproducibility).

The **final repository** (including a refined structure, documentation,
and release artifacts) will be published **after acceptance**.

---

## Repository layout

```
SynVA_V1/
├── Preprocessing_v2/                 # legacy-style preprocessing helpers
├── Preprocessing_modular_v2/         # modular preprocessing pipeline
│                                       (centerline → spline tree)
│
├── Stage1/                           # base utilities reused by Stage 2
│
├── Stage2_BranchTopology/            # SynVA-V1_TOP — GPT-2 topology decoder
├── Stage2_FlowMatching/              # TreeFlowNet base
│                                       (synvaBase, flow_v11/v12)
├── Stage2_FlowMatching_TreeGNN/      # TreeFlowNet with typed message-passing
│                                       GNN (synvaBase / synvaDepth / synvaAlign
│                                       + focal-pos / aligned-45° /
│                                       anti-curl-back ablations)
├── Stage2_FlowMatching_TreeGNN_v2/   # physiological auxiliary-loss variants
│                                       (physio-wide, synvaPhys, synvaCombined)
├── Stage2_FlowMatching_Physio/       # alternative physio-loss training
├── Stage2_AutoregressiveFM/          # autoregressive flow-matching baseline
│
├── Reconstruction_RMF/               # 5-stage spline → screened-Poisson mesh
├── Reconstruction_Loft/              # alternative loft-based reconstruction
├── SplineInterpolationMesh/          # arc-length spline-station interpolation
├── CrossSectionSDF/, sdf/            # SDF utilities used by some recon variants
│
├── HealthyVesselFoundationTSNE/      # UNI3D-G foundation t-SNE evaluation
├── HealthyVesselMeshMetrics/         # paper distribution metrics
│                                       (MMD, COV, 1-NNA, KL_deg, Spec)
├── aneurysm_removal/                 # aneurysm-decapping helper
│
├── figures/                          # paper plot-builder scripts (build_*.py)
├── scripts/                          # mesh / spline export helpers
│
├── evaluate_all.py                   # core metric implementation
├── evaluate_healthy_v2_clean.py      # v2_clean test-split evaluation wrapper
├── make_v2_clean_generate_configs.py # builds generate configs from training configs
├── make_latex_table_healthy_all.py   # final LaTeX table renderer
├── tree_functions.py                 # rel-norm ↔ absolute conversion + tree utils
├── reconstruct_mesh.py               # CLI wrapper around Reconstruction_RMF
├── preprocess_healthy_vessel.py      # raw-mesh → tree dataset entry point
├── decap_healthy_vessel.py           # aneurysm-removal entry point
├── tree_*.py                         # tree visualization / OBJ export
│
└── environment.yml                   # conda env (`vmtk_2`)
```

---

## Setup

The project pins to the `vmtk_2` conda environment captured in
`environment.yml`:

```bash
conda env create -f environment.yml
conda activate vmtk_2
```

Notable packages:

- PyTorch (geometry models, flow matching)
- HuggingFace `transformers` (GPT-2 head for the topology decoder)
- VMTK (centerline extraction during preprocessing)
- Open3D / `trimesh` / `pymeshlab` (mesh ops, screened-Poisson)
- NumPy / SciPy / scikit-learn (B-spline fitting, t-SNE)

---

## Data format

Each preprocessed tree is an `(N, 40)` float32 array in pre-order DFS:

| col   | meaning                                |
|-------|----------------------------------------|
| 0     | `k_count` (child count, 0/1/2)         |
| 1:4   | node position (3D, parent-relative)    |
| 4:28  | 24 cross-section control points (8×3)  |
| 28:40 | 12 knot values                         |

Two coordinate spaces are used:

- **rel-norm (on-disk):** parent-relative positions z-scored using
  `Stage2_FlowMatching/compute_feature_stats.py` over the training split.
- **absolute:** world coordinates after un-normalizing and accumulating
  parent-relative displacements along the tree.

Conversion helper (used at generation time and inside auxiliary
losses):

```python
from tree_functions import local_geometry_tree_to_absolute
abs_tree = local_geometry_tree_to_absolute(
    arr,
    position_slice=(1, 4),
    control_point_slices=((4, 12), (12, 20), (20, 28)),
    relative_positions=True,
    node_local_control_points=True,
    copy=True,
)
```

---

## End-to-end pipeline

```
                          ┌─────────────────────────┐
   raw vessel meshes ───▶ │ Preprocessing_modular_v2│ ──▶ rooted spline trees
                          │  (VMTK centerlines,     │       (N × 40, k_count + p + c + s)
                          │   resampling, B-spline   │
                          │   cross-section fit,     │
                          │   binarization, root)    │
                          └─────────────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────────┐
                          │  Stage2_BranchTopology   │ ──▶ topology T̂  (compressed
                          │  (SynVA-V1_TOP, GPT-2,   │       pre-order event sequence)
                          │  ℓ ∈ [0,63], δ ∈ {0,1,2})│
                          └─────────────────────────┘
                                     │
                                     ▼  T̂ (or GT topology)
                          ┌─────────────────────────┐
                          │  Stage2_FlowMatching*    │ ──▶ per-node geometry x̂
                          │  TreeFlowNet (typed-MP   │       (39-D vector per tree node)
                          │  GNN + global attention) │
                          └─────────────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────────┐
                          │  Reconstruction_RMF      │ ──▶ watertight mesh
                          │  (spline-station ×64,    │
                          │   per-segment screened   │
                          │   Poisson D=8, Boolean,  │
                          │   bif-localized Laplacian│
                          │   /Taubin, global Poisson│
                          │   D'=9)                  │
                          └─────────────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────────┐
                          │  evaluate_all.py +       │
                          │  HealthyVesselMeshMetrics│ ──▶ MMD ↓, 1-NNA → 0.5,
                          │  HealthyVesselFoundation │       KL_degree, Spec,
                          │  TSNE (UNI3D-G)          │       BifAngle → 75.9°,
                          │                          │       mesh-Chamfer ↓, t-SNE
                          └─────────────────────────┘
```

### Quick recipe

```bash
# 1) Preprocess raw meshes into rooted spline trees
python preprocess_healthy_vessel.py --config Preprocessing_modular_v2/config/healthy_vessel/pipeline_full.yaml

# 2) Train the topology decoder
python Stage2_BranchTopology/convert_dataset.py --config Stage2_BranchTopology/configs/convert_relpos_nodecp_v1.yaml
python Stage2_BranchTopology/train_branch_gpt.py --config Stage2_BranchTopology/configs/train_relpos_nodecp_v1.yaml

# 3) Train a TreeFlowNet geometry model (example: synvaBase / flow_v11)
python -m Stage2_FlowMatching.train --config Stage2_FlowMatching/configs/flow_matching_v11_v2_clean.yaml

# 4) Build generate configs from the trained checkpoints, then sample
python make_v2_clean_generate_configs.py --force
python -m Stage2_FlowMatching.generate_trees --config Stage2_FlowMatching/configs/generate_flow_v11_v2_clean.yaml

# 5) Evaluate against the biffilter_v1 test split
python evaluate_healthy_v2_clean.py --output_dir evaluation_results_healthy_all_v2_clean

# 6) Reconstruct meshes + UNI3D-G t-SNE
python reconstruct_mesh.py --config Reconstruction_RMF/reconstruct_rmf_config.yaml
python HealthyVesselFoundationTSNE/run_foundation_tsne.py --config HealthyVesselFoundationTSNE/config.yaml
```

The 14 reported TreeFlowNet ablation variants are summarized in the
table below; each variant maps to a config in
`Stage2_FlowMatching*/configs/`.

---

## Models reported in the paper

The 14 TreeFlowNet variants:

| group                       | variants                                                                                |
|-----------------------------|-----------------------------------------------------------------------------------------|
| TreeFlowNet (typed-MP GNN)  | **synvaBase**, **synvaDepth**, logit-time, focal-pos λ=2/4, focal-pos+cp-bif, **synvaAlign** (aligned-60°), aligned-45°, anti-curl-back |
| + physiological aux loss    | physio-wide θ★=120°, physio-wide-no-warp, **synvaPhys** θ★=50°, **synvaCombined**       |
| Transformer baseline        | **synvaPhysTx** (flat self-attention, no GNN)                                           |

Bold variants are the three configurations reported in the main paper
(`synvaBase`, `synvaDepth`, `synvaAlign`).

---

## Algorithmic references

The methods, training objective, and metric definitions follow the
papers below. None of these papers' code is vendored here; only the
algorithmic ideas are adopted.

- **VesselGPT** — Feldman, Sinnona, Siless, Delrieux, Iarussi.
  *VesselGPT: Autoregressive Modeling of Vascular Geometry*, MICCAI
  2025. [arXiv:2505.13318](https://arxiv.org/abs/2505.13318) — the
  original two-stage VQ-VAE + GPT-2 design that motivated this
  repository's directory layout, the VMTK-based preprocessing, and
  the spline cross-section parameterization.
- **VesselVAE** — Feldman, Sinnona, Siless, Delrieux, Iarussi.
  *VesselVAE: Recursive Variational Autoencoders for 3D Blood Vessel
  Synthesis*, CVPR 2023 — source of the per-tree radius-histogram
  cosine-similarity metric (`cos_radius`) and the recursive vessel
  representation.
- **VMTK / vmtknetworkextraction** — Antiga et al., *An image-based
  modeling framework for patient-specific computational hemodynamics*,
  MBEC 2008 — centerline extraction (called as an external CLI tool).
- **Flow Matching** — Lipman et al., ICLR 2023; **OT-CFM** — Tong
  et al., TMLR 2023 — training objective for TreeFlowNet.
- **Stable Diffusion 3 logit-normal time sampler** — Esser et al.,
  ICML 2024 — used in the `synvaAlign` time schedule.
- **Screened Poisson reconstruction** — Kazhdan & Hoppe, ACM TOG
  2013; **Taubin smoothing** — Taubin, SIGGRAPH 1995 — mesh
  reconstruction primitives.
- **Rotation-minimizing frames** — Wang et al., ACM TOG 2008;
  **Discrete elastic rods** — Bergou et al., SIGGRAPH 2008 —
  basis of the spline-station ring transport in `Reconstruction_RMF/`.
- **UNI3D-G** — Zhou et al., 2024 — foundation-model embedding used
  for the t-SNE evaluation in `HealthyVesselFoundationTSNE`.
- **OpenShape** — Liu et al., NeurIPS 2023; **Michelangelo** — Zhao
  et al., NeurIPS 2023 — alternative point-cloud foundation models
  the same evaluation pipeline can plug in.
- **Hierarchical tree-shape spectrum** — Chen et al., 2025 — `Spec`
  topology metric.
- **Finite Scalar Quantization (FSQ)** — Mentzer et al., ICLR 2024 —
  re-implemented in `Stage1/modelsMultitalk/lib/fsq_quantizer.py`.

---

## Third-party code with copyright

The following components are either adapted from third-party code or
are external dependencies whose licensed code is loaded at run time.
Each item is used in accordance with its upstream license.

### Adapted / vendored code

- **`Stage1/modelsMultitalk/lib/quantizer.py`** — `VectorQuantizer`
  adapted from
  [`CompVis/taming-transformers`](https://github.com/CompVis/taming-transformers)
  (Esser & Rombach, MIT) and the discrete-bottleneck reference
  implementation in
  [`MishaLaskin/vqvae`](https://github.com/MishaLaskin/vqvae) (MIT).
- **`sdf/`** — signed-distance-field utilities adapted from
  [`fogleman/sdf`](https://github.com/fogleman/sdf) (MIT).
- **`Reconstruction_RMF/rmf_mesh.py`** — own implementation of the
  Wang et al. 2008 / Bergou et al. 2008 rotation-minimizing-frame
  algorithm (see *Algorithmic references* above).

### Foundation-model wrappers (call upstream code at run time)

`HealthyVesselFoundationTSNE/backbones/` contains thin adapters that
import upstream model code from a user-supplied checkout. No model
code or weights are vendored.

- **Uni3D** —
  [`baaivision/Uni3D`](https://github.com/baaivision/Uni3D), Apache 2.0.
- **OpenShape** —
  [`Colin97/OpenShape_code`](https://github.com/Colin97/OpenShape_code) /
  [`OpenShape/openshape-demo-support`](https://huggingface.co/OpenShape/openshape-demo-support),
  Apache 2.0.
- **Michelangelo** —
  [`NeuralCarver/Michelangelo`](https://github.com/NeuralCarver/Michelangelo);
  verify the upstream license terms before redistribution.

### Library dependencies

- **PyTorch** (BSD-3), **NumPy / SciPy / scikit-learn / scikit-image** (BSD).
- **HuggingFace `transformers`** (Apache-2.0) — GPT-2 head used by
  `Stage2_BranchTopology/train_branch_gpt.py`.
- **`vector_quantize_pytorch`** by lucidrains (MIT) — used as an
  alternative quantizer.
- **`einops`** (MIT).
- **Open3D** (MIT), **`trimesh`** (MIT), **`pymeshlab`** (GPL-3.0).
- **VTK** (BSD-3) and **`vedo`** (MIT) — visualization.
- **`numba`** (BSD-2), **`networkx`** (BSD-3).
- **`matplotlib`** (PSF / BSD-style), **OpenCV / `cv2`** (Apache-2.0).
- **`PyYAML`** (MIT), **`tqdm`** (MIT / MPL-2.0).

External CLI tool invoked from the preprocessing pipeline:

- **VMTK / `vmtknetworkextraction`** (BSD-3) — installed separately;
  no source is shipped in this repository.

---

## Datasets

The healthy-vessel splits used in the paper (`biffilter_v1` test
split, `clean60` evaluation subset) are not included in this
repository. Raw vessel meshes are expected as `.obj` files; the
preprocessing entry point is `preprocess_healthy_vessel.py` with the
configs under `Preprocessing_modular_v2/config/healthy_vessel/`.
