# 🔬 Vision Transformer & Foundation Models — Lab Tutorial

A hands-on Jupyter notebook tutorial covering **foundation model concepts** and **Vision Transformer (ViT) internals**, applied to breast ultrasound classification using the BUS-BRA dataset.

Built for the Engineering Design / Biomedical AI lab at IIT Madras.

---

## 📖 What This Tutorial Covers

### Part 1 — Foundation Models
- What makes a model a "foundation model" (scale, self-supervision, transferability)
- The paradigm shift from task-specific supervised learning to pretrain-then-adapt
- Key examples: GPT, BERT, CLIP, MAE, SAM, BiomedCLIP

### Part 2 — From CNNs to Vision Transformers
- The locality bias of CNNs and why it limits global reasoning
- How ViTs treat images as sequences of patch tokens — the NLP analogy
- Why ViTs need massive data (or self-supervised pretraining) to shine

### Part 3 — ViT Architecture, Built from Scratch
- **Patch Embedding** — splitting an image into tokens with a single Conv2d
- **Positional Encoding** — giving the model spatial awareness
- **Multi-Head Self-Attention** — Q, K, V projections, scaled dot-product, head parallelism
- **The Transformer Block** — pre-norm, residual connections, FFN
- **CLS token** — global representation for classification

### Part 4 — Transfer Learning on Small Medical Datasets
- Why training ViT from scratch fails on BUS-BRA (~1,875 images)
- **Linear probing** — frozen encoder, train head only
- Scaling up: ViT-Tiny vs ViT-Base with ImageNet pretrained weights (via `timm`)

### Part 5 — Attention Visualization
- Extracting per-layer, per-head attention maps with forward hooks
- **Attention rollout** — propagating attention through residual connections for a global view
- Visualizing what the model "looks at" when classifying ultrasound lesions

### Part 6 — Bridging the Domain Gap with MAE
- Why ImageNet features have a ceiling on ultrasound tasks
- The MAE pretraining strategy: mask 75% of patches, reconstruct pixels
- How domain-specific pretraining (EchoCare-style) closes the gap

---

## ⚙️ Setup

### Prerequisites
- [uv](https://github.com/astral-sh/uv) — fast Python package manager
- Python 3.12+
- CUDA-capable GPU recommended (CPU works for small experiments)

### Install Dependencies

```bash
git clone https://github.com/<your-org>/vit-tutorial.git
cd vit-tutorial

# Install all dependencies from the lockfile
uv sync
```

### Activate the Environment

```bash
# On Linux/macOS
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

---

## 📂 Data Setup

The tutorial uses the **BUS-BRA** breast ultrasound dataset for the classification experiments (Parts 4 & 5).

### Directory Structure

Place the dataset under `data/o_Breast/` as follows:

```
data/
└── o_Breast/
    └── BUSBRA_Dataset/
        └── BUSBRA/
            └── BUSBRA/
                ├── bus_data.csv
                ├── Images/
                │   ├── patient_001/
                │   │   └── *.png
                │   └── ...
                └── Masks/
```

### Obtaining BUS-BRA

BUS-BRA is publicly available. Request access and download instructions can be found at:
**https://zenodo.org/record/8231412**

> **Note:** Parts 1–3 (architecture) run entirely without data. You only need BUS-BRA for Parts 4 and 5 (transfer learning and attention visualization).

---

## 🚀 Running the Tutorial

```bash
# Launch Jupyter
jupyter notebook notebooks/vit_tutorial.ipynb

# Or with JupyterLab
jupyter lab notebooks/vit_tutorial.ipynb
```

Run cells top-to-bottom. Each section is self-contained with markdown explanations before the code.

---

## 📚 Pre-Reading

Before attending the tutorial, we recommend going through the following in order:

### Foundations

| Resource | What it covers | Time |
|---|---|---|
| [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/) | The clearest visual walkthrough of transformer architecture | ~30 min read |
| [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) | The original transformer paper — skim Sections 1–3 | ~20 min |
| [An Image is Worth 16×16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) | The original ViT paper — focus on Sections 3–4 | ~25 min |

### Self-Supervised Pretraining

| Resource | What it covers | Time |
|---|---|---|
| [Masked Autoencoders Are Scalable Vision Learners — He et al. (2021)](https://arxiv.org/abs/2111.06377) | MAE: the pretraining strategy at the heart of EchoCare | ~30 min |
| [Vision Transformers, Explained — Towards Data Science](https://towardsdatascience.com/vision-transformers-explained-a9d07147e4c8/) | Annotated PyTorch walkthrough of ViT components | ~20 min read |

---

## 🎥 Video Resources

### Must-Watch Before the Tutorial

| Video | Channel | Why |
|---|---|---|
| [Transformers (Visual Intro)](https://www.youtube.com/watch?v=wjZofJX0v4M) | 3Blue1Brown | Beautiful visual intuition for embeddings and the transformer pipeline |
| [Attention in Transformers, Step by Step](https://www.youtube.com/watch?v=eMlx5fFNoYc) | 3Blue1Brown | The clearest visual explanation of Q/K/V, scaled dot-product, and multi-head attention |

### Highly Recommended

| Video | Channel | Why |
|---|---|---|
| [An Image is Worth 16×16 Words (ViT Paper Explained)](https://www.youtube.com/watch?v=TrdevFK_am4) | Yannic Kilcher | Detailed paper walkthrough with commentary on design choices |
| [MAE — Masked Autoencoders Are Scalable Vision Learners](https://www.youtube.com/watch?v=Dp6iICL2dVI) | Yannic Kilcher | Excellent explanation of MAE's asymmetric design and why 75% masking works |
| [Self-Supervised Learning Explained](https://www.youtube.com/watch?v=0AiHLQoHIBE) | Andrej Karpathy (Stanford CS231n) | Broader context on self-supervised objectives before diving into MAE |

### For Going Deeper

| Video | Channel | Why |
|---|---|---|
| [Let's Build GPT from Scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Andrej Karpathy | Builds a transformer character-by-character in PyTorch — same mechanics as ViT |
| [DINO: Self-Supervised Vision Transformers](https://www.youtube.com/watch?v=h3ij3F3cPIk) | AI Coffee Break | How DINO shows ViTs learn semantic segmentation for free — relevant to medical imaging |

---

## 🗂️ Repository Structure

```
vit-tutorial/
├── notebooks/
│   └── vit_tutorial.ipynb      # Main tutorial notebook
├── src/
│   └── data/
│       ├── datasets/
│       │   └── busbra.py       # BUS-BRA dataset class
│       ├── splits.py           # Patient-level train/val split
│       └── loaders.py          # DataLoader builder
├── data/
│   ├── figures/                # Architecture diagrams used in notebook
│   └── o_Breast/               # ← Place BUS-BRA dataset here
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 📄 Key Papers

```
Dosovitskiy et al. (2020) — An Image is Worth 16x16 Words
  https://arxiv.org/abs/2010.11929

He et al. (2021) — Masked Autoencoders Are Scalable Vision Learners
  https://arxiv.org/abs/2111.06377

Vaswani et al. (2017) — Attention Is All You Need
  https://arxiv.org/abs/1706.03762

Abnar & Zuidema (2020) — Quantifying Attention Flow in Transformers (Attention Rollout)
  https://arxiv.org/abs/2005.00928

Touvron et al. (2021) — Training Data-Efficient Image Transformers (DeiT)
  https://arxiv.org/abs/2012.12877
```

---

## 🤝 Questions

Raise an issue on this repo or reach out during the lab tutorial session.