# Water Hyperparameter Clustering Research
### Can Machine Learning Independently Recover Physically Meaningful Water Structural States?
---

## Research Overview

### Core Question
Given only computed order parameters (q, Q6, LSI, Sk, ζ) from MD trajectories, can unsupervised ML independently recover the LFTS/DNLS structural distinction that Tanaka identified through physical intuition — and can this be validated entirely independently via structure factor S(k)?

### Why This Matters
Most ML-on-water papers validate clusters against the same features used for clustering (circular). This project validates through **reciprocal space structure factors** — completely independent of the clustering step. A clean S(k) peak separation at kT1 ≈ 3/4 (LFTS) vs kD1 ≈ 1 (DNLS) is unambiguous physical proof that ML recovered physically real clusters.

---

## Assets Inventory

### Data (already available)
| Dataset                     | Model                  | File Type         | Temperature | Status      |
| --------------------------- | ---------------------- | ----------------- | ----------- | ----------- |
| swm4ndp data                | SWM4-NDP (polarizable) | .dcd .pdb<br>.mat | 253-283 K   | ✅ Available |
| TIP4P/2005 supercooled data | TIP4P/2005             | .dcd .pdb<br>.mat | 223–283 K   | ✅ Available |
| TIP5P data                  | TIP5P                  | .dcd .pdb<br>.mat | 243-263 K   | ✅ Available |

### Features (5 order parameters per molecule per frame)
| Feature | Physical meaning | Discriminating power |
|---|---|---|
| q_all | Tetrahedral order parameter | High |
| Q6_all | Bond-orientational order | Medium |
| LSI_all | Local structure index | Medium |
| Sk_all | Translational order | Low |
| zeta_all | Shell-gap descriptor (Tanaka) | Very high |

---

## Full Research Pipeline

```
┌─────────────────────────────────────────────────────┐
│                   MD SIMULATIONS                    │
│  TIP4P/2005 | TIP5P | ST2 | SWM4-NDP               │
│ T = 194–300 K | P = 1 bar | N = 1024 molecules/frame│
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              ORDER PARAMETER EXTRACTION             │
│  q_all, Q6_all, LSI_all, Sk_all, ζ_all              │
│  20 runs × 1024 molecules = 20,480 data points      │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌─────────────┐ ┌──────────┐ ┌──────────────────┐
   │  FEATURE    │ │DIMENSION │ │   DEEP LEARNING  │
   │  ABLATION   │ │REDUCTION │ │   APPROACHES     │
   │  q+ζ only   │ │DBSCAN→GMM│ │  Autoencoder     │
   │  no ζ       │ │UMAP→GMM  │ │  + clustering    │
   │  all 5      │ │UMAP+HDBSCAN│                  │
   └──────┬──────┘ └────┬─────┘ └────────┬─────────┘
          └─────────────┴────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              CLUSTER LABEL OUTPUT                   │
│  cluster_labels.csv                                 │
│  label_gmm | label_dbscan | label_umap_hdbscan ...  │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌──────────────────┐     ┌──────────────────────────┐
│  CLUSTER 0       │     │  CLUSTER 1               │
│  molecules →     │     │  molecules →             │
│  S(k|cluster=0)  │     │  S(k|cluster=1)          │
└────────┬─────────┘     └──────────┬───────────────┘
         └──────────┬───────────────┘
                    ▼
┌─────────────────────────────────────────────────────┐
│           STRUCTURE FACTOR VALIDATION               │
│                                                     │
│  Cluster 0 peak at kT1 = krOO/2π ≈ 3/4 → LFTS ✓     │
│  Cluster 1 peak at kD1 = krOO/2π ≈ 1   → DNLS ✓     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## ML Methods to Explore

### Phase 1 — Baseline Methods

| Method               | Key parameters       | Expected outcome               |
| -------------------- | -------------------- | ------------------------------ |
| GMM (all 5 features) | k=2, full covariance | Moderate separation            |
| GMM (ζ only)         | k=2                  | 1D separation, Tanaka baseline |
| GMM (q + ζ)          | k=2                  | Strong, physically motivated   |
| KMeans               | k=2                  | Weaker, spherical clusters     |
| DBSCAN → GMM         | eps then GMM         | Best of both                   |

### Phase 2 — Manifold Learning 

| Method         | Key parameters               | Expected outcome                      |
| -------------- | ---------------------------- | ------------------------------------- |
| PCA → GMM      | 2 components                 | Linear baseline                       |
| UMAP → GMM     | n_neighbors=30, min_dist=0.1 | Reveals non-linear structure          |
| UMAP → HDBSCAN | min_cluster_size=100         | Non-spherical clusters, density-aware |
| UMAP → KMeans  | 2D embedding                 | Visual validation easy                |

### Phase 3 — Deep Learning 

| Method | Architecture | Expected outcome |
|---|---|---|
| Autoencoder + GMM | 5→16→8→2→8→16→5 | Learned feature compression |
| Variational Autoencoder (VAE) + GMM | Latent dim=2 | Probabilistic cluster assignments |
| Deep Embedded Clustering (DEC) | CNN-style | Joint training of representation + clustering |


---

## Evaluation Metrics

### ML Quality Metrics
| Metric | What it measures | Target |
|---|---|---|
| Silhouette score | Cluster compactness vs separation | > 0.4 |
| BIC/AIC (GMM) | Model selection for k | Minimum at k=2 |
| Noise fraction (DBSCAN) | How many molecules are ambiguous | < 20% |
| ζ-agreement | Label agreement with ζ threshold | > 85% |

### Physical Validation Metrics (structure factor)
| Metric                  | What it measures             | Target                |
| ----------------------- | ---------------------------- | --------------------- |
| Peak position cluster 0 | Should be kT1                | Matches Tanaka Fig 2B |
| Peak position cluster 1 | Should be kD1                | Matches Tanaka Fig 2C |
| Peak separation Δk      | How cleanly did ML separate? | Maximize              |
| FSDP width Γ            | Coherence length             | Matches Tanaka Fig 4B |
| LFTS fraction s         | Order parameter              | Matches Tanaka Fig 4A |

---

## Basic guideline

### Setup & Baseline Clustering
**Goal: Clean baseline results on SWM4-NDP data with all current methods**

- [x] Upload updated `water_clustering.py` to entropie server
- [x] Run K-mean with all five features and generate clustering images
- [x] Run GMM with all five features and generate clustering images. 
- [x] Run data preprocessed with DBSCAN and them with GMM, generate the clustering images. (tried with eps=0.05, will follow on with eps = 0.06-0.10)
- [x] Run dbscan-gmm, return the structure factor result for each cluster (bad result)
- [x] Run ζ-only GMM manually, compare S(k) peaks to full 5-feature GMM
- [x] Add `--feature_subset` argument to `water_clustering.py`
- [x] Compute S(k) for each feature subset's clusters
- [x] Plot: silhouette score vs feature subset vs method (heatmap)
- [x] Plot by cluster 3d diagram similar to Tanaka's D,E,F (through plotly with html)

---

### UMAP Implementation
**Goal: Non-linear dimensionality reduction as a new clustering front**

- [x] Install umap-learn and hdbscan on entropie
  ```bash
  pip install umap-learn hdbscan --break-system-packages
  ```
- [x] Add `umap_gmm`, `hdbscan` methods to script
- [x] Visualize 2D UMAP embeddings — do two populations visually separate?

---
### Autoencoder Implementation
- [ ] Implement simple autoencoder in PyTorch or TensorFlow
  - Architecture: 5 → 32 → 16 → 2 → 16 → 32 → 5
  - Latent dim = 2 (visualizable)
  - Loss: MSE reconstruction
- [ ] Train on TIP4P/2005 240 K data (largest structural signal)
- [ ] Cluster latent space with GMM
- [ ] Compute S(k) for autoencoder-derived clusters
- [ ] Compare reconstruction quality across water models


---

### VAE + DEC
**Goal: Probabilistic and joint deep learning approaches**

- [ ] Implement Variational Autoencoder (VAE)
  - Latent space enforced to be Gaussian → natural for GMM post-clustering
  - Reparameterization trick gives differentiable cluster assignments
- [ ] Implement Deep Embedded Clustering (DEC) if time permits
  - Jointly optimizes representation + cluster assignment
- [ ] Compare latent space quality: Autoencoder vs VAE
- [ ] Compute S(k) for VAE-derived clusters
- [ ] Does VAE latent space show cleaner bimodal separation than PCA/UMAP?

---
### Data and README.md compilation
- [ ] Clean up the code and data
- [ ] Get ready for the final Readme.md file

