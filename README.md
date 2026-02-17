2026-02-16, 11:31
Tags: [[ML]] [[Unsupervised ML]]

# Water Structure ML Research Plan
### Can Machine Learning Independently Recover Physically Meaningful Water Structural States?
**Target completion: May 1, 2026 | Start: February 16, 2026 | Duration: ~10.5 weeks**

---

## Research Overview

### Core Question
Given only computed order parameters (q, Q6, LSI, Sk, ζ) from MD trajectories, can unsupervised ML independently recover the LFTS/DNLS structural distinction that Tanaka identified through physical intuition — and can this be validated entirely independently via structure factor S(k)?

### Why This Matters
Most ML-on-water papers validate clusters against the same features used for clustering (circular). This project validates through **reciprocal space structure factors** — completely independent of the clustering step. A clean S(k) peak separation at kT1 ≈ 3/4 (LFTS) vs kD1 ≈ 1 (DNLS) is unambiguous physical proof that ML recovered physically real clusters.

### Target Publication
**Journal of Chemical Physics** (primary) | **JCTC** (if multi-model + deep learning results are strong)

---

## Assets Inventory

### Data (already available)
| Dataset | Model | Temperature | Status |
|---|---|---|---|
| OrderParam_Run21_swm4ndp_T-20.0.mat | SWM4-NDP (polarizable) | 253 K | ✅ Complete |
| OrderParamZeta_Run21_swm4ndp_T-20.0.mat | SWM4-NDP | 253 K | ✅ Complete |
| OrderParam_tip4p2005_T-10_Run01.mat | TIP4P/2005 (rigid) | 263 K | ✅ Available |
| TIP4P/2005 supercooled data | TIP4P/2005 | 220–240 K | ✅ Available |
| TIP5P data | TIP5P | 194–260 K | ✅ Available |

### Code (already available)
- `water_clustering.py` — full clustering pipeline (DBSCAN, KMeans, GMM, DBSCAN+GMM)
- `compute_structure_factor.py` structureStructure factor calculation code 
- OpenMM simulation setup

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
   │  q+ζ only   │ │PCA→GMM   │ │  Autoencoder     │
   │  ζ only     │ │UMAP→GMM  │ │  + clustering    │
   │  all 5      │ │UMAP+HDBSCAN│ │                │
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
│              Cluster 0 peak at kT1                  │
│              Cluster 1 peak at kD1                  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## ML Methods to Explore

### Phase 1 — Baseline Methods (Weeks 1–3)
| Method               | Key parameters       | Expected outcome               |
| -------------------- | -------------------- | ------------------------------ |
| GMM (all 5 features) | k=2, full covariance | Moderate separation            |
| GMM (ζ only)         | k=2                  | 1D separation, Tanaka baseline |
| GMM (q + ζ)          | k=2                  | Strong, physically motivated   |
| KMeans               | k=2                  | Weaker, spherical clusters     |
| DBSCAN → GMM         | eps then GMM         | Best of both                   |

### Phase 2 — Manifold Learning (Weeks 4–5)
| Method | Key parameters | Expected outcome |
|---|---|---|
| PCA → GMM | 2 components | Linear baseline |
| UMAP → GMM | n_neighbors=30, min_dist=0.1 | Reveals non-linear structure |
| UMAP → HDBSCAN | min_cluster_size=100 | Non-spherical clusters, density-aware |
| UMAP → KMeans | 2D embedding | Visual validation easy |

### Phase 3 — Deep Learning (Weeks 6–8)
| Method | Architecture | Expected outcome |
|---|---|---|
| Autoencoder + GMM | 5→16→8→2→8→16→5 | Learned feature compression |
| Variational Autoencoder (VAE) + GMM | Latent dim=2 | Probabilistic cluster assignments |
| Deep Embedded Clustering (DEC) | CNN-style | Joint training of representation + clustering |

### Phase 4 — Feature Ablation (runs in parallel with all phases)
For every method, run with:
- All 5 features
- ζ only
- q + ζ
- q + Q6 + ζ
- PCA top-2 components

**Key metric:** Does S(k) peak separation change with feature subset?

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

## Week-by-Week Plan

### WEEK 1 : Setup & Baseline Clustering
**Goal: Clean baseline results on SWM4-NDP data with all current methods**

- [x] Upload updated `water_clustering.py` to entropie server
- [x] Run K-mean with all five features and generate clustering images
- [x] Run GMM with all five features and generate clustering images. 
- [x] Run data preprocessed with DBSCAN and them with GMM, generate the clustering images. (tried with eps=0.05, will follow on with eps = 0.06-0.10)
- [x] Run dbscan-gmm, return the structure factor result for each cluster (bad result)
- [x] Run ζ-only GMM manually, compare S(k) peaks to full 5-feature GMM
- [x] Add `--feature_subset` argument to `water_clustering.py`
- [x] Compute S(k) for each feature subset's clusters
- [ ] Plot: silhouette score vs feature subset vs method (heatmap)
- [ ] Compare LFTS fraction s vs T to Tanaka Figure 4A
- [ ] Compare coherence length λ vs T to Tanaka Figure 4B

**Deliverable:** Temperature dependence plots — key validation of physical correctness

---

### WEEK 2: UMAP Implementation
**Goal: Non-linear dimensionality reduction as a new clustering front**

- [x] Install umap-learn and hdbscan on entropie
  ```bash
  pip install umap-learn hdbscan --break-system-packages
  ```
- [ ] Add `umap_gmm`, `umap_hdbscan`, `pca_gmm` methods to script
- [ ] Run UMAP on TIP4P/2005 at -20T (strongest signal)
- [ ] Visualize 2D UMAP embeddings — do two populations visually separate?
- [ ] Compute S(k) for UMAP-derived clusters
- [ ] Compare: does UMAP S(k) match GMM S(k)?

**Deliverable:** UMAP embedding plots + S(k) comparison vs GMM

---

### WEEK 3: TIP5P + ST2 Simulations
**Goal: Multi-model comparison — does ML quality track with known model ordering?**

Tanaka's known ordering of structural signal strength: ST2 > TIP5P > TIP4P/2005 > SWM4-NDP

- [ ] Launch TIP5P simulations at 194, 220, 240, 260 K
- [ ] Launch ST2 simulations at 210, 240, 260, 280 K
- [ ] Process existing TIP5P/ST2 data if available from Diya's work
- [ ] Run GMM + UMAP on available rigid model data
- [ ] Plot silhouette score vs water model — does it follow ST2 > TIP5P > TIP4P/2005?
- [ ] Plot S(k) peak separation vs water model

**Deliverable:** Multi-model comparison table (the polarizability story)

---

### WEEK 4: Autoencoder Implementation
**Goal: Deep learning baseline**

- [ ] Implement simple autoencoder in PyTorch or TensorFlow
  - Architecture: 5 → 32 → 16 → 2 → 16 → 32 → 5
  - Latent dim = 2 (visualizable)
  - Loss: MSE reconstruction
- [ ] Train on TIP4P/2005 240 K data (largest structural signal)
- [ ] Cluster latent space with GMM
- [ ] Compute S(k) for autoencoder-derived clusters
- [ ] Compare reconstruction quality across water models

**Deliverable:** Autoencoder latent space plots + S(k) validation

---

### WEEK 5: VAE + DEC
**Goal: Probabilistic and joint deep learning approaches**

- [ ] Implement Variational Autoencoder (VAE)
  - Latent space enforced to be Gaussian → natural for GMM post-clustering
  - Reparameterization trick gives differentiable cluster assignments
- [ ] Implement Deep Embedded Clustering (DEC) if time permits
  - Jointly optimizes representation + cluster assignment
- [ ] Compare latent space quality: Autoencoder vs VAE
- [ ] Compute S(k) for VAE-derived clusters
- [ ] Does VAE latent space show cleaner bimodal separation than PCA/UMAP?

**Deliverable:** Deep learning S(k) results + comparison to classical methods

---

### WEEK 6: k > 2 Exploration
**Goal: Can ML find more than two water structural states?**

This is the most original part of the project.

- [ ] Run GMM with k = 2, 3, 4, 5 on TIP4P/2005 240 K
- [ ] Use BIC/AIC to identify optimal k
- [ ] For k=3 or k=4: compute S(k) for each sub-cluster
  - Do extra clusters have distinct S(k) peaks?
  - Or do they split one of the two main populations?
- [ ] Repeat with UMAP + HDBSCAN (lets k emerge from data naturally)
- [ ] Physical interpretation: if a third state exists, what does its ζ distribution look like?

**Deliverable:** k > 2 S(k) plots — potential new finding beyond Tanaka

---

### WEEK 7: Comprehensive Results Compilation
**Goal: Assemble all results into coherent narrative**

- [ ] Generate master comparison table: all methods × all models × all temperatures
  - Silhouette score
  - S(k) peak position cluster 0 and cluster 1
  - Peak separation Δk
  - LFTS fraction s
  - ζ-agreement score
- [ ] Identify the best-performing ML method (likely UMAP+HDBSCAN or GMM on q+ζ)
- [ ] Identify the worst-performing conditions (SWM4-NDP, high T) — this is also a finding
- [ ] Draft figures for paper

**Deliverable:** Complete results database + draft figures

---

### WEEK 9: Writing
**Goal: First draft of paper**

**Suggested paper structure:**
1. Introduction — water's two-state problem, why ML validation matters
2. Methods — MD simulation details, order parameter definitions, ML methods, S(k) calculation
3. Results
   - 3.1 Baseline: Classical ML on SWM4-NDP (establish the challenge)
   - 3.2 Rigid models: Temperature dependence of ML cluster quality
   - 3.3 Feature ablation: Which order parameters does ML need?
   - 3.4 Manifold learning: UMAP reveals structure invisible to GMM
   - 3.5 Deep learning: Autoencoder latent space as structural representation
   - 3.6 Beyond two states: Can ML find k > 2 structural populations?
4. Discussion — physical interpretation, comparison to Tanaka, implications
5. Conclusions

- [ ] Write Methods section (most straightforward, do first)
- [ ] Write Results 3.1 and 3.2 (core findings)
- [ ] Outline Discussion

**Deliverable:** Methods + core Results sections drafted

---

### WEEK 10: Polish & Submit
**Goal: Submission-ready manuscript**

- [ ] Complete all figures (publication quality, 300 dpi)
- [ ] Write Introduction and Conclusions
- [ ] Complete Discussion
- [ ] Internal review with advisor
- [ ] Submit to JCP or JCTC

---






