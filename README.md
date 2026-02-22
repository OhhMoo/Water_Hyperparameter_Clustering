# Water Structure Analysis Pipeline

A computational pipeline for identifying and characterizing structural states in supercooled water using molecular dynamics simulations. Implements the two-state thermodynamic framework of Shi & Tanaka (JACS 2020), distinguishing **Locally Favored Tetrahedral Structures (LFTS)** from **Disordered Normal-Liquid Structures (DNLS)** in TIP4P/2005 and TIP5P water models.

---

## Overview

```
MD Simulation  →  Order Parameters  →  Clustering  →  Structure Factor S(k,ζ)
   (OpenMM)         (.mat files)       (GMM / HDBSCAN)    (3D surface plots)
```

The pipeline consists of four sequential stages:

| Stage | Directory | Purpose |
|-------|-----------|---------|
| **1. MD Simulation** | `water_simulation_process/` | Run NVT trajectories; generate `.dcd` and `.pdb` files |
| **2. Order Parameters** | `water_simulation_process/` | Compute q, Q₆, LSI, Sk, ζ from trajectories → `.mat` files |
| **3. Clustering** | `clustering/` | Classify each molecule as LFTS or DNLS |
| **4. Structure Factor** | `cluster_structure/` | Compute and plot S(k) and S(k,ζ) per cluster |

---

## Requirements

```bash
conda create -n waterenv python=3.11
conda activate waterenv
conda install -c conda-forge openmm mdtraj numpy scipy matplotlib seaborn scikit-learn plotly
conda install -c conda-forge umap-learn   # for UMAP pre-processing
```

---

## Stage 1 — MD Simulation

**Scripts:** `water_simulation_process/runWater_tip5p.py`, `runWater_tip4p2005.py`

Run NVT molecular dynamics for 1024 water molecules at multiple temperatures (0°C to −30°C). Uses OpenMM with the TIP5P or TIP4P/2005 force field.

```bash
cd water_simulation_process/
python runWater_tip5p.py       # TIP5P, all temperatures (parallel)
python runWater_tip4p2005.py   # TIP4P/2005, all temperatures
```

**Output:** `water_sim_data/{model}_runs/`
- `dcd_{model}_T{temp}_N1024_Run{n}_0.dcd` — trajectory
- `inistate_{model}_T{temp}_N1024_Run{n}.pdb` — topology

---

## Stage 2 — Order Parameters

**Scripts:** `water_simulation_process/run_batch_params.py`, `run_single_condition.py`

Computes five structural order parameters per molecule per frame from the trajectory:

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Local tetrahedral order | q | Measures tetrahedral arrangement of 4 nearest neighbours |
| Bond-orientational order | Q₆ | Steinhardt order parameter |
| Local Structure Index | LSI | Gap in radial distribution at ~3.5 Å |
| Structure factor | Sk | Per-molecule contribution to S(k) |
| Tanaka ζ parameter | ζ | Primary LFTS/DNLS discriminator (Shi & Tanaka 2020) |

```bash
# Single condition
python run_single_condition.py tip5p T-20 Run01

# Batch — all runs for one model (produces 36 .mat files)
python run_batch_params.py --model tip5p
python run_batch_params.py --model tip4p2005
```

**Output:** `water_param_data/`
- `OrderParam_{model}_T{temp}_Run{n}.mat` — contains `q_all`, `Q6_all`, `LSI_all`, `Sk_all`
- `OrderParamZeta_{model}_T{temp}_Run{n}.mat` — contains `zeta_all`

---

## Stage 3 — Clustering

**Script:** `clustering/water_clustering.py`

Clusters water molecules into LFTS and DNLS states using the order parameters computed in Stage 2.

### Available Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `gmm` | Gaussian Mixture Model | Direct physical match to Tanaka two-state model |
| `dbscan_gmm` | DBSCAN noise removal → GMM | Removes transition-state molecules before GMM |
| `hdbscan_gmm` | HDBSCAN noise removal → GMM | Density-adaptive denoising; no epsilon to tune |
| `kmeans` | K-means | Hard assignment baseline |
| `dbscan` | DBSCAN only | Exploratory; identifies noise regions |
| `hdbscan` | HDBSCAN only | Density-adaptive; finds variable-size clusters |

### Optional Pre-processing

Dimensionality reduction before clustering often reveals cleaner separation:

- **`--umap`** — non-linear manifold embedding (recommended for high-dimensional feature sets)

### Example Commands

**Recommended — UMAP → HDBSCAN → GMM:**
```bash
python water_clustering.py \
  --mat_file  ../water_param_data/OrderParam_tip5p_T-20_Run01.mat \
  --zeta_file ../water_param_data/OrderParamZeta_tip5p_T-20_Run01.mat \
  --n_runs 20 --method hdbscan_gmm --umap \
  --hdbscan-min-cluster-size 100 \
  --out_dir ./tip5p_T-20_umap_hdbscan_gmm
```

**UMAP → DBSCAN → GMM:**
```bash
python water_clustering.py \
  --mat_file  ../water_param_data/OrderParam_tip5p_T-20_Run01.mat \
  --zeta_file ../water_param_data/OrderParamZeta_tip5p_T-20_Run01.mat \
  --n_runs 20 --method dbscan_gmm --umap \
  --umap-n-neighbors 15 --umap-min-dist 0.05 \
  --eps 0.05 --min_samples 30 \
  --out_dir ./tip5p_T-20_umap_dbscan_gmm
```

**GMM on ζ only (Tanaka original approach):**
```bash
python water_clustering.py \
  --mat_file  ../water_param_data/OrderParam_tip4p2005_T-20_Run01.mat \
  --zeta_file ../water_param_data/OrderParamZeta_tip4p2005_T-20_Run01.mat \
  --n_runs 20 --method gmm --features zeta_all \
  --out_dir ./tip4p2005_T-20_gmm
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mat_file` | required | Path to `OrderParam_*.mat` |
| `--zeta_file` | required | Path to `OrderParamZeta_*.mat` |
| `--n_runs` | 20 | Number of MD runs to concatenate |
| `--method` | `all` | Clustering method (see table above) |
| `--features` | all 5 | Subset of features: `q_all Q6_all LSI_all Sk_all zeta_all` |
| `--n_clusters` | 2 | Number of clusters for GMM / K-Means |
| `--confidence` | None | GMM soft-denoising threshold (0–1); marks low-probability points as noise |
| `--umap` | off | Enable UMAP pre-processing |
| `--umap-n-components` | 2 | UMAP output dimensions |
| `--umap-n-neighbors` | 15 | UMAP neighbourhood size (larger = more global) |
| `--umap-min-dist` | 0.1 | UMAP cluster tightness (smaller = tighter) |
| `--hdbscan-min-cluster-size` | 50 | HDBSCAN smallest valid cluster (~1% of data) |
| `--hdbscan-epsilon` | 0.0 | Optional distance merge threshold |
| `--out_dir` | `./clustering_results` | Output directory |

### Output

```
{out_dir}/
├── cluster_labels.csv              # Per-molecule labels (all methods)
├── {method}_scatter.png            # 2D scatter by cluster
├── {method}_zeta_distribution.png  # ζ histogram per cluster (validates LFTS/DNLS)
├── {method}_all_distributions.png  # All feature histograms
├── {method}_pairplot.png           # Feature pairplot
└── {method}_umap_embedding.png     # UMAP 2D embedding (if --umap used)
```

### Silhouette Evaluation (optional)

Sweep DBSCAN parameters to find the optimal epsilon and min_samples:

```bash
python silhouette_evaluation.py \
  --mat_file  ../water_param_data/OrderParam_tip4p2005_T-20_Run01.mat \
  --zeta_file ../water_param_data/OrderParamZeta_tip4p2005_T-20_Run01.mat \
  --eps-min 0.04 --eps-max 0.3 --eps-steps 40 \
  --min-samples-range 3 5 10 15 20 30 40 50 \
  --n_runs 20 --out_dir ./silhouette_evaluation
```

---

## Stage 4 — Structure Factor Analysis

**Scripts:** `cluster_structure/convert_cluster_labels.py`, `structure_factor_bycluster.py`

Computes the structure factor S(k) and the ζ-resolved surface S(k, ζ) per cluster, replicating Figures 2D–2E of Shi & Tanaka (JACS 2020).

### Step 4a — Convert Labels to Matrix Format

The clustering output is a flat list of labels. This step reshapes it into a (frames × molecules) matrix required by the structure factor script.

```bash
cd cluster_structure/
python convert_cluster_labels.py \
  --input  ../clustering/{out_dir}/cluster_labels.csv \
  --output ./cluster_labels_matrix_{condition}.csv \
  --n-runs 1 --n-molecules 1024 \
  --label-column label_dbscan_gmm
```

### Step 4b — Compute S(k) and S(k, ζ)

```bash
python structure_factor_bycluster.py \
  --dcd-file  ../water_sim_data/tip5p_runs/dcd_tip5p_T-20_N1024_Run01_0.dcd \
  --pdb-file  ../water_sim_data/tip5p_runs/inistate_tip5p_T-20_N1024_Run01.pdb \
  --zeta-file ../water_param_data/OrderParamZeta_tip5p_T-20_Run01.mat \
  --cluster-labels ./cluster_labels_matrix_tip5p_T-20_umap_hdbscan_gmm.csv \
  --cluster-only \
  --model-name tip5p --temperature -20 \
  --output-dir ./results_3d/tip5p_T-20_umap_hdbscan_gmm
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dcd-file` | required | DCD trajectory file |
| `--pdb-file` | required | PDB topology file |
| `--zeta-file` | None | Path to `OrderParamZeta_*.mat` (enables S(k,ζ) plots) |
| `--cluster-labels` | None | Cluster labels matrix CSV |
| `--cluster-only` | False | Skip all-atoms S(k); compute per-cluster only (faster) |
| `--model-name` | `unknown` | Water model label for plot titles |
| `--temperature` | None | Temperature in °C |
| `--rc-cutoff` | 1.5 | Neighbour cutoff distance (nm) |
| `--k-max` | 50.0 | Maximum wave number (nm⁻¹) |
| `--n-frames` | all | Limit number of frames (useful for testing) |
| `--output-dir` | `./structure_factor_results` | Output directory |

### Output

```
results_3d/{condition}/
├── structure_factor_cluster{n}_norm_{model}_T{temp}.png    # S(k) per cluster (static)
├── structure_factor_per_cluster_norm_{model}_T{temp}.png   # All clusters overlaid
├── 3d_sk_zeta_cluster{n}_{model}_T{temp}.png               # Matplotlib 3D S(k,ζ) surface
├── 3d_sk_zeta_cluster{n}_{model}_T{temp}.html              # Interactive Plotly 3D surface
├── 2d_sk_zeta_cluster{n}_{model}_T{temp}.png               # 2D contour map
├── 2d_sk_zeta_cluster{n}_{model}_T{temp}.html              # Interactive 2D contour
├── 2d_sk_zeta_combined_{model}_T{temp}.png                 # Side-by-side cluster comparison
└── 2d_sk_zeta_combined_{model}_T{temp}.html                # Interactive comparison
```

---

## Full Pipeline Example

```bash
# 1. Run MD
cd water_simulation_process/
python runWater_tip5p.py

# 2. Compute order parameters
python run_batch_params.py --model tip5p

# 3. Cluster
cd ../clustering/
python water_clustering.py \
  --mat_file  ../water_param_data/OrderParam_tip5p_T-20_Run01.mat \
  --zeta_file ../water_param_data/OrderParamZeta_tip5p_T-20_Run01.mat \
  --n_runs 20 --method hdbscan_gmm --umap \
  --out_dir ./tip5p_T-20_umap_hdbscan_gmm

# 4a. Convert labels
cd ../cluster_structure/
python convert_cluster_labels.py \
  --input  ../clustering/tip5p_T-20_umap_hdbscan_gmm/cluster_labels.csv \
  --output ./cluster_labels_matrix_tip5p_T-20_umap_hdbscan_gmm.csv \
  --n-runs 1 --n-molecules 1024 --label-column label_hdbscan_gmm

# 4b. Compute structure factor
python structure_factor_bycluster.py \
  --dcd-file  ../water_sim_data/tip5p_runs/dcd_tip5p_T-20_N1024_Run01_0.dcd \
  --pdb-file  ../water_sim_data/tip5p_runs/inistate_tip5p_T-20_N1024_Run01.pdb \
  --zeta-file ../water_param_data/OrderParamZeta_tip5p_T-20_Run01.mat \
  --cluster-labels ./cluster_labels_matrix_tip5p_T-20_umap_hdbscan_gmm.csv \
  --cluster-only --model-name tip5p --temperature -20 \
  --output-dir ./results_3d/tip5p_T-20_umap_hdbscan_gmm
```

---

## Directory Structure

```
michael/
├── water_simulation_process/   # Stage 1–2: MD runs and order parameter calculation
│   ├── MDWater.py              # OpenMM simulation engine
│   ├── runWater_tip5p.py       # TIP5P simulation launcher
│   ├── runWater_tip4p2005.py   # TIP4P/2005 simulation launcher
│   ├── run_batch_params.py     # Batch order parameter extraction
│   └── run_single_condition.py # Single condition order parameter extraction
│
├── water_sim_data/             # Raw trajectory output (.dcd, .pdb)
├── water_param_data/           # Order parameter files (.mat)
│
├── clustering/                 # Stage 3: LFTS/DNLS classification
│   ├── water_clustering.py     # Main clustering script
│   └── silhouette_evaluation.py
│
├── cluster_structure/          # Stage 4: Structure factor analysis
│   ├── convert_cluster_labels.py
│   ├── structure_factor_bycluster.py
│   ├── sk_zeta_3d.py           # S(k,ζ) computation and plotting
│   └── results_3d/             # Generated plots
│
└── calc_structure_fac/         # All-atoms S(k) (independent of clustering)
    └── compute_structure_factor.py
```

---

## Reference

Shi, R. & Tanaka, H. (2020). *Microscopic Structural Descriptor of Liquid Water*.  
**J. Am. Chem. Soc.**, 142(6), 2868–2875. https://doi.org/10.1021/jacs.9b11895
