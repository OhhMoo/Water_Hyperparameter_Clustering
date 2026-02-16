# Water Simulation Process

**Purpose**: Run molecular dynamics simulations for water models (TIP5P, TIP4P/2005) and generate order parameter MAT files from trajectory data.

---

## 📁 File Overview

### 🎯 Main Execution Scripts

| File | Purpose | Usage |
|------|---------|-------|
| **`run_batch_params.py`** | Generate order parameter MAT files from DCD trajectories (batch processing). Produces `OrderParam_*.mat` and `OrderParamZeta_*.mat` files. | `python run_batch_params.py [args]` |
| **`run_single_condition.py`** | Process ONE model/temperature/run combination. | `python run_single_condition.py <model> <temp> <run>` |

### 🧪 Simulation Setup Scripts

| File | Purpose |
|------|---------|
| **`runWater_tip5p.py`** | Run TIP5P simulations at multiple temperatures (parallel execution). |
| **`runWater_tip4p2005.py`** | Run TIP4P/2005 simulations at multiple temperatures. |
| **`runWater_tip5p_tanaka.py`** | TIP5P with Tanaka paper parameters. |
| **`runWater_tip4p2005_tanaka.py`** | TIP4P/2005 with Tanaka paper parameters. |

### 🔧 Supporting Modules

| File | Purpose |
|------|---------|
| **`MDWater.py`** | Core MD simulation engine. Handles OpenMM setup, equilibration, production runs. |
| **`molecules.py`** | Define water molecule geometries (SWM4-NDP, TIP4P/2005, TIP5P, etc.). |
| **`MolPositions.py`** | Generate initial molecular positions on a lattice. |
| **`CreateTopo.py`** | Create OpenMM topology objects for water systems. |

---

## 🚀 Quick Start

### Generate Order Parameters (Most Common)

**Single condition:**
```bash
python run_single_condition.py tip5p T-10 Run01
python run_single_condition.py tip4p2005 T-20 Run02
```

**Batch processing (all conditions):**
```bash
python run_batch_params.py --model tip5p      # TIP5P (36 MAT files)
python run_batch_params.py --model tip4p2005  # TIP4P/2005 (36 MAT files)
```

**Preview (dry run):**
```bash
python run_batch_params.py --dry-run
```

### Run MD Simulations

**TIP5P:**
```bash
python runWater_tip5p.py  # Runs all temperatures in parallel
```

**TIP4P/2005:**
```bash
python runWater_tip4p2005.py
```

---

## 📊 Output Files

### From `run_single_condition.py` or `run_batch_params.py`:
- **`OrderParam_<model>_T<temp>_<run>.mat`** - Contains: LSI, q, Sk, Q6, d5, r, g_r
- **`OrderParamZeta_<model>_T<temp>_<run>.mat`** - Contains: zeta_all (tetrahedral order)

### From `runWater_*.py`:
- **`dcd_<model>_T<temp>_N1024_<run>_0.dcd`** - MD trajectory
- **`inistate_<model>_T<temp>_N1024_<run>.pdb`** - Initial structure
- **`system_<model>_T<temp>_N1024_<run>.xml`** - System state
- **`statedata_<model>_T<temp>_N1024_<run>_0.txt`** - Energy, temperature, etc.

---

## 📝 Available Models & Temperatures

**Models:** `tip5p`, `tip4p2005`

**TIP5P Temperatures:**  
`T-25`, `T-20`, `T-18`, `T-15`, `T-10`, `T0`, `T10`, `T20`, `T30`

**TIP4P/2005 Temperatures:**  
`T-40`, `T-35`, `T-30`, `T-20`, `T-10`, `T0`, `T10`, `T20`, `T30`

**Runs:** `Run01`, `Run02`

---

## 🔗 Workflow

```
1. Run MD simulations       → runWater_*.py
   ↓ produces DCD trajectories
   
2. Generate order parameters → run_single_condition.py or run_batch_params.py
   ↓ produces MAT files
   
3. Clustering analysis      → ../clustering/water_clustering.py
   ↓ produces cluster labels
   
4. Structure factor         → ../cluster_structure/structure_factor_bycluster.py
```

---

## 💡 Tips

- **Order parameters only**: Use `run_single_condition.py` or `run_batch_params.py`
- **New simulations**: Use `runWater_*.py` (requires MD trajectories don't exist)
- **Parallel processing**: `run_batch_params.py` uses all CPU cores
- **Prerequisites**: DCD and PDB files must exist in `../water_sim_data/<model>_runs/`
