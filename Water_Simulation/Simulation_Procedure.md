# Water Simulation Process

**Purpose**: Run molecular dynamics simulations for water models (TIP5P, TIP4P/2005) and generate order parameter MAT files from trajectory data.

---

## File Overview

### Execution Scripts

| File | Purpose | Usage |
|------|---------|-------|
| **`run_batch_params.py`** | Generate order parameter MAT files from DCD trajectories (batch processing). Produces `OrderParam_*.mat` and `OrderParamZeta_*.mat` files. | `python run_batch_params.py [args]` |
| **`run_single_condition.py`** | Process ONE model/temperature/run combination. | `python run_single_condition.py <model> <temp> <run>` |

### Simulation Setup Scripts

| File | Purpose |
|------|---------|
| **`runWater_tip5p.py`** | Run TIP5P simulations at multiple temperatures (parallel execution). |
| **`runWater_tip4p2005.py`** | Run TIP4P/2005 simulations at multiple temperatures. |
| **`runWater_tip5p_tanaka.py`** | TIP5P with Tanaka paper parameters. |
| **`runWater_tip4p2005_tanaka.py`** | TIP4P/2005 with Tanaka paper parameters. |

### Supporting Modules

| File | Purpose |
|------|---------|
| **`MDWater.py`** | Core MD simulation engine. Handles OpenMM setup, equilibration, production runs. |
| **`molecules.py`** | Define water molecule geometries (SWM4-NDP, TIP4P/2005, TIP5P, etc.). |
| **`MolPositions.py`** | Generate initial molecular positions on a lattice. |
| **`CreateTopo.py`** | Create OpenMM topology objects for water systems. |

---


## Generate Order Parameters (Most Common)

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

### Run MD Simulations

**TIP5P:**
```bash
python runWater_tip5p.py  # Runs all temperatures in parallel
```

**TIP4P/2005:**
```bash
python runWater_tip4p2005.py
```

