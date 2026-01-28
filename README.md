# Bilin-Water-Clustering-Research
## The current status: 
1. The clustering algorithm are done, with clear indication with 2-3 water cluster generated. (run on the mat file)
2. Working on water structure factor analysis. Done partially

## Finished Part on water structure
1) MD data ingestion works (mostly)
Loads many .dcd files + .pdb topology into an mdtraj trajectory (traj).
Slices out oxygen atoms (traj_O_md) and gets coords_O.
Checks box lengths/angles.

2) Two structure factor implementations exist
Si_debye(...): computes a local (per-central-atom) Debye-like sum with a window function.
Si_reciprocal(...): computes a reciprocal-space estimator averaged over directions.

3) Sarah has already got the 2 peak that exist in Tanaka's paper i think....



## To-do list
1. Quick Data Allignment Data Check
  1. Grab the cluster that is under the same label (cluster 1/cluster2)
  2. Match Diya samples to MD
  3. Plot $$\zeta$$ (and any other order parameters you already have) by cluster and confirm separation.


2. Finish the physical verification and connect it back to Diya's code
  1. Compute conditional $$\S_{oo}(k)$$ by different cluster
  2. Check on  $$S(k|c=0)$$ and $$S(k|c=1)$$ (and potentially c=2)
  3. Compare the shape of the curve with Tanaka's paper
