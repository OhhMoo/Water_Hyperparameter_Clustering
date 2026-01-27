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


##To-do list
1. Align Diya’s outputs to MD trajectory 
  Decide what one Diya “sample” is (frame vs oxygen vs oxygen×frame).
  Build a mapping table: frame_id, oxygen_id, diya_cluster_label, ζ, (other order params).
  Pass condition: pick any row and retrieve the exact atoms/frames in the trajectory that produced it.

2. 
