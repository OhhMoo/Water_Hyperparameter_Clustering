#!/usr/bin/env python3
"""
Run order parameter calculations for a SINGLE condition
Usage examples:
  python run_single_condition.py tip5p T-10 Run01
  python run_single_condition.py tip4p2005 T-20 Run02
"""

import sys
import os

# Import the main processing function
sys.path.append('/home/water/WaterSimulation/michael')
from run_batch_params import process_single_dcd, find_matching_pdb

def main():
    if len(sys.argv) != 4:
        print("Usage: python run_single_condition.py <model> <temperature> <run>")
        print("\nExamples:")
        print("  python run_single_condition.py tip5p T-10 Run01")
        print("  python run_single_condition.py tip5p T-20 Run02")
        print("  python run_single_condition.py tip4p2005 T-30 Run01")
        print("  python run_single_condition.py tip4p2005 T0 Run02")
        print("\nAvailable:")
        print("  Models: tip5p, tip4p2005")
        print("  TIP5P Temps: T-25, T-20, T-18, T-15, T-10, T0, T10, T20, T30")
        print("  TIP4P2005 Temps: T-40, T-35, T-30, T-20, T-10, T0, T10, T20, T30")
        print("  Runs: Run01, Run02")
        sys.exit(1)
    
    model = sys.argv[1]
    temp = sys.argv[2]
    run = sys.argv[3]
    
    if model not in ['tip5p', 'tip4p2005']:
        print(f"Error: Invalid model '{model}'. Must be 'tip5p' or 'tip4p2005'")
        sys.exit(1)
    
    # Construct file paths
    if model == 'tip5p':
        data_dir = '/home/water/WaterSimulation/michael/tanaka_condition/water_simulation/water_sim_data/tip5p_runs'
    else:
        data_dir = '/home/water/WaterSimulation/michael/tanaka_condition/water_simulation/water_sim_data/tip4p2005_runs'
    
    dcd_file = f"{data_dir}/dcd_{model}_{temp}_N1024_{run}_0.dcd"
    
    if not os.path.exists(dcd_file):
        print(f"Error: DCD file not found: {dcd_file}")
        print(f"\nMake sure temperature is formatted correctly (e.g., T-10, T0, T10)")
        sys.exit(1)
    
    try:
        pdb_file = find_matching_pdb(dcd_file)
        output_dir = '/home/water/WaterSimulation'
        
        print(f"\n{'='*70}")
        print(f"Processing Single Condition:")
        print(f"  Model: {model}")
        print(f"  Temperature: {temp}")
        print(f"  Run: {run}")
        print(f"{'='*70}\n")
        
        success = process_single_dcd(dcd_file, pdb_file, output_dir, model)
        
        if success:
            print(f"\n{'='*70}")
            print("SUCCESS! Generated files:")
            parts = os.path.basename(dcd_file).replace('.dcd', '').split('_')
            temp_str = [p for p in parts if p.startswith('T')][0]
            run_str = [p for p in parts if p.startswith('Run')][0]
            print(f"  1. OrderParam_{model}_{temp_str}_{run_str}.mat")
            print(f"  2. OrderParamZeta_{model}_{temp_str}_{run_str}.mat")
            print(f"\nLocation: {output_dir}/")
            print(f"{'='*70}\n")
        else:
            print("\nProcessing failed. Check error messages above.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
