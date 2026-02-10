"""
remove_dataset_csv.py
=====================

Helper script to remove entries from one processed dataset CSV based on another.
Useful for removing the test set (e.g. CASF-2016) from the training set (e.g. PDBbind refined).
Matches entries based on unique_id (which is the PDB ID).
"""

import argparse
import pandas as pd
from pathlib import Path
import sys

# Get the absolute path of the current script
script_path = Path(__file__).resolve()

# Go up two levels to get the project root:
project_root = script_path.parent.parent

def parse_args():
    parser = argparse.ArgumentParser(description="Remove entries in one dataset CSV from another based on unique_id.")
    parser.add_argument("--input_csv", type=str, required=True, help="The main dataset CSV (e.g., full PDBbind).")
    parser.add_argument("--remove_csv", type=str, required=True, help="The subset CSV to remove (e.g., CASF-2016).")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the filtered CSV.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load DataFrames
    input_path = project_root / Path(args.input_csv)
    remove_path = project_root / Path(args.remove_csv)
    
    if not input_path.exists():
        print(f"Error: Input CSV {input_path} not found.")
        sys.exit(1)
    if not remove_path.exists():
        print(f"Error: Removal CSV {remove_path} not found.")
        sys.exit(1)
        
    try:
        main_df = pd.read_csv(input_path)
        remove_df = pd.read_csv(remove_path)
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        sys.exit(1)

    # Verify columns
    if 'unique_id' not in main_df.columns or 'unique_id' not in remove_df.columns:
        print("Error: Both CSVs must contain a 'unique_id' column.")
        sys.exit(1)

    # Ensure unique_id is treated as string to avoid type mismatches
    main_df['unique_id'] = main_df['unique_id'].astype(str)
    remove_df['unique_id'] = remove_df['unique_id'].astype(str)

    print(f"Main dataset: {len(main_df)} entries")
    print(f"Subset to remove: {len(remove_df)} entries")

    # 2. Identify IDs to remove
    ids_to_remove = set(remove_df['unique_id'].unique())
    
    # Check overlap
    main_ids = set(main_df['unique_id'].unique())
    ids_found = ids_to_remove.intersection(main_ids)
    ids_missing = ids_to_remove - main_ids
    
    print(f"Found {len(ids_found)} overlapping IDs to remove.")
    
    if ids_missing:
        print(f"Notice: {len(ids_missing)} IDs from the removal set were NOT found in the main dataset.")
    else:
        print("All IDs from the removal set were found in the main dataset.")

    # 3. Filter
    # We filter out rows where unique_id is in ids_to_remove
    mask = main_df['unique_id'].isin(ids_to_remove)
    filtered_df = main_df[~mask].copy()
    
    
    print(f"Final dataset size: {len(filtered_df)} entries")
    
    # 4. Verification
    expected_len = len(main_df) - mask.sum()
    if len(filtered_df) == expected_len:
        print("Verification SUCCESS: Output length matches expected length.")
    else:
        print(f"Verification FAILED: Output length {len(filtered_df)} != Expected {expected_len}")

    # 5. Save
    output_path = project_root / Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    filtered_df.to_csv(output_path, index=False)
    print(f"Saved filtered dataset to: {output_path}")

if __name__ == "__main__":
    main()
