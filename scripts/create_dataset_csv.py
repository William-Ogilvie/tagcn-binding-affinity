"""
create_dataset_csv.py
=====================

Helper script to scan a directory of protein-ligand complexes and generate
the CSV required by GraphGenerationManager.

From PDBbindv2020 the index file you want is INDEX_general_PL_data.2020

Usage:
    python create_dataset_csv.py --data_dir data/CASF-2016 --sdf_format X_ligand.sdf --pdb_format X_protein.pdb --output_csv data/CASF_2016_processed.csv --index_file data/index/INDEX_general_PL_data.2020
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
    parser = argparse.ArgumentParser(
        description="Scans a directory for protein-ligand complexes and creates a CSV for GraphGenerationManager."
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True, 
        help="Root directory containing complex folders (e.g., data/CASF-2016/)."
    )
    parser.add_argument(
        "--sdf_format", 
        type=str, 
        required=True, 
        help="Format for SDF files. Use 'X' as a placeholder for the folder name (e.g., 'X_ligand.sdf')."
    )
    parser.add_argument(
        "--pdb_format", 
        type=str, 
        required=True, 
        help="Format for PDB files. Use 'X' as a placeholder for the folder name (e.g., 'X_protein.pdb')."
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        required=True, 
        help="Path to save the output CSV (e.g., data/processed_data.csv)."
    )
    parser.add_argument(
        "--index_file", 
        type=str, 
        help="Path to the PDBbind index file (e.g., INDEX_general_PL_data.2016) to extract labels. If provided, labels will be added."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Resolve absolute path for data directory
    data_dir = (project_root / args.data_dir).resolve()
    
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.")
        sys.exit(1)

    # Load index file if provided, binding affinity 3rd column
    pdb_to_pk = None
    if args.index_file:
        print(f"Loading index file: {args.index_file}")
        index_path = project_root / args.index_file
        pdb_to_pk = {}
        with open(index_path, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                
                pdb_id = parts[0]
                try:
                    pdb_to_pk[pdb_id] = float(parts[3])
                except ValueError:
                    continue
        print(f"Loaded {len(pdb_to_pk)} affinity values from index.")
        
    print(f"Scanning directory: {data_dir}")
    
    rows = []
    
    
    # Iterate over subdirectories in the data directory
    # Sorted to ensure deterministic ordering of IDs
    for item in sorted(data_dir.iterdir()):
        if item.is_dir():
            folder_name = item.name
            
            # Construct file names based on the provided format
            sdf_name = args.sdf_format.replace("X", folder_name)
            pdb_name = args.pdb_format.replace("X", folder_name)
            
            sdf_path = item / sdf_name
            pdb_path = item / pdb_name
            
            # Check if both files exist
            if sdf_path.exists() and pdb_path.exists():
                
                # If index file is loaded, try to find the label
                label = None
                if pdb_to_pk is not None:
                    label = pdb_to_pk.get(folder_name)
                    if label is None:
                        # Skip if we require labels but can't find one
                        continue

                row_data = {
                    "unique_id": folder_name,
                    "pdb_file": str(pdb_path),
                    "sdf_file": str(sdf_path)
                }
                if label is not None:
                    row_data["pK"] = label
                
                rows.append(row_data)
                
            else: 
                print(f"Skipping {folder_name}: Missing SDF or PDB file.")
                pass
                
    if not rows:
        print("No valid complexes found matching the criteria.")
        sys.exit(0)
        
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Ensure correct column order
    cols = ["unique_id", "pdb_file", "sdf_file"]
    if pdb_to_pk is not None:
        cols.append("pK")
    df = df[cols]
    
    # Save to CSV
    output_path = (project_root / args.output_csv).resolve()
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print(f"Successfully processed {len(df)} complexes.")
    print(f"CSV saved to: {output_path}")

if __name__ == "__main__":
    main()
