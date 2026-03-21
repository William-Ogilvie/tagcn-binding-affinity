"""
create_zero_ligand_bias_datasets_csv.py
===================================
This script will create datasets for the 0 Ligand Bias benchmark from: https://academic.oup.com/bioinformatics/article/41/2/btaf040/7985708.
This script requires the zero_ligand_bias_test.csv and zero_ligand_bias_train.csv which you can find from the Data Availablity section from the 
above paper. We assume that you already have the full PDBbindv2020 (general + refined) saved in data/PDBbind. It is worth noting that OOD test isn't 
all of PDBbind but it goes beyond just the refined set.

Usage:
    python create_zero_ligand_bias_datasets_csv.py --zero_train_path data/zero_ligand_bias_train.csv --zero_test_path data/zero_ligand_bias_test.csv --pdbbind_dir data/PDBbind --sdf_format X_ligand.sdf --pdb_format X_protein.pdb --output_dir data --split_ratio 0.1 --seed 37 --index_file data/index/INDEX_general_PL_data.2020
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
        description="Generates the processed csvs for OOD test"
    )
    parser.add_argument(
        "--zero_train_path", 
        type=str, 
        required=True, 
        help="Path to the csv containing 0 Ligand Bias train split (e.g. data/zero_ligand_bias_train.csv)"
    )
    parser.add_argument(
        "--zero_test_path", 
        type=str, 
        required=True, 
        help="Path to the csv containing 0 Ligand Bias test split (e.g. data/zero_ligand_bias_test.csv)"
    )
    parser.add_argument(
        "--pdbbind_dir",
        type=str,
        required=True,
        help="Path to PDBbind"
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
        "--output_dir", 
        type=str, 
        required=True, 
        help="Path to save the output CSV (e.g., data)."
    )
    parser.add_argument(
        "--split_ratio", 
        type=float, 
        default=0.1, 
        help="Fraction of data to use for validation (default: 0.1 for 10%%)."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--index_file", 
        type=str, 
        help="Path to the PDBbind index file (e.g., INDEX_general_PL_data.2016) to extract labels. If provided, labels will be added."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Resolve absolute path for output directory and PDBbind dir
    output_dir = project_root / args.output_dir
    pdbbind_dir = project_root / args.pdbbind_dir

    # Load the 0 Ligand Bias train and test csvs
    zero_train_path = project_root / args.zero_train_path
    zero_test_path = project_root / args.zero_test_path

    df_train = pd.read_csv(zero_train_path)
    df_val = df_train.sample(frac=args.split_ratio, random_state=args.seed)
    df_train = df_train.drop(df_val.index) 
    df_test = pd.read_csv(zero_test_path)


    # Now we are going to loop through each of the train, test and val data frames and create the processed csvs
    csv_names = ["Zero_Ligand_Bias_processed_train.csv", "Zero_Ligand_Bias_processed_val.csv", "Zero_Ligand_Bias_processed_test.csv"]

    # First preload the index file and get the binding affinities pK for each complex
    # Load index file if provided, binding affinity 3rd column
    pdb_to_pk = None
    
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
        

    for df_loop, csv_name in zip([df_train, df_val, df_test], csv_names):

        new_df_data = []
        for row in df_loop.itertuples():
            # Get the complex
            pdb_id = row.key

            # Verify that this complex is indeed in PDBbind and get its absolute path
            expected_sdf_file = str(args.sdf_format).replace("X", pdb_id) 
            expected_pdb_file = str(args.pdb_format).replace("X", pdb_id) 
            expected_sdf_path = (pdbbind_dir / pdb_id / expected_sdf_file).resolve()
            expected_pdb_path = (pdbbind_dir / pdb_id / expected_pdb_file).resolve()

            if not expected_sdf_path.exists() or not expected_pdb_path.exists():
                raise FileNotFoundError(f"One of the files: {expected_pdb_path}, {expected_sdf_path} doesn't exist.")
            
            # Get pK value
            pK = pdb_to_pk[pdb_id]

            row_data = {
                "unique_id": pdb_id,
                "pdb_file": str(expected_pdb_path),
                "sdf_file": str(expected_sdf_path),
                "pK": pK
            }

            new_df_data.append(row_data)
    
        # Create dataframe and save to output directory
        new_df = pd.DataFrame(new_df_data)
        output_path = project_root / output_dir / csv_name

        new_df.to_csv(output_path, index=False)

        print(f"Succesfully processed {len(new_df)} complexes")
        print(f"Saving complexes to {output_path}")


if __name__ == "__main__":
    main()
 

