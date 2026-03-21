"""
identify_elt_bond_types.py
===========================
This script will loop through the targeted processed csv and identify all the unique elements and bond types in the sdf files.
It will return a csv giving the number of each unique element and also the number of different complexes it is present in.

Usage:
    python identify_elt_bond_types.py --processed_file data/PDBbind_minus_CASF_2016_processed.csv --output_dir data/graph_stats --output_tag PDBbind_train

    python identify_elt_bond_types.py --processed_file data/Zero_Ligand_Bias_processed_train.csv --second_processed_file data/Zero_Ligand_Bias_processed_val.csv --output_dir data/graph_stats --output_tag Zero_Ligand_Bias_train
"""

import argparse
import pandas as pd
from pathlib import Path
from tagcn_bind import GraphGenerator
import sys
from tqdm import tqdm

# Get the absolute path of the current script
script_path = Path(__file__).resolve()

# Go up two levels to get the project root:
project_root = script_path.parent.parent

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reads a processed csv and generates statistics on element type and bond types."
    )
    parser.add_argument(
        "--processed_file", 
        type=str, 
        required=True, 
        help="Relative path to processed file"
    )
    # This is used as we have saved the OOD Test csvs as two separate ones for the train and val sets
    parser.add_argument(
        "--second_processed_file",
        type=str,
        default="None",
        required=False,
        help="Second process file if needed."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Location to save output csv."
    )
    parser.add_argument(
        "--output_tag", 
        type=str, 
        required=True, 
        help="Tag for file name."
    ) 
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Resolve absolute path for processed file
    processed_file_path = (project_root / args.processed_file).resolve()

    df = pd.read_csv(processed_file_path)
    total_complexes = len(df)
    # We are going to store the bond information and elements in two dictionaries with arrays at each key,
    # the first number being a raw count and the second being a count of the number of complexes
    elt_dict = {
        "Other_Elements": [0,0],
        "Metalloids": [0,0],
        "Rare_Metals": [0,0],
        "Unknown": [0,0]
    }
    bond_dict = dict()

    metalloids_list = ["Si", "As", "Sb", "Te"]
    rare_metals = ["Be", "Mg", "Ir", "Cu", "Co", "V", "Pt", "Rh", "Os", "Re", "Zn"]
    other_elements_list = ["Fe", "Ru", "Se", "Si", "As", "Sb", "Te", "Be", "Mg", "Ir", "Cu", "Co", "V", "Pt", "Rh", "Os", "Re", "Zn", "*"] 

    # We will use an instance of the graph generator to load the sdfs
    graph_gen = GraphGenerator(1.0, 1.0, 1.0, False, [2.0], True, [2.0])

    non_carbon_sdfs = []

    for row in tqdm(df.itertuples(), desc="SDF files."):
        
        sdf_path = row[3]

        # Load the sdf
        mol = graph_gen.load_sdf(sdf_path=sdf_path)

        # Deterimine elements
        unique_elts = set()
        for atom in mol.GetAtoms():
            symbol = str(atom.GetSymbol())

            if symbol not in elt_dict.keys():
                elt_dict[symbol] = [1,0] # We add 1 at the end for the number of complexes
            else:
                elt_dict[symbol][0] += 1

            # If the symbol is from the Other_Elements bin we need to record that too
            if symbol in other_elements_list:
                elt_dict["Other_Elements"][0] += 1
                
            
            if symbol in metalloids_list:
                elt_dict["Metalloids"][0] += 1 
            elif symbol in rare_metals:
                elt_dict["Rare_Metals"][0] += 1
            elif symbol == "*":
                elt_dict["Unknown"][0] += 1
            
            unique_elts.add(symbol)

        # Increase complex count
        increased_metalloids = False
        increased_rare_metals = False
        increased_unkown = False
        increased_other_elts = False
        for symbol in unique_elts:
            elt_dict[symbol][1] += 1

            # We want to increase the second term of Other_Elements at most once per complex
            if symbol in other_elements_list and not increased_other_elts:
                elt_dict["Other_Elements"][1] += 1
                increased_other_elts = True
            
            if symbol in metalloids_list and not increased_metalloids:
                elt_dict["Metalloids"][1] += 1
                increased_metalloids = True
            
            if symbol in rare_metals and not increased_rare_metals:
                elt_dict["Rare_Metals"][1] += 1
                increased_rare_metals = True

            if symbol == "*" and not increased_unkown:
                elt_dict["Unknown"][1] += 1
                increased_unkown = True

        # For curiosity if carbon is not in the elt_dict i want to flag the sdf file
        if "C" not in unique_elts:
            non_carbon_sdfs.append(sdf_path)
            print(sdf_path)  

        # Deterimine bonds
        unique_bonds = set()
        for bond in mol.GetBonds():
            b_type = str(bond.GetBondType())

            if b_type not in bond_dict.keys():
                bond_dict[b_type] = [1,0]
            else:
                bond_dict[b_type][0] += 1
            
            unique_bonds.add(b_type)

        for b_type in unique_bonds:
            bond_dict[b_type][1] += 1

          



    # if there is a second processed file repeat the above
    if args.second_processed_file != "None":
        # Resolve absolute path for processed file
        processed_file_path = (project_root / args.second_processed_file).resolve()
        df = pd.read_csv(processed_file_path)
        total_complexes += len(df) 
        for row in tqdm(df.itertuples(), desc="SDF files."):
        
            sdf_path = row[3]

            # Load the sdf
            mol = graph_gen.load_sdf(sdf_path=sdf_path)

            # Deterimine elements
            unique_elts = set()
            for atom in mol.GetAtoms():
                symbol = str(atom.GetSymbol())

                if symbol not in elt_dict.keys():
                    elt_dict[symbol] = [1,0] # We add 1 at the end for the number of complexes
                else:
                    elt_dict[symbol][0] += 1 

                # If the symbol is from the Other_Elements bin we need to record that too
                if symbol in other_elements_list:
                    elt_dict["Other_Elements"][0] += 1

                if symbol in metalloids_list:
                    elt_dict["Metalloids"][0] += 1 
                elif symbol in rare_metals:
                    elt_dict["Rare_Metals"][0] += 1
                elif symbol == "*":
                    elt_dict["Unknown"][0] += 1
            

                unique_elts.add(symbol)

            # Increase complex count
            increased_other_elts = False
            for symbol in unique_elts:
                elt_dict[symbol][1] += 1
                # We want to increase the second term of Other_Elements at most once per complex
                if symbol in other_elements_list and not increased_other_elts:
                    elt_dict["Other_Elements"][1] += 1
                    increased_other_elts = True

                if symbol in metalloids_list and not increased_metalloids:
                    elt_dict["Metalloids"][1] += 1
                    increased_metalloids = True
                
                if symbol in rare_metals and not increased_rare_metals:
                    elt_dict["Rare_Metals"][1] += 1
                    increased_rare_metals = True

                if symbol == "*" and not increased_unkown:
                    elt_dict["Unknown"][1] += 1
                    increased_unkown = True
            # For curiosity if carbon is not in the elt_dict i want to flag the sdf file
            if "C" not in unique_elts:
                non_carbon_sdfs.append(sdf_path)
                print(sdf_path)  



            # Deterimine bonds
            unique_bonds = set()
            for bond in mol.GetBonds():
                b_type = str(bond.GetBondType())

                if b_type not in bond_dict.keys():
                    bond_dict[b_type] = [1,0]
                else:
                    bond_dict[b_type][0] += 1
                
                unique_bonds.add(b_type)

            for b_type in unique_bonds:
                bond_dict[b_type][1] += 1

    df_elts = pd.DataFrame.from_dict(elt_dict, orient='index', columns=["Raw_Count", "Complex_Occurance"])
    df_elts = df_elts.sort_values(by="Raw_Count", ascending=False)
    
    df_elts["Percent_of_Dataset"] = (df_elts["Complex_Occurance"] / total_complexes) * 100

    df_bonds = pd.DataFrame.from_dict(bond_dict, orient='index', columns=["Raw_Count", "Complex_Occurance"])
    df_bonds = df_bonds.sort_values(by="Raw_Count", ascending=False)
    df_bonds["Percent_of_Dataset"] = (df_bonds["Complex_Occurance"] / total_complexes) * 100

    # Save to output dir with the file name tag
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok = True)
    file_tag = args.output_tag

    elts_path = output_dir / f"element_statistics_{file_tag}.csv"
    bonds_path = output_dir / f"bond_statistics_{file_tag}.csv"

    df_elts.to_csv(elts_path)
    df_bonds.to_csv(bonds_path) 

    print("Finished!")
    print("Non carbon sdf paths were: ", non_carbon_sdfs)


if __name__ == "__main__":
    main() 