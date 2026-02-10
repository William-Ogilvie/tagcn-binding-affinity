"""
test_graph_gen.py
==================

This script will test graph_gen.py. Place all sdf and pdbs you would like to test inside the data/test directory, in the format:
data/test/1a30
"""
from tagcn_bind import GraphGenerator 
from pathlib import Path
import torch
import pandas as pd
import yaml

# Get the absolute path of the current script
script_path = Path(__file__).resolve()

# Go up two levels to get the project root:
project_root = script_path.parent.parent
# Test data directory
data_dir = project_root / "data" / "test"

# Graph Gen config:
graph_gen_config_path = project_root / "config" / "graph_gen_config.yml"

# Load config
with open(graph_gen_config_path, "r") as f:
        config = yaml.safe_load(f)

# Extract radial coefs and legacy mode
RcR = config["graph_generator_coef"]["RcR"]
EtaR = torch.tensor(config["graph_generator_coef"]["EtaR"])
RsR = torch.tensor(config["graph_generator_coef"]["RsR"])
legacy_mode = config["graph_generator_coef"]["legacy_mode"]
ligand_bins = config["graph_generator_coef"]["ligand_bins"]
use_og_torchani = config["graph_generator_coef"]["use_og_torchani"]
allowed_elts = config["graph_generator_coef"]["allowed_elts"]

print(RcR, EtaR, RsR, legacy_mode)

# Get atom_keys for pdb files:
pdb_atom_keys_path = project_root / "config" / "PDB_Atom_Keys.csv"

# Load the csv
atom_keys = pd.read_csv(pdb_atom_keys_path, sep=",")

# Get path to pdb and sdf files
pdb_path = project_root / "data" / "test" / "1a30" / "1a30_protein.pdb"
sdf_path = project_root / "data" / "test" / "1a30" / "1a30_ligand.sdf"

# Se ligand to test the allowed elts
#sdf_path = project_root / "data" / "test" / "1a30" / "SE4_ideal.sdf"

# Initialise GraphGenerator
graph_generator = GraphGenerator(RcR=RcR, EtaR=EtaR, RsR=RsR, legacy_mode=legacy_mode, ligand_bins=ligand_bins, use_og_torchani=use_og_torchani, allowed_elts=allowed_elts)

# Test methods:
pdb_df = graph_generator.load_pdb(pdb_path=pdb_path, atom_keys=atom_keys)
print(pdb_df.head())
sdf_mol = graph_generator.load_sdf(sdf_path=sdf_path)
print(sdf_mol)
sdf_df = graph_generator.mol_to_df(mol=sdf_mol)
print(sdf_df.head(10))
print(sdf_df["ATOM_TYPE"].unique)
print(atom_keys.head(10))
atom_map = graph_generator.create_atom_map(atom_keys=atom_keys)
print(atom_map.head(10))
atom_map.to_csv(data_dir / "atom_map.csv")
ligand_df, final_aevs = graph_generator.get_mol_aevs(protein_df=pdb_df, ligand_df=sdf_df, atom_map=atom_map)
print(ligand_df.head())

# Inspect AEVs to verify non-zero outputs
print(f"Final AEVs shape: {final_aevs.shape}")

if (final_aevs != 0).any():
    # Find the first atom (row) that has non-zero values
    non_zero_rows = torch.any(final_aevs != 0, dim=1)
    first_nz_idx = torch.where(non_zero_rows)[0][0].item()
    
    print(f"\nFirst atom with non-zero content is Atom Index: {first_nz_idx}")
    print("Non-zero values for this atom:")
    # Print only the non-zero elements for this atom
    atom_row = final_aevs[first_nz_idx]
    for i, val in enumerate(atom_row):
        if val != 0:
            print(f"  Feature {i}: {val.item():.6f}")
else:
    print("WARNING: All AEVs are zero.")

graph = graph_generator.mol_to_graph(mol=sdf_mol, aevs=final_aevs, mol_df=sdf_df)
# these can be quite big:
# print("len features ", graph[0])
# print("features: ", graph[1][0])
# print("edge index: ", graph[2])
# print("edge attr: ", graph[3])