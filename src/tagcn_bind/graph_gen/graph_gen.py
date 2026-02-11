"""
graph_gen.py
=============

This module focuses on generating graphs using a similar approach to that of AEV-PLIG,
except using more conventional approachs of BioPandas to process the .pdbs and RDKit to process the .sdf files. 
Along with changes to AEV calcuations like adding in extra bins for the dummy encoding for the elements of the ligand, rather than sending them all to 6 like in AEV-PLIG.
"""

from biopandas.pdb import PandasPdb
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem
import torch
import numpy as np
import qcelemental as qcel
import torchani
from .. import torchani_mod
from typing import List, Tuple

class GraphGenerator():
    """
    ANI-2x Symmetry Function Coefficients:

    Radial Terms (2-body interactions):
        RcR (float): Radial Cutoff. Max distance to consider a neighbor.
        RsR (list):  Radial Shifts. The specific distances (shells) being probed.
        EtaR (float): Radial Width. Controls the sharpness/thickness of the distance shells.

    Angular Terms (3-body interactions):
        RcA (float): Angular Cutoff. Max distance for atoms to form a valid angle.
        Zeta (float): Angular Focus. Higher values make the sensor more sensitive to exact angles.
        TsA (float): Angular Shift. The specific target angle (in radians/cosine) being detected.
        EtaA/RsA:    Radial parameters ensuring angles are only computed for spatially close triplets.
    """

    # Coefficients for ANI
    radial_coefs: List
    angular_coefs: List  

    # Legacy mode, if True we use the AEV-PLIG implementations and not our improvements
    legacy_mode: bool
    # If this is true use the extra ligand bins defined, otherwise send the whole ligand to 6 for AEV calculation
    ligand_bins: bool
    # If this is true use torchani over torchani_mod
    use_og_torchani: bool

    # The elements we encode for the graph (this is specifically for the graph nodes and node features, the AEVs work slightly differently)
    allowed_elts: List[str]


    def __init__(self, RcR, EtaR, RsR, legacy_mode, ligand_bins, use_og_torchani, allowed_elts, device: str = "cpu"):
        self.radial_coefs = [
            RcR,
            EtaR,
            RsR
        ]

        self.angular_coefs = [
            2.0, # RcA
            torch.tensor([1.0]), # Zeta
            torch.tensor([1.0]), # TsA
            torch.tensor([1.0]), # EstaA
            torch.tensor([1.0]) # RsA
        ] 
        self.legacy_mode = legacy_mode  
        self.ligand_bins = ligand_bins
        self.use_og_torchani = use_og_torchani
        self.allowed_elts = allowed_elts
        self.device = torch.device(device)

    def load_pdb(self, pdb_path: str, atom_keys: pd.DataFrame) -> pd.DataFrame:
        """ Loads a pdb file in BioPandas, creates PDB_ATOM column and uses atom_keys to map this to an ATOM_TYPE

        Args:
            pdb_path (str): path to pdb
            atom_keys (pd.DataFrame): dataframe used to map between "PDB_ATOM" and "ATOM_TYPE", see AEV-PLIG

        Returns:
            pd.DataFrame: dataframe with columns: ATOM_INDEX, ATOM_TYPE, X, Y, Z
        """     

        # Use BioPandas to load the PDB file
        ppdb = PandasPdb().read_pdb(pdb_path)

        # Access the DataFrame containing all the ATOM records
        df = ppdb.df["ATOM"]

        # Filter out hydrogen atoms. 
        df = df[df["element_symbol"] != "H"].copy()

        # Create the PDB_ATOM key (e.g. "ASP-OD1") for merging.
        df["PDB_ATOM"] = df["residue_name"].str.strip() + "-" + df["atom_name"].str.strip()

        # Merge with atom_keys dataframe. Inner join ensures we only
        # keep atoms that have a corresponding ATOM_TYPE in the key file.
        # This means we drop any PDB_ATOMS that do not have a matching ATOM_TYPE
        # If this occurs we throw a warning

        n_before = len(df)

        df = df.merge(atom_keys, on="PDB_ATOM", how = "inner")

        n_dropped = n_before - len(df)

        if n_dropped > 0:
            print(f"Notice: {n_dropped} atoms were dropped because they did not match any ATOM_TYPE in atom_keys.")

        # Rename the biopandas columns to match names used in AEV-PLIG
        df = df.rename(columns={
            "atom_number": "ATOM_INDEX",
            "x_coord": "X",
            "y_coord": "Y",
            "z_coord": "Z"
        })

        # Select only the required columns and sort by the atom index,
        # creating the final, clean DataFrame.
        final_df = df[["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"]].sort_values(by="ATOM_INDEX").reset_index(drop=True)

        return final_df
    
    def load_sdf(self, sdf_path: str) -> rdchem.Mol:
        """ loads an sdf file as a rdkit.Chem.rdchem.Mol
        attempts to load with sanitization (checking things like bad rings)
        if that fails attempts a more lenient loading without certain sanitizations and then
        trys to manually fix the problems afterwards.

        rdkit docs for things like UpdatePropertyCache: https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html

        Args:
            sdf_path (str): path to sdf file

        Raises:
            ValueError: first value error if inital loading fails completely
            ValueError: second value error if lenient loading fails completely

        Returns:
            rdchem.Mol: the rdchem.Mol object storing the contents of the sdf file
        """         

        # Try Strict Loading
        suppl = Chem.SDMolSupplier(str(sdf_path))

        # Check if file exists or is empty
        if not suppl or len(suppl) == 0: 
            raise ValueError(f"{sdf_path} failed to load properly")
        
        mol = suppl[0]

        # If strict loading failed, try lenient loading (removing sanitization)
        if mol is None:
            # Load without sanitization
            suppl_loose = Chem.SDMolSupplier(str(sdf_path), removeHs = False, sanitize = False)
            mol = suppl_loose[0] if len(suppl_loose) > 0 else None

            if mol is not None:
                try:
                    # Create a custom sanitization setting
                    # We catch all errors but ignore Kekulization (bad rings)
                    # and Properties (bad valences)
                    flags = Chem.SANITIZE_ALL
                    flags ^= Chem.SANITIZE_KEKULIZE
                    flags ^= Chem.SANITIZE_PROPERTIES

                    Chem.SanitizeMol(mol, sanitizeOps=flags)

                    # Manually fil properties (like ring info) that we skipped above
                    mol.UpdatePropertyCache(strict = False)

                    print(f"Loaded {sdf_path} skipping some sanitizations")
                    return mol
                except Exception as e:
                    print(f"Failed to recover {sdf_path}: {e}")
                    return None
            else:
                raise ValueError(f"{sdf_path} failed to load without sanitization")
        else:
            print(f"Loaded {sdf_path} with all sanitizations.")
            return mol
    
    def mol_to_df(self, mol: rdchem.Mol) -> pd.DataFrame:
        """ converts the rdchem.Mol object into a dataframe with the following columns:
        ATOM_INDEX, ATOM_TYPE, X, Y, Z. Only taking non hydrogen atoms

        rdkit docs for things like GetConformer: https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html

        Args:
            mol (rdchem.Mol): the rdchem.Mol object

        Returns:
            pd.DataFrame: pandas dataframe with required columns as specified above
        """        

        atoms = []
        # Gets the 3D coords for the molecule object
        conf = mol.GetConformer()

        # Loop through the atoms, taking only the non hydrogen ones, get their symbol, index and (x,y,z) coords
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "H":  # Inlcude only non-hydrogen atoms
                pos = conf.GetAtomPosition(atom.GetIdx())
                atoms.append((
                    atom.GetIdx(),
                    atom.GetSymbol(),
                    pos.x,
                    pos.y,
                    pos.z
                ))

        # Convert this list of tuples to a dataframe with required columns
        df = pd.DataFrame(atoms, columns = ["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"])
        return df
    
    def create_atom_map(self, atom_keys: pd.DataFrame) -> pd.DataFrame:
        """ creates atom_map from atom_keys, atom_map is essentially a dummy encoded for all the different atom types of atom_keys

        Args:
            atom_keys (pd.DataFrame): dataframe with columns PDB_ATOM, ATOM_TYPE

        Returns:
            pd.DataFrame: dataframe with columns ATOM_TYPE (from atom_keys) and ATOM_NR (dummy encoding for torchani)
        """         
        
        atom_map = pd.DataFrame(pd.unique(atom_keys["ATOM_TYPE"]))
        atom_map[1] = list(np.arange(len(atom_map)) + 1)
        atom_map = atom_map.rename(columns={0:"ATOM_TYPE", 1:"ATOM_NR"})

        return atom_map

    @staticmethod
    def map_ligand_atom_to_id(element_symbol: str, len_atom_map: int) -> int:
        """ this function maps the ligand elements to a dummy atomic number (ATOM_NR in the pdb). We manually set the start index based off the length of atom map
        WARNING you must always have the catch all bin as the final one, and never add coverage for "WXZ" otherwise we will have the wrong length of the species converter when
        we generate AEVs

        Args:
            element_symbol (str): the element symbol
            len_atom_map (int): length of the atom map

        Returns:
            int: the dummy atomic number encoding
        """        

        # Clean input
        elem = element_symbol.strip().upper()

        # Start one after the atom_map used to encode the protein atoms 
        start_id = len_atom_map + 1

        # Big three
        if elem == 'C':
            return start_id
        if elem == 'N':
            return start_id + 1
        if elem == 'O':
            return start_id + 2

        # "Heavies"
        if elem in ['S', 'P']:
            return start_id + 3

        # Halogens
        if elem in ['F', 'CL', 'BR', 'I']:
            return start_id + 4

        # Catch all
        return start_id + 5 

    
    def get_mol_aevs(self, protein_df: pd.DataFrame, ligand_df: pd.DataFrame, atom_map: pd.DataFrame) -> Tuple[pd.DataFrame, torch.Tensor]:
        """ This function computes the AEVs using two possible approaches: the old logic of AEV-PLIG, vs our newer refactored methods.

        The legacy version of this function is as follows:

        1. Filters protein atoms to only those within a bounding box of the ligand (the pocket).
        2. Encodes ligand atoms as 6 (this is not carbon! (or at least kinda not) in the default atom_map this will be C;4;2;2;2;0;0) and protein atoms are mapped to dummy atomic numbers (see config/atom_map.csv).
        3. Uses TorchANI_mod to compute AEVs for the combined system, torchani_mod is modified to only compute AEVs for ligand-protein pairs.
        4. Returns only the radial components of the AEVs for the ligand atoms.

        We propose two modifications, the first is to use torchani 2.4.0 rather than torchani_mod, this means computing AEVs for all possible pairs so protein - protein, and ligand - ligand as well.
        The second is to change the dummy encoding of the ligand. This dummy encoding only works because the radial AEVs only depend on the distance between the atoms (see here models AEV section: https://www.nature.com/articles/s42004-025-01428-y.pdf).
        However this means that the ligand has been entirely dummy encoded to 6 which corresponds to C;4;2;2;0;0. Instead we hypothesise it would be better to do additionally dummy encoding for the ligand to add in say S, Cl etc. but also to seperate the encodings 
        for the proteins atoms from that of the ligands. 

        Args:
            protein_df (pd.DataFrame): data frame for the protein, columns: ATOM_INDEX, ATOM_TYPE, X, Y, Z
            ligand_df (pd.DataFrame): data frame for ligand, columns: ATOM_INDEX, ATOM_TYPE, X, Y, Z
            atom_map (pd.DataFrame): the atom map that maps ATOM_TYPEs of the pdb into dummy encoded "atomic numbers"

        Returns:
            Tuple[pd.DataFrame, torch.Tensor]: the ligand data frame and a tensor of the AEVs for each of the ligand atoms
        """



        # Setup Coefficients (ANI-2x parameters)
        # Radial coefficients (from self.radial_coefs)
        RcR = self.radial_coefs[0]
        EtaR = self.radial_coefs[1]
        RsR = self.radial_coefs[2]

        # Angular coefficients (from self.angular_coefs)
        RcA = self.angular_coefs[0]
        Zeta = self.angular_coefs[1]
        TsA = self.angular_coefs[2]
        EtaA = self.angular_coefs[3]
        RsA = self.angular_coefs[4]

        # Vectorized pocket filtering
        # Define the bounding box around the ligand with a buffer
        distance_cutoff = RcR + 0.1
        min_coords = ligand_df[["X", "Y", "Z"]].min() - distance_cutoff
        max_coords = ligand_df[["X", "Y", "Z"]].max() + distance_cutoff

        # Filter protein atoms efficiently using boolean indexing
        target_df = protein_df[
            (protein_df["X"] >= min_coords["X"]) & (protein_df["X"] <= max_coords["X"]) &
            (protein_df["Y"] >= min_coords["Y"]) & (protein_df["Y"] <= max_coords["Y"]) &
            (protein_df["Z"] >= min_coords["Z"]) & (protein_df["Z"] <= max_coords["Z"])
        ].copy()

        # Map protein atoms to their atomic numbers using the pre-computed map
        target_df = target_df.merge(atom_map, on = "ATOM_TYPE", how = "left")

        # Prepare Tensors for TorchANI
        mol_len = len(ligand_df)

        # We now have a choice, in AEV-PLIG they used a hack where they encode the entire ligand as dummy atomic number 6 (see the docstring for more info)
        # or add more bins for the ligands atoms using map_ligand_atom_to_id 
        # Legacy_mode deterimines if you use our refactor or the old logic of AEV-PLIG
        # ligand_bins then determines whether you decide to stick to the old method or use our additional ligand bins
        if self.legacy_mode or not self.ligand_bins:  
            # Create atomic numbers tensor
            # ligand atoms are encoded as Carbon (6) - preserving original logic of AEV-PLIG
            ligand_species = np.ones(mol_len) * 6
            protein_species = target_df["ATOM_NR"].values

            # Species converter length (how many dummy atomic numbers do we have?)
            species_converter_len = len(atom_map)
        else:
            # Must be in the case that self.legacy_mode = False and self.ligand_bins = True, so we use our extra ligand bins method
            # Calculate the length of the atom map to determine when to start the ligand ATOM_NRs
            len_atom_map = len(atom_map)

            # Apply the mapping function to the ATOM_TYPE column
            ligand_df["ATOM_NR"] = ligand_df["ATOM_TYPE"].apply(
                lambda x: self.map_ligand_atom_to_id(element_symbol=x, len_atom_map=len_atom_map)
            )
            
            # Now get the ligand species from this column in the same way we got the protein species
            ligand_species = ligand_df["ATOM_NR"].values
            protein_species = target_df["ATOM_NR"].values

            # Assuming we never add coverage for element WXZ this should return us the catch all bin
            species_converter_len = self.map_ligand_atom_to_id(element_symbol="WXZ", len_atom_map=len_atom_map)

        atomic_nums = np.concatenate([ligand_species, protein_species]).astype(np.int64)
        atomic_nums = torch.tensor(atomic_nums, device=self.device).unsqueeze(0) # Shape: (1, Total_Atoms) 

        # Create coordinates tensor
        ligand_coords = ligand_df[["X", "Y", "Z"]].values
        protein_coords = target_df[["X", "Y", "Z"]].values

        coordinates = np.concatenate([ligand_coords, protein_coords])
        coordinates = torch.tensor(coordinates, dtype=torch.float32, device=self.device).unsqueeze(0) # Shape: (1, Total_Atoms, 3)

        # Display head of ligand_df and target_df
        # print("Ligand df head: ", ligand_df.head())
        # print("ligand species first 10 ", ligand_species[:10])
        # print("Target df head: ", target_df.head())
        # print("protein species first 10 ", protein_species[:10])
        # print("Atomic numbers first 50 ", atomic_nums[0][:50])

        # Compute AEVs
        # Define atom symbols for the species converter (H to whatever species_converter_len is which is set above)
        print("Value of species converter len: ", species_converter_len)
        atom_symbols = [qcel.periodictable.to_symbol(i) for i in range(1, species_converter_len + 1)]

        # Initialize AEV Computer
        # We again have a choice, do we use torchani_mod as in AEV-PLIG that restricts AEVs to only ligand-protein pairs (and in the original repo may or may not have also disabled CUAEV, but we have a modified version that reverts this back to the original implementation)
        # Or do we use the original torchani and compute AEVs across all possible pairs (ligand-ligand, protein-protein, ligand-protein)
        if self.legacy_mode or not self.use_og_torchani:
            aev_computer = torchani_mod.AEVComputer(RcR, RcA, EtaR, RsR,
                                            EtaA, Zeta, RsA, TsA,
                                            len(atom_symbols))
        else:
            # we are now in the case of self.legacy_mode = False and self.use_og_torchani = True, so we use og torchani
            aev_computer = torchani.AEVComputer(RcR, RcA, EtaR, RsR,
                                            EtaA, Zeta, RsA, TsA,
                                            len(atom_symbols))
        
        # Move computer to device (GPU if selected)
        aev_computer = aev_computer.to(self.device)

        # Move SpeciesConverter to GPU as well to avoid errors
        species_converter = torchani.SpeciesConverter(atom_symbols).to(self.device)
        
        # Run forward pass
        # Note: SpeciesConverter usually runs on CPU for indexing, but we pass tensors on device
        species, coords = species_converter((atomic_nums, coordinates))

        # print("Species (before aev_computer) first 50 ", species[0][:50])
        
        # If we use torchani_mod we neeed to pass mol_len so it knows where the split is between the protein and the ligand
        if self.legacy_mode or not self.use_og_torchani:
            # Ensure mol_len is a tensor for torchani_mod
            mol_len_tensor = torch.tensor([mol_len], device=coords.device)
            aev_result = aev_computer((species, coords), mol_len_tensor)
            # torchani_mod returns an object with .aevs attribute
            full_aevs = aev_result.aevs
        else:
            aev_result = aev_computer((species, coords))
            # Standard torchani returns (species, aevs) tuple
            _, full_aevs = aev_result

        # Slice and Return
        # We only want the AEVs for the ligand atoms (first mol_len indices)
        # And we only want the radial terms (based on original slicing logic)

        n_symbols = len(atom_symbols)
        n_radial_sub = len(EtaR) * len(RsR)

        # Calculate indices for radial terms: 0 to ((22 or 27 (maybe)) * 16)
        radial_indices = list(np.arange(n_symbols * n_radial_sub))

        # Extract: (Batch 0) -> (Ligand Atoms) -> (Radial features)
        final_aevs = full_aevs.squeeze(0)[:mol_len, radial_indices]

        return ligand_df, final_aevs

    @staticmethod
    def one_of_k_encoding_atoms(x: str, allowable_set: List[str]) -> List[int]:
        """ this encodes the elements of the ligand into an array of size len(allowable_set) + 1
        usually this is something like say [C, S, N, O], in this case we dummy encode all of these elements and just
        add an extra category for "other" in case an unusal element appears in the ligand 

        This is different from AEV-PLIG which uses a function similar to one_of_k_encoding_bonds for the atoms. This is less
        robust if the ligand has an unseeen element though.         

        Args:
            x (str): element symbol 
            allowable_set (List[str]): list of accepted element symbols

        Returns:
            List[int]: one hot encodings of these accepted elements + other
        """        
        
        # The encoding will have one extra spot for "other", this is different to AEV-PLIG that requires all elements of the liagand to lie in the allowed_elts array
        encoding = [0] * (len(allowable_set) + 1)
        
        try:
            index = allowable_set.index(x)
            encoding[index] = 1
        except ValueError:
            # x is not in the allowable_set so it's an "other"
            encoding[-1] = 1
        return encoding
    
    @staticmethod
    def one_of_k_encoding_bonds(x, allowable_set: list) -> List[int]:
        """ more generic version of one_of_k_encoding_atoms, this doesn't add an other category and so will throw an 
        error if we have something not in the alloable list. this is generally only safe to use for encoding bonds hence the name

        Args:
            x: object to be encoded
            allowable_set (list): list of allowed object types

        Raises:
            ValueError: throws error if x not in allowable_set

        Returns:
            List[int]: one hot encoding of the allowable objects
        """         
        # The original one_of_k_encoding of AEV-PLIG now restricted to just bonds
        if x not in allowable_set:
            raise ValueError(f"Input {x} not in allowable set {allowable_set}")
        return list(map(lambda s: int(x==s), allowable_set))
    
    def get_atom_features(self, atom:rdchem.Atom) -> np.ndarray:
        """ computes features for the atoms:
        first one hot encode the element type using one_of_k_encoding_atoms
        then get number of heavy neighbours (not hydrogen)
        then get number of hydrogens as neighbours
        then get explicit valence
        then check if aromatic
        then check if is in ring

        Args:
            atom (rdchem.Atom): the rdchem.Atom to get features for

        Returns:
            np.ndarray: an array of these features
        """        
        features = []
    
        # Atom symbol one-hot
        features.extend(self.one_of_k_encoding_atoms(atom.GetSymbol(), self.allowed_elts))
        
        # Number of heavy atom neighbours
        features.append(len([x for x in atom.GetNeighbors() if x.GetSymbol() != "H"]))

        # Number of Hyrogens
        features.append(len([x for x in atom.GetNeighbors() if x.GetSymbol() == "H"]))

        # Explicit Valence
        features.append(atom.GetExplicitValence())

        # Aromaticity
        features.append(int(atom.GetIsAromatic()))

        # In Ring
        features.append(int(atom.IsInRing()))

        return np.array(features) 
     
    def mol_to_graph(self, mol: rdchem.Mol, aevs: torch.Tensor, mol_df: pd.DataFrame) -> Tuple[int, List[np.ndarray], List[List[int]], List[List[float]]]:
        """ converts molecule and AEVs to graph representation, using the features get_atom_features, and also recording the bonds as edge features

        Args:
            mol (rdchem.Mol): the ligand to process
            aevs (torch.Tensor): the aevs for each ligand atom
            mol_df (pd.DataFrame): the processed df of the ligand, used to map the rdchem.Mol atoms to the aevs

        Raises:
            ValueError: raises error if we can't map between the atom index found from RDKit and that of the ligand_df

        Returns:
            Tuple[int, List[np.ndarray], List[List[int]], List[List[float]], int, int]: the graph stored as a tuple, the number of features, the node features, the edge index and the edge attributes (bond types), then finally the length of the chemical features, and the length of the AEVs
        """             
            
            

        features = []
        idx_to_idx = {} # Maps RDKit atom index to our 0,...,N-1 index

        # Ensures AEVs are on CPU and numpy for easy concatenation
        if isinstance(aevs, torch.Tensor):
            aevs_np = aevs.detach().cpu().numpy()
        else:
            aevs_np = aevs
            
        # Create a mapping from the atom index (in the mol object) to the index in the AEV array
        # This assumes aevs are aligned with mol_df rows
        atom_idx_to_aev_idx = {atom_idx: i for i, atom_idx in enumerate(mol_df["ATOM_INDEX"])}

        # Later on when it comes to standardising the data it will be useful to know the split point in the node features between the 
        # one hot encoded and integer value chem_feats, vs the continuous aev_feats

        # Track current index
        counter = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "H":
                idx_to_idx[atom.GetIdx()] = counter

                # Get chemical features
                chem_feats = self.get_atom_features(atom)

                # Get AEV features
                # Use the lookup to ensure we get the correct AEV for this atom index
                if atom.GetIdx() in atom_idx_to_aev_idx:
                    aev_idx = atom_idx_to_aev_idx[atom.GetIdx()]
                    aev_feats = aevs_np[aev_idx]
                else:
                    raise ValueError(f"Atom index {atom.GetIdx()} from RDKit mol not found in mol_df.")

                # Store the length of chem_feats, and aev_feats for later use, as a sanity check ensure that the length of chem_feats isn't changing across atoms, do the same for AEVs
                if counter == 0: 
                    len_chem_feats = len(chem_feats)
                    len_aev_feats = len(aev_feats)
                elif len(chem_feats) != len_chem_feats: 
                    raise ValueError(f"Atom index {atom.GetIdx} has a different length to chemical features.")  
                elif len(aev_feats) != len_aev_feats:
                    raise ValueError(f"Atom indx {atom.GetIdx} has a different length to AEV features.") 

                # Concatenate
                combined_feats = np.concatenate([chem_feats, aev_feats])
                features.append(combined_feats)

                counter += 1
        # Edges
        edges = []
        # RDKit BondType mapping to match legacy: [Single, Aromatic, Double, Triple]
        bond_encoding_list = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.AROMATIC,
                                  Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]
            
        for bond in mol.GetBonds():
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()

            # Only consider bonds between heavy atoms
            if idx1 in idx_to_idx and idx2 in idx_to_idx:
                u = idx_to_idx[idx1]
                v = idx_to_idx[idx2]

                bond_type = self.one_of_k_encoding_bonds(bond.GetBondType(), bond_encoding_list)

                # Legacy code converts to float
                bond_attr = [float(b) for b in bond_type]

                # Add undirected edge (u, v) and (v, u)
                edges.append([u, v] + bond_attr)
                edges.append([v, u] + bond_attr)

        # Sort edges by atom indices (legacy behaviour, helps with determinism)
        if edges:
            edges_df = pd.DataFrame(edges, columns = ["atom1", "atom2", "single", "aromatic", "double", "triple"])
            edges_df = edges_df.sort_values(by=["atom1", "atom2"])

            edge_index = edges_df[["atom1", "atom2"]].values.tolist()
            edge_attr = edges_df[["single", "aromatic", "double", "triple"]].values.tolist()
        else:
            edge_index = []
            edge_attr = []

        return len(features), features, edge_index, edge_attr, len_chem_feats, len_aev_feats
