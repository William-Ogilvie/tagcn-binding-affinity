# TAGCN Binding Affinity

This repo investigates the use of TAGCN instead of GATv2 within the architecture proposed by AEV-PLIG. See [Acknowledgments](#acknowledgments) for attribution. 

# Usage

## Preprocessing data

To run scripts/create_dataset_csv.py in order to generate a process csv we can pass GraphGenerationManager save your dataset in a directory, for example: data/CASF-2016. Ensure that this is structured to only have subdirectories of the form 1a30, 5c28 etc. You can specify the format of the sdf files using --sdf_format X_ligand, where X is the name of the parent folder. You can pass an index file to get the binding affinities for each protein-ligand complex. 

```bash
python create_dataset_csv.py --data_dir data/CASF-2016 --sdf_format X_ligand.sdf --pdb_format X_protein.pdb --output_csv data/CASF_2016_processed.csv --index_file data/index/INDEX_general_PL_data.2020
```

```bash
python create_dataset_csv.py --data_dir data/PDBbind --sdf_format X_ligand.sdf --pdb_format X_protein.pdb --output_csv data/PDBbind_processed.csv --index_file data/index/INDEX_general_PL_data.2020
```

To remove the test set CASF-2016 from PDBbind we use the remove_dataset_csv.py script. This returns a new csv which in this case will be PDBbind - CASF-2016.

```bash
python remove_dataset_csv.py --input_csv data/PDBbind_processed.csv --remove_csv data/CASF_2016_processed.csv --output_csv data/PDBbind_minus_CASF_2016_processed.csv
```

We use a 10% validation set for early stopping. To generate this use the create_val_dataset_csv.py script.

```bash 
python create_val_dataset_csv.py --input_csv data/PDBbind_minus_CASF_2016_processed.csv --val_csv data/PDBbind_processed_val.csv --train_csv data/PDBbind_processed_train.csv --split_ratio 0.1 --seed 37
```

To generate graphs from these csvs first configure the config/graph_generation.yml file using the template provided. Then run:

```bash
python scripts/generate_graphs.py --config_path config/graph_generation_PDBbind.yml --device auto
```

This project allows for the standardisation of the AEVs using the training set, we precompute these and then apply the standardisation in real time during training:

```bash
python scripts/scaling_script.py --config_path config/scaling_generation_PDBbind.yml
```

## Training

To train models first configure config/experiments_config.yml using the template provided then run:

```bash
python scripts/training_script.py --config_path config/experiments_config.yml --device auto
```

## Plotting

We have a script called generate_plots.py that creates plots configured using a config file like config/plotting_config.yml:

```bash
python scripts/generate_plots.py --config_path config/plotting_config.yml
```


## Modifications

If you plan on adding more features to the model please see src/tagcn_bind/utils/scaling.py. Specifically the calculate_stats method docstring as it contains important information on how to add continuous vs categorical node features to ensure we standardise properly. 


## TODO
Remeber to look at the EtaR, i suspect 19.7 is too high and causes the sparsity in AEVs we observe.

It seems the models overfit to the training data quite considerable (see models.py), have introduced new models to help combat this but could also look at increased weight decay etc. However these may need to be tuned? Have run experiments with 50% dropout and gotten slightly worse results so maybe only 20% needed, same with weight decay originally tried 0.01 have set to 0.0001. Also previously tried dropout in convolution layers have since moved to only MLP. 

Another important thing is that TAG doesn't include edge attributes like GAT does, so the edge features (bond types) aren't being passed to the model. If TAG can still perform well this would suggest these are maybe redundant if you already have AEVs as node features?

AEV-PLIG currently doesn't process hydrogens and instead has them implicitly with a hydrogen count at each heavy atom, it could be worth having explicit hydrogens? althought it will make the graphs a lot bigger...

Look at batch size. We are using the default of AEV-PLIG of 128 but it could be worth dropping to say 64 or even 32 as it may avoid oversmoothing although it would increase training time.

# Acknowledgments

## AEV-PLIG

This tool is build upon the excellent research of Ísak Valsson of the Oxford Protein Informatics Group (AEV-PLIG). AEV-PLIG is a GNN-based scoring function that predicts the binding affinity of a bound protein-ligand complex given its 3D structure. The paper is published in Nature's *Communications Chemistry* at [Narrowing the gap between machine learning scoring functions and free energy perturbation using augmented data](https://doi.org/10.1038/s42004-025-01428-y).

AEV-PLIG was first published in [How to make machine learning scoring functions competitive with FEP](https://chemrxiv.org/engage/chemrxiv/article-details/6675a38d5101a2ffa8274f62), and received the [people's poster prize at the 7th AI in Chemistry Symposium](https://www.stats.ox.ac.uk/news/isak-valsson-wins-poster-prize).

Valsson, Í., Warren, M.T., Deane, C.M. et al. Narrowing the gap between machine learning scoring functions and free energy perturbation using augmented data. Commun Chem 8, 41 (2025). https://doi.org/10.1038/s42004-025-01428-y 

See NOTICE.md for full attribution. 

## TorchANI

This project makes use of TorchANI to compute the radial AEVs as in AEV-PLIG. Inside of src/tagcn_bind there is a modified fork of TorchANI 2.4.0 called torchani_mod that was used by Ísak Valsson in the original AEV-PLIG repository. 

TorchANI was developed and is currently maintained (as of 10/02/2026) by the Roitberg group, and is available for use under an MIT license. We include the original license inside the torchani_mod directory, as well as a copy inside the NOTICE.md file. 

Pickering, I., Xue, J., Huddleston, K., Terrel, N. & Roitberg, A. E. TorchANI 2.0: An Extensible, High-Performance Library for the Design, Training, and Use of NN-IPs. J. Chem. Inf. Model. 65, 11656–11671 (2025). https://doi.org/10.1021/acs.jcim.5c01853

Gao, X., Ramezanghorbani, F., Isayev, O., Smith, J. S. & Roitberg, A. E. TorchANI: A Free and Open Source PyTorch-Based Deep Learning Implementation of the ANI Neural Network Potentials. J. Chem. Inf. Model. 60, 3408–3415 (2020). https://doi.org/10.1021/acs.jcim.0c00451

See NOTICE.md for full attribution.
