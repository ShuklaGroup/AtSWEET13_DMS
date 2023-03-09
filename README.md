# AtSWEET13_DMS
Python code used for PyRosetta calculations and the development, validation, and testing of machine learning models is included in this repository location.

Files belonging to phylogenetic analyses and structural bioinformatics were uploaded to Box due to size restrictions.

The Box link can be found at:
https://uofi.box.com/s/d8f9vllkh9s17uwvyr4ldu156jtbof1s

A tree architecture for the entire github/Box directory is depicted below:

```
.
|____SWEET_phylogeny
| |____msas
| | |____ # Multiple sequence alignments for PF03083 sequences
| | 
| |____maximum_likelihood_tree
| | |____ # Contains maximum likelihood tree for bootstrapping from RAxML
| | 
| |____bootstrapped_tree
| | |____ # Contains bootstrapped tree from IQ-TREE, along with additional analyses
| | 
| |____raw_fastas
|   |____ # PF03083 Clade 1 SWEET sequences and STREME motif analysis
|   | 
|   |____ # PF03083 Clade 2 SWEET sequences and STREME motif analysis
|   | 
|   |____ # PF03083 Clade 3 SWEET sequences and STREME motif analysis
|   | 
|   |____ # PF03083 Clade 4 SWEET sequences and STREME motif analysis
|   | 
|   |____ predicted_structures_TM4
|   | |____ # Contains predicted structures for representative clade sequences, along with cataloged TM4 contacts.
|   | 
|   |____ TM4_substitution_analysis
|     |____ # Contains STREME motif probabilities and analysis scripts/calculations used per clade
| 
|____dimensionality_reduction
| |____ # Contains AtSWEET13 DMS data and a Jupyter notebook for applying dimensionality reduction
| 
|____in_silico_dms
| |____candidate_pdbs
| | |____ # Potential PDBs obtained from https://doi.org/10.1101/2022.10.12.511964 for analysis
| | 
| |____S1_PDB84
| | |____ # Results for RosettaMP, PoPMuSiC, and mCSM ddG calculations on candidate PDB
| | 
| |____S2_PDB28
| | |____ # Results for RosettaMP, PoPMuSiC, and mCSM ddG calculations on candidate PDB
| | 
| |____S3_PDB55
| | |____ # Results for RosettaMP, PoPMuSiC, and mCSM ddG calculations on candidate PDB
| | 
| |____S4_PDB6
| | |____ # Results for RosettaMP, PoPMuSiC, and mCSM ddG calculations on candidate PDB
| | 
| |____S5_PDB52
| | |____ # Results for RosettaMP, PoPMuSiC, and mCSM ddG calculations on candidate PDB
| | 
| |____S6_PDB31
| | |____ # Results for RosettaMP, PoPMuSiC, and mCSM ddG calculations on candidate PDB
| | 
| |____csv_raw_results
| | |____ # Raw tabulated results for RosettaMP, PoPMuSiC, and mCSM ddG calculations
| | 
| |____rescale_plotting
|   |____ # Jupyter notebooks for heatmap generation
|   |____ csvs
|     |____ # contains all normalized and rescaled data from ddg/thermostability prediction software
|
|____transfer_and_zeroshot_learning
  |____ESM1v
  | |____ # contains all code and inputs for ESM1v zero shot prediction
  |
  |____ProSE
    |____ # contains all code and inputs for transfer learning using ProSE embeddings
```
