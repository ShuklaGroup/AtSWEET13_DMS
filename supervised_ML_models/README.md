# AtSWEET13 DMS Modeling with Supervised Regressors
Compilation of ML work done in 2026 for the '<i>Engineering Arabidopsis SWEET13 transporter with
enhanced and selective sucrose uptake</i>' manuscript

## Preparing DMS Data for Modeling
Kermut, and its dependencies (packages and models) were installed by cloning the [Official Repository](https://github.com/petergroth/kermut.git) for their NeurIPS 2024 Paper ([Kermut: Composite kernel regression for protein variant effects](https://www.proceedings.com/content/079/079017-0929open.pdf), and by following its installation guidelines. Its <b>'Reproducibility</b>, <b>'Installation</b>', and <b>'Data access'</b> sections will need to be followed to set up Kermut. Our AtSWEET13 DMS data can be encoded using the following command - 
```py 
python -m kermut.cmdline.preprocess_data.extract_esm2_embeddings \
    dataset=single \
    dataset.single.use_id=true \
    dataset.single.id=SWEET13  # or SWEET13_TM4 if only modeling TM4 transmembrane helix
```
### Working Local Environment used to Train and Evaluate Kermut and Tree-based Models  
```
conda env create -f environment.yml
conda activate sweetml
```

### Installation of ProteinMPNN
ProteinMPNN was installed by cloning their [Official Repository](https://github.com/dauparas/ProteinMPNN.git) ([Dauparas, J., et al. (2022) Science](https://www.science.org/doi/10.1126/science.add2187)) and following its instructions to set up the environment required to use the model for embedding.

### Make Data Cross-Validation Folds
The mutants can be divided into folds for more rigorous evaluation and minimal data leakage by running the below script (replace paths with local paths to DMS data and output folders) -
```
cd kermut
```
```py
python make_folds.py \
--base data/cv_folds_singles_substitutions/SWEET13/fold_random_5 \
--data data/sweet/SWEET13_full.csv
```
Our DMS CSV file, wild-type sequence file, and PDB crystal structure file are provided as shown below-
```text
./AtSWEET13_DMS_full_reconstructed.csv
./kermut
    └── data
        ├── AtSWEET13_TM4.csv
        ├── conditional_probs
        │   └── ProteinMPNN
        │       ├── SWEET13.npy
        │       └── SWEET13_TM4.npy
        ├── cv_folds_singles_substitutions
        │   └── SWEET13
        │       └── fold_random_5
        │           ├── test_fold_0.csv
        │           ├── test_fold_1.csv
        │           ├── test_fold_2.csv
        │           ├── test_fold_3.csv
        │           ├── test_fold_4.csv
        │           ├── train_fold_0.csv
        │           ├── train_fold_1.csv
        │           ├── train_fold_2.csv
        │           ├── train_fold_3.csv
        │           └── train_fold_4.csv
        ├── structures
        │   ├── coords
        │   │   ├── SWEET13.npy
        │   │   └── SWEET13_TM4.npy
        │   └── pdbs
        │       └── Q9FGQ2.pdb
        └── sweet
            ├── data.csv
            ├── sequence.fasta
            ├── SWEET13_full.csv
            └── SWEET13.pdb
```
## Run Kermut on AtSWEET13 Data
```py
python proteingym_benchmark.py --multirun \
dataset=single \
single.use_id=true \
single.id=SWEET13 \
cv_scheme=fold_random_5,fold_modulo_5,fold_contiguous_5 \
kernel=kermut
```
```py
python proteingym_benchmark.py \
dataset=single \
single.use_id=true \
single.id=SWEET13_TM4 \
cv_scheme=fold_random_5 \
kernel=kermut
```
### Evaluate Performance of Kermut
```py
python -m kermut.cmdline.process_results.process_model_scores \
dataset=single \
single.use_id=true \
single.id=SWEET13  # or SWEET13_TM4
```
## Train and evaluate LightGBM, XGBoost, and CatBoost models with AtSWEEt13 Data

```py
python xgb_ablation.py
```
```py
python lgb_ablation.py
```
```py
python cat_ablation.py
```
