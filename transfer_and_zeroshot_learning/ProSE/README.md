# Using ProSE Language model embeddings to predict variant effects

## Data
The MPTherm dataset and AtSW13_TM4_muts dataset is in the './data' folder, including: ProSE embeddings, labels (delta Tm), and fasta file (protein sequences)

## Hyperparameter optimization
The script 'train_ML_SVM.py' is used for hyperparamter optimization for SVM by cross validation

The script 'train_ML_RF.py' is used for hyperparamter optimization for RF by cross validation

The script 'train_sklearn_MLP_prose.py' is used for hyperparamter optimization for MLP by cross validation

All the hyperparameter optimization results of thress models are saved in the './CV_results folder'

## Prediction
After optimizing hyperparameter, we predict the lables(delta Tm) for AtSW1_TM4 mutants by using 'predict_ML.py' and 'predict_sklearn_MLP_prose.py'

The predicted results are saved in './AtSW13_predict_results' folder

## Plot
The script 'plot.py' is for plotting the performace of differents models with optimized hyperparameters
