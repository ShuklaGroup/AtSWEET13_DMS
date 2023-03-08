Using ProSE Language model embeddings to predict variant effects

The MPTherm dataset is in the './data' folder, including: ProSE embeddings, labels (delta Tm), and fasta file (protein sequences)

The script 'train_ML_SVM.py' is used for hyperparamter optimization for SVM by cross validation

The script 'train_ML_RF.py' is used for hyperparamter optimization for RF by cross validation

The script 'train_ML_MLP.py' is used for hyperparamter optimization for MLP by cross validation

All the hyperparameter optimization results of thress models are saved in the './CV_results folder'

After optimizing hyperparameter, we predict the lables(delta Tm) for AtSW1_TM4 mutants by using 'predict_ML.py' and 'predict_sklearn_MLP_prose.py'

The predicted results are saved in './AtSW13_predict_results' folder
