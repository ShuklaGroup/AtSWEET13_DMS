import numpy as np
import torch
import pandas as pd
import os
import math
import glob
from scipy import stats
import random
import pickle
from itertools import product
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, train_test_split


y = np.load('data/MPTherm_data_labels.npy')
x = np.load('data/MPTherm_avg_prose_embedding.npy')
hyper_hidden_layer = [(512), (512, 256), (512, 256, 128), (512, 256, 128, 64)]
hyper_batch_size = [32, 64]
hyper_lr = [0.01, 0.001]
hyper_comb = list(product(hyper_hidden_layer, hyper_batch_size, hyper_lr))
df = pd.DataFrame(columns = ['Hidden_layer','Batch_size','Learning rate', 'R2', 'RMSE', 'Spearman','Pearson'])

for lr_ in hyper_lr:
    for batch_size_ in hyper_batch_size:
        for j, hidden_layer in enumerate(hyper_hidden_layer):
            kfold = KFold(n_splits=5, shuffle=True, random_state = 22)
            r2_list, RMSE_list, spearman_list, pearson_list = [], [], [], [] 
            for i, (train_idx, test_idx) in enumerate(kfold.split(y)): 
                print(f"Fold {i}")
                X_train = x[train_idx] 
                y_train = y[train_idx] 
                X_test = x[test_idx] 
                y_test = y[test_idx] 
  
                regr = MLPRegressor(hidden_layer_sizes = hidden_layer, batch_size = batch_size_, learning_rate_init = lr_, early_stopping = True)   
                regr.fit(X_train, y_train) 
                y_pred = regr.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                RMSE = math.sqrt(mean_squared_error(y_test, y_pred)) 
                spearman_corr = stats.spearmanr(y_test, y_pred)
                pearson_corr = stats.pearsonr(y_test, y_pred)
                
            r2_list.append(r2) 
            RMSE_list.append(RMSE) 
            spearman_list.append(spearman_corr) 
            pearson_list.append(pearson_corr)

            metrics = {'Hidden_layer': hidden_layer, 'Batch_size': batch_size_, 'Learning_rate': lr_,'r2': np.mean(r2_list), 'RMSE': np.mean(RMSE_list), 'spearman_correlation': np.mean(spearman_list), 'pearson_correlation': np.mean(pearson_list)}
            print(metrics)
            
            with open('CV_results/MLP/MPTherm_prose_hidden_layer_' + str(j+1) +'_batch_size_' + str(batch_size_) + '_lr_' + str(lr_) +'.pkl', 'wb') as f:
                    pickle.dump(metrics, f)





















