import numpy as np
import pandas as pd
import os
import math
import glob
from scipy import stats
import time
import random
import pickle
import sys
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split



x = np.load('data/MPTherm_avg_prose_embedding.npy')
y = np.load('data/MPTherm_data_labels.npy')
hyper_para_C = [0.1, 1.0, 10]
hyper_para_kernel = ['linear', 'poly', 'rbf', 'sigmoid']
for C_ in hyper_para_C:
    for kernel_ in hyper_para_kernel:
        kfold = KFold(n_splits=10, shuffle=True, random_state = 22)
        r2_list, RMSE_list, spearman_list, pearson_list = [], [], [], [] 
        for train_idx, test_idx in kfold.split(y): 
            X_train = x[train_idx] 
            y_train = y[train_idx] 
            X_test = x[test_idx] 
            y_test = y[test_idx] 
  
            pca = PCA(n_components = 200)   

            X_train = pca.fit_transform(X_train) 
            X_test = pca.transform(X_test) 
            print(sum(pca.explained_variance_ratio_)) 
            clf = svm.SVR(C = C_, kernel=kernel_)   #SVM for regression
            clf.fit(X_train, y_train) 
            y_pred = clf.predict(X_test)
 
            r2 = r2_score(y_test, y_pred)
            RMSE = math.sqrt(mean_squared_error(y_test, y_pred)) 
            spearman_corr = stats.spearmanr(y_test, y_pred)
            pearson_corr = stats.pearsonr(y_test, y_pred)
                
            r2_list.append(r2) 
            RMSE_list.append(RMSE) 
            spearman_list.append(spearman_corr[0]) 
            pearson_list.append(pearson_corr[0])
                
         #print(spearman_list)
         metrics = {'Kernel': kernel_, 'C': C_,'r2': np.mean(r2_list), 'RMSE': np.mean(RMSE_list), 'spearman_correlation': np.mean(spearman_list), 'pearson_correlation': np.mean(pearson_list)}
         print(metrics)
         with open('CV_results/SVM/MPTherm_prose_SVM_C' + str(C_) +'_kernel_' + str(kernel_) +'.pkl', 'wb') as f:
              pickle.dump(metrics, f)




















