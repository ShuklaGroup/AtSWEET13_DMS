import numpy as np
import pandas as pd
import os
import math
import glob
from scipy import stats
import time
import random
import pickle
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split

x = np.load('data/MPTherm_avg_prose_embedding.npy')
y = np.load('data/MPTherm_data_labels.npy')

hyper_para_max_depth = [50, 100]
hyper_para_max_features = ['sqrt', 'log2']
hyper_para_n_estimators = [10, 20, 50, 100]
for max_depth_ in hyper_para_max_depth:
    for max_features_ in hyper_para_max_features:
        for n_estimators_ in hyper_para_n_estimators:
            kfold = KFold(n_splits=10, shuffle=True, random_state = 22)
            r2_list, RMSE_list, spearman_list, pearson_list = [], [], [], [] 
            for train_idx, test_idx in kfold.split(y): 
                X_train = x[train_idx] 
                y_train = y[train_idx] 
                X_test = x[test_idx] 
                y_test = y[test_idx] 
 
                pca = PCA(n_components = 35) 

                X_train = pca.fit_transform(X_train) 
                X_test = pca.transform(X_test) 
                print(sum(pca.explained_variance_ratio_)) 
                clf = RandomForestRegressor(n_estimators = n_estimators_, max_depth=max_depth_, max_features=max_features_ ,random_state=0) 
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
                metrics = {'n_estimators' : n_estimators_, 'Max_depth': max_depth_ , 'Max_features': max_features_ ,'r2': np.mean(r2_list), 'RMSE': np.mean(RMSE_list), 'spearman_correlation': np.mean(spearman_list), 'pearson_correlation': np.mean(pearson_list)}
            print(metrics)
            with open('CV_results/RF/MPTherm_prose_RF_depth' + str(max_depth_) +'_features_' + str(max_features_) + '_estimators_' + str(n_estimators_) +'.pkl', 'wb') as f:
                pickle.dump(metrics, f)





















