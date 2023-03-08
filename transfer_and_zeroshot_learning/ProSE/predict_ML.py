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
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


x = np.load('MPTherm_avg_prose_embedding.npy')
y = np.load('MPTherm_data_labels.npy')
test_x = np.load('AtSW13_TM4_avg_prose_embedding.npy')
C_ = 10
kernel_ ='rbf'

pca = PCA(n_components = 35)  
x = pca.fit_transform(x) 
test_x = pca.transform(test_x) 
print(sum(pca.explained_variance_ratio_)) 

####SVM
svm_model = svm.SVR(C = C_, kernel=kernel_)  
svm_model.fit(x, y) 
y_pred_SVM = svm_model.predict(test_x)
print(y_pred_SVM)
print(len(y_pred_SVM))
np.save('AtSW13_predict_results/AtSW13_TM4_muts_prose_prediction_SERT_SVM_result.npy', y_pred_SVM)

####RF
n_estimators_ = 100
max_depth_ = 50
max_features_ = 'log2'


RF_model = RandomForestRegressor(n_estimators = n_estimators_, max_depth=max_depth_, max_features=max_features_ ,random_state=0) 
RF_model.fit(x, y)                       
y_pred_RF = RF_model.predict(test_x)
print(y_pred_RF)
print(len(y_pred_RF))
np.save('AtSW13_predict_results/AtSW13_TM4_muts_prose_prediction_SERT_RF_result.npy', y_pred_RF)























