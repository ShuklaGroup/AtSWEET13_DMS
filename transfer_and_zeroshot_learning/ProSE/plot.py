import numpy as np
import pandas as pd
import os
import math
import glob
from scipy import stats
import random
import pickle
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor

def load_data():
    x = np.load('MPTherm_avg_prose_embedding.npy')
    y = np.load('MPTherm_data_labels.npy')
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=18, shuffle=True)
    pca = PCA(n_components = 35)   ##optimization?  95%
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test, y_train, y_test

def load_data_MLP():
    x = np.load('MPTherm_avg_prose_embedding.npy')
    y = np.load('MPTherm_data_labels.npy')
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=18, shuffle=True)
    return X_train, X_test, y_train, y_test


def plot_Therm(model, X_train, X_test, y_train, y_test, dotcolor, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #svm_model = svm.SVR(C = 10, kernel='rbf')   #SVM for regression
    #svm_model.fit(X_train, y_train) 
    #y_pred_SVM = clf.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    RMSE = math.sqrt(mean_squared_error(y_test, y_pred)) 
    spearman_corr = stats.spearmanr(y_test, y_pred)
    pearson_corr = stats.pearsonr(y_test, y_pred)

    fig,axs = plt.subplots(1,1,figsize=(9,6),constrained_layout=True)
    plt.plot(y_test, y_pred,'o',color=dotcolor)
    plt.plot([-40,20],[-40,20], '--',color = 'black')
    axs.set_xlim(-40,20)
    axs.set_xticks(range(int(-40),int(20)+1,10))
    axs.set_xticklabels(range(int(-40),int(20)+1,10))
    axs.set_ylim(-40,20)
    axs.set_yticks(range(int(-40),int(20)+1,10))
    axs.set_yticklabels(range(int(-40),int(20)+1,10))
    
    for axis in ['top','bottom','left','right']:
        axs.spines[axis].set_linewidth(2)
    axs.tick_params(width=2)
    plt.xticks(fontsize=16, fontweight="bold")
    plt.yticks(fontsize=16, fontweight="bold")
    plt.xlabel('MPTherm experimental 'r'$\Delta$Tm (${^\circ}C$)',fontsize=20, fontweight="bold")
    plt.ylabel('ProSE predicted 'r'$\Delta$Tm (${^\circ}C$)',fontsize=20, fontweight = 'bold')

    #plt.scatter(y_test, y_pred)
    plt.savefig('MPTherm_'+name + '_plot.png', dpi = 500)
    plt.close()
    return r2, RMSE, spearman_corr, pearson_corr

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    model = svm.SVR(C = 10, kernel='rbf') 
    r2, RMSE, spearman, pearson = plot_Therm(model, X_train, X_test, y_train, y_test,'blue', 'SVM')
    metrics = {'R2':r2, 'RMSE': RMSE, 'spearman_correlation': spearman, 'pearson_correlation': pearson}
    print(metrics)

    RFmodel = RandomForestRegressor(n_estimators = 100, max_depth=50, max_features='log2' ,random_state=0)
    r2_RF, RMSE_RF, spearman_RF, pearson_RF = plot_Therm(RFmodel, X_train, X_test, y_train, y_test,'gold', 'RF')
    metrics = {'R2': r2_RF, 'RMSE': RMSE_RF, 'spearman_correlation': spearman_RF, 'pearson_correlation': pearson_RF}

    print(metrics)    

    #### for MLP####
    X_train, X_test, y_train, y_test = load_data_MLP()
    regr = MLPRegressor(hidden_layer_sizes = (512, 256, 128), batch_size = 64, learning_rate_init = 0.01, early_stopping = True)
    r2, RMSE, spearman, pearson = plot_Therm(regr, X_train, X_test, y_train, y_test,'pink', 'MLP')
    metrics = {'R2': r2, 'RMSE': RMSE, 'spearman_correlation': spearman, 'pearson_correlation': pearson}
    print(metrics)






















