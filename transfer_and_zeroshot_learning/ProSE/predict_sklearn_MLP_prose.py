import numpy as np
import torch
import os
import glob
import pickle
from sklearn.neural_network import MLPRegressor

y = np.load('MPTherm_data_labels.npy')
x = np.load('MPTherm_avg_prose_embedding.npy')
test_x = np.load('AtSW13_TM4_avg_prose_embedding.npy')
hidden_layer = (512, 256, 128)
batch_size_ = 64
lr_ = 0.01
#df = pd.DataFrame(columns = ['Hidden_layer','Batch_size','Learning rate', 'R2', 'RMSE', 'Spearman','Pearson'])

regr = MLPRegressor(hidden_layer_sizes = hidden_layer, batch_size = batch_size_, learning_rate_init = lr_, early_stopping = True)
regr.fit(x, y)
y_pred = regr.predict(test_x)
np.save('AtSW13_predict_results/AtSW13_TM4_muts_prose_prediction_MLP_result.npy',y_pred)
print(y_pred)
print(len(y_pred))
























