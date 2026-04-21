import pandas as pd
import numpy as np
df=pd.read_csv("Data/SWEET13_full.csv")
np.random.seed(2024)
idx=np.random.permutation(len(df))
folds=np.array_split(idx,5)
base="Data/cv_folds"
import os
os.makedirs(base,exist_ok=True)
for i in range(5):
    test_idx=folds[i]
    train_idx=np.concatenate([folds[j] for j in range(5) if j!=i])
    df.iloc[train_idx].to_csv(f"{base}/train_fold_{i}.csv",index=False)
    df.iloc[test_idx].to_csv(f"{base}/test_fold_{i}.csv",index=False)