import numpy as np
import pandas as pd

exp = np.load('AtSWEET13_dms_exp_results_float_just_muts.npy')
df_exp = pd.DataFrame(exp)

esm1v_res = pd.read_csv('AtSWEET13_DMS_muts_labeled.csv')
exp_esm1v_res = pd.concat([esm1v_res, df_exp], axis = 1)
exp_esm1v_res = exp_esm1v_res.iloc[:,2:]
print(exp_esm1v_res)

exp_esm1v_res.to_csv('AtSWEET13_DMS_exp_ESM1v_score.csv')

cons = np.load('AtSWEET13_dms_cons_results_float_just_muts.npy')
df_cons = pd.DataFrame(cons)
cons_esm1v_res = pd.concat([esm1v_res, df_cons], axis = 1)
cons_esm1v_res = cons_esm1v_res.iloc[:,2:]
print(cons_esm1v_res)

cons_esm1v_res.to_csv('AtSWEET13_DMS_cons_ESM1v_score.csv')



