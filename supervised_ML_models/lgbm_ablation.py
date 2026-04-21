import os,json,h5py,numpy as np,pandas as pd
from scipy.stats import spearmanr,pearsonr
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor

DATA_CSV="AtSWEET13_DMS_full_reconstructed.csv"
EMB_H5="kermut/data/embeddings/substitutions_singles/ESM2/SWEET13.h5"
ZS_CSV="kermut/data/zero_shot_fitness_predictions/ESM2/650M/SWEET13.csv"
MPNN_NPY="kermut/data/conditional_probs/ProteinMPNN/SWEET13.npy"
FOLD_DIR="kermut/data/cv_folds_singles_substitutions/SWEET13/fold_random_5"
OUT_DIR="lgbm_ablation_outputs"
os.makedirs(OUT_DIR,exist_ok=True)

OUTER_FOLDS=5
RANDOM_SEED=42

BASE_PARAMS={"n_estimators":600,"learning_rate":0.03,"num_leaves":63,"max_depth":-1,"min_child_samples":20,"subsample":0.8,"colsample_bytree":0.8,"reg_alpha":0.0,"reg_lambda":1.0}

ABLATIONS={"n_estimators":[100,200,400],"learning_rate":[0.005,0.01,0.02],"num_leaves":[31,63,127],"min_child_samples":[30,60,100]}

aa="ACDEFGHIKLMNPQRSTVWY"
aa_to_i={a:i for i,a in enumerate(aa)}

def get_emb_array(h5_path):
    with h5py.File(h5_path,"r") as f:
        keys=list(f.keys())
        if "embeddings" in keys:
            return np.array(f["embeddings"])
        if len(keys)==1:
            return np.array(f[keys[0]])
        raise ValueError(f"Could not determine embedding dataset key in {h5_path}. Keys: {keys}")

def detect_mutant_col(df):
    for c in ["mutant","mutation","mut_key","variant","mut"]:
        if c in df.columns:
            return c
    if df.shape[1]==1:
        return df.columns[0]
    for c in df.columns:
        vals=df[c].astype(str).head(20)
        if vals.str.match(r"^[A-Z]\d+[A-Z]$").any():
            return c
    return df.columns[0]

def load_fold_mutants(path):
    fdf=pd.read_csv(path)
    mcol=detect_mutant_col(fdf)
    muts=fdf[mcol].astype(str).str.strip()
    return muts.tolist(),fdf

def build_lgb(params):
    return LGBMRegressor(n_estimators=params["n_estimators"],learning_rate=params["learning_rate"],num_leaves=params["num_leaves"],max_depth=params["max_depth"],min_child_samples=params["min_child_samples"],subsample=params["subsample"],colsample_bytree=params["colsample_bytree"],reg_alpha=params["reg_alpha"],reg_lambda=params["reg_lambda"],objective="regression",n_jobs=-1,random_state=RANDOM_SEED,verbosity=-1)

df=pd.read_csv(DATA_CSV)
df["mutant"]=df["mutant"].astype(str).str.strip()
df["position"]=df["mutant"].str.extract(r"(\d+)").astype(int)
df["wt"]=df["mutant"].str[0]
df["mut"]=df["mutant"].str[-1]

emb=get_emb_array(EMB_H5)
zs=pd.read_csv(ZS_CSV)
mpnn=np.load(MPNN_NPY)

zs_col="esm2_t33_650M_UR50D"
df["zero_shot"]=zs[zs_col].values
df["mpnn_prob"]=df.apply(lambda r:float(mpnn[int(r.position)-1,aa_to_i[r.mut]]),axis=1)

try:
    enc_wt=OneHotEncoder(sparse_output=False,handle_unknown="ignore")
    enc_mut=OneHotEncoder(sparse_output=False,handle_unknown="ignore")
except TypeError:
    enc_wt=OneHotEncoder(sparse=False,handle_unknown="ignore")
    enc_mut=OneHotEncoder(sparse=False,handle_unknown="ignore")

wt_oh=enc_wt.fit_transform(df[["wt"]]).astype(np.float32)
mut_oh=enc_mut.fit_transform(df[["mut"]]).astype(np.float32)
num_feats=df[["position","zero_shot","mpnn_prob"]].values.astype(np.float32)
X=np.concatenate([emb.astype(np.float32),wt_oh,mut_oh,num_feats],axis=1)
y=df["exp_score"].values.astype(np.float32)

mut_to_idx={m:i for i,m in enumerate(df["mutant"].tolist())}

print("Feature matrix shape:",X.shape)
print("Zero-shot vs exp_score Spearman:",float(spearmanr(df["zero_shot"],y).correlation))
print("ProteinMPNN(mut aa prob) vs exp_score Spearman:",float(spearmanr(df["mpnn_prob"],y).correlation))
print("Base params:",BASE_PARAMS)

all_rows=[]
best_rows=[]

for param_name,param_values in ABLATIONS.items():
    print(f"\n=== ABLATION: {param_name} ===")
    rows=[]
    for value in param_values:
        params=BASE_PARAMS.copy()
        params[param_name]=value
        oof_pred=np.zeros(len(y))
        for fold in range(OUTER_FOLDS):
            train_path=os.path.join(FOLD_DIR,f"train_fold_{fold}.csv")
            test_path=os.path.join(FOLD_DIR,f"test_fold_{fold}.csv")
            train_muts,_=load_fold_mutants(train_path)
            test_muts,_=load_fold_mutants(test_path)
            train_idx=np.array([mut_to_idx[m] for m in train_muts])
            test_idx=np.array([mut_to_idx[m] for m in test_muts])
            X_tr,y_tr=X[train_idx],y[train_idx]
            X_te,y_te=X[test_idx],y[test_idx]
            model=build_lgb(params)
            model.fit(X_tr,y_tr)
            pred=model.predict(X_te)
            oof_pred[test_idx]=pred
        s=float(spearmanr(y,oof_pred).correlation)
        p=float(pearsonr(y,oof_pred)[0])
        row={"ablated_param":param_name,"ablated_value":value,"params_json":json.dumps(params,sort_keys=True),"oof_spearman":s,"oof_pearson":p}
        rows.append(row)
        all_rows.append(row)
        print(f"{param_name}={value} OOF Spearman={s:.4f} OOF Pearson={p:.4f}")
    res=pd.DataFrame(rows).sort_values(["oof_spearman","oof_pearson"],ascending=[False,False]).reset_index(drop=True)
    out_path=os.path.join(OUT_DIR,f"lgbm_ablation_{param_name}.csv")
    res.to_csv(out_path,index=False)
    best_rows.append({"ablated_param":param_name,"best_value":res.iloc[0]["ablated_value"],"best_oof_spearman":res.iloc[0]["oof_spearman"],"best_oof_pearson":res.iloc[0]["oof_pearson"],"best_params_json":res.iloc[0]["params_json"]})
    print(f"\nTop results for {param_name}:")
    print(res[["ablated_value","oof_spearman","oof_pearson"]].to_string(index=False))
    print("Saved:",out_path)

all_res=pd.DataFrame(all_rows).sort_values(["ablated_param","oof_spearman","oof_pearson"],ascending=[True,False,False]).reset_index(drop=True)
all_path=os.path.join(OUT_DIR,"lgbm_ablation_all_results.csv")
all_res.to_csv(all_path,index=False)

best_summary=pd.DataFrame(best_rows).sort_values("best_oof_spearman",ascending=False).reset_index(drop=True)
best_path=os.path.join(OUT_DIR,"lgbm_ablation_best_summary.csv")
best_summary.to_csv(best_path,index=False)

print("\n=== BEST PER ABLATION ===")
print(best_summary.to_string(index=False))
print("\nSaved:",all_path)
print("Saved:",best_path)