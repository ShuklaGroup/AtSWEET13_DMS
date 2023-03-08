import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

exp = pd.read_csv('AtSWEET13_DMS_exp_ESM1v_score.csv')
#print(exp)
cons = pd.read_csv('AtSWEET13_DMS_cons_ESM1v_score.csv')

def plot(x, y, type_, name, xlabel_):
    spearman_corr = stats.spearmanr(x, y)
    pearson_corr = stats.pearsonr(x, y)

    fig,axs = plt.subplots(1,1,figsize=(9,6),constrained_layout=True)
    plt.scatter(x, y, s = 10, marker = 'o', color='black')
    
    axs.plot([-25,5],[-25,5], '--',color = 'grey')
    axs.set_xlim(-25,5)
    axs.set_xticks(range(int(-25),int(5)+1,5))
    axs.set_xticklabels(range(int(-25),int(5)+1,5))
    axs.set_ylim(-25,5)
    axs.set_yticks(range(int(-25),int(5)+1,5))
    axs.set_yticklabels(range(int(-25),int(5)+1,5))
    
    for axis in ['top','bottom','left','right']:
        axs.spines[axis].set_linewidth(2)
    axs.tick_params(width=2)
    plt.xticks(fontsize=16, fontweight="bold")
    plt.yticks(fontsize=16, fontweight="bold")
    plt.xlabel(xlabel_,fontsize=20, fontweight="bold")
    plt.ylabel('ESM1v',fontsize=20, fontweight = 'bold')

    #plt.scatter(y_test, y_pred)
    plt.savefig('ESM1v_' + type_ + str(name + 1) + '_plot.png', dpi = 500)
    plt.close()
    return spearman_corr, pearson_corr

if __name__ == "__main__":
    res_exp = []
    res_cons = []
    for i in range(5):
        spearman, pearson = plot(exp.iloc[:,7], exp.iloc[:,(i+2)], 'exp',i, "Consv. Score for Expression")
        res_exp.append([spearman[0], pearson[0]])
    res_exp = pd.DataFrame(np.array(res_exp))
    res_exp.columns = ['Spearman','Pearson']
    print(res_exp)
    res_exp.to_csv('AtSWEET13_DMS_exp_ESM1v_corr.csv')

    for j in range(5):
        spearman, pearson = plot(cons.iloc[:,7], cons.iloc[:,(j+2)], 'cons',j, "Consv. Score Positive Selection for Transport")
        res_cons.append([spearman[0], pearson[0]])
    res_cons = pd.DataFrame(np.array(res_cons))
    res_cons.columns = ['Spearman','Pearson']
    print(res_cons)
    res_cons.to_csv('AtSWEET13_DMS_cons_ESM1v_corr.csv')

    

