import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

exp = np.array(pd.read_csv('AtSWEET13_DMS_exp_ESM1v_score.csv'))
cons = np.array(pd.read_csv('AtSWEET13_DMS_cons_ESM1v_score.csv'))

def plot(x, y, xlim, ylim, type_, name, xlabel_):
    spearman_corr = stats.spearmanr(x, y)
    pearson_corr = stats.pearsonr(x, y)
    fig,axs = plt.subplots(1,1,figsize=(9,6),constrained_layout=True)
    Reg_model = LinearRegression()
    Reg_model.fit(x.reshape(-1, 1), y)
    r2 = Reg_model.score(x.reshape(-1,1), y)
    plt.scatter(x, y, s = 10, marker = 'o', color='black')
    print(np.max(x))
    print(np.min(x)) 
    #axs.plot([-25,5],[-25,5], '--',color = 'grey')
    m, b = Reg_model.coef_, Reg_model.intercept_
    plt.plot(x, m*x+b, '--', color='grey')
    axs.set_xlim(xlim[0],xlim[1])
    axs.set_xticks(range(xlim[0],xlim[1]+1,2))
    axs.set_xticklabels(range(xlim[0],xlim[1]+1,2))
    axs.set_ylim(ylim[0],ylim[1])
    axs.set_yticks(range(ylim[0],ylim[1]+1,5))
    axs.set_yticklabels(range(ylim[0],ylim[1]+1,5))
    
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
    return r2, spearman_corr, pearson_corr

if __name__ == "__main__":
    res_exp = []
    res_cons = []
    for i in range(5):
        r2, spearman, pearson = plot(exp[:,7], exp[:,(i+2)],[-10,4],[-25,5], 'exp',i, "Log2 Enrichment Ratio for Expression")
        res_exp.append([r2, spearman[0], pearson[0]])
    res_exp = pd.DataFrame(np.array(res_exp))
    res_exp.columns = ['R2','Spearman','Pearson']
    print(res_exp)
    res_exp.to_csv('AtSWEET13_DMS_exp_ESM1v_corr.csv')

    for j in range(5):
        r2, spearman, pearson = plot(cons[:,7], cons[:,(j+2)],[-6,4],[-25,5] ,'cons',j, "Log2 Enrichment Ratio for Transport")
        res_cons.append([r2, spearman[0], pearson[0]])
    res_cons = pd.DataFrame(np.array(res_cons))
    res_cons.columns = ['R2','Spearman','Pearson']
    print(res_cons)
    res_cons.to_csv('AtSWEET13_DMS_cons_ESM1v_corr.csv')

    

