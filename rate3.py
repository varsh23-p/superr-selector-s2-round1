import pandas as pd, numpy as np, warnings, pickle; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
b=pd.read_pickle('b5.pkl')
b['v_bat']=np.where(b.wide,np.nan,(b.runs-b.e_runs)-b.W*(b.bat_dis-b.e_dis))
b['v_bowl']=(b.e_tot-b.tot)+b.W*(b.bowl_wk-b.e_bwk)
K=400
a=b.groupby('striker_id').v_bat.sum()/(b.groupby('striker_id').v_bat.count()+K)
c=b.groupby('bowler_id').v_bowl.sum()/(b.groupby('bowler_id').v_bowl.count()+K)
b['vb']=b.v_bat+b.bowler_id.map(c).fillna(0); b['vw']=b.v_bowl+b.striker_id.map(a).fillna(0)
b['vb']-=b.vb.mean(); b['vw']-=b.vw.mean()
HPT=1.5
b['hp']=b.PI>=HPT
print('HP share',b.hp.mean().round(3),'balls',b.hp.sum(),'IPL share of HP',b.ipl[b.hp].mean().round(3))
print('sd per ball bat',b.vb.std().round(2),'bowl',b.vw.std().round(2))
b.to_pickle('b6.pkl')
