import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as HR, HistGradientBoostingClassifier as HC
from sklearn.model_selection import GroupKFold
b=pd.read_pickle('b4.pkl')
bl=(120-b.lb_before).clip(lower=0)
need=b.target-b.runs_before
b['rrr']=np.where(b.innings==2,(need*6/bl.clip(lower=1)).clip(-2,36),-1.0)
b['crr']=np.where(b.lb_before>0,b.runs_before*6/b.lb_before.clip(lower=1),7.5)
X=pd.DataFrame({'inn':b.innings,'over':b.overid,'wk':b.wk_before,'bl':bl,'runs':b.runs_before,'rrr':b.rrr,'crr':b.crr,
 'wp':b.wp_before,'PI':b.PI,'ipl':b.ipl,'pace':(b.pace_or_spin=='Pace').astype(int),'lhb':(b.batting_type=='LHB').astype(int),'ko':b.knock_out})
b['bat_dis']=(b.wkt&(b.dismissed_player_id==b.striker_id)).astype(int)
b['bowl_wk']=(b.wkt&~b.dismissal_type.isin(['RunOut','Retire Out','Obstracting the field','Obstructing the field'])).astype(int)
targets={'e_runs':('r',b.runs.astype(float)),'e_dis':('c',b.bat_dis),'e_tot':('r',b.tot.astype(float)),'e_bwk':('c',b.bowl_wk)}
for k in targets: b[k]=np.nan
for tr,te in GroupKFold(5).split(b,groups=b.match_id):
    for k,(typ,y) in targets.items():
        mdl=(HR if typ=='r' else HC)(max_depth=5,learning_rate=0.06,max_iter=200,min_samples_leaf=200,random_state=0)
        mdl.fit(X.iloc[tr],y.iloc[tr])
        b.loc[b.index[te],k]=mdl.predict(X.iloc[te]) if typ=='r' else mdl.predict_proba(X.iloc[te])[:,1]
for k,(typ,y) in targets.items():
    print(k,'mean actual',y.mean().round(4),'mean pred',b[k].mean().round(4))
# wicket run-equivalent by innings x phase
b['phase']=pd.cut(b.overid,[0,6,15,20],labels=['PP','Mid','Death'])
nw=b[~b.wkt&b.legal]
W={}
for (inn,ph),d in nw.groupby(['innings','phase']):
    slope=(d.runs*d.run_part).sum()/(d.runs**2).sum()
    wk=b[(b.innings==inn)&(b.phase==ph)&(b.bat_dis==1)]
    W[(inn,ph)]=(-wk.wkt_part.mean())/slope
    print(inn,ph,'WP/run',round(slope,4),'WP/wkt',round(-wk.wkt_part.mean(),4),'wkt=runs',round(W[(inn,ph)],1))
b['W']=[W[(i,p)] for i,p in zip(b.innings,b.phase)]
b.to_pickle('b5.pkl')
