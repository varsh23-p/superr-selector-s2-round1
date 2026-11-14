import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold
b=pd.read_pickle('b.pkl'); mm=pd.read_pickle('mm.pkl')
et=b.extras_type.fillna('none')
b['tot']=b.runs+b.wide*1+b.nb*1+b.bye*1+b.extras_2_runs+(et=='other')*1
g=b.groupby(['match_id','innings'])
b['runs_before']=g.tot.cumsum()-b.tot
t=g.tot.sum().unstack(); t.columns=['t1','t2']
# eligible matches: decided, no DLS, both innings
ok=mm[(mm.win_team_id.notna())&mm.first_inn_revised_overs.isna()&mm.second_inn_revised_overs.isna()&mm.revised_target.isna()].drop(columns=['t1','t2'])
ok=ok.merge(t,left_on='match_id',right_index=True)
ok['ipl']=ok.comp_name.str.startswith('IPL').astype(int)
print('eligible matches',len(ok), 'ipl',ok.ipl.sum())
b=b.merge(ok[['match_id','win_team_id','t1','knock_out','stage','ipl','match_date','comp_name']],on='match_id')
b['target']=b.t1+1
b['bwin']=(b.batting_team_id==b.win_team_id).astype(int)
b['lb_before']=b.groupby(['match_id','innings']).legal.cumsum()-b.legal
b['wk_before']=b.groupby(['match_id','innings']).wkt.cumsum()-b.wkt
# drop incomplete innings (very few legal balls & not all out / chase complete)
b=b.reset_index(drop=True)
def feats(inn,runs,wk,lb,target,ipl):
    bl=120-lb
    if inn==1:
        return np.column_stack([runs,wk,bl,ipl])
    need=target-runs
    rrr=np.where(bl>0,need*6/np.maximum(bl,1),99).clip(-5,40)
    return np.column_stack([need,wk,bl,rrr,ipl,target])
mono={1:[1,-1,0,0],2:[-1,-1,1,0,0,0]}
def wp_pred(models,inn,runs,wk,lb,target,ipl):
    p=models[inn].predict_proba(feats(inn,runs,wk,lb,target,ipl))[:,1]
    if inn==2:
        need=target-runs; bl=120-lb
        p=np.where(need<=0,1.0,p)
        end=(wk>=10)|(bl<=0)
        p=np.where(end&(need>0),np.where(need==1,0.5,0.0),p)
    return p
# training rows: after-state of every ball
b['runs_after']=b.runs_before+b.tot; b['wk_after']=b.wk_before+b.wkt; b['lb_after']=b.lb_before+b.legal
outs=[(0,0),(1,0),(2,0),(3,0),(4,0),(6,0),(0,1)]
# outcome probabilities by innings & over (legal balls)
L=b[b.legal]
def outcome_probs(L):
    keys=[]
    P={}
    for (inn,ov),d in L.groupby(['innings','overid']):
        pr=[]
        wk=d.wkt.mean()
        nw=d[~d.wkt]
        for r,_ in outs[:-1]:
            pr.append((nw.runs.clip(upper=6).replace({5:4})==r).sum()/len(d))
        # fold 3s/5 etc.
        pr.append(wk); pr=np.array(pr); pr=pr/pr.sum(); P[(inn,ov)]=pr
    return P
P=outcome_probs(L)
gkf=GroupKFold(5)
b['wp_before']=np.nan; b['wp_after']=np.nan; b['wp_after_nowkt']=np.nan; b['li_raw']=np.nan
for fold,(tr,te) in enumerate(gkf.split(b,groups=b.match_id)):
    d=b.iloc[tr]; models={}
    for inn in (1,2):
        di=d[d.innings==inn]
        X=feats(inn,di.runs_after.values,di.wk_after.values,di.lb_after.values,di.target.values,di.ipl.values)
        # add initial state rows
        init=di.groupby('match_id').head(1)
        Xi=feats(inn,init.runs_before.values,init.wk_before.values,init.lb_before.values,init.target.values,init.ipl.values)
        X=np.vstack([X,Xi]); y=np.concatenate([di.bwin.values,init.bwin.values])
        models[inn]=HGB(max_depth=4,learning_rate=0.06,max_iter=250,min_samples_leaf=300,monotonic_cst=mono[inn],random_state=0).fit(X,y)
    for inn in (1,2):
        idx=te[b.innings.values[te]==inn]; x=b.iloc[idx]
        args=lambda r,w,l:(inn,r,w,l,x.target.values,x.ipl.values)
        wb=wp_pred(models,*args(x.runs_before.values,x.wk_before.values,x.lb_before.values))
        wa=wp_pred(models,*args(x.runs_after.values,x.wk_after.values,x.lb_after.values))
        wn=wp_pred(models,*args(x.runs_after.values,x.wk_before.values,x.lb_after.values))
        # leverage
        pr=np.array([P[(inn,o)] for o in x.overid.values])
        li=np.zeros(len(x))
        for k,(r,w) in enumerate(outs):
            wk_=x.wk_before.values+w
            wpk=wp_pred(models,inn,x.runs_before.values+r,wk_,x.lb_before.values+1,x.target.values,x.ipl.values)
            li+=pr[:,k]*np.abs(wpk-wb)
        b.loc[x.index,['wp_before','wp_after','wp_after_nowkt','li_raw']]=np.column_stack([wb,wa,wn,li])
    print('fold',fold,'done')
# calibration check
from sklearn.metrics import brier_score_loss, roc_auc_score
for inn in (1,2):
    d=b[b.innings==inn]; print(inn,'brier',brier_score_loss(d.bwin,d.wp_before),'auc',roc_auc_score(d.bwin,d.wp_before))
b['LI']=b.li_raw/b[b.legal].li_raw.mean()
b['PI']=b.LI*(1+0.25*b.knock_out)
b.to_pickle('b2.pkl'); ok.to_pickle('ok.pkl')
print(b.PI.describe(percentiles=[.5,.75,.8,.9,.95,.99]))
print(b.groupby(['innings',pd.cut(b.overid,[0,6,10,15,20])]).PI.mean())
