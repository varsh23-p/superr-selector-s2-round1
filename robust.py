import pandas as pd, numpy as np, pickle, warnings; warnings.filterwarnings('ignore')
src=open('rate5.py').read().split("out={}")[0]
exec(src)
res={}
base={}
for role in ('bat','bowl'):
    t=pm(role); p=rate(t); base[role]=(t,p)
rng=np.random.default_rng(7)
boot={}
for role in ('bat','bowl'):
    t,p=base[role]; ms=t.match_id.unique(); cnt={}
    top10=pd.Series(0,index=p.index,dtype=float); rk=[]
    for i in range(300):
        w=pd.Series(rng.choice(ms,len(ms))).value_counts().to_dict()
        q=rate(t,w,fix=p.attrs['fix'],minhp=40,minm=10,minall=120)
        r=q.R.rank(ascending=False)
        top10[top10.index.intersection(q.index)]+= (r[top10.index.intersection(q.index)]<=10).astype(float)
        rk.append(r.reindex(p.index))
    rk=pd.concat(rk,axis=1)
    boot[role]=pd.DataFrame({'top10_pct':top10/300*100,'med_rank':rk.median(axis=1),'p10':rk.quantile(.1,axis=1),'p90':rk.quantile(.9,axis=1)})
# sensitivity
sens={}
for HPT in (1.25,1.5,2.0):
    b['hp']=b.PI>=HPT
    for role in ('bat','bowl'):
        t=pm(role); q=rate(t,minhp=40 if HPT>1.5 else 50); sens[(role,HPT)]=q.R
b['hp']=b.PI>=1.5
for role in ('bat','bowl'):
    t=pm(role)
    for wts in ((1.0,0.0),(0.6,0.4)):
        sens[(role,wts)]=rate(t,wts=wts).R
pickle.dump((boot,sens),open('robust.pkl','wb'))
for role in ('bat','bowl'):
    p=base[role][1]; top=p.R.sort_values(ascending=False).index[:10]
    print(role,'bootstrap'); print(boot[role].loc[top].round(1).T.to_string())
    for k,v in sens.items():
        if k[0]==role:
            t10=set(v.sort_values(ascending=False).index[:10]); print('  ',k[1],'overlap with base top10:',len(t10&set(top)),'/10')
