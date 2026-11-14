import pandas as pd, numpy as np, warnings, pickle; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
b=pd.read_pickle('b6.pkl')
K=400
b['rb']=np.where(b.wide,np.nan,b.runs-b.e_runs); b['wb']=np.where(b.wide,np.nan,-b.W*(b.bat_dis-b.e_dis))
b['rw']=b.e_tot-b.tot; b['ww']=b.W*(b.bowl_wk-b.e_bwk)
def eff(col,pid):
    g=b.groupby(pid)[col]; return g.sum()/(g.count()+K)
a_r,a_w=eff('rb','striker_id'),eff('wb','striker_id'); c_r,c_w=eff('rw','bowler_id'),eff('ww','bowler_id')
b['rb_a']=b.rb+b.bowler_id.map(c_r).fillna(0); b['wb_a']=b.wb+b.bowler_id.map(c_w).fillna(0)
b['rw_a']=b.rw+b.striker_id.map(a_r).fillna(0); b['ww_a']=b.ww+b.striker_id.map(a_w).fillna(0)
for c in ['rb_a','wb_a','rw_a','ww_a']: b[c]-=b[c].mean()
b.to_pickle('b7.pkl')
cfg={'bat':dict(pid='striker_id',r='rb_a',w='wb_a'),'bowl':dict(pid='bowler_id',r='rw_a',w='ww_a')}
def pm(role):
    pid,r,w=cfg[role]['pid'],cfg[role]['r'],cfg[role]['w']
    d=b[b[r].notna()].copy(); d['v']=d[r]+d[w]
    h=d[d.hp].groupby([pid,'match_id']).agg(hs=('v','sum'),hr=(r,'sum'),hw=(w,'sum'),hn=('v','size'))
    o=d[~d.hp].groupby([pid,'match_id']).agg(ns=('v','sum'),nr=(r,'sum'),nw=(w,'sum'),nn=('v','size'))
    t=h.join(o,how='outer').fillna(0).reset_index().rename(columns={pid:'pid'})
    t['succ']=((t.hs>0)&(t.hn>=3)).astype(float); t['el']=(t.hn>=3).astype(float); t['m']=(t.hn>0).astype(float)
    return t
C=['hs','hr','hw','hn','ns','nr','nw','nn','succ','el','m']
def agg(t,w=None):
    if w is not None:
        wt=t.match_id.map(w).fillna(0).values; t=t.assign(**{c:t[c]*wt for c in C})
    return t.groupby('pid')[C].sum()
def zs(s): return (s-s.mean())/s.std()
def moments(s,n):
    mu=s/n; mb=s.sum()/n.sum(); y=((mu-mb)**2).values
    X=np.column_stack([np.ones(len(n)),1/n.values]); co=np.linalg.lstsq(X,y,rcond=None)[0]
    return max(co[0],1e-5),max(co[1],0.05)
def rate(t,w=None,fix=None,minhp=50,minm=12,minall=150,wts=(0.75,0.25)):
    p=agg(t,w); p=p[(p.hn>=minhp)&(p.m>=minm)&((p.hn+p.nn)>=minall)].copy()
    if fix is None: fix={'r':moments(p.nr,p.nn),'w':moments(p.nw,p.nn)}
    sk={}
    for comp,col in (('r','nr'),('w','nw')):
        ts,ss=fix[comp]; k=ss/ts; mb=p[col].sum()/p.nn.sum()
        sk[comp]=mb+(p.nn/(p.nn+k))*(p[col]/p.nn.clip(lower=1)-mb)
    p['skill_r']=sk['r']; p['skill_w']=sk['w']; p['skill']=sk['r']+sk['w']
    p['mu_hp']=p.hs/p.hn; p['mu_non']=p.ns/p.nn.clip(lower=1)
    ss_tot=fix['r'][1]+fix['w'][1]+13  # rough total ball-level var (refined below)
    ss_tot=max(ss_tot,10.0)
    d=p.mu_hp-p.mu_non; se2=ss_tot/p.hn+ss_tot/p.nn.clip(lower=1)
    tc=max(np.mean(d**2)-np.mean(se2),0.0); tcu=max(tc,0.0004); p['lam']=tcu/(tcu+se2)
    p['clutch']=p.lam*(d-np.average(d,weights=1/se2)); p['PV']=p.skill+p.clutch
    pr=p.succ.sum()/p.el.sum(); N0=12; p['hit']=(p.succ+N0*pr)/(p.el+N0)
    p['R']=wts[0]*zs(p.PV)+wts[1]*zs(p.hit)
    p.attrs.update(fix=fix,tc=tc); return p
out={}
for role in ('bat','bowl'):
    t=pm(role); p=rate(t); out[role]=(t,p)
    print(role,len(p),{k:(round(v[0],5),round(v[1],2),round(v[1]/v[0])) for k,v in p.attrs['fix'].items()},'clutch var',round(p.attrs['tc'],6))
    ms=np.sort(t.match_id.unique()); w1={m:1 for m in ms[::2]}; w2={m:1 for m in ms[1::2]}
    fix=p.attrs['fix']; p1=rate(t,w1,fix=fix,minhp=20,minm=5,minall=60); a2=agg(t,w2); a2=a2[a2.hn>=25]; ii=p1.index.intersection(a2.index)
    raw2=a2.hs[ii]/a2.hn[ii]; a1=agg(t,w1)
    print('  n=%d predictive: R %.3f | PV %.3f | skill %.3f | skill_r %.3f | naive HP %.3f'%(len(ii),*[spearmanr(x[ii],raw2)[0] for x in (p1.R,p1.PV,p1.skill,p1.skill_r,(a1.hs/a1.hn))]))
pickle.dump(out,open('out.pkl','wb'))
