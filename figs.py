import pandas as pd, numpy as np, pickle, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
b=pd.read_pickle('b7.pkl'); out=pickle.load(open('out.pkl','rb')); final=pickle.load(open('final.pkl','rb')); boot,sens=pickle.load(open('robust.pkl','rb'))
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
NAVY='#1f2a5a'; PINK='#e0368b'; GREY='#9aa0b4'
# Fig1: PI heatmaps
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
h=b[b.legal].groupby(['innings','overid']).PI.mean().unstack(0)
ax[0].plot(h.index,h[1],color=NAVY,lw=2,label='1st innings'); ax[0].plot(h.index,h[2],color=PINK,lw=2,label='2nd innings (chase)')
ax[0].axhline(1,color=GREY,ls=':'); ax[0].axhline(1.5,color='k',ls='--',lw=.8); ax[0].text(1,1.53,'high-pressure threshold (PI ≥ 1.5)',fontsize=7)
ax[0].set_xlabel('Over'); ax[0].set_ylabel('Mean Pressure Index'); ax[0].set_title('Pressure builds late, most in chases'); ax[0].legend(frameon=False)
c=b[(b.innings==2)&b.legal&(b.lb_before<114)].copy(); c['rrb']=pd.cut(c.rrr,[-3,6,8,10,12,15,40],labels=['≤6','6-8','8-10','10-12','12-15','>15']); c['wkb']=c.wk_before.clip(upper=7)
H=c.groupby(['wkb','rrb']).PI.mean().unstack()
im=ax[1].imshow(H.values,cmap='magma_r',aspect='auto',origin='lower'); ax[1].set_xticks(range(len(H.columns))); ax[1].set_xticklabels(H.columns); ax[1].set_yticks(range(len(H.index))); ax[1].set_yticklabels([str(i) if i<7 else '7+' for i in H.index])
ax[1].set_xlabel('Required run rate'); ax[1].set_ylabel('Wickets lost'); ax[1].set_title('Chase pressure: required rate × wickets lost'); plt.colorbar(im,ax=ax[1],label='Mean PI'); ax[1].spines[:].set_visible(False)
plt.tight_layout(); plt.savefig('figs/f1.png',dpi=200); plt.close()
# Fig2: rank intervals
rk=pickle.load(open('rk.pkl','rb'))
fig,ax=plt.subplots(1,2,figsize=(10,4.2))
for a_,role,title in ((ax[0],'bat','Batters'),(ax[1],'bowl','Bowlers')):
    F=final[role].set_index('id'); S=rk[role].head(10).copy()
    if role=='bat': S=S.reindex([50284,6281,1699,6396,4260,1318,5553,5039,813,4853])
    else: S=S.reindex([2933,781,8661,34485,830,7920,6523,3644,4373,360])
    ys=np.arange(10)[::-1]
    for y,i in zip(ys,S.index):
        r=F.loc[i]; a_.plot([r.p10,r.p90],[y,y],color=GREY,lw=3,alpha=.6); a_.plot(r.medrank,y,'o',color=PINK)
    a_.set_yticks(ys); a_.set_yticklabels([f'{k+1}. {F.loc[i,"name"]}' for k,i in enumerate(S.index)]); a_.set_xscale('log'); a_.set_xticks([1,3,10,30,100]); a_.set_xticklabels([1,3,10,30,100])
    a_.axvline(10,color='k',ls='--',lw=.7); a_.set_xlabel('Rank across 300 bootstrap resamples (median ● ; 10th–90th pct bar)'); a_.set_title(title+': rank stability')
plt.tight_layout(); plt.savefig('figs/f2.png',dpi=200); plt.close()
# Fig3: HP vs usual
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
for a_,role,title,lst in ((ax[0],'bat','Batters',[50284,6281,1699,6396,4260,1318,5553,5039,813,4853]),(ax[1],'bowl','Bowlers',[2933,781,8661,34485,830,7920,6523,3644,4373,360])):
    t,p=out[role]; nm=final[role].set_index('id').name
    a_.scatter(p.mu_non*100,p.mu_hp*100,s=np.clip(p.hn/6,8,90),color=GREY,alpha=.45)
    q=p.loc[[i for i in lst if i in p.index]]; a_.scatter(q.mu_non*100,q.mu_hp*100,s=np.clip(q.hn/6,20,90),color=PINK)
    for i in q.index: a_.annotate(nm[i].split()[-1],(q.mu_non[i]*100,q.mu_hp[i]*100),fontsize=7,xytext=(3,3),textcoords='offset points')
    lim=[min(p.mu_non.min(),p.mu_hp.min())*100-5,max(p.mu_non.max(),p.mu_hp.max())*100+5]; a_.plot(lim,lim,'k:',lw=.8)
    a_.set_xlabel('Usual level (non-high-pressure), runs above expected /100 balls'); a_.set_ylabel('High-pressure, runs above exp. /100 balls'); a_.set_title(title+': pressure vs usual (raw)')
plt.tight_layout(); plt.savefig('figs/f3.png',dpi=200); plt.close()
# Validation numbers
d=b[b.legal].copy(); d['grp']=np.where(d.PI>=1.5,'High (PI≥1.5)',np.where(d.PI<1.0,'Ordinary (PI<1)','Middle'))
v=d.groupby('grp').agg(balls=('PI','size'),death=('overid',lambda s:(s>15).mean()*100),chase=('innings',lambda s:(s==2).mean()*100),
  rpb=('tot','mean'),wk=('wkt','mean'),absdwp=('wp_after',lambda s:np.nan))
d['absd']=(d.wp_after-d.wp_before).abs(); v['abs_dwp']=d.groupby('grp').absd.mean()
v['close']=d.groupby('grp').apply(lambda x:((x.wp_before>0.25)&(x.wp_before<0.75)).mean()*100)
print(v.round(3).to_string()); v.to_pickle('val.pkl')
# phase / type shares for finalists
rows=[]
for role,pid in (('bat','striker_id'),('bowl','bowler_id')):
    for i in final[role].id:
        x=b[(b[pid]==i)&(b.PI>=1.5)]
        ph=x.phase.value_counts(normalize=True)*100
        r=dict(role=role,id=i,pp=ph.get('PP',0),mid=ph.get('Mid',0),death=ph.get('Death',0),chase=(x.innings==2).mean()*100)
        if role=='bat':
            xs=x[~x.wide]; 
            for k,l in (('Pace','pace'),('Spin','spin')):
                y=xs[xs.pace_or_spin==k]; r['sr_'+l]=100*y.runs.sum()/max(len(y),1); r['n_'+l]=len(y)
        rows.append(r)
pd.DataFrame(rows).to_pickle('phase.pkl'); print(pd.DataFrame(rows).round(0).head(30).to_string())
