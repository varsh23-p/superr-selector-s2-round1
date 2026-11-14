import pandas as pd, numpy as np, pickle, warnings; warnings.filterwarnings('ignore')
boot,sens=pickle.load(open('robust.pkl','rb')); out=pickle.load(open('out.pkl','rb'))
b=pd.read_pickle('b7.pkl')
mp=pd.read_csv('data/competition_team_player_mapping.csv')
ipl26=mp[mp.comp_name.str.contains('IPL 2026')].drop_duplicates('player_id').set_index('player_id').team_name
nat=mp[~mp.comp_name.str.contains('IPL')].drop_duplicates('player_id',keep='last').set_index('player_id').team_name
final={}
for role,pid,nm in (('bat','striker_id','striker_name'),('bowl','bowler_id','bowler_name')):
    t,p=out[role]
    S=pd.concat([sens[(role,h)] for h in (1.25,1.5,2.0)],axis=1,join='inner'); S.columns=['t125','t150','t200']
    S['cons']=S.mean(axis=1); S=S.join(p[['R']].rename(columns={'R':'R_base'}),how='left')
    S['rank']=S.cons.rank(ascending=False)
    top=S.sort_values('cons',ascending=False).head(14)
    names=b.groupby(pid)[nm].agg(lambda s:s.mode()[0])
    hp=b[b.PI>=1.5]
    rows=[]
    for i in top.index:
        d=hp[hp[pid]==i]
        if role=='bat':
            f=d[~d.wide]; lg=f[f.legal]
            r=dict(balls=len(f),runs=int(f.runs.sum()),sr=100*f.runs.sum()/len(f),dis=int(f.bat_dis.sum()),bpd=len(f)/max(f.bat_dis.sum(),1),
                   boundary=100*((f.runs>=4).mean()),exp_sr=100*f.e_runs.mean())
            hand=b[b.striker_id==i].batting_type.mode()[0]; typ=hand
            mo=b[(b.striker_id==i)&(b.PI>=1.5)]
        else:
            f=d; lg=f[f.legal]
            r=dict(balls=len(lg),runs=int(f.tot.sum()),econ=6*f.tot.sum()/len(lg),wk=int(f.bowl_wk.sum()),dot=100*(f.tot==0).mean(),exp_econ=6*f.e_tot.sum()/len(lg))
            typ=b[b.bowler_id==i].bowling_type.mode()[0]
        r.update(id=i,name=names[i],team26=ipl26.get(i,'—'),nat=nat.get(i,'—'),type=typ,
                 m=int(p.m.get(i,np.nan)) if i in p.index else np.nan, ipl_share=100*d.ipl.mean(),
                 knockout=int(d.knock_out.sum()>0),
                 cons=top.cons[i],R_base=top.R_base[i],PV=p.PV.get(i,np.nan),hit=p.hit.get(i,np.nan),
                 skill=p.skill.get(i,np.nan),mu_hp=p.mu_hp.get(i,np.nan),mu_non=p.mu_non.get(i,np.nan),
                 top10=boot[role].top10_pct.get(i,np.nan),medrank=boot[role].med_rank.get(i,np.nan),p10=boot[role].p10.get(i,np.nan),p90=boot[role].p90.get(i,np.nan),
                 t125=top.t125[i],t150=top.t150[i],t200=top.t200[i])
        rows.append(r)
    F=pd.DataFrame(rows); final[role]=F
    pd.set_option('display.width',250)
    print(role); print(F.drop(columns=['id']).round(2).to_string())
pickle.dump(final,open('final.pkl','wb'))
