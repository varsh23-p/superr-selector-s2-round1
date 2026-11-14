import pandas as pd, numpy as np, re
D='/home/claude/data/'
b=pd.read_csv(D+'ball_by_ball.csv'); m=pd.read_csv(D+'matches.csv'); c=pd.read_csv(D+'competitions.csv')
b=b[b.innings.isin([1,2])].copy()
b=b.sort_values(['match_id','innings','innings_ball_num']).reset_index(drop=True)
et=b.extras_type.fillna('none')
b['wide']=(et=='wide'); b['nb']=(et=='Overstep'); b['bye']=et.isin(['legbyes','byes'])
b['legal']=~(b.wide|b.nb)
b['tot']=b.runs+b.wide*1+b.nb*1+b.bye*1+b.extras_2_runs
b['wkt']=b.is_wicket & (b.dismissal_type!='Retire Out')
g=b.groupby(['match_id','innings'])
b['runs_before']=g.tot.cumsum()-b.tot
b['wk_before']=g.wkt.cumsum()-b.wkt
b['lb_before']=g.legal.cumsum()-b.legal
tot=g.tot.sum().unstack()
tot.columns=['t1','t2']
mm=m.merge(c[['comp_id','comp_name','match_format']],on='comp_id').merge(tot,left_on='match_id',right_index=True)
# validate margin
def marg(r):
    s=str(r.result); x=re.search(r'by (\d+) runs?',s,re.I)
    return int(x.group(1)) if x else np.nan
mm['mr']=mm.apply(marg,axis=1)
v=mm.dropna(subset=['mr','t1','t2']).copy()
v['calc']=(v.t1-v.t2).abs()
print('n margin matches',len(v),'exact',(v.calc==v.mr).mean(),'within2',((v.calc-v.mr).abs()<=2).mean(),'mean diff',(v.calc-v.mr).mean())
print('revised',mm.first_inn_revised_overs.notna().sum(),mm.second_inn_revised_overs.notna().sum(),mm.revised_target.notna().sum())
print(mm.groupby('match_format').size(), mm.comp_name[mm.match_format=='T20'].value_counts())
# legal balls per innings
lb=g.legal.sum().unstack(); print(lb.describe())
print(mm.win_team_id.isna().sum(), mm.knock_out.value_counts())
b.to_pickle('b.pkl'); mm.to_pickle('mm.pkl')
