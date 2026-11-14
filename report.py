import pandas as pd, numpy as np, pickle, warnings; warnings.filterwarnings('ignore')
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
final=pickle.load(open('final.pkl','rb')); rk=pickle.load(open('rk.pkl','rb')); ph=pd.read_pickle('phase.pkl'); val=pd.read_pickle('val.pkl')
NAVY=colors.HexColor('#1f2a5a'); PINK=colors.HexColor('#e0368b'); LG=colors.HexColor('#f1f2f8')
ss=getSampleStyleSheet()
B=ParagraphStyle('B',parent=ss['BodyText'],fontName='Helvetica',fontSize=9,leading=12.2,spaceAfter=4)
H1=ParagraphStyle('H1',parent=ss['Heading1'],keepWithNext=1,fontName='Helvetica-Bold',fontSize=14,textColor=NAVY,spaceBefore=8,spaceAfter=5)
H2=ParagraphStyle('H2',parent=ss['Heading2'],keepWithNext=1,fontName='Helvetica-Bold',fontSize=10.5,textColor=PINK,spaceBefore=6,spaceAfter=3)
T=ParagraphStyle('T',parent=B,fontSize=7.2,leading=8.8,spaceAfter=0)
TB=ParagraphStyle('TB',parent=T,fontName='Helvetica-Bold',textColor=colors.white)
Ti=ParagraphStyle('Ti',fontName='Helvetica-Bold',fontSize=21,textColor=NAVY,leading=25)
Sub=ParagraphStyle('Sub',parent=B,fontSize=10,textColor=colors.HexColor('#555555'))
SM=ParagraphStyle('SM',parent=B,fontSize=7.8,leading=10,textColor=colors.HexColor('#444444'))
def P(t,s=B): return Paragraph(t,s)
bat_ids=[50284,6281,1699,6396,4260,1318,5553,5039,813,4853]
bowl_ids=[2933,781,8661,34485,830,7920,6523,3644,4373,360]
FB=final['bat'].set_index('id'); FW=final['bowl'].set_index('id'); PH=ph.set_index(['role','id'])
def phs(role,i): r=PH.loc[(role,i)]; return f"{r.pp:.0f}/{r['mid']:.0f}/{r.death:.0f}"
def mk(rows,widths,hdr):
    data=[[Paragraph(h,TB) for h in hdr]]+[[Paragraph(str(c),T) for c in r] for r in rows]
    t=Table(data,colWidths=[w*mm for w in widths],repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LG]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LINEBELOW',(0,0),(-1,-1),0.25,colors.HexColor('#d0d3e0')),('TOPPADDING',(0,0),(-1,-1),2.5),('BOTTOMPADDING',(0,0),(-1,-1),2.5),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3)]))
    return t
def short(t): return t if isinstance(t,str) and t!='—' else '—'
def team(t): return {'Royal Challengers Bengaluru':'RCB','Sunrisers Hyderabad':'SRH','Mumbai Indians':'MI','Kolkata Knight Riders':'KKR','Rajasthan Royals':'RR','Gujarat Titans':'GT','Chennai Super Kings':'CSK','Delhi Capitals':'DC','Lucknow Super Giants':'LSG','Punjab Kings':'PBKS'}.get(t,'—')
brow=[]
for k,i in enumerate(bat_ids):
    r=FB.loc[i]; brow.append([k+1,f"<b>{r['name']}</b>",team(r.team26),int(r.balls),int(r.m),f"{r.sr:.0f} <font color='#777777'>({r.exp_sr:.0f})</font>",f"{int(r.dis)} ({r.bpd:.0f})",f"{r.PV*100:+.0f}",f"{r.hit*100:.0f}%",f"{r.top10:.0f}%",phs('bat',i)])
wrow=[]
for k,i in enumerate(bowl_ids):
    r=FW.loc[i]; wrow.append([k+1,f"<b>{r['name']}</b>",team(r.team26),int(r.balls),int(r.m),f"{r.econ:.2f} <font color='#777777'>({r.exp_econ:.2f})</font>",f"{int(r.wk)}",f"{r['dot']:.0f}%",f"{r.PV*100:+.0f}",f"{r.hit*100:.0f}%",f"{r.top10:.0f}%",phs('bowl',i)])
def sr_split(i): r=PH.loc[('bat',i)]; return f"SR vs pace {r.sr_pace:.0f} / spin {r.sr_spin:.0f}"
bn={
50284:f"LHB opener-type: {PH.loc[('bat',50284)].pp:.0f}% of pressure balls in the powerplay, strike rate {FB.loc[50284].sr:.0f} against {FB.loc[50284].exp_sr:.0f} expected. Already an RR player in IPL 2026, so a retention/continuity call rather than an auction target. Smallest sample of the ten (137 balls), so the widest uncertainty.",
6281:f"LHB powerplay specialist ({PH.loc[('bat',6281)].pp:.0f}% of pressure balls) with the <i>largest</i> evidence base (96 matches). Strike rate {FB.loc[6281].sr:.0f} vs {FB.loc[6281].exp_sr:.0f} expected, and one of the best success rates ({FB.loc[6281].hit*100:.0f}% of pressure innings positive).",
1699:f"RHB middle-overs accelerator ({PH.loc[('bat',1699)]['mid']:.0f}% of pressure balls in overs 7-15), {sr_split(1699)}. Highest probability of finishing top-10 under resampling ({FB.loc[1699].top10:.0f}%).",
6396:f"RHB finisher: {PH.loc[('bat',6396)].death:.0f}% of pressure balls in the death overs, strike rate {FB.loc[6396].sr:.0f} vs {FB.loc[6396].exp_sr:.0f} expected, {PH.loc[('bat',6396)].chase:.0f}% of the time while chasing.",
4260:f"RHB in overs 7-20 ({PH.loc[('bat',4260)]['mid']:.0f}% middle / {PH.loc[('bat',4260)].death:.0f}% death), {sr_split(4260)}: better against spin. Second-highest base-threshold rating of all 169 eligible batters (after Sooryavanshi); sits 5th only because the rank-average penalises sensitivity to the pressure threshold.",
1318:f"RHB death-overs hitter ({PH.loc[('bat',1318)].death:.0f}% of pressure balls), strike rate {FB.loc[1318].sr:.0f} vs {FB.loc[1318].exp_sr:.0f}. Dismissed 12 times in 142 balls, hence a wide rank interval (3-85).",
5553:f"RHB opener ({PH.loc[('bat',5553)].pp:.0f}% powerplay): strike rate {FB.loc[5553].sr:.0f} and just {int(FB.loc[5553].dis)} dismissals in {int(FB.loc[5553].balls)} pressure balls (one per {FB.loc[5553].bpd:.0f}). Rating is the most threshold-sensitive of the ten.",
5039:f"RHB middle-overs batter; strong against spin (SR {PH.loc[('bat',5039)].sr_spin:.0f}) but only {PH.loc[('bat',5039)].sr_pace:.0f} against pace. Sample is IPL-only (100% IPL).",
813:f"RHB opener with a large sample (405 balls, 79 matches); strike rate {FB.loc[813].sr:.0f} vs {FB.loc[813].exp_sr:.0f} expected is a modest edge, but very hard to dismiss ({int(FB.loc[813].dis)} in {int(FB.loc[813].balls)}).",
4853:f"RHB middle-overs batter, only {FB.loc[4853].ipl_share:.0f}% of pressure balls in the IPL (mostly T20Is): treat the rating with extra caution. {int(FB.loc[4853].dis)} dismissals in {int(FB.loc[4853].balls)} balls.",
}
wn={
2933:f"RH fast: usable in all phases ({phs('bowl',2933)} PP/mid/death). Economy {FW.loc[2933].econ:.2f} vs {FW.loc[2933].exp_econ:.2f} expected and {FW.loc[2933]['dot']:.0f}% dots under pressure. Clear #1 on every threshold.",
781:f"Off-spin, {PH.loc[('bowl',781)]['mid']:.0f}% of pressure overs in the middle phase. Economy {FW.loc[781].econ:.2f} vs {FW.loc[781].exp_econ:.2f} expected; all pressure data is IPL.",
8661:f"Death-overs specialist ({PH.loc[('bowl',8661)].death:.0f}% of pressure balls): economy {FW.loc[8661].econ:.2f} vs {FW.loc[8661].exp_econ:.2f} expected, {int(FW.loc[8661].wk)} wickets in {int(FW.loc[8661].balls)} balls.",
34485:f"Left-arm wrist-spin in the middle overs ({PH.loc[('bowl',34485)]['mid']:.0f}%): economy {FW.loc[34485].econ:.2f} but only {int(FW.loc[34485].balls)} pressure balls and no IPL data. A Pakistan player, so not an IPL auction option: ranking is analytical.",
830:f"Powerplay/death seamer; {int(FW.loc[830].wk)} wickets in {int(FW.loc[830].balls)} pressure balls (most of the ten) and {FW.loc[830]['dot']:.0f}% dots. Highest consistency ({FW.loc[830].hit*100:.0f}% positive spells).",
7920:f"Medium-pace, {PH.loc[('bowl',7920)].death:.0f}% death overs / {PH.loc[('bowl',7920)].pp:.0f}% powerplay; economy {FW.loc[7920].econ:.2f} vs {FW.loc[7920].exp_econ:.2f} expected.",
6523:f"Fast-medium; only {FW.loc[6523].ipl_share:.0f}% of pressure balls in the IPL, mostly T20Is: larger context uncertainty.",
3644:f"Left-arm wrist-spin, {PH.loc[('bowl',3644)]['mid']:.0f}% middle overs; economy {FW.loc[3644].econ:.2f} vs {FW.loc[3644].exp_econ:.2f}. Second-highest rating at the base threshold, drops at the strictest threshold.",
4373:f"Left-arm medium, {FW.loc[4373].wk:.0f} wickets; economy {FW.loc[4373].econ:.2f} vs {FW.loc[4373].exp_econ:.2f}; {FW.loc[4373].ipl_share:.0f}% IPL sample.",
360:f"Swing bowler in the powerplay and at the death ({phs('bowl',360)}); largest bowling sample (73 matches) but a modest edge: economy {FW.loc[360].econ:.2f} vs {FW.loc[360].exp_econ:.2f} expected.",
}
def notes(ids,ds,role):
    return [[k+1,f"<b>{(FB if role=='bat' else FW).loc[i,'name']}</b>",ds[i]] for k,i in enumerate(ids)]
doc=SimpleDocTemplate('outputs/SupeRR_Selector_S2_Round1_Methodology_Report.pdf',pagesize=A4,leftMargin=16*mm,rightMargin=16*mm,topMargin=14*mm,bottomMargin=14*mm,
    title='SupeRR Selector S2 - Round 1 Methodology Report',author='Varsha Patil')
W=A4[0]-32*mm
S=[]
S+= [P('SupeRR Selector Season 2 &mdash; Round 1',Sub),P('Who Delivers When It Matters? A Win-Probability-Based Pressure Rating for IPL &amp; T20I Batters and Bowlers',Ti),Spacer(1,3),
 P('Methodology report &middot; data: IPL and T20I ball-by-ball, 2022 onwards (818 usable matches, 195,124 deliveries)',Sub),Spacer(1,6)]
S+= [P('1. Summary and final rankings',H1),
 P("<b>Approach in one paragraph.</b> I define <i>pressure</i> from the match situation itself: a delivery-level win-probability model (trained out-of-fold, by match) gives each ball a <b>Pressure Index (PI)</b>, the expected swing in win probability from the next ball, scaled so the average ball is 1.0. Balls with PI &ge; 1.5 are 'high pressure' (19.5% of all deliveries). Each ball is then scored against a context-matched expectation (over, wickets, required rate, pressure level, pace/spin, venue type) for runs and for dismissals, adjusted for opposition quality, converted into a single runs-above-expected value, and shrunk with empirical-Bayes. The rating is the best estimate of how a player performs <i>in</i> high-pressure situations, built from their performance in ordinary situations plus whatever pressure-specific deviation the data can actually support, combined with a consistency score (share of pressure innings/spells with a positive contribution)."),
 P("<b>Main finding.</b> Pressure-specific over- or under-performance is statistically indistinguishable from zero in this dataset: the estimated variance of a persistent 'clutch' trait was effectively nil, and raw pressure-only averages are very noisy (split-half rank correlation &asymp; 0.05 for batters). So the most reliable players under pressure are chiefly the players with the best <i>skill measured in the right context</i>, and the method says so rather than rewarding lucky small samples. Rank order is a probabilistic judgement, so the report shows rank-stability intervals alongside every pick.")]
lst=[[ 'Rank','Top-10 batters','Top-10 bowlers']]
rowsL=[[k+1,FB.loc[bat_ids[k],'name'],FW.loc[bowl_ids[k],'name']] for k in range(10)]
tbl=mk([[r[0],f"<b>{r[1]}</b>",f"<b>{r[2]}</b>"] for r in rowsL],[16,75,75],['Rank','Top-10 batters under pressure','Top-10 bowlers under pressure'])
S+=[Spacer(1,3),tbl,Spacer(1,3),P("Final order = average of four ranks: the model rating at three pressure thresholds (PI &ge; 1.25, 1.5, 2.0) and the median rank across 300 match-level bootstrap resamples. This stops any single arbitrary cut-off from driving the list. Ties were broken by the base-threshold rating.",SM)]
S+=[P('2. Data and preparation',H1),
 P("The pack has 883 matches, 867 of them with ball-by-ball data (470 T20I; 397 other T20 including five IPL seasons and two Emerging-Teams events), and ~201k deliveries. I kept innings 1-2 only (super overs excluded) and matches with a result and no DLS revision (818 matches: 357 IPL, 461 international/other). Cleaning notes that matter for reproducibility:"),
 P("&bull; <b>Extras columns are codes, not runs</b> (<font face='Courier'>extras1_runs</font>: 2 = wide, 3 = leg-bye, 4 = bye, 1 = no-ball, 5 = other, 6 = none). Ball value = bat runs + 1 for wides/no-balls + 1 for byes/leg-byes + <font face='Courier'>extras_2_runs</font> + 1 for 'other'. Bye/leg-bye run counts are not recoverable; I tested the assumptions against stated winning margins in 405 'won by N runs' matches (innings totals match to &plusmn;2 runs in 58%; the rest is mostly unrecoverable bye/leg-bye runs). Legal ball = not a wide and not an overstep (no-ball).<br/>&bull; <b>Wickets</b>: excluded 'Retire Out' from team wickets; run-outs are charged to the batter dismissed but not credited to the bowler.<br/>&bull; <b>Opponent/context</b>: IPL vs international is an explicit model feature (ipl flag), and knock-out matches are flagged (31 matches).<br/>&bull; The supplied PDF problem statement and CSVs were used as provided; no external data was used.")]
S+=[P('3. Defining and measuring pressure',H1),
 P("<b>Why a win-probability definition.</b> Required run rate, wickets in hand, balls remaining and closeness all matter, but they interact (8 an over with 3 wickets down in over 12 is not the same as with 7 down in over 18). A win-probability model combines them into one number and treats the first innings consistently. I fit gradient-boosted classifiers, one per innings, on every post-ball state (monotonic in runs, wickets and balls left; features: runs/needed, wickets lost, balls left, required rate, IPL flag, target). Predictions are <b>out-of-fold by match</b> (5-fold), so no match is scored by a model that saw it. Calibration: first innings Brier 0.224 / AUC 0.69 (a first innings is genuinely hard to call), chase Brier 0.145 / AUC 0.875."),
 P("<b>Pressure Index.</b> For each delivery, using the same model, I compute the win-probability result of each possible outcome (0, 1, 2, 3, 4, 6, wicket) weighted by how often that outcome occurs at that point of the innings: <i>leverage</i> = E|&Delta;WP|. PI = leverage / league-average leverage, &times; 1.25 for knock-out matches. High pressure = PI &ge; 1.5; 'ordinary' = PI &lt; 1.0."),
 Image('figs/f1.png',width=W,height=W*0.36)]
vr=val
vrows=[]
for g in ['High (PI≥1.5)','Middle','Ordinary (PI<1)']:
    r=vr.loc[g]; vrows.append([g,f"{int(r.balls):,}",f"{r.death:.0f}%",f"{r.chase:.0f}%",f"{r.abs_dwp:.3f}",f"{r.close:.0f}%",f"{r.wk*100:.1f}%"])
S+=[Spacer(1,2),mk(vrows,[34,22,24,24,34,28,24],['Situation','Legal balls','In death overs','In 2nd innings','Mean |&Delta;WP| per ball','Match still 25-75%','Wicket rate']),
 P("<b>It separates pressure from ordinary play:</b> a typical high-pressure ball moves the win probability about 6&times; as much as an ordinary one, and is far more likely to come in a close chase or at the death. It is not simply 'the last over': pressure also arises mid-innings after collapses or when the required rate climbs (right panel).",SM)]
S+=[P('4. Performance measurement and rating logic',H1),
 P("<b>Per-ball value (batters).</b> For every ball faced I compute (runs scored &minus; expected runs) &minus; W &times; (dismissed &minus; expected dismissal), where expectations come from gradient-boosted models (cross-fitted by match) using the match situation &mdash; innings, over, wickets, balls left, runs, required and current run rate, win probability, <b>PI itself</b>, IPL flag, knock-out flag, bowler pace/spin, batter handedness &mdash; but <i>no player identifiers</i>. So a batter on a 14-an-over chase is compared with what an average batter does in that same pressure, not with a league average. <b>W</b> is the run-equivalent of a wicket, derived from the win-probability model (average WP lost per wicket &divide; WP gained per run) by innings and phase: 1st inn 17.8 / 17.3 / 12.4 runs (PP / middle / death); chase 11.0 / 8.4 / 4.3 runs. Wides are charged to the bowler only."),
 P("<b>Per-ball value (bowlers).</b> (expected runs conceded &minus; runs conceded incl. extras) + W &times; (wicket credited &minus; expected wicket), per delivery bowled."),
 P("<b>Opposition quality.</b> Each player's overall (all-balls) effect is estimated with shrinkage (K = 400 balls). Batter values add back the quality of the bowler faced; bowler values add back the quality of the batter."),
 P("<b>Player baseline and 'compared with usual'.</b> Each player's <i>usual</i> level is their value over non-high-pressure balls. Their pressure estimate is: PV = shrunk usual level (runs and wicket components shrunk separately, because the run component is far more reliable than the wicket component) + shrunk (pressure &minus; usual) deviation. The prior variances come from method-of-moments on the eligible players: roughly 1,000 balls' worth of shrinkage for the run component and 500-1,500 for wickets; the deviation prior is a floor of 0.02 runs/ball standard deviation, because the data-driven estimate of persistent clutch variance was zero."),
 P("<b>Consistency.</b> Share of the player's pressure innings (batters, &ge;3 pressure balls) or spells (bowlers, &ge;3 pressure balls) with a positive net value, shrunk to the league rate with a 12-appearance prior. <b>Composite rating</b> R = 0.75 &times; z(PV) + 0.25 &times; z(consistency)."),
 P("<b>Eligibility.</b> &ge;50 pressure balls, pressure balls in &ge;12 matches, and &ge;150 balls in total: 169 batters and 170 bowlers qualify.")]
S+=[P('5. Fair comparisons',H1),
 P("<b>Sample size</b>: eligibility floors plus empirical-Bayes shrinkage, so a 20-ball hot streak cannot outrank a 400-ball body of work; the report states each player's ball and match counts. <b>Different match contexts</b>: expectations are conditioned on over, wickets, required rate, PI, innings and knock-out status; opponent quality is adjusted. <b>IPL vs T20I</b>: both are used (more pressure situations and more matches), the model has an IPL indicator, and each table shows the share of a player's pressure balls from the IPL; players whose sample is mostly T20I (e.g., Phillips, Muzarabani) carry a caution because international opposition varies more in quality than IPL line-ups. <b>Roles</b>: because a death-overs finisher and a powerplay opener face different pressure, the phase mix of each player's pressure balls is shown (PP / middle / death %).")]
S+=[P('6. Evidence for the top 10',H1),P('<b>Batters.</b> Columns: pressure balls faced; matches with pressure balls; strike rate on those balls (context-expected strike rate in grey); dismissals (balls per dismissal); PV in runs per 100 pressure balls above an average batter in the same situations (incl. the wicket cost); % of pressure innings that were net-positive; share of bootstrap resamples with the player in the top 10; phase mix.')]
S+=[mk(brow,[7,29,9,12,9,20,16,13,13,13,19],['#','Player','IPL26','Balls','M','SR (expected)','Dis. (b/dis)','PV /100','Pos. inn.','Top-10 %','PP/Mid/Death %'])]
S+=[Spacer(1,5),P('<b>Bowlers.</b> Columns: pressure balls bowled; matches; economy (context-expected economy in grey); wickets credited; dot-ball %; PV in runs per 100 pressure balls above an average bowler in the same situations; % of pressure spells that were net-positive; bootstrap top-10 share; phase mix.')]
S+=[mk(wrow,[7,29,9,12,8,21,9,10,12,12,13,19],['#','Player','IPL26','Balls','M','Econ (expected)','Wkts','Dots','PV /100','Pos. spl.','Top-10 %','PP/Mid/Death %'])]
S+=[Spacer(1,4),Image('figs/f2.png',width=W,height=W*0.42),Spacer(1,2),Image('figs/f3.png',width=W,height=W*0.38),
 P("Left/right: rank intervals show how much each pick moves when matches are resampled; the scatter shows raw pressure versus usual level (grey = all eligible players; pink = the top 10). Most top-10 players sit above the diagonal, i.e. they are better under pressure <i>and</i> in ordinary phases, but the raw gap is partly sampling noise, which is why the final PV leans on shrunk usual level.",SM)]
S+=[PageBreak(),P('7. Player notes (role and squad relevance)',H1),
 P("<b>Batters</b>",H2),mk(notes(bat_ids,bn,'bat'),[7,35,132],['#','Player','Why this player and where they fit']),Spacer(1,4),P("<b>Bowlers</b>",H2),mk(notes(bowl_ids,wn,'bowl'),[7,35,132],['#','Player','Why this player and where they fit']),
 Spacer(1,3),P("Notes on squad relevance use only the supplied data (IPL 2026 team from the mapping file, pressure phase mix, pace/spin split). Current franchise squads, retentions, overseas-slot status and auction availability are not in the data pack and must be checked separately.",SM)]
S+=[P('8. Reliability and limitations',H1),
 P("<b>Consistency checks.</b> (a) <i>Split-half</i>: using odd-numbered matches to rate players and even-numbered matches to test them, the shrunk batting rating predicts held-out pressure performance with Spearman 0.15 against 0.05 for the naive pressure-only average (3&times; better, but still modest). For <b>bowlers</b> no method predicted held-out pressure performance (about &minus;0.07): bowling pressure results here are at the edge of what ~300 balls per player can support. (b) <i>Bootstrap</i>: 300 match-level resamples; Section 6 shows rank intervals. Bumrah, Suryakumar, Klaasen, Sooryavanshi and Abhishek are the most stable (each in the top 10 in 56-73% of resamples); places 6-10 are interchangeable with the next ten or so. (c) <i>Sensitivity</i>: changing the pressure threshold keeps 5-10 of the 10 batters/7-10 of 10 bowlers; hence the consensus ranking. Changing the consistency weight (0%-40%) retains 7-9 of 10."),
 P("<b>Where the model is less certain.</b><br/>&bull; Low reliability of any pressure-only measure: a player has ~100-500 pressure balls; the wicket component is especially noisy.<br/>&bull; The 'no clutch' conclusion means rankings are driven by context-adjusted skill; they say 'who is good in the situations that matter' more than 'who changes gear under pressure'.<br/>&bull; First-innings win probability is imprecise (AUC 0.69), so first-innings pressure is less well measured than chase pressure.<br/>&bull; Team strength, pitch, dew and venue are not modelled; bye and leg-bye runs are approximated; the pack covers 2022 onwards only and mixes IPL with T20I opposition of varying quality.<br/>&bull; Role effects: openers see pressure in the powerplay, finishers at the death; the phase mix is reported but not used to rescale ratings.<br/>&bull; Player effects used for opponent adjustment include the same balls being evaluated (a small, heavily shrunk leakage).")]
S+=[P('9. Reproducibility',H1),
 P("All steps are scripted in Python (pandas, scikit-learn): (1) clean and decode extras; (2) out-of-fold win-probability models and the leverage-based Pressure Index; (3) out-of-fold expected-runs/dismissal models; (4) per-ball values with opponent adjustment; (5) empirical-Bayes ratings, consistency, bootstrap and sensitivity runs; (6) rank-average consensus. Full ranked tables (including the next ten players) are supplied in the accompanying CSV and code bundle. Fixed random seeds are used throughout.")]
def foot(c,d):
    c.saveState(); c.setFont('Helvetica',7); c.setFillColor(colors.HexColor('#888888')); c.drawString(16*mm,8*mm,'SupeRR Selector Season 2 - Round 1 methodology'); c.drawRightString(A4[0]-16*mm,8*mm,f'Page {d.page}'); c.restoreState()
doc.build(S,onFirstPage=foot,onLaterPages=foot)
print('ok')
