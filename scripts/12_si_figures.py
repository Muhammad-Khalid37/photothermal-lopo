"""Supplementary figures S1-S11 (run after scripts 02, 10 and 11)."""
from _common import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
args()
P, _ = load_pairs()
mae = lambda a, b: float(np.mean(np.abs(a - b)))
S = load_json("11_supplementary_analyses.json"); T5J = load_json("10_descriptor_models.json")
R = dict(eta2=S["clustering"]["eta2"], S7pub={q: dict(support_seen=v["support"], active_seen=v["active"], pair_seen=v["pair"]) for q, v in S["novelty"]["per_publication"].items()},
         S6=dict(fold=S["ridge_vs_training_mean"]), S7={q: dict(nested=v["nested"], a1=v["alpha_1"], alpha=v["alpha"]) for q, v in S["ridge_LI_coefficients"].items()},
         S8_ridge=S["error_share"]["ridge"], S8_rf=S["error_share"]["rf"], S9=S["intensity_mixed"],
         S8tab=[dict(d, cat=d["catalyst"].replace("-", "–")) for d in S["intensity_series"]])
y = P.dSTY.values; g = P.pub.values; pubs = sorted(set(g))
rid = np.load(C.RESULTS / "abs_ridge_ext_preds.npy"); rfS = np.load(C.RESULTS / "abs_rf_preds.npy"); rf = rfS.mean(0)
R["S6"]["sd_pred"] = float(np.std(rid, ddof=1))
F = str(C.FIGURES) + "/S"
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e','savefig.dpi':300})
B,O,A,Y,M,G,V,Rr='#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#6250d6','#e34948'; GR='#8a8f99'
def grid(ax,axis='y'): ax.grid(axis=axis,color='#e6e6e3',lw=0.6); ax.set_axisbelow(True)
U='mmol g$^{-1}$ h$^{-1}$'
# S1 workflow
fig,ax=plt.subplots(figsize=(7.2,6.4)); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,12)
boxes=[(5,11.2,'952 records, 28 publications (photothermal CO$_2$ methanation database)'),
 (5,9.9,'Split by light intensity: LI = 0 (323), LI > 0 (443), LI missing (186, not used)'),
 (5,8.6,'Exact grouping on DOI + 15 descriptors (catalyst, pretreatment, conditions)\nNaN treated as identical; only LI may differ'),
 (5,7.15,'Keep illuminated records with exactly one dark candidate\n(24 ambiguous records of P04 and P08 excluded; Table S3)'),
 (5,5.8,'96 dark–light comparisons, 74 dark references, 13 publications (P01–P13)'),
 (2.5,4.3,'Absolute response\nΔSTY = STY$_{light}$ − STY$_{dark}$\n96 pairs'),(7.5,4.3,'Relative response\nlog$_2$(STY$_{light}$/STY$_{dark}$)\n76 pairs (STY$_{dark}$ > 0)'),
 (2.5,2.5,'LOPO: baselines, nested Ridge, RF;\nmixed CV, clustering, ablation,\nalternative and mixed-effects models'),(7.5,2.5,'LOPO: GB, RF, Ridge, thermal null;\npermutation, sign, jackknife and\nselection-adjusted inference'),
 (5,0.8,'Primary-paper audit (classes A, A*, B, C), sensitivity analyses and local calibration')]
for x,yy,t in boxes:
    w=8.6 if x==5 else 4.3
    ax.add_patch(FancyBboxPatch((x-w/2,yy-0.5),w,1.0,boxstyle='round,pad=0.02,rounding_size=0.12',fc='#f0f5fc',ec=B,lw=0.9)); ax.text(x,yy,t,ha='center',va='center',fontsize=8)
ar=dict(arrowstyle='-|>',color='#52514e',lw=0.9)
for (x1,y1,x2,y2) in [(5,10.7,5,10.4),(5,9.4,5,9.1),(5,8.1,5,7.65),(5,6.65,5,6.3),(4,5.3,2.5,4.8),(6,5.3,7.5,4.8),(2.5,3.8,2.5,3.0),(7.5,3.8,7.5,3.0),(2.5,2.0,4,1.3),(7.5,2.0,6,1.3)]:
    ax.annotate('',(x2,y2),(x1,y1),arrowprops=ar)
fig.savefig(F+'1.png',bbox_inches='tight'); plt.close()
# S2 parity
fig,axs=plt.subplots(1,2,figsize=(7.2,3.5),sharey=True)
lim=[min(y.min(),rid.min(),rf.min())-10,max(y.max(),rf.max())+10]
for ax,p,t in zip(axs,[rid,rf],['Nested Ridge','Random Forest (mean of 20 seeds)']):
    ax.plot(lim,lim,color=GR,lw=0.8,ls='--'); ax.axhline(0,color='#cfcfcb',lw=0.6); ax.axvline(0,color='#cfcfcb',lw=0.6)
    ax.scatter(y,p,s=16,color=B,alpha=0.75,edgecolor='white',lw=0.4)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel(f'Observed ΔSTY ({U})'); ax.set_title(f'{t}\nMAE {(mae(y,p) if p is rid else np.mean([mae(y,q) for q in rfS])):.2f}',loc='left',fontsize=9)
axs[0].set_ylabel(f'Held-out prediction ({U})')
fig.tight_layout(); fig.savefig(F+'2.png'); plt.close()
# S3 eta2
E2=R['eta2']; lab={'LI':'Light intensity','MS':'Catalyst mass','GHSV':'GHSV','RP':'Pressure','H2CO2':'H$_2$/CO$_2$ ratio','T':'Temperature','AC':'Active-component loading','PC':'Promoter loading'}
ks=sorted(E2,key=lambda k:E2[k]['eta2'])
fig,ax=plt.subplots(figsize=(6,3.4)); v=[E2[k]['eta2'] for k in ks]
ax.barh(range(len(ks)),v,color=[GR if E2[k]['p']>0.05 else B for k in ks],height=0.6)
for i,k in enumerate(ks): ax.text(v[i]+0.01,i,f"{v[i]:.2f} (p {'< 0.001' if E2[k]['p']<0.001 else '= %.2f'%E2[k]['p']})",va='center',fontsize=8)
ax.set_yticks(range(len(ks))); ax.set_yticklabels([lab[k] for k in ks]); ax.set_xlim(0,1.25); ax.set_xlabel('Between-publication share of variance (η²)'); grid(ax,'x')
fig.tight_layout(); fig.savefig(F+'3.png'); plt.close()
# S4 recipe map: support x active, marker per publication
X=P.copy(); X['prom']=X.promoter.fillna('–')
sup=sorted(X.support.unique()); act=sorted(X.active.unique())
cols=[B,O,A,Y,M,G,V,Rr,'#1c5cab','#8c5a2b','#52514e','#b0306a','#0d8a8a']
fig,ax=plt.subplots(figsize=(7.2,4.6))
off={}
for i,q in enumerate(pubs):
    d=X[X.pub==q].drop_duplicates(['support','active','prom'])
    for r in d.itertuples():
        key=(r.support,r.active); k=off.get(key,0); off[key]=k+1
        ax.scatter(act.index(r.active)+0.18*k-0.1,sup.index(r.support),s=46,color=cols[i],edgecolor='white',lw=0.6,zorder=3,label=q if r.Index==d.index[0] else None)
ax.set_xticks(range(len(act))); ax.set_xticklabels(act); ax.set_yticks(range(len(sup))); ax.set_yticklabels(sup)
ax.set_xlabel('Active component'); ax.set_ylabel('Support'); ax.grid(color='#eeeeec',lw=0.6); ax.set_axisbelow(True)
ax.legend(ncol=1,fontsize=7,frameon=False,bbox_to_anchor=(1.01,1),loc='upper left',title='Publication',title_fontsize=7.5)
fig.tight_layout(); fig.savefig(F+'4.png'); plt.close()
# S5 per-publication novelty
pp=pd.DataFrame(R['S7pub']).T.loc[pubs]
fig,ax=plt.subplots(figsize=(7.2,3.2)); x=np.arange(len(pubs)); w=0.26
for j,(c,lb,cl) in enumerate([('support_seen','Support in training',B),('active_seen','Active component in training',O),('pair_seen','Support–active pair in training',A)]):
    ax.bar(x+(j-1)*w,pp[c]*100,w-0.03,color=cl,label=lb)
ax.set_xticks(x); ax.set_xticklabels(pubs); ax.set_ylabel('Held-out rows (%)'); ax.set_ylim(0,110); grid(ax)
ax.legend(frameon=False,fontsize=7.5,ncol=3,loc='upper center',bbox_to_anchor=(0.5,1.14))
ax.text(0.99,0.97,'Complete recipe in training: 0% of rows in every fold',transform=ax.transAxes,ha='right',va='top',fontsize=7.5,color='#52514e')
fig.tight_layout(); fig.savefig(F+'5.png'); plt.close()
# S6
fo=R['S6']['fold']; tm=[fo[q]['train_mean'] for q in pubs]; mp=[fo[q]['mean_pred'] for q in pubs]
fig,axs=plt.subplots(1,2,figsize=(7.2,3.3))
ax=axs[0]; ax.plot([10,45],[10,45],color=GR,ls='--',lw=0.8); ax.scatter(tm,mp,s=24,color=B,zorder=3)
for q,a_,b_ in zip(pubs,tm,mp):
    if abs(a_-b_)>1.4: ax.annotate(q,(a_,b_),xytext=(4,-2),textcoords='offset points',fontsize=7)
ax.set_xlabel(f'Fold training mean ΔSTY ({U})'); ax.set_ylabel(f'Mean Ridge prediction ({U})'); ax.set_title('(a) Fold level',loc='left',fontsize=9)
ax=axs[1]; o=np.argsort(y); ax.scatter(np.arange(96),y[o],s=10,color=GR,label='Observed'); ax.scatter(np.arange(96),rid[o],s=10,color=B,label='Ridge (held-out)')
ax.set_xlabel('Comparisons, ordered by observed ΔSTY'); ax.set_ylabel(f'ΔSTY ({U})'); ax.legend(frameon=False,fontsize=7.5); ax.set_title(f"(b) SD: predictions {R['S6']['sd_pred']:.1f}, observed {np.std(y):.1f}",loc='left',fontsize=9)
fig.tight_layout(); fig.savefig(F+'6.png'); plt.close()
# S7
c7=R['S7']; fig,ax=plt.subplots(figsize=(7.2,3.1)); x=np.arange(len(pubs))
ax.bar(x-0.2,[c7[q]['nested'] for q in pubs],0.38,color=B,label='Nested α (fold-specific)'); ax.bar(x+0.2,[c7[q]['a1'] for q in pubs],0.38,color=O,label='α = 1')
ax.set_xticks(x); ax.set_xticklabels([f"{q}\nα={c7[q]['alpha']:.0e}".replace('e+0','e') for q in pubs],fontsize=7); ax.set_ylabel(f'LI coefficient\n({U} per W cm$^{{-2}}$)'); grid(ax); ax.legend(frameon=False,fontsize=7.5)
fig.tight_layout(); fig.savefig(F+'7.png'); plt.close()
# S8
s8=pd.DataFrame({'Ridge':R['S8_ridge'],'Random Forest':R['S8_rf']}).loc[pubs]*100
n=pd.Series(g).value_counts().loc[pubs]/96*100
fig,ax=plt.subplots(figsize=(7.2,3.1)); ax.bar(x-0.27,n,0.26,color=GR,label='Share of comparisons'); ax.bar(x,s8['Ridge'],0.26,color=B,label='Share of Ridge absolute error'); ax.bar(x+0.27,s8['Random Forest'],0.26,color=O,label='Share of RF absolute error')
ax.set_xticks(x); ax.set_xticklabels(pubs); ax.set_ylabel('%'); grid(ax); ax.legend(frameon=False,fontsize=7.5)
fig.tight_layout(); fig.savefig(F+'8.png'); plt.close()
# S9
X=P.copy(); X['ratio']=X.dSTY/X.LI
fig,ax=plt.subplots(figsize=(7.2,3.4))
for i,q in enumerate(pubs):
    v=X.ratio[X.pub==q]; ax.scatter(np.full(len(v),i)+np.linspace(-0.12,0.12,len(v)),v,s=10,color='#b7d3f6',zorder=2)
    ax.plot([i-0.25,i+0.25],[v.median()]*2,color=B,lw=2,zorder=3)
s9=R['S9']; ax.axhspan(s9['coef']-1.96*s9['se'],s9['coef']+1.96*s9['se'],color='#fde3d6',zorder=1); ax.axhline(s9['coef'],color=O,lw=1.2,label=f"Mixed-effects pooled LI effect {s9['coef']:.1f} ± {s9['se']:.1f} (95% band)")
ax.axhline(0,color='#52514e',lw=0.6); ax.set_yscale('symlog',linthresh=10); ax.set_xticks(range(len(pubs))); ax.set_xticklabels(pubs)
ax.set_ylabel(f'ΔSTY / LI\n({U} per W cm$^{{-2}}$)'); ax.legend(frameon=False,fontsize=7.5,loc='upper left'); grid(ax)
fig.tight_layout(); fig.savefig(F+'9.png'); plt.close()
# S10
fig,axs=plt.subplots(1,2,figsize=(7.2,3.2))
for ax,s,t in zip(axs,R['S8tab'],'ab'):
    d=np.array(s['data']); o=np.argsort(d[:,0])
    ax.plot(d[o,0],d[o,1],'o-',color=GR,ms=5,lw=1.2,label=f"Observed (slope {s['slope']:.2f})")
    ax.plot(d[o,0],d[o,2],'s-',color=B,ms=4,lw=1.2,label=f"Ridge (slope {s['ridge']:.2f})"); ax.plot(d[o,0],d[o,3],'^-',color=O,ms=4,lw=1.2,label=f"RF (slope {s['rf']:.2f})")
    ax.axhline(0,color='#cfcfcb',lw=0.6); ax.set_xlabel('LI (W cm$^{-2}$)'); ax.set_title(f"({t}) {s['pub']}, {s['cat'].replace('2','$_2$')}, {s['T']:.0f} °C",loc='left',fontsize=9); ax.legend(frameon=False,fontsize=7,loc='center right' if t=='b' else 'center left')
axs[0].set_ylabel(f'ΔSTY ({U})'); fig.tight_layout(); fig.savefig(F+'10.png'); plt.close()
# S11 Table 5 values
cfg=['One-hot','Physicochemical','One-hot + dark STY','Physicochemical + dark STY']
T5n={'Ridge (extended α grid)':'Ridge (extended α)','L1 median regression':'L1 median regression','Random Forest, squared error':'RF, squared error','Random Forest, absolute error':'RF, absolute error','Gradient boosting, absolute error':'GB, absolute error'}
T5={T5n[m]:([d[c][0] for c in ['one-hot','physicochemical','one-hot + dark STY','physicochemical + dark STY']],[d[c][1] for c in ['one-hot','physicochemical','one-hot + dark STY','physicochemical + dark STY']]) for m,d in T5J['table5'].items()}
fig,ax=plt.subplots(figsize=(7.2,3.6)); x=np.arange(len(T5)); w=0.2
for j,(c,cl) in enumerate(zip(cfg,[B,O,A,V])):
    ax.bar(x+(j-1.5)*w,[T5[k][0][j] for k in T5],w-0.02,yerr=[T5[k][1][j] for k in T5],color=cl,label=c,error_kw=dict(lw=0.7,capsize=1.5))
ax.axhline(T5J['baselines']['zero'],color='#0b0b0b',lw=0.9,ls='--'); ax.text(-0.48,25.0,f"predict-zero {T5J['baselines']['zero']:.2f}",fontsize=7,ha='left',va='top',bbox=dict(fc='white',ec='none',pad=0.5))
ax.axhline(T5J['baselines']['median'],color='#52514e',lw=0.9,ls=':'); ax.text(-0.48,25.9,f"training median {T5J['baselines']['median']:.2f}",fontsize=7,ha='left',va='bottom',bbox=dict(fc='white',ec='none',pad=0.5))
ax.set_xticks(x); ax.set_xticklabels(list(T5),fontsize=8); ax.set_ylim(18,40); ax.set_xlim(-0.55,4.45); ax.set_ylabel(f'LOPO MAE ({U})'); grid(ax); ax.legend(frameon=False,fontsize=7.5,ncol=4,loc='upper center',bbox_to_anchor=(0.5,1.12))
fig.tight_layout(); fig.savefig(F+'11.png'); plt.close()
print('ok')
