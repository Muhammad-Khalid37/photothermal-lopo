"""Supplementary analyses: predictor clustering (Fig. S3, Section S3), recipe novelty (Figs. S4-S5, Table S7),
Ridge diagnostics (Figs. S6-S8), intensity series (Table S8, Fig. S10), mixed-effects models (Fig. S9,
Table S10), grouped ablation (Table S6), robustness sets (Tables S5, S11, S16), Table S15 extras."""
from _common import *
import warnings
import statsmodels.formula.api as smf
from scipy import stats
from joblib import Parallel, delayed
from sklearn.model_selection import LeaveOneGroupOut
from ptml.validation import lopo_constant, lopo_predict, nested_ridge
from ptml.models import random_forest, ridge
from ptml.stats import mae, r2
warnings.filterwarnings("ignore")
args()
P, REL = load_pairs()
y, g = P.dSTY.values, P.pub.values
pubs = sorted(set(g)); rng = np.random.default_rng(C.SEED)
rid = np.load(C.RESULTS / "abs_ridge_ext_preds.npy"); rfS = np.load(C.RESULTS / "abs_rf_preds.npy"); rf = rfS.mean(0)
alpha = load_json("02_absolute_lopo.json")["alpha_extended"]
O = {}

# --- clustering of predictors and response (Fig. S3, Section S3)
def eta2(x, gg):
    gm = pd.Series(x).groupby(gg).transform("mean").values
    return float(np.sum((gm - x.mean()) ** 2) / np.sum((x - x.mean()) ** 2))
def icc1(v, gg):
    d = pd.DataFrame(dict(y=v, g=gg)); k = d.g.nunique(); N = len(d); n = d.groupby("g").size()
    gm = d.groupby("g").y.transform("mean"); msb = np.sum((gm - d.y.mean()) ** 2) / (k - 1); msw = np.sum((d.y - gm) ** 2) / (N - k)
    n0 = (N - np.sum(n ** 2) / N) / (k - 1); return float((msb - msw) / (msb + (n0 - 1) * msw))
E2 = {}
for c in ["LI", "MS", "GHSV", "RP", "H2CO2", "T", "AC", "PC"]:
    m = P[c].notna().values; x = P[c].values[m].astype(float); gg = g[m]; e = eta2(x, gg)
    nperm = 20000 if c == "AC" else 2000
    null = np.array([eta2(x, rng.permutation(gg)) for _ in range(nperm)])
    E2[c] = dict(eta2=e, p=float((1 + (null >= e).sum()) / (nperm + 1)), n=int(m.sum()))
rk = stats.rankdata(y)
O["clustering"] = dict(eta2=E2, icc1=icc1(y, g), icc1_rank=icc1(rk, g), eta2_dSTY=eta2(y, g), eta2_rank=eta2(rk, g),
                       heterogeneity_p=float((1 + sum(eta2(y, rng.permutation(g)) >= eta2(y, g) for _ in range(2000))) / 2001))

# --- Ridge diagnostics (Figs. S6-S8)
O["ridge_vs_training_mean"] = {q: dict(mean_pred=float(rid[g == q].mean()), train_mean=float(y[g != q].mean())) for q in pubs}
coef = {}
for q in pubs:
    tr = g != q; out = {"alpha": alpha[q]}
    for lab, a in (("nested", alpha[q]), ("alpha_1", 1.0)):
        mdl = ridge(a).fit(P[tr], y[tr]); names = list(mdl[0].get_feature_names_out())
        out[lab] = float(mdl[1].coef_[names.index("num__LI")] / mdl[0].named_transformers_["num"].named_steps["scale"].scale_[C.NUM.index("LI")])
    coef[q] = out
O["ridge_LI_coefficients"] = coef
O["error_share"] = {nm: (pd.Series(np.abs(y - p)).groupby(g).sum() / np.abs(y - p).sum()).to_dict() for nm, p in (("ridge", rid), ("rf", rf))}

# --- recipe novelty (Table S7)
rec = P.support + "|" + P.active + "|" + P.promoter.fillna("-"); sm = P.support + "|" + P.active
seen = lambda col: np.array([col[i] in set(col[g != g[i]]) for i in range(len(P))])
s_s, s_a, s_p, s_r = seen(P.support), seen(P.active), seen(sm), seen(rec)
med, mean = lopo_constant(y, g, "median"), lopo_constant(y, g, "mean")
def sub(m): return dict(n=int(m.sum()), zero=mae(y[m], 0 * y[m]), median=mae(y[m], med[m]), mean=mae(y[m], mean[m]),
                        ridge=mae(y[m], rid[m]), rf=float(np.mean([mae(y[m], p[m]) for p in rfS])), rf_sd=float(np.std([mae(y[m], p[m]) for p in rfS], ddof=1)))
O["novelty"] = dict(n_recipes=int(rec.nunique()), recipes_in_several_pubs=int((P.assign(r=rec).groupby("r").pub.nunique() > 1).sum()),
                    seen=dict(support=int(s_s.sum()), active=int(s_a.sum()), both=int((s_s & s_a).sum()), pair=int(s_p.sum()), recipe=int(s_r.sum())),
                    table=dict(all=sub(np.ones(len(y), bool)), both_seen=sub(s_s & s_a), not_both=sub(~(s_s & s_a)), pair_seen=sub(s_p), pair_unseen=sub(~s_p)),
                    per_publication=pd.DataFrame(dict(pub=g, support=s_s, active=s_a, pair=s_p, recipe=s_r)).groupby("pub").mean().to_dict("index"))

# --- intensity series (Table S8, Fig. S10)
ser = []
for (q, s_, a_, p_, T), d in P.assign(pr=P.promoter.fillna("-")).groupby(["pub", "support", "active", "pr", "T"]):
    if d.LI.nunique() >= 3:
        lr = stats.linregress(d.LI, d.dSTY); t = stats.t.ppf(.975, len(d) - 2); i = d.index.values
        ser.append(dict(pub=q, catalyst=f"{a_}-{p_}/{s_}" if p_ != "-" else f"{a_}/{s_}", T=T, n=len(d), slope=lr.slope,
                        lo=lr.slope - t * lr.stderr, hi=lr.slope + t * lr.stderr, ridge=stats.linregress(d.LI, rid[i]).slope,
                        rf=stats.linregress(d.LI, rf[i]).slope, rho_ridge=stats.spearmanr(d.LI, rid[i])[0], rho_rf=stats.spearmanr(d.LI, rf[i])[0],
                        data=np.c_[d.LI, d.dSTY, rid[i], rf[i]].tolist()))
O["intensity_series"] = ser

# --- mixed-effects models (Fig. S9, Table S10)
X = P.copy(); X["ldark"] = np.log1p(X.STY_dark)
slog = lambda v: np.sign(v) * np.log1p(np.abs(v)); islog = lambda v: np.sign(v) * np.expm1(np.abs(v))
m9 = smf.mixedlm("dSTY ~ LI", X, groups=X.pub).fit(reml=True)
O["intensity_mixed"] = dict(coef=m9.params["LI"], se=m9.bse["LI"], p=m9.pvalues["LI"],
                            per_pub_median=(X.dSTY / X.LI).groupby(X.pub).median().to_dict())
Z = X.copy(); Z["sy"] = slog(y)
for c in C.NUM + ["ldark"]:
    Z[c] = X[c].fillna(X[c].median()); Z[c] = (Z[c] - Z[c].mean()) / Z[c].std()
specs = {"intercept only": "1", "LI": "LI", "LI + T": "LI + T", "LI + T + log(1+dark)": "LI + T + ldark"}
mixed = {}
for scale, resp in (("raw", "dSTY"), ("slog", "sy")):
    for nm, f in specs.items():
        fit = smf.mixedlm(f"{resp} ~ {f}", Z, groups=Z.pub).fit(reml=True); p = np.empty(len(y))
        for tr, te in LeaveOneGroupOut().split(Z, y, g):
            p[te] = smf.mixedlm(f"{resp} ~ {f}", Z.iloc[tr], groups=Z.pub.iloc[tr]).fit(reml=True).predict(Z.iloc[te])
        p = islog(p) if scale == "slog" else p
        mixed[f"{scale} | {nm}"] = dict(MAE=mae(y, p), R2=r2(y, p), su=float(fit.cov_re.iloc[0, 0]), se=float(fit.scale),
                                       fe={k: [float(fit.params[k]), float(fit.bse[k])] for k in fit.params.index if k != "Group Var"})
O["mixed_models"] = mixed

# --- grouped ablation and robustness sets (Tables S5, S6, S11, S16)
def evalset(D, num=None, cat=None):
    yy, gg = D.dSTY.values, D.pub.values; num = C.NUM if num is None else num; cat = C.CAT if cat is None else cat
    pr = np.empty(len(yy))
    for tr, te in LeaveOneGroupOut().split(D, yy, gg):
        sc = [np.mean((yy[tr] - lopo_predict(lambda a=a: ridge(a, num=num, cat=cat), D.iloc[tr], yy[tr], gg[tr])) ** 2) for a in C.RIDGE_GRID_EXT]
        a = C.RIDGE_GRID_EXT[int(np.argmin(sc))]; pr[te] = ridge(a, num=num, cat=cat).fit(D.iloc[tr], yy[tr]).predict(D.iloc[te])
    rfp = np.array(Parallel(n_jobs=-1)(delayed(lopo_predict)(lambda s=s: random_forest(s, num=num, cat=cat), D, yy, gg) for s in range(C.N_SEEDS)))
    ms = [mae(yy, q) for q in rfp]
    return dict(n=len(yy), npub=int(len(set(gg))), zero=mae(yy, 0 * yy), median=mae(yy, lopo_constant(yy, gg, "median")),
                mean=mae(yy, lopo_constant(yy, gg, "mean")), ridge=mae(yy, pr), rf=float(np.mean(ms)), rf_sd=float(np.std(ms, ddof=1))), pr, rfp
full, r0, f0 = evalset(P); f0m = np.array([mae(y, q) for q in f0]); n12 = g != "P12"
abl = {"full": full}
groups = {"catalyst descriptors": (["AC", "PC"], C.CAT), "light intensity": (["LI"], []), "temperature": (["T"], []),
          "reaction conditions": (["MS", "GHSV", "H2CO2", "RP"], [])}
for k, (nn, cc) in groups.items():
    r, pr, fp = evalset(P, [c for c in C.NUM if c not in nn], [c for c in C.CAT if c not in cc])
    d = np.array([mae(y, q) for q in fp]) - f0m
    r.update(rf_delta=float(d.mean()), rf_delta_sd=float(d.std(ddof=1)), rf_lower_seeds=int((d < 0).sum()),
             ridge_delta=r["ridge"] - full["ridge"], ridge_delta_noP12=mae(y[n12], pr[n12]) - mae(y[n12], r0[n12]))
    abl[k] = r
abl["catalyst only"] = evalset(P, ["AC", "PC"], C.CAT)[0]; abl["intensity only"] = evalset(P, ["LI"], [])[0]
O["ablation"] = abl
D15 = ["support", "active", "AC", "promoter", "PC", "CT[℃]", "CD[h]", "RT[℃]", "RD [h]", "MS", "GHSV", "H2CO2", "IGR[%]", "RP", "T"]
unver = P[D15].isna().any(axis=1).values
keep = P.sort_values(["dark_record_id", "LI", "light_record_id"], ascending=[True, False, True]).drop_duplicates("dark_record_id").index.sort_values()
sets = {"without_P02": P[g != "P02"], "P02_P06_merged": P.assign(pub=np.where(g == "P06", "P02", g)),
        "without_identical_rows": P[~((P.dSTY == 0) & (P.STY_dark > 0))], "active_only": P[~((P.STY_dark == 0) & (P.STY_light == 0))],
        "verified_80": P[~unver], "one_pair_per_dark_reference": P.loc[keep]}
O["robustness"] = {k: evalset(D.reset_index(drop=True))[0] for k, D in sets.items()}
O["unverifiable_pairs"] = P.loc[unver, ["pub", "dark_record_id", "light_record_id"]].to_dict("records")

# --- Table S15 extras (relative enhancement)
yr, gr = REL.log2r.values, REL.pub.values; ld = np.log1p(REL.STY_dark.values); T = REL["T"].values
cp = np.empty(len(yr))
for tr, te in LeaveOneGroupOut().split(REL, yr, gr):
    c = np.median((REL.STY_light - REL.STY_dark).values[tr]); d = REL.STY_dark.values[te]; cp[te] = np.log2((d + c) / d)
res = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)
Ts = (T - T.mean()) / T.std()
def within(idx):
    d = pd.DataFrame(dict(y=yr[idx], t=Ts[idx], l=ld[idx], g=gr[idx]))
    for c in "ytl": d[c] = d[c] - d.groupby("g")[c].transform("mean")
    return np.linalg.lstsq(np.c_[d.t, d.l], d.y, rcond=None)[0][0]
rp = np.unique(gr); bs = [within(np.concatenate([np.where(gr == q)[0] for q in rng.choice(rp, len(rp))])) for _ in range(C.N_BOOT)]
da = pd.DataFrame(dict(y=y, t=P["T"].values, g=g))
for c in "yt": da[c] = da[c] - da.groupby("g")[c].transform("mean")
O["table_s15"] = dict(additive_light=mae(yr, cp), median=mae(yr, lopo_constant(yr, gr, "median")),
                      partial_spearman_T_given_dark=float(np.corrcoef(res(stats.rankdata(yr), stats.rankdata(ld)), res(stats.rankdata(T), stats.rankdata(ld)))[0, 1]),
                      within_T=dict(b=float(within(np.arange(len(yr)))), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5))),
                      absolute_within_T_slope=float(np.sum(da.t * da.y) / np.sum(da.t ** 2)))
save_json(O, "11_supplementary_analyses.json")
print(json.dumps({k: O[k] for k in ["clustering", "table_s15"]}, indent=1, default=float)[:3000])
