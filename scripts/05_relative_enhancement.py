"""Post hoc relative enhancement log2(STY_light/STY_dark): Table 6, thermal null model, sign test,
jackknife, selection-adjusted bootstrap, within-publication trends, feature ablation (Sec. 3.6-3.7)."""
from _common import *
from scipy.stats import spearmanr
from ptml.validation import lopo_constant, nested_ridge, seed_average
from ptml.models import random_forest, gradient_boosting
from ptml.stats import mae, r2, cluster_bootstrap_diff, sign_test, selection_adjusted_bootstrap
from ptml.thermal import lopo_thermal, fit_one_parameter, fit_two_parameter, fit_exact

args()
_, R = load_pairs()
X = R.copy(); X["ldark"] = np.log1p(X.STY_dark)
y, g, T = X.log2r.values, X.pub.values, X["T"].values
pr = {"Training median": [lopo_constant(y, g, "median")], "Training mean": [lopo_constant(y, g, "mean")],
      "Predict no enhancement (0)": [0 * y], "Ridge (extended grid)": [nested_ridge(X, y, g, C.RIDGE_GRID_EXT)[0]]}
pr["Random Forest, one-hot"] = list(seed_average(lambda s: random_forest(s), X, y, g))
pr["Gradient boosting, one-hot"] = list(seed_average(lambda s: gradient_boosting(s), X, y, g))
pr["Gradient boosting, temperature only"] = list(seed_average(lambda s: gradient_boosting(s, num=["T"], cat=[]), X, y, g))
pr["Gradient boosting, dark yield only"] = list(seed_average(lambda s: gradient_boosting(s, num=["ldark"], cat=[]), X, y, g))
pr["Gradient boosting, temperature + dark yield"] = list(seed_average(lambda s: gradient_boosting(s, num=["T", "ldark"], cat=[]), X, y, g))
pr["Gradient boosting, one-hot + dark STY"] = list(seed_average(lambda s: gradient_boosting(s, num=C.NUM + ["ldark"]), X, y, g))
pr["Random Forest, one-hot + dark STY"] = list(seed_average(lambda s: random_forest(s, num=C.NUM + ["ldark"]), X, y, g))
p1, th1 = lopo_thermal(T, y, g, 1); p2, _ = lopo_thermal(T, y, g, 2); px, thx = lopo_thermal(T, y, g, 3)
pr["Thermal Arrhenius/dT, 1 parameter"] = [p1]; pr["Thermal Arrhenius/dT + offset, 2 parameters"] = [p2]
pr["Thermal Arrhenius exact (Ea/R, dT)"] = [px]
avg = {k: np.mean(v, 0) for k, v in pr.items()}
np.save(C.RESULTS / "rel_preds.npy", {k: np.array(v) for k, v in pr.items()}, allow_pickle=True)

med, gb = avg["Training median"], avg["Gradient boosting, one-hot"]
neg, pos = y < 0, y > 0
rows = []
for k, v in pr.items():
    ms = [mae(y, p) for p in v]
    rows.append(dict(model=k, MAE=np.mean(ms), MAE_sd=np.std(ms, ddof=1) if len(v) > 1 else np.nan,
                     MAE_neg=np.mean([mae(y[neg], p[neg]) for p in v]), MAE_pos=np.mean([mae(y[pos], p[pos]) for p in v]),
                     R2=np.mean([r2(y, p) for p in v]), rho=spearmanr(y, avg[k])[0] if np.std(avg[k]) > 0 else np.nan,
                     **{f"vs_median_{a}": b for a, b in cluster_bootstrap_diff(y - avg[k], y - med, g).items()},
                     **{f"vs_GB_{a}": b for a, b in cluster_bootstrap_diff(y - avg[k], y - gb, g).items()}))
table6 = pd.DataFrame(rows); table6.to_csv(C.RESULTS / "table6_relative_enhancement.csv", index=False)
print(table6[["model", "MAE", "MAE_neg", "MAE_pos", "R2", "vs_median_diff", "vs_median_lo", "vs_median_hi"]].round(3).to_string(index=False))

pubs = np.unique(g)
jk = [mae(y[g != q], gb[g != q]) - mae(y[g != q], med[g != q]) for q in pubs]
m2 = ~np.isin(g, ["P02", "P06"])
within = [dict(pub=q, n=int((g == q).sum()), levels=int(d["T"].nunique()), rho=float(spearmanr(d["T"], d.log2r)[0]))
          for q, d in R.groupby("pub") if d["T"].nunique() >= 3]
cands5 = ["Gradient boosting, one-hot", "Random Forest, one-hot", "Gradient boosting, one-hot + dark STY",
          "Random Forest, one-hot + dark STY", "Ridge (extended grid)"]
cands8 = cands5 + ["Gradient boosting, temperature only", "Thermal Arrhenius/dT, 1 parameter", "Thermal Arrhenius/dT + offset, 2 parameters"]

ablation = {}
groups = {"temperature": ["T"], "operating conditions": ["T", "GHSV", "H2CO2", "RP"], "light intensity": ["LI"],
          "catalyst mass": ["MS"], "catalyst descriptors": ["AC", "PC"] + C.CAT}
for k, cols in groups.items():
    num = [c for c in C.NUM if c not in cols]; cat = [c for c in C.CAT if c not in cols]
    ms = [mae(y, p) for p in seed_average(lambda s: gradient_boosting(s, num=num, cat=cat), X, y, g)]
    ablation[k] = [float(np.mean(ms)), float(np.std(ms, ddof=1))]

out = dict(counts=dict(n=len(y), positive=int(pos.sum()), negative=int(neg.sum()), zero=int((y == 0).sum()),
                       pooled_spearman_T=float(spearmanr(T, y)[0])),
           sign_test_GB=sign_test(y - gb, y - med, g), sign_test_thermal=sign_test(y - p1, y - med, g),
           jackknife=dict(range=[min(jk), max(jk)], without_P02_P06=[mae(y[m2], gb[m2]), mae(y[m2], med[m2])]),
           selection_adjusted=dict(five=selection_adjusted_bootstrap(avg, y, g, med, cands5),
                                   eight=selection_adjusted_bootstrap(avg, y, g, med, cands8)),
           thermal=dict(theta_all=fit_one_parameter(T, y), theta_folds=th1, two_param_all=list(fit_two_parameter(T, y)),
                        exact_all=list(fit_exact(T, y)), Ea_dT_kJ_per_mol_K=fit_one_parameter(T, y) * 8.314 / 1000),
           negative_pair_predictions_GB=[float(gb[neg].min()), float(gb[neg].max())],
           within_publication=within, ablation=ablation)
save_json(out, "05_relative_enhancement.json")
