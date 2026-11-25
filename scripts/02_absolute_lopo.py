"""Pre-specified primary analysis: LOPO prediction of dSTY (Tables 2, 3; Sec. 3.2, 3.3.4)."""
from _common import *
from ptml.validation import lopo_constant, nested_ridge, seed_average, lopo_predict
from ptml.models import random_forest, ridge
from ptml.stats import metrics, cluster_bootstrap_diff, mae
from sklearn.model_selection import LeaveOneGroupOut

args()
P, _ = load_pairs()
y, g = P.dSTY.values, P.pub.values
base = {k: lopo_constant(y, g, k) for k in ["zero", "median", "mean"]}
p_ext, a_ext = nested_ridge(P, y, g, C.RIDGE_GRID_EXT)
p_ext_mae, a_ext_mae = nested_ridge(P, y, g, C.RIDGE_GRID_EXT, inner="mae")
p_orig, a_orig = nested_ridge(P, y, g, C.RIDGE_GRID_ORIG)
p_inf = lopo_predict(lambda: ridge(1e6), P, y, g)
rf = seed_average(lambda s: random_forest(s), P, y, g)
np.save(C.RESULTS / "abs_rf_preds.npy", rf); np.save(C.RESULTS / "abs_ridge_ext_preds.npy", p_ext)

table3 = {k: metrics(y, v) for k, v in base.items()}
table3["ridge_extended"] = metrics(y, p_ext); table3["ridge_original"] = metrics(y, p_orig)
m = pd.DataFrame([metrics(y, p) for p in rf]); table3["random_forest"] = {c: [m[c].mean(), m[c].std()] for c in m}

err = lambda p: pd.Series(np.abs(y - p)).groupby(g).mean()
rf_pp = pd.DataFrame([err(p) for p in rf])
table2 = pd.DataFrame({"n": P.groupby("pub").size(), "median_dSTY": P.groupby("pub").dSTY.median(),
                       "MAE_mean": err(base["mean"]), "MAE_ridge_ext": err(p_ext),
                       "MAE_rf_mean": rf_pp.mean(), "MAE_rf_sd": rf_pp.std()})
table2.round(2).to_csv(C.RESULTS / "table2_per_publication.csv")

contrasts = {}
for name, p in [("ridge_extended", p_ext), ("ridge_original", p_orig)]:
    for k, b in base.items():
        contrasts[f"{name}-minus-{k}"] = dict(obs=cluster_bootstrap_diff(y - p, y - b, g),
                                              balanced=cluster_bootstrap_diff(y - p, y - b, g, balanced=True))

# Ridge diagnostics (Sec. 3.3.4)
dev, li_coef = {}, []
for tr, te in LeaveOneGroupOut().split(P, y, g):
    q = g[te][0]; dev[q] = float(p_ext[te].mean() - y[tr].mean())
    mdl = ridge(a_ext[q]).fit(P.iloc[tr], y[tr])
    names = list(mdl[0].get_feature_names_out()); sc = mdl[0].named_transformers_["num"].named_steps["scale"].scale_
    li_coef.append(float(mdl[1].coef_[names.index("num__LI")] / sc[C.NUM.index("LI")]))
share = pd.Series(np.abs(y - p_ext)).groupby(g).sum(); share = (share / share.sum() * 100).round(1)

# complete-descriptor subset (Table S5)
D15 = ["support", "active", "AC", "promoter", "PC", "CT[℃]", "CD[h]", "RT[℃]", "RD [h]", "MS", "GHSV", "H2CO2", "IGR[%]", "RP", "T"]
Q = P[P[D15].notna().all(axis=1)].reset_index(drop=True); yq, gq = Q.dSTY.values, Q.pub.values
complete = dict(n=len(Q), mean=mae(yq, lopo_constant(yq, gq, "mean")), median=mae(yq, lopo_constant(yq, gq, "median")),
                zero=mae(yq, 0 * yq), ridge=mae(yq, nested_ridge(Q, yq, gq, C.RIDGE_GRID_EXT)[0]),
                rf=float(np.mean([mae(yq, p) for p in seed_average(lambda s: random_forest(s), Q, yq, gq)])))

out = dict(table3=table3, contrasts=contrasts, alpha_extended=a_ext, alpha_extended_mae_inner=a_ext_mae,
           ridge_extended_mae_inner=mae(y, p_ext_mae), alpha_original=a_orig, ridge_fixed_1e6=mae(y, p_inf),
           ridge_diag=dict(pred_sd=float(np.std(p_ext)), obs_sd=float(np.std(y)),
                           folds_within_1p4=int(sum(abs(v) <= 1.4 for v in dev.values())), fold_dev=dev, LI_coef=li_coef,
                           error_share=share.to_dict()),
           complete_descriptor_subset=complete)
print(pd.DataFrame({k: {m: (v[m][0] if isinstance(v[m], list) else v[m]) for m in ["MAE", "R2"]} for k, v in table3.items()}).T.round(3))
save_json(out, "02_absolute_lopo.json")
