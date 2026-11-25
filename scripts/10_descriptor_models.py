"""Alternative models and descriptor sets for the absolute response (Table 5, Figure S11), physicochemical
rows of Table 6, selection-adjusted bootstrap over the 20 configurations, +/-20 % descriptor perturbation
and verifiable binary descriptors (Table S16, Section S8)."""
from _common import *
from scipy.stats import spearmanr
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from ptml.validation import lopo_constant, lopo_predict
from ptml.models import preprocessor, random_forest, gradient_boosting, ridge
from ptml.stats import mae, r2, cluster_bootstrap_diff, selection_adjusted_bootstrap
from ptml.descriptors import add_descriptors, random_factors, PHYS, BINARY

args()
N_SMALL = 5 if C.N_SEEDS >= 5 else C.N_SEEDS     # seeds for RF (absolute error) and GB in Table 5
N_PERT = 100 if C.N_SEEDS >= 5 else 5
P, R = load_pairs()
P = add_descriptors(P); P["ldark"] = np.log1p(P.STY_dark)
y, g = P.dSTY.values, P.pub.values

SETS = {"one-hot": (C.NUM, C.CAT), "physicochemical": (C.NUM + PHYS, []),
        "one-hot + dark STY": (C.NUM + ["ldark"], C.CAT), "physicochemical + dark STY": (C.NUM + PHYS + ["ldark"], [])}


def nested_ridge_set(X, y, g, num, cat, inner="mae"):
    pred = np.empty(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        sc = []
        for a in C.RIDGE_GRID_EXT:
            pi = lopo_predict(lambda a=a: ridge(a, num=num, cat=cat), X.iloc[tr], y[tr], g[tr])
            sc.append(np.mean(np.abs(y[tr] - pi)) if inner == "mae" else np.mean((y[tr] - pi) ** 2))
        a = C.RIDGE_GRID_EXT[int(np.argmin(sc))]
        pred[te] = ridge(a, num=num, cat=cat).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    return pred


L1_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def l1(alpha, num, cat):
    return make_pipeline(preprocessor(scale=True, num=num, cat=cat), QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs"))


def nested_l1(X, y, g, num, cat):
    """L1 median regression with the penalty chosen by inner LOPO (absolute error) on the training publications."""
    pred, chosen = np.empty(len(y)), []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        sc = [np.mean(np.abs(y[tr] - lopo_predict(lambda a=a: l1(a, num, cat), X.iloc[tr], y[tr], g[tr]))) for a in L1_GRID]
        a = L1_GRID[int(np.argmin(sc))]; chosen.append(a)
        pred[te] = l1(a, num, cat).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    return pred, chosen


def rf_abs(seed, num, cat):
    return make_pipeline(preprocessor(num=num, cat=cat),
                         RandomForestRegressor(random_state=seed, n_jobs=1, criterion="absolute_error", **C.RF_PARAMS))


def seeds(factory, X, y, g, n):
    return np.array(Parallel(n_jobs=-1)(delayed(lopo_predict)(lambda s=s: factory(s), X, y, g) for s in range(n)))


preds, table5 = {}, {}
for sname, (num, cat) in SETS.items():
    print("descriptor set:", sname, flush=True)
    runs = {"Ridge (extended α grid)": [nested_ridge_set(P, y, g, num, cat)],
            "L1 median regression": [nested_l1(P, y, g, num, cat)[0]],
            "Random Forest, squared error": list(seeds(lambda s: random_forest(s, num=num, cat=cat), P, y, g, C.N_SEEDS)),
            "Random Forest, absolute error": list(seeds(lambda s: rf_abs(s, num, cat), P, y, g, N_SMALL)),
            "Gradient boosting, absolute error": list(seeds(lambda s: gradient_boosting(s, num=num, cat=cat), P, y, g, N_SMALL))}
    for model, ps in runs.items():
        ms = [mae(y, p) for p in ps]
        table5.setdefault(model, {})[sname] = [float(np.mean(ms)), float(np.std(ms, ddof=1)) if len(ms) > 1 else 0.0]
        preds[f"{model} | {sname}"] = np.mean(ps, 0)
        if model == "Gradient boosting, absolute error":
            preds[f"_seed0 | {sname}"] = ps[0]
med = lopo_constant(y, g, "median")
best = min(preds, key=lambda k: mae(y, preds[k]) if not k.startswith("_") else np.inf)
best_seed0 = preds["_seed0 | " + best.split(" | ")[1]] if best.startswith("Gradient") else preds[best]
cands = [k for k in preds if not k.startswith("_")]
sel = selection_adjusted_bootstrap(preds, y, g, med, cands, n_boot=C.N_BOOT_SEL)
best_vs_median = cluster_bootstrap_diff(y - best_seed0, y - med, g)

# +/-20 % perturbation of the continuous descriptors for the best configuration (one seed per replicate)
bmodel, bset = best.split(" | ")
num_b, cat_b = SETS[bset]
facs = {"Gradient boosting, absolute error": lambda s, n, c: gradient_boosting(s, num=n, cat=c),
        "Random Forest, squared error": lambda s, n, c: random_forest(s, num=n, cat=c),
        "Random Forest, absolute error": rf_abs}
rng = np.random.default_rng(C.SEED)
def one_rep(r, f):
    Xp = add_descriptors(P.drop(columns=PHYS), factors=f); Xp["ldark"] = P.ldark
    return mae(y, lopo_predict(lambda: facs.get(bmodel, facs["Gradient boosting, absolute error"])(r, num_b, cat_b), Xp, y, g))
pert = Parallel(n_jobs=-1)(delayed(one_rep)(r, random_factors(rng)) for r in range(N_PERT)) if "physico" in bset else []

# verifiable binary descriptors in place of the continuous ones
binary = {}
for lab, num in [("binary", C.NUM + BINARY), ("binary + dark STY", C.NUM + BINARY + ["ldark"])]:
    ms = [mae(y, p) for p in seeds(lambda s: gradient_boosting(s, num=num, cat=[]), P, y, g, N_SMALL)]
    binary[f"Gradient boosting, {lab}"] = [float(np.mean(ms)), float(np.std(ms, ddof=1))]

# Table 6 rows: relative enhancement with physicochemical descriptors + dark yield
R = add_descriptors(R); R["ldark"] = np.log1p(R.STY_dark)
yr, gr = R.log2r.values, R.pub.values
medr = lopo_constant(yr, gr, "median")
rel = {}
for lab, fac in [("Random Forest, physicochemical + dark STY", lambda s: random_forest(s, num=C.NUM + PHYS + ["ldark"], cat=[])),
                 ("Gradient boosting, physicochemical + dark STY", lambda s: gradient_boosting(s, num=C.NUM + PHYS + ["ldark"], cat=[]))]:
    ps = seeds(fac, R, yr, gr, C.N_SEEDS); avg = ps.mean(0); ms = [mae(yr, p) for p in ps]; rs = [r2(yr, p) for p in ps]
    rel[lab] = dict(MAE=float(np.mean(ms)), MAE_sd=float(np.std(ms, ddof=1)), R2=float(np.mean(rs)), R2_sd=float(np.std(rs, ddof=1)),
                    rho=float(spearmanr(yr, avg)[0]), vs_median=cluster_bootstrap_diff(yr - avg, yr - medr, gr))

out = dict(table5=table5, baselines=dict(zero=mae(y, 0 * y), median=mae(y, med)), best=best, best_mae=mae(y, preds[best]),
           best_seed0_vs_median=best_vs_median, selection_adjusted=sel,
           perturbation=dict(n=len(pert), min=float(np.min(pert)) if pert else None, max=float(np.max(pert)) if pert else None,
                             mean=float(np.mean(pert)) if pert else None, model=bmodel, descriptors=bset),
           binary=binary, table6_physicochemical=rel,
           settings=dict(L1="QuantileRegressor(quantile=0.5, solver='highs'); alpha by inner LOPO over " + str(L1_GRID),
                         seeds_rf_squared=C.N_SEEDS, seeds_other=N_SMALL, perturbation_replicates=N_PERT))
pd.DataFrame({m: {s: f"{v[0]:.2f}" + (f" ± {v[1]:.2f}" if v[1] else "") for s, v in d.items()} for m, d in table5.items()}).T \
  .to_csv(C.RESULTS / "table5_alternative_models.csv")
save_json(out, "10_descriptor_models.json")
print(pd.read_csv(C.RESULTS / "table5_alternative_models.csv", index_col=0).to_string())
print("best:", best, mae(y, preds[best])); print("selection-adjusted:", sel); print("perturbation:", out["perturbation"])
print("binary:", binary); print("table 6:", {k: (round(v['MAE'], 3), round(v['R2'], 3), round(v['rho'], 3)) for k, v in rel.items()})
