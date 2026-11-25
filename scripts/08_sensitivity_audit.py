"""Re-analysis restricted to thermally comparable pairs from the primary-paper audit (Sec. 3.5, Table S3c)."""
from _common import *
from ptml.data import load_audit_classes
from ptml.validation import lopo_constant, nested_ridge, seed_average
from ptml.models import random_forest, gradient_boosting
from ptml.stats import mae, r2, cluster_bootstrap_diff
from ptml.thermal import lopo_thermal

args()
P, R = load_pairs()
P["tclass"] = load_audit_classes(P); R["tclass"] = load_audit_classes(R)
P[["pub", "DOI_corrected", "support", "active", "T", "LI", "dark_record_id", "light_record_id", "tclass"]].to_csv(
    C.RESULTS / "pair_classes.csv", index=False)
out = dict(counts=P.tclass.value_counts().to_dict())
for name, keep in [("A_confirmed", ["A"]), ("A_all", ["A", "A*"])]:
    D = P[P.tclass.isin(keep)].reset_index(drop=True); y, g = D.dSTY.values, D.pub.values
    r = dict(n_pairs=len(D), n_pubs=int(D.pub.nunique()))
    for k in ["zero", "median", "mean"]:
        r[k] = mae(y, lopo_constant(y, g, k))
    r["ridge"] = mae(y, nested_ridge(D, y, g, C.RIDGE_GRID_EXT)[0])
    r["rf"] = float(np.mean([mae(y, p) for p in seed_average(lambda s: random_forest(s), D, y, g)]))
    Q = R[R.tclass.isin(keep)].reset_index(drop=True); yr, gr = Q.log2r.values, Q.pub.values
    med = lopo_constant(yr, gr, "median")
    gb = seed_average(lambda s: gradient_boosting(s), Q, yr, gr).mean(0)
    to = seed_average(lambda s: gradient_boosting(s, num=["T"], cat=[]), Q, yr, gr).mean(0)
    th, _ = lopo_thermal(Q["T"].values, yr, gr, 1)
    r.update(rel_n=len(Q), rel_pubs=int(Q.pub.nunique()), rel_median=mae(yr, med), rel_gb=mae(yr, gb),
             rel_tonly=mae(yr, to), rel_thermal=mae(yr, th),
             gb_minus_median=cluster_bootstrap_diff(yr - gb, yr - med, gr),
             thermal_minus_median=cluster_bootstrap_diff(yr - th, yr - med, gr),
             thermal_minus_gb=cluster_bootstrap_diff(yr - th, yr - gb, gr))
    out[name] = r
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, dict)})
save_json(out, "08_sensitivity_audit.json")
