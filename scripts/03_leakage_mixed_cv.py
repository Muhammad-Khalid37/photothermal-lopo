"""Mixed random 5-fold versus publication-held-out validation (Table 4, Fig. 5; comment 12)."""
from _common import *
from ptml.data import absolute_sty_records
from ptml.validation import mixed_cv, nested_ridge, lopo_constant, seed_average
from ptml.models import random_forest
from ptml.stats import mae, r2

args()
P, _ = load_pairs()
A = absolute_sty_records(P)
out = {}
for name, X, y in [("dSTY", P, P.dSTY.values), ("absolute_STY", A, A.STY.values)]:
    g = X.pub.values
    pr, _ = nested_ridge(X, y, g, C.RIDGE_GRID_EXT)
    rf = seed_average(lambda s: random_forest(s), X, y, g)
    out[name] = dict(n=len(X),
        mixed={m: mixed_cv(X, y, m) for m in ["median", "ridge", "rf"]},
        held_out=dict(median=[mae(y, lopo_constant(y, g, "median")), r2(y, lopo_constant(y, g, "median"))],
                      ridge=[mae(y, pr), r2(y, pr)],
                      rf=[float(np.mean([mae(y, p) for p in rf])), float(np.std([mae(y, p) for p in rf], ddof=1)),
                          float(np.mean([r2(y, p) for p in rf]))]))
    print(name, json.dumps(out[name], indent=0)[:600])
save_json(out, "03_mixed_vs_lopo.json")
