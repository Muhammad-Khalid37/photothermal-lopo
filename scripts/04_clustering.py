"""Publication-level clustering: REML ICC, unequal-cluster design effect, power (Sec. 3.3.2; comment 13)."""
from _common import *
from ptml.stats import icc_and_design_effect

args()
P, _ = load_pairs()
out = icc_and_design_effect(P.dSTY.values, P.pub.values)
f = C.RESULTS / "02_absolute_lopo.json"
if f.exists():
    se = load_json("02_absolute_lopo.json")["contrasts"]["ridge_extended-minus-median"]["obs"]["se"]
    out["bootstrap_se_ridge_minus_median"] = se
    out["minimum_detectable_difference_80pct_power"] = 2.8 * se
print(json.dumps(out, indent=1))
save_json(out, "04_clustering.json")
