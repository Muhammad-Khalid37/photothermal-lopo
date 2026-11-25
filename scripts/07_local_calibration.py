"""Local calibration with fold-internal kappa (Sec. 3.7, Fig. 9a,b, Table S14; comments 9, 20)."""
from _common import *
from ptml.calibration import local_calibration, summarize
from ptml.stats import reml_components

args()
P, _ = load_pairs()
y, g = P.dSTY.values, P.pub.values
T, kappa = local_calibration(y, g)
n_j = pd.Series(g).value_counts()
common = list(n_j[n_j >= max(C.K_LIST) + 1].index)
S = summarize(T, common, n_boot=2000 if C.N_REVEAL < 100 else 10000)
T.to_csv(C.RESULTS / "calibration_per_publication.csv", index=False)
S.to_csv(C.RESULTS / "calibration_summary.csv", index=False)
su, se = reml_components(y, g)
print(S[S.method.isin(["shrink", "revmed"])].round(2).to_string(index=False))
save_json(dict(kappa_folds=kappa, kappa_full=se / su, common_publications=common), "07_local_calibration.json")
