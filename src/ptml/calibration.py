"""Local calibration: revealing k pairs of a held-out publication (Sec. 3.7, Fig. 9)."""
import numpy as np
import pandas as pd
from . import config as C
from .stats import reml_components


def local_calibration(y, groups, k_list=None, n_reveal=None, fold_kappa=True, seed=7):
    """Shrinkage estimate mu = (k*mean_revealed + kappa*global_median)/(k + kappa), kappa = s2_e/s2_u
    estimated within each training fold. A publication contributes at k only if it has >= k+1 pairs."""
    rng = np.random.default_rng(seed)
    k_list = k_list or C.K_LIST
    pubs = np.unique(groups)
    kappa, glob = {}, {}
    for q in pubs:
        tr = groups != q
        su, se = reml_components(y[tr], groups[tr]) if fold_kappa else reml_components(y, groups)
        kappa[q] = se / su; glob[q] = np.median(y[tr])
    rows = []
    for q in pubs:
        yi = y[groups == q]
        for k in k_list:
            if len(yi) < k + 1:
                continue
            acc = {m: [] for m in ["shrink", "revmed", "global", "zero"]}
            for _ in range(n_reveal or C.N_REVEAL):
                rv = rng.choice(len(yi), k, replace=False)
                rest = np.setdiff1d(np.arange(len(yi)), rv); yr = yi[rv]
                mu = (k * yr.mean() + kappa[q] * glob[q]) / (k + kappa[q])
                for m, v in [("shrink", mu), ("revmed", np.median(yr)), ("global", glob[q]), ("zero", 0.0)]:
                    acc[m].append(np.abs(yi[rest] - v).sum())
            for m, v in acc.items():
                rows.append(dict(pub=q, k=k, method=m, abs_sum=np.mean(v), n_rest=len(yi) - k))
    return pd.DataFrame(rows), kappa


def summarize(T, common, n_boot=10000, seed=11):
    """Pooled MAE per k and method, and paired bootstrap of the gain over the global median."""
    rng = np.random.default_rng(seed)
    out = []
    for subset in ["all", "common"]:
        for k in sorted(T.k.unique()):
            s = T[T.k == k]
            if subset == "common":
                s = s[s.pub.isin(common)]
            piv = s.pivot(index="pub", columns="method", values="abs_sum")
            nr = s.groupby("pub").n_rest.first()
            pubs = piv.index.values
            for m in ["shrink", "revmed", "zero", "global"]:
                d = piv[m] - piv["global"]
                bs = []
                for _ in range(n_boot):
                    pk = rng.choice(pubs, len(pubs))
                    bs.append(d.loc[pk].sum() / nr.loc[pk].sum())
                out.append(dict(subset=subset, k=k, method=m, n_pubs=len(pubs), mae=piv[m].sum() / nr.sum(),
                                global_mae=piv["global"].sum() / nr.sum(), diff=d.sum() / nr.sum(),
                                lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    return pd.DataFrame(out)
