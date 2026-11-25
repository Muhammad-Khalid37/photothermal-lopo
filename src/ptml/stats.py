"""Metrics, publication-cluster bootstrap, clustering statistics and permutation tests."""
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import LeaveOneGroupOut
from . import config as C


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2))


def metrics(y, p):
    e = y - p
    return dict(MAE=mae(y, p), RMSE=float(np.sqrt(np.mean(e ** 2))), MedAE=float(np.median(np.abs(e))),
                R2=r2(y, p), rho=float(stats.spearmanr(y, p)[0]) if np.std(p) > 0 else np.nan,
                sign_acc=float(np.mean(np.sign(p) == np.sign(y)) * 100))


def cluster_bootstrap_diff(err_model, err_ref, groups, balanced=False, n_boot=None, seed=None):
    """MAE(model) - MAE(comparator) with publication-cluster bootstrap 95% CI (negative favours model)."""
    rng = np.random.default_rng(C.SEED if seed is None else seed)
    pubs = np.unique(groups)
    d = np.abs(err_model) - np.abs(err_ref)
    sums = np.array([d[groups == q].sum() for q in pubs])
    ns = np.array([(groups == q).sum() for q in pubs])
    S = rng.integers(0, len(pubs), size=(n_boot or C.N_BOOT, len(pubs)))
    bs = (sums / ns)[S].mean(1) if balanced else sums[S].sum(1) / ns[S].sum(1)
    pt = float((sums / ns).mean() if balanced else d.mean())
    return dict(diff=pt, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), se=float(bs.std()))


def reml_components(y, groups):
    import statsmodels.formula.api as smf
    d = pd.DataFrame({"y": y, "g": groups})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = smf.mixedlm("y ~ 1", d, groups=d["g"]).fit(reml=True)
    return max(float(f.cov_re.iloc[0, 0]), 1e-12), float(f.scale)


def icc_and_design_effect(y, groups):
    su, se = reml_components(y, groups)
    icc = su / (su + se)
    n = pd.Series(groups).value_counts().values
    N = n.sum(); m_bar = n.mean(); m_tilde = (n ** 2).sum() / N
    return dict(icc=icc, kappa=se / su, m_bar=m_bar, m_tilde=m_tilde,
                de_equal=1 + (m_bar - 1) * icc, de_unequal=1 + (m_tilde - 1) * icc,
                n_eff_equal=N / (1 + (m_bar - 1) * icc), n_eff_unequal=N / (1 + (m_tilde - 1) * icc))


def sign_test(err_model, err_ref, groups):
    per = pd.DataFrame({"g": groups, "m": np.abs(err_model), "r": np.abs(err_ref)}).groupby("g").mean()
    w, n = int((per.m < per.r).sum()), int((per.m != per.r).sum())
    return dict(wins=w, n=n, p_one_sided=stats.binomtest(w, n, 0.5, "greater").pvalue,
                p_two_sided=stats.binomtest(w, n, 0.5).pvalue)


def permuted_response(y, groups, kind, rng):
    """'within': shuffle inside publications; 'offset': exchange publication means, keep deviations."""
    yp = y.copy()
    pubs = np.unique(groups)
    if kind == "within":
        for q in pubs:
            i = np.where(groups == q)[0]; yp[i] = rng.permutation(y[i])
    else:
        mu = {q: y[groups == q].mean() for q in pubs}
        perm = dict(zip(pubs, rng.permutation(pubs)))
        for q in pubs:
            i = groups == q; yp[i] = y[i] - mu[q] + mu[perm[q]]
    return yp


def selection_adjusted_bootstrap(preds, y, groups, baseline, candidates, n_boot=None, seed=99):
    """Choose the best configuration on in-bag publications; evaluate on out-of-bag publications."""
    rng = np.random.default_rng(seed)
    pubs = np.unique(groups)
    err = {k: np.abs(y - preds[k]) for k in candidates}
    eb = np.abs(y - baseline)
    D, chosen = [], []
    for _ in range(n_boot or C.N_BOOT_SEL):
        ib = np.unique(rng.choice(pubs, len(pubs))); ob = np.setdiff1d(pubs, ib)
        if len(ob) == 0:
            continue
        I, O = np.isin(groups, ib), np.isin(groups, ob)
        best = min(candidates, key=lambda k: err[k][I].mean() - eb[I].mean())
        chosen.append(best); D.append(err[best][O].mean() - eb[O].mean())
    D = np.array(D)
    return dict(n=len(D), frac_better=float((D < 0).mean()), mean=float(D.mean()),
                lo=float(np.percentile(D, 2.5)), hi=float(np.percentile(D, 97.5)),
                chosen=pd.Series(chosen).value_counts().to_dict())
