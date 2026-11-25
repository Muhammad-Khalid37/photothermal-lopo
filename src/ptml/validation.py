"""Leave-one-publication-out (LOPO) and mixed random-split validation."""
import numpy as np
from sklearn.model_selection import KFold, LeaveOneGroupOut
from . import config as C
from .models import ridge


def lopo_predict(factory, X, y, groups):
    pred = np.empty(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = factory()
        m.fit(X.iloc[tr], y[tr])
        pred[te] = m.predict(X.iloc[te])
    return pred


def lopo_constant(y, groups, kind):
    pred = np.empty(len(y))
    for tr, te in LeaveOneGroupOut().split(y, y, groups):
        pred[te] = {"zero": 0.0, "median": np.median(y[tr]), "mean": np.mean(y[tr])}[kind]
    return pred


def seed_average(factory_of_seed, X, y, groups, n_seeds=None):
    n = n_seeds or C.N_SEEDS
    return np.array([lopo_predict(lambda s=s: factory_of_seed(s), X, y, groups) for s in range(n)])


def nested_ridge(X, y, groups, grid, inner="mse"):
    """Outer LOPO; alpha chosen by inner LOPO on the training publications."""
    pred, chosen = np.empty(len(y)), {}
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        Xt, yt, gt = X.iloc[tr], y[tr], groups[tr]
        scores = []
        for a in grid:
            pi = lopo_predict(lambda a=a: ridge(a), Xt, yt, gt)
            scores.append(np.mean((yt - pi) ** 2) if inner == "mse" else np.mean(np.abs(yt - pi)))
        a = grid[int(np.argmin(scores))]
        chosen[groups[te][0]] = a
        pred[te] = ridge(a).fit(Xt, yt).predict(X.iloc[te])
    return pred, chosen


def mixed_cv(X, y, model, reps=10):
    """Repeated random 5-fold CV; returns MAE and pooled / fold-averaged R2."""
    from .stats import mae, r2
    from .models import random_forest
    out = dict(mae=[], r2_pooled=[], r2_foldavg=[])
    for rep in range(reps):
        p, fr = np.empty(len(y)), []
        for tr, te in KFold(5, shuffle=True, random_state=rep).split(X):
            if model == "median":
                p[te] = np.median(y[tr])
            elif model == "rf":
                p[te] = random_forest(rep).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            else:
                best, bs = None, np.inf
                for a in C.RIDGE_GRID_EXT:
                    pi = np.empty(len(tr))
                    for itr, ite in KFold(5, shuffle=True, random_state=rep).split(tr):
                        pi[ite] = ridge(a).fit(X.iloc[tr[itr]], y[tr[itr]]).predict(X.iloc[tr[ite]])
                    s = np.mean((y[tr] - pi) ** 2)
                    if s < bs:
                        best, bs = a, s
                p[te] = ridge(best).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            fr.append(r2(y[te], p[te]))
        out["mae"].append(mae(y, p)); out["r2_pooled"].append(r2(y, p)); out["r2_foldavg"].append(np.mean(fr))
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1))) for k, v in out.items()}
