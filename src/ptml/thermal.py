"""Purely thermal (Arrhenius / constant temperature rise) null models for log2(STY_light/STY_dark).

If light raises the catalyst temperature by a constant dT above the nominal T,
    log2(r_light / r_dark) = (Ea/R)(1/T - 1/(T + dT)) / ln 2  ~  Ea*dT / (R T^2 ln 2).
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import LeaveOneGroupOut


def x_arrhenius(T_C):
    T = np.asarray(T_C, float) + 273.15
    return 1.0 / (np.log(2) * T ** 2)


def fit_one_parameter(T_C, y):
    """y = theta * x, least absolute deviations -> x-weighted median of y/x. theta = Ea*dT/R (K^2)."""
    x = x_arrhenius(T_C); r = y / x; o = np.argsort(r); cw = np.cumsum(x[o])
    return float(r[o][np.searchsorted(cw, 0.5 * cw[-1])])


def fit_two_parameter(T_C, y):
    """y = theta * x + c (least absolute deviations)."""
    x = x_arrhenius(T_C)
    res = minimize(lambda b: np.abs(y - b[0] * x - b[1]).sum(), [fit_one_parameter(T_C, y), 0.0],
                   method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-9, maxiter=40000))
    return res.x


def fit_exact(T_C, y):
    """Separate Ea/R and dT (weakly identified; collapses to the one-parameter limit)."""
    T = np.asarray(T_C, float) + 273.15
    f = lambda p: np.exp(p[0]) / np.log(2) * (1 / T - 1 / (T + np.exp(p[1])))
    best = None
    for B0 in [3e3, 6e3, 1e4, 2e4]:
        for d0 in [2, 10, 40, 100]:
            r = minimize(lambda p: np.abs(y - f(p)).sum(), [np.log(B0), np.log(d0)], method="Nelder-Mead",
                         options=dict(maxiter=40000))
            if best is None or r.fun < best.fun:
                best = r
    return np.exp(best.x)


def lopo_thermal(T_C, y, groups, kind=1):
    T_C = np.asarray(T_C, float)
    pred, params = np.empty(len(y)), []
    for tr, te in LeaveOneGroupOut().split(y, y, groups):
        if kind == 1:
            th = fit_one_parameter(T_C[tr], y[tr]); pred[te] = th * x_arrhenius(T_C[te]); params.append(th)
        elif kind == 2:
            th, c = fit_two_parameter(T_C[tr], y[tr]); pred[te] = th * x_arrhenius(T_C[te]) + c; params.append((th, c))
        else:
            B, dT = fit_exact(T_C[tr], y[tr]); T = T_C[te] + 273.15
            pred[te] = B / np.log(2) * (1 / T - 1 / (T + dT)); params.append((B, dT))
    return pred, params
