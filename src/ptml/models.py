"""Preprocessing and model factories. All preprocessing is fitted inside each training fold."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from . import config as C


def preprocessor(scale=False, num=None, cat=None):
    num = C.NUM if num is None else num
    cat = C.CAT if cat is None else cat
    steps = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else [])
    parts = [("num", Pipeline(steps), num)]
    if cat:
        parts.append(("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat))
    return ColumnTransformer(parts)


def random_forest(seed, num=None, cat=None, **kw):
    return make_pipeline(preprocessor(num=num, cat=cat),
                         RandomForestRegressor(random_state=seed, n_jobs=1, **{**C.RF_PARAMS, **kw}))


def gradient_boosting(seed, num=None, cat=None):
    return make_pipeline(preprocessor(num=num, cat=cat), GradientBoostingRegressor(random_state=seed, **C.GB_PARAMS))


def ridge(alpha, num=None, cat=None):
    return make_pipeline(preprocessor(scale=True, num=num, cat=cat), Ridge(alpha=alpha))
