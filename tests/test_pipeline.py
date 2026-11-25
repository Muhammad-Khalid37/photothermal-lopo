"""Fast checks: matching rule, thermal model, bootstrap sign convention."""
import sys
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptml import config as C
from ptml.thermal import fit_one_parameter, x_arrhenius
from ptml.stats import cluster_bootstrap_diff


@pytest.mark.skipif(not C.RAW_DB.exists(), reason="database not downloaded")
def test_matching_reproduces_96_pairs():
    import pandas as pd
    from ptml.data import load_database, build_pairs
    pairs, amb = build_pairs(load_database())
    assert len(pairs) == 96 and pairs.DOI.nunique() == 13 and len(amb) == 24
    ref = C.DATA / "matched_pairs_reference.csv"
    if ref.exists():
        r = pd.read_csv(ref)
        assert set(zip(r.dark_record_id, r.light_record_id)) == set(zip(pairs.dark_record_id, pairs.light_record_id))


def test_thermal_model_recovers_theta():
    T = np.linspace(100, 400, 50); theta = 2.5e5
    assert abs(fit_one_parameter(T, theta * x_arrhenius(T)) - theta) / theta < 1e-6


def test_bootstrap_sign_convention():
    y = np.arange(20.0); g = np.repeat(np.arange(5), 4)
    r = cluster_bootstrap_diff(y - y, y - 0.0, g, n_boot=200)
    assert r["diff"] < 0          # a perfect model has negative (favourable) difference


def test_descriptor_table_covers_all_components():
    from ptml.data import prepare
    from ptml.descriptors import add_descriptors, PHYS
    f = C.DATA / "matched_pairs_reference.csv"
    P, _ = prepare(pd.read_csv(f, encoding="utf-8-sig")) if "STY_dark" in pd.read_csv(f, nrows=1, encoding="utf-8-sig").columns else (None, None)
    if P is None:
        pytest.skip("reference table lacks STY columns")
    X = add_descriptors(P)
    missing = X[PHYS].isna().any(axis=1)
    assert set(X.loc[missing, "support"]) <= {"-"}      # only the unsupported Ni-Fe row is imputed
