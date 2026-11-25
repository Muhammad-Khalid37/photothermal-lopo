"""Physicochemical catalyst descriptors (Table S9) used in place of one-hot catalyst identity.

Element properties: Pauling electronegativity and first ionization energy (CRC Handbook of Chemistry and
Physics; NIST Atomic Spectra Database), IUPAC group, nominal cation oxidation state of the support, and
two binary flags (noble-metal active component; reducible support). See data/catalyst_descriptors.csv."""
import numpy as np
import pandas as pd
from . import config as C

CONT = ["EN_pauling", "IE1_eV", "group"]
PHYS = (["act_EN", "act_IE", "act_group", "act_noble",
         "pro_EN", "pro_IE", "pro_group", "pro_present",
         "sup_EN", "sup_IE", "sup_group", "sup_ox", "sup_reducible"])
BINARY = ["act_noble", "sup_reducible", "pro_present"]
PERTURBABLE = ["act_EN", "act_IE", "act_group", "pro_EN", "pro_IE", "pro_group", "sup_EN", "sup_IE", "sup_group", "sup_ox"]


def load_table(path=None):
    return pd.read_csv(path or (C.DATA / "catalyst_descriptors.csv"), keep_default_na=False, na_values=[""])


def add_descriptors(X, table=None, factors=None):
    """Return a copy of X with the PHYS columns. `factors` (dict column -> per-component multiplier dict)
    is used for the +/-20 % perturbation analysis."""
    t = load_table() if table is None else table
    X = X.copy()
    for role, pre, cols in [("active", "act", "active"), ("promoter", "pro", "promoter"), ("support", "sup", "support")]:
        tt = t[t.role == role].set_index("component")
        comp = X[cols].fillna("-").astype(str)
        X[f"{pre}_EN"] = comp.map(tt.EN_pauling).astype(float)
        X[f"{pre}_IE"] = comp.map(tt.IE1_eV).astype(float)
        X[f"{pre}_group"] = comp.map(tt.group).astype(float)
        if role == "active":
            X["act_noble"] = comp.map(tt.noble).astype(float)
        if role == "promoter":
            X["pro_present"] = (comp != "-").astype(float)
        if role == "support":
            X["sup_ox"] = comp.map(tt.oxidation_state).astype(float)
            X["sup_reducible"] = comp.map(tt.reducible).astype(float)
        if factors:
            for c in [f"{pre}_EN", f"{pre}_IE", f"{pre}_group"] + (["sup_ox"] if role == "support" else []):
                if c in factors:
                    X[c] = X[c] * comp.map(factors[c]).fillna(1.0).astype(float)
    return X


def random_factors(rng, table=None, width=0.2):
    """Independent U(1-width, 1+width) multiplier for every perturbable descriptor of every component."""
    t = load_table() if table is None else table
    f = {}
    for role, pre in [("active", "act"), ("promoter", "pro"), ("support", "sup")]:
        comps = t[t.role == role].component.tolist()
        for c in [f"{pre}_EN", f"{pre}_IE", f"{pre}_group"] + (["sup_ox"] if role == "support" else []):
            f[c] = {k: rng.uniform(1 - width, 1 + width) for k in comps}
    return f
