"""Loading the 952-record database and reconstructing the matched dark-light pairs."""
import numpy as np
import pandas as pd
from . import config as C

# Publication codes P01-P13 follow the alphabetical order of the database DOIs.
# Corrected DOIs from the primary-paper audit (Table S3).
DOI_CORRECTIONS = {
    "10.1007/s11426-023-1876-10": "10.1007/s11426-023-1876-4",
    "10.1021/acso.3c07630": "10.1021/acsnano.3c07630",
    "10.1021/acso.4c02416": "10.1021/acsnano.4c02416",
}


def load_database(path=None):
    db = pd.read_excel(path or C.RAW_DB)
    for c in db.columns:
        if c not in C.DB_TEXT:
            db[c] = pd.to_numeric(db[c], errors="coerce")
    db["record_id"] = np.arange(1, len(db) + 1)          # 1-based row index used in the paper
    db["match_key"] = ["|".join(str(v) for v in r) for r in db[["DOI"] + C.MATCH_COLS].itertuples(index=False)]
    return db


def build_pairs(db):
    """Pair each illuminated record (LI > 0, STY reported) with the dark record (LI = 0, STY reported)
    of the same publication whose 15 descriptors are identical. Pairs are formed only when exactly one
    dark record matches; ambiguous groups are returned separately (Table S3b)."""
    sty, li = db[C.DB_STY], db[C.DB_LI]
    dark = db[(li == 0) & sty.notna()]
    light = db[(li > 0) & sty.notna()]
    cands = dark.groupby("match_key").record_id.apply(list)
    rows, ambiguous = [], []
    for r in light.itertuples():
        c = cands.get(r.match_key, [])
        if len(c) == 1:
            rows.append((c[0], r.record_id))
        elif len(c) > 1:
            ambiguous.append(dict(DOI=r.DOI, light_record_id=r.record_id, dark_candidates=c))
    idx = db.set_index("record_id")
    out = []
    for d_id, l_id in rows:
        d, l = idx.loc[d_id], idx.loc[l_id]
        rec = {k: d[k] for k in ["DOI"] + C.MATCH_COLS}
        rec.update(dark_record_id=d_id, light_record_id=l_id, LI=l[C.DB_LI],
                   STY_dark=d[C.DB_STY], STY_light=l[C.DB_STY],
                   X_dark=d[C.DB_X], X_light=l[C.DB_X], S_dark=d[C.DB_S], S_light=l[C.DB_S])
        out.append(rec)
    pairs = pd.DataFrame(out).sort_values(["DOI", "dark_record_id", "light_record_id"]).reset_index(drop=True)
    return pairs, pd.DataFrame(ambiguous)


def missing_li_check(db):
    """Missing-LI records that would match an illuminated record if treated as dark (comment 4)."""
    li, sty = db[C.DB_LI], db[C.DB_STY]
    lit = set(db.match_key[(li > 0) & sty.notna()])
    dk = set(db.match_key[(li == 0) & sty.notna()])
    na = db[li.isna()]
    hit = na[na.match_key.isin(lit) & ~na.match_key.isin(dk) & na[C.DB_STY].notna()]
    return na, hit


def prepare(pairs):
    """Add publication codes, short column names and the two responses."""
    P = pairs.rename(columns=C.RENAME).copy()
    dois = sorted(P.DOI.unique())
    P["pub"] = P.DOI.map({d: f"P{i + 1:02d}" for i, d in enumerate(dois)})
    P["DOI_corrected"] = P.DOI.replace(DOI_CORRECTIONS)
    P["dSTY"] = P.STY_light - P.STY_dark
    rel = P[P.STY_dark > 0].reset_index(drop=True).copy()
    rel["log2r"] = np.log2(rel.STY_light / rel.STY_dark)
    return P.reset_index(drop=True), rel


def absolute_sty_records(P):
    """170 distinct records (74 dark references + 96 illuminated) for the absolute-STY task."""
    d = P.drop_duplicates("dark_record_id").copy()
    d["LI"], d["STY"] = 0.0, d.STY_dark
    l = P.copy()
    l["STY"] = l.STY_light
    return pd.concat([d, l], ignore_index=True)


def load_audit_classes(P, path=None):
    """Thermal-comparability class per pair from data/audit_classes.csv (Table S3)."""
    a = pd.read_csv(path or (C.DATA / "audit_classes.csv"))
    pub_cls = dict(zip(a.pub, a["class"]))
    def f(r):
        if r.pub == "P06" and r.active == "Ni":      # P02 record entered under the P06 DOI
            return "B"
        return pub_cls.get(r.pub, "C")
    return P.apply(f, axis=1)
