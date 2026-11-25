"""Rebuild the 96 matched dark-light pairs from the 952-record database (Sec. 2.1, Table 1, Table S3b)."""
from _common import *
from ptml.data import load_database, build_pairs, missing_li_check, prepare

args()
db = load_database()
pairs, ambiguous = build_pairs(db)
ref = C.DATA / "matched_pairs_reference.csv"
if ref.exists():   # use the row order of the published matched table (row order affects seeded models)
    order = {k: i for i, k in enumerate(zip(*[pd.read_csv(ref)[c] for c in ["dark_record_id", "light_record_id"]]))}
    pairs = pairs.assign(_o=[order.get(k, 10**6) for k in zip(pairs.dark_record_id, pairs.light_record_id)]).sort_values("_o").drop(columns="_o").reset_index(drop=True)
pairs.to_csv(C.RESULTS / "matched_pairs.csv", index=False)
ambiguous.to_csv(C.RESULTS / "ambiguous_dark_groups.csv", index=False)
P, rel = prepare(pairs)
li = db[C.DB_LI]
na, hit = missing_li_check(db)
hit[["record_id", "DOI", C.DB_STY]].to_csv(C.RESULTS / "missing_LI_potential_pairs.csv", index=False)
same = None
if ref.exists():
    r = pd.read_csv(ref)
    same = set(zip(r.dark_record_id, r.light_record_id)) == set(zip(P.dark_record_id, P.light_record_id))
out = dict(
    table1=dict(records=len(db), publications=int(db.DOI.nunique()), LI_zero=int((li == 0).sum()),
                LI_positive=int((li > 0).sum()), LI_missing=int(li.isna().sum()),
                publications_matched=int(P.pub.nunique()), dark_references=int(P.dark_record_id.nunique()),
                illuminated_records=int(P.light_record_id.nunique()), comparisons=len(P)),
    ambiguous_illuminated_records=len(ambiguous),
    missing_LI_publications=int(na.DOI.nunique()), missing_LI_potential_pairs=len(hit),
    response=dict(positive=int((P.dSTY > 0).sum()), zero=int((P.dSTY == 0).sum()), negative=int((P.dSTY < 0).sum()),
                  rel_pairs=len(rel), rel_positive=int((rel.log2r > 0).sum()), rel_negative=int((rel.log2r < 0).sum())),
    selectivity_both=int(P.dropna(subset=["S_dark", "S_light"]).shape[0]),
    reproduces_reference_pairs=same)
print(json.dumps(out, indent=1))
save_json(out, "01_dataset.json")
