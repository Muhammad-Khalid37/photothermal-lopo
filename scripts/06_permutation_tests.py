"""Publication-preserving permutation tests for the gradient-boosting gain (comment 17).
Each permutation repeats the full LOPO workflow (seed 0). Resumable: re-running continues the files.
    python scripts/06_permutation_tests.py            # 5,000 permutations per test
    python scripts/06_permutation_tests.py --n 1000   # fewer
"""
from _common import *
import argparse
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut
from ptml.models import preprocessor
from ptml.stats import mae, permuted_response

a = args()
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=None); ap.add_argument("--jobs", type=int, default=-1)
b, _ = ap.parse_known_args()
N = b.n or C.N_PERM
_, R = load_pairs()
y, g = R.log2r.values, R.pub.values
folds = []
for tr, te in LeaveOneGroupOut().split(R, y, g):
    pre = preprocessor().fit(R.iloc[tr]); folds.append((tr, te, pre.transform(R.iloc[tr]), pre.transform(R.iloc[te])))


def gain(yv):
    p, base = np.empty(len(yv)), np.empty(len(yv))
    for tr, te, Xtr, Xte in folds:
        p[te] = GradientBoostingRegressor(random_state=0, **C.GB_PARAMS).fit(Xtr, yv[tr]).predict(Xte)
        base[te] = np.median(yv[tr])
    return mae(yv, base) - mae(yv, p)


obs = gain(y)
res = dict(observed_gain=obs)
for kind, seed in [("within", 12345), ("offset", 54321)]:
    f = C.RESULTS / f"perm_{kind}.csv"
    done = sum(1 for _ in open(f)) if f.exists() else 0
    rng = np.random.default_rng(seed); seeds = rng.integers(0, 2 ** 31 - 1, size=N)
    todo = list(range(done, N))
    for start in range(0, len(todo), 200):
        chunk = todo[start:start + 200]
        vals = Parallel(n_jobs=b.jobs)(delayed(gain)(permuted_response(y, g, kind, np.random.default_rng(int(seeds[i])))) for i in chunk)
        with open(f, "a") as fh:
            fh.writelines(f"{v}\n" for v in vals)
        print(f"  {kind}: {done + start + len(chunk)}/{N}", flush=True)
    null = np.loadtxt(f)[:N]
    k = int((null >= obs).sum())
    res[kind] = dict(n=len(null), n_ge_observed=k, p=(1 + k) / (1 + len(null)), null_p95=float(np.percentile(null, 95)))
print(json.dumps(res, indent=1))
save_json(res, "06_permutation_tests.json")
