import json, sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from ptml import config as C


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="few seeds/resamples, for a smoke test")
    a, _ = ap.parse_known_args()
    if a.quick:
        C.quick_mode()
    C.RESULTS.mkdir(exist_ok=True); C.FIGURES.mkdir(parents=True, exist_ok=True)
    return a


def load_pairs():
    from ptml.data import prepare
    f = C.RESULTS / "matched_pairs.csv"
    if not f.exists():
        sys.exit("results/matched_pairs.csv not found - run scripts/01_build_dataset.py first")
    return prepare(pd.read_csv(f))


def save_json(obj, name):
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    (C.RESULTS / name).write_text(json.dumps(obj, indent=1, default=conv))
    print(f"  -> results/{name}")


def load_json(name):
    return json.loads((C.RESULTS / name).read_text())
