"""Paths, column names and model settings used throughout the study."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW_DB = DATA / "raw" / "photothermal_methanation_database.xlsx"   # 952-record database (see data/README.md)
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# --- database columns -------------------------------------------------------
DB_STY = "STY[mmolg-1h-1]"
DB_LI = "LI[Wcm-2]"
DB_X = "XCO2[%]"
DB_S = "Smenthene[%]"
DB_TEXT = ["S", "A", "P", "DOI"]

# 15 descriptors that must agree for a dark and an illuminated record to be paired
MATCH_COLS = ["S", "A", "AC[wt%]", "P", "PC[wt%]", "CT[℃]", "CD[h]", "RT[℃]", "RD [h]",
              "MS[mg]", "GHSV[mLg-1h-1]", "H2/CO2[-]", "IGR[%]", "RP[Mpa]", "T[℃]"]

# short names used in the matched table
RENAME = {"S": "support", "A": "active", "P": "promoter", "AC[wt%]": "AC", "PC[wt%]": "PC",
          "MS[mg]": "MS", "GHSV[mLg-1h-1]": "GHSV", "H2/CO2[-]": "H2CO2", "RP[Mpa]": "RP", "T[℃]": "T"}

# predictors (Table S2)
CAT = ["support", "active", "promoter"]
NUM = ["AC", "PC", "T", "LI", "MS", "GHSV", "H2CO2", "RP"]

# --- model settings (Table S17) --------------------------------------------
RF_PARAMS = dict(n_estimators=200, min_samples_leaf=3, max_features=0.8)
GB_PARAMS = dict(loss="absolute_error", n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8)
RIDGE_GRID_ORIG = [0.1, 1, 10, 100, 1000]
RIDGE_GRID_EXT = list(np.logspace(-1, 6, 8))

N_SEEDS = 20          # tree-ensemble seeds 0-19
N_BOOT = 20000        # publication-cluster bootstrap resamples
N_BOOT_SEL = 4000     # selection-adjusted bootstrap resamples
N_PERM = 5000         # permutations per publication-preserving test
N_REVEAL = 1000       # random draws of revealed pairs per publication and k
K_LIST = [1, 2, 3, 5]
SEED = 2026


def quick_mode():
    """Reduce repetitions for a fast smoke run (python scripts/run_all.py --quick)."""
    global N_SEEDS, N_BOOT, N_BOOT_SEL, N_PERM, N_REVEAL
    N_SEEDS, N_BOOT, N_BOOT_SEL, N_PERM, N_REVEAL = 2, 500, 200, 20, 30
