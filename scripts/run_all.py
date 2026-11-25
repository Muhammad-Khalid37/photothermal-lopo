"""Run the complete analysis in order.
    python scripts/run_all.py            # full run (permutations take hours; see README)
    python scripts/run_all.py --quick    # smoke test, a few minutes
    python scripts/run_all.py --skip-permutations
"""
import subprocess, sys, time
from pathlib import Path
here = Path(__file__).parent
steps = ["01_build_dataset", "02_absolute_lopo", "03_leakage_mixed_cv", "04_clustering", "05_relative_enhancement",
         "06_permutation_tests", "07_local_calibration", "08_sensitivity_audit", "09_figures",
         "10_descriptor_models", "11_supplementary_analyses", "12_si_figures"]
extra = [a for a in sys.argv[1:] if a == "--quick"]
for s in steps:
    if s.startswith("06") and "--skip-permutations" in sys.argv:
        continue
    t = time.time(); print(f"\n=== {s} ===", flush=True)
    r = subprocess.run([sys.executable, str(here / f"{s}.py"), *extra])
    if r.returncode:
        sys.exit(f"{s} failed")
    print(f"=== {s} done in {time.time() - t:.0f} s")
