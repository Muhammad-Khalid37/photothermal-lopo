# Matched dark–light comparisons in photothermal CO₂ methanation: publication-held-out machine learning

Code and data for the manuscript

> **Matched Dark–Light Comparisons in Photothermal CO₂ Methanation: Limits of Cross-Publication Prediction and a Temperature-Dependent Relative Response**
> [Muhammad Khalid, Muhammad Rafique, Ujjwal Dhakal, Saba Arif, Lifen Li, Bachirou Guene Lougou, Mario Porru, Giambattista Gruosso, Faisal Shahzad, Yong Shuai],

The analysis rebuilds 96 matched dark–light comparisons (13 publications) from a 952-record literature database and asks, by leave-one-publication-out (LOPO) validation, whether the illumination-induced change in methane space-time yield can be predicted for unseen publications. It covers the pre-specified primary analysis (absolute ΔSTY), validation leakage under random splits, publication-level clustering, the post hoc relative enhancement log₂(STY_light/STY_dark) with a purely thermal (Arrhenius/ΔT) null model, publication-preserving permutation tests, local calibration, and a sensitivity analysis on thermally comparable pairs from a primary-paper audit.

## Repository structure

```
├── data/
│   ├── README.md                     data sources, matching rule, audit classes
│   ├── matched_pairs_reference.csv   the 96 matched pairs
│   ├── audit_classes.csv             primary-paper audit (Table S3)
│   ├── catalyst_descriptors.csv      physicochemical catalyst descriptors with sources (Table S9)
│   └── raw/                          place the 952-record database here (not distributed)
├── src/ptml/                         analysis package
│   ├── config.py                     paths, columns, model settings (Table S17)
│   ├── data.py                       loading, matching, responses, audit classes
│   ├── models.py                     preprocessing and model factories
│   ├── validation.py                 LOPO, nested Ridge, mixed random-split CV
│   ├── stats.py                      metrics, cluster bootstrap, ICC/design effect, permutations, selection-adjusted bootstrap
│   ├── thermal.py                    Arrhenius/ΔT null models
│   ├── descriptors.py                physicochemical descriptors (Table S9) and ±20 % perturbation
│   └── calibration.py                local calibration with fold-internal κ
├── scripts/                          one script per analysis (see table below)
├── tests/                            pytest checks
└── results/                          created by the scripts (JSON, CSV, figures)
```

## Installation

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python ≥ 3.10. Results in the manuscript were produced with Python 3.13 and scikit-learn 1.9.1; other scikit-learn versions can change seeded tree-ensemble results slightly.

## Data

Download the database (see `data/README.md`) and save it as `data/raw/photothermal_methanation_database.xlsx`.

## Running

```bash
python scripts/run_all.py --quick                 # smoke test, ~5 min
python scripts/run_all.py --skip-permutations     # everything except the permutation tests, ~20 min
python scripts/run_all.py                         # full run; the 2 × 5,000 permutations take several hours on a laptop
pytest -q                                         # tests
```

Each script can also be run on its own, in order. `06_permutation_tests.py` is resumable (re-running continues the files `results/perm_*.csv`) and parallel (`--jobs N`; `--n 1000` for fewer permutations).

## Scripts and manuscript items

| Script | Manuscript items |
|---|---|
| `01_build_dataset.py` | Sec. 2.1, Table 1, Table S3b (ambiguous dark groups, missing-LI check) |
| `02_absolute_lopo.py` | Tables 2, 3, S5; Sec. 3.2 contrasts; Sec. 3.3.4 Ridge diagnostics |
| `03_leakage_mixed_cv.py` | Table 4, Fig. 6 (mixed vs. publication-held-out; pooled and fold-averaged R²) |
| `04_clustering.py` | Sec. 3.3.2 (REML ICC, unequal-cluster design effect, minimum detectable difference) |
| `05_relative_enhancement.py` | Table 7, Tables S12, S13, S15, S16; Sec. 3.6–3.7 (thermal model, sign test, jackknife, selection-adjusted bootstrap, ablation) |
| `06_permutation_tests.py` | Sec. 3.6, Table S16 (within-publication and publication-mean permutation tests) |
| `07_local_calibration.py` | Sec. 3.7, Fig. 11a,b, Table S14 |
| `08_sensitivity_audit.py` | Sec. 3.5, Table S3c, Table S1 classes |
| `09_figures.py` | Figs. 4–6 and 8–11 (see "Figure numbering") |
| `10_descriptor_models.py` | Table 5, physicochemical rows of Table 7, Fig. S9, Table S16 (selection-adjusted bootstrap over 20 configurations, ±20 % descriptor perturbation, binary descriptors), Section S8 |
| `11_supplementary_analyses.py` | Fig. 7 (variance share), Section S3, Tables S5–S8, S10, S11, S15, S16; data for Figs. S4–S8 |
| `12_si_figures.py` | Fig. 2 and Figs. S1–S9 (see "Figure numbering") |

## Figure numbering

The scripts name their output files by an earlier numbering; the published figures were restyled (Times-type font, journal sizes) and renumbered. Mapping of script output to the article:

| Script output (`results/figures/`) | Article |
|---|---|
| `fig3_response_distributions.png` | Fig. 4 |
| `fig4_publication_mae.png` | Fig. 5 |
| `fig5_mixed_vs_lopo.png` | Fig. 6 |
| `fig6_light_intensity_ranges.png` | Fig. 8 |
| `fig7_P02_intensity_series.png` | Fig. 9 |
| `fig8_relative_enhancement.png` | Fig. 10 |
| `fig9_calibration_and_temperature.png` | Fig. 11 |
| `S1.png` | Fig. 2 (workflow) |
| `S2.png` | Fig. S1 |
| `S3.png` | Fig. 7 |
| `S4.png` – `S11.png` | Figs. S2 – S9 |

The data in the plots are identical; only style and numbering differ.

## Methods in brief

* **Validation:** leave-one-publication-out; every preprocessing step (median imputation, “Missing” category, one-hot encoding, standardization for Ridge) is fitted inside the training fold.
* **Models:** constant baselines (zero, training median, training mean); Ridge with α ∈ {10⁻¹,…,10⁶} chosen by inner publication-grouped LOPO (squared error); L1 median regression with its penalty chosen by inner LOPO (absolute error); Random Forest (200 trees, min_samples_leaf = 3, max_features = 0.8); gradient boosting (absolute-error loss, 100 trees, learning rate 0.1, depth 3, subsample 0.8). Tree ensembles are averaged over seeds 0–19.
* **Uncertainty:** publication-cluster bootstrap (20,000 resamples); MAE differences are model minus comparator (negative favours the model).
* **Thermal null model:** ŷ = θ / (T² ln 2) with θ = Eₐ·ΔT/R, fitted by least absolute deviations in each training fold (Sec. 3.7).
* **Local calibration:** μ = (k·mean_revealed + κ·median_global)/(k + κ), κ = σ²_ε/σ²_u from a random-intercept model fitted to the training publications.

## Key results (full run)

| Quantity | Value |
|---|---|
| LOPO MAE, ΔSTY: predict-zero / median / mean / Ridge / Random Forest | 25.32 / 25.56 / 29.09 / 29.31 / 35.60 mmol g⁻¹ h⁻¹ |
| REML ICC; effective sample size (unequal clusters) | 0.41; ≈ 19 |
| LOPO MAE, log₂ ratio: median / gradient boosting / temperature-only GB / thermal model | 1.16 / 0.76 / 0.77 / 0.79 |
| Publication-preserving permutation p (within / publication means; 5,000 each) | 0.0012 / 0.0030 |
| Best absolute configuration (GB, physicochemical descriptors); selection-adjusted share better than median | 24.46 ± 0.81 mmol g⁻¹ h⁻¹; 35% |

## Reproducibility notes

* `01_build_dataset.py` orders the rebuilt pairs as in `matched_pairs_reference.csv`; row order matters for seeded models with row subsampling.
* All random seeds are fixed in `src/ptml/config.py`. Bootstrap intervals can differ from the manuscript in the second decimal (Monte Carlo error).
* `results/` contains the tables (JSON/CSV) of the full run used for the manuscript, including the 2 × 5,000 permutation nulls, so the permutation step finishes immediately unless you delete `results/perm_*.csv`. Figures and the intermediate prediction files (`*.npy`) are not distributed; they are recreated by `run_all.py`.

## Citation

If you use this code, please cite the article above (see `CITATION.cff`).

## License

Code: MIT (see `LICENSE`). The raw database belongs to its original authors and is not redistributed here.
