# Data

| File | Content | In repository |
|---|---|---|
| `raw/photothermal_methanation_database.xlsx` | 952-record literature database of photothermal CO₂ methanation (Supplementary data `mmc2.xlsx` of Z. Song et al., *Data-driven modeling of multi-target performance in photothermal CO₂ methanation*, Fuel 407 (2026) 137563, https://doi.org/10.1016/j.fuel.2025.137563) | **No** – download it from the journal and save it under this name |
| `matched_pairs_reference.csv` | The 96 matched dark–light pairs used in the manuscript (record identifiers are 1-based row numbers of the database) | Yes |
| `audit_classes.csv` | Primary-paper audit of heating mode and temperature basis for the 13 source publications (Table S3) and the resulting thermal-comparability class | Yes |

`scripts/01_build_dataset.py` rebuilds the matched pairs from the raw database and checks that they are identical to `matched_pairs_reference.csv`.

## Matching rule

An illuminated record (LI > 0, STY reported) is paired with a dark record (LI = 0, STY reported) of the same DOI when all 15 descriptors agree (support, active component and loading, promoter and loading, calcination temperature and time, reduction temperature and time, catalyst mass, GHSV, H₂/CO₂ ratio, inert-gas ratio, pressure, reaction temperature; unreported values must be unreported in both). A pair is formed only when exactly one dark record matches; two groups at 140 °C (P04, P08; 24 illuminated records) have several candidate dark records and are excluded.

## Publication codes and audit classes

Codes P01–P13 follow the alphabetical order of the DOIs in the database. Three DOIs in the database are wrong (P02, P09, P10); `audit_classes.csv` gives the corrected ones. The P08 DOI resolves to an unrelated gas-sensor paper (class C). The Ni–Fe pair listed under P06 is a P02 record entered under the P06 DOI (class B).

Classes: **A** – temperature defined identically in both runs (same external set-point with light added, or the same measured catalyst temperature); **A\*** – as A, classified from the Supporting Information only; **B** – light-only run compared with an externally heated dark reference; **C** – source not verifiable.
