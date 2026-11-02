# Anhui monthly AOD: saved evidence and validation audit

This repository contains processed numerical evidence for the 2015-2024 Anhui AOD study: 1,343 grid centres, 161,160 monthly records, four model comparisons, fixed outer validation assignments, saved out-of-fold predictions, RF SHAP/residual outputs, and two prespecified sensitivity studies.

Repository: https://github.com/songmengqiang554-arch/Anhui-AOD-2015-2024

The primary historical explanatory model is Random Forest with 11 predictors. NTL, grid identifiers, coordinates, and year are not predictors. Archived filenames containing earlier project step numbers identify provenance, not alternative current model choices.

## Verify without training

From this directory, in a Python environment containing NumPy and pandas:

```text
python -B 07_tools/verify_saved_evidence.py --check-only
```

The command reads existing evidence and recomputes metrics/counts. It does not fit models, recalculate SHAP, run trend tests, modify the repository, or use network access. Gzipped CSVs can be read directly with pandas. The exact environment in which this check passed is recorded in `06_QA/verification_environment.json`.

## Contents

- `01_processed_data`: exact-token projection of the frozen model table, grid coordinates, and processing intermediates.
- `02_fixed_splits`: immutable original sample/fold membership, including 10 LOYO years and five spatial region labels.
- `03_analysis_scripts`: historical scientific source code with only local path literals/comments anonymized. Read `03_analysis_scripts/README.md` before use.
- `04_locked_results`: saved model predictions, locked metrics, SHAP and Moran outputs, NDVI-removal results, and trend sensitivities.
- `05_provenance`: provider citations, units, preprocessing limitations and redistribution scope.
- `06_QA`: lightweight validation receipts, not a new model reproduction.
- `07_tools`: portable read-only verifier.
- `file_manifest.csv`: relative paths, sizes and SHA256 of all files except the manifest itself.

## Statistical contracts

`sample_id` is the zero-based row index of the original frozen model CSV, never an index regenerated after sorting. Preserve this order. `month_sin` and `month_cos` are derived as sin/cos(2*pi*month/12). The 11 predictors and units are listed in `05_provenance/data_dictionary.csv`.

Fold-mean metrics and pooled OOF metrics are separate estimands. In particular, RF spatial-region fold-mean R2 is approximately 0.773623, whereas pooled OOF R2 is approximately 0.792938. Neither value substitutes for the other. Random holdout contains 32,232 test rows; nonrandom OOF schemes contain 161,160 rows.

The original NDVI interpolation can use later months. Saved NDVI-removal sensitivity preserves all 40 model-year R2 signs and negative four-model LOYO means; that result does not prove absence of original temporal look-ahead. The covariance-adjusted seasonal trend sensitivity is not 3PW and does not eliminate every form of autocorrelation. See `05_provenance/limitations.md`.

## Scope and release state

This is a public evidence and code repository, not a DOI deposit or a completed clean-environment retraining certificate. No manuscript, correspondence, email, account credential, whole third-party article PDF, raw bulk archive, or administrative map is included. Read `RIGHTS_AND_ATTRIBUTION.md` before reuse. The package does not claim a blanket licence over third-party inputs.
