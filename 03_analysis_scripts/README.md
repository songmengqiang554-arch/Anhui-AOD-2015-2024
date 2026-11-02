# Historical scientific code

These are archival source files, not the entry points for the lightweight check. The preparation process did not import or execute them. Local absolute path string literals and comments were replaced with `<LOCAL_PATH>`; a normalized Python AST comparison confirmed no other code changes.

Their historical input/output layout and optional dependencies are not made portable by that redaction. Do not execute them unchanged: some contain top-level model fitting and writing operations. Exact retraining additionally requires the historical environment, path mapping, training-row order, and Ridge inner tuning rules. This candidate certifies saved-evidence verification, not a fresh end-to-end training run.

`step58` records controlled tree model comparisons; `step59` adds training-fold-tuned Ridge; `step60` locks RF explanatory/SHAP/residual evidence; `step77` is the NDVI-removal LOYO sensitivity; `step78` records the autocorrelation sensitivity. `trend_v2` is a historical attempt and must not be interpreted as successful 3PW results. Archived filenames and output-writing text are retained as historical context, not publication-ready manuscript wording.
