from __future__ import annotations

import __future__
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback


ROOT = Path('<LOCAL_PATH>')
OUT = Path(__file__).resolve().parents[1]
S58 = ROOT / "58_CONTROLLED_MODEL_REPRODUCTION_AND_SELECTION_GATE"
S59 = ROOT / "59_FOUR_MODEL_COMPARISON_AND_FINAL_SELECTION"
S60 = ROOT / "60_RF_PRIMARY_MODEL_FINAL_EVIDENCE"
DATA = S58 / "00_frozen_inputs/Anhui_AOD_Model_Dataset_with_traceable_DEM_NTL.csv"
TREE_SOURCE = S58 / "09_logs/run_controlled_reproduction.py"
RIDGE_SOURCE = S59 / "06_logs/run_ridge_four_model_gate.py"
REGISTRY = S58 / "03_split_registry/loyo_fold_registry.csv"
EXPECTED_DATA_HASH = "a4110f4b6cae4bd91212f02c1faf78870f72d1312122fd3b840c74ff2dd6a11d"
YEARS = list(range(2015, 2025))
MODELS = ["Ridge", "RandomForest", "HistGradientBoosting", "XGBoost"]
DIRS = {k: OUT / v for k, v in {
    "input": "01_input_verification", "results": "02_results",
    "predictions": "03_predictions", "logs": "05_logs",
    "reports": "06_reports", "checksums": "07_checksums",
}.items()}
for directory in DIRS.values():
    directory.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(DIRS["logs"] / "tmp")
os.environ["TEMP"] = os.environ["TMP"]
Path(os.environ["TMP"]).mkdir(exist_ok=True)
sys.dont_write_bytecode = True


class Tee:
    def __init__(self, stream, path):
        self.stream = stream
        self.file = path.open("a", encoding="utf-8", buffering=1)

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()


sys.stdout = Tee(sys.stdout, DIRS["logs"] / "console_stdout.log")
sys.stderr = Tee(sys.stderr, DIRS["logs"] / "console_stderr.log")

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def index_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def require(condition, message):
    if not condition:
        raise RuntimeError("STRICT_CONTROL_STOP: " + message)


def original_namespace(path, names):
    """Compile selected original functions without running historical module writes."""
    source = path.read_text(encoding="utf-8-sig")
    parsed = ast.parse(source)
    constants = {}
    for node in parsed.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"FEATURES", "ALPHAS", "RANDOM_STATE", "TARGET"}:
                    constants[target.id] = ast.literal_eval(node.value)
    selected = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in selected} == set(names), f"Missing original functions in {path}")
    namespace = dict(globals())
    namespace.update(constants)
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    provenance = [{"source": str(path), "source_sha256": sha(path), "function": n.name,
                   "first_line": n.lineno, "last_line": n.end_lineno,
                   "function_text_sha256": text_hash(ast.get_source_segment(source, n))} for n in selected]
    return namespace, provenance


def protected_paths():
    selected = [DATA, TREE_SOURCE, RIDGE_SOURCE, REGISTRY,
                S58 / "00_frozen_inputs/01_frozen_dataset_fingerprint.json",
                S58 / "00_frozen_inputs/04_model_parameter_inventory.csv",
                S58 / "00_frozen_inputs/05_software_environment.txt",
                S58 / "03_split_registry/split_hashes.json",
                S58 / "05_harmonized_validation/04_LOYO_yearly_comparison.csv",
                S58 / "05_harmonized_validation/05_LOYO_summary.csv",
                S59 / "02_ridge_validation/04_Ridge_LOYO_yearly.csv",
                S59 / "02_ridge_validation/05_Ridge_predictions_OOF.csv",
                S59 / "02_ridge_validation/06_Ridge_metrics_summary.csv",
                S59 / "06_logs/environment_lock.txt",
                S60 / "07_locked_results/02_four_model_metrics_locked_source.csv"]
    selected += [S58 / f"05_harmonized_validation/{model}_LOYO_OOF_predictions.csv.gz" for model in MODELS[1:]]
    for prefix in ("72_", "73_", "74_", "75_", "76_"):
        for folder in ROOT.glob(prefix + "*"):
            if folder.is_dir():
                selected += [p for p in folder.rglob("*") if p.is_file()]
    return sorted(set(selected))


def snapshot(paths):
    return [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)} for p in paths]


def normalize_baseline(metrics):
    return pd.DataFrame({
        "model": metrics.model, "year": metrics.fold_or_year.astype(int),
        "n_train": metrics.n_train.astype(int), "n_test": metrics.n_test.astype(int),
        "R2": metrics.test_R2, "RMSE": metrics.test_RMSE, "MAE": metrics.test_MAE,
        "mean_bias": metrics.test_mean_bias_pred_minus_obs,
    })


def read_original_predictions(model):
    if model == "Ridge":
        path = S59 / "02_ridge_validation/05_Ridge_predictions_OOF.csv"
        pieces = []
        for part in pd.read_csv(path, chunksize=100000, dtype={"fold": str}):
            pieces.append(part.loc[part.validation_scheme.eq("leave_one_year_out")])
        result = pd.concat(pieces, ignore_index=True)
    else:
        path = S58 / f"05_harmonized_validation/{model}_LOYO_OOF_predictions.csv.gz"
        result = pd.read_csv(path)
        require(result.validation.eq("leave_one_year_out").all(), f"Wrong original prediction scheme: {model}")
        result["year"] = result["fold_or_year"].astype(int)
        if model == "XGBoost":
            result["predicted_AOD"] = result["predicted_AOD"].astype(np.float32)
    return result.sort_values("sample_id").reset_index(drop=True), path


def check_original_predictions(df, baseline, metric):
    audit = []
    for model in MODELS:
        pred, path = read_original_predictions(model)
        require(len(pred) == len(df), f"Original OOF size mismatch: {model}")
        require(np.array_equal(pred.sample_id, np.arange(len(df))), f"Original OOF sample IDs mismatch: {model}")
        require(np.array_equal(pred.year, df.year), f"Original OOF years mismatch: {model}")
        require(np.allclose(pred.observed_AOD, df.AOD, rtol=0, atol=1e-12), f"Original OOF target mismatch: {model}")
        for year in YEARS:
            p = pred.loc[pred.year.eq(year)]
            recomputed = metric(p.observed_AOD, p.predicted_AOD)
            stored = baseline.loc[baseline.model.eq(model) & baseline.year.eq(year)].iloc[0]
            differences = {k: abs(recomputed[k] - float(stored[k])) for k in ("R2", "RMSE", "MAE")}
            require(max(differences.values()) < 1e-10, f"Original OOF does not reproduce stored metrics: {model}/{year}")
            audit.append({"model": model, "year": year, "source": str(path),
                          **{f"abs_diff_{k}": v for k, v in differences.items()},
                          "original_prediction_dtype": "float32_restored_from_CSV" if model == "XGBoost" else "float64",
                          "status": "PASS"})
    return pd.DataFrame(audit)


def preflight(tree, ridge):
    log("PRECHECK: protected source hashes and original data/split/software contracts")
    before = snapshot(protected_paths())
    dump(DIRS["input"] / "01_protected_source_hashes_before.json", before)
    require(sha(DATA) == EXPECTED_DATA_HASH, "Frozen dataset hash differs from locked source")
    expected_env = {"python": "3.11.11", "numpy": "1.26.4", "pandas": "3.0.1", "sklearn": "1.2.2", "xgboost": "3.0.5"}
    actual_env = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                  "sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "scipy": scipy.__version__,
                  "python_executable": sys.executable,
                  "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    require(all(actual_env[k] == v for k, v in expected_env.items()), "Original numerical package versions not recovered")
    dump(DIRS["input"] / "02_environment.json", {"expected": expected_env, "actual": actual_env, "status": "PASS"})
    original_features = tree["FEATURES"]
    require(original_features == ridge["FEATURES"] and len(original_features) == 11, "Original feature lists disagree")
    features = [name for name in original_features if name != "NDVI_final"]
    require(len(features) == 10 and set(original_features) - set(features) == {"NDVI_final"}, "Unexpected feature removal")
    require(tree["RANDOM_STATE"] == ridge["RANDOM_STATE"] == 42, "Original random seeds differ")
    df = pd.read_csv(DATA)
    df.insert(0, "sample_id", np.arange(len(df), dtype=np.int64))
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    require(len(df) == 161160 and df.grid_id.nunique() == 1343, "Wrong frozen grid-month population")
    require(sorted(df.year.unique()) == YEARS and df.groupby("grid_id").size().eq(120).all(), "Incomplete time series")
    require(not df[original_features + ["AOD"]].isna().any().any(), "Unexpected missing values")
    require(not df.duplicated(["grid_id", "year", "month"]).any(), "Duplicate grid-month keys")
    X_original = df[original_features].copy()
    X = df[features].copy()
    require(X.equals(X_original.drop(columns="NDVI_final")), "Retained covariates changed")
    y = df["AOD"].copy()
    registry = pd.read_csv(REGISTRY).sort_values("sample_id").reset_index(drop=True)
    require(np.array_equal(registry.sample_id, df.sample_id), "LOYO registry/sample order differs")
    for key in ("grid_id", "year"):
        require(np.array_equal(registry[key], df[key]), f"LOYO registry {key} mismatch")
    require(np.array_equal(registry.loyo_test_year, df.year), "LOYO test-year labels mismatch")
    logical = text_hash("\n".join(f"{i},{v}" for i, v in enumerate(registry.year)))
    stored_splits = json.loads((S58 / "03_split_registry/split_hashes.json").read_text(encoding="utf-8"))
    require(logical == stored_splits["loyo_sha256"], "Original LOYO logical hash not matched")
    inventory = pd.read_csv(S58 / "00_frozen_inputs/04_model_parameter_inventory.csv", keep_default_na=False)
    parameter_audit = []
    for model in MODELS[1:]:
        params = tree["build_model"](model, 42).get_params()
        for r in inventory.loc[inventory.model.eq(model)].itertuples(index=False):
            require(str(params[r.parameter]) == str(r.value), f"Parameter mismatch: {model}/{r.parameter}")
        parameter_audit.append({"model": model, "parameters_at_base_seed": params, "loyo_seed_rule": "42 + heldout_year"})
    dump(DIRS["input"] / "03_model_configuration.json", {"tree_models": parameter_audit,
         "Ridge": {"pipeline": "StandardScaler -> Ridge", "alpha_candidates": ridge["ALPHAS"],
                   "inner_cv": "GroupKFold(5), groups=year within outer training rows",
                   "selection": "minimum mean inner RMSE; tie broken by smaller alpha",
                   "target_standardization": False, "random_state": 42},
         "original_features": original_features, "sensitivity_features": features})
    trees = pd.read_csv(S58 / "05_harmonized_validation/04_LOYO_yearly_comparison.csv")
    ridges = pd.read_csv(S59 / "02_ridge_validation/04_Ridge_LOYO_yearly.csv")
    baseline = normalize_baseline(pd.concat([trees, ridges], ignore_index=True))
    require(len(baseline) == 40 and not baseline.duplicated(["model", "year"]).any(), "Original per-year metrics incomplete")
    require(baseline.n_train.eq(145044).all() and baseline.n_test.eq(16116).all(), "Original fold sizes not recovered")
    for model in MODELS:
        require(sorted(baseline.loc[baseline.model.eq(model), "year"]) == YEARS, "Original model-years incomplete")
    audit = check_original_predictions(df, baseline, tree["metric_row"])
    save_csv(audit, DIRS["input"] / "04_original_OOF_metric_recheck.csv")
    save_csv(baseline, DIRS["results"] / "01_original_LOYO_yearly.csv")
    outer_rows, inner_rows = [], []
    all_idx = np.arange(len(df))
    for year in YEARS:
        test_idx = registry.loc[registry.year.eq(year), "sample_id"].to_numpy()
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
        require(np.array_equal(train_idx, np.flatnonzero(df.year.to_numpy() != year)), "Original tree and registry train rows differ")
        require(np.array_equal(test_idx, np.flatnonzero(df.year.to_numpy() == year)), "Original tree and registry test rows differ")
        outer_rows.append({"year": year, "n_train": len(train_idx), "n_test": len(test_idx),
                           "train_indices_sha256": index_hash(train_idx), "test_indices_sha256": index_hash(test_idx),
                           "train_years": ";".join(map(str, sorted(df.iloc[train_idx].year.unique()))),
                           "tree_random_state": 42 + year, "target_year_absent_from_training": True})
        for fold, (itr, iva) in enumerate(GroupKFold(n_splits=5).split(X.iloc[train_idx], y.iloc[train_idx], groups=df.iloc[train_idx].year), 1):
            tr, va = train_idx[itr], train_idx[iva]
            require(year not in set(df.iloc[tr].year) | set(df.iloc[va].year), "Outer test year entered inner CV")
            require(not set(df.iloc[tr].year) & set(df.iloc[va].year), "Ridge inner year groups overlap")
            inner_rows.append({"outer_year": year, "inner_fold": fold, "n_train": len(tr), "n_valid": len(va),
                               "train_ids_sha256": index_hash(tr), "valid_ids_sha256": index_hash(va),
                               "validation_years": ";".join(map(str, sorted(df.iloc[va].year.unique()))), "status": "PASS"})
    save_csv(pd.DataFrame(outer_rows), DIRS["input"] / "05_outer_fold_identity.csv")
    save_csv(pd.DataFrame(inner_rows), DIRS["input"] / "06_Ridge_inner_fold_isolation.csv")
    gate = {"status": "PASS", "rows": len(df), "grid_cells": int(df.grid_id.nunique()), "years": YEARS,
            "data_sha256": sha(DATA), "loyo_logical_sha256": logical,
            "original_features": original_features, "sensitivity_features": features,
            "other_covariates_changed": 0, "target_changed": 0, "folds_changed": 0,
            "historical_OOF_yearly_metric_checks_passed": len(audit),
            "strict_original_procedure_recovered": True}
    dump(DIRS["input"] / "07_preflight_gate.json", gate)
    ridge["FEATURES"] = features
    log("PRECHECK PASS: original software/data/40 original yearly OOF metrics/10 outer folds/50 Ridge inner folds matched")
    return df, X, y, baseline, before, gate


def summarize(yearly, scenario):
    rows = []
    for model in MODELS:
        d = yearly.loc[yearly.model.eq(model)].sort_values("year")
        rows.append({"model": model, "scenario": scenario, "n_years": len(d),
                     **{f"fold_mean_{k}": float(d[k].mean()) for k in ("R2", "RMSE", "MAE")},
                     **{f"fold_SD_{k}": float(d[k].std(ddof=1)) for k in ("R2", "RMSE", "MAE")},
                     "minimum_yearly_R2": float(d.R2.min()), "maximum_yearly_R2": float(d.R2.max()),
                     "negative_R2_year_count": int(d.R2.lt(0).sum()),
                     "negative_R2_years": ";".join(map(str, d.loc[d.R2.lt(0), "year"].tolist()))})
    return pd.DataFrame(rows)


def run():
    tree, provenance_tree = original_namespace(TREE_SOURCE, {"build_model", "metric_row"})
    ridge, provenance_ridge = original_namespace(RIDGE_SOURCE, {"make_pipeline", "choose_alpha", "fit_outer_fold", "metric_row"})
    dump(DIRS["input"] / "00_original_function_provenance.json", provenance_tree + provenance_ridge)
    df, X, y, baseline, protected_before, gate = preflight(tree, ridge)
    if "--preflight-only" in sys.argv:
        log("PREFLIGHT ONLY COMPLETE; no models fitted")
        return
    metrics = []
    alpha_details = []
    ridge_coefficients = []
    started = time.time()
    prediction_checks = []
    new_predictions = {}
    for model in MODELS:
        per_model = []
        for year in YEARS:
            train_idx = np.flatnonzero(df.year.to_numpy() != year)
            test_idx = np.flatnonzero(df.year.to_numpy() == year)
            log(f"START {model} LOYO {year}; features=10; n_train={len(train_idx)}; n_test={len(test_idx)}")
            start = time.time()
            if model == "Ridge":
                row, pred, coef, detail = ridge["fit_outer_fold"](
                    X, y, df, train_idx, test_idx, "leave_one_year_out", str(year), "group", df.year.to_numpy())
                alpha_details.append(detail)
                ridge_coefficients.append(coef)
                seed = None
            else:
                seed = tree["RANDOM_STATE"] + int(year)
                estimator = tree["build_model"](model, seed)
                require(estimator.get_params()["random_state"] == seed, "Tree per-year seed mismatch")
                estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
                values = estimator.predict(X.iloc[test_idx])
                test_metrics = tree["metric_row"](y.iloc[test_idx], values)
                row = {f"test_{k}": v for k, v in test_metrics.items()}
                row["selected_alpha"] = np.nan
                pred = df.iloc[test_idx][["sample_id", "grid_id", "year", "month"]].copy()
                pred["observed_AOD"] = y.iloc[test_idx].to_numpy()
                pred["predicted_AOD"] = values
                del estimator
            require(np.isfinite(pred.predicted_AOD.to_numpy()).all(), "Non-finite OOF predictions")
            rec = {"model": model, "year": int(year), "n_train": len(train_idx), "n_test": len(test_idx),
                   "n_features": X.shape[1], "random_state": seed, "selected_alpha": row["selected_alpha"],
                   "R2": float(row["test_R2"]), "RMSE": float(row["test_RMSE"]), "MAE": float(row["test_MAE"]),
                   "mean_bias": float(row["test_mean_bias_pred_minus_obs"]), "elapsed_seconds": time.time() - start}
            pred["model"] = model
            pred["scenario"] = "without_NDVI"
            path = DIRS["predictions"] / f"{model}_LOYO_{year}_without_NDVI.csv.gz"
            pred.to_csv(path, index=False, compression="gzip", float_format="%.17g")
            reread = pd.read_csv(path).sort_values("sample_id")
            require(np.array_equal(reread.sample_id, test_idx), "Saved OOF rows do not match test indices")
            checked = tree["metric_row"](reread.observed_AOD, reread.predicted_AOD)
            require(max(abs(checked[k] - rec[k]) for k in ("R2", "RMSE", "MAE")) < 1e-10, "Saved prediction metric mismatch")
            prediction_checks.append({"model": model, "year": year, "path": str(path), "rows": len(reread), "status": "PASS"})
            metrics.append(rec)
            per_model.append(reread)
            save_csv(pd.DataFrame(metrics), DIRS["results"] / "02_without_NDVI_LOYO_yearly.csv")
            if alpha_details:
                save_csv(pd.concat(alpha_details, ignore_index=True), DIRS["results"] / "06_Ridge_inner_alpha_selection.csv")
            dump(DIRS["logs"] / "progress.json", {"completed_outer_fits": len(metrics), "total_outer_fits": 40,
                 "latest": rec, "elapsed_seconds": time.time() - started})
            log(f"DONE {model} {year}: R2={rec['R2']:.9f}; RMSE={rec['RMSE']:.9f}; MAE={rec['MAE']:.9f}; elapsed={rec['elapsed_seconds']:.1f}s ({len(metrics)}/40)")
        allpred = pd.concat(per_model, ignore_index=True).sort_values("sample_id")
        require(np.array_equal(allpred.sample_id, np.arange(len(df))), f"Missing or repeated OOF coverage: {model}")
        require(np.allclose(allpred.observed_AOD, y, rtol=0, atol=1e-12), "OOF response altered")
        new_predictions[model] = allpred
    new = pd.DataFrame(metrics)
    require(len(new) == 40, "Sensitivity computation incomplete")
    paired = baseline.merge(new, on=["model", "year"], suffixes=("_original", "_without_NDVI"), validate="one_to_one")
    for key in ("R2", "RMSE", "MAE"):
        paired[f"delta_{key}"] = paired[f"{key}_without_NDVI"] - paired[f"{key}_original"]
    paired["R2_sign_unchanged"] = np.sign(paired.R2_original) == np.sign(paired.R2_without_NDVI)
    save_csv(paired, DIRS["results"] / "03_yearwise_paired_comparison.csv")
    summary = pd.concat([summarize(baseline, "original_11_predictors"), summarize(new, "without_NDVI_10_predictors")], ignore_index=True)
    save_csv(summary, DIRS["results"] / "04_LOYO_fold_mean_and_yearly_variability.csv")
    pooled = []
    for model in MODELS:
        oldpred, _ = read_original_predictions(model)
        for scenario, pred in [("original_11_predictors", oldpred), ("without_NDVI_10_predictors", new_predictions[model])]:
            pooled.append({"model": model, "scenario": scenario, "N": len(pred),
                           **tree["metric_row"](pred.observed_AOD, pred.predicted_AOD)})
    save_csv(pd.DataFrame(pooled), DIRS["results"] / "05_pooled_OOF_metrics_SEPARATE_FROM_FOLD_MEANS.csv")
    save_csv(pd.concat(ridge_coefficients, ignore_index=True), DIRS["results"] / "07_Ridge_standardized_coefficients.csv")
    save_csv(pd.DataFrame(prediction_checks), DIRS["input"] / "08_saved_prediction_integrity.csv")
    protected_after = snapshot([Path(record["path"]) for record in protected_before])
    dump(DIRS["input"] / "09_protected_source_hashes_after.json", protected_after)
    require(protected_before == protected_after, "A protected original file changed during computation")
    conclusions = []
    for model in MODELS:
        old = summary.loc[summary.model.eq(model) & summary.scenario.eq("original_11_predictors")].iloc[0]
        cur = summary.loc[summary.model.eq(model) & summary.scenario.eq("without_NDVI_10_predictors")].iloc[0]
        p = paired.loc[paired.model.eq(model)]
        conclusions.append({"model": model, "original_mean_R2": float(old.fold_mean_R2),
            "without_NDVI_mean_R2": float(cur.fold_mean_R2), "delta_mean_R2": float(cur.fold_mean_R2-old.fold_mean_R2),
            "negative_mean_R2_retained": bool(cur.fold_mean_R2 < 0),
            "original_R2_SD": float(old.fold_SD_R2), "without_NDVI_R2_SD": float(cur.fold_SD_R2),
            "original_negative_years": int(old.negative_R2_year_count), "without_NDVI_negative_years": int(cur.negative_R2_year_count),
            "yearwise_R2_signs_changed": int((~p.R2_sign_unchanged).sum()),
            "max_abs_yearwise_R2_delta": float(p.delta_R2.abs().max()),
            "pearson_original_vs_without_NDVI_yearly_R2": float(p.R2_original.corr(p.R2_without_NDVI))})
    result = {"status": "COMPLETED", "strict_control_gate": "PASS", "preflight": gate,
        "models": conclusions, "completed_outer_fits": len(new), "Ridge_inner_fits": len(pd.concat(alpha_details)),
        "all_four_negative_mean_R2_retained": all(r["negative_mean_R2_retained"] for r in conclusions),
        "protected_original_files_unchanged": len(protected_before), "manuscript_files_modified": 0,
        "scientific_target_or_rows_changed": 0, "only_feature_removed": "NDVI_final", "total_fit_elapsed_seconds": time.time()-started,
        "interpretation_guardrail": "This tests sensitivity of the LOYO conclusion to removing NDVI. It does not establish that original NDVI filling was free of temporal look-ahead; no broader leakage-free claim is made."}
    dump(DIRS["results"] / "08_summary.json", result)
    log("COMPLETE: 40/40 outer LOYO fits; 300 Ridge inner fits; protected original sources unchanged")
    log(json.dumps(conclusions, ensure_ascii=False))


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        log(f"STOP: {type(error).__name__}: {error}")
        traceback.print_exc()
        dump(DIRS["logs"] / "STOP_OR_FAILURE.json", {"error": str(error), "type": type(error).__name__, "no_approximate_substitution": True})
        raise
