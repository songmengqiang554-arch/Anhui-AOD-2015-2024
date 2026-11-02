"""Lightweight verification of saved data/predictions. No model training or imports."""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import warnings
import zipfile

warnings.filterwarnings("ignore", message="Pandas requires version")
import numpy as np
import pandas as pd


FEATURES = ["t2m", "d2m", "RH", "u10", "v10", "blh", "tp", "dem_traceable",
            "NDVI_final", "month_sin", "month_cos"]
EXPECTED_DATA_HASH = "076dbe16b1afe8abe5b73ab96a42b61bbba512ac4dcc2d8e689ab5d4250bd5be"
R2_TOL = 2e-7
ERROR_TOL = 2e-8


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(y, pred):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    error = pred - y
    return {"R2": float(1 - np.sum(error ** 2) / np.sum((y - y.mean()) ** 2)),
            "RMSE": float(np.sqrt(np.mean(error ** 2))), "MAE": float(np.mean(np.abs(error))),
            "mean_bias": float(np.mean(error))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    qa_dir = Path(args.output_dir).resolve() if args.output_dir else None
    checks, comparisons, fold_rows = [], [], []

    def require(item, condition, details):
        checks.append({"item": item, "status": "PASS" if bool(condition) else "FAIL", "details": details})
        if not condition:
            print(f"FAIL: {item}: {details}", flush=True)

    def read(relative, **kwargs):
        kwargs.setdefault("low_memory", False)
        path = root / relative
        if not path.exists() and path.suffix == ".csv":
            path = Path(str(path) + ".gz")
        return pd.read_csv(path, **kwargs)

    print("Checking dataset, fixed sample IDs and splits", flush=True)
    data_path = root / "01_processed_data/frozen/model_inputs.csv.gz"
    require("frozen_file_SHA256", digest(data_path) == EXPECTED_DATA_HASH, str(data_path))
    df = pd.read_csv(data_path)
    require("stable_original_sample_id", np.array_equal(df.sample_id.to_numpy(), np.arange(len(df))), "zero-based original row order")
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    require("data_shape", len(df) == 161160 and df.grid_id.nunique() == 1343,
            f"rows={len(df)}, grid_count={df.grid_id.nunique()}")
    require("complete_120_months_per_grid", df.groupby("grid_id").size().eq(120).all(), "all grids have 120 records")
    require("unique_grid_year_month", not df.duplicated(["grid_id", "year", "month"]).any(), "no duplicate grid-month")
    require("period_2015_2024", set(df.year) == set(range(2015, 2025)) and set(df.month) == set(range(1, 13)), "10 years, 12 months")
    require("finite_target_and_11_features", np.isfinite(df[["AOD"] + FEATURES].to_numpy()).all(), ",".join(FEATURES))
    require("NTL_and_identity_not_predictors", not set(FEATURES) & {"NTL", "NTL_traceable", "grid_id", "lon", "lat", "year"},
            "filename including NTL does not imply NTL as predictor")

    # This comparison retains the exact original row order; sample_id is never regenerated after sorting.
    aod = read("01_processed_data/intermediate/AOD_0p1deg_grid_monthly_2015_2024.csv")
    merged = df[["grid_id", "year", "month", "AOD"]].merge(
        aod[["grid_id", "year", "month", "AOD"]], on=["grid_id", "year", "month"],
        suffixes=("_model", "_intermediate"), validate="one_to_one")
    max_aod_diff = float(np.max(np.abs(merged.AOD_model - merged.AOD_intermediate)))
    require("AOD_intermediate_equals_frozen_target", len(merged) == len(df) and max_aod_diff == 0,
            f"n={len(merged)}; max_abs_diff={max_aod_diff}")

    registries = {}
    for label, filename in {
        "random_split": "random_split_indices.csv", "spatial_groupkfold": "grid_group_fold_registry.csv",
        "spatial_leave_region_out": "spatial_region_fold_registry.csv", "leave_one_year_out": "loyo_fold_registry.csv",
    }.items():
        reg = read(f"02_fixed_splits/{filename}")
        require(f"{label}_sample_coverage", len(reg) == len(df) and reg.sample_id.nunique() == len(df)
                and set(reg.sample_id) == set(df.sample_id), filename)
        reg = reg.set_index("sample_id").loc[df.sample_id]
        require(f"{label}_grid_identity", np.array_equal(reg.grid_id.to_numpy(), df.grid_id.to_numpy()), filename)
        registries[label] = reg
    random_reg = registries["random_split"]
    require("random_80_20", random_reg.split.eq("train").sum() == 128928 and random_reg.split.eq("test").sum() == 32232,
            str(random_reg.split.value_counts().to_dict()))
    require("grid_group_disjoint", registries["spatial_groupkfold"].groupby("grid_id").test_fold.nunique().eq(1).all()
            and registries["spatial_groupkfold"].test_fold.nunique() == 5, "five test folds; one fold per grid")
    region = registries["spatial_leave_region_out"]
    per_grid = read("02_fixed_splits/02_spatial_cluster_registry.csv").set_index("grid_id")
    require("region_labels_match_per_grid_source", np.array_equal(region.spatial_cluster.to_numpy(),
            per_grid.loc[df.grid_id, "spatial_cluster"].to_numpy()), "locked coordinates-to-label registry")
    require("region_group_disjoint", region.groupby("grid_id").spatial_cluster.nunique().eq(1).all()
            and region.spatial_cluster.nunique() == 5, "five region labels; not a new geometric contiguity proof")
    require("LOYO_year_identity", np.array_equal(registries["leave_one_year_out"].loyo_test_year.to_numpy(), df.year.to_numpy()),
            "each record assigned to its own held-out year")

    master = read("04_locked_results/step59/comparison/01_four_model_metrics_master.csv")
    ridge = read("04_locked_results/step59/ridge/05_Ridge_predictions_OOF.csv")
    original_yearly = read("04_locked_results/step77/results/01_original_LOYO_yearly.csv")
    yearly_reference_checks = []
    scheme_files = {
        "random_split": ("02_exact_reproduction", "random_test_predictions"),
        "spatial_groupkfold": ("02_exact_reproduction", "grid_group_OOF_predictions"),
        "spatial_leave_region_out": ("06_spatial_region_sensitivity", "spatial_region_OOF_predictions"),
        "leave_one_year_out": ("05_harmonized_validation", "LOYO_OOF_predictions"),
    }

    def check_prediction_frame(frame, model, scheme, fold_column):
        ids = frame.sample_id.to_numpy(dtype=int)
        valid_ids = len(ids) == len(set(ids)) and np.isin(ids, df.sample_id).all()
        expected = random_reg.index[random_reg.split.eq("test")] if scheme == "random_split" else df.sample_id
        require(f"{model}_{scheme}_prediction_coverage", valid_ids and set(ids) == set(expected), f"n={len(frame)}")
        if not valid_ids:
            raise RuntimeError("Invalid sample IDs; cannot safely verify predictions")
        y = df.loc[ids, "AOD"].to_numpy()
        target_error = float(np.max(np.abs(y - frame.observed_AOD.to_numpy())))
        require(f"{model}_{scheme}_observed_target", target_error < 1e-12, f"max_abs_diff={target_error}")
        require(f"{model}_{scheme}_finite_predictions", np.isfinite(frame.predicted_AOD.to_numpy()).all(), "saved predictions only")
        if scheme != "random_split":
            column = {"spatial_groupkfold": "test_fold", "spatial_leave_region_out": "spatial_cluster",
                      "leave_one_year_out": "loyo_test_year"}[scheme]
            require(f"{model}_{scheme}_fold_identity", np.array_equal(
                frame[fold_column].to_numpy(dtype=int), registries[scheme].loc[ids, column].to_numpy(dtype=int)), column)

    for model in ["Ridge", "RandomForest", "HistGradientBoosting", "XGBoost"]:
        for scheme, (directory, suffix) in scheme_files.items():
            print(f"Saved prediction check: {model}, {scheme}", flush=True)
            if model == "Ridge":
                frame = ridge[ridge.validation_scheme.eq(scheme)].copy()
                fold_col = "fold"
            else:
                frame = read(f"04_locked_results/step58/{directory}/{model}_{suffix}.csv.gz")
                fold_col = "fold_or_year"
            if scheme != "random_split":
                # Ridge stores random labels and numeric folds in one CSV column.
                frame[fold_col] = pd.to_numeric(frame[fold_col], errors="raise").astype("int64")
            expected_folds = 1 if scheme == "random_split" else (10 if scheme == "leave_one_year_out" else 5)
            require(f"{model}_{scheme}_fold_count", frame[fold_col].nunique() == expected_folds,
                    f"expected={expected_folds}; actual={frame[fold_col].nunique()}")
            check_prediction_frame(frame, model, scheme, fold_col)
            pooled = metrics(frame.observed_AOD, frame.predicted_AOD)
            fold_stats = []
            for fold, subset in frame.groupby(fold_col):
                values = metrics(subset.observed_AOD, subset.predicted_AOD)
                row = {"model": model, "validation": scheme, "fold": fold, "n": len(subset), **values}
                fold_rows.append(row)
                fold_stats.append(values)
                if scheme == "leave_one_year_out":
                    reference = original_yearly[(original_yearly.model == model) & (original_yearly.year == int(fold))].iloc[0]
                    ok = all(abs(values[k] - float(reference[k])) <= (R2_TOL if k == "R2" else ERROR_TOL)
                             for k in ["R2", "RMSE", "MAE"])
                    yearly_reference_checks.append(ok)
            means = {k: float(np.mean([f[k] for f in fold_stats])) for k in pooled}
            ref = master[(master.model == model) & (master.validation == scheme)].iloc[0]
            for metric in ["R2", "RMSE", "MAE", "mean_bias"]:
                tolerance = R2_TOL if metric == "R2" else ERROR_TOL
                difference = means[metric] - float(ref[metric])
                passed = abs(difference) <= tolerance
                comparisons.append({"model": model, "validation": scheme, "metric": metric,
                                    "aggregation": "test_only" if scheme == "random_split" else "fold_mean",
                                    "recalculated": means[metric], "locked": float(ref[metric]),
                                    "difference": difference, "tolerance": tolerance, "status": "PASS" if passed else "FAIL"})
            if scheme != "random_split":
                difference = pooled["R2"] - float(ref.pooled_R2)
                comparisons.append({"model": model, "validation": scheme, "metric": "R2", "aggregation": "pooled_OOF",
                                    "recalculated": pooled["R2"], "locked": float(ref.pooled_R2),
                                    "difference": difference, "tolerance": R2_TOL,
                                    "status": "PASS" if abs(difference) <= R2_TOL else "FAIL"})
        print(f"Completed saved predictions: {model}", flush=True)
    require("all_locked_master_metric_checks", all(x["status"] == "PASS" for x in comparisons),
            f"{len(comparisons)} comparisons; R2 tolerance={R2_TOL}; error tolerance={ERROR_TOL}")
    require("40_original_LOYO_yearly_reference_checks", len(yearly_reference_checks) == 40 and all(yearly_reference_checks),
            f"{sum(yearly_reference_checks)}/{len(yearly_reference_checks)}")

    print("Checking saved NDVI-removal sensitivity, not fitting models", flush=True)
    without_ref = read("04_locked_results/step77/results/02_without_NDVI_LOYO_yearly.csv")
    sensitivity_rows = []
    for path in sorted((root / "04_locked_results/step77/predictions").glob("*_without_NDVI.csv.gz")):
        frame = pd.read_csv(path)
        model = frame.model.iloc[0]
        year = int(frame.year.iloc[0])
        ids = frame.sample_id.to_numpy(dtype=int)
        expected = df.loc[df.year.eq(year), "sample_id"]
        ok = len(frame) == 16116 and frame.sample_id.nunique() == 16116 and set(ids) == set(expected)
        target_error = float(np.max(np.abs(frame.observed_AOD.to_numpy() - df.loc[ids, "AOD"].to_numpy())))
        values = metrics(frame.observed_AOD, frame.predicted_AOD)
        ref = without_ref[(without_ref.model == model) & (without_ref.year == year)].iloc[0]
        metric_ok = all(abs(values[k] - float(ref[k])) <= (R2_TOL if k == "R2" else ERROR_TOL)
                        for k in ["R2", "RMSE", "MAE"])
        original = original_yearly[(original_yearly.model == model) & (original_yearly.year == year)].iloc[0]
        sensitivity_rows.append({"model": model, "year": year, "n": len(frame), **values,
                                 "original_R2": float(original.R2),
                                 "R2_sign_unchanged": np.sign(values["R2"]) == np.sign(float(original.R2)),
                                 "sample_and_target_check": bool(ok and target_error < 1e-12),
                                 "saved_metric_check": metric_ok})
    require("NDVI_40_model_year_predictions", len(sensitivity_rows) == 40
            and len({(r["model"], r["year"]) for r in sensitivity_rows}) == 40
            and all(r["sample_and_target_check"] and r["saved_metric_check"] for r in sensitivity_rows),
            "40 saved model-year files, original targets and outer folds")
    require("NDVI_R2_signs_unchanged", all(r["R2_sign_unchanged"] for r in sensitivity_rows), "40/40 model-year signs")
    sensitivity_summary = []
    saved_summary = json.loads((root / "04_locked_results/step77/results/08_summary.json").read_text(encoding="utf-8"))
    for model in ["Ridge", "RandomForest", "HistGradientBoosting", "XGBoost"]:
        subset = [r for r in sensitivity_rows if r["model"] == model]
        value = float(np.mean([r["R2"] for r in subset]))
        orig = float(original_yearly.loc[original_yearly.model.eq(model), "R2"].mean())
        ref = next(r for r in saved_summary["models"] if r["model"] == model)
        require(f"NDVI_{model}_mean_R2_summary", abs(value - ref["without_NDVI_mean_R2"]) < R2_TOL,
                f"original={orig}, without_NDVI={value}")
        sensitivity_summary.append({"model": model, "original_mean_R2": orig, "without_NDVI_mean_R2": value,
                                    "all_yearly_signs_unchanged": all(r["R2_sign_unchanged"] for r in subset),
                                    "both_means_negative": orig < 0 and value < 0})

    print("Checking archived trend tables and independent count summaries", flush=True)
    trend = read("04_locked_results/step78/02_original_and_seasonal_reproduction.csv")
    corr = read("04_locked_results/step78/04_correlated_seasonal_sensitivity.csv")
    annual_sig = set(trend.loc[(trend.original_slope < 0) & (trend.original_p_fdr < .05), "grid_id"])
    seasonal_sig = set(trend.loc[(trend.seasonal_slope < 0) & (trend.seasonal_p_fdr < .05), "grid_id"])
    corr_sig = set(corr.loc[(corr.correlated_seasonal_slope < 0) & (corr.correlated_seasonal_p_fdr < .05), "grid_id"])
    medians = [float(trend.original_slope.median()), float(trend.seasonal_slope.median()),
               float(corr.correlated_seasonal_slope.median())]
    require("trend_1343_complete_grid_tables", len(trend) == 1343 and trend.grid_id.nunique() == 1343
            and set(corr.grid_id) == set(trend.grid_id), "saved results only; no trend test rerun")
    trend_summary = {"annual_significant": len(annual_sig), "ordinary_seasonal_significant": len(seasonal_sig),
                     "covariance_adjusted_seasonal_significant": len(corr_sig),
                     "annual_percent": 100 * len(annual_sig) / 1343, "ordinary_seasonal_percent": 100 * len(seasonal_sig) / 1343,
                     "covariance_adjusted_seasonal_percent": 100 * len(corr_sig) / 1343,
                     "ordinary_seasonal_only_vs_adjusted": len(seasonal_sig - corr_sig),
                     "annual_only_vs_adjusted": len(annual_sig - corr_sig),
                     "adjusted_only_vs_annual": len(corr_sig - annual_sig),
                     "median_annual_slope": medians[0], "median_ordinary_seasonal_slope": medians[1],
                     "median_covariance_adjusted_seasonal_slope": medians[2]}
    require("trend_significant_counts", [len(annual_sig), len(seasonal_sig), len(corr_sig)] == [1325, 1341, 1325], str(trend_summary))
    require("trend_significant_sets_not_conflated", len(seasonal_sig - corr_sig) == 16
            and len(annual_sig - corr_sig) == 5 and len(corr_sig - annual_sig) == 5,
            "16 fewer than ordinary seasonal; annual and adjusted sets differ by five each way")
    trend_ref = json.loads((root / "04_locked_results/step78/06_summary.json").read_text(encoding="utf-8"))
    expected_medians = [trend_ref[k]["median_slope"] for k in ["annual_standard_MK_BH", "ordinary_seasonal_MK_BH",
                                                             "correlation_adjusted_seasonal_MK_CC_BH"]]
    require("trend_medians_equal_saved_summary", np.allclose(medians, expected_medians, rtol=0, atol=1e-12), str(medians))

    print("Checking repository file hashes", flush=True)
    copied = read("file_manifest.csv")
    all_copy_ok = all((root / r.relative_path).stat().st_size == r.bytes and digest(root / r.relative_path) == r.sha256
                      for r in copied.itertuples(index=False))
    require("all_repository_file_hashes", all_copy_ok, f"{len(copied)} manifested files")
    actual_names = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
                    and ".git" not in p.relative_to(root).parts}
    require("manifest_covers_all_files", actual_names == set(copied.relative_path) | {"file_manifest.csv"},
            "manifest excludes only itself; no implicit output files")
    for item in checks:
        item["details"] = str(item["details"])
    failures = [c for c in checks if c["status"] == "FAIL"]
    summary = {
        "status": "PASS" if not failures else "FAIL", "check_count": len(checks), "fail_count": len(failures),
        "dataset": {"rows": len(df), "grid_count": int(df.grid_id.nunique()), "months_per_grid": 120,
                    "features": FEATURES, "sha256": digest(data_path), "AOD_intermediate_max_abs_diff": max_aod_diff},
        "master_metric_comparisons": len(comparisons), "main_prediction_sets": 16,
        "metric_R2_absolute_tolerance": R2_TOL, "metric_error_absolute_tolerance": ERROR_TOL,
        "CSV_reader": "low_memory=False; nonrandom fold identifiers normalized to int64 before grouping",
        "max_abs_locked_metric_difference": max(abs(x["difference"]) for x in comparisons),
        "NDVI_removed_saved_model_year_files": len(sensitivity_rows), "NDVI_summary": sensitivity_summary,
        "trend_summary": trend_summary, "repository_manifested_files": len(copied),
        "models_fitted": 0, "original_scientific_scripts_executed": 0,
        "raw_to_grid_reextraction_run": False, "trend_tests_rerun": False,
        "verification_scope": "saved predictions, targets, fixed registry identity, recorded metrics and counts only",
        "not_certified": ["portable end-to-end raw-to-model pipeline", "original NDVI interpolation history",
                          "data/code redistribution permission", "clean environment model reproduction"]}
    if qa_dir is not None and not args.check_only:
        if qa_dir == root or root in qa_dir.parents:
            raise ValueError("Verification output must be outside the immutable repository")
        qa_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(checks).to_csv(qa_dir / "check_inventory.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(comparisons).to_csv(qa_dir / "metric_comparisons.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(fold_rows).to_csv(qa_dir / "saved_prediction_fold_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(sensitivity_rows).to_csv(qa_dir / "NDVI_saved_prediction_checks.csv", index=False, encoding="utf-8-sig")
        (qa_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("VERIFICATION_SUMMARY_JSON=" + json.dumps(summary, ensure_ascii=True, allow_nan=False), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
