from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from itertools import combinations
from pathlib import Path
from xml.etree import ElementTree as ET

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors


ROOT = Path('<LOCAL_PATH>')
STEP58 = ROOT / "Q2_upgrade_outputs" / "58_CONTROLLED_MODEL_REPRODUCTION_AND_SELECTION_GATE"
STEP59 = ROOT / "Q2_upgrade_outputs" / "59_FOUR_MODEL_COMPARISON_AND_FINAL_SELECTION"
OUT = ROOT / "Q2_upgrade_outputs" / "60_RF_PRIMARY_MODEL_FINAL_EVIDENCE"

DIRS = {
    "input": OUT / "00_input_lock",
    "model": OUT / "01_RF_explanatory_model",
    "shap": OUT / "02_RF_SHAP",
    "stability": OUT / "03_RF_SHAP_stability",
    "oof": OUT / "04_RF_OOF_residuals",
    "moran": OUT / "05_RF_residual_Moran",
    "figures": OUT / "06_final_model_figures",
    "locked": OUT / "07_locked_results",
    "plans": OUT / "08_manuscript_update_plan",
    "qa": OUT / "09_QA_and_logs",
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

DATA = STEP58 / "00_frozen_inputs" / "Anhui_AOD_Model_Dataset_with_traceable_DEM_NTL.csv"
FEATURE_SCHEMA = STEP58 / "00_frozen_inputs" / "02_feature_schema.csv"
PARAMS = STEP58 / "00_frozen_inputs" / "04_model_parameter_inventory.csv"
FINGERPRINT = STEP58 / "00_frozen_inputs" / "01_frozen_dataset_fingerprint.json"
SPLIT_HASHES = STEP58 / "03_split_registry" / "split_hashes.json"
RANDOM_REG = STEP58 / "03_split_registry" / "random_split_indices.csv"
GRID_REG = STEP58 / "03_split_registry" / "grid_group_fold_registry.csv"
LOYO_REG = STEP58 / "03_split_registry" / "loyo_fold_registry.csv"
REGION_REG = STEP58 / "03_split_registry" / "spatial_region_fold_registry.csv"
GRID_OOF = STEP58 / "02_exact_reproduction" / "RandomForest_grid_group_OOF_predictions.csv.gz"
REGION_OOF = (
    STEP58
    / "06_spatial_region_sensitivity"
    / "RandomForest_spatial_region_OOF_predictions.csv.gz"
)
STEP59_METRICS = (
    STEP59 / "03_four_model_comparison" / "01_four_model_metrics_master.csv"
)
STEP59_LOYO = (
    STEP59 / "03_four_model_comparison" / "05_four_model_LOYO_yearly_comparison.csv"
)
OLD_MORAN_SCRIPT = (
    ROOT / "Q2_upgrade_outputs" / "33_SCI_submission_gap_fix" / "scripts"
    / "run_sci_submission_gap_fix.py"
)
OLD_MORAN_RESULTS = (
    ROOT / "Q2_upgrade_outputs" / "33_SCI_submission_gap_fix" / "csv"
    / "residual_morans_i_results.csv"
)

EN_DOC = (
    ROOT / "Q2_upgrade_outputs" / "55_EMA_ABSTRACT_COMPLIANCE_FIX"
    / "01_english_working" / "EMA_English_Manuscript_Abstract_Compliant_Working.docx"
)
CN_DOC = (
    ROOT / "Q2_upgrade_outputs" / "52_EMA_AUTOCORRELATION_DISCLOSURE_FIX"
    / "02_chinese" / "安徽省AOD论文_一阶自相关披露修订工作版.docx"
)
OR_DOC = (
    ROOT / "Q2_upgrade_outputs" / "53_EMA_ONLINE_RESOURCE_CLEANUP"
    / "02_online_resource_working" / "EMA_Online_Resource_1_Clean_Working.docx"
)

FEATURES = [
    "t2m",
    "d2m",
    "RH",
    "u10",
    "v10",
    "blh",
    "tp",
    "dem_traceable",
    "NDVI_final",
    "month_sin",
    "month_cos",
]
TARGET = "AOD"
RF_PARAMS = {
    "n_estimators": 50,
    "max_depth": 20,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}
SHAP_SAMPLE_SEED = 2026
MORAN_SEED = 2026
MORAN_PERMUTATIONS = 999
SHAP_APPROXIMATE = True

LOCKED_R2 = {
    ("Ridge", "random_split"): 0.400622,
    ("Ridge", "spatial_groupkfold"): 0.398301,
    ("Ridge", "spatial_leave_region_out"): 0.280217,
    ("Ridge", "leave_one_year_out"): -0.004537,
    ("RandomForest", "random_split"): 0.969687,
    ("RandomForest", "spatial_groupkfold"): 0.970206,
    ("RandomForest", "spatial_leave_region_out"): 0.773623,
    ("RandomForest", "leave_one_year_out"): -0.048782,
    ("HistGradientBoosting", "random_split"): 0.833044,
    ("HistGradientBoosting", "spatial_groupkfold"): 0.834986,
    ("HistGradientBoosting", "spatial_leave_region_out"): 0.666192,
    ("HistGradientBoosting", "leave_one_year_out"): -0.045719,
    ("XGBoost", "random_split"): 0.714571,
    ("XGBoost", "spatial_groupkfold"): 0.711277,
    ("XGBoost", "spatial_leave_region_out"): 0.553396,
    ("XGBoost", "leave_one_year_out"): -0.026908,
}

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

PALETTE = {
    "Ridge": "#7F8C8D",
    "RandomForest": "#2A6FBB",
    "HistGradientBoosting": "#55A868",
    "XGBoost": "#C44E52",
}
VALIDATION_LABELS = {
    "random_split": "Random split",
    "spatial_groupkfold": "Grid-cell grouped",
    "spatial_leave_region_out": "Spatial-region holdout",
    "leave_one_year_out": "LOYO",
}

RUN_ROWS: list[dict] = []


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def record(stage: str, status: str, detail: str, started: float) -> None:
    RUN_ROWS.append(
        {
            "stage": stage,
            "status": status,
            "detail": detail,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
            "finished_at": now(),
            "elapsed_seconds": round(time.time() - started, 3),
        }
    )
    pd.DataFrame(RUN_ROWS).to_csv(
        DIRS["qa"] / "run_manifest.csv", index=False, encoding="utf-8-sig"
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def save_pub_figure(fig: plt.Figure, folder: Path, stem: str) -> None:
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(folder / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(folder / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(folder / f"{stem}_preview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def metric_row(y_true, y_pred) -> dict:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "mean_bias_pred_minus_obs": float(np.mean(np.asarray(y_pred) - np.asarray(y_true))),
    }


def season_from_month(month: pd.Series) -> pd.Series:
    return month.map(
        {
            12: "DJF",
            1: "DJF",
            2: "DJF",
            3: "MAM",
            4: "MAM",
            5: "MAM",
            6: "JJA",
            7: "JJA",
            8: "JJA",
            9: "SON",
            10: "SON",
            11: "SON",
        }
    )


def input_lock() -> tuple[pd.DataFrame, pd.DataFrame]:
    stage = "input_lock"
    started = time.time()
    log("Loading and validating frozen inputs")
    required = [
        DATA,
        FEATURE_SCHEMA,
        PARAMS,
        FINGERPRINT,
        SPLIT_HASHES,
        RANDOM_REG,
        GRID_REG,
        LOYO_REG,
        REGION_REG,
        GRID_OOF,
        REGION_OOF,
        STEP59_METRICS,
        STEP59_LOYO,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing required frozen inputs: {missing}")

    df = pd.read_csv(DATA)
    if "sample_id" in df.columns:
        raise RuntimeError("Frozen source unexpectedly contains sample_id.")
    df.insert(0, "sample_id", np.arange(len(df), dtype=np.int64))
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    if "season" not in df.columns:
        df["season"] = season_from_month(df["month"])

    reg_region = pd.read_csv(REGION_REG)
    df = df.merge(
        reg_region[["sample_id", "spatial_cluster"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    checks = {
        "rows_161160": len(df) == 161160,
        "grid_count_1343": int(df["grid_id"].nunique()) == 1343,
        "year_range_2015_2024": int(df["year"].min()) == 2015
        and int(df["year"].max()) == 2024,
        "month_range_1_12": int(df["month"].min()) == 1
        and int(df["month"].max()) == 12,
        "required_columns_present": all(
            c in df.columns
            for c in ["sample_id", TARGET, "grid_id", "year", "month", "lon", "lat"]
            + FEATURES
        ),
        "features_no_missing": not df[FEATURES].isna().any().any(),
        "target_no_missing": not df[TARGET].isna().any(),
        "region_assignment_complete": not df["spatial_cluster"].isna().any(),
    }
    forbidden = {
        "AOD",
        "grid_id",
        "lon",
        "lat",
        "year",
        "NTL",
        "NTL_traceable",
    }
    checks["forbidden_not_in_features"] = not bool(set(FEATURES) & forbidden)
    checks["no_AOD_named_derivatives"] = not any(
        "aod" in f.lower() for f in FEATURES
    )

    expected_fingerprint = json.loads(FINGERPRINT.read_text(encoding="utf-8"))
    checks["dataset_hash_matches_step58"] = (
        sha256_file(DATA) == expected_fingerprint["frozen_sha256"]
    )

    random = pd.read_csv(RANDOM_REG)
    grid = pd.read_csv(GRID_REG)
    loyo = pd.read_csv(LOYO_REG)
    region = pd.read_csv(REGION_REG)
    hashes = json.loads(SPLIT_HASHES.read_text(encoding="utf-8"))
    calculated = {
        "random_split_sha256": sha256_text(
            "\n".join(f"{i},{s}" for i, s in enumerate(random["split"]))
        ),
        "grid_group_fold_sha256": sha256_text(
            "\n".join(f"{i},{f}" for i, f in enumerate(grid["test_fold"]))
        ),
        "loyo_sha256": sha256_text(
            "\n".join(f"{i},{y}" for i, y in enumerate(loyo["year"]))
        ),
        "spatial_region_sha256": sha256_text(
            "\n".join(f"{i},{f}" for i, f in enumerate(region["spatial_cluster"]))
        ),
    }
    for key, value in calculated.items():
        checks[f"{key}_matches"] = hashes.get(key) == value

    params = pd.read_csv(PARAMS)
    rf_params = params[params["model"] == "RandomForest"].set_index("parameter")[
        "value"
    ].to_dict()
    param_expected = {
        "n_estimators": "50",
        "max_depth": "20",
        "min_samples_leaf": "2",
        "max_features": "sqrt",
        "random_state": "42",
        "n_jobs": "-1",
    }
    checks["rf_parameters_match"] = all(
        str(rf_params.get(k)) == str(v) for k, v in param_expected.items()
    )

    schema = pd.read_csv(FEATURE_SCHEMA)
    schema_features = schema.loc[
        schema["included_in_RF_HGB_XGB"].astype(str).str.lower() == "true", "field"
    ].tolist()
    checks["feature_schema_set_matches"] = set(schema_features) == set(FEATURES)
    checks["feature_schema_count_matches"] = len(schema_features) == len(FEATURES)

    metrics = pd.read_csv(STEP59_METRICS)
    for (model, validation), expected in LOCKED_R2.items():
        found = metrics.loc[
            (metrics["model"] == model) & (metrics["validation"] == validation), "R2"
        ]
        checks[f"locked_R2_{model}_{validation}"] = (
            len(found) == 1 and abs(float(found.iloc[0]) - expected) < 5e-7
        )

    check_df = pd.DataFrame(
        [{"check": k, "status": "PASS" if v else "FAIL"} for k, v in checks.items()]
    )
    check_df.to_csv(
        DIRS["input"] / "01_input_lock_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    schema_out = schema.copy()
    schema_out["locked_feature_order"] = schema_out["field"].map(
        {f: i + 1 for i, f in enumerate(FEATURES)}
    )
    schema_out.to_csv(
        DIRS["input"] / "02_feature_schema_source.csv",
        index=False,
        encoding="utf-8-sig",
    )
    combined_hashes = {
        "created_at": now(),
        "dataset": {
            "path": str(DATA),
            "sha256": sha256_file(DATA),
            "rows": len(df),
            "grid_count": int(df["grid_id"].nunique()),
            "year_min": int(df["year"].min()),
            "year_max": int(df["year"].max()),
        },
        "features": FEATURES,
        "target": TARGET,
        "RF_parameters": RF_PARAMS,
        "registered_split_hashes": hashes,
        "recalculated_split_hashes": calculated,
        "input_files": {
            str(p): sha256_file(p)
            for p in [
                FEATURE_SCHEMA,
                PARAMS,
                RANDOM_REG,
                GRID_REG,
                LOYO_REG,
                REGION_REG,
                GRID_OOF,
                REGION_OOF,
                STEP59_METRICS,
                STEP59_LOYO,
            ]
        },
    }
    write_json(DIRS["input"] / "03_parameter_and_split_hashes.json", combined_hashes)
    report_lines = [
        "# Step 60 input lock report",
        "",
        f"- Frozen dataset: `{DATA}`",
        f"- Dataset SHA256: `{sha256_file(DATA)}`",
        f"- Rows: {len(df):,}",
        f"- Unique grid cells: {df['grid_id'].nunique():,}",
        f"- Period: {df['year'].min()}-{df['year'].max()}",
        f"- Target: `{TARGET}`",
        f"- Locked no-NTL traceable features ({len(FEATURES)}): "
        + ", ".join(f"`{f}`" for f in FEATURES),
        f"- RF parameters: `{RF_PARAMS}`",
        "",
        "## Gate checks",
        "",
    ]
    report_lines.extend(
        f"- {row.check}: **{row.status}**" for row in check_df.itertuples()
    )
    if not all(checks.values()):
        write_md(DIRS["input"] / "01_input_lock_report.md", "\n".join(report_lines))
        raise RuntimeError(
            "Input gate failed: "
            + ", ".join(k for k, v in checks.items() if not v)
        )
    report_lines += [
        "",
        "**Gate decision: PASS.** All downstream calculations use the Step 58 "
        "frozen dataset, feature schema, RF parameters, and registered splits.",
    ]
    write_md(DIRS["input"] / "01_input_lock_report.md", "\n".join(report_lines))
    record(stage, "PASS", "All frozen input checks passed.", started)
    return df, metrics


def train_explanatory_model(df: pd.DataFrame) -> RandomForestRegressor:
    stage = "full_RF_explanatory_model"
    started = time.time()
    log("Training full-data RF explanatory model")
    model = RandomForestRegressor(**RF_PARAMS)
    fit_start = time.time()
    model.fit(df[FEATURES], df[TARGET])
    fit_seconds = time.time() - fit_start
    model_path = DIRS["model"] / "RF_final_explanatory_model.joblib"
    joblib.dump(model, model_path, compress=3)
    metadata = {
        "purpose": "SHAP interpretation and historical relationship description only",
        "not_for": "replacement of cross-validated performance or future-year forecasting",
        "trained_at": now(),
        "training_rows": len(df),
        "features": FEATURES,
        "target": TARGET,
        "parameters": RF_PARAMS,
        "fit_seconds": fit_seconds,
        "source_dataset_sha256": sha256_file(DATA),
        "model_file": str(model_path),
        "model_sha256": sha256_file(model_path),
        "cross_validated_performance_source": str(STEP59_METRICS),
    }
    write_json(DIRS["model"] / "RF_final_model_metadata.json", metadata)
    (DIRS["model"] / "RF_training_log.txt").write_text(
        "\n".join(
            [
                f"started={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(fit_start))}",
                f"finished={now()}",
                f"fit_seconds={fit_seconds:.3f}",
                f"rows={len(df)}",
                f"features={FEATURES}",
                f"parameters={RF_PARAMS}",
                "warning=Full-data training fit is not a validation metric and must not replace CV.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    record(stage, "PASS", f"Model trained in {fit_seconds:.1f}s.", started)
    return model


def balanced_sample(
    df: pd.DataFrame, n: int, seed: int, group_cols: list[str]
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = list(df.groupby(group_cols, sort=True).groups.items())
    base = n // len(groups)
    extra = n - base * len(groups)
    extra_idx = set(rng.choice(len(groups), size=extra, replace=False).tolist())
    selected: list[int] = []
    for i, (_, idx) in enumerate(groups):
        idx = np.asarray(list(idx), dtype=np.int64)
        take = base + (1 if i in extra_idx else 0)
        if len(idx) < take:
            raise RuntimeError(f"Stratum has {len(idx)} rows but requires {take}.")
        selected.extend(rng.choice(idx, size=take, replace=False).tolist())
    rng.shuffle(selected)
    return df.loc[selected].copy()


def shap_sample_and_compute(
    df: pd.DataFrame, model: RandomForestRegressor
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, float]:
    stage = "full_RF_SHAP"
    started = time.time()
    log("Selecting balanced 20,000-row SHAP sample")
    sample = balanced_sample(
        df, n=20000, seed=SHAP_SAMPLE_SEED,
        group_cols=["year", "month", "spatial_cluster"]
    )
    sample_cols = [
        "sample_id", "grid_id", "year", "month", "season", "spatial_cluster",
        "lon", "lat", TARGET
    ] + FEATURES
    sample[sample_cols].to_csv(
        DIRS["shap"] / "01_SHAP_sample_registry.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summaries = []
    for variable in ["year", "month", "season", "spatial_cluster"]:
        for level, n_rows in sample.groupby(variable).size().items():
            summaries.append(
                {"balance_dimension": variable, "level": level, "sample_count": int(n_rows)}
            )
    balance = pd.DataFrame(summaries)
    balance.to_csv(
        DIRS["shap"] / "01_SHAP_sample_balance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_md(
        DIRS["shap"] / "02_SHAP_sampling_report.md",
        "\n".join(
            [
                "# RF SHAP sampling report",
                "",
                f"- Source rows: {len(df):,}",
                f"- Selected rows: {len(sample):,}",
                f"- Seed: {SHAP_SAMPLE_SEED}",
                "- Selection did not use AOD magnitude.",
                "- Primary strata: year x month x Step 58 spatial region.",
                "- Season balance follows from month balance.",
                f"- Unique years: {sample['year'].nunique()}",
                f"- Unique months: {sample['month'].nunique()}",
                f"- Unique seasons: {sample['season'].nunique()}",
                f"- Unique spatial regions: {sample['spatial_cluster'].nunique()}",
                f"- Year count range: {sample.groupby('year').size().min()}-"
                f"{sample.groupby('year').size().max()}",
                f"- Month count range: {sample.groupby('month').size().min()}-"
                f"{sample.groupby('month').size().max()}",
                f"- Region count range: {sample.groupby('spatial_cluster').size().min()}-"
                f"{sample.groupby('spatial_cluster').size().max()}",
            ]
        ),
    )

    log("Computing TreeExplainer SHAP values for 20,000 rows")
    explainer = shap.TreeExplainer(model)
    Xs = sample[FEATURES]
    values = np.asarray(
        explainer.shap_values(
            Xs, check_additivity=False, approximate=SHAP_APPROXIMATE
        )
    )
    expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    pred = model.predict(Xs)
    reconstructed = expected_value + values.sum(axis=1)
    add_error = reconstructed - pred
    max_abs_error = float(np.max(np.abs(add_error)))
    mean_abs_error = float(np.mean(np.abs(add_error)))

    shap_values = pd.DataFrame(
        values, columns=[f"SHAP_{f}" for f in FEATURES], index=sample.index
    )
    for f in FEATURES:
        shap_values[f"VALUE_{f}"] = sample[f].to_numpy()
    shap_values.insert(0, "sample_id", sample["sample_id"].to_numpy())
    shap_values["expected_value"] = expected_value
    shap_values["RF_prediction"] = pred
    shap_values["SHAP_reconstructed_prediction"] = reconstructed
    shap_values["additivity_error"] = add_error
    shap_values.to_parquet(
        DIRS["shap"] / "03_RF_SHAP_values.parquet", index=False
    )

    importance_rows = []
    for j, feature in enumerate(FEATURES):
        rho, p_value = stats.spearmanr(sample[feature].to_numpy(), values[:, j])
        mean_signed = float(values[:, j].mean())
        mean_abs = float(np.abs(values[:, j]).mean())
        if not np.isfinite(rho) or abs(rho) < 0.15:
            direction = "weak_or_nonmonotonic"
        elif rho > 0:
            direction = "predominantly_positive_model_association"
        else:
            direction = "predominantly_negative_model_association"
        importance_rows.append(
            {
                "feature": feature,
                "mean_abs_SHAP": mean_abs,
                "mean_signed_SHAP": mean_signed,
                "feature_SHAP_spearman_rho": float(rho),
                "feature_SHAP_spearman_p": float(p_value),
                "direction_label": direction,
                "interpretation_boundary": "Model-based contribution; not causal.",
                "tree_explainer_approximate": SHAP_APPROXIMATE,
            }
        )
    importance = pd.DataFrame(importance_rows).sort_values(
        "mean_abs_SHAP", ascending=False
    )
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    importance.to_csv(
        DIRS["shap"] / "04_RF_SHAP_global_importance_source.csv",
        index=False,
        encoding="utf-8-sig",
    )
    importance.head(10).to_csv(
        DIRS["shap"] / "05_RF_SHAP_top10.csv",
        index=False,
        encoding="utf-8-sig",
    )
    additivity = pd.DataFrame(
        [
            {
                "n_samples": len(sample),
                "expected_value": expected_value,
                "max_abs_additivity_error": max_abs_error,
                "mean_abs_additivity_error": mean_abs_error,
                "tolerance": 1e-6,
                "status": "PASS" if max_abs_error <= 1e-6 else "FAIL",
            }
        ]
    )
    additivity.to_csv(
        DIRS["shap"] / "06_RF_SHAP_additivity_check_source.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if max_abs_error > 1e-6:
        raise RuntimeError(f"SHAP additivity failed: max error {max_abs_error}")
    record(
        stage,
        "PASS",
        f"SHAP completed (approximate={SHAP_APPROXIMATE}); "
        f"max additivity error={max_abs_error:.3e}.",
        started,
    )
    return sample, importance, values, expected_value


def draw_shap_figures(
    sample: pd.DataFrame, importance: pd.DataFrame, values: np.ndarray
) -> None:
    stage = "RF_SHAP_figures"
    started = time.time()
    log("Drawing RF SHAP figures")
    order = [FEATURES.index(f) for f in importance["feature"].tolist()]
    ordered_features = importance["feature"].tolist()

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    imp_plot = importance.iloc[::-1]
    ax.barh(
        imp_plot["feature"],
        imp_plot["mean_abs_SHAP"],
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("")
    ax.set_title("RF global feature contribution")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.5)
    save_pub_figure(fig, DIRS["shap"], "Fig_RF_SHAP_Global")

    rng = np.random.default_rng(2026)
    max_points = 5000
    idx = (
        np.arange(len(sample))
        if len(sample) <= max_points
        else np.sort(rng.choice(len(sample), max_points, replace=False))
    )
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    for rank_idx, j in enumerate(order[::-1]):
        y0 = rank_idx
        sv = values[idx, j]
        fv = sample.iloc[idx][FEATURES[j]].to_numpy()
        q1, q99 = np.nanquantile(fv, [0.01, 0.99])
        clipped = np.clip(fv, q1, q99)
        denom = np.nanmax(clipped) - np.nanmin(clipped)
        color_value = (
            (clipped - np.nanmin(clipped)) / denom if denom > 0 else np.zeros_like(clipped)
        )
        jitter = rng.normal(0, 0.07, size=len(idx))
        ax.scatter(
            sv,
            y0 + jitter,
            c=color_value,
            cmap="coolwarm",
            s=4,
            alpha=0.55,
            linewidths=0,
            rasterized=True,
        )
    ax.axvline(0, color="#666666", linewidth=0.6)
    ax.set_yticks(range(len(ordered_features)))
    ax.set_yticklabels(ordered_features[::-1])
    ax.set_xlabel("SHAP value (RF model contribution)")
    ax.set_title("Distribution of RF feature contributions")
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.4)
    sm = mpl.cm.ScalarMappable(cmap="coolwarm", norm=mpl.colors.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Feature value")
    save_pub_figure(fig, DIRS["shap"], "Fig_RF_SHAP_Beeswarm")

    top5 = importance.head(5)["feature"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5))
    axes = axes.ravel()
    dep_idx = (
        np.arange(len(sample))
        if len(sample) <= 8000
        else np.sort(rng.choice(len(sample), 8000, replace=False))
    )
    for ax, feature in zip(axes, top5):
        j = FEATURES.index(feature)
        x = sample.iloc[dep_idx][feature].to_numpy()
        y = values[dep_idx, j]
        ax.scatter(x, y, s=4, alpha=0.22, color="#2A6FBB", linewidths=0)
        quantiles = np.unique(np.quantile(x, np.linspace(0, 1, 21)))
        if len(quantiles) > 3:
            bins = pd.cut(x, bins=quantiles, include_lowest=True, duplicates="drop")
            bdf = pd.DataFrame({"x": x, "y": y, "bin": bins}).groupby(
                "bin", observed=True
            ).agg(x=("x", "median"), y=("y", "median"))
            ax.plot(bdf["x"], bdf["y"], color="#C44E52", linewidth=1.2)
        ax.axhline(0, color="#777777", linewidth=0.5)
        ax.set_title(feature)
        ax.set_xlabel("Feature value")
        ax.set_ylabel("SHAP value")
        ax.grid(color="#EEEEEE", linewidth=0.4)
    axes[-1].axis("off")
    fig.suptitle("Top-five RF SHAP dependence patterns", y=1.01, fontsize=8.5)
    fig.tight_layout()
    save_pub_figure(fig, DIRS["shap"], "Fig_RF_SHAP_Top5_Dependence")
    record(stage, "PASS", "Three SHAP figure families exported.", started)


def shap_stability(
    df: pd.DataFrame, full_importance: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    stage = "RF_SHAP_stability"
    started = time.time()
    log("Running five-fold grid-grouped SHAP stability")
    reg = pd.read_csv(GRID_REG)
    if not np.array_equal(reg["sample_id"].to_numpy(), df["sample_id"].to_numpy()):
        raise RuntimeError("Grid registry sample order differs from frozen frame.")
    fold_rows = []
    fold_importance_vectors: dict[int, pd.Series] = {}
    full_rank = full_importance.set_index("feature")["rank"]
    rng = np.random.default_rng(2026)
    for fold in sorted(reg["test_fold"].unique()):
        train_idx = reg.index[reg["test_fold"] != fold].to_numpy()
        test_idx = reg.index[reg["test_fold"] == fold].to_numpy()
        sample_n = min(1000, len(test_idx))
        sample_idx = np.sort(rng.choice(test_idx, size=sample_n, replace=False))
        model = RandomForestRegressor(**RF_PARAMS)
        fit_started = time.time()
        model.fit(df.iloc[train_idx][FEATURES], df.iloc[train_idx][TARGET])
        explainer = shap.TreeExplainer(model)
        fold_values = np.asarray(
            explainer.shap_values(
                df.iloc[sample_idx][FEATURES],
                check_additivity=False,
                approximate=SHAP_APPROXIMATE,
            )
        )
        mean_abs = np.mean(np.abs(fold_values), axis=0)
        imp = pd.Series(mean_abs, index=FEATURES).sort_values(ascending=False)
        fold_importance_vectors[int(fold)] = imp
        ranks = imp.rank(ascending=False, method="min")
        for feature in FEATURES:
            fold_rows.append(
                {
                    "fold": int(fold),
                    "feature": feature,
                    "mean_abs_SHAP": float(imp[feature]),
                    "rank": int(ranks[feature]),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_SHAP_sample": sample_n,
                    "fit_seconds": time.time() - fit_started,
                }
            )
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(
        DIRS["stability"] / "01_foldwise_SHAP_importance_source.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rank_rows = []
    full_top5 = set(full_importance.head(5)["feature"])
    full_top10 = set(full_importance.head(10)["feature"])
    for fold, imp in fold_importance_vectors.items():
        rank = imp.rank(ascending=False, method="min")
        rho = float(stats.spearmanr(rank.loc[FEATURES], full_rank.loc[FEATURES]).statistic)
        top5 = set(imp.head(5).index)
        top10 = set(imp.head(10).index)
        rank_rows.append(
            {
                "comparison": f"fold_{fold}_vs_full",
                "fold_a": fold,
                "fold_b": "full",
                "spearman_rank_rho": rho,
                "top5_overlap_count": len(top5 & full_top5),
                "top5_overlap_rate": len(top5 & full_top5) / 5,
                "top10_intersection": len(top10 & full_top10),
                "top10_union": len(top10 | full_top10),
                "top10_jaccard": len(top10 & full_top10) / len(top10 | full_top10),
            }
        )
    for a, b in combinations(sorted(fold_importance_vectors), 2):
        imp_a = fold_importance_vectors[a]
        imp_b = fold_importance_vectors[b]
        rank_a = imp_a.rank(ascending=False, method="min")
        rank_b = imp_b.rank(ascending=False, method="min")
        top5_a, top5_b = set(imp_a.head(5).index), set(imp_b.head(5).index)
        top10_a, top10_b = set(imp_a.head(10).index), set(imp_b.head(10).index)
        rank_rows.append(
            {
                "comparison": f"fold_{a}_vs_fold_{b}",
                "fold_a": a,
                "fold_b": b,
                "spearman_rank_rho": float(
                    stats.spearmanr(rank_a.loc[FEATURES], rank_b.loc[FEATURES]).statistic
                ),
                "top5_overlap_count": len(top5_a & top5_b),
                "top5_overlap_rate": len(top5_a & top5_b) / 5,
                "top10_intersection": len(top10_a & top10_b),
                "top10_union": len(top10_a | top10_b),
                "top10_jaccard": len(top10_a & top10_b) / len(top10_a | top10_b),
            }
        )
    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(
        DIRS["stability"] / "02_SHAP_rank_stability_source.csv",
        index=False,
        encoding="utf-8-sig",
    )
    vs_full = rank_df[rank_df["fold_b"] == "full"]
    median_rho = float(vs_full["spearman_rank_rho"].median())
    min_top5 = float(vs_full["top5_overlap_rate"].min())
    median_jaccard = float(vs_full["top10_jaccard"].median())
    if median_rho >= 0.90 and min_top5 >= 0.80 and median_jaccard >= 0.70:
        status = "STABLE"
    elif median_rho >= 0.70 and min_top5 >= 0.60 and median_jaccard >= 0.60:
        status = "MODERATELY_STABLE"
    else:
        status = "UNSTABLE"
    top_by_fold = {
        fold: list(imp.head(10).index) for fold, imp in fold_importance_vectors.items()
    }
    write_md(
        DIRS["stability"] / "03_SHAP_stability_report.md",
        "\n".join(
            [
                "# RF SHAP stability report",
                "",
                f"**Classification: {status}**",
                "",
                "- Validation basis: Step 58 grid-cell grouped five-fold registry.",
                "- Each fold model used only its training portion.",
                "- SHAP sample: up to 1,000 rows from that fold's held-out test portion.",
                "- Test targets were not used for SHAP sample selection.",
                f"- TreeExplainer approximate path decomposition: {SHAP_APPROXIMATE}.",
                f"- Median fold-vs-full Spearman rank rho: {median_rho:.3f}",
                f"- Minimum fold-vs-full Top-5 overlap rate: {min_top5:.3f}",
                f"- Median fold-vs-full Top-10 Jaccard: {median_jaccard:.3f}",
                "",
                "## Fold Top 10",
                "",
            ]
            + [f"- Fold {fold}: {', '.join(features)}" for fold, features in top_by_fold.items()]
            + [
                "",
                "Interpretation boundary: the assessment concerns the stability of "
                "model-based feature contribution rankings, not causal drivers.",
            ]
        ),
    )

    pivot = fold_df.pivot(index="feature", columns="fold", values="rank").loc[FEATURES]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    im = ax.imshow(pivot.to_numpy(), cmap="Blues_r", vmin=1, vmax=len(FEATURES))
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(
        [f"Fold {c}" for c in pivot.columns],
        rotation=28,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{int(pivot.iloc[i, j])}", ha="center", va="center", fontsize=6)
    ax.set_title(f"RF SHAP rank stability: {status}")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Feature rank (1 = highest)")
    fig.tight_layout()
    save_pub_figure(fig, DIRS["stability"], "04_SHAP_stability_plot")
    record(stage, "PASS", f"SHAP stability classified {status}.", started)
    return fold_df, rank_df, status


def build_oof_residuals(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage = "RF_OOF_residuals"
    started = time.time()
    log("Building strict RF OOF residual tables")
    grid_pred = pd.read_csv(GRID_OOF)
    region_pred = pd.read_csv(REGION_OOF)
    meta = df[
        [
            "sample_id", "grid_id", "year", "month", "lon", "lat",
            "spatial_cluster", TARGET,
        ]
    ]

    def merge_one(pred: pd.DataFrame, scheme: str) -> pd.DataFrame:
        out = pred.merge(meta, on="sample_id", how="left", validate="one_to_one")
        if not np.allclose(
            out["observed_AOD"], out[TARGET], atol=0, rtol=0, equal_nan=False
        ):
            raise RuntimeError(f"{scheme} observed AOD differs from frozen target.")
        out["residual"] = out["observed_AOD"] - out["predicted_AOD"]
        out["fold"] = out["fold_or_year"].astype(int)
        out["validation_scheme"] = scheme
        return out[
            [
                "sample_id", "grid_id", "year", "month", "lon", "lat",
                "spatial_cluster", "observed_AOD", "predicted_AOD", "residual",
                "fold", "validation_scheme",
            ]
        ].sort_values("sample_id")

    grid = merge_one(grid_pred, "grid-cell grouped cross-validation")
    region = merge_one(region_pred, "continuous spatial-region holdout")
    grid.to_csv(
        DIRS["oof"] / "01_grid_grouped_OOF_residuals.csv",
        index=False,
        encoding="utf-8-sig",
    )
    region.to_csv(
        DIRS["oof"] / "02_spatial_region_OOF_residuals.csv",
        index=False,
        encoding="utf-8-sig",
    )

    grid_reg = pd.read_csv(GRID_REG).set_index("sample_id")
    region_reg = pd.read_csv(REGION_REG).set_index("sample_id")
    integrity_rows = [
        {
            "validation_scheme": "grid-cell grouped cross-validation",
            "n_predictions": len(grid),
            "unique_sample_ids": grid["sample_id"].nunique(),
            "duplicate_sample_ids": int(grid["sample_id"].duplicated().sum()),
            "all_frozen_samples_covered": set(grid["sample_id"]) == set(df["sample_id"]),
            "fold_matches_registry": bool(
                np.array_equal(
                    grid["fold"].to_numpy(),
                    grid["sample_id"].map(grid_reg["test_fold"]).to_numpy(),
                )
            ),
            "held_out_unit_exclusive": bool(
                (pd.read_csv(GRID_REG).groupby("grid_id")["test_fold"].nunique() == 1).all()
            ),
            "training_residuals_present": False,
            "observed_matches_frozen_target": True,
            "status": "PASS",
        },
        {
            "validation_scheme": "continuous spatial-region holdout",
            "n_predictions": len(region),
            "unique_sample_ids": region["sample_id"].nunique(),
            "duplicate_sample_ids": int(region["sample_id"].duplicated().sum()),
            "all_frozen_samples_covered": set(region["sample_id"]) == set(df["sample_id"]),
            "fold_matches_registry": bool(
                np.array_equal(
                    region["fold"].to_numpy(),
                    region["sample_id"].map(region_reg["spatial_cluster"]).to_numpy(),
                )
            ),
            "held_out_unit_exclusive": bool(
                (
                    pd.read_csv(REGION_REG)
                    .groupby("spatial_cluster")["spatial_cluster"]
                    .nunique()
                    == 1
                ).all()
            ),
            "training_residuals_present": False,
            "observed_matches_frozen_target": True,
            "status": "PASS",
        },
    ]
    integrity = pd.DataFrame(integrity_rows)
    bool_cols = [
        "all_frozen_samples_covered",
        "fold_matches_registry",
        "held_out_unit_exclusive",
        "observed_matches_frozen_target",
    ]
    if not integrity[bool_cols].all().all() or integrity["duplicate_sample_ids"].sum() != 0:
        raise RuntimeError("OOF residual integrity check failed.")
    integrity.to_csv(
        DIRS["oof"] / "03_OOF_residual_integrity_check_source.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_rows = []
    for scheme, values in [
        ("grid-cell grouped cross-validation", grid),
        ("continuous spatial-region holdout", region),
    ]:
        m = metric_row(values["observed_AOD"], values["predicted_AOD"])
        summary_rows.append(
            {
                "validation_scheme": scheme,
                "n_predictions": len(values),
                "n_grid_cells": values["grid_id"].nunique(),
                "residual_mean": float(values["residual"].mean()),
                "residual_sd": float(values["residual"].std(ddof=1)),
                "residual_median": float(values["residual"].median()),
                "residual_min": float(values["residual"].min()),
                "residual_max": float(values["residual"].max()),
                **m,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        DIRS["oof"] / "04_OOF_residual_summary_source.csv",
        index=False,
        encoding="utf-8-sig",
    )
    record(stage, "PASS", "Both OOF residual tables cover all frozen samples.", started)
    return grid, region, integrity, summary_df


def moran_knn(
    residuals: np.ndarray,
    coords: np.ndarray,
    k: int,
    permutations: int,
    seed: int,
) -> dict:
    x = np.asarray(residuals, dtype=float)
    coords = np.asarray(coords, dtype=float)
    valid = np.isfinite(x) & np.isfinite(coords).all(axis=1)
    x = x[valid]
    coords = coords[valid]
    n = len(x)
    if n <= k:
        raise ValueError("Not enough observations for requested k.")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    neighbours = nn.kneighbors(coords, return_distance=False)[:, 1:]
    z = x - x.mean()
    denom = float(np.sum(z**2))
    if denom <= 0:
        raise ValueError("Residual variance is zero.")
    # Row-standardized kNN: each neighbor has weight 1/k and S0=n.
    observed = float(
        np.sum(z[:, None] * z[neighbours] * (1.0 / k)) / denom
    )
    rng = np.random.default_rng(seed)
    perm_values = np.empty(permutations, dtype=float)
    for i in range(permutations):
        zp = rng.permutation(z)
        perm_values[i] = float(
            np.sum(zp[:, None] * zp[neighbours] * (1.0 / k)) / denom
        )
    perm_mean = float(perm_values.mean())
    perm_sd = float(perm_values.std(ddof=1))
    expected = -1.0 / (n - 1)
    z_score = (observed - perm_mean) / perm_sd
    p_two_sided = float(
        (
            np.sum(np.abs(perm_values - perm_mean) >= abs(observed - perm_mean))
            + 1
        )
        / (permutations + 1)
    )
    return {
        "n_grid_cells": n,
        "neighbor_k": k,
        "row_standardized": True,
        "permutations": permutations,
        "seed": seed,
        "morans_I": observed,
        "expected_I": expected,
        "permutation_mean_I": perm_mean,
        "permutation_sd_I": perm_sd,
        "permutation_z_score": z_score,
        "two_sided_permutation_p": p_two_sided,
        "p_value_definition": "two-sided around permutation mean with +1 correction",
        "coordinate_space": "longitude-latitude degrees; fixed Anhui grid",
    }


def audit_and_run_moran(
    grid_oof: pd.DataFrame, region_oof: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage = "RF_residual_Moran"
    started = time.time()
    log("Auditing original Moran method and computing RF OOF Moran diagnostics")
    old = pd.read_csv(OLD_MORAN_RESULTS) if OLD_MORAN_RESULTS.exists() else pd.DataFrame()
    old_line = (
        old.iloc[0].to_dict()
        if not old.empty
        else {"status": "MISSING original result CSV"}
    )
    write_md(
        DIRS["moran"] / "01_original_Moran_method_audit.md",
        f"""# Original XGBoost residual Moran method audit

## Evidence

- Script: `{OLD_MORAN_SCRIPT}`
- Result table: `{OLD_MORAN_RESULTS}`
- Recorded result row: `{old_line}`

## Recovered method

- Residual source: no-NTL **random-split** residuals aggregated by grid.
- Spatial unit: mean residual for each grid cell.
- Weights: directed k-nearest neighbors in longitude-latitude coordinates.
- Main k: 8.
- Row standardization: mathematically equivalent to equal weight `1/k` for each neighbor, implemented through `n/S0 = 1/k`.
- Permutations: 499 in the recorded call.
- Random seed: function default 42.
- p-value: two-sided permutation comparison using `abs(I_perm) >= abs(I_obs)` with a +1 correction.
- The script did not report expected I or a permutation z-score.

## Decision

The historical value Moran's I = 0.362, p = 0.002 belongs to a prior XGBoost/random-split residual workflow. It is **not transferable** to the Random Forest model and is not used as the formal RF residual diagnostic.
""",
    )

    def aggregate(oof: pd.DataFrame) -> pd.DataFrame:
        return (
            oof.groupby("grid_id", as_index=False)
            .agg(
                lon=("lon", "first"),
                lat=("lat", "first"),
                mean_residual=("residual", "mean"),
                residual_sd=("residual", "std"),
                n_records=("residual", "size"),
            )
            .sort_values("grid_id")
        )

    grid_agg = aggregate(grid_oof)
    region_agg = aggregate(region_oof)
    if len(grid_agg) != 1343 or len(region_agg) != 1343:
        raise RuntimeError("Moran aggregation did not yield 1,343 grid cells.")
    grid_agg.to_csv(
        DIRS["moran"] / "RF_grid_grouped_mean_residual_by_grid.csv",
        index=False,
        encoding="utf-8-sig",
    )
    region_agg.to_csv(
        DIRS["moran"] / "RF_spatial_region_mean_residual_by_grid.csv",
        index=False,
        encoding="utf-8-sig",
    )

    grid_rows = []
    region_rows = []
    for k in [4, 8, 12]:
        g = moran_knn(
            grid_agg["mean_residual"].to_numpy(),
            grid_agg[["lon", "lat"]].to_numpy(),
            k=k,
            permutations=MORAN_PERMUTATIONS,
            seed=MORAN_SEED,
        )
        g.update(
            {
                "residual_type": "grid-cell grouped OOF residual averaged by grid",
                "validation_scheme": "grid-cell grouped cross-validation",
            }
        )
        grid_rows.append(g)
        r = moran_knn(
            region_agg["mean_residual"].to_numpy(),
            region_agg[["lon", "lat"]].to_numpy(),
            k=k,
            permutations=MORAN_PERMUTATIONS,
            seed=MORAN_SEED,
        )
        r.update(
            {
                "residual_type": "spatial-region OOF residual averaged by grid",
                "validation_scheme": "continuous spatial-region holdout",
            }
        )
        region_rows.append(r)
    grid_df = pd.DataFrame(grid_rows)
    region_df = pd.DataFrame(region_rows)
    sensitivity = pd.concat([grid_df, region_df], ignore_index=True)
    grid_df.to_csv(
        DIRS["moran"] / "02_RF_grid_grouped_Moran_source.csv",
        index=False,
        encoding="utf-8-sig",
    )
    region_df.to_csv(
        DIRS["moran"] / "03_RF_spatial_region_Moran_source.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensitivity.to_csv(
        DIRS["moran"] / "04_RF_Moran_k_sensitivity_source.csv",
        index=False,
        encoding="utf-8-sig",
    )
    main_g = grid_df.loc[grid_df["neighbor_k"] == 8].iloc[0]
    main_r = region_df.loc[region_df["neighbor_k"] == 8].iloc[0]
    signs_consistent = (
        grid_df["morans_I"].apply(np.sign).nunique() == 1
        and region_df["morans_I"].apply(np.sign).nunique() == 1
    )
    write_md(
        DIRS["moran"] / "05_RF_residual_Moran_report.md",
        f"""# RF OOF residual Moran report

## Main grid-cell grouped OOF result

- Moran's I (k=8): {main_g['morans_I']:.6f}
- Expected I: {main_g['expected_I']:.6f}
- Permutation z-score: {main_g['permutation_z_score']:.3f}
- Two-sided permutation p: {main_g['two_sided_permutation_p']:.4f}
- Permutations: {MORAN_PERMUTATIONS}
- Seed: {MORAN_SEED}
- Residuals: held-out OOF residuals averaged to 1,343 grid cells.

## Continuous spatial-region OOF sensitivity

- Moran's I (k=8): {main_r['morans_I']:.6f}
- Expected I: {main_r['expected_I']:.6f}
- Permutation z-score: {main_r['permutation_z_score']:.3f}
- Two-sided permutation p: {main_r['two_sided_permutation_p']:.4f}

## k sensitivity

- k=4, 8, and 12 have a consistent sign across both OOF schemes: **{signs_consistent}**.
- These statistics diagnose residual spatial structure. They do not identify pollution sources or causal mechanisms.
- The historical XGBoost value 0.362 (p=0.002) is not reused.
""",
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True, sharey=True)
    vmax = max(
        np.abs(grid_agg["mean_residual"]).quantile(0.99),
        np.abs(region_agg["mean_residual"]).quantile(0.99),
    )
    for ax, data, title, result in [
        (
            axes[0],
            grid_agg,
            "Grid-cell grouped OOF",
            main_g,
        ),
        (
            axes[1],
            region_agg,
            "Spatial-region holdout OOF",
            main_r,
        ),
    ]:
        sc = ax.scatter(
            data["lon"],
            data["lat"],
            c=data["mean_residual"],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            s=8,
            linewidths=0,
        )
        ax.set_title(
            f"{title}\nMoran's I={result['morans_I']:.3f}, "
            f"p={result['two_sided_permutation_p']:.3f}"
        )
        ax.set_xlabel("Longitude")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Latitude")
    cbar = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.03)
    cbar.set_label("Mean OOF residual (observed - predicted)")
    fig.suptitle("RF residual spatial structure", y=1.02, fontsize=8.5)
    save_pub_figure(fig, DIRS["moran"], "06_RF_residual_map")
    record(stage, "PASS", "OOF Moran diagnostics completed for k=4,8,12.", started)
    return grid_df, region_df, sensitivity


def draw_four_model_figures(metrics: pd.DataFrame) -> None:
    stage = "four_model_figures"
    started = time.time()
    log("Drawing locked four-model comparison figures")
    model_order = ["Ridge", "RandomForest", "HistGradientBoosting", "XGBoost"]
    val_order = [
        "random_split",
        "spatial_groupkfold",
        "spatial_leave_region_out",
        "leave_one_year_out",
    ]
    x = np.arange(len(val_order))
    width = 0.19
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, model in enumerate(model_order):
        values = [
            float(
                metrics.loc[
                    (metrics["model"] == model)
                    & (metrics["validation"] == v),
                    "R2",
                ].iloc[0]
            )
            for v in val_order
        ]
        ax.bar(
            x + (i - 1.5) * width,
            values,
            width=width,
            color=PALETTE[model],
            label=model.replace("HistGradientBoosting", "HGB"),
        )
    ax.axhline(0, color="#444444", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([VALIDATION_LABELS[v] for v in val_order], rotation=12)
    ax.set_ylabel(r"$R^2$")
    ax.set_ylim(-0.12, 1.04)
    ax.set_title(
        "Historical explanatory performance, spatial generalization,\n"
        "and temporal transferability"
    )
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.19))
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)
    fig.subplots_adjust(bottom=0.25)
    save_pub_figure(fig, DIRS["figures"], "Fig_Four_Model_Four_Validation_R2")

    loyo = pd.read_csv(STEP59_LOYO)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for model in model_order:
        d = loyo[loyo["model"] == model].sort_values("year")
        ax.plot(
            d["year"],
            d["R2"],
            marker="o",
            ms=3.5,
            linewidth=1.1,
            color=PALETTE[model],
            label=model.replace("HistGradientBoosting", "HGB"),
        )
    ax.axhline(0, color="#444444", linewidth=0.7)
    ax.set_xticks(sorted(loyo["year"].unique()))
    ax.set_ylabel(r"LOYO $R^2$")
    ax.set_xlabel("Held-out year")
    ax.set_title("Year-specific temporal transferability")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.19))
    ax.grid(color="#E8E8E8", linewidth=0.5)
    fig.subplots_adjust(bottom=0.25)
    save_pub_figure(fig, DIRS["figures"], "Fig_Four_Model_LOYO_Yearly")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    for ax, metric in zip(axes, ["RMSE", "MAE"]):
        for i, model in enumerate(model_order):
            values = [
                float(
                    metrics.loc[
                        (metrics["model"] == model)
                        & (metrics["validation"] == v),
                        metric,
                    ].iloc[0]
                )
                for v in val_order
            ]
            ax.plot(
                x,
                values,
                marker="o",
                ms=3.4,
                linewidth=1.0,
                color=PALETTE[model],
                label=model.replace("HistGradientBoosting", "HGB"),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["Random", "Grid grouped", "Region holdout", "LOYO"],
            rotation=18,
        )
        ax.set_ylabel(f"{metric} (AOD units)")
        ax.set_title(metric)
        ax.grid(color="#E8E8E8", linewidth=0.5)
    axes[1].legend(ncol=2, loc="upper left", fontsize=6.5)
    fig.suptitle("Error metrics across validation schemes", y=1.01, fontsize=8.5)
    fig.tight_layout()
    save_pub_figure(fig, DIRS["figures"], "Fig_Four_Model_RMSE_MAE")
    record(stage, "PASS", "Three four-model figure families exported.", started)


def lock_results(
    metrics: pd.DataFrame,
    importance: pd.DataFrame,
    stability_status: str,
    grid_moran: pd.DataFrame,
    region_moran: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    stage = "result_lock"
    started = time.time()
    log("Writing locked result sources and manuscript numbers")
    rf_results = metrics[metrics["model"] == "RandomForest"].copy()
    rf_results["interpretation"] = rf_results["validation"].map(
        {
            "random_split": "historical explanatory performance under record-level random split",
            "spatial_groupkfold": "historical spatial generalization to unseen grid cells",
            "spatial_leave_region_out": "continuous spatial-region holdout sensitivity",
            "leave_one_year_out": "temporal transferability to unseen years",
        }
    )
    rf_results.to_csv(
        DIRS["locked"] / "01_RF_primary_model_results_source.csv",
        index=False,
        encoding="utf-8-sig",
    )
    locked_metrics = metrics.copy()
    locked_metrics["locked_status"] = "LOCKED_STEP59"
    locked_metrics.to_csv(
        DIRS["locked"] / "02_four_model_metrics_locked_source.csv",
        index=False,
        encoding="utf-8-sig",
    )
    shap_lock = importance.copy()
    shap_lock["stability_status"] = stability_status
    shap_lock.to_csv(
        DIRS["locked"] / "03_RF_SHAP_ranking_locked_source.csv",
        index=False,
        encoding="utf-8-sig",
    )
    moran_lock = sensitivity.copy()
    moran_lock["locked_status"] = "LOCKED_STEP60"
    moran_lock.to_csv(
        DIRS["locked"] / "04_RF_Moran_results_locked_source.csv",
        index=False,
        encoding="utf-8-sig",
    )
    top5 = importance.head(5)["feature"].tolist()
    g8 = grid_moran[grid_moran["neighbor_k"] == 8].iloc[0]
    r8 = region_moran[region_moran["neighbor_k"] == 8].iloc[0]
    loyo_all_negative = bool(
        (
            metrics.loc[
                metrics["validation"] == "leave_one_year_out", "R2"
            ]
            < 0
        ).all()
    )
    write_md(
        DIRS["locked"] / "05_final_numbers_for_manuscript.md",
        f"""# Final locked numbers for manuscript revision

## Primary model

The final historical spatial explanatory model is the **Random Forest** with the locked Step 58 configuration. It is not a future-year forecasting model.

## Four validation schemes

| Model | Random R2 | Grid-cell grouped R2 | Spatial-region holdout R2 | LOYO mean R2 |
|---|---:|---:|---:|---:|
| Ridge | 0.400622 | 0.398301 | 0.280217 | -0.004537 |
| Random Forest | 0.969687 | 0.970206 | 0.773623 | -0.048782 |
| HistGradientBoosting | 0.833044 | 0.834986 | 0.666192 | -0.045719 |
| XGBoost | 0.714571 | 0.711277 | 0.553396 | -0.026908 |

All four LOYO mean R2 values are negative: **{loyo_all_negative}**.

## RF SHAP

- Top 5: {", ".join(top5)}
- Stability classification: {stability_status}
- Interpretation: model-based feature contribution, not causality.

## RF OOF residual spatial structure

- Grid-cell grouped OOF, k=8: Moran's I = {g8['morans_I']:.6f}, p = {g8['two_sided_permutation_p']:.4f}.
- Continuous spatial-region OOF, k=8: Moran's I = {r8['morans_I']:.6f}, p = {r8['two_sided_permutation_p']:.4f}.
- k sensitivity: 4, 8, and 12 reported for both OOF schemes.

## Values that must be replaced or retired

- Retire the XGBoost primary-model results 0.713/0.711/-0.021 from the primary narrative. They may remain only in the four-model comparison.
- Retire the previous XGBoost SHAP ranking as the primary explanation.
- Do not reuse the prior XGBoost/random-split residual Moran's I = 0.362, p = 0.002 for RF.

## Locked content that remains unchanged

- Study period 2015-2024; 1,343 grid cells; 161,160 records.
- Median Theil-Sen slope and 98.66% FDR-significant decline.
- Lag-1 autocorrelation median 0.558 and standard Mann-Kendall plus FDR wording.
- Fig. 1 V3, trend figures, 32 references, AOD is not PM2.5, and NTL remains supplementary sensitivity only.
""",
    )
    record(stage, "PASS", "Locked source tables and manuscript numbers written.", started)


def extract_docx_locations(path: Path, language: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            [
                {
                    "document_path": str(path),
                    "language": language,
                    "location_type": "file",
                    "location_index": "",
                    "style_or_table": "",
                    "section_hint": "",
                    "current_text": "",
                    "matched_risk": "SOURCE_FILE_MISSING",
                    "required_action": "Confirm source path before revision.",
                    "replacement_source": "Step 60 locked results",
                    "priority": "STOP",
                }
            ]
        )
    rows = []
    heading = ""
    patterns = [
        ("XGBoost primary model", ["xgboost", "no-ntl traceable"]),
        ("Old primary metrics", ["0.713", "0.711", "-0.021", "−0.021"]),
        ("Old Moran result", ["0.362", "p = 0.002", "p=0.002"]),
        ("Old SHAP narrative", ["shap", "t2m", "month_sin", "month_cos"]),
        ("Validation terminology", ["groupkfold", "random split", "loyo"]),
        ("Figure/table caption", ["fig. 4", "fig. 5", "figure 4", "figure 5", "图4", "图5"]),
    ]
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))

    def text_of(element) -> str:
        return "".join(t.text or "" for t in element.findall(".//w:t", ns)).strip()

    def style_of(paragraph) -> str:
        pstyle = paragraph.find("./w:pPr/w:pStyle", ns)
        if pstyle is None:
            return ""
        return pstyle.attrib.get(f"{{{ns['w']}}}val", "")

    body = root.find("w:body", ns)
    paragraph_index = 0
    table_index = 0
    for child in list(body):
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = text_of(child)
            style = style_of(child)
            if text:
                if style.lower().startswith("heading") or "标题" in style:
                    heading = text
                low = text.lower()
                matched = [
                    name
                    for name, toks in patterns
                    if any(t.lower() in low for t in toks)
                ]
                if matched:
                    rows.append(
                        {
                            "document_path": str(path),
                            "language": language,
                            "location_type": "paragraph",
                            "location_index": paragraph_index,
                            "style_or_table": style,
                            "section_hint": heading,
                            "current_text": text,
                            "matched_risk": "; ".join(matched),
                            "required_action": (
                                "Review and replace the primary-model wording/numbers with Step 60 "
                                "RF evidence while preserving all trend and study-design facts."
                            ),
                            "replacement_source": (
                                "07_locked_results/05_final_numbers_for_manuscript.md; "
                                "Step 60 RF SHAP and RF OOF Moran outputs"
                            ),
                            "priority": "MUST_REVIEW",
                        }
                    )
            paragraph_index += 1
        elif tag == "tbl":
            for ri, tr in enumerate(child.findall("./w:tr", ns)):
                for ci, tc in enumerate(tr.findall("./w:tc", ns)):
                    text = text_of(tc)
                    if not text:
                        continue
                    low = text.lower()
                    matched = [
                        name
                        for name, toks in patterns
                        if any(t.lower() in low for t in toks)
                    ]
                    if matched:
                        rows.append(
                            {
                                "document_path": str(path),
                                "language": language,
                                "location_type": "table_cell",
                                "location_index": (
                                    f"table={table_index},row={ri},col={ci}"
                                ),
                                "style_or_table": f"Table {table_index + 1}",
                                "section_hint": "",
                                "current_text": text,
                                "matched_risk": "; ".join(matched),
                                "required_action": (
                                    "Update only after confirming whether this is a primary result, "
                                    "model comparison, or supplementary sensitivity statement."
                                ),
                                "replacement_source": (
                                    "07_locked_results/02_four_model_metrics_locked.xlsx and "
                                    "Step 60 figure/table replacement plan"
                                ),
                                "priority": "MUST_REVIEW",
                            }
                        )
            table_index += 1
    planned_sections = [
        ("Abstract", "Replace XGBoost primary-model metrics and SHAP/Moran summary with RF locked evidence; preserve LOYO temporal-boundary message."),
        ("Methods", "Define RF as the historical spatial explanatory primary model and keep all four validation schemes."),
        ("Results", "Report RF Random, grid-cell grouped, region holdout, and LOYO metrics; add RF SHAP stability and RF OOF Moran results."),
        ("Discussion", "Separate historical spatial explanatory performance from temporal transferability; avoid forecasting and causal language."),
        ("Conclusion", "State that high historical/spatial performance does not imply unseen-year transferability."),
        ("Figure captions", "Replace primary-model XGBoost figure captions with RF and validation-boundary captions."),
        ("Table captions", "Present RF as primary and Ridge/HGB/XGB as controlled comparators."),
    ]
    for section, action in planned_sections:
        rows.append(
            {
                "document_path": str(path),
                "language": language,
                "location_type": "planned_section",
                "location_index": "",
                "style_or_table": "",
                "section_hint": section,
                "current_text": "",
                "matched_risk": "Required Step 60 narrative update",
                "required_action": action,
                "replacement_source": "Step 60 locked evidence package",
                "priority": "REQUIRED",
            }
        )
    return pd.DataFrame(rows)


def write_manuscript_update_plans() -> None:
    stage = "manuscript_update_plans"
    started = time.time()
    log("Locating exact manuscript and Online Resource update points")
    for path, language, stem in [
        (EN_DOC, "English", "01_English_exact_update_plan_source.csv"),
        (CN_DOC, "Chinese", "02_Chinese_exact_update_plan_source.csv"),
        (OR_DOC, "Online Resource", "03_Online_Resource_exact_update_plan_source.csv"),
    ]:
        extract_docx_locations(path, language).to_csv(
            DIRS["plans"] / stem, index=False, encoding="utf-8-sig"
        )
    write_md(
        DIRS["plans"] / "04_figure_and_table_replacement_plan.md",
        """# Figure and table replacement plan

## Main manuscript

1. Replace the current XGBoost primary-model validation figure with the Step 60 four-model R2 figure or an RF-centered derivative that retains all four validation schemes.
2. Replace the current XGBoost SHAP summary and dependence figures with `Fig_RF_SHAP_Global`, `Fig_RF_SHAP_Beeswarm`, and `Fig_RF_SHAP_Top5_Dependence`.
3. Replace the current residual figure statistics with the RF grid-cell grouped OOF residual Moran result; retain spatial-region OOF as a sensitivity result.
4. Present Ridge, HGB, and XGBoost as controlled comparators, not alternative primary narratives.
5. Keep LOYO negative values visible. Do not truncate the axis at zero.

## Online Resource

1. Update the model comparison table using the locked Step 59 four-model metrics.
2. Add RF SHAP fold-stability evidence.
3. Replace the prior XGBoost/random-split Moran value with RF OOF results and k sensitivity.
4. Preserve NTL as supplementary sensitivity only.

## Terminology

- `grid-cell grouped cross-validation`
- `continuous spatial-region holdout`
- `historical explanatory performance`
- `temporal transferability`

Grid-cell grouped cross-validation prevents monthly records from the same grid cell appearing in both training and test sets, but it is not equivalent to strict extrapolation to a continuous unseen spatial region.
""",
    )
    write_md(
        DIRS["plans"] / "05_old_numbers_to_remove.md",
        """# Old numbers and claims to retire from the primary narrative

- XGBoost primary Random split R2 = 0.713.
- XGBoost primary Spatial GroupKFold R2 = 0.711.
- XGBoost primary LOYO R2 = -0.021.
- Previous XGBoost SHAP Top-5 ranking as the paper's primary explanatory ranking.
- XGBoost/random-split residual Moran's I = 0.362, p = 0.002 as if it described RF.

These values may remain only where explicitly labelled as historical XGBoost comparison results. They must not be presented as the final RF primary-model evidence.
""",
    )
    write_md(
        DIRS["plans"] / "06_content_that_remains_unchanged.md",
        """# Content locked unchanged

- Study period: 2015-2024.
- Grid cells: 1,343.
- Records: 161,160.
- AOD dataset and all covariates.
- Median Theil-Sen slope and 98.66% FDR-significant decline.
- Lag-1 autocorrelation median = 0.558.
- Standard Mann-Kendall plus FDR; no modified Mann-Kendall and no pre-whitening.
- Fig. 1 V3 and trend figures.
- Reference list: 32 entries.
- AOD is not equivalent to PM2.5.
- NTL remains supplementary sensitivity only.
- No fifth model, new pollutant, emission inventory, HYSPLIT, city-level analysis, new trend method, or new map.
""",
    )
    record(stage, "PASS", "Three source documents audited; no documents modified.", started)


def write_qa(
    model: RandomForestRegressor,
    importance: pd.DataFrame,
    stability_status: str,
    integrity: pd.DataFrame,
    grid_moran: pd.DataFrame,
    region_moran: pd.DataFrame,
) -> None:
    stage = "QA_and_hashes"
    started = time.time()
    log("Writing QA reports and SHA256 manifest")
    add = pd.read_csv(DIRS["shap"] / "06_RF_SHAP_additivity_check_source.csv")
    write_md(
        DIRS["qa"] / "SHAP_QA.md",
        f"""# SHAP QA

- Explainer: `shap.TreeExplainer`.
- TreeExplainer approximate path decomposition: {SHAP_APPROXIMATE}.
- Model: locked Step 58 Random Forest.
- Full SHAP sample: 20,000 rows.
- Selection seed: 2026.
- Target values were not used for sample selection.
- Additivity status: {add.loc[0, 'status']}.
- Maximum absolute additivity error: {add.loc[0, 'max_abs_additivity_error']:.3e}.
- RF SHAP Top 5: {", ".join(importance.head(5)['feature'])}.
- Stability classification: {stability_status}.
- Interpretation boundary: model contribution, not causality.
""",
    )
    write_md(
        DIRS["qa"] / "OOF_residual_QA.md",
        "# OOF residual QA\n\n"
        + "\n".join(
            f"- {r.validation_scheme}: {r.status}; n={r.n_predictions:,}; "
            f"unique sample IDs={r.unique_sample_ids:,}; duplicates={r.duplicate_sample_ids}; "
            f"fold registry match={r.fold_matches_registry}; training residuals={r.training_residuals_present}."
            for r in integrity.itertuples()
        )
        + "\n\nAll formal residual diagnostics use held-out predictions.\n",
    )
    g8 = grid_moran[grid_moran["neighbor_k"] == 8].iloc[0]
    r8 = region_moran[region_moran["neighbor_k"] == 8].iloc[0]
    write_md(
        DIRS["qa"] / "Moran_QA.md",
        f"""# Moran QA

- Grid-cell grouped OOF: k=8, I={g8['morans_I']:.6f}, p={g8['two_sided_permutation_p']:.4f}.
- Spatial-region OOF: k=8, I={r8['morans_I']:.6f}, p={r8['two_sided_permutation_p']:.4f}.
- Spatial units: 1,343 grid-cell mean residuals.
- Weights: row-standardized k-nearest neighbors.
- Sensitivity k: 4, 8, 12.
- Permutations: 999.
- Seed: 2026.
- The prior XGBoost/random-split value 0.362 is not reused.
- Moran's I is interpreted as residual spatial structure, not a source or causal mechanism.
""",
    )
    expected_figures = []
    for folder in [DIRS["shap"], DIRS["stability"], DIRS["moran"], DIRS["figures"]]:
        expected_figures.extend(folder.glob("*.pdf"))
    figure_rows = []
    from PIL import Image

    for pdf in sorted(expected_figures):
        try:
            raw_pdf = pdf.read_bytes()
            pages = len(re.findall(rb"/Type\s*/Page\b", raw_pdf))
            pdf_header_ok = raw_pdf[:4] == b"%PDF"
            pdf_eof_ok = b"%%EOF" in raw_pdf[-2048:]
            status = (
                "PASS"
                if pages >= 1
                and pdf.stat().st_size > 0
                and pdf_header_ok
                and pdf_eof_ok
                else "FAIL"
            )
            figure_rows.append(
                {
                    "file": str(pdf),
                    "size_bytes": pdf.stat().st_size,
                    "pages": pages,
                    "pdf_header_ok": pdf_header_ok,
                    "pdf_eof_ok": pdf_eof_ok,
                    "status": status,
                }
            )
        except Exception as exc:
            figure_rows.append(
                {
                    "file": str(pdf),
                    "size_bytes": pdf.stat().st_size if pdf.exists() else 0,
                    "pages": 0,
                    "status": f"FAIL: {exc}",
                }
            )
    for png in sorted(
        list(DIRS["shap"].glob("*.png"))
        + list(DIRS["stability"].glob("*.png"))
        + list(DIRS["moran"].glob("*.png"))
        + list(DIRS["figures"].glob("*.png"))
    ):
        try:
            with Image.open(png) as im:
                figure_rows.append(
                    {
                        "file": str(png),
                        "size_bytes": png.stat().st_size,
                        "pages": "",
                        "pixel_width": im.width,
                        "pixel_height": im.height,
                        "status": "PASS" if im.width > 500 and im.height > 400 else "FAIL",
                    }
                )
        except Exception as exc:
            figure_rows.append({"file": str(png), "status": f"FAIL: {exc}"})
    figure_df = pd.DataFrame(figure_rows)
    figure_df.to_csv(
        DIRS["qa"] / "figure_file_QA.csv", index=False, encoding="utf-8-sig"
    )
    write_md(
        DIRS["qa"] / "figure_QA.md",
        "# Figure QA\n\n"
        f"- Files checked: {len(figure_df)}\n"
        f"- PASS: {(figure_df['status'] == 'PASS').sum()}\n"
        f"- Non-PASS: {(figure_df['status'] != 'PASS').sum()}\n"
        "- Python/matplotlib was used exclusively for drawing, export, and raster preview generation.\n"
        "- Visual previews require final manual inspection; file-level PDF and pixel checks passed where marked.\n",
    )
    env = "\n".join(
        [
            f"created_at={now()}",
            f"python_executable={sys.executable}",
            f"python_version={sys.version}",
            f"platform={platform.platform()}",
            f"numpy={np.__version__}",
            f"pandas={pd.__version__}",
            f"scipy={scipy.__version__}",
            f"scikit_learn={sklearn.__version__}",
            f"shap={shap.__version__}",
            f"joblib={joblib.__version__}",
            f"matplotlib={mpl.__version__}",
            f"features={FEATURES}",
            f"RF_parameters={RF_PARAMS}",
            f"SHAP_sample_seed={SHAP_SAMPLE_SEED}",
            f"SHAP_approximate={SHAP_APPROXIMATE}",
            f"Moran_seed={MORAN_SEED}",
            f"Moran_permutations={MORAN_PERMUTATIONS}",
            "OneDrive_used=False",
        ]
    )
    (DIRS["qa"] / "environment_lock.txt").write_text(env + "\n", encoding="utf-8")

    manuscript_hashes = {
        str(path): sha256_file(path) if path.exists() else "MISSING"
        for path in [EN_DOC, CN_DOC, OR_DOC]
    }
    write_json(DIRS["qa"] / "source_manuscript_hashes_no_edit.json", manuscript_hashes)
    final_summary = f"""# Step 60 final QA summary

- Input lock: PASS.
- Frozen data and split registries changed: No.
- RF explanatory model created: Yes.
- Full-data training performance used as validation: No.
- SHAP additivity: {add.loc[0, 'status']}.
- SHAP stability: {stability_status}.
- OOF predictions complete: Yes.
- Training residuals used for formal Moran analysis: No.
- RF Moran reproducibility settings recorded: Yes.
- Figure file checks: {'PASS' if (figure_df['status'] == 'PASS').all() else 'REVIEW_REQUIRED'}.
- Manuscripts modified: No.
- Online Resource modified: No.
- OneDrive used: No.

The evidence package is ready for a separate manuscript revision step. This run stops before editing any manuscript.
"""
    write_md(DIRS["qa"] / "final_QA_summary.md", final_summary)

    # Hash everything except the hash file itself, then write a deterministic manifest.
    hash_path = DIRS["qa"] / "SHA256SUMS.txt"
    lines = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p != hash_path:
            lines.append(f"{sha256_file(p)}  {p.relative_to(OUT)}")
    hash_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    record(stage, "PASS", "QA reports and SHA256SUMS written.", started)


def main() -> None:
    total_started = time.time()
    try:
        df, metrics = input_lock()
        model = train_explanatory_model(df)
        sample, importance, shap_values, _ = shap_sample_and_compute(df, model)
        draw_shap_figures(sample, importance, shap_values)
        _, _, stability_status = shap_stability(df, importance)
        grid_oof, region_oof, integrity, _ = build_oof_residuals(df)
        grid_moran, region_moran, sensitivity = audit_and_run_moran(
            grid_oof, region_oof
        )
        draw_four_model_figures(metrics)
        lock_results(
            metrics,
            importance,
            stability_status,
            grid_moran,
            region_moran,
            sensitivity,
        )
        write_manuscript_update_plans()
        write_qa(
            model,
            importance,
            stability_status,
            integrity,
            grid_moran,
            region_moran,
        )
        log(f"Step 60 core completed in {time.time() - total_started:.1f}s")
    except Exception as exc:
        error_text = (
            f"FAILED at {now()}\n{type(exc).__name__}: {exc}\n\n"
            + traceback.format_exc()
        )
        (DIRS["qa"] / "RUN_FAILURE.txt").write_text(error_text, encoding="utf-8")
        log(error_text)
        raise


def qa_only() -> None:
    importance = pd.read_csv(
        DIRS["shap"] / "04_RF_SHAP_global_importance_source.csv"
    )
    stability_text = (
        DIRS["stability"] / "03_SHAP_stability_report.md"
    ).read_text(encoding="utf-8")
    match = re.search(r"\*\*Classification: (STABLE|MODERATELY_STABLE|UNSTABLE)\*\*", stability_text)
    if not match:
        raise RuntimeError("Cannot recover SHAP stability classification.")
    integrity = pd.read_csv(
        DIRS["oof"] / "03_OOF_residual_integrity_check_source.csv"
    )
    grid_moran = pd.read_csv(
        DIRS["moran"] / "02_RF_grid_grouped_Moran_source.csv"
    )
    region_moran = pd.read_csv(
        DIRS["moran"] / "03_RF_spatial_region_Moran_source.csv"
    )
    write_qa(
        None,
        importance,
        match.group(1),
        integrity,
        grid_moran,
        region_moran,
    )


if __name__ == "__main__":
    if os.environ.get("STEP60_QA_ONLY") == "1":
        qa_only()
    else:
        main()
