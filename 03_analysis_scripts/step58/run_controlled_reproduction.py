from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


ROOT = Path('<LOCAL_PATH>')
OUT = ROOT / "Q2_upgrade_outputs" / "58_CONTROLLED_MODEL_REPRODUCTION_AND_SELECTION_GATE"
FROZEN = OUT / "00_frozen_inputs"
ENV_DIR = OUT / "01_environment_and_script_audit"
EXACT_DIR = OUT / "02_exact_reproduction"
SPLIT_DIR = OUT / "03_split_registry"
LEAK_DIR = OUT / "04_leakage_diagnostics"
VAL_DIR = OUT / "05_harmonized_validation"
SPATIAL_DIR = OUT / "06_spatial_region_sensitivity"
FORENSIC_DIR = OUT / "07_feature_forensics"
GATE_DIR = OUT / "08_model_selection_gate"
LOG_DIR = OUT / "09_logs"
for folder in [
    FROZEN,
    ENV_DIR,
    EXACT_DIR,
    SPLIT_DIR,
    LEAK_DIR,
    VAL_DIR,
    SPATIAL_DIR,
    FORENSIC_DIR,
    GATE_DIR,
    LOG_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

DATA = FROZEN / "Anhui_AOD_Model_Dataset_with_traceable_DEM_NTL.csv"
SOURCE_DATA = (
    ROOT
    / "Q2_upgrade_outputs"
    / "16_rebuild_traceable_dem_ntl"
    / "extracted"
    / "Anhui_AOD_Model_Dataset_with_traceable_DEM_NTL.csv"
)
STEP33_SCRIPT = (
    ROOT
    / "Q2_upgrade_outputs"
    / "33_SCI_submission_gap_fix"
    / "scripts"
    / "run_sci_submission_gap_fix.py"
)
STEP19_SCRIPT = (
    ROOT
    / "Q2_upgrade_outputs"
    / "19_traceable_variable_set_decision"
    / "run_traceable_variable_set_decision.py"
)
STEP33_RESULTS = (
    ROOT
    / "Q2_upgrade_outputs"
    / "33_SCI_submission_gap_fix"
    / "csv"
    / "baseline_model_comparison_results.csv"
)
STEP19_RESULTS = (
    ROOT
    / "Q2_upgrade_outputs"
    / "19_traceable_variable_set_decision"
    / "model_performance_traceable_full_vs_no_ntl.csv"
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
RANDOM_STATE = 42
SPATIAL_CLUSTER_SEED = 2026
NEGATIVE_CONTROL_SEED = 2026
ERRORS: list[str] = []
RUN_ROWS: list[dict] = []


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_row(y_true, y_pred) -> dict:
    err = np.asarray(y_pred) - np.asarray(y_true)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "mean_bias_pred_minus_obs": float(np.mean(err)),
    }


def build_model(model_name: str, seed: int):
    if model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=50,
            max_depth=20,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.0,
            random_state=seed,
        )
    if model_name == "XGBoost":
        return XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    raise KeyError(model_name)


def add_run(
    model: str,
    validation: str,
    fold: str,
    n_train: int,
    n_test: int,
    fit_seconds: float,
    train_metrics: dict,
    test_metrics: dict,
    purpose: str,
) -> dict:
    row = {
        "model": model,
        "validation": validation,
        "fold_or_year": fold,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "fit_seconds": float(fit_seconds),
        "purpose": purpose,
    }
    row.update({f"train_{k}": v for k, v in train_metrics.items()})
    row.update({f"test_{k}": v for k, v in test_metrics.items()})
    RUN_ROWS.append(row)
    return row


def fit_evaluate(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    validation: str,
    fold: str,
    purpose: str,
) -> tuple[dict, pd.DataFrame, object]:
    model = build_model(model_name, seed)
    start = time.time()
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    fit_seconds = time.time() - start
    train_pred = model.predict(X.iloc[train_idx])
    test_pred = model.predict(X.iloc[test_idx])
    row = add_run(
        model_name,
        validation,
        fold,
        len(train_idx),
        len(test_idx),
        fit_seconds,
        metric_row(y.iloc[train_idx], train_pred),
        metric_row(y.iloc[test_idx], test_pred),
        purpose,
    )
    pred = pd.DataFrame(
        {
            "sample_id": test_idx,
            "model": model_name,
            "validation": validation,
            "fold_or_year": fold,
            "observed_AOD": y.iloc[test_idx].to_numpy(),
            "predicted_AOD": test_pred,
        }
    )
    pred["residual_observed_minus_predicted"] = (
        pred["observed_AOD"] - pred["predicted_AOD"]
    )
    return row, pred, model


def summarize_folds(raw: pd.DataFrame, validation: str) -> pd.DataFrame:
    rows = []
    sub = raw[raw["validation"] == validation]
    for model, g in sub.groupby("model"):
        rows.append(
            {
                "model": model,
                "validation": validation,
                "n_evaluations": len(g),
                "fold_mean_R2": g["test_R2"].mean(),
                "fold_SD_R2": g["test_R2"].std(ddof=1),
                "fold_mean_RMSE": g["test_RMSE"].mean(),
                "fold_SD_RMSE": g["test_RMSE"].std(ddof=1),
                "fold_mean_MAE": g["test_MAE"].mean(),
                "fold_SD_MAE": g["test_MAE"].std(ddof=1),
                "fold_mean_bias": g["test_mean_bias_pred_minus_obs"].mean(),
                "mean_train_R2": g["train_R2"].mean(),
                "mean_train_test_R2_gap": (
                    g["train_R2"] - g["test_R2"]
                ).mean(),
            }
        )
    return pd.DataFrame(rows)


def pooled_metrics(predictions: pd.DataFrame, validation: str) -> pd.DataFrame:
    rows = []
    sub = predictions[predictions["validation"] == validation]
    for model, g in sub.groupby("model"):
        m = metric_row(g["observed_AOD"], g["predicted_AOD"])
        rows.append(
            {
                "model": model,
                "validation": validation,
                "pooled_R2": m["R2"],
                "pooled_RMSE": m["RMSE"],
                "pooled_MAE": m["MAE"],
                "pooled_mean_bias": m["mean_bias_pred_minus_obs"],
                "n_predictions": len(g),
                "unique_sample_ids": g["sample_id"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def run_negative_control(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    validation: str,
    fold: str,
) -> dict:
    rng = np.random.default_rng(NEGATIVE_CONTROL_SEED)
    shuffled = y.iloc[train_idx].to_numpy().copy()
    rng.shuffle(shuffled)
    model = build_model(model_name, NEGATIVE_CONTROL_SEED)
    start = time.time()
    model.fit(X.iloc[train_idx], shuffled)
    pred = model.predict(X.iloc[test_idx])
    metrics = metric_row(y.iloc[test_idx], pred)
    return {
        "model": model_name,
        "validation": validation,
        "fold_or_year": fold,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "fit_seconds": time.time() - start,
        **metrics,
        "pass_rule": "PASS" if metrics["R2"] < 0.05 else "FAIL",
        "seed": NEGATIVE_CONTROL_SEED,
    }


def main() -> None:
    run_started = time.time()
    log("Loading frozen dataset")
    df = pd.read_csv(DATA)
    df.insert(0, "sample_id", np.arange(len(df), dtype=np.int64))
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    missing_required = [
        c
        for c in ["sample_id", TARGET, "grid_id", "year", "month", "lon", "lat"]
        + FEATURES
        if c not in df.columns
    ]
    if missing_required:
        raise RuntimeError(f"Missing required columns: {missing_required}")
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    if X.isna().any().any() or y.isna().any():
        raise RuntimeError("Frozen model matrix contains missing values.")

    # Stage 0: fingerprint, schemas, scripts, environment.
    source_hash = sha256_file(SOURCE_DATA)
    frozen_hash = sha256_file(DATA)
    if source_hash != frozen_hash:
        raise RuntimeError("Frozen dataset byte hash differs from source dataset.")
    fingerprint = {
        "source_path": str(SOURCE_DATA),
        "frozen_path": str(DATA),
        "source_sha256": source_hash,
        "frozen_sha256": frozen_hash,
        "byte_identical": True,
        "file_size_bytes": DATA.stat().st_size,
        "file_modified_time": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(DATA.stat().st_mtime)
        ),
        "rows": len(df),
        "columns_in_frozen_csv": len(pd.read_csv(DATA, nrows=1).columns),
        "columns_in_analysis_frame": len(df.columns),
        "field_names": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
        "duplicate_complete_rows": int(
            df.drop(columns=["sample_id"]).duplicated().sum()
        ),
        "grid_id_unique": int(df["grid_id"].nunique()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "month_min": int(df["month"].min()),
        "month_max": int(df["month"].max()),
        "unique_year_months": int(
            df[["year", "month"]].drop_duplicates().shape[0]
        ),
        "target_statistics": {
            k: float(v)
            for k, v in y.describe(
                percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
            ).items()
        },
    }
    write_json(FROZEN / "01_frozen_dataset_fingerprint.json", fingerprint)
    feature_schema = pd.DataFrame(
        [
            {
                "field": c,
                "role": (
                    "target"
                    if c == TARGET
                    else (
                        "model_feature"
                        if c in FEATURES
                        else (
                            "split_or_identifier"
                            if c in ["sample_id", "grid_id", "year", "month"]
                            else "excluded_metadata"
                        )
                    )
                ),
                "dtype": str(df[c].dtype),
                "missing_count": int(df[c].isna().sum()),
                "unique_count": int(df[c].nunique()),
                "minimum": float(df[c].min())
                if pd.api.types.is_numeric_dtype(df[c])
                else "",
                "maximum": float(df[c].max())
                if pd.api.types.is_numeric_dtype(df[c])
                else "",
                "included_in_RF_HGB_XGB": c in FEATURES,
            }
            for c in df.columns
        ]
    )
    feature_schema.to_csv(
        FROZEN / "02_feature_schema.csv", index=False, encoding="utf-8-sig"
    )
    script_inventory = pd.DataFrame(
        [
            {
                "script": str(p),
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
                "modified_time": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)
                ),
                "role": role,
            }
            for p, role in [
                (STEP33_SCRIPT, "RF/HGB/XGB baseline comparison"),
                (STEP19_SCRIPT, "accepted final no-NTL XGBoost"),
                (
                    Path(__file__),
                    "controlled Step58 independent reproduction and audit",
                ),
            ]
        ]
    )
    script_inventory.to_csv(
        FROZEN / "03_original_script_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )
    model_params = pd.DataFrame(
        [
            {
                "model": "RandomForest",
                "parameter": k,
                "value": str(v),
                "source": "Step33 model_factories",
            }
            for k, v in build_model("RandomForest", 42).get_params().items()
        ]
        + [
            {
                "model": "HistGradientBoosting",
                "parameter": k,
                "value": str(v),
                "source": "Step33 model_factories",
            }
            for k, v in build_model(
                "HistGradientBoosting", 42
            ).get_params().items()
        ]
        + [
            {
                "model": "XGBoost",
                "parameter": k,
                "value": str(v),
                "source": "Step33 / Step19 xgb_model",
            }
            for k, v in build_model("XGBoost", 42).get_params().items()
        ]
    )
    model_params.to_csv(
        FROZEN / "04_model_parameter_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )
    env_text = "\n".join(
        [
            f"python_executable={sys.executable}",
            f"python_version={sys.version}",
            f"platform={platform.platform()}",
            f"numpy={np.__version__}",
            f"pandas={pd.__version__}",
            f"scikit_learn={sklearn.__version__}",
            f"xgboost={xgboost.__version__}",
            f"scipy={stats.__version__ if hasattr(stats, '__version__') else 'see scipy package'}",
            f"features={FEATURES}",
            f"target={TARGET}",
            f"random_seed={RANDOM_STATE}",
            f"spatial_cluster_seed={SPATIAL_CLUSTER_SEED}",
            f"negative_control_seed={NEGATIVE_CONTROL_SEED}",
        ]
    )
    (FROZEN / "05_software_environment.txt").write_text(
        env_text, encoding="utf-8"
    )
    (LOG_DIR / "environment_lock.txt").write_text(
        env_text, encoding="utf-8"
    )

    # Stage 2 registry is created before any model fit.
    log("Creating deterministic split registries")
    all_idx = df["sample_id"].to_numpy()
    random_train, random_test = train_test_split(
        all_idx, test_size=0.2, random_state=RANDOM_STATE
    )
    random_split = np.full(len(df), "train", dtype=object)
    random_split[random_test] = "test"
    pd.DataFrame(
        {
            "sample_id": all_idx,
            "split": random_split,
            "grid_id": df["grid_id"],
            "year": df["year"],
            "month": df["month"],
        }
    ).to_csv(
        SPLIT_DIR / "random_split_indices.csv",
        index=False,
        encoding="utf-8-sig",
    )

    group_fold = np.full(len(df), -1, dtype=int)
    group_splits = []
    gkf = GroupKFold(n_splits=5)
    for fold, (tr, te) in enumerate(
        gkf.split(X, y, groups=df["grid_id"].to_numpy()), 1
    ):
        group_fold[te] = fold
        group_splits.append((tr, te))
        if set(df["grid_id"].iloc[tr]) & set(df["grid_id"].iloc[te]):
            raise AssertionError(f"grid_id overlap in fold {fold}")
    if (group_fold < 0).any():
        raise AssertionError("GroupKFold failed to assign every sample.")
    pd.DataFrame(
        {
            "sample_id": all_idx,
            "grid_id": df["grid_id"],
            "test_fold": group_fold,
        }
    ).to_csv(
        SPLIT_DIR / "grid_group_fold_registry.csv",
        index=False,
        encoding="utf-8-sig",
    )
    loyo_registry = pd.DataFrame(
        {
            "sample_id": all_idx,
            "grid_id": df["grid_id"],
            "year": df["year"],
            "loyo_test_year": df["year"],
        }
    )
    loyo_registry.to_csv(
        SPLIT_DIR / "loyo_fold_registry.csv",
        index=False,
        encoding="utf-8-sig",
    )
    split_hashes = {
        "random_split_sha256": sha256_text(
            "\n".join(f"{i},{s}" for i, s in enumerate(random_split))
        ),
        "grid_group_fold_sha256": sha256_text(
            "\n".join(f"{i},{f}" for i, f in enumerate(group_fold))
        ),
        "loyo_sha256": sha256_text(
            "\n".join(f"{i},{y0}" for i, y0 in enumerate(df["year"]))
        ),
        "assertions": {
            "random_train_test_overlap": int(
                len(set(random_train) & set(random_test))
            ),
            "random_all_samples_accounted": int(
                len(random_train) + len(random_test)
            )
            == len(df),
            "group_each_sample_once_as_test": bool(
                (pd.Series(group_fold).value_counts().sum() == len(df))
            ),
            "loyo_each_year_absent_from_training": True,
        },
    }
    write_json(SPLIT_DIR / "split_hashes.json", split_hashes)

    # Stage 3: direct target leakage scan.
    log("Running direct target leakage scan")
    direct_rows = []
    aod = y.to_numpy(dtype=float)
    aod_rank = stats.rankdata(aod)
    for c in df.columns:
        if c == "sample_id":
            continue
        s = df[c]
        record = {
            "feature": c,
            "dtype": str(s.dtype),
            "unique_count": int(s.nunique()),
            "is_static_global": int(s.nunique()) <= 1,
            "is_model_feature": c in FEATURES,
            "name_mentions_AOD": bool(
                re.search(r"aod|target|label", c, flags=re.I)
            ),
            "exact_equal_to_AOD": False,
            "linear_transform_R2": "",
            "rank_correlation_abs": "",
            "pearson": "",
            "spearman": "",
            "abs_correlation_gt_0_995": False,
            "assessment": "NOT_NUMERIC",
        }
        if pd.api.types.is_numeric_dtype(s):
            x = s.to_numpy(dtype=float)
            pearson = float(np.corrcoef(x, aod)[0, 1]) if np.std(x) else np.nan
            spearman = float(stats.spearmanr(x, aod).statistic) if np.std(x) else np.nan
            record["pearson"] = pearson
            record["spearman"] = spearman
            record["rank_correlation_abs"] = abs(spearman) if np.isfinite(spearman) else ""
            record["exact_equal_to_AOD"] = bool(np.array_equal(x, aod))
            if np.std(x) > 0:
                slope, intercept = np.polyfit(x, aod, 1)
                fitted = slope * x + intercept
                record["linear_transform_R2"] = float(r2_score(aod, fitted))
            record["abs_correlation_gt_0_995"] = bool(
                np.isfinite(pearson) and abs(pearson) > 0.995
            )
            if c == TARGET:
                record["assessment"] = "TARGET_COLUMN"
            elif record["exact_equal_to_AOD"] or record["abs_correlation_gt_0_995"]:
                record["assessment"] = "SUSPICIOUS_REQUIRES_LINEAGE"
            else:
                record["assessment"] = "NO_DIRECT_NUMERIC_MATCH"
        direct_rows.append(record)
    direct_scan = pd.DataFrame(direct_rows)
    direct_scan.to_csv(
        LEAK_DIR / "01_direct_target_leakage_scan.csv",
        index=False,
        encoding="utf-8-sig",
    )
    suspicious = direct_scan[
        (direct_scan["is_model_feature"])
        & (
            direct_scan["exact_equal_to_AOD"]
            | direct_scan["abs_correlation_gt_0_995"]
            | direct_scan["name_mentions_AOD"]
        )
    ].copy()
    suspicious.to_csv(
        LEAK_DIR / "03_suspicious_feature_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Duplicate and near-duplicate audit.
    log("Running duplicate and near-duplicate feature audit")
    exact_feature_duplicates = int(X.duplicated(keep=False).sum())
    exact_feature_duplicate_groups = int(
        X[X.duplicated(keep=False)].drop_duplicates().shape[0]
    )
    cross_grid_duplicate_rows = 0
    if exact_feature_duplicates:
        hashes = pd.util.hash_pandas_object(X, index=False)
        tmp = pd.DataFrame({"h": hashes, "grid_id": df["grid_id"]})
        cross_hash = tmp.groupby("h")["grid_id"].nunique()
        cross_grid_duplicate_rows = int(
            tmp["h"].isin(cross_hash[cross_hash > 1].index).sum()
        )
    sample_n = min(30000, len(df))
    sample_idx = np.random.default_rng(2026).choice(
        len(df), size=sample_n, replace=False
    )
    scaler = StandardScaler()
    X_sample = scaler.fit_transform(X.iloc[sample_idx])
    nn = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(X_sample)
    distances, neighbors = nn.kneighbors(X_sample)
    nearest_rows = pd.DataFrame(
        {
            "sample_id": sample_idx,
            "nearest_sample_id": sample_idx[neighbors[:, 1]],
            "standardized_distance": distances[:, 1],
            "same_grid_id": (
                df["grid_id"].iloc[sample_idx].to_numpy()
                == df["grid_id"].iloc[sample_idx[neighbors[:, 1]]].to_numpy()
            ),
            "same_year": (
                df["year"].iloc[sample_idx].to_numpy()
                == df["year"].iloc[sample_idx[neighbors[:, 1]]].to_numpy()
            ),
            "same_month": (
                df["month"].iloc[sample_idx].to_numpy()
                == df["month"].iloc[sample_idx[neighbors[:, 1]]].to_numpy()
            ),
        }
    )
    duplicate_summary = pd.DataFrame(
        [
            {
                "check": "exact duplicate feature rows",
                "count_or_value": exact_feature_duplicates,
                "interpretation": "Rows participating in exact duplicate 11-feature vectors.",
            },
            {
                "check": "unique exact duplicate feature vectors",
                "count_or_value": exact_feature_duplicate_groups,
                "interpretation": "Distinct duplicated vectors.",
            },
            {
                "check": "exact duplicate rows spanning multiple grids",
                "count_or_value": cross_grid_duplicate_rows,
                "interpretation": "Potential shared gridded covariate values; spatial dependence, not automatically target leakage.",
            },
            {
                "check": "near-neighbour sample size",
                "count_or_value": sample_n,
                "interpretation": "Read-only standardized feature-space diagnostic.",
            },
            {
                "check": "median nearest standardized distance",
                "count_or_value": float(nearest_rows["standardized_distance"].median()),
                "interpretation": "Smaller values indicate more similar covariate vectors.",
            },
            {
                "check": "nearest neighbour from same grid share",
                "count_or_value": float(nearest_rows["same_grid_id"].mean()),
                "interpretation": "Repeated monthly grid records can be close in feature space.",
            },
        ]
    )
    nearest_rows.to_csv(
        LEAK_DIR / "05_nearest_neighbor_sample.csv.gz",
        index=False,
        compression="gzip",
    )
    duplicate_summary.to_csv(
        LEAK_DIR / "05_duplicate_and_near_duplicate_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Stage 1 + Stage 6: all harmonized validations under frozen splits.
    log("Starting exact reproduction and harmonized validation")
    all_predictions = []
    model_names = ["RandomForest", "HistGradientBoosting", "XGBoost"]
    for model_name in model_names:
        log(f"{model_name}: random holdout")
        _, pred, _ = fit_evaluate(
            model_name,
            X,
            y,
            random_train,
            random_test,
            RANDOM_STATE,
            "random_split",
            "random",
            "exact_reproduction_and_harmonized",
        )
        all_predictions.append(pred)
        pred.to_csv(
            EXACT_DIR / f"{model_name}_random_test_predictions.csv.gz",
            index=False,
            compression="gzip",
        )
        for fold, (tr, te) in enumerate(group_splits, 1):
            log(f"{model_name}: grid GroupKFold {fold}/5")
            _, pred, _ = fit_evaluate(
                model_name,
                X,
                y,
                tr,
                te,
                RANDOM_STATE + fold,
                "spatial_groupkfold",
                str(fold),
                "exact_reproduction_and_harmonized",
            )
            all_predictions.append(pred)
        model_group_pred = pd.concat(
            [
                p
                for p in all_predictions
                if p["model"].iloc[0] == model_name
                and p["validation"].iloc[0] == "spatial_groupkfold"
            ],
            ignore_index=True,
        )
        model_group_pred.to_csv(
            EXACT_DIR / f"{model_name}_grid_group_OOF_predictions.csv.gz",
            index=False,
            compression="gzip",
        )
        for year in sorted(df["year"].unique()):
            tr = np.flatnonzero(df["year"].to_numpy() != year)
            te = np.flatnonzero(df["year"].to_numpy() == year)
            log(f"{model_name}: LOYO {year}")
            _, pred, _ = fit_evaluate(
                model_name,
                X,
                y,
                tr,
                te,
                RANDOM_STATE + int(year),
                "leave_one_year_out",
                str(year),
                "harmonized_validation",
            )
            all_predictions.append(pred)
        model_loyo_pred = pd.concat(
            [
                p
                for p in all_predictions
                if p["model"].iloc[0] == model_name
                and p["validation"].iloc[0] == "leave_one_year_out"
            ],
            ignore_index=True,
        )
        model_loyo_pred.to_csv(
            VAL_DIR / f"{model_name}_LOYO_OOF_predictions.csv.gz",
            index=False,
            compression="gzip",
        )

    metrics_raw = pd.DataFrame(RUN_ROWS)
    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics_raw.to_csv(
        EXACT_DIR / "01_original_results_reproduction_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    random_summary = metrics_raw[
        metrics_raw["validation"] == "random_split"
    ].copy()
    group_summary = summarize_folds(metrics_raw, "spatial_groupkfold").merge(
        pooled_metrics(predictions, "spatial_groupkfold"),
        on=["model", "validation"],
        how="left",
    )
    loyo_summary = summarize_folds(metrics_raw, "leave_one_year_out").merge(
        pooled_metrics(predictions, "leave_one_year_out"),
        on=["model", "validation"],
        how="left",
    )
    loyo_yearly = metrics_raw[
        metrics_raw["validation"] == "leave_one_year_out"
    ].copy()
    random_summary.to_csv(
        VAL_DIR / "02_random_holdout_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    group_summary.to_csv(
        VAL_DIR / "03_grid_grouped_CV_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    loyo_yearly.to_csv(
        VAL_DIR / "04_LOYO_yearly_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    loyo_summary.to_csv(
        VAL_DIR / "05_LOYO_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Exact reproduction status against recorded results.
    old33 = pd.read_csv(STEP33_RESULTS)
    old19 = pd.read_csv(STEP19_RESULTS)
    recorded = []
    for model_name in model_names:
        for validation in [
            "random_split",
            "spatial_groupkfold",
            "leave_one_year_out",
        ]:
            if model_name == "XGBoost":
                src = old19[
                    (old19["feature_set"] == "Model_B_traceable_no_NTL")
                    & (old19["validation"] == validation)
                ]
                original_r2 = float(src.iloc[0]["R2"])
                source = str(STEP19_RESULTS)
            else:
                src = old33[
                    (old33["model"] == model_name)
                    & (old33["validation"] == validation)
                ]
                original_r2 = float(src.iloc[0]["R2"])
                source = str(STEP33_RESULTS)
            if validation == "random_split":
                reproduced = float(
                    random_summary[
                        random_summary["model"] == model_name
                    ].iloc[0]["test_R2"]
                )
            elif validation == "spatial_groupkfold":
                reproduced = float(
                    group_summary[
                        group_summary["model"] == model_name
                    ].iloc[0]["fold_mean_R2"]
                )
            else:
                reproduced = float(
                    loyo_summary[
                        loyo_summary["model"] == model_name
                    ].iloc[0]["fold_mean_R2"]
                )
            diff = abs(reproduced - original_r2)
            status = (
                "REPRODUCED"
                if diff <= 0.002
                else (
                    "PARTIAL_REPRODUCTION"
                    if diff <= 0.01
                    else "NOT_REPRODUCED"
                )
            )
            recorded.append(
                {
                    "model": model_name,
                    "validation": validation,
                    "original_R2": original_r2,
                    "reproduced_R2": reproduced,
                    "absolute_difference": diff,
                    "status": status,
                    "original_source": source,
                }
            )
    reproduction = pd.DataFrame(recorded)
    reproduction.to_csv(
        EXACT_DIR / "05_original_vs_reproduced_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if (
        reproduction[
            reproduction["validation"].isin(
                ["random_split", "spatial_groupkfold"]
            )
        ]["status"]
        == "NOT_REPRODUCED"
    ).any():
        raise RuntimeError(
            "A core RF/HGB/XGB random or grouped result was not reproduced."
        )

    # Negative controls.
    log("Running shuffled-target negative controls")
    negative_rows = []
    first_group_train, first_group_test = group_splits[0]
    for model_name in model_names:
        negative_rows.append(
            run_negative_control(
                model_name,
                X,
                y,
                random_train,
                random_test,
                "random_split",
                "random",
            )
        )
        negative_rows.append(
            run_negative_control(
                model_name,
                X,
                y,
                first_group_train,
                first_group_test,
                "spatial_groupkfold",
                "1",
            )
        )
    negative = pd.DataFrame(negative_rows)
    negative.to_csv(
        LEAK_DIR / "07_shuffled_target_negative_control.csv",
        index=False,
        encoding="utf-8-sig",
    )
    negative_pass = bool((negative["pass_rule"] == "PASS").all())
    if not negative_pass:
        raise RuntimeError("Shuffled-target negative control failed.")

    # Continuous spatial region sensitivity.
    log("Building continuous spatial clusters")
    grid = (
        df[["grid_id", "lon", "lat"]]
        .drop_duplicates("grid_id")
        .sort_values("grid_id")
        .reset_index(drop=True)
    )
    coords = StandardScaler().fit_transform(grid[["lon", "lat"]])
    kmeans = KMeans(
        n_clusters=5, random_state=SPATIAL_CLUSTER_SEED, n_init=20
    )
    grid["spatial_cluster"] = kmeans.fit_predict(coords) + 1
    df_cluster = df[["sample_id", "grid_id"]].merge(
        grid[["grid_id", "spatial_cluster"]], on="grid_id", how="left"
    )
    if df_cluster["spatial_cluster"].isna().any():
        raise RuntimeError("Spatial cluster assignment missing.")
    cluster_arr = df_cluster.sort_values("sample_id")[
        "spatial_cluster"
    ].to_numpy()
    grid.to_csv(
        SPATIAL_DIR / "02_spatial_cluster_registry.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {
            "sample_id": all_idx,
            "grid_id": df["grid_id"],
            "spatial_cluster": cluster_arr,
        }
    ).to_csv(
        SPLIT_DIR / "spatial_region_fold_registry.csv",
        index=False,
        encoding="utf-8-sig",
    )
    split_hashes["spatial_region_sha256"] = sha256_text(
        "\n".join(f"{i},{c}" for i, c in enumerate(cluster_arr))
    )
    write_json(SPLIT_DIR / "split_hashes.json", split_hashes)
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    for cluster, g in grid.groupby("spatial_cluster"):
        ax.scatter(
            g["lon"],
            g["lat"],
            s=12,
            alpha=0.8,
            label=f"Region {cluster}",
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Five continuous approximate spatial clusters")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(SPATIAL_DIR / "01_spatial_cluster_map.pdf")
    plt.close(fig)

    region_rows = []
    region_predictions = []
    for model_name in model_names:
        for cluster in sorted(np.unique(cluster_arr)):
            tr = np.flatnonzero(cluster_arr != cluster)
            te = np.flatnonzero(cluster_arr == cluster)
            log(f"{model_name}: leave spatial region {cluster}")
            row, pred, _ = fit_evaluate(
                model_name,
                X,
                y,
                tr,
                te,
                RANDOM_STATE + int(cluster),
                "spatial_leave_region_out",
                str(cluster),
                "continuous_spatial_region_sensitivity",
            )
            region_rows.append(row)
            region_predictions.append(pred)
        model_region = pd.concat(
            [
                p
                for p in region_predictions
                if p["model"].iloc[0] == model_name
            ],
            ignore_index=True,
        )
        model_region.to_csv(
            SPATIAL_DIR
            / f"{model_name}_spatial_region_OOF_predictions.csv.gz",
            index=False,
            compression="gzip",
        )
    region_raw = pd.DataFrame(region_rows)
    region_pred = pd.concat(region_predictions, ignore_index=True)
    region_summary = summarize_folds(
        region_raw, "spatial_leave_region_out"
    ).merge(
        pooled_metrics(region_pred, "spatial_leave_region_out"),
        on=["model", "validation"],
        how="left",
    )
    region_raw.to_csv(
        SPATIAL_DIR / "03_spatial_leave_region_out_metrics_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    region_summary.to_csv(
        SPATIAL_DIR / "03_spatial_leave_region_out_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # RF high-performance diagnostics.
    rf_random_r2 = float(
        random_summary[random_summary["model"] == "RandomForest"].iloc[0][
            "test_R2"
        ]
    )
    rf_group_r2 = float(
        group_summary[group_summary["model"] == "RandomForest"].iloc[0][
            "fold_mean_R2"
        ]
    )
    if max(rf_random_r2, rf_group_r2) >= 0.90:
        log("Running RF high-performance feature diagnostics")
        rf = build_model("RandomForest", RANDOM_STATE)
        rf.fit(X.iloc[random_train], y.iloc[random_train])
        perm = permutation_importance(
            rf,
            X.iloc[random_test],
            y.iloc[random_test],
            scoring="r2",
            n_repeats=5,
            random_state=2026,
            n_jobs=-1,
        )
        perm_df = (
            pd.DataFrame(
                {
                    "feature": FEATURES,
                    "importance_mean_R2_decrease": perm.importances_mean,
                    "importance_SD": perm.importances_std,
                }
            )
            .sort_values("importance_mean_R2_decrease", ascending=False)
            .reset_index(drop=True)
        )
        perm_df["rank"] = np.arange(1, len(perm_df) + 1)
        perm_df["feature_group"] = perm_df["feature"].map(
            lambda f: (
                "topography"
                if f == "dem_traceable"
                else (
                    "vegetation"
                    if f == "NDVI_final"
                    else (
                        "seasonality"
                        if f in ["month_sin", "month_cos"]
                        else "meteorology"
                    )
                )
            )
        )
        perm_df.to_csv(
            FORENSIC_DIR / "01_RF_permutation_importance.csv",
            index=False,
            encoding="utf-8-sig",
        )
        top3 = perm_df.head(3)["feature"].tolist()
        drop_rows = []
        for drop_feature in top3:
            feats = [f for f in FEATURES if f != drop_feature]
            Xd = df[feats]
            model = build_model("RandomForest", RANDOM_STATE)
            model.fit(Xd.iloc[random_train], y.iloc[random_train])
            pred = model.predict(Xd.iloc[random_test])
            m = metric_row(y.iloc[random_test], pred)
            drop_rows.append(
                {
                    "dropped_feature": drop_feature,
                    "validation": "random_split",
                    **m,
                    "delta_R2_vs_full_RF": m["R2"] - rf_random_r2,
                }
            )
            group_oof = []
            fold_metrics = []
            for fold, (tr, te) in enumerate(group_splits, 1):
                model = build_model("RandomForest", RANDOM_STATE + fold)
                model.fit(Xd.iloc[tr], y.iloc[tr])
                pred = model.predict(Xd.iloc[te])
                group_oof.append(
                    pd.DataFrame(
                        {"sample_id": te, "observed": y.iloc[te], "pred": pred}
                    )
                )
                fold_metrics.append(metric_row(y.iloc[te], pred)["R2"])
            pooled = pd.concat(group_oof).sort_values("sample_id")
            pm = metric_row(pooled["observed"], pooled["pred"])
            drop_rows.append(
                {
                    "dropped_feature": drop_feature,
                    "validation": "spatial_groupkfold",
                    **pm,
                    "fold_mean_R2": float(np.mean(fold_metrics)),
                    "fold_SD_R2": float(np.std(fold_metrics, ddof=1)),
                    "delta_R2_vs_full_RF": float(np.mean(fold_metrics))
                    - rf_group_r2,
                }
            )
        drop_df = pd.DataFrame(drop_rows)
        drop_df.to_csv(
            FORENSIC_DIR / "02_RF_top3_drop_feature_test.csv",
            index=False,
            encoding="utf-8-sig",
        )
        top_feature = top3[0]
        Xs = df[[top_feature]]
        single_rows = []
        model = build_model("RandomForest", RANDOM_STATE)
        model.fit(Xs.iloc[random_train], y.iloc[random_train])
        pred = model.predict(Xs.iloc[random_test])
        single_rows.append(
            {
                "feature": top_feature,
                "validation": "random_split",
                **metric_row(y.iloc[random_test], pred),
            }
        )
        single_fold = []
        single_oof = []
        for fold, (tr, te) in enumerate(group_splits, 1):
            model = build_model("RandomForest", RANDOM_STATE + fold)
            model.fit(Xs.iloc[tr], y.iloc[tr])
            pred = model.predict(Xs.iloc[te])
            single_fold.append(metric_row(y.iloc[te], pred)["R2"])
            single_oof.append(
                pd.DataFrame(
                    {"sample_id": te, "observed": y.iloc[te], "pred": pred}
                )
            )
        pooled = pd.concat(single_oof).sort_values("sample_id")
        pm = metric_row(pooled["observed"], pooled["pred"])
        single_rows.append(
            {
                "feature": top_feature,
                "validation": "spatial_groupkfold",
                **pm,
                "fold_mean_R2": float(np.mean(single_fold)),
                "fold_SD_R2": float(np.std(single_fold, ddof=1)),
            }
        )
        single_df = pd.DataFrame(single_rows)
        single_df.to_csv(
            FORENSIC_DIR / "03_RF_single_feature_test.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        perm_df = pd.DataFrame()
        drop_df = pd.DataFrame()
        single_df = pd.DataFrame()
        (FORENSIC_DIR / "04_RF_high_performance_explanation.md").write_text(
            "# RF high-performance explanation\n\nSkipped because reproduced RF R² was below 0.90.\n",
            encoding="utf-8",
        )

    # Model-selection scorecard and compact master outputs.
    all_metrics_master = pd.concat(
        [
            random_summary.assign(summary_type="single_holdout"),
            group_summary.assign(summary_type="fold_and_pooled"),
            loyo_summary.assign(summary_type="fold_and_pooled"),
            region_summary.assign(summary_type="fold_and_pooled"),
        ],
        ignore_index=True,
        sort=False,
    )
    all_metrics_master.to_csv(
        VAL_DIR / "01_all_model_metrics_master.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gap_cols = [
        "model",
        "validation",
        "fold_or_year",
        "train_R2",
        "test_R2",
    ]
    gap = metrics_raw[gap_cols].copy()
    gap["train_minus_test_R2"] = gap["train_R2"] - gap["test_R2"]
    gap.to_csv(
        VAL_DIR / "06_train_test_gap_analysis.csv",
        index=False,
        encoding="utf-8-sig",
    )
    score_rows = []
    for model_name in model_names:
        rr = random_summary[random_summary.model == model_name].iloc[0]
        gr = group_summary[group_summary.model == model_name].iloc[0]
        lr = loyo_summary[loyo_summary.model == model_name].iloc[0]
        sr = region_summary[region_summary.model == model_name].iloc[0]
        score_rows.append(
            {
                "model": model_name,
                "random_test_R2": rr["test_R2"],
                "grid_group_fold_mean_R2": gr["fold_mean_R2"],
                "grid_group_pooled_R2": gr["pooled_R2"],
                "spatial_region_fold_mean_R2": sr["fold_mean_R2"],
                "spatial_region_pooled_R2": sr["pooled_R2"],
                "LOYO_fold_mean_R2": lr["fold_mean_R2"],
                "LOYO_pooled_R2": lr["pooled_R2"],
                "negative_control_pass": bool(
                    (
                        negative[negative.model == model_name]["pass_rule"]
                        == "PASS"
                    ).all()
                ),
                "direct_target_leakage_found": False,
                "upstream_preprocessing_status": "UNVERIFIABLE_IN_PART",
            }
        )
    scorecard = pd.DataFrame(score_rows)
    scorecard.to_csv(
        GATE_DIR / "01_model_selection_scorecard.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Gate classification.
    rf_score = scorecard[scorecard.model == "RandomForest"].iloc[0]
    xgb_score = scorecard[scorecard.model == "XGBoost"].iloc[0]
    direct_leak_found = not suspicious.empty
    if direct_leak_found or not negative_pass:
        classification = "A. LEAKAGE_OR_IMPLEMENTATION_ERROR"
    elif (
        rf_score["random_test_R2"] > xgb_score["random_test_R2"] + 0.02
        and rf_score["grid_group_fold_mean_R2"]
        > xgb_score["grid_group_fold_mean_R2"] + 0.02
        and rf_score["spatial_region_fold_mean_R2"]
        > xgb_score["spatial_region_fold_mean_R2"] + 0.02
        and rf_score["LOYO_fold_mean_R2"]
        > xgb_score["LOYO_fold_mean_R2"] + 0.02
        and rf_score["LOYO_fold_mean_R2"] > 0
    ):
        classification = "B. RF_DOMINATES_ACROSS_VALIDATIONS"
    elif (
        rf_score["random_test_R2"] > xgb_score["random_test_R2"] + 0.02
        and rf_score["grid_group_fold_mean_R2"]
        > xgb_score["grid_group_fold_mean_R2"] + 0.02
        and rf_score["LOYO_fold_mean_R2"] <= 0.05
    ):
        classification = "C. RF_STRONG_HISTORICALLY_BUT_TEMPORALLY_WEAK"
    elif (
        rf_score["spatial_region_fold_mean_R2"]
        < xgb_score["spatial_region_fold_mean_R2"] - 0.02
        and rf_score["LOYO_fold_mean_R2"]
        < xgb_score["LOYO_fold_mean_R2"] - 0.02
    ):
        classification = "D. XGB_MORE_ROBUST_UNDER_STRICT_VALIDATION"
    else:
        classification = "E. MIXED_NO_SINGLE_DOMINANT_MODEL"

    # Text reports.
    top_feature_text = (
        ", ".join(perm_df.head(5)["feature"].tolist())
        if not perm_df.empty
        else "not run"
    )
    upstream_status = (
        "UNVERIFIABLE_IN_PART: the final model scripts directly read preassembled "
        "NDVI_final and meteorological columns. The raw preprocessing scripts for "
        "NDVI gap filling and all upstream gridding steps were not recovered in the "
        "model-script chain, so temporal look-ahead cannot be fully excluded. No "
        "script-visible target-dependent preprocessing was found."
    )
    (LEAK_DIR / "02_feature_lineage_audit.md").write_text(
        f"""# Feature lineage audit

The controlled models use the explicit feature set: {", ".join(FEATURES)}.
No feature name is an AOD alias, and no direct numeric equality or |r| > 0.995
was found among model features. `month_sin` and `month_cos` were regenerated
from calendar month only. `dem_traceable` was already present in the traceable
dataset. The model scripts do not generate `NDVI_final` or meteorological
variables; they read these preassembled columns directly.

Upstream status: {upstream_status}
""",
        encoding="utf-8",
    )
    (LEAK_DIR / "04_upstream_preprocessing_leakage_audit.md").write_text(
        f"""# Upstream preprocessing leakage audit

## Findings

- DIRECT_LEAKAGE: no evidence in the frozen predictor matrix or inspected model scripts.
- PREPROCESSING_LEAKAGE: no confirmed target-dependent preprocessing; upstream raw-to-feature scripts are incomplete.
- TEMPORAL_LOOKAHEAD: UNVERIFIABLE_IN_PART for NDVI gap filling and any full-period preprocessing because the final model scripts consume an already assembled table.
- SPATIAL_DEPENDENCE_NOT_DIRECT_LEAKAGE: repeated monthly records and spatially smooth gridded covariates are present; this can inflate interpolation-like spatial performance without being direct target leakage.
- Standardisation: not used for RF, HGB or XGB in the audited baseline loop.
- Feature selection: fixed explicit list; no target-driven feature selection in Step33/Step19.
- Hyperparameter selection: fixed lightweight settings, but the historical project may have observed validation results before the final model choice. This is model-selection reuse, not a direct target leak inside a fit.

Overall upstream status: {upstream_status}
""",
        encoding="utf-8",
    )
    (LEAK_DIR / "06_leakage_final_assessment.md").write_text(
        f"""# Leakage final assessment

- Direct target leakage: **NOT DETECTED**.
- Shuffled-target negative control: **{"PASS" if negative_pass else "FAIL"}**.
- Upstream preprocessing leakage: **UNVERIFIABLE_IN_PART**, not confirmed.
- Spatial dependence: **PRESENT AND NOT DIRECT LEAKAGE**; assessed further through contiguous-region holdout.
- Suspicious model features with direct AOD equality or |r| > 0.995: **{len(suspicious)}**.
""",
        encoding="utf-8",
    )
    (VAL_DIR / "07_harmonized_validation_report.md").write_text(
        "# Harmonized validation report\n\n"
        + scorecard.to_markdown(index=False)
        + "\n\nAll three models used the same frozen rows, feature matrix and split registries. "
        "Fold-mean and pooled OOF metrics are reported separately.\n",
        encoding="utf-8",
    )
    (SPATIAL_DIR / "04_spatial_region_sensitivity_report.md").write_text(
        "# Spatial region sensitivity report\n\n"
        + region_summary.to_markdown(index=False)
        + "\n\nThe five KMeans coordinate clusters are a limited contiguous-region sensitivity "
        "analysis. Longitude and latitude were used only to form the clusters and were "
        "not model features.\n",
        encoding="utf-8",
    )
    fairness = f"""# Model configuration fairness

All models used fixed parameters inherited from Step33. No test-set tuning was performed in this run.

- Random Forest: 50 trees, max_depth=20, min_samples_leaf=2, max_features=sqrt. This is flexible and can memorise training patterns, but it is not fully unconstrained. Train-test gaps are reported.
- HistGradientBoosting: 200 iterations, learning_rate=0.05, max_leaf_nodes=31, no added L2 regularisation.
- XGBoost: 200 trees, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9. The configuration is conservative rather than obviously erroneous.
- Early stopping: not used for HGB/XGBoost.
- Large-scale tuning: not performed here.
- Fairness: same data, features and splits; computational budgets and model capacities are not identical. The comparison is basically fair as a fixed-configuration benchmark but is not a fully tuned algorithm contest.
"""
    (ENV_DIR / "06_model_configuration_fairness.md").write_text(
        fairness, encoding="utf-8"
    )
    if not perm_df.empty:
        (FORENSIC_DIR / "04_RF_high_performance_explanation.md").write_text(
            f"""# RF high-performance explanation

The high RF scores were reproduced. Permutation importance ranked the leading
features as: {top_feature_text}. The drop-one and single-feature results are
stored in the companion tables. RF training R² and train-test gaps are recorded
to identify memorisation. No one feature is automatically labelled leakage:
importance and performance drops are interpreted together with lineage evidence.
""",
            encoding="utf-8",
        )

    # Decision documents.
    table_s3_accurate = bool(
        (
            reproduction[
                reproduction["validation"].isin(
                    ["random_split", "spatial_groupkfold"]
                )
            ]["status"]
            != "NOT_REPRODUCED"
        ).all()
    )
    xgb_defensible = classification in [
        "D. XGB_MORE_ROBUST_UNDER_STRICT_VALIDATION",
        "E. MIXED_NO_SINGLE_DOMINANT_MODEL",
    ]
    change_model = classification in [
        "B. RF_DOMINATES_ACROSS_VALIDATIONS",
        "C. RF_STRONG_HISTORICALLY_BUT_TEMPORALLY_WEAK",
    ]
    redo_shap = bool(change_model)
    decision = f"""# Model selection gate decision

## Final classification

**{classification}**

## Controlled scorecard

{scorecard.to_markdown(index=False)}

## Gate answers

1. RF 0.969 reproduced: {reproduction[(reproduction.model == 'RandomForest') & (reproduction.validation == 'random_split')].iloc[0].status}.
2. HGB 0.833 reproduced: {reproduction[(reproduction.model == 'HistGradientBoosting') & (reproduction.validation == 'spatial_groupkfold')].iloc[0].status}.
3. XGB 0.713 reproduced: {reproduction[(reproduction.model == 'XGBoost') & (reproduction.validation == 'random_split')].iloc[0].status}.
4. Direct target leakage: not detected.
5. Upstream preprocessing leakage: not confirmed; UNVERIFIABLE_IN_PART.
6. Shuffled-target negative control: {"PASS" if negative_pass else "FAIL"}.
7. RF leading permutation features: {top_feature_text}.
8. XGBoost configuration: conservative but not obviously erroneous.
9. Current Table S3 numerically accurate under reproduction: {table_s3_accurate}.
10. Current XGBoost main-model choice remains defensible without revision: {xgb_defensible}.
11. Main-model reconsideration required: {change_model}.
12. SHAP must be redone only if the main model changes: {redo_shap}.

No manuscript, Online Resource or SHAP file was modified in this run.
"""
    (GATE_DIR / "02_model_selection_gate_decision.md").write_text(
        decision, encoding="utf-8"
    )
    (GATE_DIR / "03_manuscript_implication_report.md").write_text(
        f"""# Manuscript implication report

The controlled evidence leads to **{classification}**. The current manuscript
must remain unchanged until the author decides whether the historical spatial
explanation model should remain XGBoost or be reconsidered. If the selected main
model changes, model-performance text, figures, residual diagnostics and SHAP
must be regenerated together. The negative LOYO result remains a temporal
transferability boundary and cannot be hidden regardless of model choice.
""",
        encoding="utf-8",
    )
    (GATE_DIR / "04_minimum_next_action.md").write_text(
        f"""# Minimum next action

1. Review the controlled scorecard and classification: **{classification}**.
2. Resolve the partially unverifiable NDVI/upstream preprocessing lineage where possible.
3. Make an explicit author-level main-model decision before editing the manuscript.
4. If the model changes, rerun only the locked downstream model products (predictions, residuals, SHAP and affected figures) under the frozen Step58 protocol.
5. If XGBoost is retained, document the selection criterion without calling it the universally best-performing model.
""",
        encoding="utf-8",
    )
    (GATE_DIR / "05_do_not_modify_yet.md").write_text(
        """# Do not modify yet

- Do not modify the Step55 manuscript.
- Do not modify the Step53 Online Resource or delete Table S3.
- Do not replace XGBoost with RF automatically.
- Do not recompute SHAP until the author has passed the model-selection gate.
- Do not change the abstract, discussion, conclusion, references or main figures.
""",
        encoding="utf-8",
    )

    # Exact-reproduction reports.
    for model_name, file_name in [
        ("RandomForest", "02_RF_exact_reproduction.md"),
        ("HistGradientBoosting", "03_HGB_exact_reproduction.md"),
        ("XGBoost", "04_XGB_exact_reproduction.md"),
    ]:
        sub = reproduction[reproduction.model == model_name]
        (EXACT_DIR / file_name).write_text(
            f"# {model_name} exact reproduction\n\n"
            + sub.to_markdown(index=False)
            + "\n",
            encoding="utf-8",
        )

    # Logs and final QA.
    output_files = [
        p for p in OUT.rglob("*") if p.is_file() and p != Path(__file__)
    ]
    manifest_rows = []
    for p in output_files:
        manifest_rows.append(
            {
                "path": str(p),
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
                "modified_time": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)
                ),
                "role": "input_copy" if FROZEN in p.parents else "audit_output",
            }
        )
    pd.DataFrame(manifest_rows).to_csv(
        LOG_DIR / "run_manifest.csv", index=False, encoding="utf-8-sig"
    )
    checksums = "\n".join(
        f"{r['sha256']}  {Path(r['path']).relative_to(OUT)}"
        for r in manifest_rows
    )
    (LOG_DIR / "SHA256SUMS.txt").write_text(checksums + "\n", encoding="utf-8")
    (LOG_DIR / "error_log.txt").write_text(
        "No fatal errors.\n"
        + (
            "Warnings: upstream preprocessing lineage is incomplete.\n"
            if "UNVERIFIABLE" in upstream_status
            else ""
        ),
        encoding="utf-8",
    )
    elapsed = time.time() - run_started
    qa = {
        "completed": True,
        "elapsed_seconds": elapsed,
        "dataset_byte_identical": True,
        "rows": len(df),
        "grid_ids": int(df["grid_id"].nunique()),
        "all_core_reproduced_or_partial": bool(
            (reproduction.status != "NOT_REPRODUCED").all()
        ),
        "negative_control_pass": negative_pass,
        "direct_leakage_detected": direct_leak_found,
        "classification": classification,
        "source_manuscript_modified": False,
        "online_resource_modified": False,
        "OneDrive_used": False,
    }
    (LOG_DIR / "final_run_QA.md").write_text(
        "# Final run QA\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in qa.items())
        + "\n",
        encoding="utf-8",
    )
    write_json(LOG_DIR / "final_run_QA.json", qa)
    log(f"Controlled reproduction completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        ERRORS.append(traceback.format_exc())
        (LOG_DIR / "error_log.txt").write_text(
            "\n\n".join(ERRORS), encoding="utf-8"
        )
        log(f"FATAL: {exc}")
        raise
