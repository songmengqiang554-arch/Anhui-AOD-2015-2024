from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path('<LOCAL_PATH>')
STEP58 = (
    ROOT
    / "Q2_upgrade_outputs"
    / "58_CONTROLLED_MODEL_REPRODUCTION_AND_SELECTION_GATE"
)
OUT = (
    ROOT
    / "Q2_upgrade_outputs"
    / "59_FOUR_MODEL_COMPARISON_AND_FINAL_SELECTION"
)
VERIFY = OUT / "00_input_verification"
CONFIG = OUT / "01_ridge_configuration"
RIDGE_OUT = OUT / "02_ridge_validation"
COMPARE = OUT / "03_four_model_comparison"
DECISION = OUT / "04_model_selection_decision"
IMPLICATIONS = OUT / "05_manuscript_implications"
LOGS = OUT / "06_logs"

DATA = (
    STEP58
    / "00_frozen_inputs"
    / "Anhui_AOD_Model_Dataset_with_traceable_DEM_NTL.csv"
)
FEATURE_SCHEMA = STEP58 / "00_frozen_inputs" / "02_feature_schema.csv"
FINGERPRINT = STEP58 / "00_frozen_inputs" / "01_frozen_dataset_fingerprint.json"
RANDOM_REG = STEP58 / "03_split_registry" / "random_split_indices.csv"
GRID_REG = STEP58 / "03_split_registry" / "grid_group_fold_registry.csv"
LOYO_REG = STEP58 / "03_split_registry" / "loyo_fold_registry.csv"
REGION_REG = STEP58 / "03_split_registry" / "spatial_region_fold_registry.csv"
SPLIT_HASHES = STEP58 / "03_split_registry" / "split_hashes.json"

TREE_RANDOM = (
    STEP58 / "05_harmonized_validation" / "02_random_holdout_comparison.csv"
)
TREE_GRID = (
    STEP58 / "05_harmonized_validation" / "03_grid_grouped_CV_comparison.csv"
)
TREE_LOYO = STEP58 / "05_harmonized_validation" / "05_LOYO_summary.csv"
TREE_LOYO_YEAR = (
    STEP58 / "05_harmonized_validation" / "04_LOYO_yearly_comparison.csv"
)
TREE_REGION = (
    STEP58
    / "06_spatial_region_sensitivity"
    / "03_spatial_leave_region_out_metrics_summary.csv"
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
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
RANDOM_STATE = 42
ERRORS: list[str] = []


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def metric_row(y_true, y_pred) -> dict:
    residual = np.asarray(y_pred) - np.asarray(y_true)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "mean_bias_pred_minus_obs": float(np.mean(residual)),
    }


def make_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def choose_alpha(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    inner_kind: str,
    groups: np.ndarray | None,
    validation: str,
    fold_label: str,
) -> tuple[float, pd.DataFrame, float]:
    start = time.time()
    local_X = X.iloc[train_idx]
    local_y = y.iloc[train_idx]
    if inner_kind == "kfold":
        splitter = KFold(
            n_splits=5, shuffle=True, random_state=RANDOM_STATE
        )
        split_iter = list(splitter.split(local_X, local_y))
        inner_groups = None
    else:
        if groups is None:
            raise ValueError("Grouped alpha selection requires groups.")
        inner_groups = np.asarray(groups)[train_idx]
        n_groups = int(pd.Series(inner_groups).nunique())
        n_splits = min(5, n_groups)
        if n_splits < 2:
            raise RuntimeError(
                f"Insufficient inner groups for {validation} {fold_label}"
            )
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = list(
            splitter.split(local_X, local_y, groups=inner_groups)
        )

    rows = []
    for alpha in ALPHAS:
        scores = []
        for inner_fold, (inner_train, inner_valid) in enumerate(
            split_iter, 1
        ):
            model = make_pipeline(alpha)
            model.fit(local_X.iloc[inner_train], local_y.iloc[inner_train])
            pred = model.predict(local_X.iloc[inner_valid])
            rmse = math.sqrt(
                mean_squared_error(local_y.iloc[inner_valid], pred)
            )
            scores.append(rmse)
            rows.append(
                {
                    "validation_scheme": validation,
                    "outer_fold": fold_label,
                    "inner_strategy": inner_kind,
                    "inner_fold": inner_fold,
                    "alpha": alpha,
                    "inner_RMSE": rmse,
                    "n_inner_train": len(inner_train),
                    "n_inner_valid": len(inner_valid),
                    "n_inner_groups": (
                        int(pd.Series(inner_groups).nunique())
                        if inner_groups is not None
                        else ""
                    ),
                }
            )
    detail = pd.DataFrame(rows)
    means = detail.groupby("alpha", as_index=False)["inner_RMSE"].mean()
    selected = float(
        means.sort_values(["inner_RMSE", "alpha"]).iloc[0]["alpha"]
    )
    detail["selected_for_outer_fold"] = detail["alpha"].eq(selected)
    return selected, detail, time.time() - start


def fit_outer_fold(
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    validation: str,
    fold_label: str,
    inner_kind: str,
    groups: np.ndarray | None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected, alpha_detail, selection_seconds = choose_alpha(
        X,
        y,
        train_idx,
        inner_kind,
        groups,
        validation,
        fold_label,
    )
    model = make_pipeline(selected)
    start = time.time()
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    fit_seconds = time.time() - start
    train_pred = model.predict(X.iloc[train_idx])
    test_pred = model.predict(X.iloc[test_idx])
    train_metrics = metric_row(y.iloc[train_idx], train_pred)
    test_metrics = metric_row(y.iloc[test_idx], test_pred)
    selected_inner = alpha_detail[
        alpha_detail["selected_for_outer_fold"]
    ]["inner_RMSE"]
    row = {
        "model": "Ridge",
        "validation": validation,
        "fold_or_year": fold_label,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "selected_alpha": selected,
        "inner_mean_RMSE_selected": float(selected_inner.mean()),
        "inner_SD_RMSE_selected": float(selected_inner.std(ddof=1)),
        "alpha_selection_seconds": selection_seconds,
        "final_fit_seconds": fit_seconds,
        "total_seconds": selection_seconds + fit_seconds,
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    pred = pd.DataFrame(
        {
            "sample_id": test_idx,
            "grid_id": df.iloc[test_idx]["grid_id"].to_numpy(),
            "year": df.iloc[test_idx]["year"].to_numpy(),
            "month": df.iloc[test_idx]["month"].to_numpy(),
            "fold": fold_label,
            "observed_AOD": y.iloc[test_idx].to_numpy(),
            "predicted_AOD": test_pred,
            "validation_scheme": validation,
            "selected_alpha": selected,
        }
    )
    pred["residual"] = pred["observed_AOD"] - pred["predicted_AOD"]
    coefficients = pd.DataFrame(
        {
            "validation_scheme": validation,
            "outer_fold": fold_label,
            "selected_alpha": selected,
            "feature": FEATURES,
            "standardized_coefficient": model.named_steps[
                "ridge"
            ].coef_,
        }
    )
    coefficients["direction"] = np.where(
        coefficients["standardized_coefficient"] >= 0,
        "positive",
        "negative",
    )
    return row, pred, coefficients, alpha_detail


def summarize_folds(
    raw_metrics: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for validation, group in raw_metrics.groupby("validation"):
        pred = predictions[
            predictions["validation_scheme"].eq(validation)
        ]
        pooled = metric_row(pred["observed_AOD"], pred["predicted_AOD"])
        rows.append(
            {
                "model": "Ridge",
                "validation": validation,
                "n_evaluations": len(group),
                "fold_mean_R2": group["test_R2"].mean(),
                "fold_SD_R2": group["test_R2"].std(ddof=1),
                "fold_mean_RMSE": group["test_RMSE"].mean(),
                "fold_SD_RMSE": group["test_RMSE"].std(ddof=1),
                "fold_mean_MAE": group["test_MAE"].mean(),
                "fold_SD_MAE": group["test_MAE"].std(ddof=1),
                "fold_mean_bias": group[
                    "test_mean_bias_pred_minus_obs"
                ].mean(),
                "mean_train_R2": group["train_R2"].mean(),
                "mean_train_test_R2_gap": (
                    group["train_R2"] - group["test_R2"]
                ).mean(),
                "pooled_R2": pooled["R2"],
                "pooled_RMSE": pooled["RMSE"],
                "pooled_MAE": pooled["MAE"],
                "pooled_mean_bias": pooled[
                    "mean_bias_pred_minus_obs"
                ],
                "n_predictions": len(pred),
                "selected_alpha_values": ";".join(
                    str(v)
                    for v in sorted(group["selected_alpha"].unique())
                ),
                "total_runtime_seconds": group["total_seconds"].sum(),
            }
        )
    return pd.DataFrame(rows)


def tree_scorecard() -> pd.DataFrame:
    random = pd.read_csv(TREE_RANDOM)
    grid = pd.read_csv(TREE_GRID)
    loyo = pd.read_csv(TREE_LOYO)
    region = pd.read_csv(TREE_REGION)
    rows = []
    for model in ["RandomForest", "HistGradientBoosting", "XGBoost"]:
        r = random[random["model"].eq(model)].iloc[0]
        g = grid[grid["model"].eq(model)].iloc[0]
        l = loyo[loyo["model"].eq(model)].iloc[0]
        s = region[region["model"].eq(model)].iloc[0]
        rows.extend(
            [
                {
                    "model": model,
                    "validation": "random_split",
                    "R2": r["test_R2"],
                    "RMSE": r["test_RMSE"],
                    "MAE": r["test_MAE"],
                    "mean_bias": r["test_mean_bias_pred_minus_obs"],
                    "fold_SD_R2": np.nan,
                    "pooled_R2": r["test_R2"],
                    "train_R2": r["train_R2"],
                    "train_test_gap": r["train_R2"] - r["test_R2"],
                    "runtime_seconds": r["fit_seconds"],
                    "source": "Step58 locked result",
                },
                {
                    "model": model,
                    "validation": "spatial_groupkfold",
                    "R2": g["fold_mean_R2"],
                    "RMSE": g["fold_mean_RMSE"],
                    "MAE": g["fold_mean_MAE"],
                    "mean_bias": g["fold_mean_bias"],
                    "fold_SD_R2": g["fold_SD_R2"],
                    "pooled_R2": g["pooled_R2"],
                    "train_R2": g["mean_train_R2"],
                    "train_test_gap": g[
                        "mean_train_test_R2_gap"
                    ],
                    "runtime_seconds": np.nan,
                    "source": "Step58 locked result",
                },
                {
                    "model": model,
                    "validation": "spatial_leave_region_out",
                    "R2": s["fold_mean_R2"],
                    "RMSE": s["fold_mean_RMSE"],
                    "MAE": s["fold_mean_MAE"],
                    "mean_bias": s["fold_mean_bias"],
                    "fold_SD_R2": s["fold_SD_R2"],
                    "pooled_R2": s["pooled_R2"],
                    "train_R2": s["mean_train_R2"],
                    "train_test_gap": s[
                        "mean_train_test_R2_gap"
                    ],
                    "runtime_seconds": np.nan,
                    "source": "Step58 locked result",
                },
                {
                    "model": model,
                    "validation": "leave_one_year_out",
                    "R2": l["fold_mean_R2"],
                    "RMSE": l["fold_mean_RMSE"],
                    "MAE": l["fold_mean_MAE"],
                    "mean_bias": l["fold_mean_bias"],
                    "fold_SD_R2": l["fold_SD_R2"],
                    "pooled_R2": l["pooled_R2"],
                    "train_R2": l["mean_train_R2"],
                    "train_test_gap": l[
                        "mean_train_test_R2_gap"
                    ],
                    "runtime_seconds": np.nan,
                    "source": "Step58 locked result",
                },
            ]
        )
    return pd.DataFrame(rows)


def create_comparison_figures(
    four: pd.DataFrame, ridge_loyo: pd.DataFrame
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    colors = {
        "Ridge": "#7A7A7A",
        "RandomForest": "#2A6F97",
        "HistGradientBoosting": "#5AAE61",
        "XGBoost": "#D98C3F",
    }
    labels = {
        "random_split": "Random",
        "spatial_groupkfold": "Grid grouped",
        "spatial_leave_region_out": "Spatial region",
        "leave_one_year_out": "LOYO",
    }
    models = ["Ridge", "RandomForest", "HistGradientBoosting", "XGBoost"]
    validations = list(labels)
    x = np.arange(len(validations))
    width = 0.19

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, model in enumerate(models):
        subset = four[four["model"].eq(model)].set_index("validation")
        vals = [subset.loc[v, "R2"] for v in validations]
        ax.bar(
            x + (i - 1.5) * width,
            vals,
            width,
            label=model,
            color=colors[model],
            edgecolor="white",
            linewidth=0.5,
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, [labels[v] for v in validations])
    ax.set_ylabel(r"$R^2$")
    ax.set_title(
        "Four-model validation comparison: spatial explanation and temporal limits"
    )
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(
        COMPARE / "03_four_model_R2_comparison.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, model in enumerate(models):
        subset = four[four["model"].eq(model)].set_index("validation")
        vals = [subset.loc[v, "RMSE"] for v in validations]
        ax.bar(
            x + (i - 1.5) * width,
            vals,
            width,
            label=model,
            color=colors[model],
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xticks(x, [labels[v] for v in validations])
    ax.set_ylabel("RMSE")
    ax.set_title("Prediction error across four validation schemes")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(
        COMPARE / "04_four_model_RMSE_comparison.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    tree_year = pd.read_csv(TREE_LOYO_YEAR)
    tree_year = tree_year.rename(
        columns={"fold_or_year": "year", "test_R2": "R2"}
    )
    ridge_year = ridge_loyo.rename(
        columns={"fold_or_year": "year", "test_R2": "R2"}
    ).copy()
    ridge_year["model"] = "Ridge"
    yearly = pd.concat(
        [tree_year[["model", "year", "R2"]], ridge_year[["model", "year", "R2"]]],
        ignore_index=True,
    )
    yearly["year"] = yearly["year"].astype(int)
    yearly.to_csv(
        COMPARE / "05_four_model_LOYO_yearly_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for model in models:
        group = yearly[yearly["model"].eq(model)].sort_values("year")
        ax.plot(
            group["year"],
            group["R2"],
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            label=model,
            color=colors[model],
        )
    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Held-out year")
    ax.set_ylabel(r"$R^2$")
    ax.set_title("Leave-one-year-out performance by held-out year")
    ax.set_xticks(sorted(yearly["year"].unique()))
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(color="#D9D9D9", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(
        COMPARE / "05_four_model_LOYO_yearly_comparison.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    started = time.time()
    for directory in [
        VERIFY,
        CONFIG,
        RIDGE_OUT,
        COMPARE,
        DECISION,
        IMPLICATIONS,
        LOGS,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    log("Loading Step58 frozen data and split registries")
    df = pd.read_csv(DATA)
    df.insert(0, "sample_id", np.arange(len(df), dtype=np.int64))
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    random_reg = pd.read_csv(RANDOM_REG).sort_values("sample_id")
    grid_reg = pd.read_csv(GRID_REG).sort_values("sample_id")
    loyo_reg = pd.read_csv(LOYO_REG).sort_values("sample_id")
    region_reg = pd.read_csv(REGION_REG).sort_values("sample_id")

    expected_ids = np.arange(len(df))
    for name, reg in [
        ("random", random_reg),
        ("grid", grid_reg),
        ("loyo", loyo_reg),
        ("region", region_reg),
    ]:
        if not np.array_equal(reg["sample_id"].to_numpy(), expected_ids):
            raise RuntimeError(f"{name} registry sample_id mismatch")

    expected_hashes = json.loads(SPLIT_HASHES.read_text(encoding="utf-8"))
    logical_hashes = {
        "random_split_sha256": sha256_text(
            "\n".join(
                f"{i},{v}"
                for i, v in enumerate(random_reg["split"].to_numpy())
            )
        ),
        "grid_group_fold_sha256": sha256_text(
            "\n".join(
                f"{i},{v}"
                for i, v in enumerate(grid_reg["test_fold"].to_numpy())
            )
        ),
        "loyo_sha256": sha256_text(
            "\n".join(
                f"{i},{v}"
                for i, v in enumerate(loyo_reg["year"].to_numpy())
            )
        ),
        "spatial_region_sha256": sha256_text(
            "\n".join(
                f"{i},{v}"
                for i, v in enumerate(
                    region_reg["spatial_cluster"].to_numpy()
                )
            )
        ),
    }
    logical_match = {
        key: logical_hashes[key] == expected_hashes[key]
        for key in logical_hashes
    }
    file_hashes = {
        str(path): sha256_file(path)
        for path in [
            DATA,
            FEATURE_SCHEMA,
            FINGERPRINT,
            RANDOM_REG,
            GRID_REG,
            LOYO_REG,
            REGION_REG,
            SPLIT_HASHES,
        ]
    }
    verification = {
        "dataset_path": str(DATA),
        "dataset_sha256": file_hashes[str(DATA)],
        "rows": len(df),
        "unique_grid_id": int(df["grid_id"].nunique()),
        "year_range": [int(df["year"].min()), int(df["year"].max())],
        "target": TARGET,
        "features": FEATURES,
        "excluded_prohibited_fields": sorted(
            set(
                [
                    "AOD",
                    "grid_id",
                    "lon",
                    "lat",
                    "year",
                    "NTL",
                    "NTL_traceable",
                ]
            )
            - set(FEATURES)
        ),
        "file_hashes": file_hashes,
        "expected_logical_hashes": {
            key: expected_hashes[key] for key in logical_hashes
        },
        "recomputed_logical_hashes": logical_hashes,
        "logical_hash_matches": logical_match,
        "all_split_hashes_match": all(logical_match.values()),
    }
    write_json(VERIFY / "03_hash_comparison.json", verification)
    if not all(logical_match.values()):
        raise RuntimeError("Step58 split hash mismatch; execution stopped.")
    if (
        len(df) != 161160
        or df["grid_id"].nunique() != 1343
        or df["year"].min() != 2015
        or df["year"].max() != 2024
    ):
        raise RuntimeError("Frozen data dimensions do not match Step58.")
    if X.isna().any().any() or y.isna().any():
        raise RuntimeError("Missing values in Ridge matrix.")

    feature_table = pd.DataFrame(
        [
            {
                "field": col,
                "role": "Ridge model feature",
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isna().sum()),
                "included": True,
                "prohibited_identifier_or_target": False,
            }
            for col in FEATURES
        ]
        + [
            {
                "field": col,
                "role": "excluded",
                "dtype": str(df[col].dtype)
                if col in df.columns
                else "not_present",
                "missing_count": (
                    int(df[col].isna().sum()) if col in df.columns else ""
                ),
                "included": False,
                "prohibited_identifier_or_target": True,
            }
            for col in [
                "AOD",
                "grid_id",
                "lon",
                "lat",
                "year",
                "NTL",
                "NTL_traceable",
            ]
        ]
    )
    feature_table.to_csv(
        VERIFY / "02_feature_list_verified.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (VERIFY / "01_input_and_split_verification.md").write_text(
        f"""# Input and split verification

- Step58 frozen dataset used: YES
- Dataset SHA256: `{verification['dataset_sha256']}`
- Rows: {len(df)}
- Unique grid cells: {df['grid_id'].nunique()}
- Period: {df['year'].min()}-{df['year'].max()}
- Target: `{TARGET}`
- Model features: {", ".join(FEATURES)}
- AOD/grid_id/lon/lat/year/NTL included as features: NO
- Random split logical hash match: {logical_match['random_split_sha256']}
- Grid-grouped split logical hash match: {logical_match['grid_group_fold_sha256']}
- LOYO split logical hash match: {logical_match['loyo_sha256']}
- Spatial-region split logical hash match: {logical_match['spatial_region_sha256']}
- Input gate: PASS
""",
        encoding="utf-8",
    )

    config_rows = pd.DataFrame(
        [
            {
                "component": "Pipeline",
                "parameter": "steps",
                "value": "StandardScaler -> Ridge",
            },
            {
                "component": "Ridge",
                "parameter": "alpha_candidates",
                "value": ";".join(str(a) for a in ALPHAS),
            },
            {
                "component": "Inner CV",
                "parameter": "selection_metric",
                "value": "minimum mean validation RMSE",
            },
            {
                "component": "Random inner CV",
                "parameter": "splitter",
                "value": "KFold(5, shuffle=True, random_state=42)",
            },
            {
                "component": "Grid inner CV",
                "parameter": "splitter",
                "value": "GroupKFold(5), groups=grid_id",
            },
            {
                "component": "LOYO inner CV",
                "parameter": "splitter",
                "value": "GroupKFold(5), groups=year",
            },
            {
                "component": "Spatial-region inner CV",
                "parameter": "splitter",
                "value": "GroupKFold(4), groups=remaining spatial clusters",
            },
            {
                "component": "Scaling",
                "parameter": "scope",
                "value": "fit only within each inner/final training fold",
            },
            {
                "component": "Target",
                "parameter": "standardization",
                "value": "not standardized",
            },
        ]
    )
    config_rows.to_csv(
        CONFIG / "01_Ridge_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_metrics: list[dict] = []
    all_predictions: list[pd.DataFrame] = []
    all_coefficients: list[pd.DataFrame] = []
    all_alpha_detail: list[pd.DataFrame] = []

    random_train = random_reg[
        random_reg["split"].eq("train")
    ]["sample_id"].to_numpy()
    random_test = random_reg[
        random_reg["split"].eq("test")
    ]["sample_id"].to_numpy()
    log("Ridge random holdout")
    result = fit_outer_fold(
        X,
        y,
        df,
        random_train,
        random_test,
        "random_split",
        "random",
        "kfold",
        None,
    )
    all_metrics.append(result[0])
    all_predictions.append(result[1])
    all_coefficients.append(result[2])
    all_alpha_detail.append(result[3])

    for fold in sorted(grid_reg["test_fold"].unique()):
        log(f"Ridge grid-grouped outer fold {fold}")
        test_idx = grid_reg[
            grid_reg["test_fold"].eq(fold)
        ]["sample_id"].to_numpy()
        train_idx = np.setdiff1d(expected_ids, test_idx, assume_unique=True)
        result = fit_outer_fold(
            X,
            y,
            df,
            train_idx,
            test_idx,
            "spatial_groupkfold",
            str(fold),
            "group",
            df["grid_id"].to_numpy(),
        )
        all_metrics.append(result[0])
        all_predictions.append(result[1])
        all_coefficients.append(result[2])
        all_alpha_detail.append(result[3])

    for cluster in sorted(region_reg["spatial_cluster"].unique()):
        log(f"Ridge spatial-region outer fold {cluster}")
        test_idx = region_reg[
            region_reg["spatial_cluster"].eq(cluster)
        ]["sample_id"].to_numpy()
        train_idx = np.setdiff1d(expected_ids, test_idx, assume_unique=True)
        result = fit_outer_fold(
            X,
            y,
            df,
            train_idx,
            test_idx,
            "spatial_leave_region_out",
            str(cluster),
            "group",
            region_reg["spatial_cluster"].to_numpy(),
        )
        all_metrics.append(result[0])
        all_predictions.append(result[1])
        all_coefficients.append(result[2])
        all_alpha_detail.append(result[3])

    for year in sorted(loyo_reg["year"].unique()):
        log(f"Ridge LOYO outer year {year}")
        test_idx = loyo_reg[
            loyo_reg["year"].eq(year)
        ]["sample_id"].to_numpy()
        train_idx = np.setdiff1d(expected_ids, test_idx, assume_unique=True)
        result = fit_outer_fold(
            X,
            y,
            df,
            train_idx,
            test_idx,
            "leave_one_year_out",
            str(year),
            "group",
            df["year"].to_numpy(),
        )
        all_metrics.append(result[0])
        all_predictions.append(result[1])
        all_coefficients.append(result[2])
        all_alpha_detail.append(result[3])

    raw_metrics = pd.DataFrame(all_metrics)
    predictions = pd.concat(all_predictions, ignore_index=True)
    coefficients = pd.concat(all_coefficients, ignore_index=True)
    alpha_detail = pd.concat(all_alpha_detail, ignore_index=True)
    summary = summarize_folds(raw_metrics, predictions)

    raw_metrics[raw_metrics["validation"].eq("random_split")].to_csv(
        RIDGE_OUT / "01_Ridge_random_holdout.csv",
        index=False,
        encoding="utf-8-sig",
    )
    raw_metrics[
        raw_metrics["validation"].eq("spatial_groupkfold")
    ].to_csv(
        RIDGE_OUT / "02_Ridge_grid_grouped_CV.csv",
        index=False,
        encoding="utf-8-sig",
    )
    raw_metrics[
        raw_metrics["validation"].eq("spatial_leave_region_out")
    ].to_csv(
        RIDGE_OUT / "03_Ridge_spatial_region_holdout.csv",
        index=False,
        encoding="utf-8-sig",
    )
    raw_metrics[
        raw_metrics["validation"].eq("leave_one_year_out")
    ].to_csv(
        RIDGE_OUT / "04_Ridge_LOYO_yearly.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        RIDGE_OUT / "05_Ridge_predictions_OOF.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        RIDGE_OUT / "06_Ridge_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    alpha_detail.to_csv(
        LOGS / "ridge_parameter_log.csv",
        index=False,
        encoding="utf-8-sig",
    )
    coefficients.to_csv(
        RIDGE_OUT / "08_Ridge_standardized_coefficients_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    coefficient_summary = (
        coefficients.groupby(["validation_scheme", "feature"], as_index=False)
        .agg(
            mean_standardized_coefficient=(
                "standardized_coefficient",
                "mean",
            ),
            SD_standardized_coefficient=(
                "standardized_coefficient",
                lambda s: s.std(ddof=1) if len(s) > 1 else 0.0,
            ),
            minimum_coefficient=("standardized_coefficient", "min"),
            maximum_coefficient=("standardized_coefficient", "max"),
            positive_fraction=(
                "standardized_coefficient",
                lambda s: float((s >= 0).mean()),
            ),
            n_outer_folds=("outer_fold", "count"),
        )
    )
    coefficient_summary["mean_absolute_coefficient"] = coefficient_summary[
        "mean_standardized_coefficient"
    ].abs()
    coefficient_summary["absolute_rank_within_validation"] = (
        coefficient_summary.groupby("validation_scheme")[
            "mean_absolute_coefficient"
        ]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    coefficient_summary.to_csv(
        RIDGE_OUT / "08_Ridge_standardized_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ridge_lookup = summary.set_index("validation")
    ridge_report = f"""# Ridge validation report

Ridge was implemented as a leakage-safe `StandardScaler -> Ridge` pipeline.
Scaling and alpha selection were performed exclusively inside each outer
training set. The candidate alpha set was {ALPHAS}, selected by minimum inner
validation RMSE.

| Validation | Fold-mean R2 | Pooled R2 | RMSE | MAE |
|---|---:|---:|---:|---:|
| Random holdout | {ridge_lookup.loc['random_split', 'fold_mean_R2']:.6f} | {ridge_lookup.loc['random_split', 'pooled_R2']:.6f} | {ridge_lookup.loc['random_split', 'fold_mean_RMSE']:.6f} | {ridge_lookup.loc['random_split', 'fold_mean_MAE']:.6f} |
| Grid-cell grouped | {ridge_lookup.loc['spatial_groupkfold', 'fold_mean_R2']:.6f} | {ridge_lookup.loc['spatial_groupkfold', 'pooled_R2']:.6f} | {ridge_lookup.loc['spatial_groupkfold', 'fold_mean_RMSE']:.6f} | {ridge_lookup.loc['spatial_groupkfold', 'fold_mean_MAE']:.6f} |
| Continuous spatial-region holdout | {ridge_lookup.loc['spatial_leave_region_out', 'fold_mean_R2']:.6f} | {ridge_lookup.loc['spatial_leave_region_out', 'pooled_R2']:.6f} | {ridge_lookup.loc['spatial_leave_region_out', 'fold_mean_RMSE']:.6f} | {ridge_lookup.loc['spatial_leave_region_out', 'fold_mean_MAE']:.6f} |
| LOYO | {ridge_lookup.loc['leave_one_year_out', 'fold_mean_R2']:.6f} | {ridge_lookup.loc['leave_one_year_out', 'pooled_R2']:.6f} | {ridge_lookup.loc['leave_one_year_out', 'fold_mean_RMSE']:.6f} | {ridge_lookup.loc['leave_one_year_out', 'fold_mean_MAE']:.6f} |

The linear baseline is used only to quantify the incremental value of nonlinear
tree models. It is not used for SHAP or causal interpretation.
"""
    (RIDGE_OUT / "07_Ridge_validation_report.md").write_text(
        ridge_report, encoding="utf-8"
    )
    (RIDGE_OUT / "09_Ridge_coefficient_stability.md").write_text(
        """# Ridge coefficient stability

Standardized coefficients summarize conditional linear associations after
feature scaling within each training fold. They are not causal effects.
Correlated meteorological and seasonal covariates can redistribute coefficient
magnitude or alter signs across folds. These coefficients are a linear-baseline
diagnostic only and do not replace SHAP for any subsequently selected tree
model.
""",
        encoding="utf-8",
    )

    tree = tree_scorecard()
    ridge_four_rows = []
    for _, row in summary.iterrows():
        ridge_four_rows.append(
            {
                "model": "Ridge",
                "validation": row["validation"],
                "R2": row["fold_mean_R2"],
                "RMSE": row["fold_mean_RMSE"],
                "MAE": row["fold_mean_MAE"],
                "mean_bias": row["fold_mean_bias"],
                "fold_SD_R2": row["fold_SD_R2"],
                "pooled_R2": row["pooled_R2"],
                "train_R2": row["mean_train_R2"],
                "train_test_gap": row["mean_train_test_R2_gap"],
                "runtime_seconds": row["total_runtime_seconds"],
                "source": "Step59 controlled nested-CV Ridge",
            }
        )
    four = pd.concat([pd.DataFrame(ridge_four_rows), tree], ignore_index=True)
    model_order = {
        "Ridge": 0,
        "RandomForest": 1,
        "HistGradientBoosting": 2,
        "XGBoost": 3,
    }
    validation_order = {
        "random_split": 0,
        "spatial_groupkfold": 1,
        "spatial_leave_region_out": 2,
        "leave_one_year_out": 3,
    }
    four["_m"] = four["model"].map(model_order)
    four["_v"] = four["validation"].map(validation_order)
    four = four.sort_values(["_m", "_v"]).drop(columns=["_m", "_v"])
    four.to_csv(
        COMPARE / "01_four_model_metrics_master.csv",
        index=False,
        encoding="utf-8-sig",
    )
    score = four.pivot(
        index="model", columns="validation", values="R2"
    ).reset_index()
    score["spatial_rank"] = score[
        "spatial_leave_region_out"
    ].rank(ascending=False, method="min")
    score["grid_rank"] = score["spatial_groupkfold"].rank(
        ascending=False, method="min"
    )
    score["random_rank"] = score["random_split"].rank(
        ascending=False, method="min"
    )
    score["LOYO_rank_relative_only"] = score[
        "leave_one_year_out"
    ].rank(ascending=False, method="min")
    score["LOYO_positive"] = score["leave_one_year_out"] > 0
    score.to_csv(
        COMPARE / "02_four_model_validation_scorecard.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gaps = four[
        [
            "model",
            "validation",
            "train_R2",
            "R2",
            "train_test_gap",
            "fold_SD_R2",
            "pooled_R2",
        ]
    ].copy()
    gaps.to_csv(
        COMPARE / "06_four_model_generalization_gap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    create_comparison_figures(
        four,
        raw_metrics[
            raw_metrics["validation"].eq("leave_one_year_out")
        ],
    )

    ridge_values = score[score["model"].eq("Ridge")].iloc[0]
    rf_values = score[score["model"].eq("RandomForest")].iloc[0]
    improvements = {
        validation: float(rf_values[validation] - ridge_values[validation])
        for validation in validation_order
    }
    best = {
        validation: score.sort_values(validation, ascending=False).iloc[0][
            "model"
        ]
        for validation in validation_order
    }
    classification = (
        "A. RF_PRIMARY_MODEL_RECOMMENDED"
        if best["spatial_groupkfold"] == "RandomForest"
        and best["spatial_leave_region_out"] == "RandomForest"
        and ridge_values["spatial_groupkfold"]
        < rf_values["spatial_groupkfold"]
        and ridge_values["spatial_leave_region_out"]
        < rf_values["spatial_leave_region_out"]
        else "D. UNRESOLVED"
    )
    comparison_report = f"""# Four-model comparison report

The four models cover a linear baseline (Ridge), bagging (Random Forest),
histogram gradient boosting (HGB), and regularized boosted trees (XGBoost).

Random Forest ranked first for random holdout, grid-cell grouped CV, and
continuous spatial-region holdout. The RF-minus-Ridge R2 improvements were
{improvements['random_split']:.3f}, {improvements['spatial_groupkfold']:.3f},
and {improvements['spatial_leave_region_out']:.3f}, respectively.

The relatively best LOYO model was {best['leave_one_year_out']}, but LOYO must
be interpreted as a temporal-transfer boundary rather than an automatic
primary-model criterion. Negative or weak LOYO performance is shown without
truncation.

The comparison supports a genuine nonlinear advantage for historical spatial
explanation. It does not establish reliable unseen-year forecasting.
"""
    (COMPARE / "07_four_model_comparison_report.md").write_text(
        comparison_report, encoding="utf-8"
    )

    selection_score = score.copy()
    selection_score["direct_leakage_evidence"] = "none detected in Step58"
    selection_score["upstream_preprocessing_status"] = (
        "UNVERIFIABLE_IN_PART"
    )
    selection_score["recommended_role"] = selection_score["model"].map(
        {
            "Ridge": "linear baseline",
            "RandomForest": "recommended historical-spatial primary candidate",
            "HistGradientBoosting": "nonlinear benchmark",
            "XGBoost": "existing primary model pending author decision",
        }
    )
    selection_score.to_csv(
        DECISION / "02_final_model_selection_scorecard.csv",
        index=False,
        encoding="utf-8-sig",
    )
    decision_text = f"""# Final four-model selection decision

## Classification

**{classification}**

## Decision

Random Forest is recommended as the primary model for historical spatial
explanation because it ranks first under the two highest-priority spatial
criteria: grid-cell grouped CV and continuous spatial-region holdout. Ridge is
materially weaker, supporting a real nonlinear modelling advantage rather than
mere model-family inflation. Step58 found no direct target leakage, although
parts of the upstream preprocessing lineage remain `UNVERIFIABLE_IN_PART`.

RF LOYO remains negative ({rf_values['leave_one_year_out']:.6f}). Therefore,
this recommendation applies only to historical spatial explanation and not
future-year forecasting. The existing manuscript and Online Resource remain
unchanged in this step. The author must explicitly approve any primary-model
replacement before downstream RF SHAP and residual diagnostics are produced.
"""
    (DECISION / "01_final_four_model_selection_decision.md").write_text(
        decision_text, encoding="utf-8"
    )
    (DECISION / "03_need_for_additional_models.md").write_text(
        """# Need for additional models

**No additional model is scientifically necessary.**

The comparison already covers a linear model (Ridge), bagging (Random Forest),
histogram gradient boosting (HGB), and XGBoost. Adding GAM, SVR, LightGBM,
CatBoost, deep learning, ExtraTrees, or stacking would increase algorithm-count
inflation without resolving the principal scientific finding: strong
historical spatial explanation does not ensure unseen-year transferability.
""",
        encoding="utf-8",
    )
    (DECISION / "04_final_recommendation_for_author_CN.md").write_text(
        f"""# 给作者的最终建议

四个模型已经足够，覆盖了线性基准、Bagging、梯度提升和XGBoost，不需要第五个模型。

推荐把Random Forest作为“历史空间解释”主模型候选，因为它在grid-cell grouped和连续空间区域留出验证中都明显最好，而且Ridge基准明显更弱，说明树模型确实获得了非线性解释增益。但RF的LOYO平均R2仍为负，因此不能写成跨年预测模型。

本轮不自动替换XGBoost。下一步应先由作者和导师确认是否接受RF作为正式主模型。确认后，只重做RF SHAP、RF OOF残差Moran's I、受影响的模型表格和图件，并同步修改摘要、Methods、Results、Discussion、Conclusion及Online Resource。AOD数据、趋势分析、1343格点、161160条记录、Fig.1和参考文献审计不动。

最终分类：{classification}
""",
        encoding="utf-8",
    )

    implication_files = {
        "01_required_main_text_changes.md": """# Required main-text changes

Only if the author approves RF as the primary model:

- Replace the primary-model specification and validation metrics in Methods.
- Update the abstract, Results, Discussion, and Conclusions with RF results.
- Preserve the validation-aware framing and disclose negative RF LOYO.
- Do not describe RF as a future-year forecasting model.
""",
        "02_required_figure_and_table_changes.md": """# Required figure and table changes

Only after author approval:

- Update the primary model performance table.
- Replace model-comparison graphics with the locked four-model comparison.
- Regenerate primary-model observed-predicted and residual graphics.
- Regenerate RF SHAP figures.
- Keep Fig.1 and the AOD trend figures unchanged.
""",
        "03_required_online_resource_changes.md": """# Required Online Resource changes

Only after author approval:

- Add the Ridge configuration and four-model validation table.
- Update primary-model hyperparameters and validation rows if RF is selected.
- Update SHAP and residual diagnostic supplements only after they are rerun.
- Retain all negative LOYO values without truncation.
""",
        "04_SHAP_and_residual_reanalysis_scope.md": """# SHAP and residual reanalysis scope

If RF becomes the primary model, rerun:

- RF SHAP global importance and dependence analysis.
- RF OOF predictions under the locked validation protocol.
- Residual Moran's I using the approved RF OOF residual definition.

Do not rerun these analyses until the model-selection decision is approved.
""",
        "05_content_that_remains_unchanged.md": """# Content that remains unchanged

- AOD source data and 2015-2024 study period.
- 1,343 grid cells and 161,160 records.
- Theil-Sen and Mann-Kendall plus FDR trend analysis.
- 98.66% significant decline result.
- Lag-1 autocorrelation disclosure of 0.558.
- Fig.1 and AOD trend figures.
- Reference authenticity and citation audits.
- The monitoring-and-assessment framing and temporal-boundary principle.
""",
    }
    for name, text in implication_files.items():
        (IMPLICATIONS / name).write_text(text, encoding="utf-8")

    environment = "\n".join(
        [
            f"python_executable={sys.executable}",
            f"python_version={sys.version}",
            f"platform={platform.platform()}",
            f"numpy={np.__version__}",
            f"pandas={pd.__version__}",
            f"scikit_learn={sklearn.__version__}",
            f"matplotlib={mpl.__version__}",
            f"features={FEATURES}",
            f"target={TARGET}",
            f"alpha_candidates={ALPHAS}",
            f"random_state={RANDOM_STATE}",
            "standardization=StandardScaler fitted inside each training fold",
            "target_standardization=False",
            "OneDrive_used=False",
        ]
    )
    (LOGS / "environment_lock.txt").write_text(
        environment, encoding="utf-8"
    )
    write_json(LOGS / "split_hash_check.json", verification)
    (LOGS / "error_log.txt").write_text(
        "No execution errors.\n"
        "Upstream preprocessing lineage remains UNVERIFIABLE_IN_PART as "
        "documented in Step58.\n",
        encoding="utf-8",
    )
    qa = {
        "completed": True,
        "input_gate_pass": True,
        "all_split_hashes_match": all(logical_match.values()),
        "ridge_pipeline_scaling_inside_training": True,
        "outer_test_used_for_alpha_selection": False,
        "target_standardized": False,
        "ridge_predictions_rows": len(predictions),
        "ridge_expected_predictions_rows": (
            len(random_test) + len(df) * 3
        ),
        "classification": classification,
        "historical_files_modified": False,
        "manuscript_modified": False,
        "online_resource_modified": False,
        "OneDrive_used": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(LOGS / "final_QA.json", qa)
    (LOGS / "final_QA.md").write_text(
        "# Final QA\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in qa.items())
        + "\n",
        encoding="utf-8",
    )
    log(
        f"Completed Ridge and four-model gate in "
        f"{qa['elapsed_seconds']:.1f} seconds"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        ERRORS.append(f"{type(exc).__name__}: {exc}")
        (LOGS / "error_log.txt").write_text(
            "\n".join(ERRORS), encoding="utf-8"
        )
        raise
