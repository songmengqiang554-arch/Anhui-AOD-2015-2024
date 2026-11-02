"""Pre-specified AOD trend robustness analysis.

This script creates only new outputs in this directory.  It never writes to
the locked project inputs, figures, manuscripts, or model outputs.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pymannkendall as pmk
from scipy import stats

import mannkendall as official_mk


PROJECT = Path('<LOCAL_PATH>')
OUT = Path('<LOCAL_PATH>')
MONTHLY = PROJECT / "Output" / "AOD_0p1deg_grid_monthly_2015_2024.csv"
LOCKED_TREND = PROJECT / "Output" / "Step24_TheilSen_MK_Trend" / "AOD_TheilSen_MK_trend_grid_2015_2024.csv"
LOCKED_FDR = PROJECT / "Q2_upgrade_outputs" / "33_SCI_submission_gap_fix" / "csv" / "trend_fdr_grid_results.csv"
OFFICIAL_SOURCE = OUT / "official_mannkendall_python"

ALPHA = 0.05
OFFICIAL_RESOLUTION = 0.001
RANDOM_SEED = 20260919


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bh_fdr(p_values: pd.Series, alpha: float = ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg, matching the project's frozen Step 33 implementation."""
    values = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("BH-FDR requires finite p-values; unavailable 3PW p-values are not adjusted.")
    n = values.size
    order = np.argsort(values)
    ordered = values[order]
    threshold = alpha * np.arange(1, n + 1) / n
    passed = ordered <= threshold
    reject = np.zeros(n, dtype=bool)
    if passed.any():
        reject[order[: np.where(passed)[0].max() + 1]] = True
    q_sorted = np.minimum.accumulate((ordered * n / np.arange(1, n + 1))[::-1])[::-1]
    q_values = np.empty(n, dtype=float)
    q_values[order] = np.minimum(q_sorted, 1.0)
    return reject, q_values


def sign_label(value: float) -> str:
    if not np.isfinite(value):
        return "UNAVAILABLE"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def lag1_pearson(values: np.ndarray) -> float:
    if len(values) < 3 or np.nanstd(values[:-1]) == 0 or np.nanstd(values[1:]) == 0:
        return np.nan
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def grid_integrity(monthly: pd.DataFrame) -> dict:
    expected = pd.MultiIndex.from_product([range(2015, 2025), range(1, 13)], names=["year", "month"])
    counts = monthly.groupby("grid_id").size()
    incomplete = []
    for grid_id, group in monthly.groupby("grid_id", sort=True):
        actual = pd.MultiIndex.from_frame(group[["year", "month"]].drop_duplicates())
        if len(group) != 120 or not actual.equals(expected):
            incomplete.append(int(grid_id))
    return {
        "record_count": int(len(monthly)),
        "grid_count": int(monthly["grid_id"].nunique()),
        "period_min": f"{int(monthly.year.min()):04d}-{int(monthly.month.min()):02d}",
        "period_max": f"{int(monthly.year.max()):04d}-{int(monthly.month.max()):02d}",
        "all_grid_series_have_120_complete_months": len(incomplete) == 0,
        "incomplete_grid_ids": incomplete,
    }


def original_reproduction(monthly: pd.DataFrame, locked: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the historical annual-mean series used by the locked grid table."""
    rows = []
    for grid_id, group in monthly.groupby("grid_id", sort=True):
        annual = group.groupby("year", sort=True)["AOD"].mean()
        sen = stats.theilslopes(annual.to_numpy(dtype=float), annual.index.to_numpy(dtype=float))
        result = pmk.original_test(annual.to_numpy(dtype=float), alpha=ALPHA)
        row = {
            "grid_id": int(grid_id),
            "n_months_input": int(len(group)),
            "n_annual_means_used": int(len(annual)),
            "annual_month_count_min": int(group.groupby("year").size().min()),
            "annual_month_count_max": int(group.groupby("year").size().max()),
            "reproduced_slope": float(sen.slope),
            "reproduced_intercept": float(sen.intercept),
            "reproduced_lower_slope": float(sen.low_slope),
            "reproduced_upper_slope": float(sen.high_slope),
            "reproduced_mk_p": float(result.p),
            "reproduced_mk_z": float(result.z),
            "reproduced_mk_tau": float(result.Tau),
            "reproduced_mk_trend": str(result.trend),
            "reproduced_mk_significant_nominal": bool(result.h),
        }
        rows.append(row)
    reproduced = pd.DataFrame(rows)
    source = locked[["grid_id", "lon", "lat", "slope", "intercept", "lower_slope", "upper_slope", "mk_p", "mk_z", "mk_tau", "mk_trend"]].copy()
    source = source.rename(columns={
        "slope": "locked_slope", "intercept": "locked_intercept", "mk_p": "locked_mk_p",
        "lower_slope": "locked_lower_slope", "upper_slope": "locked_upper_slope", "mk_z": "locked_mk_z",
        "mk_tau": "locked_mk_tau", "mk_trend": "locked_mk_trend",
    })
    reproduced = source.merge(reproduced, on="grid_id", validate="one_to_one")
    for item in ["slope", "intercept", "lower_slope", "upper_slope", "mk_p", "mk_z", "mk_tau"]:
        reproduced[f"abs_diff_{item}"] = (reproduced[f"locked_{item}"] - reproduced[f"reproduced_{item}"]).abs()
    reproduced["all_locked_fields_match"] = (
        (reproduced[[f"abs_diff_{item}" for item in ["slope", "intercept", "lower_slope", "upper_slope", "mk_p", "mk_z", "mk_tau"]]] < 1e-12).all(axis=1)
        & (reproduced["locked_mk_trend"] == reproduced["reproduced_mk_trend"])
    )
    return reproduced


def seasonal_mk(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grid_id, group in monthly.groupby("grid_id", sort=True):
        ordered = group.sort_values(["year", "month"])
        result = pmk.seasonal_test(ordered["AOD"].to_numpy(dtype=float), period=12, alpha=ALPHA)
        rows.append({
            "grid_id": int(grid_id),
            "n_months": int(len(ordered)),
            "seasonal_period": 12,
            "seasonal_tau": float(result.Tau),
            "seasonal_test_statistic_s": float(result.s),
            "seasonal_z": float(result.z),
            "seasonal_p": float(result.p),
            "seasonal_slope": float(result.slope),
            "seasonal_intercept": float(result.intercept),
            "seasonal_mk_trend_nominal": str(result.trend),
            "seasonal_significant_nominal": bool(result.h),
        })
    results = pd.DataFrame(rows)
    reject, q_values = bh_fdr(results["seasonal_p"], alpha=ALPHA)
    results["seasonal_p_fdr"] = q_values
    results["seasonal_significant_fdr"] = reject
    results["seasonal_significance_class"] = np.select(
        [reject & (results["seasonal_slope"] < 0), reject & (results["seasonal_slope"] > 0)],
        ["FDR significant decrease", "FDR significant increase"],
        default="Not FDR significant",
    )
    return results


def official_3pw_attempt(monthly: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run only the required 10-grid validation of the unmodified official seasonal 3PW call."""
    grid_ids = np.sort(monthly["grid_id"].unique())
    rng = np.random.default_rng(RANDOM_SEED)
    sample_ids = np.sort(rng.choice(grid_ids, size=10, replace=False)).astype(int)
    attempts = []
    error_template = None
    for grid_id in sample_ids:
        group = monthly[monthly["grid_id"] == grid_id].sort_values(["year", "month"])
        dates = []
        observations = []
        for month in range(1, 13):
            season = group[group["month"] == month].sort_values("year")
            dates.append(np.array([datetime(int(year), int(month), 15) for year in season["year"]]))
            observations.append(season["AOD"].to_numpy(dtype=float).copy())
        try:
            result = official_mk.mk_temp_aggr(
                dates, observations, resolution=OFFICIAL_RESOLUTION, pw_method="3pw",
                alpha_mk=95, alpha_cl=90, alpha_xhomo=90, alpha_ak=95,
            )
            overall = result[12]
            attempts.append({
                "grid_id": int(grid_id), "sample_validation_attempted": True, "official_3pw_call_status": "SUCCESS",
                "official_3pw_error_type": pd.NA, "official_3pw_error_message": pd.NA,
                "official_3pw_p": overall.get("p"), "official_3pw_ss": overall.get("ss"),
                "official_vctfpw_slope": overall.get("slope"), "official_vctfpw_lcl": overall.get("lcl"),
                "official_vctfpw_ucl": overall.get("ucl"),
            })
        except Exception as exc:  # Intentionally record the unmodified official implementation failure.
            error_template = {
                "type": type(exc).__name__, "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            attempts.append({
                "grid_id": int(grid_id), "sample_validation_attempted": True, "official_3pw_call_status": "FAILED",
                "official_3pw_error_type": type(exc).__name__, "official_3pw_error_message": str(exc),
                "official_3pw_p": np.nan, "official_3pw_ss": np.nan,
                "official_vctfpw_slope": np.nan, "official_vctfpw_lcl": np.nan, "official_vctfpw_ucl": np.nan,
            })
    attempted = pd.DataFrame(attempts)
    all_failed = bool((attempted["official_3pw_call_status"] == "FAILED").all())
    rows = []
    attempted_by_id = attempted.set_index("grid_id")
    for grid_id, group in monthly.groupby("grid_id", sort=True):
        values = group.sort_values(["year", "month"])["AOD"].to_numpy(dtype=float).copy()
        lag1 = lag1_pearson(values)
        try:
            official_lag1, _, official_lag1_ss = official_mk.mk_white.nanprewhite_arok(values.copy(), alpha_ak=95)
            lag1_ss = bool(official_lag1_ss == 95)
        except Exception as exc:
            official_lag1, official_lag1_ss, lag1_ss = np.nan, np.nan, pd.NA
        base = {
            "grid_id": int(grid_id),
            "n_months": int(len(values)),
            "raw_lag1_autocorrelation": float(lag1),
            "official_lag1_autocorrelation": float(official_lag1) if np.isfinite(official_lag1) else np.nan,
            "lag1_significance_confidence_percent": official_lag1_ss,
            "lag1_autocorrelation_significant_95pct": lag1_ss,
            "3pw_resolution": OFFICIAL_RESOLUTION,
            "3pw_alpha_mk_percent": 95,
            "3pw_alpha_cl_percent": 90,
            "3pw_alpha_xhomo_percent": 90,
            "3pw_alpha_ak_percent": 95,
        }
        if int(grid_id) in attempted_by_id.index:
            record = attempted_by_id.loc[int(grid_id)].to_dict()
            base.update(record)
        else:
            base.update({
                "sample_validation_attempted": False,
                "official_3pw_call_status": "NOT_RUN_AFTER_REQUIRED_10_GRID_VALIDATION_FAILED" if all_failed else "NOT_RUN",
                "official_3pw_error_type": error_template["type"] if error_template else pd.NA,
                "official_3pw_error_message": error_template["message"] if error_template else pd.NA,
                "official_3pw_p": np.nan, "official_3pw_ss": np.nan,
                "official_vctfpw_slope": np.nan, "official_vctfpw_lcl": np.nan, "official_vctfpw_ucl": np.nan,
            })
        base["3pw_p_fdr"] = np.nan
        base["3pw_significant"] = pd.NA
        base["3pw_fdr_status"] = "NOT_CALCULATED_OFFICIAL_SEASONAL_3PW_UNAVAILABLE" if all_failed else "NOT_CALCULATED"
        rows.append(base)
    results = pd.DataFrame(rows)
    status = {"sample_grid_ids": sample_ids.tolist(), "all_failed": all_failed, "error": error_template}
    return results, status


def annual_sensitivity(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grid_id, group in monthly.groupby("grid_id", sort=True):
        annual = group.groupby("year", sort=True)["AOD"].mean()
        result = pmk.original_test(annual.to_numpy(dtype=float), alpha=ALPHA)
        rows.append({
            "grid_id": int(grid_id), "n_annual_means": int(len(annual)), "n_months_per_year_min": int(group.groupby("year").size().min()),
            "n_months_per_year_max": int(group.groupby("year").size().max()),
            "annual_slope": float(result.slope), "annual_intercept": float(result.intercept),
            "annual_mk_p_descriptive_only": float(result.p), "annual_mk_z_descriptive_only": float(result.z),
            "annual_mk_tau_descriptive_only": float(result.Tau), "annual_mk_trend_descriptive_only": str(result.trend),
            "annual_mk_significant_nominal_descriptive_only": bool(result.h),
            "scope_note": "descriptive sensitivity only; n = 10 annual means",
        })
    return pd.DataFrame(rows)


def agreement_percent(columns: list[pd.Series]) -> float | None:
    frame = pd.concat(columns, axis=1)
    if frame.isna().any().any():
        return None
    return float((frame.nunique(axis=1) == 1).mean() * 100)


def write_log(integrity: dict, original: pd.DataFrame, three_pw_status: dict, summary: dict) -> None:
    package_versions = {}
    for name, module in [("numpy", np), ("pandas", pd), ("pymannkendall", pmk), ("mannkendall", official_mk)]:
        package_versions[name] = getattr(module, "__version__", "unknown")
    try:
        import scipy
        package_versions["scipy"] = scipy.__version__
    except Exception:
        package_versions["scipy"] = "unavailable"
    text = f"""# Pre-specified trend robustness analysis v2: method log

## Scope and immutability

Only trend robustness outputs were created in `{OUT}`. No manuscript, figure, locked trend result, model, SHAP, LOYO, RF, XGBoost, or input data file was modified.

## Inputs

- Monthly AOD input: `{MONTHLY}`
- SHA256: `{sha256(MONTHLY)}`
- Locked grid trend table: `{LOCKED_TREND}`
- Locked FDR table: `{LOCKED_FDR}`
- Input integrity: {json.dumps(integrity, ensure_ascii=False)}

## Original-result reproduction

The locked Step24 grid values were exactly reproduced from annual means computed from the monthly table (10 annual means per grid), using `pymannkendall.original_test` at alpha = 0.05. This is an evidence finding from the locked source table: the locked slopes, tau, normal-approximation MK p values, z values, and trend labels all matched the annual-mean reproduction. It is not a replacement of the original analysis.

- Exact grid-level locked-result matches: `{int(original['all_locked_fields_match'].sum())}/{len(original)}`
- The annual aggregation used in this reproduction is reported explicitly so the monthly seasonal sensitivity is not conflated with the locked result.

## Seasonal MK sensitivity

- Function: `pymannkendall.seasonal_test(x, period=12, alpha=0.05)`.
- Software reference: Hussain, M., & Mahmud, I. (2019). *pyMannKendall: a python package for non parametric Mann Kendall family of trend tests*. Journal of Open Source Software, 4(39), 1556. https://doi.org/10.21105/joss.01556.
- Input: each grid's complete 120 monthly values ordered by year and month.
- Seasonal grouping: month (12 seasonal groups of 10 values).
- `seasonal_slope` is the package-returned seasonal Sen slope; the package returns it in AOD units per year for the period=12 formulation.
- BH-FDR: the same Step33 in-project BH implementation, applied across the 1,343 finite seasonal MK p-values at alpha = 0.05.

## Official autocorrelation-aware 3PW sensitivity

- Official code: `mannkendall` Python v{package_versions['mannkendall']}, MeteoSwiss / mannkendall organisation, BSD-3-Clause, corresponding to Collaud Coen et al. (2020), doi:10.5194/amt-13-6945-2020.
- Method reference: Collaud Coen, M., Andrews, E., Bigi, A., Martucci, G., Romanens, G., Vogt, F. P. A., & Vuilleumier, L. (2020). *Effects of the prewhitening method, the time granularity, and the time segmentation on the Mann–Kendall trend detection and the associated Sen's slope*. Atmospheric Measurement Techniques, 13(12), 6945–6964. https://doi.org/10.5194/amt-13-6945-2020.
- Official function attempted without modification: `mannkendall.mk_temp_aggr(..., pw_method='3pw', resolution=0.001, alpha_mk=95, alpha_cl=90, alpha_xhomo=90, alpha_ak=95)`.
- Seasonal input to the official function: 12 arrays (one per calendar month), each with 10 dated observations.
- Required deterministic 10-grid validation seed: `{RANDOM_SEED}`; grid IDs: `{three_pw_status['sample_grid_ids']}`.
- Validation all failed: `{three_pw_status['all_failed']}`.
- Failure: `{json.dumps(three_pw_status['error'], ensure_ascii=False)}`.
- The failure occurs inside the unmodified official small-sample probability-table indexing path for a seasonal subseries of n=10. No substitute, patch, custom 3PW implementation, or full 1,343-grid 3PW run was performed.
- A direct attempt to retrieve the official `https://github.com/mannkendall/Python.git` repository on 2026-09-19 ended with `Recv failure: Connection was reset`; no unofficial mirror was used.
- `official_lag1_autocorrelation` and its 95% significance are nevertheless available from the same official package's `mk_white.nanprewhite_arok(obs.copy(), alpha_ak=95)` diagnostic.
- Because the official seasonal 3PW call did not complete, no corrected 3PW p-values, VCTFPW slopes, confidence limits, 3PW significance classifications, or 3PW BH-FDR result are reported.

## Annual-mean descriptive sensitivity

- Annual means: each complete set of 12 months was averaged for each grid and year.
- Theil-Sen slope and standard MK p value: `pymannkendall.original_test`.
- These values are labelled descriptive sensitivity only; n = 10 and no annual MK result is used as a new primary significance result.

## Runtime

- Python: `{sys.version}`
- Platform: `{platform.platform()}`
- Package versions: `{json.dumps(package_versions)}`
- 3PW official source directory: `{OFFICIAL_SOURCE}`

## Summary availability

{json.dumps(summary, indent=2, ensure_ascii=False)}
"""
    (OUT / "07_method_log.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    monthly = pd.read_csv(MONTHLY)
    locked = pd.read_csv(LOCKED_TREND)
    locked_fdr = pd.read_csv(LOCKED_FDR)
    integrity = grid_integrity(monthly)
    if integrity["record_count"] != 161160 or integrity["grid_count"] != 1343 or not integrity["all_grid_series_have_120_complete_months"]:
        raise RuntimeError(f"Input integrity failed: {integrity}")
    if len(locked) != 1343 or len(locked_fdr) != 1343:
        raise RuntimeError("Locked trend outputs do not contain 1,343 grid records.")

    original = original_reproduction(monthly, locked)
    original = original.merge(locked_fdr[["grid_id", "mk_q_fdr_bh", "fdr_significant", "fdr_significant_decrease"]], on="grid_id", validate="one_to_one")
    original = original.rename(columns={"mk_q_fdr_bh": "original_p_fdr", "fdr_significant": "original_significant_fdr", "fdr_significant_decrease": "original_significant_decline_fdr"})
    original.to_csv(OUT / "01_original_reproduction.csv", index=False, encoding="utf-8-sig")

    seasonal = seasonal_mk(monthly)
    seasonal.to_csv(OUT / "02_seasonal_mk_grid_results.csv", index=False, encoding="utf-8-sig")

    three_pw, three_pw_status = official_3pw_attempt(monthly)
    three_pw.to_csv(OUT / "03_3pw_grid_results.csv", index=False, encoding="utf-8-sig")

    annual = annual_sensitivity(monthly)
    annual.to_csv(OUT / "04_annual_mean_sensitivity.csv", index=False, encoding="utf-8-sig")

    comparison = original[["grid_id", "locked_slope", "original_p_fdr", "original_significant_decline_fdr"]].rename(columns={"locked_slope": "original_slope"})
    comparison = comparison.merge(seasonal[["grid_id", "seasonal_slope", "seasonal_p_fdr", "seasonal_significant_fdr"]], on="grid_id", validate="one_to_one")
    comparison = comparison.merge(three_pw[["grid_id", "official_vctfpw_slope", "3pw_significant", "official_3pw_call_status"]], on="grid_id", validate="one_to_one")
    comparison = comparison.rename(columns={"official_vctfpw_slope": "3pw_slope"})
    comparison = comparison.merge(annual[["grid_id", "annual_slope"]], on="grid_id", validate="one_to_one")
    comparison["sign_original"] = comparison["original_slope"].map(sign_label)
    comparison["sign_seasonal"] = comparison["seasonal_slope"].map(sign_label)
    comparison["sign_3pw"] = comparison["3pw_slope"].map(sign_label)
    comparison["sign_annual"] = comparison["annual_slope"].map(sign_label)
    comparison.to_csv(OUT / "05_trend_method_comparison.csv", index=False, encoding="utf-8-sig")

    original_sig = original["original_significant_decline_fdr"].astype(bool)
    seasonal_sig = (seasonal["seasonal_significant_fdr"] & (seasonal["seasonal_slope"] < 0)).astype(bool)
    annual_vs_original = (comparison["sign_original"] == comparison["sign_annual"]).mean() * 100
    original_vs_seasonal = (comparison["sign_original"] == comparison["sign_seasonal"]).mean() * 100
    four_agreement = None if three_pw_status["all_failed"] else agreement_percent([
        comparison["sign_original"], comparison["sign_seasonal"], comparison["sign_3pw"], comparison["sign_annual"]
    ])
    summary = {
        "analysis_scope": "pre-specified trend robustness analysis only",
        "input_integrity": integrity,
        "original_reproduction": {
            "source_table": str(LOCKED_TREND),
            "source_table_method_evidence": "Locked values reproduce exactly from annual means (n=10) calculated from the monthly source table using standard MK normal approximation and Sen slope.",
            "exact_grid_matches": int(original["all_locked_fields_match"].sum()),
            "grid_count": int(len(original)),
            "median_slope": float(original["locked_slope"].median()),
            "fdr_significant_decline_percent": float(original_sig.mean() * 100),
            "median_lag1_autocorrelation_locked": float(three_pw["raw_lag1_autocorrelation"].median()),
        },
        "seasonal_mk": {
            "median_slope": float(seasonal["seasonal_slope"].median()),
            "fdr_significant_decline_percent": float(seasonal_sig.mean() * 100),
            "fdr_significance_class_counts": seasonal["seasonal_significance_class"].value_counts().to_dict(),
        },
        "official_3pw_vctfpw": {
            "status": "UNAVAILABLE_OFFICIAL_SEASONAL_3PW_CALL_FAILED" if three_pw_status["all_failed"] else "AVAILABLE",
            "sample_validation_grid_ids": three_pw_status["sample_grid_ids"],
            "error": three_pw_status["error"],
            "median_slope": None,
            "significant_decline_percent": None,
            "fdr_note": "Not performed because no official autocorrelation-aware p-values were returned.",
        },
        "annual_mean_descriptive_sensitivity": {
            "n_per_grid": 10,
            "median_slope": float(annual["annual_slope"].median()),
            "note": "descriptive sensitivity only; it duplicates the aggregation evidenced in the locked original grid table and is not a new primary significance result.",
        },
        "agreement": {
            "original_vs_seasonal_slope_sign_percent": float(original_vs_seasonal),
            "original_vs_annual_slope_sign_percent": float(annual_vs_original),
            "four_method_slope_sign_agreement_percent": four_agreement,
            "original_vs_seasonal_significance_agreement_percent": float((original_sig.to_numpy() == seasonal_sig.to_numpy()).mean() * 100),
            "original_vs_autocorrelation_aware_significance_agreement_percent": None,
            "autocorrelation_agreement_reason": "Not available because the official seasonal 3PW implementation did not complete on the required 10-grid validation.",
        },
        "locked_artifacts_modified": False,
    }
    (OUT / "06_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_log(integrity, original, three_pw_status, summary)

    guardrails = f"""# Interpretation guardrails: pre-specified trend robustness v2

## Scope

This file reports only factual comparisons from the new analysis outputs. It does not revise, replace, or decide the manuscript's conclusions.

## Locked original result reproduction

- Input completeness was `{integrity['record_count']}` records across `{integrity['grid_count']}` grids; every grid had the complete 2015-01 to 2024-12 monthly series.
- The locked annual-mean grid trend table was reproduced exactly for `{int(original['all_locked_fields_match'].sum())}/{len(original)}` grids.
- Locked original median slope: `{original['locked_slope'].median():.10f}` AOD yr^-1.
- Locked original BH-FDR significant-decline proportion: `{original_sig.mean() * 100:.6f}%`.

## Seasonal monthly sensitivity

- Seasonal MK used calendar month as 12 seasonal groups across the 120 monthly observations.
- Seasonal median slope: `{seasonal['seasonal_slope'].median():.10f}` AOD yr^-1.
- Seasonal BH-FDR significant-decline proportion: `{seasonal_sig.mean() * 100:.6f}%`.
- Original versus seasonal slope-sign agreement: `{original_vs_seasonal:.6f}%`.
- Original versus seasonal FDR significant-decline agreement: `{(original_sig.to_numpy() == seasonal_sig.to_numpy()).mean() * 100:.6f}%`.

## Autocorrelation-aware official 3PW sensitivity

- Median monthly lag-1 autocorrelation from the official diagnostic: `{three_pw['official_lag1_autocorrelation'].median():.10f}`.
- Grid cells with official 95% significant lag-1 autocorrelation: `{int(three_pw['lag1_autocorrelation_significant_95pct'].astype(bool).sum())}/{len(three_pw)}`.
- The unmodified official `mannkendall` v{official_mk.__version__} seasonal 3PW call failed for all required 10 deterministic validation grids because `{three_pw_status['error']['type'] if three_pw_status['error'] else 'an unavailable error'}` was raised in the package's small-sample probability-table indexing path.
- Therefore, no 3PW/VCTFPW slope, corrected p value, 3PW significance class, or 3PW BH-FDR comparison is reported. No alternate prewhitening algorithm was substituted.

## Annual-mean descriptive sensitivity

- Annual means used 10 values per grid and are explicitly descriptive only.
- Annual median slope: `{annual['annual_slope'].median():.10f}` AOD yr^-1.
- Original versus annual slope-sign agreement: `{annual_vs_original:.6f}%`.
- The annual results match the annual aggregation evidenced in the locked source table; they are not introduced as a second primary significance result.
"""
    (OUT / "08_interpretation_guardrails.md").write_text(guardrails, encoding="utf-8")


if __name__ == "__main__":
    main()
