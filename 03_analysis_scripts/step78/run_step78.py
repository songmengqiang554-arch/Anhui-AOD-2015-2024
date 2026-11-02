"""Read-only inputs; bounded, pre-specified trend sensitivity outputs only."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
import platform
import sys
import traceback
import warnings
from urllib.request import Request, urlopen
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pymannkendall as pmk
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path('<LOCAL_PATH>')
OUT = Path(__file__).resolve().parents[1]
OLD = ROOT / "67_EMA_REVISION_2026-09/06_ANALYSIS/trend_robustness_v2"
AOD = Path('<LOCAL_PATH>')
LOCKED = Path('<LOCAL_PATH>')
FDR = ROOT / "33_SCI_submission_gap_fix/csv/trend_fdr_grid_results.csv"
SEED = 20261016
ALPHA = 0.05
SAMPLE_IDS = [132, 277, 319, 711, 869, 1011, 1051, 1059, 1157, 1325]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def target(name):
    p = (OUT / name).resolve()
    assert p.is_relative_to(OUT) and "onedrive" not in str(p).lower()
    if p.exists():
        raise FileExistsError(f"Refusing to overwrite {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def json_out(name, obj):
    target(name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv_out(name, rows):
    pd.DataFrame(rows).to_csv(target(name), index=False, encoding="utf-8-sig", float_format="%.17g")


def bh(p):
    p = np.asarray(p, dtype=float)
    assert len(p) == 1343 and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    order = np.argsort(p, kind="stable")
    qsort = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    q = np.empty_like(p)
    q[order] = np.minimum(qsort, 1)
    re, qr, _, _ = multipletests(p, alpha=ALPHA, method="fdr_bh")
    assert np.allclose(q, qr, atol=1e-14, rtol=1e-13)
    assert np.array_equal(q <= ALPHA, re)
    return q


def corr_skt_batch(x):
    """Complete-data Hirsch-Slack covariance sum; RKT continuity correction.

    x axes: replicate, year, season. Source: rkt.R lines 295-326.
    Sum of the covariance matrix is evaluated without materializing it.
    """
    x = np.asarray(x, dtype=float)
    assert x.ndim == 3 and np.isfinite(x).all()
    b, n, m = x.shape
    i, j = np.triu_indices(n, 1)
    d = np.sign(x[:, j, :] - x[:, i, :])
    pair_score = d.sum(axis=2)
    s = pair_score.sum(axis=1)
    ranks = stats.rankdata(x, method="average", axis=1)
    sum_ranks = ranks.sum(axis=2)
    ksum = (pair_score ** 2).sum(axis=1)
    gsum = (sum_ranks ** 2).sum(axis=1)
    var = (ksum + 4 * gsum - (m ** 2) * n * (n + 1) ** 2) / 3
    z = np.zeros(b)
    valid = var > 0
    z[valid] = (s[valid] - np.sign(s[valid])) / np.sqrt(var[valid])
    p = np.ones(b)
    p[valid] = 2 * stats.norm.sf(np.abs(z[valid]))
    p[(~valid) & (s != 0)] = np.nan
    return s, var, z, p


def independent_covariance(x):
    """Literal complete-data covariance formula, independent of batch sums."""
    n, m = x.shape
    ranks = np.column_stack([stats.rankdata(x[:, g], method="average") for g in range(m)])
    gamma = np.empty((m, m))
    score = sum(np.sign(x[j] - x[i]).sum() for i in range(n - 1) for j in range(i + 1, n))
    for g in range(m):
        for h in range(m):
            k = sum(np.sign((x[j, g] - x[i, g]) * (x[j, h] - x[i, h]))
                    for i in range(n - 1) for j in range(i + 1, n))
            gamma[g, h] = (k + 4 * np.dot(ranks[:, g], ranks[:, h]) - n * (n + 1) ** 2) / 3
    return float(score), float(gamma.sum())


def collect_protected():
    paths = set()
    for directory in ROOT.iterdir():
        if directory.is_dir() and any(directory.name.startswith(str(n) + "_") for n in range(72, 78)):
            paths.update(p for p in directory.rglob("*") if p.is_file())
    paths.update([AOD, LOCKED, FDR])
    paths.update(p for p in OLD.glob("*") if p.is_file())
    paths.update((OLD / "official_mannkendall_python/mannkendall").glob("*.py"))
    for p in sorted(paths):
        assert "onedrive" not in str(p).lower()
        yield {"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha(p)}


def fetch_sources():
    sources = [
        ("rkt_R_source.txt", "https://raw.githubusercontent.com/cran/rkt/master/R/rkt.R"),
        ("rkt_manual.pdf", "https://cran.r-project.org/web/packages/rkt/rkt.pdf"),
        ("Hirsch_Slack_1984_primary_article.pdf", "https://manatee.wateratlas.usf.edu/upload/documents/Hirsch_et_al-1984-Water_Resources_Research.pdf"),
        ("Collaud_Coen_2020_primary_article.pdf", "https://amt.copernicus.org/articles/13/6945/2020/amt-13-6945-2020.pdf"),
    ]
    records = []
    for name, url in sources:
        rec = {"url": url, "access_time": datetime.now().isoformat(), "file": str(OUT / "source_evidence" / name)}
        try:
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0 research-method-validation"}), timeout=35) as response:
                data = response.read()
                rec.update(http_status=response.status, content_type=response.headers.get("Content-Type"), bytes=len(data))
            if name.endswith(".pdf"):
                assert data.startswith(b"%PDF"), "Not a PDF response"
            target("source_evidence/" + name).write_bytes(data)
            rec.update(status="ARCHIVED", sha256=hashlib.sha256(data).hexdigest())
        except Exception as e:
            rec.update(status="DOWNLOAD_FAILED", error=repr(e))
        records.append(rec)
        print("SOURCE", name, rec["status"], flush=True)
    target("source_evidence/pymannkendall_installed_source.txt").write_text(inspect.getsource(sys.modules["pymannkendall.pymannkendall"]), encoding="utf-8")
    json_out("source_evidence/source_access_records.json", records)


def code_validation():
    rng = np.random.default_rng(SEED)
    cases = []
    for k in range(100):
        cases.append(("continuous_" + str(k), rng.normal(size=(10, 12))))
    for k in range(30):
        cases.append(("ties_" + str(k), np.round(rng.normal(size=(10, 12)), 1)))
    cases.extend([
        ("constant", np.ones((10, 12))),
        ("identical_monotone_seasons", np.tile(np.arange(10)[:, None], (1, 12))),
        ("balanced_increasing_decreasing", np.outer(np.arange(10), np.tile([1., -1.], 6))),
    ])
    rows = []
    for name, x in cases:
        s, var, z, p = [a[0] for a in corr_skt_batch(x[None])]
        si, vi = independent_covariance(x)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            mk = pmk.correlated_seasonal_test(x, period=12, alpha=.05)
        expected_nocc = 2 * stats.norm.sf(abs(s / np.sqrt(var))) if var > 0 else np.nan
        checks = [abs(s - si) <= 1e-10, abs(var - vi) <= 1e-10,
                  abs(s - mk.s) <= 1e-10, abs(var - mk.var_s) <= 1e-10]
        if var > 0:
            checks.append(abs(expected_nocc - mk.p) <= 1e-10)
        perm = rng.permutation(12)
        a2 = corr_skt_batch(x[:, perm][None])
        reverse = corr_skt_batch(x[::-1][None])
        checks += [abs(a2[0][0] - s) <= 1e-10, abs(a2[1][0] - var) <= 1e-10,
                   abs(reverse[0][0] + s) <= 1e-10, abs(reverse[3][0] - p) <= 1e-10]
        rows.append({"case": name, "S": s, "varS": var, "literal_S": si, "literal_varS": vi,
                     "pmk_S": mk.s, "pmk_varS": mk.var_s, "p_primary_with_continuity": p,
                     "p_pmk_without_continuity": float(mk.p) if np.isfinite(mk.p) else None,
                     "all_checks_pass": bool(all(checks)), "warnings": "; ".join(str(w.message) for w in ws)})
    csv_out("05_method_validation/01_formula_crosscheck.csv", rows)
    passed = all(r["all_checks_pass"] for r in rows)
    print("FORMULA_VALIDATION", len(rows), "passed", passed, flush=True)
    return passed, len(rows)


def null_validation():
    rng = np.random.default_rng(SEED)
    rows = []
    scenes = [(0., 0., "CORE"), (.2, 0., "CORE"), (.4, 0., "CORE"), (.6, 0., "CORE"),
              (.9, 0., "STRESS"), (.9, .8, "STRESS")]
    for phi, theta, role in scenes:
        errors = rng.normal(size=(20000, 361))
        series = np.zeros((20000, 360))
        series[:, 0] = errors[:, 0] / np.sqrt(1 - phi ** 2)
        for t in range(1, 360):
            series[:, t] = phi * series[:, t-1] + errors[:, t] - theta * errors[:, t-1]
        x = series[:, -120:].reshape(-1, 10, 12)
        x = x + 2 * np.sin(2 * np.pi * np.arange(12) / 12)[None, None, :]
        s, var, z, p = corr_skt_batch(x)
        if (var <= 0).any() or not np.isfinite(p).all():
            raise RuntimeError("Invalid variance or p in prespecified calibration")
        for alpha, limit in [(.05, .075), (.01, .020), (.001, .005)]:
            k = int((p < alpha).sum())
            ci = stats.binomtest(k, len(p)).proportion_ci(.95, method="wilson")
            rows.append({"scene_role": role, "phi": phi, "theta": theta, "n_years": 10, "n_months": 120,
                         "replicates": len(p), "alpha": alpha, "rejections": k, "rejection_rate": k/len(p),
                         "wilson95_low": ci.low, "wilson95_high": ci.high, "diagnostic_gate_limit": limit,
                         "core_gate_pass": k/len(p) <= limit if role == "CORE" else None})
        print("NULL_VALIDATION", phi, theta, "alpha05", rows[-3]["rejection_rate"], flush=True)
    csv_out("05_method_validation/02_null_simulation_calibration.csv", rows)
    return all(r["core_gate_pass"] for r in rows if r["scene_role"] == "CORE")


def load_data():
    data = pd.read_csv(AOD).sort_values(["grid_id", "year", "month"]).reset_index(drop=True)
    assert len(data) == 161160 and data.grid_id.nunique() == 1343
    assert data.AOD.notna().all() and np.isfinite(data.AOD).all()
    assert not data.duplicated(["grid_id", "year", "month"]).any()
    ids = data.grid_id.unique()
    expected = np.array([(y, m) for y in range(2015, 2025) for m in range(1, 13)])
    assert np.array_equal(data[["year", "month"]].to_numpy(), np.tile(expected, (1343, 1)))
    return data, ids, data.AOD.to_numpy().reshape(1343, 10, 12)


def lag(x, k=1):
    if np.std(x[:-k]) == 0 or np.std(x[k:]) == 0:
        return np.nan
    return float(np.corrcoef(x[:-k], x[k:])[0, 1])


def threepw_verification(data):
    package = OLD / "official_mannkendall_python/mannkendall"
    try:
        spec = importlib.util.spec_from_file_location("mannkendall", package / "__init__.py", submodule_search_locations=[str(package)])
        mk = importlib.util.module_from_spec(spec)
        sys.modules["mannkendall"] = mk
        spec.loader.exec_module(mk)
    except Exception as e:
        rows = [{"grid_id": None, "status": "IMPORT_FAILED_NO_GRID_CALLS",
                 "exception": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}]
        json_out("05_method_validation/04_official_3pw_recheck.json", {"source": str(package), "attempts": rows,
                 "historical_validation": str(OLD / "07_method_log.md"),
                 "historical_ten_grid_IndexError_not_replaced": True})
        print("3PW_IMPORT_FAILED: no algorithm/source/package patch attempted", repr(e), flush=True)
        return rows
    rows = []
    for grid in SAMPLE_IDS:
        g = data[data.grid_id == grid]
        dates, vals = [], []
        for month in range(1, 13):
            month_data = g[g.month == month]
            dates.append(np.array([datetime(int(y), month, 15) for y in month_data.year]))
            vals.append(month_data.AOD.to_numpy().copy())
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                result = mk.mk_temp_aggr(dates, vals, resolution=.001, pw_method="3pw", alpha_mk=95,
                                         alpha_cl=90, alpha_xhomo=90, alpha_ak=95)
            rows.append({"grid_id": grid, "status": "COMPLETED_NOT_USED_WITHOUT_NEW_PROTOCOL", "result_repr": repr(result)})
        except Exception as e:
            rows.append({"grid_id": grid, "status": "FAILED_UNMODIFIED_OFFICIAL_CODE",
                         "exception": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()})
    json_out("05_method_validation/04_official_3pw_recheck.json", {"version": mk.__version__, "source": str(package), "attempts": rows})
    print("3PW_RECHECK", sum(r["status"].startswith("FAILED") for r in rows), "/", len(rows), "failed", flush=True)
    return rows


def extract_manuscript():
    specs = [
        ROOT / "74_EMA_R24_FINAL_LANGUAGE_AND_FORMAT_QA_2026-10-14/01_Manuscript/EMA_Manuscript_R24_FINAL_LANGUAGE_AND_FORMAT_QA_CANDIDATE.docx",
        ROOT / "72_EMA_R24_FINAL_SUBMISSION_CANDIDATE_2026-10-13/02_Supplementary/EMA_Online_Resource_1_R24_FINAL.docx",
    ]
    terms = ["98.66", "99.85", "autocorr", "Mann", "seasonal", "3PW", "0.558"]
    rows = []
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    for p in specs:
        with zipfile.ZipFile(p) as doc:
            xml = ET.fromstring(doc.read("word/document.xml"))
        for index, paragraph in enumerate(xml.findall(".//w:p", ns), 1):
            text = "".join(n.text or "" for n in paragraph.findall(".//w:t", ns))
            if any(t.lower() in text.lower() for t in terms):
                rows.append({"file": str(p), "ooxml_paragraph_index_including_tables": index, "text": text})
    csv_out("source_evidence/manuscript_supplement_trend_passages.csv", rows)


def prepare():
    print("Snapshot protected sources", flush=True)
    csv_out("01_protected_sources_before.csv", list(collect_protected()))
    versions = {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "pymannkendall", "statsmodels"]}
    json_out("05_method_validation/00_environment.json", {"python": sys.version, "executable": sys.executable,
             "platform": platform.platform(), "versions": versions, "seed": SEED})
    fetch_sources()
    validate()


def validate():
    extract_manuscript()
    formula_pass, n = code_validation()
    calibration_pass = null_validation()
    json_out("05_method_validation/03_gate_decision.json", {"formula_pass": formula_pass, "formula_cases": n,
             "calibration_core_pass": calibration_pass, "gate": "PASS" if formula_pass and calibration_pass else "STOP",
             "scope": "Conditional correlated-seasonal sensitivity only; not arbitrary temporal-dependence correction"})


def analyze():
    gate = json.loads((OUT / "05_method_validation/03_gate_decision.json").read_text(encoding="utf-8"))
    if gate["gate"] != "PASS":
        raise RuntimeError("Method validation STOP. No grid sensitivity permitted.")
    data, ids, x = load_data()
    prior_original = pd.read_csv(OLD / "01_original_reproduction.csv").set_index("grid_id").loc[ids]
    prior_seasonal = pd.read_csv(OLD / "02_seasonal_mk_grid_results.csv").set_index("grid_id").loc[ids]
    locked = pd.read_csv(LOCKED).set_index("grid_id").loc[ids]
    # Reuse the locked pandas aggregation path to preserve floating-point ties.
    annual = data.groupby(["grid_id", "year"], sort=True).AOD.mean().to_numpy().reshape(1343, 10)
    orig_rows, diag_rows = [], []
    for idx, grid in enumerate(ids):
        a, month = annual[idx], x[idx].reshape(120)
        sen = stats.theilslopes(a, np.arange(2015, 2025), alpha=.95)
        mk = pmk.original_test(a, alpha=.05)
        sk = pmk.seasonal_test(month, period=12, alpha=.05)
        for field, value in [("slope", sen.slope), ("intercept", sen.intercept), ("lower_slope", sen.low_slope),
                             ("upper_slope", sen.high_slope), ("mk_p", mk.p), ("mk_z", mk.z), ("mk_tau", mk.Tau)]:
            assert np.isclose(locked.loc[grid, field], value, rtol=0, atol=1e-12), (grid, field)
        for field, value in [("seasonal_slope", sk.slope), ("seasonal_p", sk.p), ("seasonal_z", sk.z),
                             ("seasonal_test_statistic_s", sk.s)]:
            assert np.isclose(prior_seasonal.loc[grid, field], value, rtol=0, atol=1e-12), (grid, field)
        orig_rows.append({"grid_id": int(grid), "n_annual": 10, "n_monthly": 120,
                          "original_slope": sen.slope, "original_low_slope": sen.low_slope, "original_high_slope": sen.high_slope,
                          "original_S": mk.s, "original_varS": mk.var_s, "original_p": mk.p,
                          "seasonal_S": sk.s, "seasonal_varS": sk.var_s, "seasonal_p": sk.p, "seasonal_slope": sk.slope})
        annual_res = a - sen.slope * np.arange(10)
        detrended = (month - sen.slope * np.arange(120)/12).reshape(10, 12)
        residual = (detrended - detrended.mean(axis=0)).reshape(120)
        within_month_r = []
        for m in range(12):
            slope_m = stats.theilslopes(x[idx, :, m], np.arange(10)).slope
            within_month_r.append(lag(x[idx, :, m] - slope_m*np.arange(10)))
        diag_rows.append({"grid_id": int(grid), "annual_raw_r1": lag(a), "annual_sen_detrended_r1": lag(annual_res),
                          "monthly_raw_r1": lag(month), "monthly_detrended_deseasonal_r1": lag(residual),
                          "monthly_detrended_deseasonal_r12": lag(residual, 12),
                          "monthly_detrended_deseasonal_r24": lag(residual, 24),
                          "calendar_month_detrended_annual_r1_median": float(np.median(within_month_r)),
                          "calendar_month_detrended_annual_r1_max": float(np.max(within_month_r)),
                          "calendar_month_detrended_annual_r1_min": float(np.min(within_month_r))})
        if (idx+1) % 200 == 0:
            print("REPRODUCTION_DIAGNOSTIC", idx+1, "/", len(ids), flush=True)
    original = pd.DataFrame(orig_rows)
    original["original_p_fdr"] = bh(original.original_p)
    original["seasonal_p_fdr"] = bh(original.seasonal_p)
    assert np.allclose(original.original_p_fdr, prior_original.original_p_fdr, atol=1e-12, rtol=0)
    assert np.allclose(original.seasonal_p_fdr, prior_seasonal.seasonal_p_fdr, atol=1e-12, rtol=0)
    csv_out("02_original_and_seasonal_reproduction.csv", original)
    diagnostic = pd.DataFrame(diag_rows)
    csv_out("03_autocorrelation_diagnostics.csv", diagnostic)
    s, var, z, p = corr_skt_batch(x)
    assert np.isfinite(p).all() and (var > 0).all()
    assert np.allclose(s, original.seasonal_S, rtol=0, atol=1e-12)
    q = bh(p)
    sensitivity = pd.DataFrame({"grid_id": ids, "n_years": 10, "months_per_year": 12, "correlated_seasonal_S": s,
                "original_seasonal_varS": original.seasonal_varS, "correlated_seasonal_varS": var,
                "variance_ratio": var / original.seasonal_varS, "correlated_seasonal_z_CC": z,
                "correlated_seasonal_p_CC": p, "correlated_seasonal_p_fdr": q,
                "correlated_seasonal_slope": original.seasonal_slope,
                "significant_decline_fdr": (q<=ALPHA)&(original.seasonal_slope<0),
                "original_p_fdr": original.original_p_fdr, "original_seasonal_p_fdr": original.seasonal_p_fdr,
                "slope_not_reestimated_or_prewhitened": True})
    check_rows = []
    for idx in range(len(ids)):
        result = pmk.correlated_seasonal_test(x[idx], period=12, alpha=.05)
        no_cc = 2*stats.norm.sf(abs(s[idx]/np.sqrt(var[idx])))
        ok = bool(abs(result.s-s[idx])<1e-10 and abs(result.var_s-var[idx])<1e-10
                  and abs(result.slope-original.seasonal_slope.iloc[idx])<1e-10 and abs(result.p-no_cc)<1e-10)
        assert ok, ids[idx]
        check_rows.append({"grid_id": int(ids[idx]), "pmk_S": result.s, "pmk_varS": result.var_s,
                           "pmk_slope": result.slope, "pmk_p_without_CC": result.p,
                           "primary_p_with_CC": p[idx], "check_pass": ok})
    csv_out("05_method_validation/05_all_grid_software_crosscheck.csv", check_rows)
    csv_out("04_correlated_seasonal_sensitivity.csv", sensitivity)
    summarize()


def summarize():
    data, ids, _ = load_data()
    original = pd.read_csv(OUT / "02_original_and_seasonal_reproduction.csv")
    diagnostic = pd.read_csv(OUT / "03_autocorrelation_diagnostics.csv")
    sensitivity = pd.read_csv(OUT / "04_correlated_seasonal_sensitivity.csv")
    assert np.array_equal(original.grid_id, ids) and np.array_equal(sensitivity.grid_id, ids)
    q = sensitivity.correlated_seasonal_p_fdr.to_numpy()
    var = sensitivity.correlated_seasonal_varS.to_numpy()
    assert np.allclose(q, bh(sensitivity.correlated_seasonal_p_CC), atol=1e-12, rtol=0)
    failure = threepw_verification(data)
    def method_stats(slopes, qs):
        mask = (np.asarray(qs)<=ALPHA)&(np.asarray(slopes)<0)
        return {"median_slope": float(np.median(slopes)), "negative_slope_count": int((np.asarray(slopes)<0).sum()),
                "fdr_significant_decline_count": int(mask.sum()), "fdr_significant_decline_percent": float(mask.mean()*100)}
    a = method_stats(original.original_slope, original.original_p_fdr)
    b = method_stats(original.seasonal_slope, original.seasonal_p_fdr)
    c = method_stats(original.seasonal_slope, q)
    median_diag = {col: {"median": float(diagnostic[col].median()), "p05": float(diagnostic[col].quantile(.05)),
                        "p95": float(diagnostic[col].quantile(.95)), "min": float(diagnostic[col].min()),
                        "max": float(diagnostic[col].max())} for col in diagnostic if col != "grid_id"}
    s1 = (original.original_p_fdr<=ALPHA).to_numpy()
    s2 = (original.seasonal_p_fdr<=ALPHA).to_numpy()
    s3 = q<=ALPHA
    summary = {"created": datetime.now().isoformat(), "input": str(AOD), "input_sha256": sha(AOD),
               "integrity": {"grid_count": 1343, "records": 161160, "annual_n": 10, "monthly_n": 120,
                             "period": "2015-01 through 2024-12", "complete_grid_count": 1343},
               "reproduction": {"all_1343_original_values_reproduced": True, "all_1343_seasonal_values_reproduced": True},
               "annual_standard_MK_BH": a, "ordinary_seasonal_MK_BH": b,
               "correlation_adjusted_seasonal_MK_CC_BH": c,
               "comparison": {"new_minus_ordinary_seasonal_pp": c["fdr_significant_decline_percent"]-b["fdr_significant_decline_percent"],
                              "new_minus_annual_pp_not_same_test": c["fdr_significant_decline_percent"]-a["fdr_significant_decline_percent"],
                              "annual_vs_seasonal_slope_sign_agreement_percent": float((np.sign(original.original_slope)==np.sign(original.seasonal_slope)).mean()*100),
                              "annual_vs_correlated_seasonal_class_agreement_percent": float((s1==s3).mean()*100),
                              "seasonal_vs_correlated_seasonal_class_agreement_percent": float((s2==s3).mean()*100),
                              "ordinary_seasonal_significant_but_new_not_count": int((s2&~s3).sum()),
                              "annual_significant_but_new_not_count": int((s1&~s3).sum()),
                              "new_significant_but_ordinary_seasonal_not_count": int((s3&~s2).sum()),
                              "median_variance_ratio": float(np.median(var/original.seasonal_varS)),
                              "median_abs_relative_slope_difference_annual_seasonal_percent": float(np.median(abs((original.seasonal_slope-original.original_slope)/original.original_slope))*100)},
               "autocorrelation_diagnostics": median_diag,
               "threepw": {"current_grid_calls": sum(r["grid_id"] is not None for r in failure),
                            "current_import_failed": any(r["status"].startswith("IMPORT_FAILED") for r in failure),
                            "current_grid_call_failures": sum(r["status"].startswith("FAILED") for r in failure),
                            "historical_unmodified_official_grid_calls": 10, "historical_IndexError_failures": 10,
                            "corrected_p_values_available": False, "full_grid_3pw_not_run": True},
               "annual_autocorrelation_corrected_significant_percent": None,
               "annual_correction_status": "NOT_VERIFIED: n=10; no validated unconditional correction produced",
               "limitations": ["Correlation-adjusted seasonal result is conditional, not full long-memory correction",
                               "Year-to-year persistence is not eliminated by the seasonal covariance formula",
                               "Slope is original seasonal Sen, not a new prewhitened slope",
                               "Simulation assesses specified null scenarios, not arbitrary data-generating processes",
                               "BH assumes suitable dependence conditions; no spatial-field independence claimed"]}
    json_out("06_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ["annual_standard_MK_BH", "ordinary_seasonal_MK_BH", "correlation_adjusted_seasonal_MK_CC_BH", "comparison"]}, indent=2), flush=True)


def finish():
    before = pd.read_csv(OUT / "01_protected_sources_before.csv")
    changed, missing = [], []
    for row in before.itertuples(index=False):
        p = Path(row.path)
        if not p.exists():
            missing.append(str(p))
        elif p.stat().st_size != row.size_bytes or sha(p) != row.sha256:
            changed.append(str(p))
    json_out("09_scope_integrity_check.json", {"protected_files_checked": len(before), "modified_files": changed,
              "missing_files": missing, "protected_inputs_unchanged": not changed and not missing,
              "models_trained": 0, "manuscripts_modified": 0, "figures_modified": 0,
              "output_directory": str(OUT), "onedrive_used": False})
    assert not changed and not missing
    rows = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            rows.append({"relative_path": str(p.relative_to(OUT)), "full_path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)})
    csv_out("10_SHA256_manifest.csv", rows)
    print("FINAL_INTEGRITY", len(before), "unchanged; manifest", len(rows), "files", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "validate", "analyze", "summarize", "finish"])
    arg = parser.parse_args()
    if arg.stage == "finish":
        finish()
    else:
        with target("logs/" + arg.stage + "_stdout.txt").open("x", encoding="utf-8") as log:
            class Tee:
                def write(self, s):
                    sys.__stdout__.write(s)
                    log.write(s)
                    log.flush()
                def flush(self):
                    log.flush()
                    sys.__stdout__.flush()
            sys.stdout = Tee()
            try:
                {"prepare": prepare, "validate": validate, "analyze": analyze, "summarize": summarize}[arg.stage]()
            except Exception:
                traceback.print_exc(file=sys.stdout)
                raise
            finally:
                sys.stdout = sys.__stdout__
