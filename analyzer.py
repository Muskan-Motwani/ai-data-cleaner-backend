"""
Profiles a dataset AND detects concrete data-quality issues — missing values,
duplicate rows, outliers, inconsistent types, high-cardinality columns — so
the LLM gets a clear, structured picture of what's actually wrong, not just
raw stats.
"""

import math
import pandas as pd


def profile_and_detect_issues(df: pd.DataFrame) -> dict:
    issues = []
    columns_info = []

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        issues.append({
            "type": "duplicate_rows",
            "detail": f"{duplicate_count} duplicate rows found",
            "severity": "medium" if duplicate_count < len(df) * 0.05 else "high",
        })

    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        null_pct = _safe_round(series.isnull().mean() * 100) or 0.0

        col_info = {
            "name": col,
            "dtype": dtype,
            "null_percentage": null_pct,
            "unique_count": int(series.nunique()),
        }

        if null_pct > 0:
            severity = "high" if null_pct > 30 else ("medium" if null_pct > 5 else "low")
            issues.append({
                "type": "missing_values",
                "column": col,
                "detail": f"{null_pct}% missing",
                "severity": severity,
            })

        if pd.api.types.is_numeric_dtype(series):
            col_info["type_category"] = "numeric"
            col_info["min"] = _safe_round(series.min())
            col_info["max"] = _safe_round(series.max())
            col_info["mean"] = _safe_round(series.mean())

            # simple IQR-based outlier detection
            non_null = series.dropna()
            if len(non_null) > 10:
                q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    outlier_count = int(((non_null < lower) | (non_null > upper)).sum())
                    if outlier_count > 0:
                        outlier_pct = _safe_round(outlier_count / len(non_null) * 100) or 0.0
                        issues.append({
                            "type": "outliers",
                            "column": col,
                            "detail": f"{outlier_count} potential outliers ({outlier_pct}%) outside [{_safe_round(lower)}, {_safe_round(upper)}]",
                            "severity": "medium" if outlier_pct < 5 else "high",
                        })

        elif pd.api.types.is_datetime64_any_dtype(series):
            col_info["type_category"] = "datetime"
        else:
            col_info["type_category"] = "categorical"
            sample = series.dropna().unique().tolist()[:5]
            col_info["sample_values"] = [_json_safe(v) for v in sample]

            if series.nunique() > 50:
                issues.append({
                    "type": "high_cardinality",
                    "column": col,
                    "detail": f"{series.nunique()} unique values — may need grouping or encoding strategy",
                    "severity": "low",
                })

            # detect a numeric-looking column stored as text (common messy-data issue)
            non_null = series.dropna().astype(str)
            if len(non_null) > 0:
                numeric_like = non_null.str.replace(",", "", regex=False).str.match(r"^-?\d+\.?\d*$")
                if numeric_like.mean() > 0.9:
                    issues.append({
                        "type": "wrong_dtype",
                        "column": col,
                        "detail": "Looks numeric but is stored as text",
                        "severity": "medium",
                    })

        columns_info.append(col_info)

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "duplicate_rows": duplicate_count,
        "columns": columns_info,
        "issues": issues,
    }


def _safe_round(value, digits: int = 2):
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, digits)
    except (TypeError, ValueError):
        return None


def _json_safe(value):
    if value is None:
        return None
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
    except TypeError:
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
    return value
