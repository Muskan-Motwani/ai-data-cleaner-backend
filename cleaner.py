"""
Applies cleaning steps (either AI-suggested or user-picked) to a dataframe.
Each step is a small dict like:
  {"action": "fill_missing", "column": "age", "strategy": "median"}
  {"action": "drop_column", "column": "unnamed_0"}
  {"action": "drop_duplicates"}
  {"action": "convert_dtype", "column": "price", "target_type": "numeric"}
  {"action": "remove_outliers", "column": "salary"}

Unknown/unsupported actions are skipped (logged, not crashed on) so one bad
step from the AI doesn't take down the whole cleaning run.
"""

import pandas as pd


def _is_text_column(series: pd.Series) -> bool:
    """True for string/object columns, across both old (object) and new
    (pandas 2.1+ StringDtype) pandas string handling."""
    return series.dtype == object or pd.api.types.is_string_dtype(series)


def apply_cleaning_steps(df: pd.DataFrame, steps: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    log = []

    for step in steps:
        action = step.get("action")
        column = step.get("column")

        try:
            if action == "drop_duplicates":
                before = len(df)
                df = df.drop_duplicates()
                log.append(f"Dropped {before - len(df)} duplicate rows")

            elif action == "drop_column":
                if column in df.columns:
                    df = df.drop(columns=[column])
                    log.append(f"Dropped column '{column}'")

            elif action == "fill_missing":
                if column not in df.columns:
                    continue
                strategy = step.get("strategy", "mean")
                if strategy == "mean" and pd.api.types.is_numeric_dtype(df[column]):
                    df[column] = df[column].fillna(df[column].mean())
                elif strategy == "median" and pd.api.types.is_numeric_dtype(df[column]):
                    df[column] = df[column].fillna(df[column].median())
                elif strategy == "mode":
                    mode_val = df[column].mode()
                    if len(mode_val) > 0:
                        df[column] = df[column].fillna(mode_val[0])
                elif strategy == "zero":
                    df[column] = df[column].fillna(0)
                elif strategy == "unknown":
                    df[column] = df[column].fillna("Unknown")
                elif strategy == "drop_rows":
                    df = df.dropna(subset=[column])
                log.append(f"Filled missing values in '{column}' using strategy: {strategy}")

            elif action == "convert_dtype":
                if column not in df.columns:
                    continue
                target = step.get("target_type")
                if target == "numeric":
                    df[column] = pd.to_numeric(
                        df[column].astype(str).str.replace(",", "", regex=False), errors="coerce"
                    )
                elif target == "datetime":
                    df[column] = pd.to_datetime(df[column], errors="coerce")
                elif target == "category":
                    df[column] = df[column].astype("category")
                log.append(f"Converted '{column}' to {target}")

            elif action == "remove_outliers":
                if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
                    continue
                q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    before = len(df)
                    df = df[(df[column] >= lower) & (df[column] <= upper) | df[column].isna()]
                    log.append(f"Removed {before - len(df)} outlier rows from '{column}'")

            elif action == "strip_whitespace":
                if column in df.columns and _is_text_column(df[column]):
                    df[column] = df[column].astype(str).str.strip()
                    log.append(f"Stripped whitespace in '{column}'")

            elif action == "standardize_case":
                if column in df.columns and _is_text_column(df[column]):
                    case = step.get("case", "lower")
                    if case == "lower":
                        df[column] = df[column].astype(str).str.lower()
                    elif case == "upper":
                        df[column] = df[column].astype(str).str.upper()
                    elif case == "title":
                        df[column] = df[column].astype(str).str.title()
                    log.append(f"Standardized case in '{column}' to {case}")

            elif action == "create_feature":
                # Feature engineering: e.g. extract year/month from a date column,
                # or a ratio between two numeric columns.
                new_col = step.get("new_column_name")
                method = step.get("method")
                source = step.get("source_column")
                if method == "extract_year" and source in df.columns:
                    df[new_col] = pd.to_datetime(df[source], errors="coerce").dt.year
                elif method == "extract_month" and source in df.columns:
                    df[new_col] = pd.to_datetime(df[source], errors="coerce").dt.month
                elif method == "ratio":
                    col_a, col_b = step.get("column_a"), step.get("column_b")
                    if col_a in df.columns and col_b in df.columns:
                        df[new_col] = df[col_a] / df[col_b].replace(0, pd.NA)
                log.append(f"Created feature '{new_col}' using {method}")

            else:
                log.append(f"Skipped unsupported action: {action}")

        except Exception as e:
            log.append(f"Failed step '{action}' on '{column}': {e}")

    return df, log
