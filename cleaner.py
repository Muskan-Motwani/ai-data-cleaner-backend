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

Memory note: this avoids unnecessary intermediate copies — important on
free-tier hosting (512MB RAM), where repeated full-dataframe copies on a
10k+ row dataset can exceed the limit and crash the process (OOM / exit 139).
"""

import gc
import pandas as pd


def _is_text_column(series: pd.Series) -> bool:
    """True for string/object columns, across both old (object) and new
    (pandas 2.1+ StringDtype) pandas string handling."""
    return series.dtype == object or pd.api.types.is_string_dtype(series)


def apply_cleaning_steps(df: pd.DataFrame, steps: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    # Work on the dataframe in place where possible instead of repeatedly
    # copying it — one copy up front is enough.
    df = df.copy()
    log = []

    for step in steps:
        action = step.get("action")
        column = step.get("column")

        try:
            if action == "drop_duplicates":
                before = len(df)
                df.drop_duplicates(inplace=True)
                df.reset_index(drop=True, inplace=True)
                log.append(f"Dropped {before - len(df)} duplicate rows")

            elif action == "drop_column":
                if column in df.columns:
                    df.drop(columns=[column], inplace=True)
                    log.append(f"Dropped column '{column}'")

            elif action == "fill_missing":
                if column not in df.columns:
                    continue
                strategy = step.get("strategy", "mean")
                if strategy == "mean" and pd.api.types.is_numeric_dtype(df[column]):
                    df[column].fillna(df[column].mean(), inplace=True)
                elif strategy == "median" and pd.api.types.is_numeric_dtype(df[column]):
                    df[column].fillna(df[column].median(), inplace=True)
                elif strategy == "mode":
                    mode_val = df[column].mode()
                    if len(mode_val) > 0:
                        df[column].fillna(mode_val[0], inplace=True)
                elif strategy == "zero":
                    df[column].fillna(0, inplace=True)
                elif strategy == "unknown":
                    df[column].fillna("Unknown", inplace=True)
                elif strategy == "drop_rows":
                    df.dropna(subset=[column], inplace=True)
                    df.reset_index(drop=True, inplace=True)
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
                    keep_mask = df[column].between(lower, upper) | df[column].isna()
                    df = df.loc[keep_mask].reset_index(drop=True)
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

        # Free any temporary objects from this step before moving to the next
        # one — matters on constrained-memory free-tier hosting.
        gc.collect()

    return df, log
