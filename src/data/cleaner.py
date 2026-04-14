import argparse
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from loader import load_csv
from quality import check_data_quality, print_quality_report


# Load variables from the .env file at the project root into os.environ
load_dotenv(Path(__file__).parents[2] / ".env")

DATA_DIR = Path(__file__).parents[2] / "data"
CLEANED_CSV = DATA_DIR / "cleaned.csv"

# Target column (same source as quality.py)
TARGET_COL = os.getenv("TARGET_COL")


def clean_data(df: pd.DataFrame,target_col: str = None,time_series: bool = False,) -> tuple:
    """
    Clean a DataFrame by handling nulls, removing duplicates, and coercing dtypes.
    Saves cleaned data to data/cleaned.csv and re-runs the quality gate.

    Args:
        df: Raw input DataFrame.
        target_col: Name of the target column; defaults to TARGET_COL from .env.
        time_series: If True, forward-fill non-target columns instead of dropping rows.

    Returns:
        cleaned_df: Cleaned DataFrame.
        quality_result: Result dict from check_data_quality.
    """
    if target_col is None:
        target_col = TARGET_COL

    df = df.copy()

    # ------------------------------------------------------------------
    # Step 1 — Drop columns with > 50% nulls
    # ------------------------------------------------------------------
    null_rate = df.isnull().mean()
    cols_to_drop = null_rate[null_rate > 0.5].index.tolist()
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} column(s) with >50% nulls: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

    # ------------------------------------------------------------------
    # Step 2 — Drop rows where target is null
    # ------------------------------------------------------------------
    if target_col and target_col in df.columns:
        before = len(df)
        df = df.dropna(subset=[target_col])
        dropped = before - len(df)
        if dropped:
            print(f"Dropped {dropped:,} row(s) where target '{target_col}' was null")

    # ------------------------------------------------------------------
    # Step 3 — Handle remaining nulls in non-target columns
    # ------------------------------------------------------------------
    other_cols = [c for c in df.columns if c != target_col]
    if time_series:
        df[other_cols] = df[other_cols].ffill()
        # Any leading nulls that couldn't be forward-filled must be dropped
        before = len(df)
        df = df.dropna(subset=other_cols)
        dropped = before - len(df)
        if dropped:
            print(f"Dropped {dropped:,} row(s) with leading nulls after forward-fill")
    else:
        before = len(df)
        df = df.dropna(subset=other_cols)
        dropped = before - len(df)
        if dropped:
            print(f"Dropped {dropped:,} row(s) with nulls in non-target columns")

    # ------------------------------------------------------------------
    # Step 4 — Remove exact duplicate rows (keep first)
    # ------------------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(keep="first")
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped:,} duplicate row(s)")

    # ------------------------------------------------------------------
    # Step 5 — Coerce dtypes: numeric → float/int, categorical → str
    # ------------------------------------------------------------------
    for col in df.columns:
        if col == target_col:
            continue
        if pd.api.types.is_object_dtype(df[col]):
            coerced = pd.to_numeric(df[col], errors="coerce")
            # Accept numeric cast only if it introduced no new nulls beyond what already existed
            original_null_count = df[col].isna().sum()
            if coerced.isna().sum() == original_null_count:
                df[col] = coerced
            else:
                df[col] = df[col].astype(str)
        # Columns already typed float/int need no action

    # ------------------------------------------------------------------
    # Step 6 — Save cleaned CSV
    # ------------------------------------------------------------------
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_CSV, index=False)
    print(f"Saved cleaned data to {CLEANED_CSV}")

    # ------------------------------------------------------------------
    # Step 7 — Re-run quality gate on cleaned data
    # ------------------------------------------------------------------
    quality_result = check_data_quality(df)

    return df, quality_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean a raw CSV dataset from data/")
    parser.add_argument("filename",nargs="?", default="creditcard.csv",
                        help="CSV filename inside data/ (e.g. creditcard.csv). Defaults to creditcard.csv.",
    )
    parser.add_argument("--time-series",action="store_true",
                        help="Forward-fill non-target columns instead of dropping rows with nulls.",
    )
    args = parser.parse_args()

    raw_df = load_csv(args.filename)
    print(f"Loaded '{args.filename}': {len(raw_df):,} rows x {raw_df.shape[1]} columns")
    cleaned_df, quality_result = clean_data(raw_df, time_series=args.time_series)
    print(f"\nBefore: {len(raw_df):,} rows")
    print(f"After : {len(cleaned_df):,} rows")
    print_quality_report(quality_result)
