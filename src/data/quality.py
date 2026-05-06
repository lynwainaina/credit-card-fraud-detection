import argparse
import json
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from src.data.loader import load_csv


# Load variables from the .env file at the project root into os.environ
load_dotenv(Path(__file__).parents[2] / ".env")

DATA_DIR = Path(__file__).parents[2] / "data"

# ---------------------------------------------------------------------------
# Schema: expected columns and their required dtypes.
# Extend this dict in .env when the dataset changes.
# ---------------------------------------------------------------------------
REQUIRED_SCHEMA = json.loads(os.getenv("REQUIRED_SCHEMA"))

# Target column for classification checks
TARGET_COL = os.getenv("TARGET_COL")

# Numeric bounds: {column: (min_allowed, max_allowed)}
# None means "no bound on that side" — stored as null in .env JSON
_raw_bounds = json.loads(os.getenv("NUMERIC_BOUNDS"))
NUMERIC_BOUNDS = {col: tuple(v for v in bounds) for col, bounds in _raw_bounds.items()}

# Imbalance threshold: warn if the minority class is below this fraction
IMBALANCE_WARN_THRESHOLD = float(os.getenv("IMBALANCE_WARN_THRESHOLD"))   # 5 %
# Null rate thresholds
NULL_CRITICAL_THRESHOLD = float(os.getenv("NULL_CRITICAL_THRESHOLD"))     # 50 % → failure
NULL_WARN_THRESHOLD     = float(os.getenv("NULL_WARN_THRESHOLD"))         # 20 % → warning


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def _check_schema(df: pd.DataFrame, failures: list, warnings: list) -> None:
    """Check 1 — required columns exist and have the correct dtype.
    
    Args:
        df: pandas DataFrame containing the data.
        failures: list of failure messages.
        warnings: list of warning messages.
    Returns:
        None
    """
    for col, expected_dtype in REQUIRED_SCHEMA.items():
        if col not in df.columns:
            failures.append(f"Schema: missing required column '{col}'")
        elif str(df[col].dtype) != expected_dtype:
            failures.append(
                f"Schema: column '{col}' has dtype '{df[col].dtype}', expected '{expected_dtype}'"
            )


def _check_row_count(df: pd.DataFrame, failures: list, warnings: list) -> None:
    """Check 2 — dataset has enough rows to be meaningful.
    
    Args:
        df: pandas DataFrame containing the data.
        failures: list of failure messages.
        warnings: list of warning messages.
    Returns:
        None
    """
    n = len(df)
    if n < 100:
        failures.append(f"Row count: only {n:,} rows — minimum required is 100")
    elif n < 1000:
        warnings.append(f"Row count: {n:,} rows — consider collecting more data (recommended ≥ 1,000)")


def _check_null_rates(df: pd.DataFrame, failures: list, warnings: list) -> dict:
    """Check 3 — no column exceeds the critical null-rate threshold.
    
    Args:
        df: pandas DataFrame containing the data.
        failures: list of failure messages.
        warnings: list of warning messages.
    Returns:
        null_counts: dictionary of column names and their null counts.
    """
    null_counts = {}
    for col in df.columns:
        null_rate = df[col].isnull().mean()
        null_counts[col] = int(df[col].isnull().sum())
        if null_rate > NULL_CRITICAL_THRESHOLD:
            failures.append(
                f"Null rate: '{col}' is {null_rate:.1%} null — exceeds critical threshold of {NULL_CRITICAL_THRESHOLD:.0%}"
            )
        elif null_rate > NULL_WARN_THRESHOLD:
            warnings.append(
                f"Null rate: '{col}' is {null_rate:.1%} null — exceeds warning threshold of {NULL_WARN_THRESHOLD:.0%}"
            )
    return null_counts


def _check_value_ranges(df: pd.DataFrame, failures: list, warnings: list) -> None:
    """Check 4 — numeric columns stay within sensible bounds specified in NUMERIC_BOUNDS
    
    Args:
        df: pandas DataFrame containing the data.
        failures: list of failure messages.
        warnings: list of warning messages.
    Returns:
        None
    """
    for col, (low, high) in NUMERIC_BOUNDS.items():
        if col not in df.columns:
            continue
        if low is not None and (df[col] < low).any():
            n_violations = int((df[col] < low).sum())
            failures.append(
                f"Value range: '{col}' has {n_violations:,} value(s) below minimum ({low})"
            )
        if high is not None and (df[col] > high).any():
            n_violations = int((df[col] > high).sum())
            failures.append(
                f"Value range: '{col}' has {n_violations:,} value(s) above maximum ({high})"
            )


def _check_target_distribution(df: pd.DataFrame, failures: list, warnings: list) -> dict:
    """Check 5 — classification target has ≥ 2 classes, each with ≥ 5% share.
    
    Args:
        df: pandas DataFrame containing the data.
        failures: list of failure messages.
        warnings: list of warning messages.
    Returns:
        distribution: dictionary of target class names and their proportions.
        
    Example:  Target distribution; {'0': 0.9983, '1': 0.0017}
        a minority class below 5% is widely considered problematic for most classifiers 
        for fraud detection you'd typically address this with:
        - class_weight='balanced' in sklearn models
        - SMOTE oversampling on the training set only
        - Optimising for F1/AUC-PR rather than accuracy
    """
    distribution = {}
    if TARGET_COL not in df.columns:
        # Schema check will already flag this; skip silently here
        return distribution
    # count the proportion of class 1 and class 0
    value_counts = df[TARGET_COL].value_counts(normalize=True).sort_index()
    # Target distribution of the two classes
    distribution = {str(target): round(float(proportion), 4) for target, proportion in value_counts.items()}
    # check if the target column has at least 2 classes, otherwise the model cant learn from one class
    if len(value_counts) < 2:
        failures.append(
            f"Target distribution: '{TARGET_COL}' has only {len(value_counts)} class — need at least 2"
        )
        return distribution
    #  find the smallest proportion using .min() and find the label of that class using .idxmin() 
    minority_share = value_counts.min()
    minority_class = value_counts.idxmin()
    if minority_share < IMBALANCE_WARN_THRESHOLD:
        warnings.append(
            f"Target distribution: class '{minority_class}' represents only "
            f"{minority_share:.2%} of data — dataset is heavily imbalanced"
        )

    return distribution


# ---------------------------------------------------------------------------
# Main gate function
# ---------------------------------------------------------------------------

def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Run 5 data quality checks on a DataFrame.

    Returns a dict with keys:
        success    — True if no critical failures
        failures   — list of critical error messages
        warnings   — list of non-critical concern messages
        statistics — counts and distributions collected during checks
    """
    failures = []
    warnings = []

    _check_schema(df, failures, warnings)
    _check_row_count(df, failures, warnings)
    null_counts = _check_null_rates(df, failures, warnings)
    _check_value_ranges(df, failures, warnings)
    target_distribution = _check_target_distribution(df, failures, warnings)

    statistics = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_nulls_by_column": null_counts,
        "target_distribution": target_distribution,
    }

    return {
        "success": len(failures) == 0,
        "failures": failures,
        "warnings": warnings,
        "statistics": statistics,
    }


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_quality_report(result: dict) -> None:
    status = "PASSED" if result["success"] else "FAILED"
    print(f"\n{'='*50}")
    print(f"  Data Quality Gate: {status}")
    print(f"{'='*50}")

    if result["failures"]:
        print(f"\nCritical failures ({len(result['failures'])}):")
        for msg in result["failures"]:
            print(f"  [FAIL] {msg}")

    if result["warnings"]:
        print(f"\nWarnings ({len(result['warnings'])}):")
        for msg in result["warnings"]:
            print(f"  [WARN] {msg}")

    if not result["failures"] and not result["warnings"]:
        print("\n  All checks passed with no warnings.")

    stats = result["statistics"]
    print("\nStatistics:")
    print(f"  Rows        : {stats['total_rows']:,}")
    print(f"  Columns     : {stats['total_columns']}")

    nulls = {col: n for col, n in stats["total_nulls_by_column"].items() if n > 0}
    print(f"  Null counts : {nulls if nulls else 'none'}")

    if stats["target_distribution"]:
        print(f"  Target dist : {stats['target_distribution']}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data quality checks on a CSV dataset")
    parser.add_argument("filename", nargs="?", default="creditcard.csv",
                        help="CSV filename inside data/. Defaults to creditcard.csv.")
    args = parser.parse_args()

    df = load_csv(args.filename)
    result = check_data_quality(df)
    print_quality_report(result)
