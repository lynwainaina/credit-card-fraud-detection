"""
Feature engineering for credit card fraud detection.

Transforms the cleaned DataFrame (Time, V1-V28, Amount, Class) into a
richer feature set by adding domain-specific, statistical, and interaction
features that help the model separate fraudulent from legitimate transactions.
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import argparse
from src.data.loader import load_csv


_V_COLS      = [f'V{i}' for i in range(1, 29)]
_TOP_SIGNAL  = ['V14', 'V17', 'V12', 'V10']   # highest |corr| with Class from EDA

logger = logging.getLogger(__name__)

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer new features from the cleaned credit card transactions DataFrame.

    Args:
        df: Cleaned DataFrame with columns Time, V1-V28, Amount, and
            optionally Class.

    Returns:
        New DataFrame containing the original columns plus 11 engineered
        features (5 domain-specific, 3 statistical, 3 interaction).
    """
    out = df.copy()
    # Domain-specific features -These encode established knowledge of how credit card fraud manifests.
    # Amount is strongly right-skewed: most transactions are small but a long tail extends into the thousands.  
    # Log-transforming stabilises variance and prevents large amounts from dominating distance-based and linear models.
    # np.log1p(x): Calculates the natural logarithm of one plus the input ln(1+x)
    # np.log(x): Calculates the natural logarithm ln(x) - It is undefined at x = 0, and returns -inf thus loosing precision
    out['log_amount'] = np.log1p(df['Amount'])
    
    # Time records seconds elapsed since the dataset's first transaction. 
    # Applying the modulo 86,400 operator to a Unix timestamp converts it into a value representing the number of seconds passed since midnight (UTC).
    # Modulo 86 400 recovers an approximate intra-day signal.  
    # Fraud is disproportionately concentrated in late-night hours (00–6 AM) when cardholders are least likely to notice and respond to alerts.
    out['hour_of_day'] = (df['Time'] % 86400) / 3600

    # Binary flag for transactions between midnight and 6 AM.  Gives tree-based models a clean categorical split instead of 
    # having to discover the non-linear hour boundary inside a continuous feature.
    out['is_night'] = (out['hour_of_day'] < 6).astype(np.int8)

    # Card-testing pattern: fraudsters who obtain stolen credentials often run a micropayment (< $1) to verify the card is live before executing larger fraudulent charges.  
    # This flag directly encodes that behaviour.
    out['is_micropayment'] = (df['Amount'] < 1.0).astype(np.int8)

    # High-value transactions carry greater financial exposure and are a frequent fraud target.  
    # An explicit threshold flag makes the risk boundary visible to linear models that cannot discover it from Amount alone.
    out['is_high_value'] = (df['Amount'] > 500.0).astype(np.int8)

    # -------------------------------------------------------------------------
    # Statistical features derived from PCA components (V1–V28). V1–V28 are orthogonal directions in the original feature space.  
    # Aggregate statistics across all 28 dimensions capture how far a transaction sits from the dense normal-behaviour cluster.
    # Euclidean distance from the PCA origin across all 28 components.
    # Fraud transactions cluster away from the normal region, so a large magnitude acts as a global anomaly indicator without committing to any single feature direction.
    out['pca_magnitude'] = np.sqrt((df[_V_COLS] ** 2).sum(axis=1))

    # Focused version of the magnitude using only the 4 PCA features that EDA identified as most correlated with Class.  
    # Concentrating on the high-signal directions reduces noise from components that carry little fraud information.
    out['high_signal_magnitude'] = np.sqrt((df[_TOP_SIGNAL] ** 2).sum(axis=1))

    # Count of PCA features whose |value| exceeds 3 (≈ 3 standard deviations).
    # Legitimate transactions rarely show extreme values across multiple independent PCA dimensions simultaneously; 
    # fraud transactions often do, making this count a lightweight multi-dimensional anomaly score.
    out['pca_extreme_count'] = (df[_V_COLS].abs() > 3).sum(axis=1).astype(np.int16)

    # -------------------------------------------------------------------------
    # Interaction features
    # Two features together can carry more signal than either does separately.
    # High PCA anomaly combined with a large transaction amount are the joint conditions that maximise fraud probability.  
    # The product creates a continuous score that ranks combinations of both risk factors and cannot be recovered by either feature independently.
    out['amount_pca_risk'] = out['log_amount'] * out['high_signal_magnitude']

    # A nighttime large-value transaction combines two independent risk factors.
    # Neither the time flag nor the value flag alone distinguishes this scenario; their product creates a new indicator for the riskiest joint condition.
    out['night_high_value'] = out['is_night'] * out['is_high_value']

    # V14 and V17 are the two PCA features most correlated with fraud.  Their product amplifies the signal when both dimensions are simultaneously
    # anomalous — a transaction extreme in only one direction contributes little; a transaction extreme in both is flagged strongly.
    out['v14_v17_product'] = df['V14'] * df['V17']
    return out


def select_features(df: pd.DataFrame, corr_threshold: float = 0.95, var_multiplier: float = 0.01) -> tuple[list[str], pd.DataFrame]:
    """
    Remove highly correlated and low-variance numeric features.

    Two-pass filter:
      1. Correlation: for each pair with |corr| > corr_threshold, the later
         column (by position) is dropped; the earlier one is kept.
      2. Variance: features whose variance falls below
         var_multiplier * mean(all remaining variances) are dropped.

    Args:
        df: DataFrame whose numeric columns are the candidate features.
        corr_threshold: Absolute correlation cutoff. Default 0.95.
        var_multiplier: Fraction of mean variance used as the low-variance
            cutoff. Default 0.01.

    Returns:
        selected: List of retained column names.
        reduced:  DataFrame containing only the retained columns.
    """
    numeric = df.select_dtypes(include="number")
    cols = list(numeric.columns)

    # correlation filter :
    # Correlation pass — iterates column pairs using the upper triangle of the absolute correlation matrix. For each pair exceeding        
    # corr_threshold (default 0.95), the later column is dropped (first column is kept). Already-dropped columns are skipped to avoid redundant comparisons. 
    corr = numeric.corr().abs()
    to_drop = set()
    for i, col_i in enumerate(cols):
        if col_i in to_drop:
            continue
        for col_j in cols[i + 1:]:
            if col_j in to_drop:
                continue
            if corr.loc[col_i, col_j] > corr_threshold:
                to_drop.add(col_j)
                logger.info("Dropped (correlation) '%s': corr('%s', '%s') = %.4f > %.2f",
                            col_j, col_j, col_i, corr.loc[col_i, col_j], corr_threshold)
    after_corr = [c for c in cols if c not in to_drop]

    # variance filter:
    # Variance pass — computed only on the columns that survived pass 1. 
    # overall_variance is the mean variance across those survivors. However, if any of the columns has outliers eg 'Time' column compared to other columns, using mean returns skewed results
    # the threshold used here will ne  var_multiplier * overall_variance (default 0.01 * median_variance).
    # Using median variance ensures robustness to outliers 
    variances = numeric[after_corr].var()
    overall_variance = variances.median()
    variance_threshold = var_multiplier * overall_variance
    low_var = variances[variances < variance_threshold].index.tolist()
    for col in low_var:
        logger.info("Dropped (low variance) '%s': var = %.6f < threshold %.6f "
                    "(%.2f * overall %.6f)", col, variances[col], variance_threshold, var_multiplier, overall_variance)

    selected = [c for c in after_corr if c not in low_var]
    logger.info("select_features: retained %d / %d features "
                "(%d dropped for correlation, %d for low variance).", len(selected), len(cols), len(to_drop), len(low_var))
    return selected, df[selected]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features from the cleaned credit card transactions DataFrame.
    Args:
        df: Cleaned DataFrame with columns Time, V1-V28, Amount, and
            optionally Class.
    Returns:
        New DataFrame containing the original columns plus 11 engineered
        features (5 domain-specific, 3 statistical, 3 interaction).
    """
    raw = df.copy()
    # Create new features
    engineered = create_features(raw)
    new_cols = [c for c in engineered.columns if c not in raw.columns]
    print(f"\nEngineered features ({len(new_cols)}):")
    for col in new_cols:
        s = engineered[col]
        print(f"  {col:<25}  min={s.min():>10.4f}  max={s.max():>10.4f}  mean={s.mean():>10.4f}")
    print(f"\nFinal shape: {engineered.shape[0]:,} rows x {engineered.shape[1]} columns")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run feature engineering on a cleaned CSV dataset from data/")                                  
    parser.add_argument("filename", nargs="?", default="cleaned.csv",
                        help="CSV filename inside data. Defaults to cleaned.csv.")                  
    args = parser.parse_args()  
    raw = load_csv(args.filename)                                                                                       
    print(f"Loaded '{args.filename}': {len(raw):,} rows x {raw.shape[1]} columns") 
    engineer_features(raw)
    