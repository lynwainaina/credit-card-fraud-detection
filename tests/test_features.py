import numpy as np
import pandas as pd
import pytest
from src.features.engineering import create_features, select_features

ENGINEERED_COLS = [
    "log_amount", "hour_of_day", "is_night", "is_micropayment", "is_high_value",
    "pca_magnitude", "high_signal_magnitude", "pca_extreme_count",
    "amount_pca_risk", "night_high_value", "v14_v17_product"]

_BASE_COLS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]

# create_features keeps all 31 original columns and appends 11 new ones → 42 total.
# EXPECTED_FEATURE_COUNT excludes Class → 41.
EXPECTED_FEATURE_COUNT = 41


@pytest.fixture(scope="module")
def cleaned_df():
    """Synthetic cleaned DataFrame with the same schema as cleaned.csv."""
    rng = np.random.default_rng(42)
    n = 500
    data = {f"V{i}": rng.standard_normal(n) for i in range(1, 29)}
    data["Time"] = rng.uniform(0, 172_800, n)
    data["Amount"] = np.abs(rng.standard_normal(n)) * 100
    data["Class"] = rng.choice([0, 1], size=n, p=[0.998, 0.002])
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def engineered_df(cleaned_df):
    # engineer_features prints a summary but has no return statement;
    # create_features is the function that builds and returns the enriched DataFrame.
    return create_features(cleaned_df)


# ── column presence & count ───────────────────────────────────────────────────

def test_all_engineered_columns_present(engineered_df):
    missing = [c for c in ENGINEERED_COLS if c not in engineered_df.columns]
    assert not missing, f"Missing engineered columns: {missing}"


def test_original_columns_preserved(engineered_df):
    missing = [c for c in _BASE_COLS if c not in engineered_df.columns]
    assert not missing, f"create_features dropped original columns: {missing}"


def test_feature_count(engineered_df):
    feature_cols = [c for c in engineered_df.columns if c != "Class"]
    assert len(feature_cols) == EXPECTED_FEATURE_COUNT, (
        f"Expected {EXPECTED_FEATURE_COUNT} features, got {len(feature_cols)}: "
        f"{feature_cols}"
    )


# ── data integrity ────────────────────────────────────────────────────────────

def test_no_nan_values(engineered_df):
    null_counts = engineered_df.isnull().sum()
    bad = null_counts[null_counts > 0]
    assert bad.empty, f"NaN values found:\n{bad}"


def test_row_count_preserved(engineered_df, cleaned_df):
    assert len(engineered_df) == len(cleaned_df)


def test_class_column_preserved(engineered_df, cleaned_df):
    assert "Class" in engineered_df.columns
    pd.testing.assert_series_equal(engineered_df["Class"], cleaned_df["Class"])


# ── feature ranges ────────────────────────────────────────────────────────────

def test_hour_of_day_range(engineered_df):
    # (Time % 86400) / 3600 produces [0, 24), not [0, 23]
    h = engineered_df["hour_of_day"]
    assert (h >= 0).all() and (h < 24).all()


def test_binary_flag_columns(engineered_df):
    # night_high_value is the product of two binary flags so it is also binary
    for col in ("is_night", "is_high_value", "is_micropayment", "night_high_value"):
        assert engineered_df[col].isin([0, 1]).all(), f"{col} is not binary"


def test_log_amount_non_negative(engineered_df):
    assert (engineered_df["log_amount"] >= 0).all()


def test_pca_magnitude_non_negative(engineered_df):
    assert (engineered_df["pca_magnitude"] >= 0).all()


def test_high_signal_magnitude_non_negative(engineered_df):
    assert (engineered_df["high_signal_magnitude"] >= 0).all()


def test_pca_extreme_count_non_negative(engineered_df):
    assert (engineered_df["pca_extreme_count"] >= 0).all()


# ── feature logic correctness ─────────────────────────────────────────────────

def _make_controlled_df() -> pd.DataFrame:
    """Four-row DataFrame with known, hand-verifiable values for formula tests."""
    return pd.DataFrame({
        **{f"V{i}": [0.0, 4.0, -4.0, 1.0] for i in range(1, 29)},
        "Time":   [0.0,    3_600.0,  21_600.0, 79_200.0],  # 00:00, 01:00, 06:00, 22:00
        "Amount": [0.5,    1.0,      100.0,    600.0],      # micro, boundary, normal, high
        "Class":  [0,      0,        1,        0],
    })


def test_log_amount_formula():
    df = _make_controlled_df()
    out = create_features(df)
    pd.testing.assert_series_equal(
        out["log_amount"], np.log1p(df["Amount"]), check_names=False
    )


def test_hour_of_day_formula():
    df = _make_controlled_df()
    out = create_features(df)
    pd.testing.assert_series_equal(
        out["hour_of_day"], (df["Time"] % 86_400) / 3_600, check_names=False
    )


def test_is_night_logic():
    df = _make_controlled_df()
    out = create_features(df)
    # 00:00 → night, 01:00 → night, 06:00 → NOT night, 22:00 → NOT night
    assert list(out["is_night"]) == [1, 1, 0, 0]


def test_is_micropayment_logic():
    df = _make_controlled_df()
    out = create_features(df)
    # 0.5 < 1 → micro; 1.0, 100.0, 600.0 → not micro
    assert list(out["is_micropayment"]) == [1, 0, 0, 0]


def test_is_high_value_logic():
    df = _make_controlled_df()
    out = create_features(df)
    # 600.0 > 500 → high; 0.5, 1.0, 100.0 → not high
    assert list(out["is_high_value"]) == [0, 0, 0, 1]


def test_night_high_value_logic():
    df = _make_controlled_df()
    out = create_features(df)
    pd.testing.assert_series_equal(
        out["night_high_value"], out["is_night"] * out["is_high_value"],
        check_names=False
    )


def test_v14_v17_product_formula():
    df = _make_controlled_df()
    out = create_features(df)
    pd.testing.assert_series_equal(
        out["v14_v17_product"], df["V14"] * df["V17"], check_names=False
    )


def test_amount_pca_risk_formula():
    df = _make_controlled_df()
    out = create_features(df)
    pd.testing.assert_series_equal(
        out["amount_pca_risk"], out["log_amount"] * out["high_signal_magnitude"],
        check_names=False
    )


def test_pca_extreme_count_formula():
    df = _make_controlled_df()
    out = create_features(df)
    # Row 0: all V = 0.0  → |0| > 3 is False for all 28 → count = 0
    # Row 1: all V = 4.0  → |4| > 3 is True  for all 28 → count = 28
    assert out["pca_extreme_count"].iloc[0] == 0
    assert out["pca_extreme_count"].iloc[1] == 28


# ── select_features ───────────────────────────────────────────────────────────

def test_select_features_returns_tuple(engineered_df):
    result = select_features(engineered_df)
    assert isinstance(result, tuple) and len(result) == 2
    selected, reduced = result
    assert isinstance(selected, list)
    assert isinstance(reduced, pd.DataFrame)


def test_select_features_cols_match_df(engineered_df):
    selected, reduced = select_features(engineered_df)
    assert list(reduced.columns) == selected


def test_select_features_drops_correlated_column():
    rng = np.random.default_rng(0)
    n = 300
    a = rng.standard_normal(n)
    df = pd.DataFrame({
        "A": a,
        "B": a + rng.normal(0, 1e-6, n),  # near-duplicate of A → corr ≈ 1.0
        "C": rng.standard_normal(n),
    })
    selected, _ = select_features(df, corr_threshold=0.95)
    assert "A" in selected
    assert "B" not in selected
    assert "C" in selected


def test_select_features_drops_low_variance_column():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "A": rng.standard_normal(n),
        "B": np.full(n, 5.0),        # zero variance → always below threshold
        "C": rng.standard_normal(n),
    })
    selected, _ = select_features(df)
    assert "B" not in selected
    assert "A" in selected
    assert "C" in selected


def test_select_features_retains_independent_columns():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "A": rng.standard_normal(n),
        "B": rng.standard_normal(n),
        "C": rng.standard_normal(n),
    })
    selected, reduced = select_features(df)
    assert set(selected) == {"A", "B", "C"}
    assert list(reduced.columns) == selected
