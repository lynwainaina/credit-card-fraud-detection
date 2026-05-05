import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.data.quality import check_data_quality, print_quality_report

ROOT = Path(__file__).parents[1]


# ---------------------------------------------------------------------------
# Fixtures and helpers

# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cleaned_df():
    return pd.read_csv(ROOT / "data" / "cleaned.csv")


def _make_valid_df(n: int = 1000, fraud_frac: float = 0.05) -> pd.DataFrame:
    """Synthetic DataFrame resembling the credit card data, that satisfies every quality check."""
    rng = np.random.default_rng(42)
    n_fraud = int(n * fraud_frac)
    n_legit = n - n_fraud
    data = {
        "Time":   rng.uniform(0, 172800, n).astype("float64"),
        "Amount": rng.uniform(0.01, 5000, n).astype("float64"),
        "Class":  np.array([0] * n_legit + [1] * n_fraud, dtype="int64"),
    }
    for i in range(1, 29):
        data[f"V{i}"] = rng.standard_normal(n).astype("float64")
    return pd.DataFrame(data)


@pytest.fixture
def valid_df():
    return _make_valid_df()


@pytest.fixture
def bad_df():
    """Wrong columns — fails schema check."""
    return pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})


@pytest.fixture
def null_df():
    """Correct columns but all-NaN values — fails null-rate and dtype checks."""
    cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]
    return pd.DataFrame({c: [np.nan] * 200 for c in cols})


# ---------------------------------------------------------------------------
# Passing / success cases
# ---------------------------------------------------------------------------

def test_passes_on_cleaned_data(cleaned_df):
    result = check_data_quality(cleaned_df)
    assert result["success"] is True


def test_passes_on_synthetic_data(valid_df):
    result = check_data_quality(valid_df)
    assert result["success"] is True
    assert result["failures"] == []


def test_result_structure(valid_df):
    result = check_data_quality(valid_df)
    assert set(result.keys()) == {"success", "failures", "warnings", "statistics"}
    stats = result["statistics"]
    assert "total_rows" in stats
    assert "total_columns" in stats
    assert "total_nulls_by_column" in stats
    assert "target_distribution" in stats


# ---------------------------------------------------------------------------
# Schema check (_check_schema)
# ---------------------------------------------------------------------------

def test_fails_missing_columns(bad_df):
    result = check_data_quality(bad_df)
    assert result["success"] is False
    assert any("missing required column" in f for f in result["failures"])


def test_fails_wrong_dtype():
    df = _make_valid_df()
    df["Class"] = df["Class"].astype("float64")  # schema expects int64
    result = check_data_quality(df)
    assert result["success"] is False
    assert any("dtype" in f and "Class" in f for f in result["failures"])


# ---------------------------------------------------------------------------
# Row count check (_check_row_count)
# ---------------------------------------------------------------------------

def test_fails_too_few_rows():
    df = _make_valid_df(n=50, fraud_frac=0.1)  # 50 rows < 100 minimum
    result = check_data_quality(df)
    assert result["success"] is False
    assert any("Row count" in f for f in result["failures"])


def test_warns_low_row_count():
    df = _make_valid_df(n=500)  # 100 ≤ n < 1000 → warning, not failure
    result = check_data_quality(df)
    assert result["success"] is True
    assert any("Row count" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Null rate check (_check_null_rates)
# ---------------------------------------------------------------------------

def test_fails_all_nulls(null_df):
    result = check_data_quality(null_df)
    assert result["success"] is False


def test_fails_critical_null_rate():
    df = _make_valid_df()
    df.loc[df.sample(frac=0.55, random_state=0).index, "Amount"] = np.nan
    result = check_data_quality(df)
    assert result["success"] is False
    assert any("Amount" in f and "null" in f.lower() for f in result["failures"])


def test_warns_high_null_rate():
    df = _make_valid_df()
    df.loc[df.sample(frac=0.30, random_state=0).index, "Amount"] = np.nan
    result = check_data_quality(df)
    assert result["success"] is True
    assert any("Amount" in w and "null" in w.lower() for w in result["warnings"])


def test_null_counts_in_statistics():
    df = _make_valid_df()
    df.loc[0, "Amount"] = np.nan
    result = check_data_quality(df)
    assert result["statistics"]["total_nulls_by_column"]["Amount"] == 1


# ---------------------------------------------------------------------------
# Value range check (_check_value_ranges)
# ---------------------------------------------------------------------------

def test_fails_negative_amount():
    df = _make_valid_df()
    df.loc[0, "Amount"] = -1.0
    result = check_data_quality(df)
    assert result["success"] is False
    assert any("Amount" in f and "below minimum" in f for f in result["failures"])


def test_fails_negative_time():
    df = _make_valid_df()
    df.loc[0, "Time"] = -1.0
    result = check_data_quality(df)
    assert result["success"] is False
    assert any("Time" in f and "below minimum" in f for f in result["failures"])


# ---------------------------------------------------------------------------
# Target distribution check (_check_target_distribution)
# ---------------------------------------------------------------------------

def test_fails_single_class():
    df = _make_valid_df()
    df["Class"] = np.zeros(len(df), dtype="int64")  # all class 0
    result = check_data_quality(df)
    assert result["success"] is False
    assert any("only" in f.lower() for f in result["failures"])


def test_warns_imbalanced_target(cleaned_df):
    # cleaned_df has ~0.17% fraud — well below the 5% warning threshold
    result = check_data_quality(cleaned_df)
    assert any("imbalanced" in w for w in result["warnings"])


def test_target_distribution_in_statistics(valid_df):
    result = check_data_quality(valid_df)
    dist = result["statistics"]["target_distribution"]
    assert "0" in dist and "1" in dist
    assert abs(sum(dist.values()) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# print_quality_report (smoke tests)
# ---------------------------------------------------------------------------

def test_print_report_passed(valid_df, capsys):
    print_quality_report(check_data_quality(valid_df))
    assert "PASSED" in capsys.readouterr().out


def test_print_report_failed(bad_df, capsys):
    print_quality_report(check_data_quality(bad_df))
    assert "FAILED" in capsys.readouterr().out
