import numpy as np
import pandas as pd
import pytest
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from src.models.train_models import build_models, _test_metrics, save_model

ROOT = Path(__file__).parents[1]
PRODUCTION_MODEL_PATH = ROOT / "models" / "production_model.pkl"
FEATURES_CSV = ROOT / "data" / "features.csv"


@pytest.fixture(scope="module")
def synthetic_data():
    """Small labeled dataset for unit-testing model functions without disk I/O."""
    rng = np.random.default_rng(42)
    n = 300
    X = pd.DataFrame(
        rng.standard_normal((n, 28)), columns=[f"V{i}" for i in range(1, 29)]
    )
    y = pd.Series(np.array([0] * 285 + [1] * 15), name="Class")
    return X, y


@pytest.fixture(scope="module")
def small_trained_rf(synthetic_data):
    """Minimal RandomForest trained on synthetic data — no saved artifact needed."""
    X, y = synthetic_data
    model = RandomForestClassifier(
        n_estimators=5, random_state=42, class_weight="balanced"
    )
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def production_model():
    """Load the saved production model; skip gracefully if not yet trained."""
    if not PRODUCTION_MODEL_PATH.exists():
        pytest.skip(
            f"No model at {PRODUCTION_MODEL_PATH} — run src/models/train_models.py first"
        )
    return joblib.load(PRODUCTION_MODEL_PATH)


@pytest.fixture(scope="module")
def features_sample():
    """100-row feature sample from features.csv; skip if the file is absent."""
    if not FEATURES_CSV.exists():
        pytest.skip("features.csv not found — run feature engineering first")
    return pd.read_csv(FEATURES_CSV).drop(columns=["Class"]).sample(100, random_state=42)


# ---------------------------------------------------------------------------
# build_models
# ---------------------------------------------------------------------------

def test_build_models_returns_both_models():
    models = build_models(scale_pos_weight=100.0)
    assert set(models.keys()) == {"RandomForest", "XGBoost"}


def test_build_models_rf_is_balanced():
    models = build_models(scale_pos_weight=100.0)
    assert models["RandomForest"].class_weight == "balanced"


def test_build_models_xgb_scale_pos_weight():
    spw = 200.0
    models = build_models(scale_pos_weight=spw)
    assert models["XGBoost"].get_params()["scale_pos_weight"] == spw


def test_build_models_predict_interface():
    for name, model in build_models(scale_pos_weight=100.0).items():
        assert hasattr(model, "predict"), f"{name} missing predict"
        assert hasattr(model, "predict_proba"), f"{name} missing predict_proba"


# ---------------------------------------------------------------------------
# _test_metrics
# ---------------------------------------------------------------------------

def test_metrics_returns_expected_keys(small_trained_rf, synthetic_data):
    X, y = synthetic_data
    result = _test_metrics(small_trained_rf, X, y)
    assert set(result.keys()) == {"accuracy", "precision", "recall", "f1", "auc_roc"}


def test_metrics_values_in_unit_interval(small_trained_rf, synthetic_data):
    X, y = synthetic_data
    result = _test_metrics(small_trained_rf, X, y)
    for key, val in result.items():
        assert 0.0 <= val <= 1.0, f"{key} = {val:.4f} is outside [0, 1]"


def test_metrics_auc_above_chance(small_trained_rf, synthetic_data):
    X, y = synthetic_data
    result = _test_metrics(small_trained_rf, X, y)
    assert result["auc_roc"] > 0.5, "AUC-ROC should exceed the random-chance baseline of 0.5"


# ---------------------------------------------------------------------------
# save_model
# ---------------------------------------------------------------------------

def test_save_model_creates_pkl_file(small_trained_rf, tmp_path, monkeypatch):
    import src.models.train_models as tm
    monkeypatch.setattr(tm, "MODELS_DIR", tmp_path)
    path = save_model(small_trained_rf, "TestModel")
    assert path.exists()
    assert path.name == "testmodel.pkl"


def test_save_model_artifact_is_loadable(small_trained_rf, tmp_path, monkeypatch):
    import src.models.train_models as tm
    monkeypatch.setattr(tm, "MODELS_DIR", tmp_path)
    path = save_model(small_trained_rf, "TestModel")
    loaded = joblib.load(path)
    assert hasattr(loaded, "predict") and hasattr(loaded, "predict_proba")


# ---------------------------------------------------------------------------
# Prediction interface — verified on an in-test trained model (no disk artifact)
# ---------------------------------------------------------------------------

def test_predict_returns_binary_labels(small_trained_rf, synthetic_data):
    X, _ = synthetic_data
    preds = small_trained_rf.predict(X)
    assert set(preds).issubset({0, 1}), f"Unexpected label values: {set(preds)}"


def test_predict_length_matches_input(small_trained_rf, synthetic_data):
    X, _ = synthetic_data
    assert len(small_trained_rf.predict(X)) == len(X)


def test_predict_proba_shape(small_trained_rf, synthetic_data):
    X, _ = synthetic_data
    proba = small_trained_rf.predict_proba(X)
    assert proba.shape == (len(X), 2), f"Expected ({len(X)}, 2), got {proba.shape}"


def test_predict_proba_in_range(small_trained_rf, synthetic_data):
    X, _ = synthetic_data
    proba = small_trained_rf.predict_proba(X)
    assert proba.shape[1] == 2
    assert (proba >= 0).all() and (proba <= 1).all()


def test_predict_proba_rows_sum_to_one(small_trained_rf, synthetic_data):
    X, _ = synthetic_data
    proba = small_trained_rf.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_fraud_probability_maps_to_column_index_1(small_trained_rf):
    """predict_proba[:, 1] must be the fraud-class probability, not the legitimate-class."""
    assert small_trained_rf.classes_[1] == 1, (
        "Class 1 (fraud) must correspond to column index 1 in predict_proba output"
    )


# ---------------------------------------------------------------------------
# Integration tests — require a saved production model and features.csv
# Skipped automatically when no trained artifact is present (e.g., fresh clone)
# ---------------------------------------------------------------------------

def test_production_model_has_predict_interface(production_model):
    assert hasattr(production_model, "predict")
    assert hasattr(production_model, "predict_proba")


def test_production_model_predict_on_real_features(production_model, features_sample):
    preds = production_model.predict(features_sample)
    assert set(preds).issubset({0, 1})
    assert len(preds) == len(features_sample)


def test_production_model_proba_shape(production_model, features_sample):
    proba = production_model.predict_proba(features_sample)
    assert proba.shape == (len(features_sample), 2)


def test_production_model_proba_in_range(production_model, features_sample):
    proba = production_model.predict_proba(features_sample)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_production_model_proba_sums_to_one(production_model, features_sample):
    proba = production_model.predict_proba(features_sample)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
