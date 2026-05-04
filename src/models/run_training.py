"""
Track experiments with MLflow - You need to log all models tried so you can compare later:
End-to-end MLflow training pipeline for credit card fraud detection.
Runs two experiment configurations and logs everything to MLflow:

  baseline    — LogisticRegression with default settings. Establishes the
                reference point a fraud analyst can compare all future runs against.

  tuned_best  — XGBoost with Optuna-optimised hyperparameters loaded from
                models/best_params.json. This becomes the production artifact.

For each run MLflow records:
  Parameters  : model name + every hyperparameter (fully reproducible from UI)
  Metrics     : AUC-ROC, Recall, F1, Precision, Accuracy on both train and test
  Artifact    : serialised model file saved with joblib

After both runs the tuned model is promoted to models/production_model.pkl.

Usage
─────
    python -m run_training          # train and log
    mlflow server --host 127.0.0.1 --port 5000  # start tracking UI  
    open http://localhost:5000                  # inspect runs
"""

from __future__ import annotations
import json
import logging
import os
import tempfile
import joblib
import mlflow
from pathlib import Path
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from src.models.baseline import load_features

PROJECT_ROOT = Path(__file__).parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"

load_dotenv(PROJECT_ROOT / ".env")
TARGET = os.getenv("TARGET_COL")
FEATURES_FILE = os.getenv("FEATURES_FILE", "features.csv")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def compute_metrics(model, X, y, prefix: str) -> dict[str, float]:
    """
    Return a flat dict of five metrics, each prefixed with 'train_' or 'test_'.

    Using both train and test metrics in the same MLflow run makes it easy to
    spot overfitting at a glance in the UI: a large train/test AUC gap means
    the model has memorised the training set rather than learnt the fraud signal.
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {f"{prefix}_auc_roc":   roc_auc_score(y, y_prob),
            f"{prefix}_recall":    recall_score(y, y_pred, zero_division=0),
            f"{prefix}_f1":        f1_score(y, y_pred, zero_division=0),
            f"{prefix}_precision": precision_score(y, y_pred, zero_division=0),
            f"{prefix}_accuracy":  accuracy_score(y, y_pred)
            }


def build_configs(best_params: dict) -> dict:
    """
    Return an ordered dict mapping config name → {model, mlflow_params}.

    mlflow_params is logged verbatim so every run is fully reproducible
    by reading its parameter page in the MLflow UI.
    """
    # Baseline: LogisticRegression with sklearn defaults which is the reference point for comparison
    lr = LogisticRegression()
    lr_mlflow_params = {"model_name":   "LogisticRegression",
                        "C":            lr.C,
                        "max_iter":     lr.max_iter,
                        "solver":       lr.solver,
                        "class_weight": str(lr.class_weight),
                        "penalty":      lr.penalty
                        }
    # Best model: XGBoost with Optuna-tuned hyperparameters, which is the production garde model
    xgb = XGBClassifier(**best_params, eval_metric="auc", verbosity=0, random_state=42)
    xgb_mlflow_params = {"model_name": "XGBoost_tuned", **best_params}
    return {"baseline":   {"model": lr,  "params": lr_mlflow_params},
            "tuned_best": {"model": xgb, "params": xgb_mlflow_params}
            }

def run_model_train():
    # ── Load & split ──────────────────────────────────────────────────────────
    print(f"Loading {FEATURES_FILE} ...")
    X, y = load_features(FEATURES_FILE)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f" {len(X):,} rows | {X.shape[1]} features | {n_pos:,} fraud ({y.mean() * 100:.3f}%)\n")
    # stratify=y preserves the 0.17% fraud ratio in both splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # ── Load Optuna best params ───────────────────────────────────────────────
    params_path = MODELS_DIR / "best_params.json"
    with open(params_path) as f:
        best_params = json.load(f)
    logger.info("Loaded best params from %s", params_path)
    
    # ── MLflow setup ──────────────────────────────────────────────────────────
    # tell MLflow where to persist run data to a local file system in mlruns directory
    # In production you'd swap this for a remote server URI.                                                 
    tracking_uri = f"file://{PROJECT_ROOT}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    # set a named container for related runs. Think of it as a folder grouping all your fraud-detection iterations together.
    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT)
    logger.info("Tracking URI : %s", tracking_uri)
    logger.info("Experiment   : %s (id=%s)", experiment.name, experiment.experiment_id)
    # ── Training loop ─────────────────────────────────────────────────────────
    configs = build_configs(best_params)
    trained: dict = {}

    # iterate over baseline and tuned models
    for config_name, cfg in configs.items():
        model  = cfg["model"]
        params = cfg["params"]
        print(f"[{config_name}] Training and logging to MLflow ...")
        with mlflow.start_run(run_name=config_name):
            # Train on full training set
            model.fit(X_train, y_train)
            # Compute metrics on both splits
            train_metrics = compute_metrics(model, X_train, y_train, prefix="train")
            test_metrics  = compute_metrics(model, X_test,  y_test,  prefix="test")
            # Log everything to the active run i.e params like n_estimators, learning_rate
            mlflow.log_params(params)
            # Enables side-by-side comparison in the UI for metrics like AUC; train vs. test gap reveals overfitting
            mlflow.log_metrics({**train_metrics, **test_metrics})

            # Serialize with joblib and log as a run artifact
            with tempfile.TemporaryDirectory() as tmpdir:
                artifact_file = Path(tmpdir) / f"{config_name}.pkl"
                # tuned XGBoost gets written to production_model.pkl. 
                # In a mature MLOps setup you'd replace this with mflow.register_model() and use the Model Registry to manage staging → production transitions with approval workflows.       
                joblib.dump(model, artifact_file)
                # One-click download of any past model version saved as a serialized pkl model
                mlflow.log_artifact(str(artifact_file), artifact_path="model")
            run_id = mlflow.active_run().info.run_id
        print(f"  Run ID : {run_id}")
        print(f"  Train  — AUC {train_metrics['train_auc_roc']:.4f} | Recall {train_metrics['train_recall']:.4f} | "
              f"F1 {train_metrics['train_f1']:.4f}")
        print(f"  Test   — AUC {test_metrics['test_auc_roc']:.4f} | Recall {test_metrics['test_recall']:.4f} | "
              f"F1 {test_metrics['test_f1']:.4f}\n")
        trained[config_name] = model
    # ── Promote tuned model to production ─────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    prod_path = MODELS_DIR / "production_model.pkl"
    joblib.dump(trained["tuned_best"], prod_path)
    print(f"Production model saved → {prod_path}")
    
if __name__ == "__main__":
    run_model_train()
    # ── Instructions to open the UI ───────────────────────────────────────────
    print(f"\nTo view all runs in the MLflow UI:\n"
        f"  mlflow server --host 127.0.0.1 --port 5000\n"
        f"  open http://localhost:5000")
