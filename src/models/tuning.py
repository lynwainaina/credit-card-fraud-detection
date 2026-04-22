"""
Hyperparameter tuning for XGBoost using Optuna.

XGBoost was selected as the best model from src/models/train_models.py - (CV AUC 0.9774 vs Random Forest 0.9458).

Results of tuning the XGboost: 
The tuned model catches 82% of frauds vs 77% before — at a default threshold of 0.5 that's an extra ~30 frauds caught per 56k transactions. Precision drops
because the model is now more aggressive about flagging borderline cases. For a fraud analyst, this is the right trade-off — a missed fraud is an
unrecoverable loss, while a false alert is just a phone call to the cardholder. The threshold can be adjusted up if false-positive volume is too high without
retraining.

Search strategy:
Optuna uses Tree-structured Parzen Estimators (TPE) — a Bayesian approach that builds a probability model over the hyperparameter space and samples
from the region most likely to improve the objective, so it converges faster than grid or random search.

Objective: 
maximise mean AUC-ROC across 5-fold StratifiedKFold on the training set. 
StratifiedKFold is mandatory here because the fraud class is only 0.17% of the data — random splits would produce folds with zero or near-zero fraud samples.

Each trial:
  1. Optuna proposes a hyperparameter configuration.
  2. cross_val_score fits 5 clones of XGBoost on 5 folds and returns AUC-ROC.
  3. The mean score is reported back to Optuna.
  4. Optuna updates its TPE model and proposes a better config next trial.
"""

from __future__ import annotations
import json
import logging
import os
import joblib
import optuna
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier
from src.models.baseline import load_features


MODELS_DIR = Path(__file__).parents[2] / "models"
load_dotenv(Path(__file__).parents[2] / ".env")
TARGET = os.getenv("TARGET_COL")
FEATURES_FILE = os.getenv("FEATURES_FILE")
N_TRIALS = int(os.getenv("N_TRIALS"))
CV_FOLDS = int(os.getenv("CV_FOLDS"))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Silence Optuna's own INFO chatter — we print a one-liner per trial instead
optuna.logging.set_verbosity(optuna.logging.WARNING)

    
def make_objective(X_train, y_train, scale_pos_weight: float, cv: StratifiedKFold):
    """
    Return an Optuna objective function closed over the training data.
    The search space covers the hyperparameters that most affect XGBoost performance on tabular, imbalanced data:
    Optuna Search Space is (9 hyperparameters) for this example
    n_estimators      — more trees improve fit but increase training time.
    max_depth         — deeper trees capture interactions but risk overfitting.
    learning_rate     — smaller rates need more trees; controls step size in gradient descent.
    subsample         — row sampling per tree; reduces variance.
    colsample_bytree  — feature sampling per tree; reduces correlation between trees and speeds up training.
    min_child_weight  — minimum sum of instance weights in a leaf; higher values prevent the model from fitting rare-fraud leaves with too few samples.
    gamma             — minimum loss-reduction required to make a split; higher values prune trees more aggressively.
    reg_alpha         — L1 regularisation; encourages sparsity.
    reg_lambda        — L2 regularisation; shrinks weights toward zero.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {"n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "scale_pos_weight": scale_pos_weight,
                "eval_metric": "auc",
                "verbosity": 0,
                "random_state": 42}
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        mean_auc = scores.mean()
        logger.info(
            "Trial %3d | AUC %.4f ± %.4f | n_est=%d depth=%d lr=%.4f",
            trial.number, mean_auc, scores.std(),
            params["n_estimators"], params["max_depth"], params["learning_rate"],
        )
        return mean_auc
    return objective


def evaluate(model: XGBClassifier, X_test, y_test) -> dict:
    """Compute and print all classification metrics on the test set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "auc_roc":   roc_auc_score(y_test, y_prob)
        }
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1       : {metrics['f1']:.4f}")
    print(f"AUC-ROC  : {metrics['auc_roc']:.4f}")
    return metrics


if __name__ == "__main__":
    # ── Load training data and split to train and test
    print(f"Loading {FEATURES_FILE} ...")
    X, y = load_features(FEATURES_FILE)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"{len(X):,} rows | {X.shape[1]} features | " 
          f"{n_pos:,} fraud ({y.mean() * 100:.3f}%)")
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")
    scale_pos_weight = n_neg / n_pos
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    
    # ── Optuna study ──────────────────────────────────────────────────────────
    print(f"Starting Optuna search ({N_TRIALS} trials, {CV_FOLDS}-fold CV) ...")
    print("-" * 60)
    # create_study - primary method for initializing a new Study object in Optuna, which manages the hyperparameter optimization process. 
    # storage - Defines where trial history is saved. None (default): Uses InMemoryStorage. Results are lost once the program terminates
    # direction - Sets the optimization goal, "maximize": Optimization aims for the highest objective value.
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42), 
                                study_name="xgboost_fraud_tuning")
    # Once a study object is created, you execute the optimization loop using the study.optimize method. 
    # Objective Function: a user-defined function that takes a trial object as an argument, suggests hyperparameters, and returns a single numerical value 
    # n_trials: The total number of trials to run.
    study.optimize(make_objective(X_train, y_train, scale_pos_weight, cv),
                   n_trials=N_TRIALS, show_progress_bar=True)
    print("-" * 60)
    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"  CV AUC: {study.best_value:.4f}")
    
    # ── Save best hyperparameters ─────────────────────────────────────────────
    best_params = dict(study.best_params)
    best_params["scale_pos_weight"] = scale_pos_weight

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    params_path = MODELS_DIR / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest hyperparameters saved → {params_path}")
    print(json.dumps(best_params, indent=2))

    # ── Train final model with best params ────────────────────────────────────
    print("\nTraining final XGBoost with best hyperparameters ...")
    final_model = XGBClassifier(
        **best_params,
        eval_metric="auc",
        verbosity=0,
        random_state=42,
    )
    final_model.fit(X_train, y_train)

    # ── Evaluate on test set and save metrics ──────────────────────────────────────────────────
    print("\nTest-set metrics (tuned model):")
    print("")
    metrics_path = MODELS_DIR / "tuned_model_metrics.json"
    metric = evaluate(final_model, X_test, y_test)
    with open(metrics_path, "w") as f:
        json.dump(metric, f, indent=2)
    

    # ── Save tuned model ──────────────────────────────────────────────────────
    model_path = MODELS_DIR / "tuned_model.pkl"
    joblib.dump(final_model, model_path)
    print(f"\nTuned model saved → {model_path}")
