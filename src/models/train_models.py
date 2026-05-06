"""
Train and compare Random Forest and XGBoost classifiers for credit card fraud detection.

Model selection rationale
─────────────────────────
Both models are tree-based ensembles and therefore inherently explainable via feature importance and SHAP values — which a fraud analyst can use to justify flagging a transaction.

Random Forest
  • Builds many independent decision trees and averages their votes.
  • `class_weight='balanced'` up-weights each fraud sample by n_samples /
    (n_classes * n_fraud) so every tree sees the minority class in proportion.
  • Low variance due to bagging; robust to outliers and non-linear boundaries.
  • Feature importance via mean impurity decrease (MDI).

XGBoost
  • Sequentially builds trees that focus on the residual errors of prior trees,
    meaning rare, hard-to-classify fraud cases receive increasing attention.
  • `scale_pos_weight = n_negative / n_positive` rescales the gradient for the
    positive class so each fraud sample contributes ~600× more to the loss.
  • Native SHAP integration enables per-prediction, per-feature explanations —
    critical for fraud analysts who must explain decisions to cardholders.
  • Typically yields higher AUC than Random Forest on tabular, imbalanced data.

Evaluation
──────────
Primary metric: AUC-ROC — measures ranking quality across all thresholds and is insensitive to class imbalance, making it the right CV scorer here.
Secondary metrics printed at test time: Precision, Recall, F1 (threshold=0.5).
CV uses StratifiedKFold so every fold preserves the original fraud ratio.
StratifiedKFold-  a cross-validation technique used to evaluate machine learning models, particularly for classification problems with imbalanced datasets
"""

from __future__ import annotations
import logging
import os
import time
import joblib
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier
from src.models.baseline import load_features

MODELS_DIR = Path(__file__).parents[2] / "models"
load_dotenv(Path(__file__).parents[2] / ".env")
TARGET = os.getenv("TARGET_COL")
FEATURES_FILE = os.getenv("FEATURES_FILE")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def build_models(scale_pos_weight: float) -> dict:
  """
    Return the two candidate models pre-configured for class imbalance.

    Args:
        scale_pos_weight: Ratio n_negative / n_positive passed to XGBoost.
  """
  return {"RandomForest": RandomForestClassifier(class_weight="balanced",n_estimators=100,random_state=42,n_jobs=-1),
            "XGBoost": XGBClassifier(scale_pos_weight=scale_pos_weight,n_estimators=100,random_state=42,eval_metric="auc",
                                verbosity=0)}


def _test_metrics(model, X_test, y_test) -> dict:
  y_pred = model.predict(X_test)
  y_prob = model.predict_proba(X_test)[:, 1]
  return {
      "accuracy":  accuracy_score(y_test, y_pred),
      "precision": precision_score(y_test, y_pred, zero_division=0),
      "recall":    recall_score(y_test, y_pred, zero_division=0),
      "f1":        f1_score(y_test, y_pred, zero_division=0),
      "auc_roc":   roc_auc_score(y_test, y_prob),
  }


def save_model(model, name: str) -> Path:
  MODELS_DIR.mkdir(parents=True, exist_ok=True)
  path = MODELS_DIR / f"{name.lower()}.pkl"
  joblib.dump(model, path)
  return path


def run_training()-> pd.DataFrame:
  # ── Load features dataset that contains feature engineered features
  print(f"Loading {FEATURES_FILE} ...")
  X, y = load_features(FEATURES_FILE)
  n_pos = int(y.sum())
  n_neg = len(y) - n_pos
  print(f"  {len(X):,} rows | {X.shape[1]} features | " 
        f"{n_pos:,} fraud ({y.mean() * 100:.3f}%)")
  # stratify=y keeps fraud ratio identical in train and test
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
  print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")
  # ── Train & evaluate the ML models
  scale_pos_weight = n_neg / n_pos
  models = build_models(scale_pos_weight)
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  # ── Run cross-validation and evaluation
  rows = []
  for name, model in models.items():
      print(f"[{name}] Running 5-fold CV ...")
      # run cross validation to prevent over fitting and allow model to generalize on new dataset
      cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
      print(f"[{name}] Fitting final model on full training set ...")
      t0 = time.perf_counter()
      model.fit(X_train, y_train)
      train_time = time.perf_counter() - t0
      tm = _test_metrics(model, X_test, y_test)
      rows.append({"Model": name,
                    "CV AUC Mean":     round(cv_scores.mean(), 4),
                    "CV AUC Std":      round(cv_scores.std(),  4),
                    "Test AUC-ROC":    round(tm["auc_roc"],   4),
                    "Test F1":         round(tm["f1"],         4),
                    "Test Recall":     round(tm["recall"],     4),
                    "Test Precision":  round(tm["precision"],  4), 
                    "Train Time (s)":  round(train_time, 1)})

      out = save_model(model, name)
      print(f"[{name}] Saved → {out}\n")
  return pd.DataFrame(rows)
  


if __name__ == "__main__":
  # create comparison table
  comparison = run_training()
  sep = "=" * 90
  print(sep)
  print("MODEL COMPARISON")
  print(sep)
  print(comparison.to_string(index=False))
  print(sep)

  # ── Recommendation and save best model metrics to path ────────────────────────────────────────────────────────
  best_row = comparison.loc[comparison["CV AUC Mean"].idxmax()]
  best_name = best_row["Model"]
  
  MODELS_DIR.mkdir(parents=True, exist_ok=True)
  params_path = MODELS_DIR / f"{best_name}_metrics.json"
  best_metrics = dict(best_row[['CV AUC Mean', 'Test AUC-ROC', 'Test Recall', 'Test F1']])
  with open(params_path, "w") as f:
      json.dump(best_metrics, f, indent=2)
  print(f"\nBest hyperparameters saved → {params_path}")
  print(f"\nRecommendation: {best_name}")
  
  print(f"  CV AUC {best_row['CV AUC Mean']:.4f} ± {best_row['CV AUC Std']:.4f}  |  "
        f"Test AUC {best_row['Test AUC-ROC']:.4f}  |  "
        f"Recall {best_row['Test Recall']:.4f}  |  "
        f"F1 {best_row['Test F1']:.4f}")
  print("\nFor a fraud analyst the most important metric is Recall: every missed fraud"
        "\n(false negative) is a financial loss to the bank. Pick the model with the"
        "\nhighest CV AUC as the primary ranking, then verify it also leads on Recall."
        "\nBoth models expose feature importance and SHAP values for decision explanation.")
