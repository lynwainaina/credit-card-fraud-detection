"""
Baseline model for credit card fraud detection.

Trains a LogisticRegression classifier on features.csv, evaluates it on
a held-out test set, and saves the fitted model to models/baseline.pkl.
"""

from __future__ import annotations
import logging
import os
import joblib
from dotenv import load_dotenv
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from src.data.loader import load_csv


MODELS_DIR = Path(__file__).parents[2] / "models"
load_dotenv(Path(__file__).parents[2] / ".env")
TARGET = os.getenv("TARGET_COL")
features_file = os.getenv("FEATURES_FILE") 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def load_features(filename:str):
    "Load the feature-engineered dataset using the load_csv helper function and return X, y."
    df = load_csv(filename)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def train(X_train, y_train) -> LogisticRegression:
    "Fit a LogisticRegression() model with default settings."
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def evaluate(model: LogisticRegression, X_test, y_test) -> dict:
    "Compute and print classification metrics on the test set."
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {"accuracy": accuracy_score(y_test, y_pred),
               "precision": precision_score(y_test, y_pred),
               "recall": recall_score(y_test, y_pred),
               "f1": f1_score(y_test, y_pred),
               "auc_roc": roc_auc_score(y_test, y_prob)}

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1       : {metrics['f1']:.4f}")
    print(f"AUC-ROC  : {metrics['auc_roc']:.4f}")
    return metrics


def save_model(model: LogisticRegression, filename: str = "baseline.pkl") -> Path:
    "Save the fitted model to models/<filename> using joblib."
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)
    return path

def run_baseline()-> None:
    print("Loading features.csv ...")
    X, y = load_features(features_file)
    print(f"  {len(X):,} rows, {X.shape[1]} features, "
          f"{int(y.sum()):,} positives ({y.mean() * 100:.3f}%)")
    # split to train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTrain: {len(X_train):,} rows  |  Test: {len(X_test):,} rows \nTraining LogisticRegression ...")
    model = train(X_train, y_train)
    print("\nTest-set metrics:")
    evaluate(model, X_test, y_test)
    out = save_model(model)
    print(f"\nModel saved → {out}")

    
if __name__ == "__main__":
    run_baseline()

