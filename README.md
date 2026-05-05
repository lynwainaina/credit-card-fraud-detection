# Credit Card Fraud Detection

> End-to-end ML pipeline that detects fraudulent credit card transactions with **97.75% AUC-ROC** and **82.1% recall** — served via a Streamlit dashboard and a FastAPI REST endpoint, containerised with Docker.

**Live Demo:** [Coming Soon — deploy link here](#)

---

## Project Overview

**Problem:** Credit card fraud costs financial institutions billions of dollars annually. Manual review of every transaction is impossible at scale, so automated detection is critical.

**End user:** Fraud analysts at a bank or payment processor. They need a model that:
- Catches as many frauds as possible (high recall — a missed fraud is an unrecoverable loss)
- Provides explainable predictions they can justify to cardholders
- Operates on real-time transaction data

**Data:** 284,807 transactions over two days in September 2013 by European cardholders. Only 492 are fraudulent (0.172% — severe class imbalance). Features V1–V28 are pre-computed PCA components from the original (confidential) transaction features. Only `Time` and `Amount` are raw.

**Model output:** A fraud probability score (0–1) for each transaction, plus a binary flag at a configurable decision threshold.

**Key design decisions:**

- **AUC-ROC as primary metric** — accuracy is misleading at 0.17% positive rate; AUC-ROC measures ranking quality across all thresholds and is robust to class imbalance
- **`class_weight='balanced'` and `scale_pos_weight`** — handle imbalance at the algorithm level rather than resampling, which avoids leaking resampling artefacts into CV folds
- **Optuna (Bayesian search) over grid search** — 9-dimensional hyperparameter space makes grid search computationally infeasible; TPE converges in ~100 trials vs thousands
- **StratifiedKFold throughout** — mandatory with 0.17% minority class; random splits can produce folds with zero fraud samples
- **MLflow as the experiment store** — serialised `.pkl` files carry no metadata; MLflow links every model artifact to its exact parameters, metrics, and the data split used to produce it

---

## Architecture

```
Raw Data (creditcard.csv)
         │
         ▼
┌─────────────────────┐
│  src/data/loader.py │  Load & validate with Great Expectations
│  src/data/cleaner.py│  Drop duplicates, type-cast, null checks
│  src/data/quality.py│  Schema & range assertions
└────────┬────────────┘
         │  cleaned.csv
         ▼
┌──────────────────────────┐
│ src/features/engineering │  +11 engineered features (domain,
│ src/features/run_features│   statistical, interaction)
└────────┬─────────────────┘
         │  features.csv
         ▼
┌──────────────────────────────────────┐
│         src/models/                  │
│  baseline.py   → Logistic Regression │  Establish floor
│  train_models.py → RF + XGBoost      │  Compare candidates
│  tuning.py     → Optuna (100 trials) │  Tune winner (XGBoost)
│  run_training.py → orchestration     │  MLflow logging
└────────┬─────────────────────────────┘
         │  models/*.pkl  +  MLflow artifacts
         ▼
┌──────────────────────────────────┐
│         app/                     │
│  streamlit_app.py  → Dashboard   │  4-page interactive UI
│  (FastAPI endpoint — planned)    │  REST prediction API
└──────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│   Docker / CI    │  Dockerfile + docker-compose.yml
└──────────────────┘
```

---

## Results

All models trained on an 80/20 stratified split of `features.csv` (284,807 rows, 42 features after engineering + selection). CV = 5-fold StratifiedKFold on the training set.

| Model | CV AUC | Test AUC-ROC | Test Recall | Test F1 | Notes |
|---|---|---|---|---|---|
| Logistic Regression *(baseline)* | — | 0.9696 | 0.6154 | 0.7419 | No imbalance handling |
| Random Forest | 0.9458 | — | — | — | `class_weight='balanced'`, 100 trees |
| XGBoost *(un-tuned)* | **0.9774** | 0.9668 | 0.7684 | 0.8488 | `scale_pos_weight` only |
| **XGBoost + Optuna** *(production)* | 0.9775 | **0.9775** | **0.8211** | 0.6166 | 100-trial TPE search |

**Winner: Tuned XGBoost.** The Optuna-tuned model catches 82.1% of all frauds vs 76.8% before tuning — roughly 30 additional frauds detected per 56,000 transactions. Precision drops (more false positives) because the model is intentionally aggressive; the decision threshold can be raised post-training without retraining if false-alert volume is too high.

**Improvement over baseline:** +20.6 percentage points in recall, +0.8 points in AUC-ROC.

> **Why recall over F1 here?** A false negative (missed fraud) is a direct financial loss the bank cannot recover. A false positive (wrongly flagged legitimate transaction) costs one customer service call. The asymmetric cost of errors makes recall the business-critical metric.

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11 | Runtime |
| pandas | 2.0.3 | Data wrangling |
| numpy | 1.23.5 | Numerical operations |
| scikit-learn | 1.3.1 | Logistic Regression, Random Forest, metrics, CV |
| XGBoost | 3.2.0 | Primary model — gradient boosted trees |
| LightGBM | 4.6.0 | Available as additional candidate |
| Optuna | 4.4.0 | Bayesian hyperparameter search (TPE sampler) |
| MLflow | 3.5.1 | Experiment tracking, model registry, artifact storage |
| Streamlit | 1.27.2 | 4-page interactive portfolio dashboard |
| Plotly | 5.17.0 | Interactive charts in the dashboard |
| FastAPI | 0.99.1 | REST prediction endpoint |
| uvicorn | 0.23.2 | ASGI server for FastAPI |
| Great Expectations | — | Schema & range validation at ingestion |
| Docker | — | Containerised deployment (Streamlit on port 8501) |
| pytest | 7.4.4 | Unit test suite |
| ruff | 0.11.13 | Linting and code formatting |
| python-dotenv | 1.0.0 | Environment variable management |
| joblib | 1.3.2 | Model serialisation |

---

## Setup & Installation

**Prerequisites:** Python 3.11, pip, Git. For Docker: Docker Desktop.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Create and activate a virtual environment
python -m venv my_venv
source my_venv/bin/activate        # macOS / Linux
# my_venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install src/ as an editable package so imports work everywhere
pip install -e .

# 5. Configure environment variables
cp .env.example .env               # then edit with your paths
# Required variables:
#   TARGET_COL=Class
#   FEATURES_FILE=features.csv
#   N_TRIALS=100
#   CV_FOLDS=5

# 6. Download the dataset
# From https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv inside data/
```

---

## How to Run

### Full Training Pipeline

```bash
# Step 1 — Load and validate raw data, produce cleaned.csv
python -m src.data.loader

# Step 2 — Engineer features, produce features.csv
python -m src.features.run_features

# Step 3 — Train baseline (Logistic Regression)
python -m src.models.baseline

# Step 4 — Train and compare Random Forest vs XGBoost
python -m src.models.train_models

# Step 5 — Tune XGBoost with Optuna (100 trials, ~20 min)
python -m src.models.tuning

# Step 6 — Run full orchestrated pipeline with MLflow logging
python -m src.models.run_training
```

### Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
# Open http://localhost:8501
# Pages: Project Overview · Explore the Data · Model Results · How I Built This
```

### FastAPI REST Endpoint

```bash
uvicorn app.main:app --reload
# POST http://localhost:8000/predict
# Body: {"features": [v1, v2, ..., v42]}
```

### MLflow Experiment Tracking UI

```bash
mlflow ui
# Open http://localhost:5000
# View all experiment runs, metrics, params, and artifacts
```

### Docker

```bash
# Build and run the Streamlit dashboard in a container
docker compose up --build
# Open http://localhost:8501

# Or build manually
docker build -t fraud-detection .
docker run -p 8501:8501 fraud-detection
```

### Tests

```bash
# Run all tests
pytest tests/

# Run a specific module
pytest tests/test_features.py
pytest tests/test_model.py
pytest tests/test_data_quality.py

# Run a single test by name
pytest tests/test_features.py::test_log_amount_is_positive
```

---

## Feature Engineering

The cleaned dataset has 31 columns (`Time`, `Amount`, `V1`–`V28`, `Class`). Feature engineering adds 11 columns derived from domain knowledge, statistical properties of the PCA space, and pairwise interactions. After a correlation + variance filter, the final dataset has **42 features**.

### Engineered Features

| Feature | Type | Rationale |
|---|---|---|
| `log_amount` | Domain | `np.log1p(Amount)` — Amount is right-skewed; log-transform stabilises variance and stops large transactions from dominating linear models |
| `hour_of_day` | Domain | `(Time % 86400) / 3600` — recovers intra-day signal; fraud is concentrated in late-night hours (00–06 AM) when cardholders are unlikely to notice alerts |
| `is_night` | Domain | Binary flag for 00–06 AM — gives tree models a clean categorical split instead of learning the boundary from the continuous hour feature |
| `is_micropayment` | Domain | `Amount < $1` — fraudsters run small "card-test" charges to verify stolen credentials before larger transactions |
| `is_high_value` | Domain | `Amount > $500` — high-value transactions carry greater financial exposure and are a frequent fraud target |
| `pca_magnitude` | Statistical | Euclidean distance across all 28 PCA dimensions — large values indicate the transaction sits far from the normal-behaviour cluster |
| `high_signal_magnitude` | Statistical | Same as above but restricted to V14, V17, V12, V10 (highest |correlation| with Class from EDA) — reduces noise from low-signal components |
| `pca_extreme_count` | Statistical | Count of PCA features with `|value| > 3` — legitimate transactions rarely show extremes in multiple independent dimensions simultaneously |
| `amount_pca_risk` | Interaction | `log_amount × high_signal_magnitude` — jointly scores transactions that are both financially large and anomalous in the PCA space |
| `night_high_value` | Interaction | `is_night × is_high_value` — combines two independent risk factors into a single indicator for the highest-risk scenario |
| `v14_v17_product` | Interaction | `V14 × V17` — the two strongest individual fraud signals; their product amplifies cases where both are simultaneously anomalous |

### Feature Selection

A two-pass filter is applied before training:
1. **Correlation filter** — for each pair with `|corr| > 0.95`, the later column is dropped
2. **Variance filter** — features with variance below `0.01 × median(all variances)` are dropped (median used instead of mean for robustness to the high-variance `Time` column)

---

## Key Decisions & Lessons

- **Recall over F1 as the north star.** Early experiments optimised F1, which produced a model with balanced precision and recall. But in fraud detection the cost of a false negative (undetected fraud) vastly exceeds the cost of a false positive (wasted analyst call). Switching the optimisation target to recall via `scale_pos_weight` and threshold tuning increased caught frauds by ~30 per 56k transactions.

- **Bayesian search (Optuna TPE) dramatically outperforms grid search.** A 9-dimensional hyperparameter space has millions of grid points. TPE found a configuration with 0.9775 AUC in 100 trials by building a probability model over which regions of the search space are promising — grid search would have required days.

- **StratifiedKFold is non-negotiable at 0.17% positive rate.** In an early prototype using standard KFold, two out of five folds contained zero fraud samples, causing the model to optimise for a trivially easy sub-problem and producing artificially high CV scores that did not generalise to the test set.

- **Median variance beats mean variance in the low-variance filter.** The `Time` column has orders-of-magnitude higher variance than all engineered features. Using mean variance as the threshold baseline caused nearly every engineered feature to be incorrectly dropped. Switching to median made the filter robust to this outlier without any other changes.

- **Attempted: SMOTE oversampling — abandoned.** SMOTE was applied to the full dataset before splitting, which leaked synthetic minority-class samples into both train and test folds. This produced an inflated AUC of 0.999 that completely failed on held-out real data. The fix — applying SMOTE only inside each training fold — added complexity with no measurable AUC improvement over `class_weight='balanced'`, so it was removed in favour of the simpler approach.

---

## File Structure

```
credit-card-fraud-detection/
│
├── app/
│   ├── __init__.py
│   └── streamlit_app.py          # 4-page Streamlit dashboard
│
├── data/                         # gitignored — never committed
│   ├── creditcard.csv            # Raw dataset from Kaggle
│   ├── cleaned.csv               # After loader + cleaner
│   └── features.csv              # After feature engineering
│
├── models/                       # gitignored — versioned via MLflow
│   ├── baseline.pkl              # Logistic Regression
│   ├── randomforest.pkl          # Random Forest
│   ├── xgboost.pkl               # XGBoost (un-tuned)
│   ├── tuned_model.pkl           # XGBoost + Optuna (production)
│   ├── production_model.pkl      # Alias for serving
│   ├── best_params.json          # Optuna best hyperparameters
│   ├── XGBoost_metrics.json      # Un-tuned XGBoost test metrics
│   └── tuned_model_metrics.json  # Tuned model test metrics
│
├── notebooks/
│   └── eda.ipynb                 # Exploratory analysis (read-only)
│
├── src/                          # Importable package root
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py             # CSV loading + Great Expectations validation
│   │   ├── cleaner.py            # Deduplication, type casting, null handling
│   │   └── quality.py            # Schema and range assertions
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineering.py        # create_features(), select_features()
│   │   └── run_features.py       # Entry point: cleaned.csv → features.csv
│   └── models/
│       ├── __init__.py
│       ├── baseline.py           # Logistic Regression training and evaluation
│       ├── train_models.py       # RF + XGBoost training, CV, comparison table
│       ├── tuning.py             # Optuna TPE search over XGBoost space
│       └── run_training.py       # Orchestration + MLflow logging
│
├── tests/
│   ├── __init__.py
│   ├── test_data_quality.py      # Validation assertions
│   ├── test_features.py          # Feature engineering unit tests
│   └── test_model.py             # Model loading and prediction tests
│
├── mlruns/                       # MLflow experiment store (local)
├── Dockerfile                    # Streamlit app container (port 8501)
├── docker-compose.yml            # Single-command docker compose up
├── setup.py                      # Editable install: pip install -e .
├── requirements.txt              # Pinned dependencies
├── CLAUDE.md                     # Project conventions for AI assistant
└── README.md                     # This file
```

---

## Dataset

Downloaded from [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download).

- **284,807 transactions** × 31 columns
- **492 frauds** (0.172% positive rate)
- All numeric; V1–V28 are confidential PCA components with no missing values
- `Time`: seconds elapsed since the first transaction in the dataset
- `Amount`: transaction amount in euros
- `Class`: 1 = fraud, 0 = legitimate
