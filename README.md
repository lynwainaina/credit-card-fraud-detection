# Credit Card Fraud Detection

It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

## Dataset Description
- The dataset has been downloaded from [kaggle.com](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download) 
- The dataset contains transactions made by credit cards in September 2013 by European cardholders.
- This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. - The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
- It contains only numerical input variables which are the result of a PCA transformation.
- Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. 
- Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. 
- The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. 
- Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

## Setup

Install the project in editable mode so `src/` is importable as a package:

```bash
pip install -e .
pip install -r requirements.txt
```

## Common Commands

```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_<module>.py

# Run a single test by name
pytest tests/test_<module>.py::test_<function_name>

# Start the FastAPI server
uvicorn app.main:app --reload

# Start the Streamlit dashboard
streamlit run app/dashboard.py

# Start the MLflow tracking UI
mlflow ui
```

## Architecture

The project follows a layered pipeline: raw data → features → model → serving.

- **`src/data/`** — Data ingestion and validation. Loads raw data and validates it with Great Expectations before anything downstream consumes it.
- **`src/features/`** — Feature engineering. Transforms validated data into model-ready feature sets. Both training and inference must use the same transformations to avoid train-serve skew.
- **`src/models/`** — Model training, evaluation, and prediction. Logs experiments and artifacts to MLflow. Saved model artifacts go to `models/` (gitignored).
- **`app/`** — Serving layer. FastAPI exposes a REST prediction endpoint; Streamlit provides an interactive dashboard. Both import from `src/` via the editable install.
- **`tests/`** — Unit tests mirroring the `src/` structure.
- **`notebooks/`** — Exploratory analysis only. Production logic lives in `src/`, not notebooks.
- **`data/`** — Raw and processed data (gitignored, never committed).
- **`models/`** — Serialized model artifacts (gitignored, versioned via MLflow instead).

## Exploratory Data Analysis

Full analysis: [`notebooks/eda.ipynb`](notebooks/eda.ipynb)

**Dataset:** 284,807 transactions × 31 columns — `Time`, `Amount`, 28 anonymised PCA features (`V1`–`V28`), and target `Class`.  
**Feature types:** All numeric. `Time` and `Amount` are raw; `V1`–`V28` are pre-computed PCA components with no missing values.

**Key findings:**

- **Severe class imbalance:** Only ~0.17% of transactions are fraud. Accuracy is meaningless here — optimise for AUC-PR or F1.
- **Strong separability in PCA features:** `V14`, `V17`, `V12`, and `V10` have the highest absolute correlation with `Class` and show clear distributional separation between classes in box plots.
- **`Amount` is right-skewed:** Apply `np.log1p` before scaling to reduce the effect of large outliers.
- **`V1`–`V28` are orthogonal:** Near-zero pairwise correlations confirm no redundancy — no further dimensionality reduction needed.

**Modeling implications:** Use `class_weight='balanced'` or SMOTE (training split only); log-transform `Amount`; evaluate on AUC-PR rather than accuracy.

## Note

- `src/` is the importable package root (configured in `setup.py`). Import as `from data.loader import ...`, `from models.train import ...`, etc.
- MLflow is the source of truth for experiment tracking and model versioning — do not rely on file timestamps in `models/` for version control.
- Data validation (Great Expectations) runs at ingestion time in `src/data/`; downstream code can assume data conforms to the defined expectations.
