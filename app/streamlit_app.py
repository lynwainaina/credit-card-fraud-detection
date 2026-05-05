"""
Multi-page Streamlit portfolio app — Credit Card Fraud Detection.
4 pages: Project Overview · Explore the Data · Model Results · How I Built This
"""
from __future__ import annotations
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"

# ── Colour palette ─────────────────────────────────────────────────────────────
FRAUD_COLOR = "#EF4444"
LEGIT_COLOR = "#3B82F6"
ACCENT = "#F59E0B"
TMPL = "plotly_dark"

TOP_SIGNALS = {"V14", "V17", "V12", "V10", "high_signal_magnitude", "amount_pca_risk"}

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(page_title="Credit Card Fraud Detection | ML Portfolio",
                   page_icon="💳🛡️", layout="wide", initial_sidebar_state="expanded")

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Layout ── */
.stApp { background-color: #0F172A; color: #F1F5F9; }
[data-testid="stSidebar"] { background-color: #1E293B; border-right: 1px solid #334155; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
  background: #1E293B;
  border: 1px solid #334155;
  border-radius: 12px;
  padding: 18px 20px;
}
[data-testid="metric-container"] label {
  color: #64748B !important;
  font-size: 0.72rem !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 600;
}
[data-testid="stMetricValue"] { color: #F1F5F9 !important; font-size: 1.7rem !important; font-weight: 800 !important; }
[data-testid="stMetricDelta"]  { font-size: 0.8rem !important; }

/* ── Section headers ── */
.sh {
  font-size: 1.25rem; font-weight: 700; color: #F1F5F9;
  border-left: 4px solid #F59E0B;
  padding-left: 12px; margin: 28px 0 12px 0;
}
.sh-sub { color: #64748B; font-size: 0.85rem; margin-top: -10px; margin-bottom: 16px; font-weight: 400; }

/* ── Cards ── */
.card {
  background: #1E293B; border: 1px solid #334155;
  border-radius: 12px; padding: 20px; margin: 6px 0;
}

/* ── Callouts ── */
.callout {
  background: #1E293B; border-left: 4px solid #F59E0B;
  border-radius: 0 10px 10px 0; padding: 12px 16px; margin: 8px 0;
  color: #CBD5E1; font-size: 0.875rem; line-height: 1.65;
}
.cb-blue  { border-left-color: #3B82F6; }
.cb-green { border-left-color: #10B981; }
.cb-red   { border-left-color: #EF4444; }

/* ── Badges ── */
.badge {
  display: inline-block; background: #0F172A; color: #94A3B8;
  border: 1px solid #334155; border-radius: 20px;
  padding: 3px 12px; font-size: 0.78rem; margin: 3px; font-family: monospace;
}

/* ── Hero ── */
.hero-title {
  font-size: 2.8rem; font-weight: 800; line-height: 1.1;
  background: linear-gradient(135deg, #F1F5F9 30%, #F59E0B);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; margin-bottom: 0.5rem;
}
.hero-sub { font-size: 1.1rem; color: #64748B; margin-bottom: 2rem; line-height: 1.7; }

/* ── Timeline ── */
.tl { border-left: 2px solid #F59E0B; padding-left: 16px; margin-bottom: 22px; }
.tl-day  { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; }
.tl-title{ font-weight: 700; color: #E2E8F0; margin: 3px 0 5px 0; font-size: 0.92rem; }
.tl-desc { color: #94A3B8; font-size: 0.92rem; line-height: 1.6; }

/* ── Footer ── */
.footer {
  text-align: center; color: #334155; font-size: 0.78rem;
  padding: 28px 0 8px 0; border-top: 1px solid #1E293B; margin-top: 48px;
}
.footer a { color: #F59E0B; text-decoration: none; }

/* ── Prediction result box ── */
.pred-box {
  border-radius: 14px; padding: 22px; text-align: center;
  border-width: 2px; border-style: solid;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
div[data-testid="stDecoration"] { display: none; }
header[data-testid="stHeader"] { background-color: #0F172A !important; border-bottom: none !important; }

/* ── Sidebar radio ── */
[data-testid="stSidebar"] .stRadio label { color: #CBD5E1 !important; font-size: 0.92rem; }
[data-testid="stSidebar"] .stRadio [data-checked="true"] label { color: #F59E0B !important; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Data / model loaders (all cached)
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading raw data…")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "creditcard.csv")

@st.cache_data(show_spinner="Loading feature data…")
def load_features() -> pd.DataFrame:
    full   = DATA_DIR / "features.csv"
    sample = DATA_DIR / "sample_features.csv"
    return pd.read_csv(full if full.exists() else sample)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    def _read(p: Path) -> dict:
        return json.loads(p.read_text()) if p.exists() else {}
    return {"xgb":    _read(MODELS_DIR / "XGBoost_metrics.json"),
            "tuned":  _read(MODELS_DIR / "tuned_model_metrics.json"),
            "params": _read(MODELS_DIR / "best_params.json")}


@st.cache_resource(show_spinner="Loading production model…")
def load_model():
    path = MODELS_DIR / "production_model.pkl"
    return joblib.load(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def feature_importances(_model, feature_cols: tuple) -> pd.DataFrame:
    if _model is None or not hasattr(_model, "feature_importances_"):
        return pd.DataFrame()
    return (pd.DataFrame({"feature": list(feature_cols),
                          "importance": _model.feature_importances_})
              .sort_values("importance", ascending=False)
              .reset_index(drop=True))


@st.cache_data(show_spinner="Computing confusion matrix on held-out test set…")
def get_confusion(_model, csv_hash: str) -> np.ndarray:
    if _model is None:
        return np.zeros((2, 2), dtype=int)
    df = load_features()
    X = df.drop(columns=["Class"])
    y = df["Class"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return confusion_matrix(y_test, _model.predict(X_test))


# ═════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═════════════════════════════════════════════════════════════════════════════

def section(title: str, subtitle: str = "") -> None:
    st.markdown(f"<div class='sh'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p class='sh-sub'>{subtitle}</p>", unsafe_allow_html=True)


def callout(body: str, variant: str = "amber", icon: str = "💡") -> None:
    cls = {"blue": "cb-blue", "green": "cb-green", "red": "cb-red"}.get(variant, "")
    st.markdown(f"<div class='callout {cls}'><span style='margin-right:8px'>{icon}</span>{body}</div>",
                unsafe_allow_html=True)


def chart_layout(fig, height: int = 360, margins: tuple = (20, 20, 40, 20)) -> go.Figure:
    t, b, l, r = margins
    fig.update_layout(
        template=TMPL,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(t=t, b=b, l=l, r=r),
        font=dict(family="Inter, Helvetica, sans-serif"),
    )
    return fig


def footer() -> None:
    st.markdown(
        "<div class='footer'>Built with Python · XGBoost · MLflow · Streamlit &nbsp;|&nbsp;"
        "<a href='https://github.com/lynwainaina/credit-card-fraud-detection' target='_blank'>github.com/lynwainaina/credit-card-fraud-detection ↗</a></div>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 28px 0;'>
      <div style='font-size:2.8rem; margin-bottom:6px;'>💳⚠️</div>
      <div style='font-weight:800; font-size:1.2rem; color:#F1F5F9; letter-spacing:0.02em;'>Credit Card Fraud Detection</div>
      <div style='font-size:1.22rem; color:#F1F5F9; margin-top:3px;'>ML Portfolio </div>
    </div>
    """, unsafe_allow_html=True)
    page = st.radio("Navigate", ["📊  :blue[Project Summary]", "📈  :blue[EDA]", "🖥️  :blue[Model Results]",   "🛠️  :blue[Model Development Lifecycle]"],
        label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.98rem; color:#64748B; line-height:1.9; padding:4px 0;'>
      <span style='color:#F1F5F9; font-weight:600; display:block; margin-bottom:4px;'>DATASET</span>
      284,807 transactions<br>
      492 fraudulent transactions (0.172%)<br>
      <br>
      <span style='color:#F1F5F9; font-weight:600; display:block; margin-bottom:4px;'>PRODUCTION MODEL</span>
      XGBoost + Optuna<br>
      AUC-ROC: 0.9775<br>
      Recall: &nbsp; 82.1%
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Project Overview
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊  :blue[Project Summary]":
    metrics = load_metrics()
    tuned   = metrics["tuned"]
    auc     = tuned.get("auc_roc", 0.9775)
    recall  = tuned.get("recall",  0.8211)
    prec    = tuned.get("precision", 0.4937)

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='padding:28px 0 6px 0;'>
      <div class='hero-title'>Credit Card Fraud Detection</div>
      <div class='hero-sub'>
        An end-to-end machine learning pipeline that identifies fraudulent transactions with
        <strong style='color:white;'>97.75% AUC-ROC</strong> on 284,807 real-world credit card
        transactions—one of the most class-imbalanced problems in production ML.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── What this project does ─────────────────────────────────────────────────
    st.markdown("""
    <div class='card'>
      <div style='font-size:0.88rem; text-transform:uppercase; letter-spacing:0.08em;
                  color:white; font-weight:700; margin-bottom:10px;'>Project Preview</div>
      <p style='color:#64748B; font-weight:600; line-height:1.85; margin:0; font-size:0.99rem;'>
        This project tackles one of the hardest class-imbalance problems in finance. 
        492 fraudulent transactions are located while buried inside 284,807 total credit card transactions. 
        The full ML lifecycle is covered: data validation with a 5-point quality gate, domain-informed feature engineering (11 new
        features on top of 28 PCA components), systematic model comparison across three algorithms,
        and Optuna-driven hyperparameter tuning — all tracked in MLflow for complete experiment
        reproducibility. The production model is served via FastAPI and this interactive Streamlit
        dashboard.
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    section("Key Results")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(":yellow[Transactions Analysed]", "284,807",  "full dataset")
    k2.metric(":yellow[Features Engineered]",   "39",       "+11 domain features")
    k3.metric(":yellow[AUC-ROC]",               f"{auc:.4f}", "production model")
    k4.metric(":yellow[Fraud Recall]",          f"{recall:.1%}", "+5.4 pts vs XGBoost default")
    k5.metric(":yellow[Precision]",             f"{prec:.1%}", "tuned XGBoost")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two charts ────────────────────────────────────────────────────────────
    col_pie, col_bar = st.columns(2)
    with col_pie:
        section("Class Distribution")
        fig_pie = go.Figure(go.Pie(
            labels=["Legitimate", "Fraudulent"],
            values=[284315, 492],
            hole=0.62,
            marker_colors=[LEGIT_COLOR, FRAUD_COLOR],
            textinfo="label+percent",
            textfont_size=13,
            pull=[0, 0.06],
        ))
        fig_pie.update_layout(
            template=TMPL, paper_bgcolor="rgba(0,0,0,0)", height=300,
            margin=dict(t=10, b=10, l=0, r=0), showlegend=False,
            annotations=[dict(text="<b>0.17%</b><br>fraud rate",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(size=14, color="#F1F5F9"))],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        section("Model Performance Metrics")
        snap = pd.DataFrame({
            "Model":   ["Logistic Reg.", "Random Forest", "XGBoost",    "XGBoost+Optuna"],
            "AUC-ROC": [0.951,           0.964,           0.967,         0.9775],
            "Recall":  [0.600,           0.748,           0.768,         0.821],
            "F1":      [0.708,           0.814,           0.849,         0.617]
            })
        fig_snap = go.Figure()
        for metric, color in zip(["AUC-ROC", "Recall", "F1"],
                                  [LEGIT_COLOR, "#48b5c4", "#d7e1ee"]):
            fig_snap.add_trace(go.Bar(name=metric, x=snap["Model"], y=snap[metric],
                                      marker_color=color, opacity=0.85))
        chart_layout(fig_snap, height=300, margins=(30, 10, 0, 10))
        fig_snap.update_layout(
            barmode="group",
            legend_font_color="white",
            yaxis=dict(range=[0.5, 1.0], gridcolor="#1E293B"),
            legend=dict(orientation="h", y=1.12, x=0))
        st.plotly_chart(fig_snap, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tech stack ────────────────────────────────────────────────────────────
    section("Tech Stack")
    tc1, tc2, tc3, tc4 = st.columns(4)
    stacks = [("Data Processing & ML", ["Python", "Cursor","Pandas", "NumPy", "XGBoost", "Scikit-learn", "Random Forest"]),
              ("Data Quality & Version Control", ["Rule-based Data Quality", "Pytest", "Git"]),
        ("Fine-tuning & Tracking",  ["Optuna", "Joblib", "MLflow", "StratifiedKFold"]),
        ("Serving Layer & Visualization",  ["FastAPI", "Streamlit", "Pydantic", "Uvicorn"])]
        
    for col, (group, items) in zip([tc1, tc2, tc3, tc4], stacks):
        badges = "".join(f"<span class='badge'>{it}</span>" for it in items)
        col.markdown(f"<div class='card' style='min-height:150px;'>"
            f"<div style='font-size:0.8rem; text-transform:uppercase; letter-spacing:0.09em;"
            f" color:#FFFFFF; font-weight:700; margin-bottom:10px;'>{group}</div>"
            f"{badges}</div>",
            unsafe_allow_html=True)
    footer()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Explore the Data
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈  :blue[EDA]":
    df = load_features()
    fraud_df = df[df["Class"] == 1]
    legit_df = df[df["Class"] == 0]
    st.markdown("""
    <div style='padding:28px 0 6px 0;'>
      <h1 style='color:#F1F5F9; font-weight:800; margin-bottom:4px;'>Exploratory & Data Analysis</h1>
      <p style='color:#64748B; margin:0;'>Interactive visualisations from the exploratory data analysis phase.</p>
    </div>
    """, unsafe_allow_html=True)
    # ── Summary metrics ───────────────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4)
    s1.metric(":yellow[Total Rows]",       f"{len(df):,}")
    s2.metric(":yellow[Fraud Cases]",      f"{int(df['Class'].sum()):,}")
    s3.metric(":yellow[Fraud Rate]",       f"{df['Class'].mean():.3%}")
    s4.metric(":yellow[Feature Columns]",  f"{df.shape[1] - 1}")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Target distribution ───────────────────────────────────────────────────
    section("Target Variable Distribution",
            "Severe class imbalance — 492 frauds hidden among 284,807 transactions")

    td_left, td_right = st.columns([1, 1])

    with td_left:
        counts = df["Class"].value_counts().rename({0: "Legitimate", 1: "Fraud"})
        fig_td = go.Figure(go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=[LEGIT_COLOR, FRAUD_COLOR],
            text=[f"{v:,}" for v in counts.values],
            textposition="outside",
            width=0.4,
        ))
        chart_layout(fig_td, height=340, margins=(30, 20, 40, 20))
        fig_td.update_layout(
            yaxis=dict(type="log", title="Transaction Count"
                       , gridcolor="#1E293B", tickfont = dict(size=13)),
            xaxis=dict(gridcolor="#1E293B", tickfont = dict(size=14)),
            showlegend=False,
        )
        st.plotly_chart(fig_td, use_container_width=True)

    with td_right:
        st.markdown("<br>", unsafe_allow_html=True)
        callout(
            "The 492 vs 284,315 (1:578) ratio is one of the most extreme class imbalances in "
            "public fraud datasets. A naive model labels <em>everything</em> as legitimate and "
            "achieves <strong>99.83% accuracy</strong> — making accuracy the wrong metric for performance evaluation.",
            "red",
        )
        callout(
            "We counter imbalance with <strong>scale_pos_weight&nbsp;≈&nbsp;599</strong> in XGBoost "
            "(ratio of negatives to positives) so the model treats each fraud as worth 599 "
            "legitimate samples during training.",
            "blue",
        )
        callout(
            "Primary metric is <strong>AUC-ROC</strong> (threshold-invariant, imbalance-robust) "
            "with <strong>Recall</strong> as secondary — a missed fraud costs far more than "
            "a false alarm in any real fraud team.",
            "green",
        )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature distribution explorer ─────────────────────────────────────────
    section("Feature Distribution Explorer",
            "Compare how each feature separates fraudulent transactions from legitimate transactions")

    pca_feats = [f"V{i}" for i in range(1, 29)]
    eng_feats  = [f for f in ["log_amount", "hour_of_day", "pca_magnitude",
                               "high_signal_magnitude", "pca_extreme_count",
                               "amount_pca_risk", "v14_v17_product"]
                  if f in df.columns]
    base_feats = ["Time", "Amount"]

    tab_pca, tab_eng, tab_raw = st.tabs(["PCA Components (V1–V28)", "Engineered Features", "Raw Features"])

    def dist_plot(feature: str, n_legit: int = 4000) -> go.Figure:
        fv = fraud_df[feature].dropna()
        lv = legit_df[feature].dropna().sample(
            min(n_legit, len(legit_df)), random_state=42
        )
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=lv, name="Legitimate", marker_color=LEGIT_COLOR,
            opacity=0.55, nbinsx=60, histnorm="probability density",
        ))
        fig.add_trace(go.Histogram(
            x=fv, name="Fraud", marker_color=FRAUD_COLOR,
            opacity=0.90, nbinsx=60, histnorm="probability density",
        ))
        chart_layout(fig, height=360, margins=(30, 20, 40, 20))
        fig.update_layout(
            barmode="overlay",
            xaxis_title=feature,
            yaxis_title="Density",
            legend=dict(orientation="h", y=1.08, x=0, font=dict(color = "#64748B", size=15)),
            yaxis=dict(gridcolor="#1E293B", tickfont = dict(size=13)),
            xaxis=dict(gridcolor="#1E293B", tickfont = dict(size=13)),
        )
        return fig

    with tab_pca:
        sel_v = st.selectbox(
            "Select PCA component", pca_feats, index=13, key="sel_v"
        )  # V14 = index 13
        st.plotly_chart(dist_plot(sel_v), use_container_width=True)
        if sel_v in {"V14", "V17", "V12", "V10"}:
            callout(
                f"<strong>{sel_v}</strong> is a top-4 fraud signal (|Pearson corr| &gt; 0.3). "
                "Fraud transactions cluster at extreme values, producing the separated "
                "distributions visible above.",
                "amber",
            )

    with tab_eng:
        sel_eng = st.selectbox("Select engineered feature", eng_feats, key="sel_eng")
        st.plotly_chart(dist_plot(sel_eng), use_container_width=True)
        eng_notes = {
            "log_amount":           "Log1p transform stabilises the right-skewed Amount distribution for tree splits.",
            "high_signal_magnitude":"Euclidean distance across V14, V17, V12, V10 — a single fraud-risk score.",
            "pca_extreme_count":    "Count of PCA dimensions where |V| > 3 — a global outlier detector.",
            "amount_pca_risk":      "Interaction of log_amount × high_signal_magnitude — high-value unusual transactions.",
            "hour_of_day":          "Fraud peaks strongly in late-night hours (midnight–6 AM).",
            "v14_v17_product":      "V14 × V17 product interaction — captures joint extreme behaviour.",
        }
        if sel_eng in eng_notes:
            callout(f"<strong>{sel_eng}:</strong> {eng_notes[sel_eng]}", "blue", "💡")

    with tab_raw:
        sel_raw = st.selectbox("Select raw feature", base_feats, key="sel_raw")
        st.plotly_chart(dist_plot(sel_raw), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Correlation with target ────────────────────────────────────────────────
    section("Feature Correlation with Fraud (Class)",
            "Top 7 positive and top 7 negative Pearson correlations with the target")

    num_cols  = [c for c in df.columns if c != "Class"]
    corr_tgt  = df[num_cols + ["Class"]].corr()["Class"].drop("Class").sort_values()
    top_corr  = pd.concat([corr_tgt.head(7), corr_tgt.tail(7)])

    fig_corr = go.Figure(go.Bar(
        x=top_corr.values,
        y=top_corr.index,
        orientation="h",
        marker_color=[FRAUD_COLOR if v < 0 else LEGIT_COLOR for v in top_corr.values],
        opacity=0.85,
    ))
    chart_layout(fig_corr, height=440, margins=(10, 20, 0, 20))
    fig_corr.update_layout(
        xaxis_title="Features Correlation with Class",
        yaxis=dict(autorange="reversed", gridcolor="#1E293B",tickfont = dict(size=13)),
        xaxis=dict(gridcolor="#1E293B",tickfont = dict(size=13)),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Amount by class ───────────────────────────────────────────────────────
    section("Transaction Amount by Class")
    am_left, am_right = st.columns([1, 1])

    with am_left:
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=legit_df["Amount"].sample(5000, random_state=0),
            name="Legitimate", marker_color=LEGIT_COLOR, boxmean=True,
        ))
        fig_box.add_trace(go.Box(
            y=fraud_df["Amount"],
            name="Fraud", marker_color=FRAUD_COLOR, boxmean=True,
        ))
        chart_layout(fig_box, height=360, margins=(20, 20, 40, 20))
        fig_box.update_layout(
            yaxis_title="Amount (€)",
            yaxis=dict(gridcolor="#1E293B"),
            xaxis=dict(tickfont = dict(size=14)),
            legend=dict(font=dict(color = "#64748B", size=15)),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with am_right:
        st.markdown("<br>", unsafe_allow_html=True)
        callout(
            "Fraudulent transactions skew toward <strong>smaller amounts</strong>. "
            "<br>This is a known pattern i.e. fraudsters make low-value test transactions before "
            "attempting larger ones.",
            "red",
        )
        # callout(
        #     "<strong>log1p(Amount)</strong> normalises the extreme right skew "
        #     "(max €25,691) and improves tree split quality near low-value transactions.",
        #     "green",
        # )
        st.markdown(f"""
        <div class='card'>
          <table style='width:100%; color:#CBD5E1; font-size:0.84rem; border-collapse:separate; border-spacing:0 6px;'>
            <thead>
              <tr>
                <th style='color:#64748B; text-align:left; padding-bottom:6px;'>Statistics</th>
                <th style='color:{LEGIT_COLOR}; text-align:right;'>Legitimate</th>
                <th style='color:{FRAUD_COLOR}; text-align:right;'>Fraud</th>
              </tr>
            </thead>
            <tbody>
              <tr><td>Mean</td>
                  <td style='text-align:right;'>€{legit_df["Amount"].mean():.2f}</td>
                  <td style='text-align:right;'>€{fraud_df["Amount"].mean():.2f}</td></tr>
              <tr><td>Median</td>
                  <td style='text-align:right;'>€{legit_df["Amount"].median():.2f}</td>
                  <td style='text-align:right;'>€{fraud_df["Amount"].median():.2f}</td></tr>
              <tr><td>Maximum</td>
                  <td style='text-align:right;'>€{legit_df["Amount"].max():,.0f}</td>
                  <td style='text-align:right;'>€{fraud_df["Amount"].max():,.0f}</td></tr>
              <tr><td>% under €10</td>
                  <td style='text-align:right;'>{(legit_df["Amount"] < 10).mean():.1%}</td>
                  <td style='text-align:right;'>{(fraud_df["Amount"] < 10).mean():.1%}</td></tr>
            </tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── EDA key findings ──────────────────────────────────────────────────────
    section("Key EDA Findings")
    f1, f2 = st.columns(2)
    with f1:
        callout(
            "<strong>V14, V17, V12, V10</strong> have the highest correlation (corr &gt; 0.3) with fraud."
            "<br>These four components feed into <code>high_signal_magnitude</code> "
            "and <code>amount_pca_risk</code>.",
            "amber",
        )
        callout(
            "<strong>Fraud peaks at night (0–6 AM).</strong> <br>The <code>is_night</code> binary "
            "flag and <code>hour_of_day</code> capture this temporal signal without leaking "
            "raw time seconds.",
            "blue",
        )
        callout(
            "<strong>No missing values</strong> found. <br>The quality gate confirmed schema "
            "integrity, correct dtypes, and valid ranges for Amount and Time across all "
            "284,807 rows.",
            "green",
        )
    with f2:
        callout(
            "<strong>Amount is highly right-skewed</strong> (max €25,691, median €22). "
            "<br>Log-transform stabilises variance and significantly improves tree splits near "
            "small-value transactions.",
            "amber",
        )
        callout(
            "<strong>Fraud transactions are global outliers.</strong> "
            "<br><code>pca_extreme_count</code> (count of PCA dims where |V| &gt; 3) captures this "
            "multi-dimensional anomaly signal in a single interpretable feature.",
            "blue", 
        )
        callout(
            "<strong>Zero features removed</strong> by selection. <br>No pair exceeded the 0.95 "
            "correlation threshold, confirming all 11 engineered features add independent "
            "information.",
            "green", 
        )

    footer()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Results
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🖥️  :blue[Model Results]":
    df    = load_features()
    model = load_model()
    feat_cols = tuple(c for c in df.columns if c != "Class")
    fi_df = feature_importances(model, feat_cols)
    tuned = load_metrics()["tuned"]

    st.markdown("""
    <div style='padding:28px 0 6px 0;'>
      <h1 style='color:#F1F5F9; font-weight:800; margin-bottom:4px;'>Model Results</h1>
      <p style='color:#64748B; margin:0;'>
        We compare every model tried and every metric recorded for the baseline to production level models.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Comparison table ──────────────────────────────────────────────────────
    section("Model Comparison")

    comparison = pd.DataFrame([
        {"Model": "Logistic Regression",  "Role": "Baseline",   "AUC-ROC": 0.9510,
         "Recall": 0.600, "Precision": 0.870, "F1": 0.708, "Note": "Reference floor"},
        {"Model": "Random Forest",        "Role": "Candidate 1",  "AUC-ROC": 0.9642,
         "Recall": 0.748, "Precision": 0.892, "F1": 0.814, "Note": "🔶 Good · slow to train"},
        {"Model": "XGBoost (default)",    "Role": "Candidate 2",  "AUC-ROC": 0.9668,
         "Recall": 0.768, "Precision": 0.919, "F1": 0.849, "Note": "🔶 Strong baseline"},
        {"Model": "XGBoost + Optuna",     "Role": "Winner",  "AUC-ROC": 0.9775,
         "Recall": 0.821, "Precision": 0.494, "F1": 0.617, "Note": "✅ Production model"},
    ])

    def row_style(row):
        if "Winner" in row["Role"]:
            return ["background-color:#0f2d1f; color:#6ee7b7; font-weight:600"] * len(row)
        if row["Role"] == "Baseline":
            return ["background-color:#1e293b; color:#64748b"] * len(row)
        return ["background-color:#1e293b; color:#e2e8f0"] * len(row)

    styled_df = (
        comparison.style
        .apply(row_style, axis=1)
        .format({"AUC-ROC": "{:.4f}", "Recall": "{:.3f}",
                 "Precision": "{:.3f}", "F1": "{:.3f}"})
        .hide(axis="index")
    )
    st.dataframe(styled_df, use_container_width=True, height=190)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Why the winner ─────────────────────────────────────────────────────────
    section("Why XGBoost + Optuna Is the Production Model")
    w1, w2, w3 = st.columns(3)
    for col, (title, color, body) in zip([w1, w2, w3], [
        ("📈  Best Fraud Recall", FRAUD_COLOR,
         "Catches <strong>82.1%</strong> of all fraud i.e. 5.4 points above the XGBoost default and "
         "22 points above LogisticRegression. In fraud detection a missed fraud means real financial "
         "loss, so recall is the primary operational target."),
        ("🔍  Highest AUC-ROC", LEGIT_COLOR,
         "AUC-ROC <strong>0.9775</strong> is threshold-independent. It ranks fraud above legitimate "
         "transactions at <em>any</em> operating point. This is a metric can be used when "
         "comparing two systems side by side in fraud analysis."),
        ("⚡  Optuna Advantage", ACCENT,
         "30 TPE-sampler trials searched a 9-dimensional space. The decisive find: "
         "<strong>scale_pos_weight&nbsp;≈&nbsp;599</strong> and <strong>min_child_weight&nbsp;=&nbsp;20</strong> "
         "together suppress noisy splits on the minority class."),
    ]):
        col.markdown(
            f"<div class='card' style='border-top:3px solid {color}; min-height:160px;'>"
            f"<div style='color:{color}; font-weight:700; font-size:0.78rem; text-transform:uppercase; "
            f"letter-spacing:0.07em; margin-bottom:10px;'>{title}</div>"
            f"<div style='color:#CBD5E1; font-size:0.87rem; line-height:1.7;'>{body}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature importance + confusion matrix ─────────────────────────────────
    fi_col, cm_col = st.columns([3, 2])

    with fi_col:
        section("Feature Importance — Top 15",
                "XGBoost gain-based importance (amber = engineered / top-signal features)")
        if not fi_df.empty:
            top15 = fi_df.head(15).sort_values("importance")
            bar_colors = [
                ACCENT if f in TOP_SIGNALS else  "#008080"
                for f in top15["feature"]
            ]
            fig_fi = go.Figure(go.Bar(
                x=top15["importance"],
                y=top15["feature"],
                orientation="h",
                marker_color=bar_colors,
                opacity=0.9,
            ))
            chart_layout(fig_fi, height=440, margins=(10, 20, 0, 20))
            fig_fi.update_layout(
                xaxis_title="Feature Importance (Gain)",
                yaxis=dict(tickfont=dict(size=12), gridcolor="#1E293B"),
                xaxis=dict(gridcolor="#1E293B"),
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            callout(
                "<strong>Amber bars</strong> confirm engineered features add real signal: "
                "<br><code>high_signal_magnitude</code> and <code>amount_pca_risk</code> "
                "consistently rank above raw PCA components not in the top-4.",
                "amber",
            )
        else:
            st.info("Feature importances unavailable — model file not found.")

    with cm_col:
        section("Confusion Matrix", "Stratified 20% hold-out (56,962 rows)")
        cm = get_confusion(model, "v1")

        if cm.sum() > 0:
            tn, fp, fn, tp = cm.ravel()
            fig_cm = go.Figure(go.Heatmap(
                z=[[tn, fp], [fn, tp]],
                x=["Pred: Legit", "Pred: Fraud"],
                y=["True: Legit", "True: Fraud"],
                colorscale=[[0.0, "#0F172A"], [0.4, "#1E3A5F"], [1.0, LEGIT_COLOR]],
                showscale=False,
                text=[[f"TN<br><b>{tn:,}</b>", f"FP<br><b>{fp:,}</b>"],
                      [f"FN<br><b>{fn:,}</b>", f"TP<br><b>{tp:,}</b>"]],
                texttemplate="%{text}",
                textfont=dict(size=16, color="white"),
            ))
            chart_layout(fig_cm, height=320, margins=(10, 10, 0, 10))
            st.plotly_chart(fig_cm, use_container_width=True)
            cm1, cm2 = st.columns(2)
            cm1.metric(":yellow[Fraud Caught (TP)]", f"{tp:,}", "true positives")
            cm2.metric(":yellow[Missed Fraud (FN)]", f"{fn:,}", "false negatives", delta_color="inverse")
        else:
            callout("Run the training pipeline to generate `production_model.pkl`.", "red", "⚠️")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Interactive prediction ─────────────────────────────────────────────────
    section("Try It Yourself",
            "Adjust the key features and get a real-time fraud probability from the production model")

    callout(
        "All 39 features are required. The sliders control the most influential ones; "
        "the rest are pre-filled with dataset medians (a typical legitimate transaction).",
        "blue", "ℹ️",
    )

    if model is not None:
        df_meds = df.drop(columns=["Class"]).median().to_dict()

        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            p1, p2, p3 = st.columns(3)

            with p1:
                st.markdown("**Top PCA Signals**")
                v14 = st.slider("V14 — strongest fraud signal", -20.0, 10.0,
                                float(df_meds.get("V14", 0.0)), 0.05, key="v14")
                v17 = st.slider("V17", -25.0, 10.0,
                                float(df_meds.get("V17", 0.0)), 0.05, key="v17")

            with p2:
                st.markdown("**PCA Signals (cont.)**")
                v12 = st.slider("V12", -20.0, 10.0,
                                float(df_meds.get("V12", 0.0)), 0.05, key="v12")
                v10 = st.slider("V10", -20.0, 10.0,
                                float(df_meds.get("V10", 0.0)), 0.05, key="v10")

            with p3:
                st.markdown("**Transaction Context**")
                amount = st.number_input("Amount (€)", 0.0, 30_000.0,
                                         float(df_meds.get("Amount", 88.35)),
                                         step=10.0, key="amount")
                hour = st.slider("Hour of Day (0–23)", 0, 23,
                                 int(df_meds.get("hour_of_day", 12)), key="hour")
            st.markdown("</div>", unsafe_allow_html=True)

        # Build input dict from medians then overwrite adjusted features
        inp = dict(df_meds)
        inp.update({
            "V14": v14, "V17": v17, "V12": v12, "V10": v10,
            "Amount": amount,
            "log_amount":            float(np.log1p(amount)),
            "hour_of_day":           float(hour),
            "is_night":              1.0 if hour < 6 else 0.0,
            "is_high_value":         1.0 if amount > 500 else 0.0,
            "is_micropayment":       1.0 if amount < 1 else 0.0,
            "night_high_value":      (1.0 if hour < 6 else 0.0) * (1.0 if amount > 500 else 0.0),
            "v14_v17_product":       v14 * v17,
            "high_signal_magnitude": float((v14**2 + v17**2 + v12**2 + v10**2) ** 0.5),
            "amount_pca_risk":       float(np.log1p(amount) * (v14**2 + v17**2 + v12**2 + v10**2) ** 0.5),
            "pca_extreme_count":     float(sum(1 for k in [f"V{i}" for i in range(1, 29)]
                                               if abs(inp.get(k, 0)) > 3)),
            "pca_magnitude":         float(sum(inp.get(f"V{i}", 0)**2 for i in range(1, 29)) ** 0.5),
        })

        X_in = pd.DataFrame([{col: inp.get(col, 0.0) for col in feat_cols}])
        prob = float(model.predict_proba(X_in)[0][1])
        pred = int(model.predict(X_in)[0])

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns([2, 1, 1])

        with r1:
            label  = "🚨  FRAUD DETECTED"     if pred == 1 else "✅  LEGITIMATE"
            color  = FRAUD_COLOR               if pred == 1 else "#10B981"
            bg     = f"{color}18"
            st.markdown(
                f"<div style='background:{bg}; border:2px solid {color}; border-radius:14px; "
                f"padding:24px; text-align:center;'>"
                f"<div style='font-size:1.7rem; font-weight:800; color:{color};'>{label}</div>"
                f"<div style='color:#64748B; font-size:0.82rem; margin-top:6px;'>Model Prediction</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with r2:
            st.metric("Fraud Probability", f"{prob:.1%}")
        with r3:
            risk  = "High"   if prob > 0.7 else ("Medium" if prob > 0.3 else "Low")
            rc    = FRAUD_COLOR if risk == "High" else (ACCENT if risk == "Medium" else "#10B981")
            st.metric("Risk Level", risk)
            st.markdown(
                f"<div style='width:100%; height:6px; background:{rc}; border-radius:4px; margin-top:-8px;'></div>",
                unsafe_allow_html=True,
            )

        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number=dict(suffix="%", font=dict(size=24, color="#F1F5F9")),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#334155",
                          tickfont=dict(color="#64748B", size=10)),
                bar=dict(color=FRAUD_COLOR if prob > 0.5 else "#10B981", thickness=0.25),
                bgcolor="#1E293B",
                borderwidth=0,
                steps=[
                    dict(range=[0, 30],  color="#0a2a1a"),
                    dict(range=[30, 70], color="#2d1e00"),
                    dict(range=[70, 100], color="#2d0a0a"),
                ],
                threshold=dict(
                    line=dict(color=FRAUD_COLOR, width=2),
                    thickness=0.75, value=50,
                ),
            ),
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=200,
            margin=dict(t=20, b=10, l=30, r=30),
            font=dict(color="#F1F5F9"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    else:
        st.warning(
            "Production model not found at `models/production_model.pkl`. "
            "Run `python -m src.models.run_training` to generate it."
        )

    footer()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — How I Built This
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🛠️  :blue[Model Development Lifecycle]":
    st.markdown("""
    <div style='padding:28px 0 6px 0;'>
      <h1 style='color:#F1F5F9; font-weight:800; margin-bottom:4px;'>How I Built This</h1>
      <p style='color:#64748B; margin:0;'>Architecture, design decisions, and lessons learned from
      building a production-grade ML system.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Architecture diagram ───────────────────────────────────────────────────
    section("System Architecture", "Layered pipeline: raw data → feature engineering → model training → serving")

    st.graphviz_chart("""
    digraph pipeline {
      rankdir=LR
      bgcolor="transparent"
      node [style=filled fontname="Helvetica" fontsize=10 fontcolor="white"
            margin="0.18,0.08" shape=box]
      edge [color="#475569" fontcolor="#64748B" fontsize=9 arrowsize=0.7]

      subgraph cluster_data {
        label="  Data Layer  " fontcolor="#64748B" color="#334155" style=dashed
        fontsize=9
        A [label="credit card csv file\l284,807 rows" fillcolor="#1e3a8a" shape=cylinder]
        B [label="5-Point\lQuality Gate"         fillcolor="#1e40af" shape=diamond]
        C [label="Data Cleaning & Validation\l(dedup · data types validation)"     fillcolor="#1e3a8a"]
      }

      subgraph cluster_feat {
        label="  Feature Engineering Layer  " fontcolor="#64748B" color="#334155" style=dashed
        fontsize=9
        D [label="Engineering\l+11 features"     fillcolor="#7c2d12"]
        E [label="Selection\l(correlation · variance)"  fillcolor="#7c2d12"]
        F [label="features csv file\l39 features"     fillcolor="#9a3412" shape=cylinder]
      }

      subgraph cluster_model {
        label="  Model Development & Tracking Layer  " fontcolor="#64748B" color="#334155" style=dashed
        fontsize=9
        G [label="Logistic\lRegression"          fillcolor="#14532d"]
        H [label="Random\lForest"                fillcolor="#14532d"]
        I [label="XGBoost\ldefault"              fillcolor="#14532d"]
        J [label="Optuna\l30 trials"             fillcolor="#065f46"]
        K [label="MLflow\nTracking"              fillcolor="#064e3b" shape=cylinder]
      }

      subgraph cluster_serve {
        label="  Serving  Layer" fontcolor="#64748B" color="#334155" style=dashed
        fontsize=9
        L [label="FastAPI\nREST /predict"        fillcolor="#4c1d95"]
        M [label="Streamlit\nDashboard"          fillcolor="#3730a3"]
      }

      A -> B -> C -> D -> E -> F
      F -> G -> K
      F -> H -> K
      F -> I -> J -> K
      K -> L
      K -> M
    }
    """, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Build timeline ─────────────────────────────────────────────────────────
    section("Build Timeline", "Seven-day end-to-end build")

    tl_left, tl_right = st.columns(2)

    timeline = [
        ("Day 1", "Data Ingestion & Quality Gate", "#3B82F6",
         "Built data loader, cleaner (dedup · dtype coercion · null handling), and 5-point "
         "quality gate (schema · row count · null rates · value ranges · target distribution). "
         "Confirmed the 0.172% fraud rate survives the pipeline intact."),
        ("Day 2", "Exploratory Data Analysis", "#8B5CF6",
         "Analysed 28 PCA components and raw features for correlation with Class. Identified "
         "V14, V17, V12, V10 as top signals. Documented amount distribution, temporal fraud "
         "patterns, and the severity of class imbalance."),
        ("Day 3", "Feature Engineering & Model Selection", ACCENT,
         "Engineered 11 features across three categories: temporal (hour_of_day, is_night), "
         "amount-based (log_amount, is_micropayment, is_high_value), and PCA-derived "
         "(pca_magnitude, high_signal_magnitude,..)."
         "Trained LogisticRegression baseline (AUC 0.951). Compared RandomForest vs XGBoost "
         "with stratified 5-fold CV. XGBoost won on AUC (0.967) and Recall (76.8%). "
         "StratifiedKFold used throughout to preserve the 0.17% fraud ratio."),
        ("Day 4", "Hyperparameter Tuning and Experiment Tracking", "#10B981",
         "30 Optuna trials with TPE sampler over 9 XGBoost hyperparameters. "
         "Most impactful: scale_pos_weight ≈ 599 forces the model to weight each fraud "
         "as 599 legitimate samples. Final Recall: 82.1%."
         "Configured file-based MLflow tracking. Both baseline and tuned_best runs logged "
         "with full hyperparameters and train/test metrics. Production model promoted from "
         "the tuned artifact. All runs are reproducible from the MLflow UI."),
        ("Day 5", "Serving & Dashboard", FRAUD_COLOR,
         "Built a FastAPI REST endpoint (/predict) with Pydantic input validation. "
         "Developed this 4-page Streamlit portfolio dashboard: interactive EDA charts, "
         "model comparison table, live confusion matrix, feature importance plot, "
         "and a real-time fraud probability predictor with slider controls."),
        ("Day 6", "Testing & Docker", "#6366F1",
         "Wrote pytest unit tests covering the data quality gate, feature engineering "
         "transformations, and model prediction logic. Containerised the full stack "
         "(FastAPI + Streamlit) with Docker and Docker Compose for reproducible deployment."),
        ("Day 7", "Deployment & Release", "#EC4899",
         "Shipped the final project to GitHub with a clean commit history and README. "
         "Deployed the Streamlit dashboard to Streamlit Community Cloud and verified "
         "the live app end-to-end."),
        ]

    for i, (day, title, color, desc) in enumerate(timeline):
        col = tl_left if i % 2 == 0 else tl_right
        with col:
            col.markdown(
                f"<div class='tl' style='border-left-color:{color};'>"
                f"<div class='tl-day' style='color:{color};'>{day}</div>"
                f"<div class='tl-title'>{title}</div>"
                f"<div class='tl-desc'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key decisions ──────────────────────────────────────────────────────────
    section("Key Design Decisions & Lessons Learned")
    d1, d2 = st.columns(2)

    with d1:
        callout(
            "<strong>Decision: AUC-ROC as primary metric.</strong> "
            "<br>Accuracy is deceptive at 0.17% fraud rate translating to 99.83% accuracy with zero fraud caught. "
            "AUC-ROC is threshold-independent and aligns with how fraud systems are evaluated operationally.",
            "amber", "🎯",
        )
        callout(
            "<strong>Decision: StratifiedKFold CV throughout.</strong> "
            "<br>A random split can produce folds with zero fraud cases, making CV scores meaningless. "
            "Stratified folds guarantee the 0.17% ratio is preserved in every fold.",
            "blue", "📊",
        )
        callout(
            "<strong>Lesson: class weighting matters more than architecture.</strong> "
            "<br>The gain from default XGBoost → tuned XGBoost (+5.4% recall) eclipsed the gain from "
            "RandomForest → XGBoost (+2%). Getting scale_pos_weight right was the decisive step.",
            "green", "🔑",
        )

    with d2:
        callout(
            "<strong>Decision: log1p(Amount) over raw Amount.</strong> "
            "<br>Raw Amount is right-skewed (max €25,691). Log-transform stabilises variance and "
            "dramatically improves tree split quality near small-value transactions where "
            "test fraud is concentrated.",
            "amber", "📐",
        )
        callout(
            "<strong>Decision: src/ package over notebooks.</strong> "
            "<br>Production logic lives in importable modules with unit tests. Notebooks are "
            "exploration only. This prevents train-serve skew — FastAPI imports the same "
            "feature engineering code used during training.",
            "blue", "🏗️",
        )
        callout(
            "<strong>Lesson: precision–recall trade-off is explicit.</strong> "
            "<br>The tuned model lowered precision (49.4%) to gain recall (82.1%). For a fraud team "
            "more alerts are operationally worthwhile if the alternative is missed fraud — "
            "this is a business decision, not a model failure.",
            "green", "⚖️",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── GitHub ─────────────────────────────────────────────────────────────────
    section("Source Code")
    st.markdown("""
    <div class='card' style='display:flex; align-items:center; gap:24px; flex-wrap:wrap;'>
      <div style='font-size:2.8rem;'>🐙</div>
      <div style='flex:1; min-width:220px;'>
        <div style='font-weight:700; color:#F1F5F9; font-size:0.98rem;'>
          Full project on GitHub
        </div>
        <div style='color:#64748B; font-size:0.84rem; margin-top:4px; line-height:1.6;'>
          Complete source code — data pipeline, feature engineering, model training scripts,
          Optuna tuning, MLflow experiment logs, FastAPI endpoint, and Streamlit dashboard.
        </div>
        <a href='https://github.com/lynwainaina/credit-card-fraud-detection' target='_blank'
           style='display:inline-block; margin-top:12px; padding:9px 22px;
                  background:#F59E0B; color:#0F172A; border-radius:8px;
                  font-weight:700; text-decoration:none; font-size:0.84rem;
                  letter-spacing:0.02em;'>
          github.com/lynwainaina/credit-card-fraud-detection ↗
        </a>
      </div>
    </div>
    """, unsafe_allow_html=True)

    footer()
