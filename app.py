"""
app.py — Fake News Detector Streamlit Web Application
======================================================
...
"""

from __future__ import annotations
from typing import Optional, List, Dict, Tuple

import sys
import pathlib
import json
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
SRC_DIR      = PROJECT_ROOT / "src"
PLOTS_DIR    = PROJECT_ROOT / "plots"
MODELS_DIR   = PROJECT_ROOT / "models"
DATA_DIR     = PROJECT_ROOT / "data"

sys.path.insert(0, str(SRC_DIR))

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector | AI-Powered NLP",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Google Font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Hero header */
  .hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #E63946 0%, #457B9D 50%, #2DC653 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    line-height: 1.1;
  }
  .hero-subtitle {
    font-size: 1.1rem;
    color: #6B7280;
    margin-top: 0.3rem;
  }
  .badge {
    display: inline-block;
    background: linear-gradient(135deg, #1e3a5f, #457B9D);
    color: white;
    padding: 4px 14px;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-top: 8px;
  }

  /* Verdict banners */
  .verdict-fake {
    background: linear-gradient(135deg, #E63946, #c1121f);
    color: white;
    padding: 28px 32px;
    border-radius: 16px;
    text-align: center;
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: 2px;
    box-shadow: 0 8px 30px rgba(230,57,70,0.35);
    animation: pulse-red 2s infinite;
  }
  .verdict-real {
    background: linear-gradient(135deg, #2DC653, #1a7a2e);
    color: white;
    padding: 28px 32px;
    border-radius: 16px;
    text-align: center;
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: 2px;
    box-shadow: 0 8px 30px rgba(45,198,83,0.35);
  }

  @keyframes pulse-red {
    0%   { box-shadow: 0 8px 30px rgba(230,57,70,0.35); }
    50%  { box-shadow: 0 8px 40px rgba(230,57,70,0.55); }
    100% { box-shadow: 0 8px 30px rgba(230,57,70,0.35); }
  }

  /* Metric cards */
  .metric-card {
    background: #1E1E2E;
    border: 1px solid #2d2d44;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
    transition: transform 0.2s;
  }
  .metric-card:hover { transform: translateY(-2px); }
  .metric-card .label {
    font-size: 0.75rem;
    color: #9CA3AF;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }
  .metric-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #F1F5F9;
  }

  /* Word tags */
  .word-tag-fake {
    display: inline-block;
    background: rgba(230,57,70,0.18);
    color: #E63946;
    border: 1px solid rgba(230,57,70,0.4);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 3px;
  }
  .word-tag-real {
    display: inline-block;
    background: rgba(45,198,83,0.18);
    color: #2DC653;
    border: 1px solid rgba(45,198,83,0.4);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 3px;
  }

  /* Section header */
  .section-header {
    font-size: 1.2rem;
    font-weight: 700;
    color: #F1F5F9;
    border-left: 4px solid #457B9D;
    padding-left: 12px;
    margin: 20px 0 12px 0;
  }

  /* Info box */
  .info-box {
    background: rgba(69,123,157,0.12);
    border: 1px solid rgba(69,123,157,0.3);
    border-radius: 10px;
    padding: 14px 18px;
    color: #CBD5E1;
    font-size: 0.9rem;
    line-height: 1.6;
  }

  /* Sidebar override */
  section[data-testid="stSidebar"] {
    background: #0F172A;
  }

  /* Button override */
  .stButton > button {
    background: linear-gradient(135deg, #1e3a5f, #457B9D) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(69,123,157,0.4) !important;
  }

  /* Sample buttons */
  div[data-testid="column"] .stButton > button {
    width: 100%;
  }

  /* Credibility score bar */
  .progress-container {
    width: 100%;
    background: #1E2A3A;
    border-radius: 50px;
    height: 18px;
    overflow: hidden;
    margin: 8px 0;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.3);
  }
  .progress-bar-red   { height: 18px; border-radius: 50px; background: linear-gradient(90deg, #E63946, #c1121f); transition: width 1s ease; }
  .progress-bar-orange{ height: 18px; border-radius: 50px; background: linear-gradient(90deg, #F4A261, #e76f51); transition: width 1s ease; }
  .progress-bar-green { height: 18px; border-radius: 50px; background: linear-gradient(90deg, #2DC653, #1a7a2e); transition: width 1s ease; }

  /* Sample article card */
  .article-card {
    background: #1E1E2E;
    border: 1px solid #2d2d44;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 14px;
    font-size: 0.87rem;
    color: #CBD5E1;
    line-height: 1.6;
  }
  .article-card .card-title {
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 8px;
    font-size: 0.95rem;
  }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING HELPERS (cached)
# =============================================================================

@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame | None:
    """Load model metrics CSV generated by train.py."""
    path = MODELS_DIR / "model_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_stats() -> dict:
    """Load dataset stats JSON generated by train.py."""
    path = MODELS_DIR / "dataset_stats.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_raw_samples() -> Tuple[List[Dict], List[Dict]]:
    """
    Load a handful of real fake/real articles from the CSV files for display.
    Returns (fake_samples, real_samples) — each a list of dicts.
    """
    fake_path = DATA_DIR / "Fake.csv"
    real_path = DATA_DIR / "True.csv"
    fake_samples, real_samples = [], []

    if fake_path.exists():
        df = pd.read_csv(fake_path).dropna(subset=["title", "text"])
        for _, row in df.sample(min(3, len(df)), random_state=42).iterrows():
            fake_samples.append({"title": row["title"], "text": row["text"][:300]})

    if real_path.exists():
        df = pd.read_csv(real_path).dropna(subset=["title", "text"])
        for _, row in df.sample(min(3, len(df)), random_state=42).iterrows():
            real_samples.append({"title": row["title"], "text": row["text"][:300]})

    return fake_samples, real_samples


# =============================================================================
# SAMPLE ARTICLES (hardcoded for demo)
# =============================================================================

FAKE_EXAMPLE_TITLE = "BREAKING: Scientists Confirm Drinking Bleach Cures All Diseases"
FAKE_EXAMPLE_TEXT  = (
    "In a shocking revelation that mainstream media refuses to cover, "
    "a group of underground scientists have confirmed that drinking a small "
    "amount of household bleach every morning can cure cancer, diabetes, and "
    "COVID-19. The government has been suppressing this information for years "
    "to protect Big Pharma profits. Share this before it gets deleted! "
    "Our sources inside the CDC have confirmed this but fear losing their jobs "
    "if they speak out publicly. The elite don't want you to know this truth."
)

REAL_EXAMPLE_TITLE = "Federal Reserve raises interest rates by 0.25 percent"
REAL_EXAMPLE_TEXT  = (
    "The Federal Reserve raised its benchmark interest rate by a quarter "
    "percentage point on Wednesday, continuing its campaign to bring inflation "
    "under control. The decision by the Federal Open Market Committee was "
    "unanimous. Fed Chair stated that the committee remains strongly committed "
    "to returning inflation to its 2 percent objective. The rate increase brings "
    "the federal funds rate to its highest level in 15 years. Economists and "
    "analysts had widely anticipated the move following recent economic data."
)


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar(metrics_df: Optional[pd.DataFrame], stats: dict) -> None:
    """Render the sidebar with about, metrics, and dataset stats."""
    with st.sidebar:
        st.markdown("## 🧠 Fake News Detector")
        st.markdown("---")

        st.markdown("### 📖 About")
        st.markdown("""
This app uses **Logistic Regression** trained on TF-IDF features extracted
from ~44,000 real and fake news articles.

**Pipeline:**
1. Clean & normalise text
2. Vectorise with TF-IDF (10k features)
3. Logistic Regression classification
4. Return verdict + confidence
        """)
        st.markdown("---")

        # Model performance metrics
        st.markdown("### 📊 Model Performance")
        if metrics_df is not None:
            lr_row = metrics_df[metrics_df["model"] == "Logistic Regression"]
            if not lr_row.empty:
                row = lr_row.iloc[0]
                cols = st.columns(2)
                cols[0].metric("Accuracy",  f"{row['accuracy']*100:.2f}%")
                cols[0].metric("Precision", f"{row['precision']*100:.2f}%")
                cols[1].metric("Recall",    f"{row['recall']*100:.2f}%")
                cols[1].metric("F1 Score",  f"{row['f1']*100:.2f}%")
        else:
            st.info("Train the model first to see metrics.")

        st.markdown("---")

        # Dataset stats
        st.markdown("### 📦 Dataset")
        if stats:
            st.markdown(f"**Total Articles:** {int(stats.get('total', 0)):,}")
            st.markdown(f"**FAKE Articles:** {int(stats.get('fake', 0)):,}")
            st.markdown(f"**REAL Articles:** {int(stats.get('real', 0)):,}")
        else:
            st.info("Run train.py to generate stats.")

        st.markdown("---")

        st.markdown("### 🛠️ How to Use")
        st.markdown("""
1. Paste a news article in the **Detector** tab
2. Optionally add the article title
3. Click **Analyze Article**
4. View verdict, confidence score, and key words
        """)

        st.markdown("---")
        st.markdown('<div style="font-size:0.75rem;color:#6B7280;">Built with scikit-learn + NLTK + Streamlit</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 1 — DETECTOR
# =============================================================================

def credibility_bar_html(score: int) -> str:
    """Return HTML for a colored credibility progress bar."""
    if score <= 40:
        bar_class = "progress-bar-red"
    elif score <= 70:
        bar_class = "progress-bar-orange"
    else:
        bar_class = "progress-bar-green"

    return f"""
    <div class="progress-container">
      <div class="{bar_class}" style="width:{score}%;"></div>
    </div>
    <div style="font-size:0.8rem;color:#9CA3AF;text-align:right;">
      Credibility: {score}/100
    </div>
    """


def render_detector_tab() -> None:
    """Render the main article analysis tab."""
    st.markdown('<div class="section-header">📰 Paste Your News Article</div>', unsafe_allow_html=True)

    # Sample article quick-fill buttons
    col1, col2 = st.columns(2)
    if col1.button("🔴 Try Fake News Example", key="btn_fake"):
        st.session_state["article_title"] = FAKE_EXAMPLE_TITLE
        st.session_state["article_text"]  = FAKE_EXAMPLE_TEXT

    if col2.button("🟢 Try Real News Example", key="btn_real"):
        st.session_state["article_title"] = REAL_EXAMPLE_TITLE
        st.session_state["article_text"]  = REAL_EXAMPLE_TEXT

    # Input fields
    article_title = st.text_input(
        "Article Title (optional)",
        value=st.session_state.get("article_title", ""),
        placeholder="Enter the news headline here...",
        key="input_title",
    )
    article_text = st.text_area(
        "Article Body",
        value=st.session_state.get("article_text", ""),
        height=300,
        placeholder="Paste your news article here...",
        key="input_text",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_clicked = st.button("🔍 Analyze Article", key="btn_analyze", use_container_width=True)

    if analyze_clicked:
        combined = f"{article_title} {article_text}".strip()
        if not combined:
            st.error("Please paste a news article before clicking Analyze.")
            return

        # ── Run prediction ─────────────────────────────────────────────────
        with st.spinner("Analyzing article..."):
            try:
                from predict import predict
                result = predict(article_text, title=article_title)
            except FileNotFoundError as e:
                st.error(f"⚠️ {e}")
                return
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return

        # ── Warning ────────────────────────────────────────────────────────
        if result.get("warning"):
            st.warning(result["warning"])

        st.markdown("---")

        # ── Verdict ────────────────────────────────────────────────────────
        verdict = result["classification"]
        if verdict == "FAKE":
            st.markdown(f'<div class="verdict-fake">🚨 FAKE NEWS DETECTED 🚨</div>', unsafe_allow_html=True)
        elif verdict == "REAL":
            st.markdown(f'<div class="verdict-real">✅ REAL NEWS</div>', unsafe_allow_html=True)
        else:
            st.warning("Could not determine verdict.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Metrics row ────────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        credibility = result["credibility_score"]

        with m1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">Credibility Score</div>
              <div class="value">{credibility}/100</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">Confidence</div>
              <div class="value">{result['confidence']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">REAL Probability</div>
              <div class="value">{result['raw_proba_real']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Credibility progress bar ───────────────────────────────────────
        st.markdown(credibility_bar_html(credibility), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Top contributing words ─────────────────────────────────────────
        top_words = result.get("top_contributing_words", [])
        if top_words:
            st.markdown('<div class="section-header">🔑 Key Influencing Words</div>', unsafe_allow_html=True)
            html_tags = ""
            for w in top_words:
                tag_class = "word-tag-fake" if w["direction"] == "FAKE" else "word-tag-real"
                icon = "↓" if w["direction"] == "FAKE" else "↑"
                html_tags += f'<span class="{tag_class}">{icon} {w["word"]}</span>'
            st.markdown(html_tags, unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:0.8rem;color:#6B7280;margin-top:6px;">'
                '🔴 Red = pushed toward FAKE &nbsp;&nbsp; 🟢 Green = pushed toward REAL</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Plain-English explanation ──────────────────────────────────────
        st.markdown('<div class="section-header">💡 What This Means</div>', unsafe_allow_html=True)
        if verdict == "FAKE":
            explanation = (
                f"The model classified this article as **FAKE** with **{result['confidence']:.1f}% confidence**. "
                f"The credibility score of **{credibility}/100** indicates the article exhibits strong linguistic "
                "patterns associated with misinformation — such as sensational language, conspiracy framing, "
                "and unverified claims. The highlighted red words were the strongest indicators of fake content."
            )
        elif verdict == "REAL":
            explanation = (
                f"The model classified this article as **REAL** with **{result['confidence']:.1f}% confidence**. "
                f"The credibility score of **{credibility}/100** suggests the text uses neutral, factual language "
                "typical of legitimate journalism — measured tone, specific details, and attributable statements. "
                "The highlighted green words were the strongest indicators of real content."
            )
        else:
            explanation = "The model could not confidently classify this article."

        st.markdown(f'<div class="info-box">{explanation}</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 2 — MODEL INSIGHTS
# =============================================================================

def render_model_insights_tab(metrics_df: Optional[pd.DataFrame]) -> None:
    """Render model performance metrics and plots."""
    st.markdown('<div class="section-header">📊 Model Comparison</div>', unsafe_allow_html=True)

    if metrics_df is not None:
        # Style the dataframe
        styled = metrics_df.copy()
        styled["accuracy"]  = styled["accuracy"].map("{:.4f}".format)
        styled["precision"] = styled["precision"].map("{:.4f}".format)
        styled["recall"]    = styled["recall"].map("{:.4f}".format)
        styled["f1"]        = styled["f1"].map("{:.4f}".format)
        styled["roc_auc"]   = styled["roc_auc"].map("{:.4f}".format)
        styled.columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("Model metrics not found. Please run `python src/train.py`.")

    st.markdown("---")

    # ── Plots ──────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
        cm_path = PLOTS_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.info("Plot not found. Run train.py to generate.")

    with c2:
        st.markdown('<div class="section-header">📈 ROC Curve</div>', unsafe_allow_html=True)
        roc_path = PLOTS_DIR / "roc_curve.png"
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)
        else:
            st.info("Plot not found. Run train.py to generate.")

    st.markdown("---")
    comp_path = PLOTS_DIR / "model_comparison.png"
    if comp_path.exists():
        st.markdown('<div class="section-header">🏆 Model Comparison Chart</div>', unsafe_allow_html=True)
        st.image(str(comp_path), use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📖 Metric Glossary</div>', unsafe_allow_html=True)
    metrics_info = {
        "Accuracy": "Percentage of all articles correctly classified (FAKE or REAL).",
        "Precision": "Of all predicted REAL articles, what fraction were actually real? High precision = fewer false 'REALs'.",
        "Recall": "Of all actual REAL articles, what fraction did we catch? High recall = fewer missed real news.",
        "F1 Score": "Harmonic mean of Precision and Recall. Best single metric when classes are balanced.",
        "ROC-AUC": "Area Under the ROC Curve. 1.0 = perfect classifier, 0.5 = random guessing.",
    }
    for metric, desc in metrics_info.items():
        st.markdown(f"**{metric}:** {desc}")


# =============================================================================
# TAB 3 — DATA INSIGHTS
# =============================================================================

def render_data_insights_tab() -> None:
    """Render EDA charts, top words, and sample articles."""
    st.markdown('<div class="section-header">📏 Article Length Distribution</div>', unsafe_allow_html=True)
    ld_path = PLOTS_DIR / "article_length_distribution.png"
    if ld_path.exists():
        st.image(str(ld_path), use_container_width=True)
    else:
        st.info("Plot not found. Run train.py.")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">🔴 Top FAKE News Words</div>', unsafe_allow_html=True)
        fw_path = PLOTS_DIR / "top_fake_words.png"
        if fw_path.exists():
            st.image(str(fw_path), use_container_width=True)
        else:
            st.info("Plot not found. Run train.py.")

    with col2:
        st.markdown('<div class="section-header">🟢 Top REAL News Words</div>', unsafe_allow_html=True)
        rw_path = PLOTS_DIR / "top_real_words.png"
        if rw_path.exists():
            st.image(str(rw_path), use_container_width=True)
        else:
            st.info("Plot not found. Run train.py.")

    st.markdown("---")

    # ── EDA Findings ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔍 Key EDA Findings</div>', unsafe_allow_html=True)
    st.markdown("""
- **Balanced dataset**: The Kaggle dataset contains ~23,000 fake and ~21,000 real articles — minimal class imbalance.
- **Article length**: Real news articles tend to be longer and more detailed than fake news.
- **Fake news language**: Fake articles frequently use sensational verbs (e.g. *breaking*, *shocking*, *exposed*) and emotional appeals.
- **Real news language**: Real articles rely more on formal vocabulary, specific proper nouns, and attribution language (*said*, *according to*).
- **TF-IDF is effective**: High-dimensional sparse features from TF-IDF (10,000 features) give Logistic Regression exceptional accuracy.
- **Subject distribution**: Fake news over-indexes on political and conspiracy topics; real news spans business, politics, and world events evenly.
    """)

    st.markdown("---")

    # ── Sample articles from dataset ───────────────────────────────────────
    st.markdown('<div class="section-header">📄 Sample Articles from Dataset</div>', unsafe_allow_html=True)
    fake_samples, real_samples = load_raw_samples()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔴 FAKE News Samples**")
        for s in fake_samples:
            st.markdown(f"""
            <div class="article-card">
              <div class="card-title">{s['title']}</div>
              {s['text']}...
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("**🟢 REAL News Samples**")
        for s in real_samples:
            st.markdown(f"""
            <div class="article-card">
              <div class="card-title">{s['title']}</div>
              {s['text']}...
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# MAIN APP
# =============================================================================

def main() -> None:
    """Entry point — renders the full Streamlit application."""
    metrics_df = load_metrics()
    stats      = load_stats()

    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown(
        '<h1 class="hero-title">🔍 Fake News Detector</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-subtitle">Powered by Logistic Regression + TF-IDF NLP · '
        'Trained on 44,000+ real & fake articles</p>',
        unsafe_allow_html=True,
    )

    # Show accuracy badge if available
    if metrics_df is not None:
        lr_row = metrics_df[metrics_df["model"] == "Logistic Regression"]
        if not lr_row.empty:
            acc = lr_row.iloc[0]["accuracy"] * 100
            st.markdown(
                f'<span class="badge">✅ Model Accuracy: {acc:.2f}%</span>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────
    render_sidebar(metrics_df, stats)

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🔍 Detector",
        "📊 Model Insights",
        "📈 Data Insights",
    ])

    with tab1:
        render_detector_tab()

    with tab2:
        render_model_insights_tab(metrics_df)

    with tab3:
        render_data_insights_tab()


if __name__ == "__main__":
    main()
