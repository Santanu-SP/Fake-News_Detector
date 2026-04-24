"""
predict.py — Prediction Module
===============================
Loads the saved TF-IDF vectorizer and Logistic Regression model, cleans an
input article, and returns a rich prediction object including:
  - classification  : 'FAKE' or 'REAL'
  - confidence      : probability % of the predicted class
  - credibility_score : 0-100  (0 = definitely fake, 100 = definitely real)
  - top_contributing_words : top 10 words that most influenced the prediction

Author:  Fake News Detector Project
License: MIT
"""

from __future__ import annotations
from typing import List, Dict, Any

import pathlib
import warnings
import numpy as np
import joblib
import sys

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"

# Add src/ to path so preprocess can be imported regardless of CWD
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from preprocess import clean_text

# ── Minimum readable article length ───────────────────────────────────────────
MIN_WORD_COUNT = 10


def _load_artifacts() -> tuple:
    """
    Load the saved vectorizer and model from disk.

    Returns
    -------
    (vectorizer, model) tuple — both sklearn objects
    """
    vec_path   = MODELS_DIR / "tfidf_vectorizer.pkl"
    model_path = MODELS_DIR / "lr_model.pkl"

    # Fallback to best model if LR-specific file not found
    if not model_path.exists():
        model_path = MODELS_DIR / "fake_news_model.pkl"

    if not vec_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "Model files not found. Please run 'python src/train.py' first "
            "to train the model and generate the required .pkl files."
        )

    vectorizer = joblib.load(vec_path)
    model      = joblib.load(model_path)
    return vectorizer, model


def get_top_contributing_words(
    text_tfidf,
    vectorizer,
    model,
    n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Identify the top-N words in the input that most influenced the prediction.

    Method:
        element-wise product of TF-IDF weights (how important the word is
        in this article) × LR coefficients (how much the model associates
        those words with FAKE/REAL).

    A positive contribution → pushes toward REAL
    A negative contribution → pushes toward FAKE

    Parameters
    ----------
    text_tfidf : Sparse TF-IDF matrix for the single input article
    vectorizer : Fitted TfidfVectorizer
    model      : Fitted LogisticRegression
    n          : Number of top words to return

    Returns
    -------
    List of dicts: [{"word": str, "contribution": float, "direction": "FAKE"|"REAL"}, ...]
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]  # LR coefficients

    # TF-IDF weights for this document (sparse → dense)
    tfidf_weights = np.array(text_tfidf.todense()).flatten()

    # Contribution = tfidf_weight × coefficient
    contributions = tfidf_weights * coefs

    # Get indices of top N by absolute contribution, only for present tokens
    present_idx = np.where(tfidf_weights > 0)[0]
    if len(present_idx) == 0:
        return []

    top_n = min(n, len(present_idx))
    top_abs = np.argsort(np.abs(contributions[present_idx]))[-top_n:][::-1]
    top_idx = present_idx[top_abs]

    top_words = []
    for idx in top_idx:
        word = feature_names[idx]
        contrib = float(contributions[idx])
        direction = "REAL" if contrib > 0 else "FAKE"
        top_words.append({"word": word, "contribution": contrib, "direction": direction})

    return top_words


def predict(text: str, title: str = "") -> dict:
    """
    Predict whether a news article is FAKE or REAL.

    Parameters
    ----------
    text  : Body of the news article
    title : (Optional) headline of the article — will be prepended to text

    Returns
    -------
    dict with keys:
        classification       : 'FAKE' or 'REAL'
        confidence           : float, probability of predicted class (0-100)
        credibility_score    : int, 0 (fake) to 100 (real) — P(REAL) * 100
        top_contributing_words : list of dicts (word, contribution, direction)
        warning              : str or None — empty input / short article / non-English
        raw_proba_fake       : float, raw probability of FAKE class
        raw_proba_real       : float, raw probability of REAL class
    """
    result = {
        "classification": None,
        "confidence": 0.0,
        "credibility_score": 50,
        "top_contributing_words": [],
        "warning": None,
        "raw_proba_fake": 0.5,
        "raw_proba_real": 0.5,
    }

    # ── Edge case: empty input ─────────────────────────────────────────────
    combined = f"{title} {text}".strip()
    if not combined:
        result["warning"] = "Input is empty. Please paste a news article."
        result["classification"] = "UNKNOWN"
        return result

    # ── Edge case: very short article ─────────────────────────────────────
    word_count = len(combined.split())
    if word_count < MIN_WORD_COUNT:
        result["warning"] = (
            f"Article is very short ({word_count} words). "
            f"Results may be unreliable. Please provide at least {MIN_WORD_COUNT} words."
        )

    # ── Edge case: non-English text (best-effort detection) ───────────────
    try:
        from langdetect import detect
        lang = detect(combined[:500])
        if lang != "en":
            result["warning"] = (
                f"Text appears to be in language '{lang}'. "
                "This model was trained on English articles only — results may be inaccurate."
            )
    except Exception:
        pass  # langdetect optional; skip if not installed or detection fails

    # ── Load model artifacts ───────────────────────────────────────────────
    vectorizer, model = _load_artifacts()

    # ── Preprocess and vectorize ───────────────────────────────────────────
    cleaned = clean_text(combined)
    text_tfidf = vectorizer.transform([cleaned])

    # ── Predict ───────────────────────────────────────────────────────────
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(text_tfidf)[0]   # [P(FAKE), P(REAL)]
        p_fake, p_real = float(proba[0]), float(proba[1])
    else:
        # Linear models without predict_proba — use sigmoid of decision function
        score = float(model.decision_function(text_tfidf)[0])
        p_real = 1 / (1 + np.exp(-score))
        p_fake = 1 - p_real

    predicted_class = "REAL" if p_real >= 0.5 else "FAKE"
    confidence      = max(p_fake, p_real) * 100
    credibility     = round(p_real * 100)

    # ── Top contributing words ─────────────────────────────────────────────
    # Only supported for LogisticRegression (needs coef_ attribute)
    top_words = []
    if hasattr(model, "coef_"):
        top_words = get_top_contributing_words(text_tfidf, vectorizer, model)

    result.update({
        "classification": predicted_class,
        "confidence": round(confidence, 2),
        "credibility_score": credibility,
        "top_contributing_words": top_words,
        "raw_proba_fake": round(p_fake * 100, 2),
        "raw_proba_real": round(p_real * 100, 2),
    })

    return result


# =============================================================================
# MAIN — Quick CLI test
# =============================================================================

if __name__ == "__main__":
    sample_fake = (
        "BREAKING: Scientists Confirm Drinking Bleach Cures All Diseases. "
        "In a shocking revelation that mainstream media refuses to cover, "
        "a group of underground scientists have confirmed that drinking a small "
        "amount of household bleach every morning can cure cancer, diabetes, and "
        "COVID-19. The government has been suppressing this information for years "
        "to protect Big Pharma profits."
    )

    sample_real = (
        "Federal Reserve raises interest rates by 0.25 percent. "
        "The Federal Reserve raised its benchmark interest rate by a quarter "
        "percentage point on Wednesday, continuing its campaign to bring inflation "
        "under control. The decision by the Federal Open Market Committee was unanimous. "
        "Fed Chair stated the committee remains strongly committed to returning inflation "
        "to its 2 percent objective."
    )

    for label, text in [("FAKE", sample_fake), ("REAL", sample_real)]:
        print(f"\n{'='*60}")
        print(f"Expected: {label}")
        res = predict(text)
        print(f"Predicted: {res['classification']}")
        print(f"Confidence: {res['confidence']:.1f}%")
        print(f"Credibility Score: {res['credibility_score']}/100")
        if res["warning"]:
            print(f"Warning: {res['warning']}")
        print("Top contributing words:")
        for w in res["top_contributing_words"][:5]:
            print(f"  {w['word']:20s}  {w['direction']}  ({w['contribution']:+.4f})")
