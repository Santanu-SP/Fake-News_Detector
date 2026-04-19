"""
train.py — Model Training, Evaluation, and Plot Generation
===========================================================
Trains four ML classifiers on TF-IDF features extracted from cleaned news text,
evaluates them on a held-out test set, generates all required plots, and saves
the best model + vectorizer to disk.

─── Why Logistic Regression for Fake News Detection? ────────────────────────
# 1. TF-IDF produces high-dimensional sparse features — LR handles this well
#    via efficient implementations (lbfgs solver uses L-BFGS-B optimisation).
# 2. Coefficients are directly interpretable: positive coefs → REAL signal,
#    negative coefs → FAKE signal — great for interview explanations.
# 3. Fast to train even on large datasets with tens of thousands of features.
# 4. predict_proba gives calibrated confidence scores via the logistic sigmoid.
# 5. Regularisation (C parameter) prevents overfitting on noisy news data.
# 6. Industry baseline — always start simple before going deep learning.
─────────────────────────────────────────────────────────────────────────────

Author:  Fake News Detector Project
License: MIT
"""

from __future__ import annotations
from typing import List, Dict, Any

import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; safe for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve,
)
from tabulate import tabulate

# Local import
from preprocess import load_and_merge_data, preprocess_dataframe

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
PLOTS_DIR    = PROJECT_ROOT / "plots"
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Plot styling ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
PALETTE = {"FAKE": "#E63946", "REAL": "#2DC653"}


# =============================================================================
# TF-IDF FEATURE EXTRACTION
# =============================================================================

def build_tfidf_features(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Fit a TF-IDF vectorizer on the training split and transform both splits.

    TF-IDF (Term Frequency-Inverse Document Frequency) converts raw text into
    a numerical matrix where each cell reflects how important a word is to a
    document relative to the entire corpus.  Using bigrams (ngram_range=(1,2))
    captures two-word phrases like "breaking news" or "fake report".

    Parameters
    ----------
    df           : Preprocessed DataFrame with 'clean_content' and 'label'
    test_size    : Fraction of data reserved for evaluation
    random_state : Seed for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test, vectorizer
    """
    X = df["clean_content"]
    y = df["label"]

    # Stratified split → preserves class ratio in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,   # ensures equal fake/real proportions in each split
    )

    print(f"[INFO] Train size: {len(X_train):,} | Test size: {len(X_test):,}")

    # TF-IDF vectorizer configuration
    vectorizer = TfidfVectorizer(
        max_features=10_000,   # vocabulary cap — enough for news, not overfit
        ngram_range=(1, 2),    # unigrams + bigrams for phrase capture
        sublinear_tf=True,     # apply log(1+tf) to dampen frequency dominance
        min_df=2,              # ignore tokens appearing in fewer than 2 docs
        max_df=0.95,           # ignore tokens appearing in >95% of docs
    )

    # Fit ONLY on training data to prevent data leakage into the test set
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    print(f"[INFO] Vocabulary size: {len(vectorizer.vocabulary_):,} features")

    # Persist the fitted vectorizer — must be loaded alongside the model
    vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    joblib.dump(vectorizer, vec_path)
    print(f"[INFO] Vectorizer saved → {vec_path}")

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def get_models() -> dict:
    """
    Return a dictionary of model name → sklearn estimator.

    All models chosen because they operate natively on sparse TF-IDF matrices.

    Logistic Regression:        Primary model — see interview notes at top.
    Multinomial Naive Bayes:    Fast probabilistic baseline for text tasks.
    Linear SVM (SGD):           A.k.a. LinearSVC via SGD; strong for text.
    Passive Aggressive:         Online learner; widely cited in fake news papers.
    """
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,             # inverse regularisation — larger = less penalty
            max_iter=1000,     # enough iterations for convergence on TF-IDF
            solver="lbfgs",    # efficient for multiclass problems
            random_state=RANDOM_STATE,
        ),
        "Multinomial NB": MultinomialNB(
            alpha=0.1,         # additive (Laplace) smoothing
        ),
        "Linear SVM": SGDClassifier(
            loss="hinge",      # hinge loss → linear SVM objective
            max_iter=1000,
            random_state=RANDOM_STATE,
            tol=1e-3,
        ),
        "Passive Aggressive": PassiveAggressiveClassifier(
            max_iter=1000,
            random_state=RANDOM_STATE,
            tol=1e-3,
        ),
    }


# =============================================================================
# EVALUATION HELPER
# =============================================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str,
) -> dict:
    """
    Compute classification metrics for a fitted model.

    Parameters
    ----------
    model      : Fitted sklearn estimator
    X_test     : TF-IDF test matrix
    y_test     : Ground-truth labels
    model_name : String label used in output tables

    Returns
    -------
    dict with keys: model, accuracy, precision, recall, f1, roc_auc, y_pred, y_proba
    """
    y_pred = model.predict(X_test)

    # predict_proba is not available for SVM/PAC — fall back to decision scores
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Platt scaling approximation: normalise decision function to [0,1]
        scores = model.decision_function(X_test)
        y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)

    return {
        "model":     model_name,
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "roc_auc":   auc,
        "y_pred":    y_pred,
        "y_proba":   y_proba,
    }


# =============================================================================
# PLOT GENERATION
# =============================================================================

def plot_confusion_matrix(y_test, y_pred, save_path: pathlib.Path) -> None:
    """Save a seaborn heatmap confusion matrix with counts and percentages."""
    cm = confusion_matrix(y_test, y_pred)
    total = cm.sum()

    # Build annotation labels: count + percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i,j]:,}\n({cm[i,j]/total*100:.1f}%)"

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=annot, fmt="", cmap="RdYlGn",
        linewidths=0.5, linecolor="white",
        xticklabels=["FAKE", "REAL"],
        yticklabels=["FAKE", "REAL"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    ax.set_title("Confusion Matrix — Logistic Regression", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Confusion matrix saved → {save_path}")


def plot_roc_curves(results: List[Dict[str, Any]], y_test, save_path: pathlib.Path) -> None:
    """Plot ROC curves for all models on a single axes with AUC in legend."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#E63946", "#457B9D", "#2DC653", "#F4A261"]
    for res, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(
            fpr, tpr,
            label=f"{res['model']} (AUC = {res['roc_auc']:.4f})",
            color=color, lw=2,
        )

    # Diagonal baseline (random classifier)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Baseline (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — All Models", fontsize=14, pad=15)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] ROC curve saved → {save_path}")


def plot_top_words(
    vectorizer: TfidfVectorizer,
    model,
    label: int,
    n: int = 25,
    save_path: pathlib.Path = None,
) -> None:
    """
    Bar chart of the top N words most associated with a class label,
    extracted from Logistic Regression coefficients.

    For a binary LR model:
        coef_ > 0  →  associated with class 1 (REAL)
        coef_ < 0  →  associated with class 0 (FAKE)
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]  # shape: (n_features,)

    if label == 0:  # FAKE → most negative coefficients
        top_idx = np.argsort(coefs)[:n]
        color = "#E63946"
        title = f"Top {n} Indicator Words — FAKE News"
        top_words = feature_names[top_idx]
        top_coefs = np.abs(coefs[top_idx])[::-1]
        top_words = top_words[::-1]
    else:           # REAL → most positive coefficients
        top_idx = np.argsort(coefs)[-n:][::-1]
        color = "#2DC653"
        title = f"Top {n} Indicator Words — REAL News"
        top_words = feature_names[top_idx]
        top_coefs = coefs[top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(n), top_coefs, color=color, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(n))
    ax.set_yticklabels(top_words, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient Magnitude", fontsize=11)
    ax.set_title(title, fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Top words chart saved → {save_path}")


def plot_article_length_distribution(df: pd.DataFrame, save_path: pathlib.Path) -> None:
    """Overlapping histogram of article word-count for FAKE vs REAL."""
    df = df.copy()
    df["word_count"] = df["content"].apply(lambda t: len(str(t).split()))

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, name, color in [(0, "FAKE", "#E63946"), (1, "REAL", "#2DC653")]:
        subset = df[df["label"] == label]["word_count"]
        ax.hist(subset, bins=60, alpha=0.55, color=color, label=f"{name} (median={int(subset.median())})")

    ax.set_xlabel("Article Word Count", fontsize=12)
    ax.set_ylabel("Number of Articles", fontsize=12)
    ax.set_title("Article Length Distribution — FAKE vs REAL", fontsize=14, pad=15)
    ax.legend(fontsize=10)
    ax.set_xlim(left=0)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Article length distribution saved → {save_path}")


def plot_model_comparison(results: List[Dict[str, Any]], save_path: pathlib.Path) -> None:
    """Grouped bar chart comparing Accuracy, Precision, Recall, F1 for all models."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    model_names = [r["model"] for r in results]

    x = np.arange(len(model_names))
    width = 0.2
    colors = ["#457B9D", "#E63946", "#2DC653", "#F4A261"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [r[metric] for r in results]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize(), color=color, alpha=0.88)
        # Add value labels on top of each bar
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — All Metrics", fontsize=14, pad=15)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Model comparison chart saved → {save_path}")


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def train_and_evaluate() -> None:
    """
    End-to-end training pipeline:
        1. Load & preprocess data
        2. TF-IDF feature extraction
        3. Train all four models
        4. Evaluate on held-out test set
        5. Generate and save all plots
        6. Save best model to disk
    """
    # ── 1. Data ───────────────────────────────────────────────────────────────
    df = load_and_merge_data()
    df = preprocess_dataframe(df)

    # ── 2. Features ───────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, vectorizer = build_tfidf_features(df)

    # ── 3. Train ──────────────────────────────────────────────────────────────
    models = get_models()
    fitted_models = {}
    results = []

    print("\n" + "="*60)
    print("  TRAINING AND EVALUATION")
    print("="*60)

    for name, model in models.items():
        print(f"\n[TRAIN] {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        res = evaluate_model(model, X_test, y_test, name)
        results.append(res)

        print(f"  Accuracy  : {res['accuracy']:.4f}")
        print(f"  Precision : {res['precision']:.4f}")
        print(f"  Recall    : {res['recall']:.4f}")
        print(f"  F1 Score  : {res['f1']:.4f}")
        print(f"  ROC-AUC   : {res['roc_auc']:.4f}")

    # ── 4. Print comparison table ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("  MODEL COMPARISON TABLE")
    print("="*60)
    table_data = [
        [r["model"], f"{r['accuracy']:.4f}", f"{r['precision']:.4f}",
         f"{r['recall']:.4f}", f"{r['f1']:.4f}", f"{r['roc_auc']:.4f}"]
        for r in results
    ]
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    print(tabulate(table_data, headers=headers, tablefmt="github"))

    # ── 5. Best model (primary: Logistic Regression by design) ───────────────
    lr_res = next(r for r in results if r["model"] == "Logistic Regression")
    lr_model = fitted_models["Logistic Regression"]

    print("\n── Full Classification Report (Logistic Regression) ──────────")
    print(classification_report(y_test, lr_res["y_pred"], target_names=["FAKE", "REAL"]))

    # Determine best model by F1 score for saving
    best_res = max(results, key=lambda r: r["f1"])
    best_model = fitted_models[best_res["model"]]
    print(f"\n[INFO] Best model by F1 score: {best_res['model']} (F1={best_res['f1']:.4f})")

    # Save best model
    model_path = MODELS_DIR / "fake_news_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"[INFO] Best model saved → {model_path}")

    # Also save LR separately if it's not the best, for predict.py (interpretability)
    lr_path = MODELS_DIR / "lr_model.pkl"
    joblib.dump(lr_model, lr_path)

    # Save results metrics for app.py consumption
    metrics_df = pd.DataFrame([{
        "model": r["model"],
        "accuracy": r["accuracy"],
        "precision": r["precision"],
        "recall": r["recall"],
        "f1": r["f1"],
        "roc_auc": r["roc_auc"],
    } for r in results])
    metrics_df.to_csv(MODELS_DIR / "model_metrics.csv", index=False)

    # Save dataset stats for app.py consumption  
    stats = {
        "total": len(df),
        "fake": int((df["label"] == 0).sum()),
        "real": int((df["label"] == 1).sum()),
        "best_model": best_res["model"],
        "best_accuracy": best_res["accuracy"],
        "best_f1": best_res["f1"],
    }
    pd.Series(stats).to_json(MODELS_DIR / "dataset_stats.json")

    # ── 6. Generate plots ─────────────────────────────────────────────────────
    print("\n[INFO] Generating evaluation plots...")

    # 6a. Confusion matrix
    plot_confusion_matrix(
        y_test, lr_res["y_pred"],
        PLOTS_DIR / "confusion_matrix.png",
    )

    # 6b. ROC curve (all models)
    plot_roc_curves(results, y_test, PLOTS_DIR / "roc_curve.png")

    # 6c. Top fake words
    plot_top_words(
        vectorizer, lr_model, label=0, n=25,
        save_path=PLOTS_DIR / "top_fake_words.png",
    )

    # 6d. Top real words
    plot_top_words(
        vectorizer, lr_model, label=1, n=25,
        save_path=PLOTS_DIR / "top_real_words.png",
    )

    # 6e. Article length distribution
    plot_article_length_distribution(df, PLOTS_DIR / "article_length_distribution.png")

    # 6f. Model comparison
    plot_model_comparison(results, PLOTS_DIR / "model_comparison.png")

    print("\n" + "="*60)
    print(f"  TRAINING COMPLETE")
    print(f"  Best model : {best_res['model']}")
    print(f"  F1 Score   : {best_res['f1']:.4f}")
    print(f"  ROC-AUC    : {best_res['roc_auc']:.4f}")
    print(f"  Plots saved to: {PLOTS_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    train_and_evaluate()
