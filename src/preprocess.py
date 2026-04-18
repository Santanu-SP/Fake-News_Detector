"""
preprocess.py — Data Loading, Merging, and Text Cleaning Pipeline
=================================================================
This module handles:
  1. Loading Fake.csv and True.csv from the data/ folder
  2. Merging them into a single shuffled DataFrame
  3. Cleaning news text using an NLP pipeline
  4. Returning a ready-to-train dataset

Author:  Fake News Detector Project
License: MIT
"""

from __future__ import annotations
from typing import Optional

import re
import string
import pathlib
import pandas as pd
import nltk

# ── Download required NLTK assets (runs once, skips if already present) ──────
# 'stopwords'     → common English words (the, is, at…) that carry no signal
# 'punkt'         → sentence/word tokenizer
# 'wordnet'       → lexical database used by the lemmatizer
# 'omw-1.4'       → Open Multilingual Wordnet (extends WordNet coverage)
for pkg in ["stopwords", "punkt", "wordnet", "omw-1.4", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg.startswith("punkt") else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Global objects — instantiated once to avoid repeated overhead
STOP_WORDS: set = set(stopwords.words("english"))
LEMMATIZER: WordNetLemmatizer = WordNetLemmatizer()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# =============================================================================
# 1. DATA LOADING & MERGING
# =============================================================================

def load_and_merge_data(
    fake_path: Optional[pathlib.Path] = None,
    real_path: Optional[pathlib.Path] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load Fake.csv and True.csv, assign labels, merge, and shuffle.

    Labels:
        Fake.csv  → label = 0  (FAKE)
        True.csv  → label = 1  (REAL)

    Parameters
    ----------
    fake_path    : Path to Fake.csv  (defaults to data/Fake.csv)
    real_path    : Path to True.csv  (defaults to data/True.csv)
    random_state : Seed for reproducible shuffling

    Returns
    -------
    pd.DataFrame with columns: title, text, subject, date, label, content
    """
    fake_path = fake_path or DATA_DIR / "Fake.csv"
    real_path = real_path or DATA_DIR / "True.csv"

    # ── Load raw CSVs ──────────────────────────────────────────────────────
    print("[INFO] Loading datasets...")
    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)

    # ── Assign binary labels ───────────────────────────────────────────────
    # 0 = FAKE, 1 = REAL — standard convention for binary classification
    fake_df["label"] = 0
    real_df["label"] = 1

    # ── Merge into one DataFrame ───────────────────────────────────────────
    df = pd.concat([fake_df, real_df], ignore_index=True)

    # ── Shuffle to avoid order-based leakage during cross-validation ───────
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # ── Combine title + text into a single 'content' field ────────────────
    # Merging title with body gives the model more signal per sample.
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

    # ── Drop rows that are entirely empty ─────────────────────────────────
    df.dropna(subset=["content", "label"], inplace=True)
    df = df[df["content"].str.strip() != ""]

    # ── Print dataset statistics for transparency ──────────────────────────
    total = len(df)
    n_fake = (df["label"] == 0).sum()
    n_real = (df["label"] == 1).sum()
    print(f"[INFO] Dataset loaded successfully.")
    print(f"       Total samples : {total:,}")
    print(f"       FAKE articles : {n_fake:,}  ({n_fake/total*100:.1f}%)")
    print(f"       REAL articles : {n_real:,}  ({n_real/total*100:.1f}%)")

    return df


# =============================================================================
# 2. TEXT CLEANING
# =============================================================================

def clean_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline for a single news article string.

    Pipeline steps (in order):
        1. Lowercase            → normalises casing, "The" == "the"
        2. Remove URLs          → http/https/www links add no semantic value
        3. Remove HTML tags     → strip any embedded markup (<p>, <br> etc.)
        4. Remove punctuation   → punctuation is noise for bag-of-words models
        5. Remove numbers       → raw digits rarely carry category signal
        6. Tokenise             → split string into list of word tokens
        7. Remove stopwords     → drop high-frequency, low-signal words
        8. Lemmatise            → reduce inflected forms to their base lemma
                                   (e.g. "running" → "run", "better" → "good")
                                   Preferred over stemming for news text because
                                   it produces real English words.
        9. Re-join tokens       → return cleaned string

    Parameters
    ----------
    text : Raw article string

    Returns
    -------
    str : Cleaned, lemmatised text ready for TF-IDF vectorisation
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs (http://, https://, www.)
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # 3. Remove HTML tags (e.g. <p>, <br/>, <div class="...">)
    text = re.sub(r"<[^>]+>", "", text)

    # 4. Remove punctuation and special characters
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 5. Remove digits
    text = re.sub(r"\d+", "", text)

    # 6. Tokenise — splits on whitespace and punctuation boundaries
    tokens = word_tokenize(text)

    # 7. Remove stopwords and very short tokens (len < 2 adds noise)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    # 8. Lemmatise — converts each token to its dictionary base form
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    # 9. Re-join into a single cleaned string
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply clean_text() to the entire 'content' column of a DataFrame.

    Also prints a before/after example so the cleaning effect is visible.

    Parameters
    ----------
    df : DataFrame with a 'content' column (output of load_and_merge_data)

    Returns
    -------
    pd.DataFrame : Same DataFrame with a new 'clean_content' column added
    """
    print("\n[INFO] Preprocessing text (this may take a minute)...")

    # ── Show a before/after example for transparency ───────────────────────
    example_raw = df["content"].iloc[0]
    example_clean = clean_text(example_raw)
    print("\n── Before cleaning ─────────────────────────────────────────────")
    print(example_raw[:300])
    print("\n── After cleaning ──────────────────────────────────────────────")
    print(example_clean[:300])
    print("────────────────────────────────────────────────────────────────\n")

    # Apply clean_text to every row (progress printed every 5k rows)
    cleaned = []
    for i, text in enumerate(df["content"]):
        cleaned.append(clean_text(text))
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1:,} / {len(df):,} articles...")

    df = df.copy()
    df["clean_content"] = cleaned

    # Drop any rows where cleaning produced an empty string
    before = len(df)
    df = df[df["clean_content"].str.strip() != ""].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[WARN] Dropped {dropped} rows that became empty after cleaning.")

    print(f"[INFO] Preprocessing complete. {len(df):,} articles ready.\n")
    return df


# =============================================================================
# 3. MAIN — Run as standalone script
# =============================================================================

if __name__ == "__main__":
    # Load and merge
    df = load_and_merge_data()

    # Clean text
    df = preprocess_dataframe(df)

    # Quick sanity check
    print(df[["label", "clean_content"]].head(3))
    print(f"\nColumn dtypes:\n{df.dtypes}")
