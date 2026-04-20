```
███████╗ █████╗ ██╗  ██╗███████╗    ███╗   ██╗███████╗██╗    ██╗███████╗
██╔════╝██╔══██╗██║ ██╔╝██╔════╝    ████╗  ██║██╔════╝██║    ██║██╔════╝
█████╗  ███████║█████╔╝ █████╗      ██╔██╗ ██║█████╗  ██║ █╗ ██║███████╗
██╔══╝  ██╔══██║██╔═██╗ ██╔══╝      ██║╚██╗██║██╔══╝  ██║███╗██║╚════██║
██║     ██║  ██║██║  ██╗███████╗    ██║ ╚████║███████╗╚███╔███╔╝███████║
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚═╝  ╚═══╝╚══════╝ ╚══╝╚══╝ ╚══════╝
                D E T E C T O R   —   A I - P o w e r e d   N L P
```

# 📰 Fake News Detector

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.x-green?style=flat-square)](https://nltk.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

> A production-grade, end-to-end **Fake News Detection** system built with classical NLP and Machine Learning.  
> Paste any news article and get an instant verdict with confidence scores and explainable word attributions.

---

## 🌍 Why Fake News Detection Matters

Misinformation spreads 6× faster than truthful news on social media (MIT, 2018).  
Automated detection at scale is a critical NLP problem combining:
- **Text classification** — the core supervised learning task
- **Feature engineering** — TF-IDF, n-grams, stylometric features
- **Model interpretability** — understanding *why* an article is flagged

This project demonstrates a complete, interview-ready ML pipeline from raw CSV data to a deployed Streamlit web app.

---

## 📦 Dataset

| Property | Details |
|---|---|
| Source | [Kaggle — Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) |
| Files | `Fake.csv` (~23,000 articles) + `True.csv` (~21,000 articles) |
| Columns | `title`, `text`, `subject`, `date` |
| Labels | Fake → 0, Real → 1 |
| Balance | ~52% Fake / ~48% Real (minimal class imbalance) |

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| ML / NLP | scikit-learn, NLTK |
| Data | pandas, NumPy |
| Visualization | matplotlib, seaborn, WordCloud |
| Web App | Streamlit |
| Serialization | joblib |
| Notebook | Jupyter |

---

## 🏗️ Project Architecture

```
fake-news-detector/
├── data/
│   ├── Fake.csv                       ← Raw fake articles
│   └── True.csv                       ← Raw real articles
├── notebooks/
│   └── exploration.ipynb              ← EDA + model experiments
├── models/
│   ├── fake_news_model.pkl            ← Best trained model
│   ├── lr_model.pkl                   ← Logistic Regression (for explainability)
│   ├── tfidf_vectorizer.pkl           ← Fitted TF-IDF vectorizer
│   ├── model_metrics.csv              ← Performance comparison table
│   └── dataset_stats.json            ← Dataset statistics
├── plots/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── top_fake_words.png
│   ├── top_real_words.png
│   ├── article_length_distribution.png
│   └── model_comparison.png
├── src/
│   ├── preprocess.py                  ← Text cleaning pipeline
│   ├── train.py                       ← Model training + evaluation
│   └── predict.py                     ← Prediction logic
├── app.py                             ← Streamlit web app
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

```
Raw Article
     │
     ▼
┌─────────────────────────────┐
│   Text Preprocessing        │
│   • Lowercase               │
│   • Remove URLs, HTML       │
│   • Remove punctuation      │
│   • Remove stopwords        │
│   • Lemmatisation (NLTK)    │
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│   TF-IDF Vectorisation      │
│   • 10,000 features         │
│   • Unigrams + Bigrams      │
│   • Sublinear TF scaling    │
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│   Logistic Regression       │
│   • Trained on 80% data     │
│   • C=1.0, lbfgs solver     │
│   • predict_proba output    │
└─────────────────────────────┘
     │
     ▼
  FAKE / REAL
  + Confidence %
  + Credibility Score (0-100)
  + Top 10 influencing words
```

---

## 📊 Model Performance Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **~99%** | **~99%** | **~99%** | **~99%** | **~99%** |
| Passive Aggressive | ~99% | ~99% | ~99% | ~99% | ~99% |
| Linear SVM | ~99% | ~99% | ~99% | ~99% | ~99% |
| Multinomial NB | ~94% | ~94% | ~94% | ~94% | ~97% |

> *Exact scores generated after training. Run `python src/train.py` to see live results.*

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place files in `data/`:
```
data/Fake.csv
data/True.csv
```

### 4. Train the model
```bash
cd src
python train.py
```
This will:
- Preprocess all articles
- Extract TF-IDF features
- Train 4 models and compare them
- Save the best model to `models/`
- Generate all 6 plots to `plots/`

### 5. Launch the Streamlit app
```bash
cd ..   # back to project root
streamlit run app.py
```

### 6. (Optional) Run the Jupyter notebook
```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## 🔍 Key EDA Findings

- **Balanced dataset**: ~52% fake, ~48% real — no class imbalance problem
- **Article length**: Real articles are typically longer and more detailed
- **Fake news lexicon**: Sensational verbs — *breaking*, *exposed*, *conspiracy*, *truth*
- **Real news lexicon**: Attribution language — *said*, *according to*, *reported*, *officials*
- **TF-IDF effectiveness**: 10,000 TF-IDF features + Logistic Regression achieves near-perfect accuracy
- **Bigrams matter**: Phrases like "fake news", "mainstream media", "deep state" are strong FAKE indicators

---

## 🖼️ Screenshots

| Detector Tab | Model Insights |
|---|---|
| *(Add screenshot after running app)* | *(Add screenshot after running app)* |

---

## 🧠 Why Logistic Regression?

Logistic Regression is the **ideal choice** for this task — here's why:

| Reason | Explanation |
|---|---|
| **Sparse features** | TF-IDF creates high-dimensional sparse matrices; LR handles these efficiently via lbfgs |
| **Interpretability** | Positive coefficients → REAL indicators; Negative → FAKE indicators |
| **Speed** | Trains in seconds even with 10k features × 44k samples |
| **Calibrated probabilities** | `predict_proba` gives reliable confidence scores via logistic sigmoid |
| **Regularization** | C parameter controls overfitting; crucial for noisy news text |
| **Industry standard** | Always establish a classical baseline before jumping to BERT |

---

## 🔮 Future Improvements

- [ ] **BERT / RoBERTa** — contextual embeddings for much richer representations
- [ ] **Fact-checking API** — cross-reference claims against external databases
- [ ] **Source credibility scoring** — flag known unreliable domains
- [ ] **Browser extension** — real-time detection while browsing news online
- [ ] **Multi-language support** — extend to non-English fake news
- [ ] **Real-time news ingestion** — live RSS feed classification pipeline
- [ ] **Ensemble methods** — gradient boosting + transformer hybrid

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file for details.

---

*Built as a resume-worthy Data Science portfolio project demonstrating end-to-end ML engineering.*
