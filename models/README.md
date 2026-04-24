# Models directory — auto-populated by train.py

After running `python src/train.py`, this folder will contain:
  - fake_news_model.pkl   ← Best model (by F1 score)
  - lr_model.pkl          ← Logistic Regression (for interpretability)
  - tfidf_vectorizer.pkl  ← Fitted TF-IDF vectorizer
  - model_metrics.csv     ← Performance comparison table
  - dataset_stats.json    ← Dataset statistics for Streamlit app
