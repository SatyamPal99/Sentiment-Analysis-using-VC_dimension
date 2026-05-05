# Sentiment Analysis (VC-dimension & Comparisons)

Lightweight Streamlit app and helpers to train TF‑IDF + linear classifiers for sentiment, explore how model capacity (via TF‑IDF max features — a practical proxy for VC‑dimension) affects overfitting, and compare results across datasets (local CSV vs HuggingFace datasets such as `tweet_eval` or `imdb`).

This README is a compact, up-to-date guide for running the app and using the new comparison and VC-experiment features.

Quick facts
- Code is organized as a small package: `sentiment/` contains modular helpers: `data_utils`, `train_utils`, `infer_utils`, `vc_utils`, `plot_utils`, `hf_utils`.
- `app.py` is the Streamlit UI that ties everything together.
- Models and vectorizers are cached under `saved_models/` with dataset-aware filenames.

Requirements
- Python 3.8+ recommended.
- Core dependencies are in `requirements.txt`. For HuggingFace dataset support install `datasets`.

Minimal setup
1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app

```bash
streamlit run app.py
```

Core features (short)
- Train or load cached Logistic Regression / Linear SVM models using TF‑IDF.
- Adjust `Max Features` (TF‑IDF vocabulary size) — a working proxy for model capacity.
- View Accuracy, F1, Precision, Recall, and confusion matrix for the active dataset.
- Compare against a HuggingFace dataset (default `tweet_eval/sentiment`) — the app will download, normalize labels, and train/evaluate a separate model.
- Run a VC‑experiment: sweep TF‑IDF `max_features` (configurable list), compute train/test accuracy per value, and plot train/test curves (local vs HF datasets). Useful to visualize overfitting trends.

Notes about datasets and labels
- Local CSV: place at `data/sentiment_dataset.csv`. Required columns: `text`, `label` (labels: `positive` / `negative`).
- HuggingFace datasets: the HF loader tries to normalize common label schemes and will drop neutral/other classes to keep a binary task; it reports how many rows were dropped.

How model caching works
- Saved files in `saved_models/` include the dataset tag in the filename. This prevents collisions when comparing local and HF models.

Plots & visualizations
- Single-dataset dashboard shows: train/test error bars, confusion matrix, core metrics.
- Comparison mode adds: metrics comparison chart and train/test error comparison between datasets.
- VC experiment plots train & test accuracy vs TF‑IDF `max_features` for one or more datasets; the Y-axis is auto-scaled to the observed accuracy range for better visibility.

Performance & tips
- VC experiments train several models — they can be slow. Use the sidebar options to reduce sample size or reduce the number of feature points.
- Use `LinearSVC` for slightly faster runs on some setups; `LogisticRegression` (saga) is also supported.

Troubleshooting
- Missing dataset: app will error and stop; add CSV at `data/sentiment_dataset.csv` with `text`/`label` columns.
- HF dataset issues: install `datasets` (`pip install datasets`) and ensure network access.
- Corrupt model/vectorizer: delete the corresponding files under `saved_models/` and retrain.

Development notes
- The code is modular — helper modules live in `sentiment/` and are easy to test independently.
- If you plan to add new datasets or model types, implement small helpers in `sentiment/` and wire them into `app.py`.

If you'd like, I can:
- Add simple unit tests for `sentiment/hf_utils.py` and `sentiment/plot_utils.py`.
- Add caching for long VC-experiment runs.
- Provide a small sample `data/sentiment_dataset.csv` template.

Enjoy exploring VC‑dimension and comparing models!
