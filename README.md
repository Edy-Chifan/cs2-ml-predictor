# CS2 Professional Match Outcome Predictor

An end-to-end Machine Learning pipeline that predicts competitive Counter-Strike 2 (CS2) professional match outcomes using historical series telemetry, leakage-free temporal feature engineering, and map-specific simulations.

---

## Architecture Overview

1. **Scraping (`extract_matches.py`)**: Crawls professional series from HLTV using `DrissionPage` and `BeautifulSoup` with anti-bot bypass and incremental JSON storage.
2. **Relational Database (`load_to_postgres.py`)**: Stores telemetry in a normalized **PostgreSQL** schema (`teams`, `matches`, `match_maps`) using automated upserts via `psycopg2`.
3. **Feature Engineering (`build_features.py`)**: Extracts historical and 90-day rolling win rates, map performance, and H2H records across 650+ series without future data leakage, maintaining a 50.69% balanced target.
4. **Model Training (`train_model.py`)**: Evaluates linear and tree-based classifiers using chronological train/test splits (434 train / 131 out-of-sample test).
5. **Inference CLI (`predict_match.py`)**: Simulates series outcomes (BO1, BO3, BO5) and exact score probabilities based on map vetoes.

---

## Model Performance (Out-of-Sample Test)

Chronological test set (August 2026 – October 2026):

| Model | Accuracy | ROC-AUC | Log Loss |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | **61.07%** | **0.6629** | 0.6889 |
| **Random Forest** | 60.31% | 0.6522 | 0.6527 |

**Top Features:** `diff_map_wr_all`, `diff_map_wr_90d`, `diff_series_wr_90d`, `t1_matches_all`.
