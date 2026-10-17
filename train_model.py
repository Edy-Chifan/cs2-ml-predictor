import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report, confusion_matrix

def train_and_evaluate():
    # Load dataset
    df = pd.read_csv("cs2_ml_dataset.csv")
    print(f"Total meciuri în dataset: {len(df)}")

    # Chronological sorting by match_date and match_id
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values(by=["match_date", "match_id"]).reset_index(drop=True)

    feature_cols = [
        # Stats & Counts
        "t1_matches_all", "t2_matches_all",
        "t1_matches_90d", "t2_matches_90d",
        "h2h_matches_all", "h2h_matches_90d",
        # Winrate series & Diffs
        "t1_series_wr_all", "t2_series_wr_all", "diff_series_wr_all",
        "t1_series_wr_90d", "t2_series_wr_90d", "diff_series_wr_90d",
        "h2h_wr_all", "h2h_wr_90d",
        # Winrate maps & Diffs
        "t1_map_wr_all", "t2_map_wr_all", "diff_map_wr_all",
        "t1_map_wr_90d", "t2_map_wr_90d", "diff_map_wr_90d"
    ]

    target_col = "target_team1_won"

    # Train/Test Split Chronological (80% Train, 20% Test)
    # No Data Leakage
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Burn-in period eliminated
    train_df = train_df[(train_df["t1_matches_all"] >= 2) & (train_df["t2_matches_all"] >= 2)]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    print(f"Match Train (historical): {len(X_train)}")
    print(f"Match Test (future):   {len(X_test)}")
    print(f"Interval Train: {train_df['match_date'].min().strftime('%Y-%m-%d')} -> {train_df['match_date'].max().strftime('%Y-%m-%d')}")
    print(f"Interval Test:  {test_df['match_date'].min().strftime('%Y-%m-%d')} -> {test_df['match_date'].max().strftime('%Y-%m-%d')}")
    print("=" * 60)

    # Model 1: Logistic Regression (Linear, baseline)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)

    lr_preds = lr.predict(X_test)
    lr_probs = lr.predict_proba(X_test)[:, 1]

    print("\n--- 1. LOGISTIC REGRESSION ---")
    print(f"Accuracy:  {accuracy_score(y_test, lr_preds):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, lr_probs):.4f}")
    print(f"Log Loss:  {log_loss(y_test, lr_probs):.4f}")

    # Model 2: Random Forest (Non-liniar, tree-based)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=10, random_state=42)
    rf.fit(X_train, y_train)

    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]

    print("\n--- 2. RANDOM FOREST ---")
    print(f"Accuracy:  {accuracy_score(y_test, rf_preds):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, rf_probs):.4f}")
    print(f"Log Loss:  {log_loss(y_test, rf_probs):.4f}")

    # Feature Importances
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n--- TOP 7 IMPORTANT FEATURES (RANDOM FOREST) ---")
    print(importances.head(7).to_string())

    joblib.dump(rf, "cs2_model.pkl")
    joblib.dump(feature_cols, "model_features.pkl")

if __name__ == "__main__":
    train_and_evaluate()