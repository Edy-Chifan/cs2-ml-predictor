import os
import random
from datetime import timedelta
from dotenv import load_dotenv
import pandas as pd
import psycopg2

load_dotenv()

# Fix the random seed for reproducibility
random.seed(42)

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", 5432)
}

def load_data_from_db():
    connection = psycopg2.connect(**DB_CONFIG)

    matches_query = """
        SELECT
            m.id,
            m.match_date,
            m.team1_id,
            t1.name AS team1_name,
            m.team2_id,
            t2.name AS team2_name,
            m.winner_team_id
        FROM matches m
        JOIN teams t1 ON m.team1_id = t1.id
        JOIN teams t2 ON m.team2_id = t2.id
        WHERE m.match_date IS NOT NULL
          AND m.winner_team_id IS NOT NULL
        ORDER BY m.match_date ASC, m.id ASC;
    """
    df_matches = pd.read_sql(matches_query, connection)

    maps_query = """
        SELECT
            mm.match_id,
            m.match_date,
            m.team1_id,
            m.team2_id,
            mm.winner_team_id
        FROM match_maps mm
        JOIN matches m ON mm.match_id = m.id
        WHERE m.match_date IS NOT NULL
          AND mm.winner_team_id IS NOT NULL;
    """
    df_maps = pd.read_sql(maps_query, connection)
    connection.close()

    df_matches["match_date"] = pd.to_datetime(df_matches["match_date"])
    df_maps["match_date"] = pd.to_datetime(df_maps["match_date"])

    return df_matches, df_maps

def build_features(df_matches, df_maps):
    dataset = []

    for _, row in df_matches.iterrows():
        current_date = row["match_date"]
        cutoff_date = current_date - timedelta(days=90)

        # Random swap to avoid bias in team order
        swap = random.choice([True, False])
        if not swap:
            t1_id, t1_name = row["team1_id"], row["team1_name"]
            t2_id, t2_name = row["team2_id"], row["team2_name"]
        else:
            t1_id, t1_name = row["team2_id"], row["team2_name"]
            t2_id, t2_name = row["team1_id"], row["team1_name"]

        past_matches = df_matches[df_matches["match_date"] < current_date]
        past_maps = df_maps[df_maps["match_date"] < current_date]

        # SERIES STATS & COUNTS
        # Team 1 All-time
        t1_all = past_matches[(past_matches["team1_id"] == t1_id) | (past_matches["team2_id"] == t1_id)]
        t1_n_all = len(t1_all)
        t1_wr_all = (t1_all["winner_team_id"] == t1_id).sum() / t1_n_all if t1_n_all > 0 else 0.5

        # Team 1 Recent 90d
        t1_rec = t1_all[t1_all["match_date"] >= cutoff_date]
        t1_n_90d = len(t1_rec)
        t1_wr_90d = (t1_rec["winner_team_id"] == t1_id).sum() / t1_n_90d if t1_n_90d > 0 else t1_wr_all

        # Team 2 All-time
        t2_all = past_matches[(past_matches["team1_id"] == t2_id) | (past_matches["team2_id"] == t2_id)]
        t2_n_all = len(t2_all)
        t2_wr_all = (t2_all["winner_team_id"] == t2_id).sum() / t2_n_all if t2_n_all > 0 else 0.5

        # Team 2 Recent 90d
        t2_rec = t2_all[t2_all["match_date"] >= cutoff_date]
        t2_n_90d = len(t2_rec)
        t2_wr_90d = (t2_rec["winner_team_id"] == t2_id).sum() / t2_n_90d if t2_n_90d > 0 else t2_wr_all

        # HEAD TO HEAD 
        h2h_all = past_matches[
            ((past_matches["team1_id"] == t1_id) & (past_matches["team2_id"] == t2_id)) |
            ((past_matches["team1_id"] == t2_id) & (past_matches["team2_id"] == t1_id))
        ]
        h2h_n_all = len(h2h_all)
        h2h_wr_all = (h2h_all["winner_team_id"] == t1_id).sum() / h2h_n_all if h2h_n_all > 0 else 0.5

        h2h_rec = h2h_all[h2h_all["match_date"] >= cutoff_date]
        h2h_n_90d = len(h2h_rec)
        h2h_wr_90d = (h2h_rec["winner_team_id"] == t1_id).sum() / h2h_n_90d if h2h_n_90d > 0 else h2h_wr_all

        # MAP STATS & COUNTS
        # Maps Team 1
        t1_maps = past_maps[(past_maps["team1_id"] == t1_id) | (past_maps["team2_id"] == t1_id)]
        t1_maps_n_all = len(t1_maps)
        t1_map_wr_all = (t1_maps["winner_team_id"] == t1_id).sum() / t1_maps_n_all if t1_maps_n_all > 0 else 0.5

        t1_maps_rec = t1_maps[t1_maps["match_date"] >= cutoff_date]
        t1_maps_n_90d = len(t1_maps_rec)
        t1_map_wr_90d = (t1_maps_rec["winner_team_id"] == t1_id).sum() / t1_maps_n_90d if t1_maps_n_90d > 0 else t1_map_wr_all

        # Maps Team 2
        t2_maps = past_maps[(past_maps["team1_id"] == t2_id) | (past_maps["team2_id"] == t2_id)]
        t2_maps_n_all = len(t2_maps)
        t2_map_wr_all = (t2_maps["winner_team_id"] == t2_id).sum() / t2_maps_n_all if t2_maps_n_all > 0 else 0.5

        t2_maps_rec = t2_maps[t2_maps["match_date"] >= cutoff_date]
        t2_maps_n_90d = len(t2_maps_rec)
        t2_map_wr_90d = (t2_maps_rec["winner_team_id"] == t2_id).sum() / t2_maps_n_90d if t2_maps_n_90d > 0 else t2_map_wr_all

        target = 1 if row["winner_team_id"] == t1_id else 0

        dataset.append({
            "match_id": row["id"],
            "match_date": current_date.strftime("%Y-%m-%d"),
            "team1": t1_name,
            "team2": t2_name,
            # Stats & Counts
            "t1_matches_all": t1_n_all,
            "t2_matches_all": t2_n_all,
            "t1_matches_90d": t1_n_90d,
            "t2_matches_90d": t2_n_90d,
            "h2h_matches_all": h2h_n_all,
            "h2h_matches_90d": h2h_n_90d,
            # Winrates & Diffs
            "t1_series_wr_all": round(t1_wr_all, 3),
            "t2_series_wr_all": round(t2_wr_all, 3),
            "diff_series_wr_all": round(t1_wr_all - t2_wr_all, 3),
            "t1_series_wr_90d": round(t1_wr_90d, 3),
            "t2_series_wr_90d": round(t2_wr_90d, 3),
            "diff_series_wr_90d": round(t1_wr_90d - t2_wr_90d, 3),
            "h2h_wr_all": round(h2h_wr_all, 3),
            "h2h_wr_90d": round(h2h_wr_90d, 3),
            # Maps Winrates & Diffs
            "t1_map_wr_all": round(t1_map_wr_all, 3),
            "t2_map_wr_all": round(t2_map_wr_all, 3),
            "diff_map_wr_all": round(t1_map_wr_all - t2_map_wr_all, 3),
            "t1_map_wr_90d": round(t1_map_wr_90d, 3),
            "t2_map_wr_90d": round(t2_map_wr_90d, 3),
            "diff_map_wr_90d": round(t1_map_wr_90d - t2_map_wr_90d, 3),
            # Target
            "target_team1_won": target
        })

    return pd.DataFrame(dataset)

if __name__ == "__main__":
    df_matches, df_maps = load_data_from_db()
    print(f"Generating features for {len(df_matches)} matches...")
    df_features = build_features(df_matches, df_maps)
    
    # Verify the distribution of the target variable
    balance = df_features["target_team1_won"].mean()
    print(f"Distribution of target_team1_won: {balance:.2%}")
    
    df_features.to_csv("cs2_ml_dataset.csv", index=False)
    print("Saved to cs2_ml_dataset.csv!")