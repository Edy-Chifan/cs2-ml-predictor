import os
import argparse
import joblib
import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", 5432)
}

def get_team_id(cursor, team_name):
    cursor.execute("SELECT id, name FROM teams WHERE LOWER(name) = LOWER(%s) LIMIT 1;", (team_name,))
    row = cursor.fetchone()
    if row:
        return row[0], row[1]
    
    cursor.execute("SELECT id, name FROM teams WHERE LOWER(name) LIKE LOWER(%s) LIMIT 1;", (f"%{team_name}%",))
    row = cursor.fetchone()
    if row:
        return row[0], row[1]
    return None, None

def compute_team_stats(cursor, team_id, ref_date):
    """Calculate aggregated historical and rolling 90-day statistics."""
    cutoff_90d = ref_date - timedelta(days=90)

    # All-time series
    cursor.execute("""
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE winner_team_id = %s)
        FROM matches
        WHERE (team1_id = %s OR team2_id = %s)
          AND match_date < %s
          AND winner_team_id IS NOT NULL;
    """, (team_id, team_id, team_id, ref_date))
    all_matches, all_wins = cursor.fetchone()

    # Rolling 90-day series
    cursor.execute("""
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE winner_team_id = %s)
        FROM matches
        WHERE (team1_id = %s OR team2_id = %s)
          AND match_date >= %s AND match_date < %s
          AND winner_team_id IS NOT NULL;
    """, (team_id, team_id, team_id, cutoff_90d, ref_date))
    m90_matches, m90_wins = cursor.fetchone()

    # All-time maps
    cursor.execute("""
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE mm.winner_team_id = %s)
        FROM match_maps mm
        JOIN matches m ON mm.match_id = m.id
        WHERE (m.team1_id = %s OR m.team2_id = %s)
          AND m.match_date < %s
          AND mm.winner_team_id IS NOT NULL;
    """, (team_id, team_id, team_id, ref_date))
    all_maps, all_map_wins = cursor.fetchone()

    # Rolling 90-day maps
    cursor.execute("""
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE mm.winner_team_id = %s)
        FROM match_maps mm
        JOIN matches m ON mm.match_id = m.id
        WHERE (m.team1_id = %s OR m.team2_id = %s)
          AND m.match_date >= %s AND m.match_date < %s
          AND mm.winner_team_id IS NOT NULL;
    """, (team_id, team_id, team_id, cutoff_90d, ref_date))
    m90_maps, m90_map_wins = cursor.fetchone()

    series_wr_all = (all_wins / all_matches) if all_matches and all_matches > 0 else 0.5
    series_wr_90d = (m90_wins / m90_matches) if m90_matches and m90_matches > 0 else series_wr_all
    map_wr_all = (all_map_wins / all_maps) if all_maps and all_maps > 0 else 0.5
    map_wr_90d = (m90_map_wins / m90_maps) if m90_maps and m90_maps > 0 else map_wr_all

    return {
        "matches_all": all_matches or 0,
        "matches_90d": m90_matches or 0,
        "series_wr_all": round(series_wr_all, 3),
        "series_wr_90d": round(series_wr_90d, 3),
        "map_wr_all": round(map_wr_all, 3),
        "map_wr_90d": round(map_wr_90d, 3)
    }

def compute_h2h_stats(cursor, t1_id, t2_id, ref_date):
    """Calculate head-to-head metrics between the two teams."""
    cutoff_90d = ref_date - timedelta(days=90)

    cursor.execute("""
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE winner_team_id = %s)
        FROM matches
        WHERE ((team1_id = %s AND team2_id = %s) OR (team1_id = %s AND team2_id = %s))
          AND match_date < %s
          AND winner_team_id IS NOT NULL;
    """, (t1_id, t1_id, t2_id, t2_id, t1_id, ref_date))
    h2h_all, h2h_all_wins = cursor.fetchone()

    cursor.execute("""
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE winner_team_id = %s)
        FROM matches
        WHERE ((team1_id = %s AND team2_id = %s) OR (team1_id = %s AND team2_id = %s))
          AND match_date >= %s AND match_date < %s
          AND winner_team_id IS NOT NULL;
    """, (t1_id, t1_id, t2_id, t2_id, t1_id, cutoff_90d, ref_date))
    h2h_90d, h2h_90d_wins = cursor.fetchone()

    h2h_wr_all = (h2h_all_wins / h2h_all) if h2h_all and h2h_all > 0 else 0.5
    h2h_wr_90d = (h2h_90d_wins / h2h_90d) if h2h_90d and h2h_90d > 0 else h2h_wr_all

    return {
        "h2h_matches_all": h2h_all or 0,
        "h2h_matches_90d": h2h_90d or 0,
        "h2h_wr_all": round(h2h_wr_all, 3),
        "h2h_wr_90d": round(h2h_wr_90d, 3)
    }

def compute_map_specific_wr(cursor, team_id, map_name):
    """Calculate historical win rate on a specific map."""
    cursor.execute("""
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE mm.winner_team_id = %s)
        FROM match_maps mm
        JOIN matches m ON mm.match_id = m.id
        WHERE (m.team1_id = %s OR m.team2_id = %s)
          AND LOWER(mm.map_name) LIKE LOWER(%s)
          AND mm.winner_team_id IS NOT NULL;
    """, (team_id, team_id, team_id, f"%{map_name}%"))
    total_played, wins = cursor.fetchone()
    total = total_played or 0
    w = wins or 0
    wr = (w / total) if total > 0 else 0.5
    return total, w, wr

def predict_match(t1_name, t2_name, match_format="BO3", maps=None):
    if not os.path.exists("cs2_model.pkl") or not os.path.exists("model_features.pkl"):
        print("Error: cs2_model.pkl or model_features.pkl not found! Run train_model.py first.")
        return

    model = joblib.load("cs2_model.pkl")
    feature_cols = joblib.load("model_features.pkl")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    t1_id, t1_real_name = get_team_id(cur, t1_name)
    t2_id, t2_real_name = get_team_id(cur, t2_name)

    if not t1_id:
        print(f"Error: Team '{t1_name}' was not found in database.")
        cur.close(); conn.close()
        return
    if not t2_id:
        print(f"Error: Team '{t2_name}' was not found in database.")
        cur.close(); conn.close()
        return

    ref_date = datetime.now()
    t1_stats = compute_team_stats(cur, t1_id, ref_date)
    t2_stats = compute_team_stats(cur, t2_id, ref_date)
    h2h_stats = compute_h2h_stats(cur, t1_id, t2_id, ref_date)

    sample = {
        "t1_matches_all": t1_stats["matches_all"],
        "t2_matches_all": t2_stats["matches_all"],
        "t1_matches_90d": t1_stats["matches_90d"],
        "t2_matches_90d": t2_stats["matches_90d"],
        "h2h_matches_all": h2h_stats["h2h_matches_all"],
        "h2h_matches_90d": h2h_stats["h2h_matches_90d"],

        "t1_series_wr_all": t1_stats["series_wr_all"],
        "t2_series_wr_all": t2_stats["series_wr_all"],
        "diff_series_wr_all": round(t1_stats["series_wr_all"] - t2_stats["series_wr_all"], 3),

        "t1_series_wr_90d": t1_stats["series_wr_90d"],
        "t2_series_wr_90d": t2_stats["series_wr_90d"],
        "diff_series_wr_90d": round(t1_stats["series_wr_90d"] - t2_stats["series_wr_90d"], 3),

        "h2h_wr_all": h2h_stats["h2h_wr_all"],
        "h2h_wr_90d": h2h_stats["h2h_wr_90d"],

        "t1_map_wr_all": t1_stats["map_wr_all"],
        "t2_map_wr_all": t2_stats["map_wr_all"],
        "diff_map_wr_all": round(t1_stats["map_wr_all"] - t2_stats["map_wr_all"], 3),

        "t1_map_wr_90d": t1_stats["map_wr_90d"],
        "t2_map_wr_90d": t2_stats["map_wr_90d"],
        "diff_map_wr_90d": round(t1_stats["map_wr_90d"] - t2_stats["map_wr_90d"], 3)
    }

    df_sample = pd.DataFrame([sample])[feature_cols]
    base_p1 = model.predict_proba(df_sample)[0, 1]

    print("=" * 68)
    print(f"  CS2 MATCH PREDICTION: {t1_real_name} vs {t2_real_name} [{match_format.upper()}]")
    print("=" * 68)
    print(f"{t1_real_name:<20} | 90d Series WR: {t1_stats['series_wr_90d']*100:.1f}% ({t1_stats['matches_90d']} matches)")
    print(f"{t2_real_name:<20} | 90d Series WR: {t2_stats['series_wr_90d']*100:.1f}% ({t2_stats['matches_90d']} matches)")
    print(f"Head-to-Head (H2H):  {h2h_stats['h2h_matches_all']} matches | {t1_real_name} win rate: {h2h_stats['h2h_wr_all']*100:.1f}%")
    print("-" * 68)

    fmt = match_format.upper()

    if maps and len(maps) > 0:
        print("\n--- MAP-SPECIFIC BREAKDOWN ---")
        map_probs = []
        for m in maps:
            m_clean = m.strip()
            t1_played, t1_w, t1_wr = compute_map_specific_wr(cur, t1_id, m_clean)
            t2_played, t2_w, t2_wr = compute_map_specific_wr(cur, t2_id, m_clean)
            
            diff_wr = t1_wr - t2_wr
            map_p1 = float(np.clip(base_p1 + (diff_wr * 0.25), 0.05, 0.95))
            map_probs.append(map_p1)

            print(f"Map {m_clean:<10}: {t1_real_name} ({t1_wr*100:.0f}%, {t1_w}/{t1_played}) vs {t2_real_name} ({t2_wr*100:.0f}%, {t2_w}/{t2_played}) => P({t1_real_name}) = {map_p1*100:.1f}%")

        if fmt == "BO1":
            final_p1 = map_probs[0]
            winner = t1_real_name if final_p1 >= 0.5 else t2_real_name
            print(f"\n--- BO1 SERIES RESULT ---")
            print(f"Predicted Winner: {winner}")
            print(f"Win Probabilities:")
            print(f"  {t1_real_name:<20}: {final_p1*100:.2f}%")
            print(f"  {t2_real_name:<20}: {(1 - final_p1)*100:.2f}%")

        elif fmt == "BO3":
            if len(map_probs) < 3:
                print(f"\nWarning: BO3 requires 3 maps to simulate accurately. Provided: {len(map_probs)} maps.")
                final_p1 = base_p1
            else:
                p1, p2, p3 = map_probs[0], map_probs[1], map_probs[2]
                q1, q2, q3 = 1 - p1, 1 - p2, 1 - p3

                p_2_0 = p1 * p2
                p_2_1 = (p1 * q2 * p3) + (q1 * p2 * p3)
                p_0_2 = q1 * q2
                p_1_2 = (q1 * p2 * q3) + (p1 * q2 * q3)

                final_p1 = p_2_0 + p_2_1
                winner = t1_real_name if final_p1 >= 0.5 else t2_real_name

                print(f"\n--- BO3 SERIES RESULT ---")
                print(f"Predicted Winner: {winner}")
                print(f"Series Win Probabilities:")
                print(f"  {t1_real_name:<20}: {final_p1*100:.2f}%")
                print(f"  {t2_real_name:<20}: {(1 - final_p1)*100:.2f}%")
                print(f"\nExact Score Probabilities:")
                print(f"  {t1_real_name} 2 - 0: {p_2_0*100:.1f}%")
                print(f"  {t1_real_name} 2 - 1: {p_2_1*100:.1f}%")
                print(f"  {t2_real_name} 2 - 1: {p_1_2*100:.1f}%")
                print(f"  {t2_real_name} 2 - 0: {p_0_2*100:.1f}%")

        elif fmt == "BO5":
            if len(map_probs) < 5:
                print(f"\nWarning: BO5 requires 5 maps to simulate accurately. Provided: {len(map_probs)} maps.")
            else:
                p = map_probs[:5]
                q = [1 - x for x in p]

                # Exact outcomes for Team 1 winning (First to 3 maps):
                # 3-0: wins first 3
                p_3_0 = p[0] * p[1] * p[2]

                # 3-1: exactly 2 wins in first 3, then wins map 4
                p_3_1 = ((p[0]*p[1]*q[2]) + (p[0]*q[1]*p[2]) + (q[0]*p[1]*p[2])) * p[3]

                # 3-2: exactly 2 wins in first 4, then wins map 5
                four_map_2wins = (
                    (p[0]*p[1]*q[2]*q[3]) + (p[0]*q[1]*p[2]*q[3]) + (p[0]*q[1]*q[2]*p[3]) +
                    (q[0]*p[1]*p[2]*q[3]) + (q[0]*p[1]*q[2]*p[3]) + (q[0]*q[1]*p[2]*p[3])
                )
                p_3_2 = four_map_2wins * p[4]

                # Exact outcomes for Team 2 winning:
                # 0-3: team 2 wins first 3
                p_0_3 = q[0] * q[1] * q[2]

                # 1-3: team 2 gets 2 wins in first 3, then wins map 4
                p_1_3 = ((q[0]*q[1]*p[2]) + (q[0]*p[1]*q[2]) + (p[0]*q[1]*q[2])) * q[3]

                # 2-3: team 2 gets 2 wins in first 4, then wins map 5
                four_map_2losses = (
                    (q[0]*q[1]*p[2]*p[3]) + (q[0]*p[1]*q[2]*p[3]) + (q[0]*p[1]*p[2]*q[3]) +
                    (p[0]*q[1]*q[2]*p[3]) + (p[0]*q[1]*p[2]*q[3]) + (p[0]*p[1]*q[2]*q[3])
                )
                p_2_3 = four_map_2losses * q[4]

                final_p1 = p_3_0 + p_3_1 + p_3_2
                winner = t1_real_name if final_p1 >= 0.5 else t2_real_name

                print(f"\n--- BO5 SERIES RESULT ---")
                print(f"Predicted Winner: {winner}")
                print(f"Series Win Probabilities:")
                print(f"  {t1_real_name:<20}: {final_p1*100:.2f}%")
                print(f"  {t2_real_name:<20}: {(1 - final_p1)*100:.2f}%")
                print(f"\nExact Score Probabilities:")
                print(f"  {t1_real_name} 3 - 0: {p_3_0*100:.1f}%")
                print(f"  {t1_real_name} 3 - 1: {p_3_1*100:.1f}%")
                print(f"  {t1_real_name} 3 - 2: {p_3_2*100:.1f}%")
                print(f"  {t2_real_name} 3 - 2: {p_2_3*100:.1f}%")
                print(f"  {t2_real_name} 3 - 1: {p_1_3*100:.1f}%")
                print(f"  {t2_real_name} 3 - 0: {p_0_3*100:.1f}%")

    else:
        # Generic series prediction when maps are not yet specified
        print(f"\nGeneral Series Prediction ({fmt}):")
        print(f"  {t1_real_name}: {base_p1*100:.2f}%")
        print(f"  {t2_real_name}: {(1 - base_p1)*100:.2f}%")

    cur.close()
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS2 Professional Match Outcome Predictor")
    parser.add_argument("--t1", type=str, required=True, help="Name of Team 1")
    parser.add_argument("--t2", type=str, required=True, help="Name of Team 2")
    parser.add_argument("--format", type=str, default="BO3", choices=["BO1", "BO3", "BO5", "bo1", "bo3", "bo5"], help="Match format: BO1, BO3, or BO5")
    parser.add_argument("--maps", type=str, default=None, help="Comma-separated list of maps in played order (e.g., Mirage,Inferno,Ancient,Nuke,Dust2)")

    args = parser.parse_args()
    map_list = [m.strip() for m in args.maps.split(",")] if args.maps else None
    predict_match(args.t1, args.t2, args.format, map_list)