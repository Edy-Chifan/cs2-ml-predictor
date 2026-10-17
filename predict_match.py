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
    """Găsește ID-ul și numele oficial al unei echipe."""
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
    """Calculează statisticile folosind winner_team_id."""
    cutoff_90d = ref_date - timedelta(days=90)

    # 1. Toate seriile anterioare
    cursor.execute("""
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE winner_team_id = %s)
        FROM matches
        WHERE (team1_id = %s OR team2_id = %s)
          AND match_date < %s
          AND winner_team_id IS NOT NULL;
    """, (team_id, team_id, team_id, ref_date))
    all_matches, all_wins = cursor.fetchone()

    # 2. Seriile din ultimele 90 de zile
    cursor.execute("""
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE winner_team_id = %s)
        FROM matches
        WHERE (team1_id = %s OR team2_id = %s)
          AND match_date >= %s AND match_date < %s
          AND winner_team_id IS NOT NULL;
    """, (team_id, team_id, team_id, cutoff_90d, ref_date))
    m90_matches, m90_wins = cursor.fetchone()

    # 3. Toate hărțile
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

    # 4. Hărțile din ultimele 90 de zile
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
    """Calculează meciurile directe anterioare între cele două echipe."""
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
    """Calculează winrate-ul pe o hartă specifică folosind mm.winner_team_id."""
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
        print("Eroare: cs2_model.pkl sau model_features.pkl lipsesc!")
        return

    model = joblib.load("cs2_model.pkl")
    feature_cols = joblib.load("model_features.pkl")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    t1_id, t1_real_name = get_team_id(cur, t1_name)
    t2_id, t2_real_name = get_team_id(cur, t2_name)

    if not t1_id:
        print(f"Echipa '{t1_name}' nu a fost găsită în baza de date.")
        cur.close(); conn.close()
        return
    if not t2_id:
        print(f"Echipa '{t2_name}' nu a fost găsită în baza de date.")
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

    print("=" * 65)
    print(f"  PREDICȚIE MECI: {t1_real_name} vs {t2_real_name} [{match_format.upper()}]")
    print("=" * 65)
    print(f"{t1_real_name:<20} | 90d Series WR: {t1_stats['series_wr_90d']*100:.1f}% ({t1_stats['matches_90d']} meciuri)")
    print(f"{t2_real_name:<20} | 90d Series WR: {t2_stats['series_wr_90d']*100:.1f}% ({t2_stats['matches_90d']} meciuri)")
    print(f"Meciuri directe (H2H): {h2h_stats['h2h_matches_all']} partide | Rata victorii {t1_real_name}: {h2h_stats['h2h_wr_all']*100:.1f}%")
    print("-" * 65)

    if maps and len(maps) > 0:
        print("\n--- ANALIZĂ PE HĂRȚI ---")
        map_probs = []
        for m in maps:
            m_clean = m.strip()
            t1_played, t1_w, t1_wr = compute_map_specific_wr(cur, t1_id, m_clean)
            t2_played, t2_w, t2_wr = compute_map_specific_wr(cur, t2_id, m_clean)
            
            diff_wr = t1_wr - t2_wr
            map_p1 = float(np.clip(base_p1 + (diff_wr * 0.25), 0.05, 0.95))
            map_probs.append(map_p1)

            print(f"Harta {m_clean:<10}: {t1_real_name} ({t1_wr*100:.0f}%, {t1_w}/{t1_played}) vs {t2_real_name} ({t2_wr*100:.0f}%, {t2_w}/{t2_played}) => P({t1_real_name}) = {map_p1*100:.1f}%")

        if match_format.upper() == "BO1":
            final_p1 = map_probs[0]
            print(f"\nPredicție Finală BO1:")
            print(f"  {t1_real_name}: {final_p1*100:.2f}%")
            print(f"  {t2_real_name}: {(1 - final_p1)*100:.2f}%")

        elif match_format.upper() == "BO3" and len(map_probs) >= 3:
            p1, p2, p3 = map_probs[0], map_probs[1], map_probs[2]
            p_2_0 = p1 * p2
            p_2_1 = (p1 * (1 - p2) * p3) + ((1 - p1) * p2 * p3)
            p_0_2 = (1 - p1) * (1 - p2)
            p_1_2 = ((1 - p1) * p2 * (1 - p3)) + (p1 * (1 - p2) * (1 - p3))
            
            final_p1 = p_2_0 + p_2_1
            winner = t1_real_name if final_p1 >= 0.5 else t2_real_name
            print(f"\n--- REZULTAT SERIE BO3 ---")
            print(f"Câștigător probabil: {winner}")
            print(f"Șanse victorie:")
            print(f"  {t1_real_name:<20}: {final_p1*100:.2f}%")
            print(f"  {t2_real_name:<20}: {(1 - final_p1)*100:.2f}%")
            print(f"\nProbabilități scor exact:")
            print(f"  {t1_real_name} 2 - 0: {p_2_0*100:.1f}%")
            print(f"  {t1_real_name} 2 - 1: {p_2_1*100:.1f}%")
            print(f"  {t2_real_name} 2 - 1: {p_1_2*100:.1f}%")
            print(f"  {t2_real_name} 2 - 0: {p_0_2*100:.1f}%")
    else:
        print(f"\nPredicție Generală Serie ({match_format.upper()}):")
        print(f"  {t1_real_name}: {base_p1*100:.2f}%")
        print(f"  {t2_real_name}: {(1 - base_p1)*100:.2f}%")

    cur.close()
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS2 Match Outcome Predictor")
    parser.add_argument("--t1", type=str, required=True, help="Nume echipa 1")
    parser.add_argument("--t2", type=str, required=True, help="Nume echipa 2")
    parser.add_argument("--format", type=str, default="BO3", help="Format: BO1 sau BO3")
    parser.add_argument("--maps", type=str, default=None, help="Hărți separate prin virgulă (ex: Mirage,Inferno,Ancient)")

    args = parser.parse_args()
    map_list = [m.strip() for m in args.maps.split(",")] if args.maps else None
    predict_match(args.t1, args.t2, args.format, map_list)