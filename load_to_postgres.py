import json
from datetime import datetime
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", 5432)
}

INPUT_FILE = "matches_raw.json"

def clean_match_data(date_str: str):
    if not date_str:
        return None
    try:
        return datetime.strptime(f"{date_str.strip()} 2026", "%d %b %Y").date()
    except Exception:
        return None

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        matches = json.load(f)

    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()

    cursor.execute("SELECT name, id FROM teams;")
    teams = dict(cursor.fetchall())

    def get_team_id(team_name: str) -> int:
        team_name = team_name.strip()
        if team_name in teams:
            return teams[team_name]

        cursor.execute("INSERT INTO teams (name) VALUES (%s) ON CONFLICT (name) DO UPDATE SET name=EXCLUDED.name RETURNING id;", 
                        (team_name,))
        team_id = cursor.fetchone()[0]
        teams[team_name] = team_id
        return team_id

    inserted_matches = 0

    for match in matches:
        teams1_id = get_team_id(match["team1"])
        teams2_id = get_team_id(match["team2"])
        winner_id = get_team_id(match["series_winner"])
        match_date = clean_match_data(match.get("date", ""))

        cursor.execute("""
            INSERT INTO matches (match_url, event_id, event_name, match_date, team1_id, team2_id, series_score, winner_team_id, maps_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (match_url) DO UPDATE SET
                series_score = EXCLUDED.series_score
            RETURNING id;
        """, (
            match["match_url"],
            match.get("event_id"),
            match.get("event_name"),
            match_date,
            teams1_id,
            teams2_id,
            match.get("series_score"),
            winner_id,
            match.get("maps_count", len(match.get("maps", [])))
        ))

        match_id = cursor.fetchone()[0]

        cursor.execute("DELETE FROM match_maps WHERE match_id = %s;", (match_id,))

        for map in match.get("maps", []):
            map_winner_id = get_team_id(map["winner"])

            cursor.execute("""
                INSERT INTO match_maps (match_id, map_name, team1_score, team2_score, winner_team_id)
                VALUES (%s, %s, %s, %s, %s);
            """, (
                match_id,
                map.get("map_name"),
                map.get("team1_score"),
                map.get("team2_score"),
                map_winner_id
            ))

        inserted_matches += 1

    connection.commit()
    cursor.close()
    connection.close()
    print(f"Inserted {inserted_matches} matches into the database.")

if __name__ == "__main__":
    main()