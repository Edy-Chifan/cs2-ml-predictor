import json
import os
import re
import time
from datetime import date
from bs4 import BeautifulSoup
from curl_cffi import requests

BASE_URL = "https://www.hltv.org"
OUTPUT_FILE = "matches_raw.json"

# endDate is today's date in YYYY-MM-DD format
TODAY_STR = date.today().strftime("%Y-%m-%d")
EVENTS_ARCHIVE_URL = (
    f"https://www.hltv.org/events/archive"
    f"?startDate=2026-01-01&endDate={TODAY_STR}&prizeMin=175000&prizeMax=2000000"
)

# mimicking a modern Chrome browser to avoid bot detection
session = requests.Session(impersonate="chrome124")

def get_target_events():
    print(f"Fetching tournaments up to {TODAY_STR}...")

    res = session.get(EVENTS_ARCHIVE_URL, timeout=15)
    if res.status_code != 200:
        print(f"Error fetching tournaments: HTTP {res.status_code}")
        return []

    soup = BeautifulSoup(res.text, "html.parser")
    events = []
    seen_ids = set()

    for a in soup.select("a[href*='/events/']"):
        href = a.get("href", "")
        match = re.search(r"^/events/(\d+)/([^/?#]+)", href)
        if match:
            event_id, event_slug = match.groups()
            if event_id not in seen_ids:
                seen_ids.add(event_id)
                events.append({
                    "id": event_id,
                    "name": a.get_text(strip=True) or event_slug,
                    "url": f"{BASE_URL}/events/{event_id}/{event_slug}"
                })

    print(f"Found {len(events)} eligible Tier 1 tournaments.")
    return events

def get_matches_for_event(event_id: str):
    """Fetch all completed matches for a tournament using the HLTV results page."""
    results_url = f"{BASE_URL}/results?event={event_id}"
    time.sleep(1.0)

    res = session.get(results_url, timeout=15)
    if res.status_code != 200:
        return []

    soup = BeautifulSoup(res.text, "html.parser")
    match_links = []

    for a in soup.select(".result-con a"):
        href = a.get("href", "")
        if href.startswith("/matches/"):
            match_links.append(BASE_URL + href)

    return match_links

def extract_all():
    events = get_target_events()
    if not events:
        print("No eligible tournaments found.")
        return

    # 1. Incremental loading: we keep the matches already downloaded previously
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                extracted_records = json.load(f)
            print(f"Found {len(extracted_records)} existing matches in '{OUTPUT_FILE}'.")
        except Exception:
            extracted_records = []
    else:
        extracted_records = []

    # Maintain a set of seen match URLs
    seen_matches = {rec["match_url"] for rec in extracted_records if "match_url" in rec}
    new_matches_count = 0

   # Iterate through each tournament and extract matches
    for ev_idx, ev in enumerate(events, start=1):
        print(f"\n==================================================")
        print(f"[{ev_idx}/{len(events)}] Championship: {ev['name']} (ID: {ev['id']})")
        print(f"==================================================")

        matches = get_matches_for_event(ev["id"])
        print(f"Total matches in championship: {len(matches)}")

        for m_idx, m_url in enumerate(matches, start=1):
            if m_url in seen_matches:
                continue

            try:
                time.sleep(1.2)  # Delay to avoid being blocked by HLTV
                m_res = session.get(m_url, timeout=15)
                if m_res.status_code != 200:
                    continue

                m_soup = BeautifulSoup(m_res.text, "html.parser")

                team_nodes = m_soup.select(".teamsBox .teamName")
                if len(team_nodes) < 2:
                    continue
                team1 = team_nodes[0].get_text(strip=True)
                team2 = team_nodes[1].get_text(strip=True)

                date_el = m_soup.select_one(".date")
                match_date = date_el.get_text(strip=True) if date_el else ""

                # Extracting map results
                map_holders = m_soup.select(".mapholder")
                played_maps = []
                t1_maps_won = 0
                t2_maps_won = 0

                for holder in map_holders:
                    # Skip maps that don't have results
                    # Map 3 of a BO3 which ended 2-0
                    if not holder.select_one(".results-center"):
                        continue

                    map_name_el = holder.select_one(".mapname")
                    map_name = map_name_el.get_text(strip=True) if map_name_el else "Unknown"

                    scores = holder.select(".results-team-score")
                    if len(scores) == 2:
                        try:
                            s1 = int(scores[0].get_text(strip=True))
                            s2 = int(scores[1].get_text(strip=True))
                            map_winner = team1 if s1 > s2 else team2

                            if s1 > s2:
                                t1_maps_won += 1
                            else:
                                t2_maps_won += 1

                            played_maps.append({
                                "map_name": map_name,
                                "team1_score": s1,
                                "team2_score": s2,
                                "winner": map_winner
                            })
                        except ValueError:
                            continue

                if not played_maps:
                    continue

                series_winner = team1 if t1_maps_won > t2_maps_won else team2

                extracted_records.append({
                    "match_url": m_url,
                    "event_id": ev["id"],
                    "event_name": ev["name"],
                    "date": match_date,
                    "team1": team1,
                    "team2": team2,
                    "series_score": f"{t1_maps_won}:{t2_maps_won}",
                    "series_winner": series_winner,
                    "maps_count": len(played_maps),
                    "maps": played_maps
                })

                seen_matches.add(m_url)
                new_matches_count += 1
                print(f"[{m_idx}/{len(matches)}] New match added: {team1} vs {team2} ({t1_maps_won}:{t2_maps_won})")

            except Exception as err:
                print(f"Error processing match {m_url}: {err}")
                continue

    # Save the updated records to the output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(extracted_records, f, indent=4, ensure_ascii=False)

    print(f"\nCompleted! {new_matches_count} new matches added. Total in '{OUTPUT_FILE}': {len(extracted_records)}.")

if __name__ == "__main__":
    extract_all()