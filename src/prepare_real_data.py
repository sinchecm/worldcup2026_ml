from __future__ import annotations

from collections import defaultdict
import re
import tempfile
import unicodedata
import urllib.request

import pandas as pd
import pdfplumber

from config import Config

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
SHOOTOUTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
CONFED_URL = "https://raw.githubusercontent.com/fivethirtyeight/data/master/fifa/fifa_countries_audience.csv"
REGULATIONS_URL = "https://digitalhub.fifa.com/m/636f5c9c6f29771f/original/FWC2026_regulations_EN.pdf"
FIFA_SOURCE_1930_1978 = "https://www.fifa.com/en/tournaments/mens/worldcup/articles/world-cup-champions-1930-1978-uruguay-italy-germany-brazil-england-argentina"
FIFA_SOURCE_1982_2022 = "https://www.fifa.com/en/tournaments/mens/worldcup/articles/world-cup-champions-1982-2022-italy-argentina-germany-brazil-france-spain"

WORLD_CUP_2026_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

HISTORY_ROWS = [
    {"year": 1930, "champion": "Uruguay", "runner_up": "Argentina", "source_url": FIFA_SOURCE_1930_1978, "match_type": "Final"},
    {"year": 1934, "champion": "Italy", "runner_up": "Czechoslovakia", "source_url": FIFA_SOURCE_1930_1978, "match_type": "Final"},
    {"year": 1938, "champion": "Italy", "runner_up": "Hungary", "source_url": FIFA_SOURCE_1930_1978, "match_type": "Final"},
    {"year": 1950, "champion": "Uruguay", "runner_up": "Brazil", "source_url": FIFA_SOURCE_1930_1978, "match_type": "Final group decider"},
    {"year": 1954, "champion": "Germany", "runner_up": "Hungary", "source_url": FIFA_SOURCE_1930_1978, "match_type": "Final"},
    {"year": 1958, "champion": "Brazil", "runner_up": "Sweden", "source_url": FIFA_SOURCE_1930_1978, "match_type": "Final"},
    {"year": 1962, "champion": "Brazil", "runner_up": "Czechoslovakia", "source_url": FIFA_SOURCE_1930_1978, "match_type": "Final"},
    {"year": 1966, "champion": "England", "runner_up": "Germany", "source_url": FIFA_SOURCE_1930_1978, "match_type": "Final"},
    {"year": 1970, "champion": "Brazil", "runner_up": "Italy", "source_url": FIFA_SOURCE_1930_1978, "match_type": "Final"},
    {"year": 1974, "champion": "Germany", "runner_up": "Netherlands", "source_url": FIFA_SOURCE_1930_1978, "match_type": "Final"},
    {"year": 1978, "champion": "Argentina", "runner_up": "Netherlands", "source_url": FIFA_SOURCE_1930_1978, "match_type": "Final"},
    {"year": 1982, "champion": "Italy", "runner_up": "Germany", "source_url": FIFA_SOURCE_1982_2022, "match_type": "Final"},
    {"year": 1986, "champion": "Argentina", "runner_up": "Germany", "source_url": FIFA_SOURCE_1982_2022, "match_type": "Final"},
    {"year": 1990, "champion": "Germany", "runner_up": "Argentina", "source_url": FIFA_SOURCE_1982_2022, "match_type": "Final"},
    {"year": 1994, "champion": "Brazil", "runner_up": "Italy", "source_url": FIFA_SOURCE_1982_2022, "match_type": "Final"},
    {"year": 1998, "champion": "France", "runner_up": "Brazil", "source_url": FIFA_SOURCE_1982_2022, "match_type": "Final"},
    {"year": 2002, "champion": "Brazil", "runner_up": "Germany", "source_url": FIFA_SOURCE_1982_2022, "match_type": "Final"},
    {"year": 2006, "champion": "Italy", "runner_up": "France", "source_url": FIFA_SOURCE_1982_2022, "match_type": "Final"},
    {"year": 2010, "champion": "Spain", "runner_up": "Netherlands", "source_url": FIFA_SOURCE_1982_2022, "match_type": "Final"},
    {"year": 2014, "champion": "Germany", "runner_up": "Argentina", "source_url": FIFA_SOURCE_1982_2022, "match_type": "Final"},
    {"year": 2018, "champion": "France", "runner_up": "Croatia", "source_url": FIFA_SOURCE_1982_2022, "match_type": "Final"},
    {"year": 2022, "champion": "Argentina", "runner_up": "France", "source_url": FIFA_SOURCE_1982_2022, "match_type": "Final"},
]

ALIASES = {
    "curacao": "curaçao",
    "cote divoire": "ivory coast",
    "cote d ivoire": "ivory coast",
    "cape verde": "cape verde",
    "cabo verde": "cape verde",
    "republic of ireland": "ireland",
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "ir iran": "iran",
    "turkiye": "turkey",
    "türkiye": "turkey",
    "usa": "united states",
    "congo dr": "dr congo",
    "congo democratic republic": "dr congo",
    "democratic republic of congo": "dr congo",
    "bosnia-herzegovina": "bosnia and herzegovina",
}

QUALIFIED_CONFEDS = {
    "Algeria": "CAF", "Argentina": "CONMEBOL", "Australia": "AFC", "Austria": "UEFA", "Belgium": "UEFA",
    "Bosnia and Herzegovina": "UEFA", "Brazil": "CONMEBOL", "Canada": "CONCACAF", "Cape Verde": "CAF",
    "Colombia": "CONMEBOL", "Croatia": "UEFA", "Curaçao": "CONCACAF", "Czechia": "UEFA", "DR Congo": "CAF",
    "Ecuador": "CONMEBOL", "Egypt": "CAF", "England": "UEFA", "France": "UEFA", "Germany": "UEFA",
    "Ghana": "CAF", "Haiti": "CONCACAF", "Iran": "AFC", "Iraq": "AFC", "Ivory Coast": "CAF",
    "Japan": "AFC", "Jordan": "AFC", "Mexico": "CONCACAF", "Morocco": "CAF", "Netherlands": "UEFA",
    "New Zealand": "OFC", "Norway": "UEFA", "Panama": "CONCACAF", "Paraguay": "CONMEBOL", "Portugal": "UEFA",
    "Qatar": "AFC", "Saudi Arabia": "AFC", "Scotland": "UEFA", "Senegal": "CAF", "South Africa": "CAF",
    "South Korea": "AFC", "Spain": "UEFA", "Sweden": "UEFA", "Switzerland": "UEFA", "Tunisia": "CAF",
    "Turkey": "UEFA", "United States": "CONCACAF", "Uruguay": "CONMEBOL", "Uzbekistan": "AFC",
}
ANNEX_COLS = ["slot_1A", "slot_1B", "slot_1D", "slot_1E", "slot_1G", "slot_1I", "slot_1K", "slot_1L"]


def normalize_name(value: str) -> str:
    s = str(value).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.replace("&", "and").replace("-", " ")
    s = " ".join(s.split())
    return ALIASES.get(s, s)


def k_factor(tournament: str) -> float:
    t = str(tournament).lower()
    if "world cup" in t and "qualification" not in t and "qualif" not in t:
        return 60.0
    if "euro" in t or "copa am" in t or "african cup" in t or "asian cup" in t or "gold cup" in t:
        return 50.0
    if "qualification" in t or "qualif" in t:
        return 40.0
    if "nations league" in t:
        return 35.0
    if "friendly" in t:
        return 20.0
    return 30.0


def build_elo_ratings(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    ratings = defaultdict(lambda: 1500.0)
    rows = []
    for row in matches.itertuples(index=False):
        rh = ratings[row.home_team]
        ra = ratings[row.away_team]
        home_adv = 80.0 if not bool(row.neutral) else 0.0
        exp_home = 1.0 / (1.0 + 10 ** ((ra - (rh + home_adv)) / 400.0))
        exp_away = 1.0 - exp_home
        if row.home_score > row.away_score:
            s_home, s_away = 1.0, 0.0
        elif row.home_score < row.away_score:
            s_home, s_away = 0.0, 1.0
        else:
            s_home = s_away = 0.5
        k = k_factor(row.tournament)
        ratings[row.home_team] = rh + k * (s_home - exp_home)
        ratings[row.away_team] = ra + k * (s_away - exp_away)
        rows.append({"team": row.home_team, "rating_date": row.date, "rating": round(ratings[row.home_team], 3)})
        rows.append({"team": row.away_team, "rating_date": row.date, "rating": round(ratings[row.away_team], 3)})
    return pd.DataFrame(rows)


def build_team_metadata(matches: pd.DataFrame, confed_df: pd.DataFrame) -> pd.DataFrame:
    confed_map = {normalize_name(country): confed for country, confed in zip(confed_df["country"], confed_df["confederation"])}
    teams = sorted(set(matches["home_team"]).union(matches["away_team"]))
    rows = []
    for team in teams:
        confed = QUALIFIED_CONFEDS.get(team) or confed_map.get(normalize_name(team), "UNKNOWN")
        rows.append({"team": team, "confederation": confed, "host_country_flag": int(team in {"Canada", "Mexico", "United States"})})
    return pd.DataFrame(rows)


def build_groups_table() -> pd.DataFrame:
    return pd.DataFrame([{"group": group, "team": team} for group, teams in WORLD_CUP_2026_GROUPS.items() for team in teams])


def build_annex_c_mapping() -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        urllib.request.urlretrieve(REGULATIONS_URL, tmp.name)
        rows = []
        with pdfplumber.open(tmp.name) as doc:
            for page in doc.pages[79:]:
                text = page.extract_text() or ""
                for line in text.splitlines():
                    m = re.match(r"^(\d+)\s+(3[A-L])\s+(3[A-L])\s+(3[A-L])\s+(3[A-L])\s+(3[A-L])\s+(3[A-L])\s+(3[A-L])\s+(3[A-L])\s*$", line.strip())
                    if m:
                        rows.append(m.groups())
    if len(rows) != 495:
        raise ValueError(f"Expected 495 Annexe C rows, got {len(rows)}")
    df = pd.DataFrame(rows, columns=["option", *ANNEX_COLS])
    df["option"] = df["option"].astype(int)
    df["combo_key"] = df[ANNEX_COLS].apply(lambda row: "".join(sorted(slot[-1] for slot in row)), axis=1)
    return df[["option", "combo_key", *ANNEX_COLS]].sort_values("option").reset_index(drop=True)


def build_squad_hooks() -> pd.DataFrame:
    teams = [team for teams in WORLD_CUP_2026_GROUPS.values() for team in teams]
    return pd.DataFrame({
        "team": teams,
        "snapshot_date": ["2026-06-01"] * len(teams),
        "squad_strength_delta": [0.0] * len(teams),
        "attack_delta": [0.0] * len(teams),
        "defense_delta": [0.0] * len(teams),
        "availability_delta": [0.0] * len(teams),
        "source_url": [""] * len(teams),
        "notes": ["Fill with pre-tournament squad adjustments"] * len(teams),
    })


def build_squad_sources_template() -> pd.DataFrame:
    teams = [team for teams in WORLD_CUP_2026_GROUPS.values() for team in teams]
    return pd.DataFrame({
        "team": teams,
        "as_of_date": ["2026-06-01"] * len(teams),
        "metric_name": [""] * len(teams),
        "metric_value": [""] * len(teams),
        "source_name": [""] * len(teams),
        "source_url": [""] * len(teams),
        "notes": ["Document source used to justify squad hook adjustments"] * len(teams),
    })


def build_past_world_cup_finals(matches: pd.DataFrame, shootouts: pd.DataFrame) -> pd.DataFrame:
    wc = matches[matches["tournament"] == "FIFA World Cup"].copy()
    wc["date"] = pd.to_datetime(wc["date"]).dt.normalize()
    shootouts = shootouts.copy()
    shootouts["date"] = pd.to_datetime(shootouts["date"]).dt.normalize()
    rows = []
    for item in HISTORY_ROWS:
        year = item["year"]
        sub = wc[wc["date"].dt.year == year].copy()
        champion = item["champion"]
        runner_up = item["runner_up"]
        if year == 1950:
            final = sub[((sub["home_team"] == "Brazil") & (sub["away_team"] == "Uruguay")) | ((sub["home_team"] == "Uruguay") & (sub["away_team"] == "Brazil"))].sort_values("date").tail(1)
            note = "No official final was played in 1950; this was the decisive final-group match."
        else:
            candidate = sub[((sub["home_team"] == champion) & (sub["away_team"] == runner_up)) | ((sub["home_team"] == runner_up) & (sub["away_team"] == champion))].sort_values(["date", "home_score", "away_score"])
            final = candidate.tail(1)
            note = ""
        if final.empty:
            raise ValueError(f"Could not locate final data for {year}")
        r = final.iloc[0]
        ft_score = f"{int(r['home_score'])}-{int(r['away_score'])}"
        shoot = shootouts[(shootouts["date"] == r["date"]) & (shootouts["home_team"] == r["home_team"]) & (shootouts["away_team"] == r["away_team"])]
        penalty_winner = shoot.iloc[0]["winner"] if not shoot.empty else ""
        score_display = ft_score if penalty_winner == "" else f"{ft_score} (pens: {penalty_winner})"
        rows.append({
            "year": year,
            "final_date": r["date"].date().isoformat(),
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "home_score": int(r["home_score"]),
            "away_score": int(r["away_score"]),
            "score_display": score_display,
            "champion": champion,
            "runner_up": runner_up,
            "penalty_winner": penalty_winner,
            "match_type": item["match_type"],
            "note": note,
            "fifa_source_url": item["source_url"],
            "results_source_url": RESULTS_URL,
        })
    return pd.DataFrame(rows)


def main() -> None:
    cfg = Config()
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    matches = pd.read_csv(RESULTS_URL)
    matches.to_csv(cfg.matches_path, index=False)
    matches["date"] = pd.to_datetime(matches["date"]).dt.normalize()

    shootouts = pd.read_csv(SHOOTOUTS_URL)
    shootouts.to_csv(cfg.shootouts_path, index=False)

    confed_df = pd.read_csv(CONFED_URL)
    team_metadata = build_team_metadata(matches, confed_df)
    team_metadata.to_csv(cfg.team_metadata_path, index=False)

    ratings = build_elo_ratings(matches)
    ratings.to_csv(cfg.ratings_path, index=False)

    groups = build_groups_table()
    groups.to_csv(cfg.groups_path, index=False)

    annex_c = build_annex_c_mapping()
    annex_c.to_csv(cfg.third_place_mapping_path, index=False)

    squad_hooks = build_squad_hooks()
    squad_hooks.to_csv(cfg.squad_hooks_path, index=False)
    squad_sources = build_squad_sources_template()
    squad_sources.to_csv(cfg.squad_sources_template_path, index=False)

    history = build_past_world_cup_finals(matches, shootouts)
    history.to_csv(cfg.history_finals_path, index=False)

    print(f"Saved real-source files to {cfg.raw_dir}")
    print(f"Matches rows: {len(matches):,}")
    print(f"Shootouts rows: {len(shootouts):,}")
    print(f"Ratings rows: {len(ratings):,}")
    print(f"Metadata rows: {len(team_metadata):,}")
    print(f"Groups rows: {len(groups):,}")
    print(f"Annexe C rows: {len(annex_c):,}")
    print(f"Squad hooks rows: {len(squad_hooks):,}")
    print(f"History rows: {len(history):,}")


if __name__ == "__main__":
    main()
