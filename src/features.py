from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_NUMERIC = [
    "is_neutral",
    "same_confederation",
    "team_a_host_flag",
    "team_b_host_flag",
    "team_a_rating",
    "team_b_rating",
    "rating_gap_ab",
    "team_a_form_ppg_5",
    "team_b_form_ppg_5",
    "form_gap_ppg_5",
    "team_a_gd_5",
    "team_b_gd_5",
    "gd_gap_5",
    "team_a_gf_10",
    "team_b_gf_10",
    "team_a_ga_10",
    "team_b_ga_10",
    "team_a_opp_avg_rating_10",
    "team_b_opp_avg_rating_10",
    "team_a_rest_days",
    "team_b_rest_days",
    "rest_gap_ab",
]
FEATURE_CATEGORICAL = ["competition_type", "home_confederation", "away_confederation"]


@dataclass
class TeamState:
    results: Deque[dict]
    last_match_date: pd.Timestamp | None = None


TOURNAMENT_MAP = {
    "FIFA World Cup": "world_cup",
    "World Cup": "world_cup",
    "Friendly": "friendly",
    "UEFA Nations League": "continental_competitive",
    "CONCACAF Nations League": "continental_competitive",
    "AFC Asian Cup": "continental_major",
    "UEFA Euro": "continental_major",
    "Copa América": "continental_major",
    "African Cup of Nations": "continental_major",
}



def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(transformers=[("num", numeric_pipe, FEATURE_NUMERIC), ("cat", categorical_pipe, FEATURE_CATEGORICAL)])



def normalize_competition(tournament: str) -> str:
    t = str(tournament)
    if "qualification" in t.lower() or "qualif" in t.lower():
        return "qualifier"
    return TOURNAMENT_MAP.get(t, "other_competitive")



def _latest_rating_lookup(ratings: pd.DataFrame, match_df: pd.DataFrame, team_col: str, out_prefix: str) -> pd.DataFrame:
    left = match_df[["match_id", "date", team_col]].rename(columns={team_col: "team"}).sort_values("date")
    right = ratings.sort_values("rating_date")
    merged = pd.merge_asof(left, right, left_on="date", right_on="rating_date", by="team", direction="backward", allow_exact_matches=False)
    cols = ["match_id", "rating"] + (["rank"] if "rank" in merged.columns else [])
    return merged[cols].rename(columns={"rating": f"{out_prefix}_rating", "rank": f"{out_prefix}_rank"})



def attach_prematch_ratings(matches: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    base = matches.copy()
    home = _latest_rating_lookup(ratings, base, "home_team", "home")
    away = _latest_rating_lookup(ratings, base, "away_team", "away")
    out = base.merge(home, on="match_id", how="left").merge(away, on="match_id", how="left")
    global_mean = float(ratings["rating"].mean()) if len(ratings) else 1500.0
    out["home_rating"] = out["home_rating"].fillna(global_mean)
    out["away_rating"] = out["away_rating"].fillna(global_mean)
    if "home_rank" in out.columns:
        out["home_rank"] = out["home_rank"].fillna(out["home_rank"].median())
        out["away_rank"] = out["away_rank"].fillna(out["away_rank"].median())
    return out



def _make_team_state_dict() -> Dict[str, TeamState]:
    return defaultdict(lambda: TeamState(results=deque(maxlen=20), last_match_date=None))



def _state_snapshot(state: TeamState, current_date: pd.Timestamp) -> dict:
    res = list(state.results)
    last5 = res[-5:]
    last10 = res[-10:]
    rest_days = int((current_date - state.last_match_date).days) if res else 30

    def _avg(records: List[dict], key: str, default: float = 0.0) -> float:
        return float(np.mean([r[key] for r in records])) if records else default

    def _ppg(records: List[dict]) -> float:
        return float(np.mean([r["points"] for r in records])) if records else 1.0

    return {
        "form_ppg_5": _ppg(last5),
        "form_ppg_10": _ppg(last10),
        "gd_5": _avg(last5, "gd"),
        "gf_10": _avg(last10, "gf"),
        "ga_10": _avg(last10, "ga"),
        "opp_avg_rating_10": _avg(last10, "opp_rating", 1500.0),
        "rest_days": rest_days,
        "matches_seen": len(res),
    }



def build_team_snapshots(matches: pd.DataFrame, ratings: pd.DataFrame, as_of_date: str | pd.Timestamp) -> pd.DataFrame:
    cutoff = pd.Timestamp(as_of_date)
    with_ratings = attach_prematch_ratings(matches, ratings)
    with_ratings = with_ratings[with_ratings["date"] < cutoff].sort_values(["date", "match_id"]).reset_index(drop=True)
    states = _make_team_state_dict()
    for row in with_ratings.itertuples(index=False):
        home_points = 3 if row.home_score > row.away_score else 1 if row.home_score == row.away_score else 0
        away_points = 3 if row.away_score > row.home_score else 1 if row.home_score == row.away_score else 0
        states[row.home_team].results.append({"points": home_points, "gf": row.home_score, "ga": row.away_score, "gd": row.home_score - row.away_score, "opp_rating": row.away_rating})
        states[row.away_team].results.append({"points": away_points, "gf": row.away_score, "ga": row.home_score, "gd": row.away_score - row.home_score, "opp_rating": row.home_rating})
        states[row.home_team].last_match_date = row.date
        states[row.away_team].last_match_date = row.date
    all_teams = sorted(set(matches["home_team"]).union(matches["away_team"]).union(set(ratings["team"])))
    return pd.DataFrame([{**_state_snapshot(states[team], cutoff), "team": team} for team in all_teams])



def build_model_table(matches: pd.DataFrame, ratings: pd.DataFrame, team_metadata: pd.DataFrame) -> pd.DataFrame:
    df = attach_prematch_ratings(matches, ratings)
    meta = team_metadata[["team", "confederation", "host_country_flag"]].copy()
    df = df.merge(meta.rename(columns={"team": "home_team", "confederation": "home_confederation", "host_country_flag": "home_host_flag"}), on="home_team", how="left")
    df = df.merge(meta.rename(columns={"team": "away_team", "confederation": "away_confederation", "host_country_flag": "away_host_flag"}), on="away_team", how="left")
    df["competition_type"] = df["tournament"].map(normalize_competition)
    df["same_confederation"] = (df["home_confederation"].fillna("UNK") == df["away_confederation"].fillna("UNK")).astype(int)
    states = _make_team_state_dict()
    rows = []
    for row in df.sort_values(["date", "match_id"]).itertuples(index=False):
        home_snap = _state_snapshot(states[row.home_team], row.date)
        away_snap = _state_snapshot(states[row.away_team], row.date)
        target = 2 if row.home_score > row.away_score else 1 if row.home_score == row.away_score else 0
        rows.append({
            "match_id": row.match_id,
            "match_date": row.date,
            "team_a": row.home_team,
            "team_b": row.away_team,
            "is_neutral": int(row.neutral),
            "competition_type": row.competition_type,
            "home_confederation": row.home_confederation,
            "away_confederation": row.away_confederation,
            "same_confederation": int(row.same_confederation),
            "team_a_host_flag": int(row.home_host_flag or 0),
            "team_b_host_flag": int(row.away_host_flag or 0),
            "team_a_rating": float(row.home_rating),
            "team_b_rating": float(row.away_rating),
            "rating_gap_ab": float(row.home_rating - row.away_rating),
            "team_a_form_ppg_5": home_snap["form_ppg_5"],
            "team_b_form_ppg_5": away_snap["form_ppg_5"],
            "form_gap_ppg_5": home_snap["form_ppg_5"] - away_snap["form_ppg_5"],
            "team_a_gd_5": home_snap["gd_5"],
            "team_b_gd_5": away_snap["gd_5"],
            "gd_gap_5": home_snap["gd_5"] - away_snap["gd_5"],
            "team_a_gf_10": home_snap["gf_10"],
            "team_b_gf_10": away_snap["gf_10"],
            "team_a_ga_10": home_snap["ga_10"],
            "team_b_ga_10": away_snap["ga_10"],
            "team_a_opp_avg_rating_10": home_snap["opp_avg_rating_10"],
            "team_b_opp_avg_rating_10": away_snap["opp_avg_rating_10"],
            "team_a_rest_days": home_snap["rest_days"],
            "team_b_rest_days": away_snap["rest_days"],
            "rest_gap_ab": home_snap["rest_days"] - away_snap["rest_days"],
            "target_result_3class": target,
            "target_goal_diff": int(row.home_score - row.away_score),
            "target_goals_a": int(row.home_score),
            "target_goals_b": int(row.away_score),
        })
        home_points = 3 if row.home_score > row.away_score else 1 if row.home_score == row.away_score else 0
        away_points = 3 if row.away_score > row.home_score else 1 if row.home_score == row.away_score else 0
        states[row.home_team].results.append({"points": home_points, "gf": row.home_score, "ga": row.away_score, "gd": row.home_score - row.away_score, "opp_rating": row.away_rating})
        states[row.away_team].results.append({"points": away_points, "gf": row.away_score, "ga": row.home_score, "gd": row.away_score - row.home_score, "opp_rating": row.home_rating})
        states[row.home_team].last_match_date = row.date
        states[row.away_team].last_match_date = row.date
    out = pd.DataFrame(rows)
    num_cols = out.select_dtypes(include=["number"]).columns
    out[num_cols] = out[num_cols].fillna(0)
    return out.sort_values(["match_date", "match_id"]).reset_index(drop=True)



def build_future_match_features(team_a: str, team_b: str, as_of_date: str | pd.Timestamp, ratings: pd.DataFrame, snapshots: pd.DataFrame, team_metadata: pd.DataFrame, competition_type: str = "world_cup", is_neutral: int = 1) -> pd.DataFrame:
    as_of_date = pd.Timestamp(as_of_date)
    latest_ratings = ratings[ratings["rating_date"] < as_of_date].sort_values(["team", "rating_date"]).groupby("team", as_index=False).tail(1)[["team", "rating"]]
    rating_map = dict(zip(latest_ratings["team"], latest_ratings["rating"]))
    snap_map = snapshots.set_index("team").to_dict("index")
    meta_map = team_metadata.set_index("team").to_dict("index")
    a = snap_map.get(team_a, {"form_ppg_5": 1.0, "gd_5": 0.0, "gf_10": 1.0, "ga_10": 1.0, "opp_avg_rating_10": 1500.0, "rest_days": 30})
    b = snap_map.get(team_b, {"form_ppg_5": 1.0, "gd_5": 0.0, "gf_10": 1.0, "ga_10": 1.0, "opp_avg_rating_10": 1500.0, "rest_days": 30})
    a_rating = float(rating_map.get(team_a, ratings["rating"].mean()))
    b_rating = float(rating_map.get(team_b, ratings["rating"].mean()))
    a_meta = meta_map.get(team_a, {"confederation": "UNKNOWN", "host_country_flag": 0})
    b_meta = meta_map.get(team_b, {"confederation": "UNKNOWN", "host_country_flag": 0})
    return pd.DataFrame([{
        "team_a": team_a,
        "team_b": team_b,
        "is_neutral": int(is_neutral),
        "competition_type": competition_type,
        "home_confederation": a_meta.get("confederation", "UNKNOWN"),
        "away_confederation": b_meta.get("confederation", "UNKNOWN"),
        "same_confederation": int(a_meta.get("confederation", "UNKNOWN") == b_meta.get("confederation", "UNKNOWN")),
        "team_a_host_flag": int(a_meta.get("host_country_flag", 0)),
        "team_b_host_flag": int(b_meta.get("host_country_flag", 0)),
        "team_a_rating": a_rating,
        "team_b_rating": b_rating,
        "rating_gap_ab": a_rating - b_rating,
        "team_a_form_ppg_5": a["form_ppg_5"],
        "team_b_form_ppg_5": b["form_ppg_5"],
        "form_gap_ppg_5": a["form_ppg_5"] - b["form_ppg_5"],
        "team_a_gd_5": a["gd_5"],
        "team_b_gd_5": b["gd_5"],
        "gd_gap_5": a["gd_5"] - b["gd_5"],
        "team_a_gf_10": a["gf_10"],
        "team_b_gf_10": b["gf_10"],
        "team_a_ga_10": a["ga_10"],
        "team_b_ga_10": b["ga_10"],
        "team_a_opp_avg_rating_10": a["opp_avg_rating_10"],
        "team_b_opp_avg_rating_10": b["opp_avg_rating_10"],
        "team_a_rest_days": a["rest_days"],
        "team_b_rest_days": b["rest_days"],
        "rest_gap_ab": a["rest_days"] - b["rest_days"],
    }])
