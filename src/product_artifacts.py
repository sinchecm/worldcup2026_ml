from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from simulate import build_matchup_cache


def save_model_bundle(models: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(models, f)
    return output_path


def build_team_profiles(
    groups_df: pd.DataFrame,
    ratings: pd.DataFrame,
    snapshots: pd.DataFrame,
    team_metadata: pd.DataFrame,
    champion_probs: pd.DataFrame,
    sim_date: str,
) -> pd.DataFrame:
    as_of = pd.Timestamp(sim_date)
    latest_ratings = (
        ratings[ratings["rating_date"] < as_of]
        .sort_values(["team", "rating_date"])
        .groupby("team", as_index=False)
        .tail(1)[["team", "rating", "rating_date"]]
        .rename(columns={"rating": "latest_rating", "rating_date": "latest_rating_date"})
    )
    out = groups_df.merge(latest_ratings, on="team", how="left")
    out = out.merge(team_metadata[["team", "confederation", "host_country_flag"]], on="team", how="left")
    out = out.merge(
        snapshots[["team", "form_ppg_5", "form_ppg_10", "gd_5", "gf_10", "ga_10", "opp_avg_rating_10", "rest_days", "matches_seen"]],
        on="team",
        how="left",
    )
    out = out.merge(champion_probs, on="team", how="left")
    out = out.rename(
        columns={
            "group": "group_name",
            "host_country_flag": "is_host",
        }
    )
    out["latest_rating"] = out["latest_rating"].round(2)
    return out.sort_values(["champion_pct", "latest_rating"], ascending=[False, False]).reset_index(drop=True)


def build_baseline_matchups(
    models: dict,
    groups_df: pd.DataFrame,
    ratings: pd.DataFrame,
    snapshots: pd.DataFrame,
    team_metadata: pd.DataFrame,
    squad_hooks: pd.DataFrame,
    sim_date: str,
) -> pd.DataFrame:
    teams = sorted(groups_df["team"].unique().tolist())
    matchup_cache = build_matchup_cache(
        models["classifier"],
        models["goal_model_a"],
        models["goal_model_b"],
        teams,
        ratings,
        snapshots,
        team_metadata,
        squad_hooks,
        sim_date,
    )
    rows = []
    for (team_a, team_b), entry in matchup_cache.items():
        rows.append(
            {
                "team_a": team_a,
                "team_b": team_b,
                "prob_team_a_loss": float(entry["probs"][0]),
                "prob_draw": float(entry["probs"][1]),
                "prob_team_a_win": float(entry["probs"][2]),
                "lambda_a": float(entry["lambda_a"]),
                "lambda_b": float(entry["lambda_b"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["team_a", "team_b"]).reset_index(drop=True)


def build_contender_benchmark(
    matches: pd.DataFrame,
    ratings: pd.DataFrame,
    history_finals: pd.DataFrame,
    tournament_years: tuple[int, ...] = (2010, 2014, 2018, 2022),
) -> pd.DataFrame:
    wc = matches[matches["tournament"] == "FIFA World Cup"].copy()
    finals_lookup = history_finals.set_index("year").to_dict("index")
    rows = []
    for year in tournament_years:
        year_wc = wc[wc["date"].dt.year == year].copy()
        participants = sorted(set(year_wc["home_team"]).union(year_wc["away_team"]))
        if not participants:
            continue
        cutoff = pd.Timestamp(f"{year}-06-01")
        rating_slice = (
            ratings[(ratings["rating_date"] < cutoff) & (ratings["team"].isin(participants))]
            .sort_values(["team", "rating_date"])
            .groupby("team", as_index=False)
            .tail(1)[["team", "rating", "rating_date"]]
        )
        if rating_slice.empty:
            continue
        rating_slice = rating_slice.sort_values("rating", ascending=False).reset_index(drop=True)
        rating_slice["contender_rank"] = rating_slice.index + 1
        final_info = finals_lookup.get(year, {})
        champion = final_info.get("champion", "")
        runner_up = final_info.get("runner_up", "")
        rating_slice["year"] = year
        rating_slice["champion"] = champion
        rating_slice["runner_up"] = runner_up
        rating_slice["champion_flag"] = (rating_slice["team"] == champion).astype(int)
        rating_slice["runner_up_flag"] = (rating_slice["team"] == runner_up).astype(int)
        rating_slice["finalist_flag"] = ((rating_slice["team"] == champion) | (rating_slice["team"] == runner_up)).astype(int)
        rows.append(rating_slice)
    if not rows:
        return pd.DataFrame(columns=["year", "team", "rating", "rating_date", "contender_rank", "champion", "runner_up", "champion_flag", "runner_up_flag", "finalist_flag"])
    out = pd.concat(rows, ignore_index=True)
    return out[["year", "team", "rating", "rating_date", "contender_rank", "champion", "runner_up", "champion_flag", "runner_up_flag", "finalist_flag"]].sort_values(["year", "contender_rank"]).reset_index(drop=True)
