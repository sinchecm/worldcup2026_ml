from __future__ import annotations

from itertools import combinations
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from features import build_future_match_features
from train import FEATURE_CATEGORICAL, FEATURE_NUMERIC

ANNEX_WINNER_COLUMNS = ["slot_1A", "slot_1B", "slot_1D", "slot_1E", "slot_1G", "slot_1I", "slot_1K", "slot_1L"]
R32_FIXED = {
    73: ("2A", "2B"),
    75: ("1F", "2C"),
    76: ("1C", "2F"),
    78: ("2E", "2I"),
    83: ("2K", "2L"),
    84: ("1H", "2J"),
    86: ("1J", "2H"),
    88: ("2D", "2G"),
}
R32_THIRD = {
    74: ("1E", "slot_1E"),
    77: ("1I", "slot_1I"),
    79: ("1A", "slot_1A"),
    80: ("1L", "slot_1L"),
    81: ("1D", "slot_1D"),
    82: ("1G", "slot_1G"),
    85: ("1B", "slot_1B"),
    87: ("1K", "slot_1K"),
}
ROUND_MAPS = {
    89: (74, 77),
    90: (73, 75),
    91: (76, 78),
    92: (79, 80),
    93: (83, 84),
    94: (81, 82),
    95: (86, 88),
    96: (85, 87),
    97: (89, 90),
    98: (93, 94),
    99: (91, 92),
    100: (95, 96),
    101: (97, 98),
    102: (99, 100),
    104: (101, 102),
}



def _predict_match(model, feature_row: pd.DataFrame) -> np.ndarray:
    X = feature_row[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
    return model.predict_proba(X)[0]



def _latest_squad_hook_map(squad_hooks: pd.DataFrame, sim_date: str) -> dict:
    if squad_hooks.empty:
        return {}
    eligible = squad_hooks[squad_hooks["snapshot_date"] <= pd.Timestamp(sim_date)].copy()
    if eligible.empty:
        eligible = squad_hooks.copy()
    latest = eligible.sort_values(["team", "snapshot_date"]).groupby("team", as_index=False).tail(1)
    return latest.set_index("team").to_dict("index")



def _apply_squad_adjustments(entry: dict, squad_a: dict, squad_b: dict) -> dict:
    squad_gap = float(squad_a.get("squad_strength_delta", 0.0) - squad_b.get("squad_strength_delta", 0.0))
    availability_gap = float(squad_a.get("availability_delta", 0.0) - squad_b.get("availability_delta", 0.0))
    attack_vs_defense_a = float(squad_a.get("attack_delta", 0.0) - squad_b.get("defense_delta", 0.0))
    attack_vs_defense_b = float(squad_b.get("attack_delta", 0.0) - squad_a.get("defense_delta", 0.0))

    probs = entry["probs"].copy()
    shift = 0.18 * squad_gap + 0.12 * attack_vs_defense_a + 0.08 * availability_gap
    probs[2] *= np.exp(shift)
    probs[0] *= np.exp(-shift)
    probs = probs / probs.sum()

    lambda_a = float(np.clip(entry["lambda_a"] * np.exp(0.15 * attack_vs_defense_a + 0.08 * availability_gap + 0.05 * squad_gap), 0.05, 5.0))
    lambda_b = float(np.clip(entry["lambda_b"] * np.exp(0.15 * attack_vs_defense_b - 0.08 * availability_gap - 0.05 * squad_gap), 0.05, 5.0))
    return {**entry, "probs": probs, "lambda_a": lambda_a, "lambda_b": lambda_b}



def build_matchup_cache(classifier_model, goal_model_a, goal_model_b, teams: list[str], ratings: pd.DataFrame, snapshots: pd.DataFrame, team_metadata: pd.DataFrame, squad_hooks: pd.DataFrame, sim_date: str) -> Dict[Tuple[str, str], dict]:
    cache: Dict[Tuple[str, str], dict] = {}
    squad_map = _latest_squad_hook_map(squad_hooks, sim_date)
    for team_a, team_b in combinations(sorted(teams), 2):
        feat = build_future_match_features(team_a, team_b, sim_date, ratings, snapshots, team_metadata, competition_type="world_cup", is_neutral=1)
        probs_ab = _predict_match(classifier_model, feat)
        lambda_a = float(np.clip(goal_model_a.predict(feat[FEATURE_NUMERIC + FEATURE_CATEGORICAL])[0], 0.05, 5.0))
        lambda_b = float(np.clip(goal_model_b.predict(feat[FEATURE_NUMERIC + FEATURE_CATEGORICAL])[0], 0.05, 5.0))
        base_ab = {"probs": probs_ab, "lambda_a": lambda_a, "lambda_b": lambda_b}
        base_ba = {"probs": np.array([probs_ab[2], probs_ab[1], probs_ab[0]]), "lambda_a": lambda_b, "lambda_b": lambda_a}
        cache[(team_a, team_b)] = _apply_squad_adjustments(base_ab, squad_map.get(team_a, {}), squad_map.get(team_b, {}))
        cache[(team_b, team_a)] = _apply_squad_adjustments(base_ba, squad_map.get(team_b, {}), squad_map.get(team_a, {}))
    return cache



def init_group_table(groups_df: pd.DataFrame, ratings: pd.DataFrame, sim_date: str) -> pd.DataFrame:
    latest_ratings = (
        ratings[ratings["rating_date"] < pd.Timestamp(sim_date)]
        .sort_values(["team", "rating_date"])
        .groupby("team", as_index=False)
        .tail(1)[["team", "rating"]]
    )
    table = groups_df.copy()
    table = table.merge(latest_ratings, on="team", how="left")
    table["rating"] = table["rating"].fillna(ratings["rating"].mean())
    for col in ["pts", "gf", "ga", "gd", "wins", "draws", "losses"]:
        table[col] = 0
    return table



def simulate_group_stage(groups_df: pd.DataFrame, ratings: pd.DataFrame, sim_date: str, matchup_cache: Dict[Tuple[str, str], dict], rng: np.random.Generator) -> pd.DataFrame:
    table = init_group_table(groups_df, ratings, sim_date)
    for group_name, sub in groups_df.groupby("group"):
        teams = sub["team"].tolist()
        if len(teams) != 4:
            raise ValueError(f"Group {group_name} does not have 4 teams")
        for team_a, team_b in combinations(teams, 2):
            entry = matchup_cache[(team_a, team_b)]
            goals_a = int(rng.poisson(entry["lambda_a"]))
            goals_b = int(rng.poisson(entry["lambda_b"]))
            for team, gf_add, ga_add in [(team_a, goals_a, goals_b), (team_b, goals_b, goals_a)]:
                table.loc[table["team"] == team, "gf"] += gf_add
                table.loc[table["team"] == team, "ga"] += ga_add
                table.loc[table["team"] == team, "gd"] += gf_add - ga_add
            if goals_a > goals_b:
                table.loc[table["team"] == team_a, ["pts", "wins"]] += [3, 1]
                table.loc[table["team"] == team_b, ["losses"]] += [1]
            elif goals_b > goals_a:
                table.loc[table["team"] == team_b, ["pts", "wins"]] += [3, 1]
                table.loc[table["team"] == team_a, ["losses"]] += [1]
            else:
                table.loc[table["team"] == team_a, ["pts", "draws"]] += [1, 1]
                table.loc[table["team"] == team_b, ["pts", "draws"]] += [1, 1]
    ranked_parts = []
    for _, sub in table.groupby("group"):
        ranked = sub.sort_values(["pts", "gd", "gf", "rating"], ascending=[False, False, False, False]).reset_index(drop=True)
        ranked["group_place"] = ranked.index + 1
        ranked_parts.append(ranked)
    return pd.concat(ranked_parts, ignore_index=True)



def _build_slot_lookup(group_table: pd.DataFrame) -> Dict[str, str]:
    return {f"{row.group_place}{row.group}": row.team for row in group_table.itertuples(index=False)}



def _best_third_placed(group_table: pd.DataFrame) -> pd.DataFrame:
    return group_table[group_table["group_place"] == 3].sort_values(["pts", "gd", "gf", "rating"], ascending=[False, False, False, False]).head(8).reset_index(drop=True)



def build_round_of_32_fixtures(group_table: pd.DataFrame, third_place_mapping: pd.DataFrame) -> tuple[Dict[int, Tuple[str, str]], list[str]]:
    slot_to_team = _build_slot_lookup(group_table)
    best_thirds = _best_third_placed(group_table)
    for row in best_thirds.itertuples(index=False):
        slot_to_team[f"3{row.group}"] = row.team
    combo_key = "".join(sorted(best_thirds["group"].tolist()))
    match_row = third_place_mapping[third_place_mapping["combo_key"] == combo_key]
    if match_row.empty:
        raise ValueError(f"Missing Annexe C mapping for third-place combination {combo_key}")
    mapping = match_row.iloc[0].to_dict()

    fixtures: Dict[int, Tuple[str, str]] = {}
    for match_no, (slot_a, slot_b) in R32_FIXED.items():
        fixtures[match_no] = (slot_to_team[slot_a], slot_to_team[slot_b])
    for match_no, (winner_slot, third_slot_col) in R32_THIRD.items():
        fixtures[match_no] = (slot_to_team[winner_slot], slot_to_team[mapping[third_slot_col]])
    return fixtures, best_thirds["team"].tolist()



def _choose_knockout_winner(team_a: str, team_b: str, matchup_cache: Dict[Tuple[str, str], dict], rng: np.random.Generator) -> str:
    entry = matchup_cache[(team_a, team_b)]
    goals_a = int(rng.poisson(entry["lambda_a"]))
    goals_b = int(rng.poisson(entry["lambda_b"]))
    if goals_a > goals_b:
        return team_a
    if goals_b > goals_a:
        return team_b
    et_a = int(rng.poisson(max(entry["lambda_a"] / 3.0, 0.01)))
    et_b = int(rng.poisson(max(entry["lambda_b"] / 3.0, 0.01)))
    if et_a > et_b:
        return team_a
    if et_b > et_a:
        return team_b
    p_team_a = float(entry["probs"][2] / (entry["probs"][0] + entry["probs"][2])) if (entry["probs"][0] + entry["probs"][2]) > 0 else 0.5
    return team_a if rng.random() < p_team_a else team_b



def simulate_tournament(models: dict, groups_df: pd.DataFrame, ratings: pd.DataFrame, snapshots: pd.DataFrame, team_metadata: pd.DataFrame, third_place_mapping: pd.DataFrame, squad_hooks: pd.DataFrame, sim_date: str, n_sims: int = 200, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    teams = sorted(groups_df["team"].unique().tolist())
    matchup_cache = build_matchup_cache(models["classifier"], models["goal_model_a"], models["goal_model_b"], teams, ratings, snapshots, team_metadata, squad_hooks, sim_date)

    deep_runs: Dict[str, Dict[str, int]] = {}
    for _ in range(n_sims):
        group_table = simulate_group_stage(groups_df, ratings, sim_date, matchup_cache, rng)
        round_of_32_fixtures, best_thirds = build_round_of_32_fixtures(group_table, third_place_mapping)
        qualifiers = group_table[group_table["group_place"] <= 2]["team"].tolist() + best_thirds
        winners: Dict[int, str] = {}
        for match_no in range(73, 89):
            a, b = round_of_32_fixtures[match_no]
            winners[match_no] = _choose_knockout_winner(a, b, matchup_cache, rng)
        for match_no in [89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 104]:
            src_a, src_b = ROUND_MAPS[match_no]
            winners[match_no] = _choose_knockout_winner(winners[src_a], winners[src_b], matchup_cache, rng)

        for team in qualifiers:
            deep_runs.setdefault(team, {"round_of_32": 0, "round_of_16": 0, "quarterfinal": 0, "semifinal": 0, "final": 0, "champion": 0})
            deep_runs[team]["round_of_32"] += 1
        for team in [winners[m] for m in range(73, 89)]:
            deep_runs.setdefault(team, {"round_of_32": 0, "round_of_16": 0, "quarterfinal": 0, "semifinal": 0, "final": 0, "champion": 0})
            deep_runs[team]["round_of_16"] += 1
        for team in [winners[m] for m in range(89, 97)]:
            deep_runs[team]["quarterfinal"] += 1
        for team in [winners[m] for m in range(97, 101)]:
            deep_runs[team]["semifinal"] += 1
        for team in [winners[101], winners[102]]:
            deep_runs[team]["final"] += 1
        deep_runs[winners[104]]["champion"] += 1

    rows = []
    for team in teams:
        stats = deep_runs.get(team, {"round_of_32": 0, "round_of_16": 0, "quarterfinal": 0, "semifinal": 0, "final": 0, "champion": 0})
        rows.append({
            "team": team,
            "round_of_32_pct": stats["round_of_32"] / n_sims,
            "round_of_16_pct": stats["round_of_16"] / n_sims,
            "quarterfinal_pct": stats["quarterfinal"] / n_sims,
            "semifinal_pct": stats["semifinal"] / n_sims,
            "final_pct": stats["final"] / n_sims,
            "champion_pct": stats["champion"] / n_sims,
        })
    return pd.DataFrame(rows).sort_values("champion_pct", ascending=False).reset_index(drop=True)
