from __future__ import annotations

from itertools import combinations
from typing import Dict, Tuple

import numpy as np
import pandas as pd

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


def matchup_table_to_cache(matchups: pd.DataFrame) -> Dict[Tuple[str, str], dict]:
    cache: Dict[Tuple[str, str], dict] = {}
    for row in matchups.itertuples(index=False):
        cache[(row.team_a, row.team_b)] = {
            "probs": np.array([row.prob_team_a_loss, row.prob_draw, row.prob_team_a_win], dtype=float),
            "lambda_a": float(row.lambda_a),
            "lambda_b": float(row.lambda_b),
        }
    return cache


def apply_scenario_adjustments(matchups: pd.DataFrame, adjustments: dict[str, dict]) -> pd.DataFrame:
    adjusted = matchups.copy()
    for idx, row in adjusted.iterrows():
        a = adjustments.get(row["team_a"], {})
        b = adjustments.get(row["team_b"], {})
        squad_gap = float(a.get("squad_strength_delta", 0.0) - b.get("squad_strength_delta", 0.0))
        availability_gap = float(a.get("availability_delta", 0.0) - b.get("availability_delta", 0.0))
        attack_vs_defense_a = float(a.get("attack_delta", 0.0) - b.get("defense_delta", 0.0))
        attack_vs_defense_b = float(b.get("attack_delta", 0.0) - a.get("defense_delta", 0.0))

        probs = np.array([row["prob_team_a_loss"], row["prob_draw"], row["prob_team_a_win"]], dtype=float)
        shift = 0.18 * squad_gap + 0.12 * attack_vs_defense_a + 0.08 * availability_gap
        probs[2] *= np.exp(shift)
        probs[0] *= np.exp(-shift)
        probs = probs / probs.sum()

        lambda_a = float(np.clip(row["lambda_a"] * np.exp(0.15 * attack_vs_defense_a + 0.08 * availability_gap + 0.05 * squad_gap), 0.05, 5.0))
        lambda_b = float(np.clip(row["lambda_b"] * np.exp(0.15 * attack_vs_defense_b - 0.08 * availability_gap - 0.05 * squad_gap), 0.05, 5.0))

        adjusted.loc[idx, "prob_team_a_loss"] = probs[0]
        adjusted.loc[idx, "prob_draw"] = probs[1]
        adjusted.loc[idx, "prob_team_a_win"] = probs[2]
        adjusted.loc[idx, "lambda_a"] = lambda_a
        adjusted.loc[idx, "lambda_b"] = lambda_b
    return adjusted


def _build_rating_map(group_table: pd.DataFrame) -> dict:
    if "rating" not in group_table.columns:
        return {}
    return dict(zip(group_table["team"], group_table["rating"]))


def init_group_table(groups_df: pd.DataFrame, team_profiles: pd.DataFrame) -> pd.DataFrame:
    table = groups_df.copy()
    rating_map = dict(zip(team_profiles["team"], team_profiles.get("latest_rating", pd.Series(dtype=float))))
    table["rating"] = table["team"].map(rating_map).fillna(float(team_profiles.get("latest_rating", pd.Series([1500.0])).mean()))
    for col in ["pts", "gf", "ga", "gd", "wins", "draws", "losses"]:
        table[col] = 0
    return table


def simulate_group_stage(groups_df: pd.DataFrame, team_profiles: pd.DataFrame, matchup_cache: Dict[Tuple[str, str], dict], rng: np.random.Generator) -> pd.DataFrame:
    table = init_group_table(groups_df, team_profiles)
    for _, sub in groups_df.groupby("group"):
        teams = sub["team"].tolist()
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


def choose_knockout_winner(team_a: str, team_b: str, matchup_cache: Dict[Tuple[str, str], dict], rng: np.random.Generator) -> str:
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
    denom = entry["probs"][0] + entry["probs"][2]
    p_team_a = float(entry["probs"][2] / denom) if denom > 0 else 0.5
    return team_a if rng.random() < p_team_a else team_b


def simulate_tournament_from_matchups(
    groups_df: pd.DataFrame,
    team_profiles: pd.DataFrame,
    third_place_mapping: pd.DataFrame,
    matchups: pd.DataFrame,
    n_sims: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    teams = sorted(groups_df["team"].unique().tolist())
    matchup_cache = matchup_table_to_cache(matchups)

    deep_runs: Dict[str, Dict[str, int]] = {}
    for _ in range(n_sims):
        group_table = simulate_group_stage(groups_df, team_profiles, matchup_cache, rng)
        round_of_32_fixtures, best_thirds = build_round_of_32_fixtures(group_table, third_place_mapping)
        qualifiers = group_table[group_table["group_place"] <= 2]["team"].tolist() + best_thirds
        winners: Dict[int, str] = {}
        for match_no in range(73, 89):
            a, b = round_of_32_fixtures[match_no]
            winners[match_no] = choose_knockout_winner(a, b, matchup_cache, rng)
        for match_no in [89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 104]:
            src_a, src_b = ROUND_MAPS[match_no]
            winners[match_no] = choose_knockout_winner(winners[src_a], winners[src_b], matchup_cache, rng)

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
        rows.append(
            {
                "team": team,
                "round_of_32_pct": stats["round_of_32"] / n_sims,
                "round_of_16_pct": stats["round_of_16"] / n_sims,
                "quarterfinal_pct": stats["quarterfinal"] / n_sims,
                "semifinal_pct": stats["semifinal"] / n_sims,
                "final_pct": stats["final"] / n_sims,
                "champion_pct": stats["champion"] / n_sims,
            }
        )
    return pd.DataFrame(rows).sort_values("champion_pct", ascending=False).reset_index(drop=True)
