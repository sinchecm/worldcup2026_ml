from __future__ import annotations

from pathlib import Path
import pandas as pd


REQUIRED_MATCH_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
}

REQUIRED_RATING_COLUMNS = {"team", "rating_date", "rating"}
SQUAD_HOOK_COLUMNS = {
    "team",
    "snapshot_date",
    "squad_strength_delta",
    "attack_delta",
    "defense_delta",
    "availability_delta",
}



def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")



def load_matches(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, REQUIRED_MATCH_COLUMNS, "matches.csv")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["neutral"] = df["neutral"].astype(str).str.upper().map({"TRUE": True, "FALSE": False}).fillna(False)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    if "match_id" not in df.columns:
        df["match_id"] = (
            df["date"].dt.strftime("%Y%m%d")
            + "_"
            + df["home_team"].str.replace(" ", "_", regex=False)
            + "_vs_"
            + df["away_team"].str.replace(" ", "_", regex=False)
            + "_"
            + df.groupby(["date", "home_team", "away_team"]).cumcount().add(1).astype(str)
        )
    return df.sort_values(["date", "match_id"]).reset_index(drop=True)



def load_ratings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, REQUIRED_RATING_COLUMNS, "ratings.csv")
    df = df.copy()
    df["rating_date"] = pd.to_datetime(df["rating_date"]).dt.normalize()
    df["rating"] = df["rating"].astype(float)
    if "rank" in df.columns:
        df["rank"] = df["rank"].astype(float)
    return df.sort_values(["team", "rating_date"]).reset_index(drop=True)



def load_team_metadata(path: Path, matches: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
    else:
        teams = sorted(set(matches["home_team"]).union(matches["away_team"]))
        df = pd.DataFrame({"team": teams})
    df = df.copy()
    if "confederation" not in df.columns:
        df["confederation"] = "UNKNOWN"
    if "host_country_flag" not in df.columns:
        df["host_country_flag"] = df["team"].isin(["USA", "United States", "Mexico", "Canada"]).astype(int)
    df["host_country_flag"] = df["host_country_flag"].astype(int)
    return df.drop_duplicates(subset=["team"]).reset_index(drop=True)



def load_groups(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing group file: {path}. Expected columns: group, team")
    df = pd.read_csv(path)
    _require_columns(df, {"group", "team"}, "worldcup_2026_groups.csv")
    return df.sort_values(["group", "team"]).reset_index(drop=True)



def load_squad_hooks(path: Path, groups_df: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        _require_columns(df, SQUAD_HOOK_COLUMNS, "squad_strength_hooks.csv")
    else:
        df = pd.DataFrame({
            "team": groups_df["team"].tolist(),
            "snapshot_date": ["2026-06-01"] * len(groups_df),
            "squad_strength_delta": [0.0] * len(groups_df),
            "attack_delta": [0.0] * len(groups_df),
            "defense_delta": [0.0] * len(groups_df),
            "availability_delta": [0.0] * len(groups_df),
        })
    df = df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.normalize()
    for col in ["squad_strength_delta", "attack_delta", "defense_delta", "availability_delta"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
    if "source_url" not in df.columns:
        df["source_url"] = ""
    if "notes" not in df.columns:
        df["notes"] = ""
    return df.sort_values(["team", "snapshot_date"]).reset_index(drop=True)
