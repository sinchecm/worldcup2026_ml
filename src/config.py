from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parents[1]

    @property
    def raw_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def charts_dir(self) -> Path:
        return self.output_dir / "charts"

    @property
    def matches_path(self) -> Path:
        return self.raw_dir / "matches.csv"

    @property
    def shootouts_path(self) -> Path:
        return self.raw_dir / "shootouts.csv"

    @property
    def history_finals_path(self) -> Path:
        return self.raw_dir / "past_world_cup_finals.csv"

    @property
    def ratings_path(self) -> Path:
        return self.raw_dir / "ratings.csv"

    @property
    def team_metadata_path(self) -> Path:
        return self.raw_dir / "team_metadata.csv"

    @property
    def groups_path(self) -> Path:
        return self.raw_dir / "worldcup_2026_groups.csv"

    @property
    def third_place_mapping_path(self) -> Path:
        return self.raw_dir / "annex_c_third_place_mapping.csv"

    @property
    def squad_hooks_path(self) -> Path:
        return self.raw_dir / "squad_strength_hooks.csv"

    @property
    def squad_sources_template_path(self) -> Path:
        return self.raw_dir / "squad_strength_sources_template.csv"

    train_start: str = "2010-01-01"
    selection_train_end: str = "2022-12-31"
    selection_calib_start: str = "2023-01-01"
    selection_calib_end: str = "2023-12-31"
    evaluation_test_start: str = "2024-01-01"
    deploy_train_end: str = "2023-12-31"
    deploy_calib_start: str = "2024-01-01"
    deploy_calib_end: str = "2024-12-31"
    simulation_date: str = "2026-06-01"
    n_sims: int = 200
    random_state: int = 42
