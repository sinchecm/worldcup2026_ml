from __future__ import annotations

import pandas as pd

from config import Config
from features import build_model_table, build_team_snapshots
from ingest import load_groups, load_matches, load_ratings, load_squad_hooks, load_team_metadata
from product_artifacts import (
    build_baseline_matchups,
    build_contender_benchmark,
    build_team_profiles,
    save_model_bundle,
)
from reporting import save_calibration_plot, save_classifier_feature_importance, save_goal_model_feature_importance
from simulate import simulate_tournament
from train import save_metrics, train_and_evaluate



def main() -> None:
    cfg = Config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.charts_dir.mkdir(parents=True, exist_ok=True)

    matches = load_matches(cfg.matches_path)
    ratings = load_ratings(cfg.ratings_path)
    team_metadata = load_team_metadata(cfg.team_metadata_path, matches)
    groups_df = load_groups(cfg.groups_path)
    squad_hooks = load_squad_hooks(cfg.squad_hooks_path, groups_df)
    third_place_mapping = pd.read_csv(cfg.third_place_mapping_path)

    model_table = build_model_table(matches, ratings, team_metadata)
    model_table = model_table[model_table["match_date"] >= pd.Timestamp(cfg.train_start)].copy()
    model_table.to_parquet(cfg.output_dir / "model_matches.parquet", index=False)

    models, metrics, artifacts = train_and_evaluate(
        model_df=model_table,
        selection_train_end=cfg.selection_train_end,
        selection_calib_start=cfg.selection_calib_start,
        selection_calib_end=cfg.selection_calib_end,
        evaluation_test_start=cfg.evaluation_test_start,
        deploy_train_end=cfg.deploy_train_end,
        deploy_calib_start=cfg.deploy_calib_start,
        deploy_calib_end=cfg.deploy_calib_end,
        random_state=cfg.random_state,
    )
    save_metrics(metrics, cfg.output_dir / "metrics.json")

    save_classifier_feature_importance(artifacts["selected_eval_classifier"], artifacts["X_test"], artifacts["y_test"], cfg.charts_dir)
    save_goal_model_feature_importance(artifacts["goal_eval_model_a"], artifacts["goal_eval_model_b"], artifacts["X_test"], artifacts["y_test_a"], artifacts["y_test_b"], cfg.charts_dir)
    save_calibration_plot(artifacts["selected_eval_classifier"], artifacts["X_test"], artifacts["y_test"], cfg.charts_dir)

    snapshots = build_team_snapshots(matches, ratings, cfg.simulation_date)
    champion_probs = simulate_tournament(
        models=models,
        groups_df=groups_df,
        ratings=ratings,
        snapshots=snapshots,
        team_metadata=team_metadata,
        third_place_mapping=third_place_mapping,
        squad_hooks=squad_hooks,
        sim_date=cfg.simulation_date,
        n_sims=cfg.n_sims,
        random_state=cfg.random_state,
    )
    champion_probs.to_csv(cfg.output_dir / "champion_probabilities.csv", index=False)

    snapshots.to_csv(cfg.output_dir / "team_snapshots.csv", index=False)
    save_model_bundle(models, cfg.output_dir / "model_bundle.pkl")
    build_baseline_matchups(
        models=models,
        groups_df=groups_df,
        ratings=ratings,
        snapshots=snapshots,
        team_metadata=team_metadata,
        squad_hooks=squad_hooks,
        sim_date=cfg.simulation_date,
    ).to_csv(cfg.output_dir / "baseline_matchups.csv", index=False)
    build_team_profiles(
        groups_df=groups_df,
        ratings=ratings,
        snapshots=snapshots,
        team_metadata=team_metadata,
        champion_probs=champion_probs,
        sim_date=cfg.simulation_date,
    ).to_csv(cfg.output_dir / "team_profiles.csv", index=False)
    history_finals = pd.read_csv(cfg.history_finals_path)
    build_contender_benchmark(matches, ratings, history_finals).to_csv(cfg.output_dir / "contender_benchmark.csv", index=False)

    print("Saved:")
    print(f"  - {cfg.output_dir / 'model_matches.parquet'}")
    print(f"  - {cfg.output_dir / 'metrics.json'}")
    print(f"  - {cfg.output_dir / 'champion_probabilities.csv'}")
    print(f"  - {cfg.output_dir / 'team_snapshots.csv'}")
    print(f"  - {cfg.output_dir / 'model_bundle.pkl'}")
    print(f"  - {cfg.output_dir / 'baseline_matchups.csv'}")
    print(f"  - {cfg.output_dir / 'team_profiles.csv'}")
    print(f"  - {cfg.output_dir / 'contender_benchmark.csv'}")
    print(f"  - {cfg.charts_dir / 'classifier_feature_importance.png'}")
    print(f"  - {cfg.charts_dir / 'goal_model_feature_importance.png'}")
    print(f"  - {cfg.charts_dir / 'calibration_plots.png'}")
    print("Top 10 title probabilities:")
    print(champion_probs.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
