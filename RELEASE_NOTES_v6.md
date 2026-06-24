# Release Notes - v6.0.0 Product Upgrade

This package upgrades the repository from repo-polish status into a more interactive product build.

## New product features
- Scenario lab in Streamlit for squad-strength, attack, defense, and availability what-if simulations
- Team comparison view with head-to-head probabilities and expected goals
- Team profile view with group peers, form, ratings, and tournament-stage probabilities
- Historical contender benchmark for recent World Cups based on pre-tournament participant ratings
- New generated product artifacts for dashboard use:
  - `outputs/team_profiles.csv`
  - `outputs/baseline_matchups.csv`
  - `outputs/contender_benchmark.csv`
  - `outputs/team_snapshots.csv`
  - `outputs/model_bundle.pkl`

## Notes
- The historical benchmark is a contender-ranking benchmark, not a full retro tournament simulation.
- The scenario lab runs from precomputed matchup baselines and applies squad-adjustment heuristics consistently across the tournament simulator.

## Suggested next step
Build a true V7 product release with:
- full historical tournament backtesting
- saved scenario presets
- bracket/path explorer visualizations
- experiment registry and run history
