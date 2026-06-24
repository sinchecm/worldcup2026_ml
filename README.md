# World Cup 2026 Prediction Model + Streamlit Dashboard

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Version-v5%20repo%20polish-blue)

A GitHub-ready repository for predicting the **2026 FIFA World Cup champion** with a supervised match model, Poisson-based score simulation, official 48-team tournament logic, and a Streamlit dashboard for forecasts, diagnostics, squad hooks, and historical World Cup finals.

Visual architecture reference: [pipeline diagram](https://www.genspark.ai/api/files/s/Ph22vxQw?cache_control=3600)

## What this repo includes

- End-to-end data preparation from public football results and source-aligned metadata
- Supervised pre-match feature engineering
- Calibrated 3-class match prediction model
- Poisson goal layer for more realistic knockout simulations
- Official **2026 FIFA World Cup** format support: 12 groups, top two plus eight best third-placed teams, and Annexe C routing
- Streamlit dashboard with:
  - 2026 champion probabilities
  - scenario lab for what-if squad adjustments
  - team comparison view with head-to-head probabilities and expected goals
  - team profiles with group peers, ratings, and stage probabilities
  - model diagnostics and calibration plots
  - squad-strength hooks
  - past World Cup final teams and winners
  - historical contender benchmark for recent World Cups

The repo uses a public historical international-results backbone, FIFA’s official 2026 tournament rules/bracket materials, and official FIFA champion-history pages for the finals-history dashboard. [martj42/international_results](https://github.com/martj42/international_results) [FIFA 2026 regulations](https://digitalhub.fifa.com/m/636f5c9c6f29771f/original/FWC2026_regulations_EN.pdf) [FIFA 2026 bracket](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/knockout-stage-match-schedule-bracket) [FIFA 1930–1978 champions](https://www.fifa.com/en/tournaments/mens/worldcup/articles/world-cup-champions-1930-1978-uruguay-italy-germany-brazil-england-argentina) [FIFA 1982–2022 champions](https://www.fifa.com/en/tournaments/mens/worldcup/articles/world-cup-champions-1982-2022-italy-argentina-germany-brazil-france-spain)

## GitHub-ready repo structure

```text
worldcup2026_ml/
├── .streamlit/
│   └── config.toml
├── data/
│   └── raw/
│       ├── matches.csv
│       ├── shootouts.csv
│       ├── ratings.csv
│       ├── team_metadata.csv
│       ├── worldcup_2026_groups.csv
│       ├── annex_c_third_place_mapping.csv
│       ├── squad_strength_hooks.csv
│       ├── squad_strength_sources_template.csv
│       └── past_world_cup_finals.csv
├── outputs/
│   ├── champion_probabilities.csv
│   ├── metrics.json
│   ├── model_matches.parquet
│   └── charts/
│       ├── calibration_plots.png
│       ├── classifier_feature_importance.csv
│       ├── classifier_feature_importance.png
│       ├── goal_model_feature_importance.csv
│       └── goal_model_feature_importance.png
├── src/
│   ├── config.py
│   ├── ingest.py
│   ├── features.py
│   ├── reporting.py
│   ├── train.py
│   ├── simulate.py
│   ├── prepare_real_data.py
│   └── main.py
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── RELEASE_NOTES_v5.md
├── requirements.txt
├── requirements_pipeline.txt
├── streamlit_app.py
└── README.md
```

## Quick start

### Option A — run the dashboard only

Use this if you only want to open the existing forecast dashboard.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Option B — rebuild data, retrain, and resimulate

Use this if you want to reproduce the pipeline locally.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements_pipeline.txt
python src/prepare_real_data.py
python src/main.py
streamlit run streamlit_app.py
```

## Repository extras

This package now includes:
- `LICENSE` for reuse terms
- `CHANGELOG.md` for version tracking
- `CONTRIBUTING.md` for contributor workflow
- `.github/workflows/ci.yml` for GitHub Actions syntax checks
- issue and pull request templates for cleaner collaboration
- `RELEASE_NOTES_v5.md` for this package handoff

## Makefile shortcuts

```bash
make check         # syntax check
make prepare-data  # rebuild source-ready raw files
make train         # rebuild outputs and charts
make app           # launch Streamlit dashboard
```

## Deployment to Streamlit Community Cloud

1. Push this repo to GitHub
2. Create a new app in Streamlit Community Cloud
3. Select the repository and branch
4. Set the app entrypoint to `streamlit_app.py`
5. Keep `requirements.txt` as the dashboard dependency file

`requirements.txt` is intentionally lightweight for deployment. Use `requirements_pipeline.txt` for full model rebuilding and offline retraining.

Streamlit Community Cloud reflects updates from your GitHub repository automatically after new pushes, and dependency changes trigger a fuller redeploy cycle. This makes the repo compatible with scheduled refresh automation through GitHub Actions.

## Daily automation and redeploy

This repo includes `.github/workflows/daily-refresh-redeploy.yml`.

What it does:
- runs once per day at 08:00 UTC
- also supports manual runs from the GitHub Actions tab
- refreshes source-ready data with `python src/prepare_real_data.py`
- rebuilds prediction outputs with `python src/main.py`
- commits refreshed files in `data/raw/` and `outputs/`
- pushes those changes back to `main`
- stops automatically after `2026-07-19`

Because Streamlit Community Cloud updates from GitHub pushes, those workflow commits function as the redeploy trigger for the app.

### One-time GitHub setup

Make sure the repository allows GitHub Actions to write back to the repo. The workflow requests `contents: write`, but your repository or organization settings must also permit that behavior.

### Manual run

Open the repository on GitHub, go to **Actions**, choose **Daily Refresh and Streamlit Redeploy**, and use **Run workflow**.

## Core files

### `src/prepare_real_data.py`
Builds source-ready files used by the project:
- downloads and stores historical international match results
- generates Elo-style ratings history
- builds 2026 groups and Annexe C mapping
- prepares squad hook templates
- creates `past_world_cup_finals.csv` for the history dashboard

### `src/features.py`
Builds the supervised learning table with strictly pre-match features such as rating gap, form, goal-difference trends, rest days, confederation context, and host flags.

### `src/train.py`
Trains candidate classifiers, calibrates probabilities, evaluates on time-based splits, and trains Poisson goal models. If XGBoost is not installed, the pipeline still runs using logistic-regression-based candidates.

### `src/simulate.py`
Simulates the full 2026 tournament using the trained classifier, Poisson goal expectations, group ranking logic, and official third-place routing.

### `streamlit_app.py`
Deployable dashboard with four main sections:
- 2026 prediction
- past finals & winners
- model diagnostics
- squad hooks

## Data assets

### Historical matches
`data/raw/matches.csv` comes from the public international football results repository and provides the match backbone used for both feature engineering and historical validation. [martj42/international_results](https://github.com/martj42/international_results)

### 2026 tournament structure
The repository follows FIFA’s 48-team tournament format and knockout routing references for 2026. [FIFA 2026 bracket](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/knockout-stage-match-schedule-bracket) [FIFA 2026 regulations](https://digitalhub.fifa.com/m/636f5c9c6f29771f/original/FWC2026_regulations_EN.pdf)

### Historical finals dashboard
`data/raw/past_world_cup_finals.csv` aligns FIFA’s official champion/runner-up history with match-level final score rows from the public results dataset. This is what powers the dashboard view for past finalists and winners. [FIFA 1930–1978 champions](https://www.fifa.com/en/tournaments/mens/worldcup/articles/world-cup-champions-1930-1978-uruguay-italy-germany-brazil-england-argentina) [FIFA 1982–2022 champions](https://www.fifa.com/en/tournaments/mens/worldcup/articles/world-cup-champions-1982-2022-italy-argentina-germany-brazil-france-spain) [martj42/international_results](https://github.com/martj42/international_results)

## Current artifacts

After a successful pipeline run, the main deliverables are:
- `outputs/model_matches.parquet`
- `outputs/metrics.json`
- `outputs/champion_probabilities.csv`
- `outputs/team_profiles.csv`
- `outputs/baseline_matchups.csv`
- `outputs/contender_benchmark.csv`
- `outputs/team_snapshots.csv`
- `outputs/model_bundle.pkl`
- `outputs/charts/*.png`

## Typical workflow

### Refresh data and rebuild the model

```bash
python src/prepare_real_data.py
python src/main.py
```

### Launch the dashboard

```bash
streamlit run streamlit_app.py
```

### Adjust squads before simulation

1. Edit `data/raw/squad_strength_hooks.csv`
2. Document your evidence in `data/raw/squad_strength_sources_template.csv`
3. Re-run `python src/main.py`
4. Refresh the Streamlit app

## Recommended next versions

Good next upgrades for this repository are:
- backtesting page for 2010, 2014, 2018, and 2022
- scenario simulator for injuries and analyst overrides
- team comparison cards
- tournament path explorer
- experiment tracking for versioned model runs

## Notes

- This repo is designed as a practical forecasting and simulation framework, not a claim of certainty.
- Tournament probabilities are sensitive to squad availability, draw routing, and late-form changes.
- For production-quality forecasting, consider enriching squad strength, lineups, injuries, and event-level features from richer open-data sources such as StatsBomb open data. [StatsBomb Open Data](https://github.com/statsbomb/open-data)
