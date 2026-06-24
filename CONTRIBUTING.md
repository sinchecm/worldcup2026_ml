# Contributing

Thanks for contributing to the World Cup 2026 prediction project.

## Development setup

### Dashboard-only setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### Full pipeline setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements_pipeline.txt
```

## Common commands
```bash
make check
make prepare-data
make train
make app
```

## Contribution guidelines
- Keep all modeling features strictly pre-match to avoid target leakage.
- Prefer reproducible, source-documented changes to raw data assets.
- If you adjust squad-strength hooks, document your rationale in the source template CSV.
- If you change tournament logic, verify it against the FIFA 2026 regulations and bracket references.
- Keep dashboard dependencies lightweight in `requirements.txt`.
- Put heavy offline training dependencies in `requirements_pipeline.txt`.

## Pull requests
Please include:
- what changed
- why it changed
- whether outputs were regenerated
- whether README or deployment instructions changed
- screenshots if UI changes are visible

## Data and sources
Use authoritative sources whenever possible. The project currently relies on FIFA references for historical tournament framing and a public international football results backbone for match-level data. [FIFA 2026 regulations](https://digitalhub.fifa.com/m/636f5c9c6f29771f/original/FWC2026_regulations_EN.pdf) [FIFA 2026 bracket](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/knockout-stage-match-schedule-bracket) [martj42/international_results](https://github.com/martj42/international_results)
