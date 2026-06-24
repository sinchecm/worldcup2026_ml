# Release Notes - v6.1.0 Automation Add-on

This add-on prepares the repository for unattended daily refresh and redeploy.

## Added
- GitHub Actions workflow: `.github/workflows/daily-refresh-redeploy.yml`
- daily scheduled refresh at 08:00 UTC
- manual trigger support through `workflow_dispatch`
- automatic stop condition after 2026-07-19
- automatic commit-and-push of refreshed `data/raw/` and `outputs/` artifacts

## Deployment effect
When this workflow pushes updated prediction files back to the repository, Streamlit Community Cloud should reflect those GitHub updates automatically.

## Important setup note
The repository must have GitHub Actions enabled, and the workflow must be allowed to write contents to the repository.
