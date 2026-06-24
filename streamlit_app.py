from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from simulate_runtime import apply_scenario_adjustments, simulate_tournament_from_matchups  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
CHARTS = OUT / "charts"

HISTORY_SOURCE_1 = "https://www.fifa.com/en/tournaments/mens/worldcup/articles/world-cup-champions-1930-1978-uruguay-italy-germany-brazil-england-argentina"
HISTORY_SOURCE_2 = "https://www.fifa.com/en/tournaments/mens/worldcup/articles/world-cup-champions-1982-2022-italy-argentina-germany-brazil-france-spain"
RESULTS_SOURCE = "https://github.com/martj42/international_results"
FORMAT_SOURCE = "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/knockout-stage-match-schedule-bracket"
REGS_SOURCE = "https://digitalhub.fifa.com/m/636f5c9c6f29771f/original/FWC2026_regulations_EN.pdf"
PIPELINE_DIAGRAM = "https://www.genspark.ai/api/files/s/Ph22vxQw?cache_control=3600"

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="wide")


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text())


def pct(series: pd.Series) -> pd.Series:
    return (series * 100).round(1)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        st.error(f"Missing required file: {label} ({path.name})")
        st.stop()


def scenario_adjustment_controls(team: str, defaults: dict | None = None) -> dict:
    defaults = defaults or {}
    with st.expander(team, expanded=True):
        squad = st.slider(f"{team} squad strength", -1.0, 1.0, float(defaults.get("squad_strength_delta", 0.0)), 0.05, key=f"squad_{team}")
        attack = st.slider(f"{team} attack delta", -1.0, 1.0, float(defaults.get("attack_delta", 0.0)), 0.05, key=f"attack_{team}")
        defense = st.slider(f"{team} defense delta", -1.0, 1.0, float(defaults.get("defense_delta", 0.0)), 0.05, key=f"defense_{team}")
        availability = st.slider(f"{team} availability delta", -1.0, 1.0, float(defaults.get("availability_delta", 0.0)), 0.05, key=f"avail_{team}")
    return {
        "squad_strength_delta": squad,
        "attack_delta": attack,
        "defense_delta": defense,
        "availability_delta": availability,
    }


def load_required_assets() -> dict:
    required = {
        OUT / "champion_probabilities.csv": "simulation output",
        OUT / "metrics.json": "model metrics",
        RAW / "past_world_cup_finals.csv": "historical finals dataset",
        RAW / "squad_strength_hooks.csv": "squad hooks dataset",
        OUT / "team_profiles.csv": "team profiles artifact",
        OUT / "baseline_matchups.csv": "baseline matchup artifact",
        OUT / "contender_benchmark.csv": "historical contender benchmark",
        RAW / "worldcup_2026_groups.csv": "group file",
        RAW / "annex_c_third_place_mapping.csv": "Annexe C mapping",
    }
    for path, label in required.items():
        require_file(path, label)
    return {
        "champion_probs": load_csv(OUT / "champion_probabilities.csv"),
        "history": load_csv(RAW / "past_world_cup_finals.csv"),
        "squad_hooks": load_csv(RAW / "squad_strength_hooks.csv"),
        "metrics": load_metrics(OUT / "metrics.json"),
        "team_profiles": load_csv(OUT / "team_profiles.csv"),
        "baseline_matchups": load_csv(OUT / "baseline_matchups.csv"),
        "contender_benchmark": load_csv(OUT / "contender_benchmark.csv"),
        "groups_df": load_csv(RAW / "worldcup_2026_groups.csv"),
        "third_place_mapping": load_csv(RAW / "annex_c_third_place_mapping.csv"),
    }


assets = load_required_assets()
champion_probs = assets["champion_probs"].sort_values("champion_pct", ascending=False).reset_index(drop=True)
history = assets["history"].sort_values("year").reset_index(drop=True)
squad_hooks = assets["squad_hooks"]
metrics = assets["metrics"]
team_profiles = assets["team_profiles"].sort_values("champion_pct", ascending=False).reset_index(drop=True)
baseline_matchups = assets["baseline_matchups"]
contender_benchmark = assets["contender_benchmark"]
groups_df = assets["groups_df"]
third_place_mapping = assets["third_place_mapping"]

team_list = sorted(team_profiles["team"].unique().tolist())
group_lookup = dict(zip(team_profiles["team"], team_profiles["group_name"]))
profile_lookup = team_profiles.set_index("team").to_dict("index")
default_hook_lookup = squad_hooks.sort_values(["team", "snapshot_date"]).groupby("team", as_index=False).tail(1).set_index("team").to_dict("index")

titles = history.groupby("champion").size().sort_values(ascending=False).rename("titles")
finalist_appearances = (
    pd.concat([
        history[["champion"]].rename(columns={"champion": "team"}),
        history[["runner_up"]].rename(columns={"runner_up": "team"}),
    ])
    .groupby("team")
    .size()
    .sort_values(ascending=False)
    .rename("final_appearances")
)

favorite = champion_probs.iloc[0]
most_titles_team = titles.index[0]
most_titles = int(titles.iloc[0])
most_finals_team = finalist_appearances.index[0]
most_finals = int(finalist_appearances.iloc[0])
selected_classifier = metrics.get("selected_classifier")
selected_classifier_metrics = metrics.get("classifier_candidate_metrics", {}).get(selected_classifier, {})

st.title("⚽ World Cup 2026 Prediction Dashboard")
st.caption("Forecasting, scenario testing, team comparison, model diagnostics, and World Cup history in one Streamlit product.")

with st.sidebar:
    st.header("Navigate")
    st.markdown(
        "- 2026 forecast\n"
        "- Scenario lab\n"
        "- Team compare\n"
        "- Team profiles\n"
        "- History & benchmark\n"
        "- Model diagnostics\n"
        "- Squad hooks"
    )
    st.header("Reference links")
    st.markdown(
        f"- [Pipeline diagram]({PIPELINE_DIAGRAM})\n"
        f"- [FIFA 2026 bracket]({FORMAT_SOURCE})\n"
        f"- [FIFA 2026 regulations]({REGS_SOURCE})\n"
        f"- [Historical champions 1930–1978]({HISTORY_SOURCE_1})\n"
        f"- [Historical champions 1982–2022]({HISTORY_SOURCE_2})\n"
        f"- [International results dataset]({RESULTS_SOURCE})"
    )

m1, m2, m3, m4 = st.columns(4)
m1.metric("Projected favorite", favorite["team"])
m2.metric("Top title probability", f"{favorite['champion_pct']:.1%}")
m3.metric("Most successful nation", f"{most_titles_team} ({most_titles})")
m4.metric("Most finals appearances", f"{most_finals_team} ({most_finals})")

tab_forecast, tab_scenario, tab_compare, tab_profiles, tab_history, tab_diag, tab_hooks = st.tabs([
    "2026 prediction",
    "Scenario lab",
    "Team compare",
    "Team profiles",
    "History & benchmark",
    "Model diagnostics",
    "Squad hooks",
])

with tab_forecast:
    left, right = st.columns([1.2, 0.8])
    with left:
        st.subheader("Champion probabilities")
        bar_df = champion_probs.head(15).copy()
        bar_df["Champion %"] = bar_df["champion_pct"] * 100
        st.bar_chart(bar_df.set_index("team")["Champion %"], height=420)

        table_df = champion_probs.copy()
        for col in ["round_of_32_pct", "round_of_16_pct", "quarterfinal_pct", "semifinal_pct", "final_pct", "champion_pct"]:
            table_df[col] = pct(table_df[col])
        table_df = table_df.rename(
            columns={
                "team": "Team",
                "round_of_32_pct": "Round of 32 %",
                "round_of_16_pct": "Round of 16 %",
                "quarterfinal_pct": "Quarterfinal %",
                "semifinal_pct": "Semifinal %",
                "final_pct": "Final %",
                "champion_pct": "Champion %",
            }
        )
        st.dataframe(table_df, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Model summary")
        st.json(
            {
                "selected_classifier": selected_classifier,
                "classifier_metrics": selected_classifier_metrics,
                "goal_model_metrics": metrics.get("goal_model_metrics", {}),
                "evaluation_test_rows": metrics.get("evaluation_test_rows"),
            }
        )
        st.subheader("Forecast notes")
        st.markdown(
            """
            - 48-team FIFA 2026 structure with Annexe C third-place routing
            - calibrated match classifier plus Poisson goal layer
            - source-ready historical match backbone and finals history
            - squad hooks can be edited and rerun offline before redeployment
            """
        )
        st.download_button(
            "Download title probabilities CSV",
            data=(OUT / "champion_probabilities.csv").read_bytes(),
            file_name="champion_probabilities.csv",
            mime="text/csv",
        )

with tab_scenario:
    st.subheader("Scenario lab")
    st.caption("Stress-test the tournament by changing squad strength, attack, defense, or availability for selected teams.")

    left, right = st.columns([0.95, 1.05])
    with left:
        selected_teams = st.multiselect("Teams to adjust", team_list, default=team_list[:2])
        sim_count = st.slider("Scenario simulations", min_value=100, max_value=2000, value=300, step=100)
        random_seed = st.number_input("Random seed", min_value=1, max_value=100000, value=42, step=1)
        adjustments: dict[str, dict] = {}
        for team in selected_teams:
            adjustments[team] = scenario_adjustment_controls(team, default_hook_lookup.get(team, {}))
        run_scenario = st.button("Run scenario simulation")
    with right:
        st.markdown("**Baseline favorite order**")
        st.dataframe(
            champion_probs.head(10).assign(champion_pct=lambda d: pct(d["champion_pct"])).rename(columns={"team": "Team", "champion_pct": "Champion %"})[["Team", "Champion %"]],
            use_container_width=True,
            hide_index=True,
        )
        st.info("Tip: increase attack and squad strength for one contender, or lower availability for an injured star-dependent team, then rerun.")

    if run_scenario:
        adjusted_matchups = apply_scenario_adjustments(baseline_matchups, adjustments)
        scenario_probs = simulate_tournament_from_matchups(
            groups_df=groups_df,
            team_profiles=team_profiles,
            third_place_mapping=third_place_mapping,
            matchups=adjusted_matchups,
            n_sims=int(sim_count),
            random_state=int(random_seed),
        )
        scenario_view = scenario_probs.merge(champion_probs[["team", "champion_pct"]], on="team", how="left", suffixes=("_scenario", "_baseline"))
        scenario_view["delta_pp"] = (scenario_view["champion_pct_scenario"] - scenario_view["champion_pct_baseline"]) * 100
        scenario_view = scenario_view.sort_values("champion_pct_scenario", ascending=False).reset_index(drop=True)

        s1, s2 = st.columns([1.05, 0.95])
        with s1:
            plot_df = scenario_view.head(12).copy()
            plot_df["Scenario Champion %"] = plot_df["champion_pct_scenario"] * 100
            st.bar_chart(plot_df.set_index("team")["Scenario Champion %"], height=360)
        with s2:
            movers = scenario_view.reindex(scenario_view["delta_pp"].abs().sort_values(ascending=False).index).head(8)
            st.markdown("**Biggest movers vs baseline**")
            st.dataframe(
                movers.assign(
                    champion_pct_scenario=lambda d: pct(d["champion_pct_scenario"]),
                    champion_pct_baseline=lambda d: pct(d["champion_pct_baseline"]),
                    delta_pp=lambda d: d["delta_pp"].round(2),
                ).rename(
                    columns={
                        "team": "Team",
                        "champion_pct_scenario": "Scenario %",
                        "champion_pct_baseline": "Baseline %",
                        "delta_pp": "Delta pp",
                    }
                )[["Team", "Scenario %", "Baseline %", "Delta pp"]],
                use_container_width=True,
                hide_index=True,
            )
        st.subheader("Scenario tournament table")
        st.dataframe(
            scenario_view.assign(
                champion_pct_scenario=lambda d: pct(d["champion_pct_scenario"]),
                champion_pct_baseline=lambda d: pct(d["champion_pct_baseline"]),
                delta_pp=lambda d: d["delta_pp"].round(2),
            ).rename(
                columns={
                    "team": "Team",
                    "round_of_32_pct": "Round of 32 %",
                    "round_of_16_pct": "Round of 16 %",
                    "quarterfinal_pct": "Quarterfinal %",
                    "semifinal_pct": "Semifinal %",
                    "final_pct": "Final %",
                    "champion_pct_scenario": "Scenario Champion %",
                    "champion_pct_baseline": "Baseline Champion %",
                    "delta_pp": "Delta pp",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

with tab_compare:
    st.subheader("Team compare")
    c1, c2 = st.columns(2)
    with c1:
        team_a = st.selectbox("Team A", team_list, index=team_list.index("Argentina") if "Argentina" in team_list else 0)
    with c2:
        fallback_index = team_list.index("Spain") if "Spain" in team_list else min(1, len(team_list) - 1)
        team_b = st.selectbox("Team B", team_list, index=fallback_index)

    if team_a == team_b:
        st.warning("Choose two different teams to compare.")
    else:
        row = baseline_matchups[(baseline_matchups["team_a"] == team_a) & (baseline_matchups["team_b"] == team_b)]
        if row.empty:
            st.error("No baseline matchup found for the selected teams.")
        else:
            row = row.iloc[0]
            pa = profile_lookup[team_a]
            pb = profile_lookup[team_b]

            pcol1, pcol2, pcol3, pcol4 = st.columns(4)
            pcol1.metric(f"{team_a} win", f"{row['prob_team_a_win']:.1%}")
            pcol2.metric("Draw", f"{row['prob_draw']:.1%}")
            pcol3.metric(f"{team_b} win", f"{row['prob_team_a_loss']:.1%}")
            pcol4.metric("Expected goals", f"{row['lambda_a']:.2f} - {row['lambda_b']:.2f}")

            left, right = st.columns(2)
            with left:
                st.markdown(f"### {team_a}")
                st.dataframe(
                    pd.DataFrame([
                        ["Group", pa.get("group_name")],
                        ["Confederation", pa.get("confederation")],
                        ["Latest rating", round(float(pa.get("latest_rating", 0.0)), 2)],
                        ["5-match form PPG", round(float(pa.get("form_ppg_5", 0.0)), 2)],
                        ["5-match goal diff", round(float(pa.get("gd_5", 0.0)), 2)],
                        ["Champion probability", f"{float(pa.get('champion_pct', 0.0)):.1%}"],
                    ], columns=["Metric", "Value"]),
                    use_container_width=True,
                    hide_index=True,
                )
            with right:
                st.markdown(f"### {team_b}")
                st.dataframe(
                    pd.DataFrame([
                        ["Group", pb.get("group_name")],
                        ["Confederation", pb.get("confederation")],
                        ["Latest rating", round(float(pb.get("latest_rating", 0.0)), 2)],
                        ["5-match form PPG", round(float(pb.get("form_ppg_5", 0.0)), 2)],
                        ["5-match goal diff", round(float(pb.get("gd_5", 0.0)), 2)],
                        ["Champion probability", f"{float(pb.get('champion_pct', 0.0)):.1%}"],
                    ], columns=["Metric", "Value"]),
                    use_container_width=True,
                    hide_index=True,
                )

            comparison_df = pd.DataFrame(
                {
                    "Metric": ["Latest rating", "Form PPG (5)", "Goal diff (5)", "Goals for (10)", "Goals against (10)", "Rest days"],
                    team_a: [pa.get("latest_rating"), pa.get("form_ppg_5"), pa.get("gd_5"), pa.get("gf_10"), pa.get("ga_10"), pa.get("rest_days")],
                    team_b: [pb.get("latest_rating"), pb.get("form_ppg_5"), pb.get("gd_5"), pb.get("gf_10"), pb.get("ga_10"), pb.get("rest_days")],
                }
            )
            st.subheader("Side-by-side feature snapshot")
            st.dataframe(comparison_df.round(2), use_container_width=True, hide_index=True)

with tab_profiles:
    st.subheader("Team profiles")
    profile_team = st.selectbox("Profile team", team_list, index=team_list.index(favorite["team"]))
    profile_row = team_profiles[team_profiles["team"] == profile_team].iloc[0]
    group_peers = team_profiles[team_profiles["group_name"] == profile_row["group_name"]].sort_values("champion_pct", ascending=False)

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Group", profile_row["group_name"])
    a2.metric("Latest rating", f"{profile_row['latest_rating']:.1f}")
    a3.metric("Champion probability", f"{profile_row['champion_pct']:.1%}")
    a4.metric("Most likely deepest stage", max([
        ("Round of 32", profile_row["round_of_32_pct"]),
        ("Round of 16", profile_row["round_of_16_pct"]),
        ("Quarterfinal", profile_row["quarterfinal_pct"]),
        ("Semifinal", profile_row["semifinal_pct"]),
        ("Final", profile_row["final_pct"]),
        ("Champion", profile_row["champion_pct"]),
    ], key=lambda x: x[1])[0])

    left, right = st.columns([0.95, 1.05])
    with left:
        stage_df = pd.DataFrame(
            {
                "Stage": ["Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final", "Champion"],
                "Probability %": [
                    profile_row["round_of_32_pct"] * 100,
                    profile_row["round_of_16_pct"] * 100,
                    profile_row["quarterfinal_pct"] * 100,
                    profile_row["semifinal_pct"] * 100,
                    profile_row["final_pct"] * 100,
                    profile_row["champion_pct"] * 100,
                ],
            }
        )
        st.bar_chart(stage_df.set_index("Stage")["Probability %"], height=320)
    with right:
        st.markdown("**Group peers**")
        st.dataframe(
            group_peers.assign(
                champion_pct=lambda d: pct(d["champion_pct"]),
                latest_rating=lambda d: d["latest_rating"].round(1),
            ).rename(columns={"team": "Team", "champion_pct": "Champion %", "latest_rating": "Rating"})[["Team", "Rating", "Champion %"]],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Full team profile")
    profile_view = pd.DataFrame([
        ["Confederation", profile_row["confederation"]],
        ["Host flag", int(profile_row["is_host"])],
        ["Latest rating date", profile_row["latest_rating_date"]],
        ["Form PPG last 5", round(float(profile_row["form_ppg_5"]), 2)],
        ["Form PPG last 10", round(float(profile_row["form_ppg_10"]), 2)],
        ["Goal diff last 5", round(float(profile_row["gd_5"]), 2)],
        ["Goals for last 10", round(float(profile_row["gf_10"]), 2)],
        ["Goals against last 10", round(float(profile_row["ga_10"]), 2)],
        ["Opposition average rating last 10", round(float(profile_row["opp_avg_rating_10"]), 2)],
        ["Rest days proxy", round(float(profile_row["rest_days"]), 1)],
        ["Matches in rolling history", int(profile_row["matches_seen"])],
    ], columns=["Metric", "Value"])
    st.dataframe(profile_view, use_container_width=True, hide_index=True)

with tab_history:
    st.subheader("Past World Cup final teams and winners")
    h1, h2 = st.columns(2)
    with h1:
        st.markdown("**Championship titles by nation**")
        st.bar_chart(titles.head(12), height=320)
    with h2:
        st.markdown("**Final appearances by nation**")
        st.bar_chart(finalist_appearances.head(12), height=320)

    st.dataframe(
        history[["year", "champion", "runner_up", "score_display", "match_type", "final_date"]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Pre-tournament contender benchmark")
    st.caption("This benchmark ranks actual tournament participants by their pre-tournament rating snapshot. It is a historical contender check, not a full retro simulation.")
    benchmark_year = st.selectbox("Benchmark year", sorted(contender_benchmark["year"].unique().tolist()), index=len(sorted(contender_benchmark["year"].unique().tolist())) - 1)
    year_df = contender_benchmark[contender_benchmark["year"] == benchmark_year].copy()
    champion_row = year_df[year_df["champion_flag"] == 1]
    runner_row = year_df[year_df["runner_up_flag"] == 1]
    b1, b2, b3 = st.columns(3)
    if not champion_row.empty:
        b1.metric("Champion contender rank", int(champion_row.iloc[0]["contender_rank"]))
    if not runner_row.empty:
        b2.metric("Runner-up contender rank", int(runner_row.iloc[0]["contender_rank"]))
    b3.metric("Champion in top 5", "Yes" if (not champion_row.empty and int(champion_row.iloc[0]["contender_rank"]) <= 5) else "No")

    st.dataframe(
        year_df.head(12).assign(
            rating=lambda d: d["rating"].round(1),
            champion_flag=lambda d: d["champion_flag"].map({1: "Champion", 0: ""}),
            runner_up_flag=lambda d: d["runner_up_flag"].map({1: "Runner-up", 0: ""}),
        ).rename(columns={"team": "Team", "rating": "Pre-tournament rating", "contender_rank": "Rank", "champion_flag": "Champion", "runner_up_flag": "Runner-up"})[["Rank", "Team", "Pre-tournament rating", "Champion", "Runner-up"]],
        use_container_width=True,
        hide_index=True,
    )

with tab_diag:
    st.subheader("Feature importance")
    d1, d2 = st.columns(2)
    with d1:
        st.image(str(CHARTS / "classifier_feature_importance.png"), caption="Classifier feature importance")
    with d2:
        st.image(str(CHARTS / "goal_model_feature_importance.png"), caption="Goal-model feature importance")

    st.subheader("Calibration")
    st.image(str(CHARTS / "calibration_plots.png"), caption="Probability calibration plots")
    with st.expander("Raw importance tables"):
        st.dataframe(load_csv(CHARTS / "classifier_feature_importance.csv"), use_container_width=True, hide_index=True)
        st.dataframe(load_csv(CHARTS / "goal_model_feature_importance.csv"), use_container_width=True, hide_index=True)

with tab_hooks:
    st.subheader("Squad-strength hooks")
    st.caption("Edit this CSV offline before rerunning the pipeline to reflect injuries, absences, or analyst overrides.")
    st.dataframe(squad_hooks, use_container_width=True, hide_index=True)
    hleft, hright = st.columns(2)
    with hleft:
        st.download_button(
            "Download squad hooks CSV",
            data=(RAW / "squad_strength_hooks.csv").read_bytes(),
            file_name="squad_strength_hooks.csv",
            mime="text/csv",
        )
    with hright:
        st.download_button(
            "Download squad source template CSV",
            data=(RAW / "squad_strength_sources_template.csv").read_bytes(),
            file_name="squad_strength_sources_template.csv",
            mime="text/csv",
        )
    st.markdown(
        """
        Suggested workflow:
        1. update `squad_strength_hooks.csv`
        2. document your evidence in `squad_strength_sources_template.csv`
        3. rerun `python src/main.py`
        4. refresh the Streamlit app
        """
    )
