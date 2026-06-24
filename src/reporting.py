from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from train import FEATURE_CATEGORICAL, FEATURE_NUMERIC

CLASS_LABELS = {0: "Away win", 1: "Draw", 2: "Team A win"}



def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def save_classifier_feature_importance(model, X_eval: pd.DataFrame, y_eval: pd.Series, output_dir: Path) -> Path:
    _ensure_dir(output_dir)
    result = permutation_importance(
        model,
        X_eval[FEATURE_NUMERIC + FEATURE_CATEGORICAL],
        y_eval,
        scoring="neg_log_loss",
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )
    importances = pd.DataFrame({
        "feature": FEATURE_NUMERIC + FEATURE_CATEGORICAL,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importances.to_csv(output_dir / "classifier_feature_importance.csv", index=False)

    top = importances.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#2563eb", alpha=0.85)
    ax.set_title("Classifier feature importance\nPermutation importance on holdout set")
    ax.set_xlabel("Importance (drop in negative log loss)")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out = output_dir / "classifier_feature_importance.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out



def save_goal_model_feature_importance(goal_model_a, goal_model_b, X_eval: pd.DataFrame, y_eval_a: pd.Series, y_eval_b: pd.Series, output_dir: Path) -> Path:
    _ensure_dir(output_dir)
    res_a = permutation_importance(
        goal_model_a,
        X_eval[FEATURE_NUMERIC + FEATURE_CATEGORICAL],
        y_eval_a,
        scoring="neg_mean_poisson_deviance",
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )
    res_b = permutation_importance(
        goal_model_b,
        X_eval[FEATURE_NUMERIC + FEATURE_CATEGORICAL],
        y_eval_b,
        scoring="neg_mean_poisson_deviance",
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )
    imp = pd.DataFrame({
        "feature": FEATURE_NUMERIC + FEATURE_CATEGORICAL,
        "team_a_goal_importance": res_a.importances_mean,
        "team_b_goal_importance": res_b.importances_mean,
    })
    imp["combined_importance"] = imp[["team_a_goal_importance", "team_b_goal_importance"]].abs().mean(axis=1)
    imp = imp.sort_values("combined_importance", ascending=False)
    imp.to_csv(output_dir / "goal_model_feature_importance.csv", index=False)

    top = imp.head(15).iloc[::-1]
    y_pos = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(y_pos - 0.18, top["team_a_goal_importance"], height=0.35, label="Team A goals", color="#16a34a")
    ax.barh(y_pos + 0.18, top["team_b_goal_importance"], height=0.35, label="Team B goals", color="#f59e0b")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["feature"])
    ax.set_title("Goal model feature importance\nPermutation importance on holdout set")
    ax.set_xlabel("Importance (drop in negative Poisson deviance score)")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = output_dir / "goal_model_feature_importance.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out



def save_calibration_plot(model, X_eval: pd.DataFrame, y_eval: pd.Series, output_dir: Path) -> Path:
    _ensure_dir(output_dir)
    proba = model.predict_proba(X_eval[FEATURE_NUMERIC + FEATURE_CATEGORICAL])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for idx, ax in enumerate(axes):
        y_bin = (y_eval.to_numpy() == idx).astype(int)
        frac_pos, mean_pred = calibration_curve(y_bin, proba[:, idx], n_bins=10, strategy="quantile")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
        ax.plot(mean_pred, frac_pos, marker="o", color="#2563eb")
        ax.set_title(CLASS_LABELS[idx])
        ax.set_xlabel("Predicted probability")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Observed frequency")
    fig.suptitle("Calibration plots for 3-class match model")
    fig.tight_layout()
    out = output_dir / "calibration_plots.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out
