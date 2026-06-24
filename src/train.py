from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_poisson_deviance
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

from features import build_preprocessor

FEATURE_NUMERIC = [
    "is_neutral",
    "same_confederation",
    "team_a_host_flag",
    "team_b_host_flag",
    "team_a_rating",
    "team_b_rating",
    "rating_gap_ab",
    "team_a_form_ppg_5",
    "team_b_form_ppg_5",
    "form_gap_ppg_5",
    "team_a_gd_5",
    "team_b_gd_5",
    "gd_gap_5",
    "team_a_gf_10",
    "team_b_gf_10",
    "team_a_ga_10",
    "team_b_ga_10",
    "team_a_opp_avg_rating_10",
    "team_b_opp_avg_rating_10",
    "team_a_rest_days",
    "team_b_rest_days",
    "rest_gap_ab",
]
FEATURE_CATEGORICAL = ["competition_type", "home_confederation", "away_confederation"]
TARGET = "target_result_3class"
TARGET_GOALS_A = "target_goals_a"
TARGET_GOALS_B = "target_goals_b"



def _subset(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if start is not None:
        out = out[out["match_date"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["match_date"] <= pd.Timestamp(end)]
    return out.copy()



def build_candidate_classifiers(random_state: int) -> Dict[str, Pipeline]:
    logistic = Pipeline(
        steps=[
            ("prep", build_preprocessor()),
            ("model", LogisticRegression(max_iter=2000, C=1.0)),
        ]
    )
    candidates: Dict[str, Pipeline] = {"logistic_regression": logistic}

    if XGBClassifier is not None:
        xgb = Pipeline(
            steps=[
                ("prep", build_preprocessor()),
                (
                    "model",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=3,
                        eval_metric="mlogloss",
                        n_estimators=250,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        min_child_weight=2,
                        reg_lambda=1.0,
                        tree_method="hist",
                        random_state=random_state,
                        n_jobs=2,
                    ),
                ),
            ]
        )
        candidates["xgboost"] = xgb

    return candidates



def build_poisson_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", build_preprocessor()),
            ("model", PoissonRegressor(alpha=0.3, max_iter=500)),
        ]
    )



def multiclass_brier_score(y_true, proba, classes) -> float:
    y_onehot = pd.get_dummies(pd.Categorical(y_true, categories=classes)).to_numpy()
    return float(((proba - y_onehot) ** 2).sum(axis=1).mean())



def _get_classes(model):
    if hasattr(model, "classes_"):
        return model.classes_
    estimator = getattr(model, "estimator", None)
    if estimator is not None and hasattr(estimator, "classes_"):
        return estimator.classes_
    if hasattr(model, "named_steps") and "model" in model.named_steps and hasattr(model.named_steps["model"], "classes_"):
        return model.named_steps["model"].classes_
    raise AttributeError("Could not determine class labels from model")



def fit_calibrated_classifier(base_model: Pipeline, X_train, y_train, X_calib, y_calib):
    fitted = clone(base_model)
    fitted.fit(X_train, y_train)
    if len(X_calib) < 50:
        return fitted
    calibrated = CalibratedClassifierCV(fitted, method="sigmoid", cv="prefit")
    calibrated.fit(X_calib, y_calib)
    return calibrated



def evaluate_classifier(model, X_eval, y_eval) -> dict:
    proba = model.predict_proba(X_eval)
    preds = model.predict(X_eval)
    classes = _get_classes(model)
    return {
        "accuracy": float(accuracy_score(y_eval, preds)),
        "log_loss": float(log_loss(y_eval, proba, labels=classes)),
        "brier_multiclass": float(multiclass_brier_score(y_eval, proba, classes)),
    }



def _clip_lambda(values):
    return values.clip(0.05, 5.0)



def fit_goal_models(X_train, y_train_a, y_train_b):
    model_a = build_poisson_model()
    model_b = build_poisson_model()
    model_a.fit(X_train, y_train_a)
    model_b.fit(X_train, y_train_b)
    return model_a, model_b



def evaluate_goal_models(model_a, model_b, X_eval, y_eval_a, y_eval_b) -> dict:
    pred_a = _clip_lambda(pd.Series(model_a.predict(X_eval)))
    pred_b = _clip_lambda(pd.Series(model_b.predict(X_eval)))
    return {
        "team_a_goal_mae": float(mean_absolute_error(y_eval_a, pred_a)),
        "team_b_goal_mae": float(mean_absolute_error(y_eval_b, pred_b)),
        "team_a_mean_poisson_deviance": float(mean_poisson_deviance(y_eval_a, pred_a)),
        "team_b_mean_poisson_deviance": float(mean_poisson_deviance(y_eval_b, pred_b)),
    }



def train_and_evaluate(
    model_df: pd.DataFrame,
    selection_train_end: str,
    selection_calib_start: str,
    selection_calib_end: str,
    evaluation_test_start: str,
    deploy_train_end: str,
    deploy_calib_start: str,
    deploy_calib_end: str,
    random_state: int,
):
    train_df = _subset(model_df, end=selection_train_end)
    calib_df = _subset(model_df, start=selection_calib_start, end=selection_calib_end)
    test_df = _subset(model_df, start=evaluation_test_start)

    X_train = train_df[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
    y_train = train_df[TARGET]
    X_calib = calib_df[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
    y_calib = calib_df[TARGET]
    X_test = test_df[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
    y_test = test_df[TARGET]

    candidate_metrics = {}
    fitted_eval_classifiers = {}
    for name, base_model in build_candidate_classifiers(random_state).items():
        calibrated_model = fit_calibrated_classifier(base_model, X_train, y_train, X_calib, y_calib)
        candidate_metrics[name] = evaluate_classifier(calibrated_model, X_test, y_test)
        fitted_eval_classifiers[name] = calibrated_model

    best_name = min(candidate_metrics, key=lambda name: candidate_metrics[name]["log_loss"])

    eval_goal_model_a, eval_goal_model_b = fit_goal_models(X_train, train_df[TARGET_GOALS_A], train_df[TARGET_GOALS_B])
    goal_eval_metrics = evaluate_goal_models(eval_goal_model_a, eval_goal_model_b, X_test, test_df[TARGET_GOALS_A], test_df[TARGET_GOALS_B])

    deploy_train_df = _subset(model_df, end=deploy_train_end)
    deploy_calib_df = _subset(model_df, start=deploy_calib_start, end=deploy_calib_end)
    X_deploy_train = deploy_train_df[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
    y_deploy_train = deploy_train_df[TARGET]
    X_deploy_calib = deploy_calib_df[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
    y_deploy_calib = deploy_calib_df[TARGET]

    final_classifier = fit_calibrated_classifier(
        build_candidate_classifiers(random_state)[best_name],
        X_deploy_train,
        y_deploy_train,
        X_deploy_calib,
        y_deploy_calib,
    )

    X_goal_deploy = pd.concat([X_deploy_train, X_deploy_calib], axis=0)
    y_goal_deploy_a = pd.concat([deploy_train_df[TARGET_GOALS_A], deploy_calib_df[TARGET_GOALS_A]], axis=0)
    y_goal_deploy_b = pd.concat([deploy_train_df[TARGET_GOALS_B], deploy_calib_df[TARGET_GOALS_B]], axis=0)
    final_goal_model_a, final_goal_model_b = fit_goal_models(X_goal_deploy, y_goal_deploy_a, y_goal_deploy_b)

    metrics = {
        "selection_train_rows": int(len(train_df)),
        "selection_calibration_rows": int(len(calib_df)),
        "evaluation_test_rows": int(len(test_df)),
        "deployment_train_rows": int(len(deploy_train_df)),
        "deployment_calibration_rows": int(len(deploy_calib_df)),
        "selected_classifier": best_name,
        "classifier_candidate_metrics": candidate_metrics,
        "goal_model_metrics": goal_eval_metrics,
    }
    models = {
        "classifier": final_classifier,
        "goal_model_a": final_goal_model_a,
        "goal_model_b": final_goal_model_b,
    }
    artifacts = {
        "selected_eval_classifier": fitted_eval_classifiers[best_name],
        "goal_eval_model_a": eval_goal_model_a,
        "goal_eval_model_b": eval_goal_model_b,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_a": test_df[TARGET_GOALS_A],
        "y_test_b": test_df[TARGET_GOALS_B],
        "selected_classifier_name": best_name,
    }
    return models, metrics, artifacts



def save_metrics(metrics: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(metrics, indent=2))
