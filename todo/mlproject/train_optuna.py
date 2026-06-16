"""Optimisation d'hyperparametres avec Optuna.

Seance 6 - TP Optuna
    Optimise les hyperparametres de trois familles (RF, XGBoost, LightGBM)
    avec Optuna (sampler TPE), compare et persiste le meilleur.
    Suivi MLflow via la configuration partagee (mlproject.tracking).
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

# TODO (S6-1)
import optuna
import optuna.samplers
from sklearn.model_selection import cross_val_score

from mlproject.config import MODEL_DIR, MODEL_NAME, RANDOM_STATE
from mlproject.data import load_data, split
from mlproject.evaluation import log_shap_summary
from mlproject.features import build_preprocessor
from mlproject.tracking import setup_experiment, log_dataset

# TODO (S6-2) : imports des modeles
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    name: str
    suggest_params: Callable
    build_estimator: Callable[[dict], ClassifierMixin]


def build_model_specs() -> list[ModelSpec]:
    """Construire la liste des familles de modeles a optimiser."""
    # TODO (S6-2)
    def rf_suggest(trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }

    def rf_build(params: dict) -> ClassifierMixin:
        return cast(
            ClassifierMixin,
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params),
        )

    def xgb_suggest(trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }

    def xgb_build(params: dict) -> ClassifierMixin:
        return cast(
            ClassifierMixin,
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
                **params,
            ),
        )

    def lgbm_suggest(trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
        }

    def lgbm_build(params: dict) -> ClassifierMixin:
        return cast(
            ClassifierMixin,
            LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, **params),
        )

    return [
        ModelSpec(name="random_forest", suggest_params=rf_suggest, build_estimator=rf_build),
        ModelSpec(name="xgboost", suggest_params=xgb_suggest, build_estimator=xgb_build),
        ModelSpec(name="lightgbm", suggest_params=lgbm_suggest, build_estimator=lgbm_build),
    ]


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", estimator),
        ]
    )


def objective(trial, spec: ModelSpec, x_train, y_train, cv: int) -> float:
    """Fonction objectif Optuna : ROC AUC moyen en validation croisee."""
    # TODO (S6-3)
    params = spec.suggest_params(trial)
    estimator = spec.build_estimator(params)
    pipeline = build_pipeline(estimator)
    scores = cross_val_score(pipeline, x_train, y_train, scoring="roc_auc", cv=cv)
    return float(scores.mean())


def run_study(spec: ModelSpec, x_train, y_train, n_trials: int, cv: int):
    """Lancer l'etude Optuna pour une famille de modeles."""
    # TODO (S6-4)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    # TODO (S6-5)
    study.optimize(
        lambda trial: objective(trial, spec, x_train, y_train, cv),
        n_trials=n_trials,
    )
    return study


@dataclass
class FamilyResult:
    spec: ModelSpec
    study: Any
    best_pipeline: Pipeline
    test_roc_auc: float
    preds: np.ndarray


def optimize_family(
    spec: ModelSpec,
    x_train,
    y_train,
    x_test,
    y_test,
    n_trials: int,
    cv: int,
) -> FamilyResult:
    """Optimiser une famille avec Optuna et l'evaluer sur le test."""
    logger.info("Optimisation de %s (n_trials=%d, cv=%d)", spec.name, n_trials, cv)
    study = run_study(spec, x_train, y_train, n_trials=n_trials, cv=cv)

    best_pipeline = build_pipeline(spec.build_estimator(study.best_params))
    best_pipeline.fit(x_train, y_train)
    proba = best_pipeline.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    test_roc_auc = float(roc_auc_score(y_test, proba))

    logger.info(
        "%s : cv_roc_auc=%.3f  test_roc_auc=%.3f  params=%s",
        spec.name,
        study.best_value,
        test_roc_auc,
        study.best_params,
    )
    return FamilyResult(
        spec=spec,
        study=study,
        best_pipeline=best_pipeline,
        test_roc_auc=test_roc_auc,
        preds=preds,
    )


def log_family_to_mlflow(
    result: FamilyResult,
    x_test,
    y_test,
    n_trials: int,
    cv: int,
    register_as: str | None = None,
) -> None:
    """Logger une famille de modeles dans un run MLflow imbrique."""
    with mlflow.start_run(run_name=result.spec.name, nested=True):
        mlflow.set_tag("model_family", result.spec.name)
        mlflow.set_tag("sampler", "TPE")
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("cv", cv)

        # TODO (S6-6)
        for trial in result.study.trials:
            with mlflow.start_run(run_name=f"{result.spec.name}-trial-{trial.number}", nested=True):
                mlflow.log_params(trial.params)
                if trial.value is not None:
                    mlflow.log_metric("cv_roc_auc", trial.value)

        mlflow.log_params(result.study.best_params)
        mlflow.log_metric("cv_roc_auc", result.study.best_value)
        mlflow.log_metric("test_roc_auc", result.test_roc_auc)

        cm = confusion_matrix(y_test, result.preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        ax.set_title(f"Matrice de confusion : {result.spec.name}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        report_dict = cast(dict, classification_report(y_test, result.preds, output_dict=True))
        mlflow.log_dict(report_dict, "classification_report.json")
        report_text = cast(str, classification_report(y_test, result.preds))
        mlflow.log_text(report_text, "classification_report.txt")

        log_shap_summary(result.best_pipeline, x_test, result.spec.name)

        signature = infer_signature(x_test, result.best_pipeline.predict(x_test))
        _model_info = mlflow.sklearn.log_model(
            result.best_pipeline,
            name="model",
            signature=signature,
            input_example=x_test.iloc[:5],
            registered_model_name=register_as,
        )

        # TODO (S6-7 bonus) : documenter la version dans le registry (non implemente)


def describe_registered_version(
    name: str,
    version: int,
    result: FamilyResult,
    n_trials: int,
    cv: int,
) -> None:
    """Documenter une version enregistree dans le Model Registry (bonus S6-7)."""
    # TODO (S6-7 bonus) : non implemente pour l'instant
    raise NotImplementedError


def optimize(n_trials: int = 30, cv: int = 5, use_mlflow: bool = True) -> list[FamilyResult]:
    """Optimiser RF / XGBoost / LightGBM avec Optuna et sauvegarder le meilleur."""
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    if use_mlflow:
        setup_experiment()
        logger.info("Suivi MLflow configure via mlproject.tracking")

    results = [
        optimize_family(spec, x_train, y_train, x_test, y_test, n_trials=n_trials, cv=cv)
        for spec in build_model_specs()
    ]
    results.sort(key=lambda r: r.test_roc_auc, reverse=True)

    best = results[0]
    logger.info("Meilleure famille : %s (test_roc_auc=%.3f)", best.spec.name, best.test_roc_auc)

    if use_mlflow:
        with mlflow.start_run(run_name="optuna-compare"):
            log_dataset(df, context="training")
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("cv", cv)
            mlflow.set_tag("best_model", best.spec.name)
            for result in results:
                register_as = MODEL_NAME if result is best else None
                log_family_to_mlflow(
                    result, x_test, y_test, n_trials, cv, register_as=register_as
                )
        logger.info("Meilleur modele enregistre dans le registry sous '%s'", MODEL_NAME)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.best_pipeline, MODEL_DIR / "model.joblib")
    logger.info("Modele sauvegarde dans %s", MODEL_DIR / "model.joblib")

    return results


def main() -> None:
    """Point d'entree en ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-trials", type=int, default=30, help="Nombre d'essais Optuna par famille de modeles"
    )
    parser.add_argument("--cv", type=int, default=5, help="Nombre de plis de validation croisee")
    parser.add_argument(
        "--no-mlflow",
        dest="use_mlflow",
        action="store_false",
        help="Desactive le suivi MLflow (utile sans serveur de tracking)",
    )
    args = parser.parse_args()
    optimize(n_trials=args.n_trials, cv=args.cv, use_mlflow=args.use_mlflow)


if __name__ == "__main__":
    main()
