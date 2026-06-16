"""Configuration partagee du suivi MLflow.

Seance 5 - TP MLflow Tracking (suite)
    Centralise la configuration du tracking pour eviter de la dupliquer dans
    chaque script d'entrainement, et ajoute la tracabilite des donnees
    (dataset lineage).
"""
from __future__ import annotations

import logging

import mlflow
import mlflow.data
import pandas as pd

from mlproject.config import (
    DATA_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_EXPERIMENT_DESCRIPTION,
    MLFLOW_EXPERIMENT_TAGS,
    MLFLOW_TRACKING_URI,
    TARGET,
)

logger = logging.getLogger(__name__)


def setup_experiment() -> None:
    """Configurer le tracking MLflow et les metadonnees de l'experience."""
    # TODO (S5-8)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT)
    client = mlflow.MlflowClient()
    if MLFLOW_EXPERIMENT_DESCRIPTION:
        client.set_experiment_tag(
            experiment.experiment_id,
            "mlflow.note.content",
            MLFLOW_EXPERIMENT_DESCRIPTION,
        )
    for key, value in MLFLOW_EXPERIMENT_TAGS.items():
        client.set_experiment_tag(experiment.experiment_id, key, value)


def log_dataset(df: pd.DataFrame, context: str, name: str = "dataset") -> None:
    """Logger un dataset MLflow dans le run courant (tracabilite donnees -> modele).

    Parameters
    ----------
    df : pandas.DataFrame
        Donnees a referencer (features + cible).
    context : str
        Role du dataset dans le run, par exemple "training" ou "evaluation".
    name : str, optional
        Nom logique du dataset, par defaut "dataset".
    """
    # TODO (S5-9)
    dataset = mlflow.data.from_pandas(
        df, source=str(DATA_PATH), targets=TARGET, name=name
    )
    mlflow.log_input(dataset, context=context)
