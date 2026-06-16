"""Configuration centrale du projet de classification.
C'est le SEUL fichier a adapter pour brancher votre propre jeu de donnees :
data.py, features.py et les scripts d'entrainement lisent toutes leurs
colonnes via ces constantes. Voir tp/TP_S0_projet_personnel.md.
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
# TODO (S0-1) : chemin vers votre fichier de donnees (CSV) place dans data/
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
# TODO (S0-2) : nom de la colonne cible binaire (valeurs 0/1)
TARGET = "Class"
# TODO (S0-3) : colonnes numeriques de votre dataset
NUMERIC_FEATURES: list[str] = [
    "Time", "Amount",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
]
# TODO (S0-4) : colonnes categorielles (peut rester vide : [])
CATEGORICAL_FEATURES: list[str] = []
RANDOM_STATE = 42
# Surcouche via variables d'environnement (principe 12-factor)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "classification-baseline")
MODEL_NAME = os.getenv("MODEL_NAME", "classifier")
# --- Infrastructure fournie (pas a modifier pour brancher le dataset) ---------
# Description et tags de l'experience MLflow (utilises par tracking.py, TP S5).
MLFLOW_EXPERIMENT_DESCRIPTION = os.getenv(
    "MLFLOW_EXPERIMENT_DESCRIPTION",
    "Detection de fraude par carte bancaire - cours MLOps",
)
def _parse_tags(raw: str) -> dict[str, str]:
    """Parser une chaine `cle=valeur,cle2=valeur2` en dictionnaire de tags."""
    tags: dict[str, str] = {}
    for pair in raw.split(","):
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        key, value = key.strip(), value.strip()
        if key:
            tags[key] = value
    return tags
MLFLOW_EXPERIMENT_TAGS = _parse_tags(os.getenv("MLFLOW_EXPERIMENT_TAGS", "course=mlops"))
# Seuils de la porte qualite (evaluate.py, TP S11) : modele rejete si en dessous.
EVAL_ROC_AUC_MIN = float(os.getenv("EVAL_ROC_AUC_MIN", "0.65"))
EVAL_F1_MIN = float(os.getenv("EVAL_F1_MIN", "0.55"))
# URL de l'API FastAPI ciblee par les clients (DAG de previsions, predict_client).
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
