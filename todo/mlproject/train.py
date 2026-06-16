"""Entrainement du modele de classification (baseline).

Seance 5 - TP MLflow Tracking
    Ce script entraine et evalue un modele AVEC suivi d'experience MLflow.
"""
from __future__ import annotations

import argparse

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from mlproject.config import MODEL_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT
from mlproject.data import load_data, split
from mlproject.features import build_preprocessor

# TODO (S5-1) : importer mlflow et mlflow.sklearn
import mlflow
import mlflow.sklearn


def build_model(c: float = 1.0, max_iter: int = 1000) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", LogisticRegression(C=c, max_iter=max_iter)),
        ]
    )


def train(c: float = 1.0, max_iter: int = 1000) -> dict:
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    # TODO (S5-2) : configurer l'URI de tracking et l'experience
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # TODO (S5-3) : ouvrir un run englobant l'entrainement et l'evaluation
    with mlflow.start_run(run_name=f"logreg-c{c}"):
        model = build_model(c=c, max_iter=max_iter)
        model.fit(x_train, y_train)

        proba = model.predict_proba(x_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        metrics = {
            "f1": float(f1_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
        }
        print(f"f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")

        # TODO (S5-4) : logger les parametres
        mlflow.log_params({"c": c, "max_iter": max_iter, "model": "logreg"})
        # TODO (S5-5) : logger les metriques
        mlflow.log_metrics(metrics)
        # TODO (S5-6) : logger le modele
        mlflow.sklearn.log_model(model, name="model")

        # TODO (S5-7 bonus) : matrice de confusion en artefact
        from sklearn.metrics import ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        ConfusionMatrixDisplay.from_predictions(y_test, preds)
        plt.savefig("confusion.png")
        plt.close()
        mlflow.log_artifact("confusion.png")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter)


if __name__ == "__main__":
    main()
