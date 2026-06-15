# TP — Séance 5 : Suivi d'expériences avec MLflow Tracking

**Durée : 1 h 10 · Pré-requis : baseline fonctionnelle (`PYTHONPATH=todo python -m mlproject.train`), config.py adapté à votre dataset (TP S0)**

## Objectifs
- Lancer un serveur MLflow local.
- Instrumenter un entraînement pour tracer paramètres, métriques, modèle et artefacts.
- Comparer plusieurs runs dans l'interface MLflow.

## Contexte
Le fichier `todo/mlproject/train.py` entraîne un modèle de classification mais **ne garde aucune trace**
des expériences : impossible de comparer deux essais ou de retrouver le modèle d'un run passé.
Vous allez corriger cela avec MLflow. Les emplacements à compléter sont signalés par des
marqueurs `TODO (S5-n)`.

## Étape 1 — Démarrer le serveur de tracking (10 min)
Deux options, au choix :

```bash
# Option A — local, sans Docker
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Option B — via le docker-compose fourni
docker compose up -d mlflow
```
Ouvrez http://localhost:5000 : l'UI doit s'afficher (encore vide).

## Étape 2 — Brancher le client MLflow (`TODO S5-1, S5-2`)
Dans `train.py` :
- importez `mlflow` et `mlflow.sklearn` ;
- au début de `train()`, pointez le client sur le serveur et nommez l'expérience :

```python
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)   # déjà dans config.py
mlflow.set_experiment(MLFLOW_EXPERIMENT)
```

## Étape 3 — Encadrer l'entraînement dans un run (`TODO S5-3`)
Englobez l'entraînement **et** l'évaluation dans un bloc :
```python
with mlflow.start_run(run_name=f"logreg-c{c}"):
    ...
```

## Étape 4 — Logger params, métriques et modèle (`TODO S5-4 à S5-6`)
À l'intérieur du run :
```python
mlflow.log_params({"c": c, "max_iter": max_iter, "model": "logreg"})
mlflow.log_metrics(metrics)                       # {"f1": ..., "roc_auc": ...}
mlflow.sklearn.log_model(model, name="model")
```

## Étape 5 — Générer plusieurs runs et comparer (15 min)
```bash
PYTHONPATH=todo python -m mlproject.train --c 0.1
PYTHONPATH=todo python -m mlproject.train --c 1.0
PYTHONPATH=todo python -m mlproject.train --c 10.0
```
Dans l'UI : sélectionnez les 3 runs → **Compare**. Identifiez le meilleur `roc_auc` et
observez le modèle enregistré comme artefact de chaque run.

## Bonus (`TODO S5-7`)
Sauvegardez la matrice de confusion en image et loggez-la :
```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
ConfusionMatrixDisplay.from_predictions(y_test, preds)
plt.savefig("confusion.png"); mlflow.log_artifact("confusion.png")
```

## Critères de réussite
- [ ] Chaque exécution crée un run visible dans l'UI sous l'expérience `classification-baseline`.
- [ ] Params et métriques sont consultables et comparables entre runs.
- [ ] Le modèle est récupérable depuis l'onglet *Artifacts* d'un run.

## Pour aller plus loin (préparation S6)
Activez `mlflow.sklearn.autolog()` avant le `fit` et observez ce qui est capturé
automatiquement. On enregistrera le meilleur modèle dans le **Model Registry** en séance 6.
