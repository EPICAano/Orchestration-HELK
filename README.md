# Projet fil rouge MLOps — Détection de fraude par carte bancaire

Journal de bord du projet personnel (module MLOps, ESGI/IABD).
Squelette du cours : voir `todo/README.md`.

## Problématique

Classification binaire : détecter les **transactions frauduleuses** par carte
bancaire parmi un flux de transactions.

- `1` = transaction frauduleuse
- `0` = transaction légitime

**En une phrase :** prédire cette cible permet à un système de paiement de
signaler une transaction suspecte en temps réel, avant la perte d'argent.

## Jeu de données

**Credit Card Fraud Detection** — Machine Learning Group, Université Libre de
Bruxelles (ULB). Transactions de cartes bancaires européennes, septembre 2013.

- Source : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Fichier : `data/creditcard.csv` (~150 Mo, non versionné)
- Volume : 284 807 transactions, dont 492 fraudes (~0,172 %) — **fortement
  déséquilibré**.
- Colonnes :
  - `Time`, `Amount` : numériques brutes.
  - `V1`…`V28` : composantes issues d'une ACP (anonymisation), numériques.
  - `Class` : cible binaire (0/1).

Toutes les variables explicatives sont numériques, donc
`CATEGORICAL_FEATURES = []`.

> Le fichier `data/creditcard.csv` n'est pas versionné (volumineux). Pour
> reproduire : télécharger depuis le lien Kaggle ci-dessus et le placer dans
> `data/`.

## Avancement par séance

| Séance | État | Détail |
|--------|------|--------|
| S0 — Brancher le dataset | ✅ Fait | `mlproject/config.py` adapté (DATA_PATH, TARGET, NUMERIC_FEATURES, CATEGORICAL_FEATURES). Entraînement vérifié. |
| S5 — MLflow | ⏳ À faire | `mlproject/train.py` |
| S6 — Optuna + Registry | ⏳ À faire | `mlproject/train_optuna.py` |
| S7 — AutoML + SHAP | ⏳ À faire | `mlproject/train_models.py` |
| S8 — Docker train | ⏳ À faire | `docker/Dockerfile.train` |
| S12 — API FastAPI | ⏳ À faire | `mlproject/api.py` |
| S14 — docker-compose | ⏳ À faire | `docker-compose.yml` |
| S14bis — Frontend Streamlit | ⏳ À faire | `frontend/app.py` |
| S17 — Airflow | ⏳ À faire | `dags/retrain_dag.py` |

## S0 — Résultat

Commande exécutée depuis `todo/` :

```bash
uv run python -m mlproject.train
```

Sortie obtenue :
f1=0.717  roc_auc=0.961

Les deux métriques dépassent 0,5 : le critère de réussite S0 est rempli.

## Notes d'installation (spécifiques à ce poste)

Le `pyproject.toml` était absent du dépôt de cours ; il a été reconstruit à
partir des imports réels du code (`mlproject`, `frontend`, `dags`). À remplacer
si l'enseignant publie le fichier officiel.

- Python épinglé à **3.13** (`.python-version`) — `llvmlite`/`numba` ne se
  construisent pas sur Python 3.14.
- `numba>=0.61` forcé pour obtenir un `llvmlite` récent avec wheel (sinon échec
  de compilation).
- **Airflow non inclus** dans les dépendances : à ajouter en S17 (tourne via
  Docker dans la stack du cours).

Installation :

```bash
make -C todo install
```

## Remarque modèle

Dataset très déséquilibré : surveiller `roc_auc` plutôt que `accuracy`. Si le
`f1` chute sur de futures variantes, envisager `class_weight="balanced"` (piste
S5).
