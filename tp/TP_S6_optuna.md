# TP — Séance 6 : Optimisation d'hyperparamètres avec Optuna

**Durée : 1 h · Pré-requis : MLflow Tracking opérationnel (Séance 5)**

## Objectifs
- Comprendre la différence entre une recherche exhaustive (`GridSearchCV`) et une
  recherche bayésienne (Optuna, sampler TPE).
- Définir un espace de recherche et une fonction objectif pour plusieurs familles de
  modèles.
- Suivre chaque essai (trial) dans MLflow, comparer les familles entre elles et
  enregistrer la meilleure dans le Model Registry.

## Contexte
Dans `todo/mlproject/train_models.py`, les hyperparamètres de Random Forest, XGBoost et
LightGBM sont optimisés par `GridSearchCV` : toutes les combinaisons de chaque grille
sont testées. Pour un espace de recherche large ou continu, c'est coûteux. Optuna
explore l'espace de façon plus intelligente : chaque essai s'appuie sur les résultats
précédents pour orienter le suivant (sampler TPE).

Ce TP reprend les **mêmes trois familles de modèles** (`random_forest`, `xgboost`,
`lightgbm`), mais chacune est optimisée par Optuna. Une étude Optuna distincte est
lancée par famille, puis les meilleures pipelines de chaque famille sont comparées sur
le jeu de test : seule la meilleure est enregistrée dans le Model Registry.

Le squelette à compléter se trouve dans `todo/mlproject/train_optuna.py` : complétez les
`TODO S6-n`.

## Étape 1 — Importer Optuna (`TODO S6-1`)
```python
import optuna
import optuna.samplers
from sklearn.model_selection import cross_val_score
```

## Étape 2 — Définir les espaces de recherche par famille (`TODO S6-2`)
`build_model_specs()` retourne une liste de `ModelSpec`, un par famille de modèles.
Chaque `ModelSpec` porte :
- `suggest_params(trial)` : échantillonne les hyperparamètres via `trial.suggest_*` ;
- `build_estimator(params)` : construit l'estimateur scikit-learn correspondant.

```python
from typing import cast

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from mlproject.config import RANDOM_STATE

ModelSpec(
    name="random_forest",
    suggest_params=lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
    },
    build_estimator=lambda params: RandomForestClassifier(random_state=RANDOM_STATE, **params),
)
```
Procédez de même pour `xgboost` (`n_estimators` 100-300, `max_depth` 3-10,
`learning_rate` en échelle log entre 0.01 et 0.3) et `lightgbm` (`n_estimators` 50-300,
`num_leaves` 15-127, `learning_rate` en échelle log entre 0.01 et 0.3, `max_depth`
3-12).

> `log=True` sur `learning_rate` : Optuna échantillonne uniformément en échelle
> logarithmique, plus adapté pour ce type de paramètre (0.01 et 0.1 sont aussi "proches"
> que 0.1 et 1.0).

> Pour LightGBM, le type de retour de `build_estimator` doit être compatible avec
> `ClassifierMixin` : utilisez `cast(ClassifierMixin, LGBMClassifier(...))` si mypy se
> plaint.

## Étape 3 — Fonction objectif (`TODO S6-3`)
```python
def objective(trial, spec, x_train, y_train, cv):
    params = spec.suggest_params(trial)
    pipeline = build_pipeline(spec.build_estimator(params))
    scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="roc_auc")
    return scores.mean()
```
Optuna **maximise** la valeur retournée par cette fonction. La même fonction
`objective` est réutilisée pour les trois familles : seul `spec` change.

## Étape 4 — Créer et lancer une étude par famille (`TODO S6-4, S6-5`)
```python
def run_study(spec, x_train, y_train, n_trials, cv):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective(trial, spec, x_train, y_train, cv),
        n_trials=n_trials,
    )
    return study
```
`optimize_family` (déjà fourni) appelle `run_study`, réentraîne le meilleur pipeline de
la famille sur `x_train`/`y_train` et calcule son `test_roc_auc`.

## Étape 5 — Logger chaque essai dans MLflow (`TODO S6-6`)
Dans `log_family_to_mlflow`, chaque famille est un run imbriqué (`nested=True`) sous le
run parent `optuna-compare`. À l'intérieur, chaque `trial` devient lui-même un run
imbriqué :
```python
for trial in result.study.trials:
    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        mlflow.log_params(trial.params)
        mlflow.log_metric("cv_roc_auc", trial.value)
```

## Étape 6 — Lancer et comparer
```bash
PYTHONPATH=todo python -m mlproject.train_optuna --n-trials 30
```
`optimize()` (déjà fourni) lance une étude Optuna par famille, trie les résultats par
`test_roc_auc` décroissant et n'enregistre que la meilleure famille dans le Model
Registry.

Dans l'UI MLflow (expérience `classification-baseline`) :
- ouvrez le run parent `optuna-compare` et ses runs imbriqués `random_forest`,
  `xgboost`, `lightgbm`, chacun avec ses propres essais `trial-0`, `trial-1`, ... ;
- comparez le `test_roc_auc` des trois familles : le tag `best_model` sur le run
  parent indique celle qui a été enregistrée ;
- comparez ces résultats avec ceux du run `compare-models` produit par
  `train_models.py` (GridSearchCV) : avec un budget d'essais comparable (ou inférieur),
  Optuna doit s'approcher du meilleur score, voire le dépasser.

## Étape 7 — Documenter la version dans le Model Registry (`TODO S6-7 bonus`)
Une version de modèle sans contexte (juste un numéro) est difficile à interpréter dans
l'UI. Complétez `describe_registered_version` pour ajouter une description et des tags
via `mlflow.MlflowClient` :
```python
def describe_registered_version(name, version, result, n_trials, cv):
    client = mlflow.MlflowClient()
    description = (
        f"Modele {result.spec.name} optimise par Optuna "
        f"(sampler TPE, n_trials={n_trials}, cv={cv}).\n"
        f"Meilleurs hyperparametres : {result.study.best_params}\n"
        f"Metriques : cv_roc_auc={result.study.best_value:.3f}, "
        f"test_roc_auc={result.test_roc_auc:.3f}"
    )
    client.update_model_version(name=name, version=str(version), description=description)
    tags = {
        "model_family": result.spec.name,
        "search_method": "optuna-tpe",
        "n_trials": str(n_trials),
        "cv": str(cv),
        "cv_roc_auc": f"{result.study.best_value:.4f}",
        "test_roc_auc": f"{result.test_roc_auc:.4f}",
    }
    for key, value in tags.items():
        client.set_model_version_tag(name=name, version=str(version), key=key, value=value)
```
Dans `log_family_to_mlflow`, renommez `_model_info` en `model_info` puis appelez cette
fonction si `register_as` est défini et que `model_info.registered_model_version is not
None`. Dans l'UI, ouvrez *Models > classifier > Version N* : la description et
les tags sont visibles sans rouvrir le run.

## Critères de réussite
- [ ] `python -m mlproject.train_optuna --n-trials 30` s'exécute sans erreur.
- [ ] Le run parent `optuna-compare` et un run par famille (`random_forest`,
      `xgboost`, `lightgbm`) sont visibles dans l'UI, chacun avec ses essais imbriqués.
- [ ] Le meilleur modèle (toutes familles confondues) est enregistré dans le Model
      Registry sous `classifier`.
- [ ] `models/model.joblib` est mis à jour avec le meilleur pipeline.
- [ ] (Bonus) La version enregistrée affiche une description et des tags
      (`model_family`, `search_method`, `n_trials`, `cv`, `cv_roc_auc`, `test_roc_auc`).

## Pour aller plus loin
- **Pruning** : passez un `pruner` (ex. `optuna.pruners.MedianPruner()`) à
  `optuna.create_study` pour arrêter tôt les essais peu prometteurs.
- **Visualisation** : `optuna.visualization.plot_optimization_history(study)` et
  `plot_param_importances(study)` (nécessite `plotly`).
- **Comparaison systématique** : relancez `train_models.py` (GridSearchCV) et
  `train_optuna.py` (Optuna) avec le même budget de temps, comparez `roc_auc` et le
  nombre de configurations explorées, famille par famille.
