# TP - Seance 7 : AutoML (GridSearchCV) et explicabilite avec SHAP

**Duree : 1 h · Pre-requis : MLflow Tracking et Model Registry operationnels (Seances 5 et 6)**

## Objectifs
- Comparer plusieurs familles de modeles (Random Forest, XGBoost, LightGBM) via une
  recherche d'hyperparametres en grille (`GridSearchCV`).
- Suivre chaque famille dans MLflow (runs imbriques) et n'enregistrer que la meilleure
  dans le Model Registry.
- Ajouter un graphique d'explicabilite SHAP (importance des variables) comme artefact
  MLflow.

## Contexte
Jusqu'ici (S5), un seul modele (regression logistique) etait entraine et suivi dans
MLflow. Pour ce TP, on entraine et compare **trois familles de modeles** sur le meme
jeu de donnees, chacune optimisee independamment, puis on ne garde que la meilleure
(selon le ROC AUC de test). C'est une forme simple d'AutoML : explorer
automatiquement plusieurs algorithmes et configurations, et selectionner le meilleur.

On y ajoute aussi **SHAP** (SHapley Additive exPlanations), qui explique l'importance
de chaque variable dans les predictions du modele retenu - utile pour justifier une
decision (pourquoi ce client est-il predit "a risque" ?) et detecter des biais.

Le squelette à compléter se trouve dans `todo/mlproject/train_models.py` : completez les
`TODO S7-n`. `mlproject.evaluation.log_shap_summary` (deja fourni) calcule et loggue le
graphique SHAP.

## Etape 1 - Importer les modeles et GridSearchCV (`TODO S7-1`)
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from mlproject.config import RANDOM_STATE
```

## Etape 2 - Definir les grilles d'hyperparametres par famille (`TODO S7-2`)
`build_model_specs()` retourne une liste de `ModelSpec`, un par famille de modeles.
Chaque `ModelSpec` porte un `estimator` (instance scikit-learn) et un `param_grid`
(grille `GridSearchCV`, cles prefixees par `clf__` car le classifieur est la derniere
etape du pipeline) :

```python
ModelSpec(
    name="random_forest",
    estimator=RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid={
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_leaf": [1, 2],
    },
)
```
Procedez de meme pour `xgboost` (`XGBClassifier(random_state=RANDOM_STATE,
eval_metric="logloss", n_jobs=-1)`, grille `clf__n_estimators` [100, 200],
`clf__max_depth` [3, 5], `clf__learning_rate` [0.1, 0.01]) et `lightgbm`
(`LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)`, grille `clf__n_estimators`
[100, 200], `clf__num_leaves` [31, 63], `clf__learning_rate` [0.1, 0.01]).

## Etape 3 - Optimiser chaque famille avec GridSearchCV (`TODO S7-3`)
```python
def optimize_model(spec, x_train, y_train, x_test, y_test, cv=5, scoring="roc_auc"):
    search = GridSearchCV(
        estimator=build_pipeline(spec.estimator),
        param_grid=spec.param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)

    best = search.best_estimator_
    proba = best.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return FitResult(
        name=spec.name,
        best_estimator=best,
        best_params=search.best_params_,
        cv_score=float(search.best_score_),
        f1=float(f1_score(y_test, preds)),
        roc_auc=float(roc_auc_score(y_test, proba)),
        preds=preds,
    )
```
`GridSearchCV` essaie **toutes les combinaisons** de la grille (contrairement a Optuna
qui en echantillonne intelligemment un sous-ensemble, voir TP S6).

## Etape 4 - Logger chaque famille dans MLflow (`TODO S7-4a`, `TODO S7-4b`)
Dans `log_run_to_mlflow`, chaque famille est un run imbrique (`nested=True`) sous le
run parent `compare-models` :
```python
mlflow.log_params(result.best_params)
mlflow.log_metrics({
    f"cv_{scoring}": result.cv_score,
    "f1": result.f1,
    "roc_auc": result.roc_auc,
})
```
Puis, apres le rapport de classification, ajoutez le graphique SHAP :
```python
log_shap_summary(result.best_estimator, x_test, result.name)
```
`log_shap_summary` (deja fourni dans `mlproject.evaluation`) calcule les valeurs SHAP sur
un echantillon du jeu de test via `shap.TreeExplainer` (compatible Random Forest,
XGBoost, LightGBM) et loggue un `summary_plot` comme artefact `shap_summary.png`.

## Etape 5 - Lancer et comparer
```bash
PYTHONPATH=todo python -m mlproject.train_models --cv 5 --scoring roc_auc
```
Dans l'UI MLflow (experience `classification-baseline`) :
- ouvrez le run parent `compare-models` et ses runs imbriques `random_forest`,
  `xgboost`, `lightgbm` ;
- pour chaque run, onglet *Artifacts* : `confusion_matrix.png`,
  `classification_report.json`, et **`shap_summary.png`** (importance des variables) ;
- le tag `best_model` sur le run parent indique la famille enregistree dans le Model
  Registry sous `classifier`.

Dans le frontend (`make frontend`), onglet "Evaluation", selectionnez cette nouvelle
version : le graphique SHAP doit s'afficher dans la section "Importance des variables
(SHAP)".

## Etape 6 - Documenter la version dans le Model Registry (`TODO S7-5 bonus`)
Comme au TP S6, completez `describe_registered_version` pour ajouter une description
et des tags via `mlflow.MlflowClient` :
```python
def describe_registered_version(name, version, result, cv, scoring):
    client = mlflow.MlflowClient()
    description = (
        f"Modele {result.name} optimise par GridSearchCV (cv={cv}, scoring={scoring}).\n"
        f"Meilleurs hyperparametres : {result.best_params}\n"
        f"Metriques de test : f1={result.f1:.3f}, roc_auc={result.roc_auc:.3f}"
    )
    client.update_model_version(name=name, version=str(version), description=description)
    tags = {
        "model_family": result.name,
        "search_method": "GridSearchCV",
        "cv": str(cv),
        "scoring": scoring,
        "f1": f"{result.f1:.4f}",
        "roc_auc": f"{result.roc_auc:.4f}",
    }
    for key, value in tags.items():
        client.set_model_version_tag(name=name, version=str(version), key=key, value=value)
```
Dans `log_run_to_mlflow`, appelez cette fonction si `register_as` est defini et que
`model_info.registered_model_version is not None`.

## Criteres de reussite
- [ ] `python -m mlproject.train_models` s'execute sans erreur.
- [ ] Le run parent `compare-models` et un run par famille (`random_forest`,
      `xgboost`, `lightgbm`) sont visibles dans l'UI, chacun avec ses metriques.
- [ ] Chaque run de famille contient l'artefact `shap_summary.png`.
- [ ] Le meilleur modele (toutes familles confondues) est enregistre dans le Model
      Registry sous `classifier`, et `models/model.joblib` est mis a jour.
- [ ] (Bonus) La version enregistree affiche une description et des tags
      (`model_family`, `search_method`, `cv`, `scoring`, `f1`, `roc_auc`).
- [ ] Le frontend (onglet "Evaluation") affiche le graphique SHAP pour cette version.

## Pour aller plus loin
- **Comparaison avec Optuna** : relancez `train_optuna.py` (TP S6) avec un budget
  d'essais comparable et comparez `test_roc_auc` famille par famille avec
  `train_models.py` (GridSearchCV).
- **SHAP par instance** : `shap.force_plot` ou `shap.waterfall_plot` pour expliquer
  une prediction individuelle (utile pour justifier un refus de credit, par exemple).
- **Importance globale vs locale** : le `summary_plot` (global) montre l'effet moyen
  de chaque variable sur l'ensemble du jeu de test ; les graphiques par instance
  montrent l'effet pour un client precis.
