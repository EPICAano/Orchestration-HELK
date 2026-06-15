# TP - Seance 0 : Choisir votre dataset et votre problematique de classification

**Duree : 30-45 min · Pre-requis : aucun (a faire avant la Seance 5)**

## Objectif
Le dossier `todo/` contient un projet de classification **generique et autonome** (package
`mlproject`) que vous adaptez a VOTRE propre cas d'usage. Tout le pipeline (chargement,
pre-processing, entrainement, API, frontend, Airflow) est pilote par les constantes de
`todo/mlproject/config.py` : c'est le seul fichier a modifier pour brancher un nouveau jeu de
donnees. La seule contrainte : il doit s'agir d'un probleme de **classification binaire**
(deux classes, ex. fraude/non-fraude, defaut/pas de defaut, spam/non-spam, churn/pas de
churn, sain/malade...).

## Etape 0 - Mettre en place l'environnement
Travaillez depuis la racine du projet en pointant Python sur le dossier `todo/` :
```bash
make install            # installe les dependances
export PYTHONPATH=todo   # rend le package `mlproject` importable
```
Toutes les commandes des TP suivants utilisent `PYTHONPATH=todo python -m mlproject.<module>`.

## Etape 1 - Choisir un dataset et une problematique
Choisissez un jeu de donnees tabulaire avec :
- une colonne cible **binaire** (0/1, ou deux valeurs convertibles en 0/1) ;
- des colonnes numeriques (ages, montants, durees, compteurs...) ;
- (optionnel) des colonnes categorielles (type de contrat, categorie, region...).

Exemples de problematiques de classification binaire : detection de fraude, defaut de
paiement, attrition (client ou employe), conversion publicitaire, detection de spam,
diagnostic medical binaire (sur un dataset pedagogique de type Kaggle).

Placez votre fichier CSV dans `data/`, par exemple `data/dataset.csv`, et identifiez :
le nom de la colonne cible, la liste des colonnes numeriques, la liste des colonnes
categorielles.

## Etape 2 - Adapter `todo/mlproject/config.py`
C'est le **seul fichier a modifier** pour brancher votre dataset. Completez les
`TODO (S0-n)` :
```python
# TODO (S0-1)
DATA_PATH = ROOT / "data" / "dataset.csv"   # votre fichier CSV
# TODO (S0-2)
TARGET = "ma_colonne_cible"                  # colonne binaire (0/1)
# TODO (S0-3)
NUMERIC_FEATURES = ["age", "montant", "duree"]   # vos colonnes numeriques
# TODO (S0-4)
CATEGORICAL_FEATURES = ["categorie", "region"]   # vos colonnes categorielles (peut etre [])

MLFLOW_EXPERIMENT = "mon-projet-baseline"   # nom de l'experience MLflow
MODEL_NAME = "mon-projet-classifier"        # nom du modele dans le Model Registry
```
Verifiez que :
- `TARGET` ne contient que des valeurs `0`/`1` (sinon, ajoutez un mapping dans un script de
  preparation qui ecrit le CSV final dans `data/`) ;
- chaque colonne de `NUMERIC_FEATURES`/`CATEGORICAL_FEATURES` existe dans votre CSV et n'est
  pas la cible.

`data.py` et `features.py` (deja fournis dans `todo/mlproject/`) lisent automatiquement ces
constantes : vous n'avez pas a les modifier.

## Etape 3 - Verifier que le pipeline tourne
```bash
PYTHONPATH=todo python -m mlproject.train
```
Vous devriez voir s'afficher `f1=... roc_auc=...` sans erreur. Une erreur pointant vers
`features.py`/`data.py` signale en general un nom de colonne mal orthographie ou des valeurs
manquantes non gerees dans `config.py`.

## Etape 4 - Adapter l'API et le frontend (a partir de la Seance 12)
A la Seance 12, le schema d'entree `Features` (`todo/mlproject/api.py`) devra reprendre
**exactement** les colonnes de `NUMERIC_FEATURES` + `CATEGORICAL_FEATURES`, avec leurs types
et contraintes : le `DataFrame` construit dans `/predict` doit avoir les memes colonnes qu'a
l'entrainement. Le formulaire du frontend (Seance 14 bis) reprend egalement ces champs.

## Feuille de route du semestre
Une fois `config.py` adapte, suivez les TP dans l'ordre (squelettes dans `todo/mlproject/`) :

1. `tp/TP_S5_mlflow.md` (`todo/mlproject/train.py`) : suivi MLflow de la baseline.
2. `tp/TP_S6_optuna.md` (`todo/mlproject/train_optuna.py`) : optimisation Optuna + Registry.
3. `tp/TP_S7_automl_shap.md` (`todo/mlproject/train_models.py`) : comparaison de modeles
   (GridSearchCV) + explicabilite SHAP.
4. `tp/TP_S8_docker.md` (`todo/docker/Dockerfile.train`) : conteneuriser l'entrainement.
5. `tp/TP_S12_fastapi.md` (`todo/mlproject/api.py`) : exposer le modele via une API FastAPI.
6. `tp/TP_S14_docker_compose.md` et `tp/TP_S14_bis_streamlit.md` : stack complete et frontend.
7. `tp/TP_S17_airflow.md` (`todo/dags/retrain_dag.py`) : planifier le re-entrainement.

## Criteres de reussite
- [ ] `todo/mlproject/config.py` reference votre dataset (`DATA_PATH`, `TARGET`,
      `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`).
- [ ] `PYTHONPATH=todo python -m mlproject.train` s'execute sans erreur et affiche des
      metriques `f1`/`roc_auc` superieures au hasard (> 0.5).
- [ ] `MLFLOW_EXPERIMENT` et `MODEL_NAME` sont nommes d'apres votre projet.
- [ ] Vous savez expliquer en une phrase la problematique metier (qui sont les `1`, qui sont
      les `0`, et pourquoi predire cette cible est utile).

## Pour aller plus loin
- Dataset desequilibre (peu de `1`) : surveillez `roc_auc` plutot que `accuracy`, et
  envisagez `class_weight="balanced"` (a discuter en seance).
- Sans colonnes categorielles, `CATEGORICAL_FEATURES = []` fonctionne tel quel (le
  `ColumnTransformer` ignore simplement cette branche).
