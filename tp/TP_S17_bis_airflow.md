# TP - Seance 17 bis : Comprendre le fonctionnement d'Airflow

**Duree : 30 min · Pre-requis : avoir complete `dags/churn_dag.py` (TP_S17) et demarre
le service `airflow` (`make airflow`)**

## Objectif
Comprendre, etape par etape, les concepts d'Airflow et comment ils s'articulent dans
le DAG `churn_retraining` de ce projet.

## Concepts de base

**DAG (Directed Acyclic Graph)** : un pipeline defini en Python, un graphe oriente sans
cycle. Chaque fichier `.py` dans `dags/` qui contient un objet `DAG` est scanne par
Airflow et devient un pipeline planifiable.

**Task / Operator** : chaque noeud du DAG est une tache, creee via un operator. Dans
`churn_dag.py`, on utilise `PythonOperator`, qui execute une fonction Python
(`task_generate_data`, `task_train`, `task_check_quality`).

**Dependances** : l'operateur `>>` definit l'ordre d'execution :
```python
generate >> train_task >> check
```
`check_quality` ne demarre que si `train` a reussi, qui ne demarre que si
`generate_data` a reussi.

**Schedule** : `schedule="0 3 * * 1"` (syntaxe cron) dit a Airflow "cree un DAG run tous
les lundis a 3h". `catchup=False` evite de rejouer tous les runs manques depuis
`start_date`.

## Les composants qui font tourner tout ca

1. **Scheduler** : tourne en boucle, regarde l'heure et les DAGs actifs (non "paused"),
   et cree des DAG runs quand l'echeance arrive. Il decide aussi quelles taches sont
   pretes a etre executees (dependances satisfaites) et les envoie a l'executor.

2. **Executor** : execute reellement les taches. Dans notre setup
   (`AIRFLOW__CORE__EXECUTOR: SequentialExecutor`), les taches s'executent une par une,
   sequentiellement, dans le meme processus - adapte au mode standalone/SQLite, pas a
   la prod (en prod on utilise `CeleryExecutor` ou `KubernetesExecutor` pour du
   parallelisme).

3. **Metadata DB** : base (ici SQLite, fichier dans le volume `airflow_data`) qui
   stocke l'etat de tout : DAGs, runs, taches, XComs, utilisateurs... C'est la
   "memoire" d'Airflow.

4. **Webserver** : l'UI sur http://localhost:8080. Permet de voir les DAGs, declencher
   des runs, consulter les logs de chaque tache, activer/desactiver (pause/unpause) un
   DAG.

5. **DAG File Processor** : scanne periodiquement le dossier `dags/` pour detecter de
   nouveaux DAGs ou des erreurs de syntaxe/import (visible via
   `airflow dags list-import-errors`).

## XCom (cross-communication)

Mecanisme pour passer une petite valeur d'une tache a une autre via la metadata DB :
```python
context["ti"].xcom_push(key="f1", value=metrics["f1"])   # dans task_train
...
f1 = context["ti"].xcom_pull(task_ids="train", key="f1")  # dans task_check_quality
```
C'est ainsi que `check_quality` recupere le score f1 calcule par `train`.

## Deroulement concret pour `churn_retraining`

1. Le scheduler voit que c'est lundi 3h (ou tu declenches manuellement via
   `airflow dags trigger churn_retraining`) -> il cree un DAG run (identifie par
   `manual__...` ou `scheduled__...`).
2. Il instancie les 3 task instances : `generate_data`, `train`, `check_quality`,
   toutes a `None`/`queued`.
3. `generate_data` n'a pas de dependance amont -> l'executor la lance : elle appelle
   `scripts.generate_data.main()`, regenere `data/churn.csv`. Etat -> `success`.
4. Le scheduler voit que `generate_data` est `success` -> `train` devient eligible ->
   executee : appelle `churn.train.train()`, ecrit `models/model.joblib`, pousse `f1`
   dans XCom. Etat -> `success`.
5. Idem pour `check_quality` : recupere `f1` via XCom, leve une exception si
   `f1 < 0.55` (le DAG run passerait alors en `failed`, avec retry automatique selon
   `default_args` : 1 retry, attente 2 min).
6. Quand les 3 taches sont `success`, le DAG run global passe a `success`.

## Specificite de notre setup (standalone)

`airflow standalone` (commande du service `airflow` dans `docker-compose.yml`) lance
dans un seul conteneur : webserver + scheduler + triggerer + initialisation de la base
SQLite + creation d'un utilisateur admin, le tout en `SequentialExecutor`. C'est le
mode le plus simple pour decouvrir Airflow localement, pas pour la production (pas de
parallelisme, SQLite ne supporte qu'un seul writer).

Le code du projet (`src/churn`, `scripts/`) est copie dans l'image
(`docker/Dockerfile.airflow`) et accessible via
`PYTHONPATH=/opt/airflow/project/src:/opt/airflow/project`, ce qui permet aux imports
`from churn.train import train` et `from scripts.generate_data import main` de
fonctionner dans les fonctions Python des taches.

## Pour aller plus loin

- Inspecter l'etat d'un run en CLI :
  `docker compose --profile airflow exec airflow airflow tasks states-for-dag-run churn_retraining <run_id>`
- Lister les erreurs d'import des DAGs :
  `docker compose --profile airflow exec airflow airflow dags list-import-errors`
- Mettre `QUALITY_THRESHOLD` a `0.99` (TP_S17, etape 6) et observer le run passer en
  `failed` sur `check_quality` uniquement.
