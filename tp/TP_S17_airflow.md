# TP — Séance 17 : Orchestrer le ré-entraînement avec Airflow

**Durée : 1 h 15 · Pré-requis : Docker, et la baseline du projet fonctionnelle**

## Objectifs
- Démarrer une instance Airflow locale.
- Écrire un DAG enchaînant génération de données → entraînement → contrôle qualité.
- Échanger une valeur entre tâches via XCom et planifier le pipeline.

## Contexte
Lancer l'entraînement à la main n'est pas tenable : on veut un **ré-entraînement planifié et
supervisé**. On modélise le pipeline en DAG Airflow. Le squelette à compléter (marqueurs
`TODO S17-n`) est dans `todo/dags/retrain_dag.py` : un DAG simple à 3 tâches (préparation des
données → entraînement → contrôle qualité). Pour aller plus loin (DAG plus avancé : boucle de
feedback, comparaison de modèles), voir `tp/TP_S17_bis_airflow.md`.

## Étape 1 — Démarrer Airflow (15 min)
Utilisez le `docker-compose` officiel d'Airflow :
```bash
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/stable/docker-compose.yaml'
mkdir -p ./dags ./logs ./plugins
echo "AIRFLOW_UID=$(id -u)" > .env
docker compose up airflow-init
docker compose up -d
```
UI sur http://localhost:8080 (login `airflow` / `airflow`). Copiez `todo/dags/retrain_dag.py`
ainsi que votre package `todo/mlproject/` (et tout script de préparation de données) dans le
volume `dags/` (ou montez le projet) pour que les imports fonctionnent.

## Étape 2 — Préparation des données (`TODO S17-1`)
Appelez le script/fonction qui (re)génère ou rafraîchit votre jeu de données dans `data/` :
```python
def task_prepare_data(**context):
    from votre_module_de_preparation import main as prepare
    prepare()
```

## Étape 3 — Entraînement + XCom (`TODO S17-2`)
```python
def task_train(**context):
    from mlproject.train import train
    metrics = train()
    context["ti"].xcom_push(key="f1", value=metrics["f1"])
```

## Étape 4 — Contrôle qualité (`TODO S17-3`)
La tâche échoue si le modèle est sous le seuil → le pipeline ne « livre » pas un mauvais modèle :
```python
def task_check_quality(**context):
    f1 = context["ti"].xcom_pull(task_ids="train", key="f1")
    if f1 < QUALITY_THRESHOLD:
        raise ValueError(f"f1={f1:.3f} < seuil {QUALITY_THRESHOLD}")
    print(f"Qualité OK : f1={f1:.3f}")
```

## Étape 5 — Planification & dépendances (`TODO S17-4, S17-5`)
- Réglez `schedule` (ex. `"0 3 * * 1"` = tous les lundis à 3 h).
- Déclarez l'ordre en bas du DAG :
```python
prepare >> train_task >> check
```

## Étape 6 — Exécuter & observer
Dans l'UI : activez le DAG `model_retraining`, déclenchez-le (*Trigger DAG*), puis suivez la
**Grid / Graph view**. Inspectez les logs de chaque tâche et la valeur d'XCom de `train`.
Testez le garde-fou en montant temporairement `QUALITY_THRESHOLD` à `0.99`.

## Critères de réussite
- [ ] Le DAG apparaît sans erreur d'import dans l'UI.
- [ ] Les 3 tâches s'exécutent dans l'ordre et passent au vert.
- [ ] `check_quality` lit bien la métrique poussée par `train` (XCom).
- [ ] Un seuil trop haut fait échouer `check_quality` (et lui seul).

## Pour aller plus loin
Ajoutez une tâche d'enregistrement du modèle dans le MLflow Registry après `check_quality`,
et une notification (e-mail / Slack) en cas d'échec via un *callback* `on_failure_callback`.
