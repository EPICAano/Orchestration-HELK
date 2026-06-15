# TP — Séance 8 : Conteneuriser l'entraînement avec Docker

**Durée : 1 h · Pré-requis : Docker installé (`docker --version`)**

## Objectifs
- Écrire un `Dockerfile` from scratch.
- Comprendre le cache de layers et l'optimiser.
- Construire une image et exécuter l'entraînement dans un conteneur.

## Contexte
On veut que n'importe quel collègue puisse lancer l'entraînement **sans installer Python ni
les dépendances** sur sa machine : « ça marche chez moi » ne suffit pas. On empaquette donc
tout dans une image. Le squelette à compléter (marqueurs `TODO S8-n`) est dans
`todo/docker/Dockerfile.train`.

## Étape 1 — Image de base & dossier de travail (`TODO S8-1, S8-2`)
```dockerfile
FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app
```
> Pourquoi `-slim` ? Image plus petite (~150 Mo vs ~1 Go) et surface d'attaque réduite.

## Étape 2 — Dépendances en exploitant le cache (`TODO S8-3`)
Copiez d'abord **les métadonnées seules**, installez, puis copiez le code. Ainsi, tant que
`pyproject.toml` ne change pas, Docker réutilise le layer d'installation (build rapide).
```dockerfile
COPY pyproject.toml ./
RUN pip install --no-cache-dir .
```

## Étape 3 — Code source (`TODO S8-4`)
Copiez votre package (et, le cas échéant, votre script de préparation de données) :
```dockerfile
COPY todo/mlproject/ ./mlproject/
```

## Étape 4 — Données + commande par défaut (`TODO S8-5`)
Rendez vos données disponibles dans l'image (copie du CSV depuis `data/`, ou exécution de
votre script de préparation), puis définissez la commande par défaut :
```dockerfile
COPY data/ ./data/
CMD ["python", "-m", "mlproject.train"]
```

## Étape 5 — Build & run
```bash
docker build -f todo/docker/Dockerfile.train -t mlproject-train .
docker run --rm mlproject-train
# récupérer le modèle produit sur l'hôte via un volume :
docker run --rm -v "$PWD/models:/app/models" mlproject-train
ls models/   # model.joblib doit apparaître
```

## Étape 6 — Observer le cache de layers
Relancez le build : il doit être quasi instantané. Modifiez une ligne de `train.py`,
rebuild : seuls les derniers layers se reconstruisent (l'install des deps reste en cache).
Inversez l'ordre des `COPY` (code avant deps) et constatez la perte du cache.

## Étape 7 — Passer un paramètre & une variable d'environnement
```bash
docker run --rm mlproject-train python -m mlproject.train --c 0.1
docker run --rm -e MLFLOW_EXPERIMENT=classification-docker mlproject-train
```

## Critères de réussite
- [ ] `docker build` réussit et `docker run` affiche `f1=... roc_auc=...`.
- [ ] Le `model.joblib` est récupéré sur l'hôte via le volume.
- [ ] Le second build profite du cache (install des dépendances non rejouée).

## Pour aller plus loin (préparation S9)
Transformez le Dockerfile en **multi-stage build** (un stage `builder` pour installer,
un stage `runtime` minimal) et comparez la taille finale avec `docker images`.
