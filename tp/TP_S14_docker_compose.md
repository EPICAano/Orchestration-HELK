# TP - Seance 14 : Orchestrer la stack avec Docker Compose

**Duree : 1 h · Pre-requis : images d'entrainement et d'API construites (TP S8/S12),
un modele entraine dans le volume `models_data`**

## Objectifs
- Completer un `docker-compose.yml` qui orchestre quatre services : MLflow,
  entrainement, API et frontend.
- Distinguer les services actifs par defaut des services optionnels (profil `train`).
- Comprendre le reseau interne de docker compose (resolution par nom de service),
  `depends_on`/`healthcheck` et les volumes nommes.

## Contexte
Jusqu'ici (TP S8), chaque conteneur etait lance isolement avec `docker run`. Une vraie
stack fait collaborer plusieurs services : MLflow (tracking + registry), un service
d'entrainement, l'API FastAPI (charge le modele et l'expose) et le frontend Streamlit
(consomme l'API). Le squelette a completer est `todo/docker-compose.yml` (marqueurs
`TODO S14-n`) ; il s'appuie sur les images `todo/docker/Dockerfile.{train,api,frontend}`.

## Etape 1 - Lire les services fournis
Ouvrez `todo/docker-compose.yml`. Les services `mlflow` et `train` sont fournis. Repondez :
- Quel port `mlflow` expose-t-il ? Ou sont stockees ses donnees (volume) ?
- Pourquoi `train` porte-t-il `profiles: [train]` ? Que se passe-t-il si on lance
  `docker compose up -d` sans option ?
- Comment `train` ecrit-il le modele la ou l'API pourra le lire (volume partage) ?

## Etape 2 - Completer le service `api` (`TODO S14-1, S14-2, S14-3`)
Sous le service `api` :
- `TODO S14-1` : exposez le port `8000` (`ports: ["8000:8000"]`) ;
- `TODO S14-2` : montez le volume `models_data` en lecture seule sur `/app/models`
  (`models_data:/app/models:ro`) pour lire le modele produit par `train` ;
- `TODO S14-3` : ajoutez un `healthcheck` qui interroge `GET /health` (voir l'exemple
  en commentaire).

## Etape 3 - Completer le service `frontend` (`TODO S14-4, S14-5, S14-6`)
Sous le service `frontend` :
- `TODO S14-4` : definissez `API_URL: http://api:8000` dans `environment`. Le nom `api`
  est resolu par le DNS interne de docker compose (voir Etape 5) ;
- `TODO S14-5` : exposez le port `8501` ;
- `TODO S14-6` : attendez que l'API soit saine avant de demarrer
  (`depends_on: api: condition: service_healthy`).

## Etape 4 - Lancer la stack
```bash
# 1. suivi MLflow
docker compose -f todo/docker-compose.yml up -d mlflow
# 2. entrainement one-shot (profil train) -> alimente models_data
docker compose -f todo/docker-compose.yml --profile train run --rm train
# 3. API + frontend
docker compose -f todo/docker-compose.yml up -d api frontend

docker compose -f todo/docker-compose.yml ps     # etat (healthy, running...)
docker compose -f todo/docker-compose.yml logs -f api
```
Ouvrez http://localhost:8501 (frontend) et http://localhost:5000 (MLflow). Faites une
prediction via le frontend.

Repondez : pourquoi `train` utilise-t-il `run --rm` (conteneur jetable) alors que `api`
et `frontend` utilisent `up -d` (services durables) ?

## Etape 5 - Reseau interne : resoudre les services par leur nom
Dans le reseau cree par docker compose, chaque service est joignable par son **nom de
service** (`api`, `mlflow`) depuis les autres conteneurs : c'est pourquoi le frontend
vise `http://api:8000`.
```bash
# Depuis l'interieur du reseau : resout vers l'IP du conteneur api
docker compose -f todo/docker-compose.yml exec frontend \
  python -c "import socket; print(socket.gethostbyname('api'))"

# Depuis votre machine : echoue (le nom 'api' n'existe pas hors du reseau compose)
python -c "import socket; print(socket.gethostbyname('api'))"
```

## Etape 6 - `healthcheck`, `depends_on` et volumes
- A quoi sert le `healthcheck` de l'API ? Que se passerait-il si `frontend` demarrait
  avant que l'API ne reponde sur `/health` (sans la condition `service_healthy`) ?
- Les volumes `mlflow_data` et `models_data` sont declares en bas du fichier. Apres un
  `docker compose down` (sans `-v`) puis un nouveau `up`, les experiences MLflow et le
  modele sont-ils conserves ? Et avec `docker compose down -v` ?
```bash
docker compose -f todo/docker-compose.yml down     # conserve les volumes
docker compose -f todo/docker-compose.yml down -v   # supprime les volumes
```

## Criteres de reussite
- [ ] `docker compose -f todo/docker-compose.yml config -q` ne renvoie aucune erreur.
- [ ] La stack (`mlflow`, `api`, `frontend`) demarre et chaque service repond sur son
      port ; `train` n'est lance qu'avec `--profile train`.
- [ ] Le frontend joint l'API via `http://api:8000` (DNS interne) et une prediction
      aboutit.
- [ ] Vous savez expliquer ce que conservent/suppriment `down` et `down -v`.

## Pour aller plus loin
- `docker compose config` : affiche la configuration resolue (profils, variables
  substituees).
- Ajoutez un service de base de donnees (ex. `mysql` ou `postgres`) avec son propre
  `healthcheck`, et faites-en dependre l'API si vous ajoutez un journal des previsions.
- Comparez la taille des images (`docker images`) entre `Dockerfile.train`,
  `Dockerfile.api` et `Dockerfile.frontend` : laquelle est la plus lourde, et pourquoi ?
