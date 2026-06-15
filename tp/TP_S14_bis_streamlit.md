# TP - Seance 14 bis : Frontend de test avec Streamlit

**Duree : 1 h · Pre-requis : API FastAPI fonctionnelle (TP S12) et un modele entraine**

## Objectifs
- Construire un frontend Streamlit minimal qui appelle l'API `/predict`.
- Adapter le formulaire de saisie aux variables de VOTRE dataset.
- Afficher proprement le resultat de la prediction et gerer les erreurs d'appel.

## Contexte
Une API REST n'est pas pratique a tester a la main. Streamlit permet de construire en
quelques lignes une interface web qui appelle l'API et affiche le resultat. Le squelette
a completer est `todo/frontend/app.py` (marqueurs `TODO S14bis-n`). Il lit l'URL de
l'API depuis la variable d'environnement `API_URL` (utile en docker compose, ou l'API
est joignable via le nom de service `api`).

## Etape 0 - Lancer le frontend
```bash
PYTHONPATH=todo uvicorn mlproject.api:app --reload   # terminal 1 : l'API
PYTHONPATH=todo streamlit run todo/frontend/app.py    # terminal 2 : le frontend
```
Ouvrez http://localhost:8501. Le squelette affiche deja un champ "URL de l'API" et deux
onglets ("Prediction", "Historique").

## Etape 1 - Adapter le formulaire a votre dataset (`TODO S14bis-1`)
Dans l'onglet "Prediction", remplacez les deux champs d'exemple par les colonnes de
VOTRE dataset (memes noms que le schema `Features` de l'API, cf TP S12) :
- `st.number_input(...)` pour chaque variable numerique (`NUMERIC_FEATURES`) ;
- `st.selectbox(...)` pour chaque variable categorielle (`CATEGORICAL_FEATURES`).

## Etape 2 - Construire le payload (`TODO S14bis-2`)
Construisez le dictionnaire `payload` avec les **memes cles** que le schema `Features`
de l'API (un couple `{nom_colonne: valeur}` par variable saisie). C'est ce dict qui est
envoye en JSON a `POST {api_url}/predict`.

## Etape 3 - Afficher le resultat (`TODO S14bis-3`)
En cas de succes, la reponse contient `prediction` (0/1) et `probability`. Affichez-les
lisiblement, par exemple :
```python
col1, col2 = st.columns(2)
col1.metric("Classe predite", result["prediction"])
col2.metric("Probabilite", f"{result['probability']:.1%}")
st.progress(result["probability"])
```
Notez que l'appel est deja encadre par un `try/except httpx.HTTPError` : si l'API est
injoignable, un message d'erreur s'affiche au lieu d'un plantage.

## Etape 4 - Tester
Saisissez un profil "classe 1 probable" puis un profil "classe 0 probable" (selon votre
metier) et comparez les probabilites. Arretez l'API (terminal 1) et refaites une
prediction : verifiez que le message d'erreur s'affiche proprement.

## Etape 5 - Historique des previsions (`TODO S14bis-4 bonus`)
L'onglet "Historique" affiche un message par defaut. Si vous ajoutez a votre API un
endpoint `GET /predictions` (journal des previsions en base), recuperez-le ici :
```python
rows = httpx.get(f"{api_url}/predictions", timeout=10.0).json()
st.dataframe(pd.DataFrame(rows), width="stretch")
```

## Criteres de reussite
- [ ] Le formulaire reprend les variables de votre dataset (memes noms que `Features`).
- [ ] Une prediction via le formulaire affiche `prediction` et `probability` de facon
      lisible.
- [ ] Couper l'API fait apparaitre un message d'erreur (pas une exception brute).
- [ ] Le frontend fonctionne avec `API_URL=http://api:8000` quand il tourne dans docker
      compose (TP S14).

## Pour aller plus loin
- `@st.cache_data(ttl=30)` autour d'un appel `GET /predictions` pour limiter les
  requetes repetees.
- Ajoutez un onglet "Suivi du modele" qui interroge le Model Registry MLflow
  (`mlflow.MlflowClient().search_model_versions(...)`) pour lister les versions de
  `MODEL_NAME`.
- Conteneurisez le frontend (`todo/docker/Dockerfile.frontend`) et integrez-le a la
  stack docker compose (TP S14).
