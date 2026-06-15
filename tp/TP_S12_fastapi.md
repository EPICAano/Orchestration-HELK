# TP — Séance 12 : Exposer le modèle via une API FastAPI

**Durée : 1 h · Pré-requis : un modèle entraîné (`python -m mlproject.train` → `models/model.joblib`)**

## Objectifs
- Créer une API d'inférence avec FastAPI.
- Valider les entrées et structurer les sorties avec pydantic.
- Charger le modèle au démarrage et servir des prédictions.

## Contexte
Le modèle vit pour l'instant dans un fichier `.joblib` : inutilisable par une application tierce.
On l'expose derrière une API HTTP. Le squelette à compléter se trouve dans
`todo/mlproject/api.py` : `/health` est déjà fourni, complétez les `TODO S12-n`. Adaptez le
schéma d'entrée (`Features`) aux colonnes de VOTRE dataset (`config.py`, cf TP S0).

## Étape 0 — Lancer l'API telle quelle
```bash
PYTHONPATH=todo uvicorn mlproject.api:app --reload
```
Ouvrez http://127.0.0.1:8000/docs : seul `/health` est présent. C'est le point de départ.

## Étape 1 — Schéma d'entrée + exemple de payload (`TODO S12-1`)
Déclarez dans `Features` les **mêmes colonnes que `NUMERIC_FEATURES` +
`CATEGORICAL_FEATURES` de votre `config.py`**, avec leurs types et contraintes, puis
ajoutez un exemple de payload pour enrichir la doc `/docs`. Exemple générique (à
remplacer par VOS variables) :
```python
class Features(BaseModel):
    feature_numerique: float = Field(..., ge=0)
    feature_categorielle: str

    model_config = {
        "json_schema_extra": {
            "examples": [{"feature_numerique": 1.0, "feature_categorielle": "A"}]
        }
    }
```

## Étape 2 — Schéma de sortie (`TODO S12-2`)
```python
class PredictionOut(BaseModel):
    prediction: int
    probability: float
```

## Étape 3 — Charger le modèle au démarrage (`TODO S12-3`)
Utilisez le `lifespan` pour charger une seule fois au boot (pas à chaque requête) :
```python
import joblib
from contextlib import asynccontextmanager
from mlproject.config import MODEL_DIR

ml = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml["model"] = joblib.load(MODEL_DIR / "model.joblib")
    yield
    ml.clear()

app = FastAPI(title="Classification API", version="0.1.0", lifespan=lifespan)
```

## Étape 4 — Endpoint `/predict` (`TODO S12-4`)
```python
import pandas as pd
from fastapi import HTTPException

@app.post("/predict", response_model=PredictionOut)
def predict(features: Features) -> PredictionOut:
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    row = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(row)[0, 1])
    return PredictionOut(prediction=int(proba >= 0.5), probability=round(proba, 4))
```

## Étape 5 — Tester
Via `/docs` (bouton *Try it out*) ou en ligne de commande (adaptez le payload à VOS
colonnes) :
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"feature_numerique": 1.0, "feature_categorielle": "A"}'
```
Testez aussi une entrée invalide (valeur hors contrainte, ex. un champ `ge=0` à `-1`) :
l'API doit renvoyer **422** automatiquement.

## Bonus (`TODO S12-5`)
Ajoutez `GET /model-info` renvoyant la version servie (ex. lue depuis une variable d'env
`MODEL_VERSION`).

## Critères de réussite
- [ ] `/docs` liste `/health`, `/predict` (et `/model-info` en bonus).
- [ ] Une entrée valide renvoie `prediction` et `probability`.
- [ ] Une entrée invalide renvoie automatiquement un code 422.

## Pour aller plus loin (préparation S13)
Remplacez le chargement `joblib` par un chargement depuis le **MLflow Model Registry**
(`mlflow.pyfunc.load_model("models:/classifier/Staging")`).
