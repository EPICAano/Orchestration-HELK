"""Frontend Streamlit : tester l'API de classification (fraude CB).

Seance 14 bis - TP Streamlit
    Application qui appelle l'API FastAPI (TP S12) pour predire si une
    transaction est frauduleuse.
    Lancement : `PYTHONPATH=todo streamlit run todo/frontend/app.py`
"""
from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Detection de fraude", layout="wide")
st.title("Demonstrateur de detection de fraude bancaire")

api_url = st.text_input("URL de l'API", value=API_URL)

predict_tab, history_tab = st.tabs(["Prediction", "Historique"])

with predict_tab:
    st.subheader("Tester l'endpoint /predict")

    with st.form("predict_form"):
        # TODO (S14bis-1) : champs adaptes au dataset fraude (Time, Amount, V1..V28)
        col1, col2 = st.columns(2)
        with col1:
            time = st.number_input("Time", min_value=0.0, value=0.0)
        with col2:
            amount = st.number_input("Amount", min_value=0.0, value=149.62)

        st.caption("Composantes PCA V1 a V28 (peuvent etre negatives)")
        v_values = {}
        v_cols = st.columns(4)
        for i in range(1, 29):
            with v_cols[(i - 1) % 4]:
                v_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.4f")

        submitted = st.form_submit_button("Predire")

    if submitted:
        # TODO (S14bis-2) : payload avec les memes cles que le schema Features
        payload = {"Time": time, "Amount": amount, **v_values}
        try:
            response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as exc:
            st.error(f"Appel a l'API impossible : {exc}")
        else:
            # TODO (S14bis-3) : affichage lisible du resultat
            prediction = result["prediction"]
            probability = result["probability"]
            if prediction == 1:
                st.error("Transaction predite : FRAUDULEUSE")
            else:
                st.success("Transaction predite : LEGITIME")
            st.metric("Probabilite de fraude", f"{probability:.2%}")
            st.progress(min(max(probability, 0.0), 1.0))

with history_tab:
    st.subheader("Historique des previsions")
    # TODO (S14bis-4 bonus) : endpoint GET /predictions non implemente
    st.info("Aucun journal de previsions : ajoutez un endpoint /predictions a l'API (bonus).")
    _ = pd
