"""Frontend Streamlit : detection de fraude bancaire.

Projet realise par Hajar Elkadouri.
"""
from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Detection de fraude", page_icon="🛡️", layout="wide")

with st.sidebar:
    st.title("🛡️ Detection de fraude")
    st.markdown("**Projet realise par**")
    st.markdown("### Hajar Elkadouri")
    st.divider()
    st.markdown(
        "Projet MLOps - ESGI / IABD\n\n"
        "Detection de transactions frauduleuses par carte bancaire "
        "(dataset Credit Card Fraud, modele Random Forest)."
    )
    st.divider()
    api_url = st.text_input("URL de l'API", value=API_URL)
    if st.button("Tester la connexion a l'API"):
        try:
            r = httpx.get(f"{api_url}/health", timeout=5.0)
            r.raise_for_status()
            st.success("API joignable")
        except httpx.HTTPError:
            st.error("API injoignable")

st.title("Demonstrateur de detection de fraude bancaire")
st.caption("Saisissez les caracteristiques d'une transaction pour estimer le risque de fraude.")

predict_tab, history_tab = st.tabs(["Prediction", "Historique"])

with predict_tab:
    st.subheader("Tester l'endpoint /predict")

    with st.form("predict_form"):
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
        payload = {"Time": time, "Amount": amount, **v_values}
        try:
            response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as exc:
            st.error(f"Appel a l'API impossible : {exc}")
        else:
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
    st.info("Aucun journal de previsions : ajoutez un endpoint /predictions a l'API (bonus).")
    _ = pd
