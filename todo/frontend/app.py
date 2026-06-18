"""Frontend Streamlit : detection de fraude bancaire.

Projet realise par Hajar Elkadouri - ESGI / IABD.
"""
from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://127.0.0.1:5000")

st.set_page_config(page_title="Detection de fraude", page_icon="🛡️", layout="wide")

with st.sidebar:
    st.title("🛡️ Detection de fraude")
    st.markdown("**Projet realise par**")
    st.markdown("### Hajar Elkadouri")
    st.caption("ESGI / IABD - Projet MLOps")
    st.divider()

    st.markdown("**API d'inference**")
    api_url = st.text_input("URL de l'API", value=API_URL)
    st.link_button("Ouvrir la documentation de l'API", f"{api_url}/docs")
    if st.button("Tester la connexion a l'API"):
        try:
            r = httpx.get(f"{api_url}/health", timeout=5.0)
            r.raise_for_status()
            st.success("API joignable")
        except httpx.HTTPError:
            st.error("API injoignable")

    st.divider()
    st.markdown("**Suivi des experiences (MLflow)**")
    mlflow_url = st.text_input("URL de MLflow", value=MLFLOW_URL)
    st.link_button("Ouvrir MLflow", mlflow_url)
    st.caption("Renseignez l'URL de votre serveur MLflow (local ou distant).")

    st.divider()
    st.caption(
        "Modele : Random Forest (roc_auc 0.972)\n\n"
        "Dataset : Credit Card Fraud (ULB)"
    )

st.title("Demonstrateur de detection de fraude bancaire")
st.caption("Saisissez une transaction pour estimer le risque de fraude, ou explorez le modele et les donnees.")

predict_tab, model_tab, data_tab = st.tabs(["Prediction", "Modele", "Dataset"])

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
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.metric("Probabilite de fraude", f"{probability:.2%}")
            with col_b:
                st.write("Niveau de risque")
                st.progress(min(max(probability, 0.0), 1.0))

with model_tab:
    st.subheader("Comparaison des modeles (seance AutoML)")
    st.caption("Trois familles optimisees par recherche d'hyperparametres, evaluees en ROC AUC.")
    scores = pd.DataFrame(
        {"ROC AUC": [0.972, 0.961, 0.958]},
        index=["Random Forest", "XGBoost", "LightGBM"],
    )
    st.bar_chart(scores)
    st.dataframe(scores, use_container_width=True)
    st.info("Modele retenu : Random Forest (meilleur ROC AUC = 0.972). C'est lui qui sert les predictions.")

with data_tab:
    st.subheader("Repartition du jeu de donnees")
    st.caption("Dataset Credit Card Fraud (ULB) : 284 807 transactions, fortement desequilibre.")
    repartition = pd.DataFrame(
        {"Nombre de transactions": [284315, 492]},
        index=["Legitimes", "Fraudes"],
    )
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(repartition)
    with col2:
        st.metric("Total transactions", "284 807")
        st.metric("Fraudes", "492")
        st.metric("Taux de fraude", "0.17 %")
    st.warning(
        "Le tres fort desequilibre (0.17 % de fraudes) explique l'usage du ROC AUC "
        "comme metrique principale plutot que la simple precision."
    )
