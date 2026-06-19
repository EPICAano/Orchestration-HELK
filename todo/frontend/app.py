"""Frontend Streamlit : detection de fraude bancaire.

Projet realise par Hajar Elkadouri - ESGI / IABD.
"""
from __future__ import annotations

import os

import httpx
import plotly.graph_objects as go
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://127.0.0.1:5000")

ACCENT = "#00D4B8"
GREEN = "#2ECC71"
RED = "#E74C3C"
ORANGE = "#F39C12"
GRID = "#2A2F3B"

st.set_page_config(page_title="Detection de fraude", page_icon="🛡️", layout="wide")


def style_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


with st.sidebar:
    st.title("🛡️ Detection de fraude")
    st.markdown("**Projet realise par**")
    st.markdown(f"<h3 style='color:{ACCENT};margin-top:-8px'>Hajar Elkadouri</h3>", unsafe_allow_html=True)
    st.caption("ESGI / IABD - Projet MLOps")
    st.divider()

    st.markdown("**API d'inference**")
    api_url = st.text_input("URL de l'API", value=API_URL)
    st.link_button("📖 Documentation de l'API", f"{api_url}/docs", use_container_width=True)
    if st.button("🔌 Tester la connexion", use_container_width=True):
        try:
            r = httpx.get(f"{api_url}/health", timeout=5.0)
            r.raise_for_status()
            st.success("API joignable")
        except httpx.HTTPError:
            st.error("API injoignable")

    st.divider()
    st.markdown("**Suivi des experiences (MLflow)**")
    mlflow_url = st.text_input("URL de MLflow", value=MLFLOW_URL)
    st.link_button("📊 Ouvrir MLflow", mlflow_url, use_container_width=True)

    st.divider()
    st.caption("Modele : Random Forest (ROC AUC 0.972)\n\nDataset : Credit Card Fraud (ULB)")

st.title("Demonstrateur de detection de fraude bancaire")
st.caption("Estimez le risque de fraude d'une transaction, ou explorez le modele et les donnees.")

predict_tab, model_tab, data_tab = st.tabs(["🔎 Prediction", "🤖 Modele", "📦 Dataset"])

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

        submitted = st.form_submit_button("Predire", use_container_width=True)

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
            res_col, gauge_col = st.columns([1, 1])
            with res_col:
                if prediction == 1:
                    st.error("### Transaction predite : FRAUDULEUSE")
                else:
                    st.success("### Transaction predite : LEGITIME")
                st.metric("Probabilite de fraude", f"{probability:.2%}")
            with gauge_col:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": " %", "font": {"color": "#FAFAFA"}},
                    title={"text": "Niveau de risque"},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#FAFAFA"},
                        "bar": {"color": ACCENT},
                        "steps": [
                            {"range": [0, 40], "color": GREEN},
                            {"range": [40, 70], "color": ORANGE},
                            {"range": [70, 100], "color": RED},
                        ],
                    },
                ))
                st.plotly_chart(style_fig(gauge), use_container_width=True)

with model_tab:
    st.subheader("Comparaison des modeles (seance AutoML)")
    st.caption("Trois familles optimisees par recherche d'hyperparametres, evaluees en ROC AUC.")
    models = ["Random Forest", "XGBoost", "LightGBM"]
    roc = [0.972, 0.961, 0.958]
    colors = [ACCENT, "#5B8FF9", "#9B59B6"]
    bar = go.Figure(go.Bar(
        x=models, y=roc, marker_color=colors,
        text=[f"{v:.3f}" for v in roc], textposition="outside",
    ))
    bar.update_yaxes(range=[0.9, 1.0], title="ROC AUC")
    st.plotly_chart(style_fig(bar), use_container_width=True)
    st.info("Modele retenu : Random Forest (meilleur ROC AUC = 0.972). C'est lui qui sert les predictions.")

with data_tab:
    st.subheader("Repartition du jeu de donnees")
    st.caption("Dataset Credit Card Fraud (ULB) : 284 807 transactions, fortement desequilibre.")
    pie_col, stat_col = st.columns([1, 1])
    with pie_col:
        pie = go.Figure(go.Pie(
            labels=["Legitimes", "Fraudes"],
            values=[284315, 492],
            hole=0.55,
            marker_colors=[GREEN, RED],
        ))
        pie.update_traces(textinfo="percent")
        st.plotly_chart(style_fig(pie), use_container_width=True)
    with stat_col:
        st.metric("Total transactions", "284 807")
        st.metric("Fraudes", "492")
        st.metric("Taux de fraude", "0.17 %")
    st.warning(
        "Le tres fort desequilibre (0.17 % de fraudes) explique l'usage du ROC AUC "
        "comme metrique principale plutot que la simple precision."
    )
