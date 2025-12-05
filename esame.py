import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Configurazione della pagina
st.set_page_config(page_title="Dashboard KPI Videogiochi", layout="wide")
st.title("🎮 Dashboard Analisi Videogiochi")

DATA_DIR = Path("data.csv")

# Colonne feature
FEATURE_COLS = ["Platform", "Year_of_Release", "Genre", "Publisher" ,"Critic_Score", "Critic_Count", "User_Score", "User_Count", "Developer", "Rating"]

# Etichette feature
FEATURE_LABELS = {
    "Platform": "Piattaforma",
    "Year_of_Release": "Anno di rilascio",
    "Genre": "Genere",
    "Publisher": "Distributore",
    "Critic_Score": "Valutazione critica",
    "Critic_Count": "Numero di recensioni critica",
    "User_Score": "Valutazione utenti",
    "User_Count": "Numero di recensioni utenti",
    "Developer": "Sviluppatore",
    "Rating": "Classificazione",
}

# Etichetta target
TARGET_LABEL = "Numero di copie vendute globalmente"

# ----LOAD DATA----
@st.cache_data
def load_data() -> pd.DataFrame:
    game_sales = pd.read_csv(DATA_DIR)
    return game_sales

# ----TRAIN MODEL----
@st.cache_resource
def train_model(df: pd.DataFrame,  max_depth, min_samples_leaf):

    bins = [-float('inf'), 0.1, 1, float('inf')]
    labels = ["Flop", "Mid", "Hit"]

    df["Sales_Class"] = pd.cut(df["Global_Sales"], bins=bins, labels=labels)


    X = df[FEATURE_COLS]
    y = df["Sales_Class"]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)


    metrics = {
        "acc_train": acc_train,
        "acc_test": acc_test,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    return model, metrics


# ---------------- MAIN APP ---------------- 
df = load_data()
if not df.empty:

    # SIDEBAR -------------------------------------------------
    st.sidebar.header("Filtri base")
    max_depth = st.sidebar.slider("max_depth", min_value=1, max_value=10, value=5)
    min_samples_leaf = st.sidebar.slider("min_samples_leaf", min_value=1, max_value=10, value=5)


    model, metrics = train_model(df,  max_depth, min_samples_leaf)

    # ---------------------------------------------------------
    # SEZIONE KPI (WIDGET)
    # ---------------------------------------------------------
    st.subheader("Indicatori Chiave")
    
    # Calcoli
    n_giochi = len(df)
    tot_vendite = df['Global_Sales'].sum()
    pct_over_1m = (len(df[df['Global_Sales'] > 1]) / n_giochi) * 100

    # Colonne per i widget
    col1, col2, col3 = st.columns(3)
    col1.metric("Numero Giochi", f"{n_giochi}")
    col2.metric("Vendite Globali Totali", f"{tot_vendite:.2f} M")
    col3.metric("% Giochi > 1MLN", f"{pct_over_1m:.1f}%")

    st.markdown("---")

    # ---------------------------------------------------------
    # DROPDOWN (EXPANDER) PER HEAD DATASET
    # ---------------------------------------------------------
    with st.expander("📂 Clicca per vedere l'head del dataset (data.csv)"):
        st.dataframe(df.head())

    st.markdown("---")

    # ---------------------------------------------------------
    # TAB (PAGINE NELLA STESSA SCHERMATA)
    # ---------------------------------------------------------
    # Creazione delle tab
    tab_grafici, tab_modello = st.tabs(["📊 Pagina 1: Grafici", "🤖 Pagina 2: Modello"])

    # Contenuto Tab 1
    with tab_grafici:
        st.header("Sezione Grafici")
        # Placeholder grafico
        st.info("Qui verranno visualizzati i grafici (Placeholder)")
        
        # Esempio: un grafico vuoto o semplice per mostrare dove andrebbe
        chart_placeholder = st.empty()
        # Se volessi mettere un grafico vero in futuro:
        # st.bar_chart(df['Global_Sales'].head(10))

    # Contenuto Tab 2
    with tab_modello:
        st.subheader("Metriche del modello")

        col1, col2 = st.columns(2)
        col1.metric("Accuracy su train", f"{metrics['acc_train']:.2%}")
        col2.metric("Accuracy su test", f"{metrics['acc_test']:.2%}")

else:
    st.write("Impossibile caricare i dati.")