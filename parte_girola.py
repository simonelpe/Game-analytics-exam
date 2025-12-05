from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


st.set_page_config(page_title="Games Analysis", layout="wide")

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

def apply_filters(
    df: pd.DataFrame,
    year_start: int,
    year_end: int,
    platform: str, 
    genre: str,
    ranked_by: str,
) -> pd.DataFrame:
    
    df_filtered = df.copy()

    if platform:
        df_filtered = df_filtered[df_filtered["Platform"].isin(platform)]

    if genre:
        df_filtered = df_filtered[df_filtered["Genre"].isin(genre)]

    df_filtered = df_filtered[df_filtered["Year_of_Release"].between(year_start, year_end)]

    df_filtered.sort_values(ranked_by, ascending=False, inplace=True)

    return df_filtered

@st.cache_resource
def train_model(df: pd.DataFrame,  max_depth, min_samples_leaf):
    """
    Allena un RandomForest e calcola:
    - accuracy su train e test
    - baseline (classe più frequente)
    - importanza delle feature
    """
    df["Sales_Class"] = pd.qcut(df["Global_Sales"], q=3, labels=["Flop", "Mid", "Hit"])


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
# ---------------- MAIN APP ---------------- #

# ---- FILTRI SIDEBAR ---- #
st.sidebar.header("Filtri base")
max_depth = st.sidebar.slider("max_depth", min_value=1, max_value=10, value=5)
min_samples_leaf = st.sidebar.slider("min_samples_leaf", min_value=1, max_value=10, value=5)

df = load_data()
model, metrics = train_model(df,  max_depth, min_samples_leaf)


#---- METRICHE ---- #
st.subheader("Metriche del modello")

col1, col2 = st.columns(2)
col1.metric("Accuracy su train", f"{metrics['acc_train']:.2%}")
col2.metric("Accuracy su test", f"{metrics['acc_test']:.2%}")



"""
Il management vuole usare questi dati per:

Capire su quali piattaforme e generi ha funzionato meglio il catalogo nel tempo.
Valutare quanto contano recensioni e rating sul successo commerciale di un gioco.
Avere un prototipo di modello ML che aiuti a stimare la probabilità che un nuovo gioco diventi un “HIT” (es. ≥ 1M copie).
(Nice to have) Un modo rapido per interrogare i dati in linguaggio naturale.
"""

