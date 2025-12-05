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

dx, sx = st.columns(2)

fig = px.scatter(
    df,
    x='Critic_Score',
    y='Global_Sales',
    color_discrete_sequence=['blue'],
    labels={'Critic_Score':'Critic Score', 'Global_Sales':'Global Sales (milioni)'},
    title='Impatto delle Recensioni sul Successo Commerciale',
    hover_data=['Name','Platform']
)

# Aggiungiamo anche User_Score nello stesso grafico
fig2 = px.scatter(
    df,
    x='User_Score',
    y='Global_Sales',
    color_discrete_sequence=['orange'],
    hover_data=['Name','Platform']
)

dx, sx = st.columns(2)
with dx:
    st.plotly_chart(fig, use_container_width=True)
with sx:
    st.plotly_chart(fig2, use_container_width=True)

threshold = df['Global_Sales'].quantile(0.95)
df_filtered = df[df['Global_Sales'] <= threshold]

# Box plot interattivo senza outlier estremi
fig_rating = px.box(
    df_filtered,
    x='Rating',
    y='Global_Sales',
    points='all',  # mostra anche tutti i punti
    color='Rating',
    title='Distribuzione delle Vendite Globali per Rating (outlier rimossi)',
    labels={'Global_Sales':'Global Sales (milioni)', 'Rating':'Rating'}
)

# Mostra il grafico su Streamlit
st.plotly_chart(fig_rating, use_container_width=True)