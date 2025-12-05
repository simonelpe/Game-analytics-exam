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

RANKED_BY_LIST   = ['EU_Sales','NA_Sales','JP_Sales','Other_Sales','Global_Sales','Critic_Score','User_Score']

# Etichette feature
FEATURE_LABELS = {
    "Platform": "Piattaforma",
    "Year_of_Release": "Anno di rilascio",
    "Genre": "Genere",
    "Publisher": "Distributore",
    'EU_Sales': 'Vendite in Europa',
    'NA_Sales': 'Vendite in Nord America',
    'JP_Sales': 'Vendite in Giappone',
    'Other_Sales': 'Vendite nel resto del mondo',
    'Global_Sales': 'Vendite globali',
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



# ----UTILITY FUNSCTIONS----
def apply_filters(
    df: pd.DataFrame,
    year_start: int,
    year_end: int,
    platform: str = None, 
    genre: str = None,
    ranked_by: str = 'Global_Sales',
) -> pd.DataFrame:
    
    df_filtered = df.copy()

    if platform:
        df_filtered = df_filtered[df_filtered["Platform"].isin(platform)]

    if genre:
        df_filtered = df_filtered[df_filtered["Genre"].isin(genre)]

    df_filtered = df_filtered[df_filtered["Year_of_Release"].between(year_start, year_end)]

    df_filtered.sort_values(ranked_by, ascending=False, inplace=True)

    return df_filtered



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

    # ---------------- SIDEBAR ----------------
    with st.sidebar:

        print(df["Year_of_Release"].dtype)  # tipo della colonna
        print(df["Year_of_Release"].unique())
        st.header("Filtri grafici")

        st.header("Filtri base")

        year_min = int(df["Year_of_Release"].min())
        year_max = int(df["Year_of_Release"].max())
        platform_list = sorted(df["Platform"].unique())
        genre_list = sorted(df["Genre"].dropna().unique())

        year_start, year_end = st.slider(
            "Intervallo di anni",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max),
            step=1,
        )

        selected_platform = st.sidebar.multiselect(
            "COnsoles (opzionale)",
            platform_list,
            default=[],
        )

        selected_genre = st.multiselect(
            "Generi (opzionale)",
            genre_list,
            default=[],
        )

        ranked_by = st.selectbox('Ordianto per', RANKED_BY_LIST)

        df_filtered = apply_filters(
            df=df,
            year_start=year_start,
            year_end=year_end,
            platform=selected_platform, 
            genre=selected_genre,
            ranked_by=ranked_by
        )

        with st.expander("Filtri modello"):
            max_depth = st.slider("max_depth", min_value=1, max_value=10, value=5)
            min_samples_leaf = st.slider("min_samples_leaf", min_value=1, max_value=10, value=5)
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

        df_platform = apply_filters(
            df=df,
            year_start=year_start,
            year_end=year_end,
            platform=selected_platform,
            genre=selected_genre,
        ).groupby(['Year_of_Release', 'Platform'])[ranked_by].sum().reset_index()

        df_genre = apply_filters(
            df=df,
            year_start=year_start,
            year_end=year_end,
            platform=selected_platform,
            genre=selected_genre,
        ).groupby(['Year_of_Release', 'Genre'])[ranked_by].sum().reset_index()
        
        first_bar = px.bar(
            df_filtered.head(10),
            x="Name",
            y=ranked_by,
            title="Top 10 Giochi secondo i filtri per"
        )
        st.plotly_chart(first_bar, height=500, use_container_width=True)

        st.subheader("Trend di vendite per console")
        col_a, col_b = st.columns(2)

        platform_line = px.line(
            df_platform, 
            x='Year_of_Release', 
            y=ranked_by,
            color='Platform',
            title="Singoli")
        platform_line.update_layout(xaxis_title="Anno di rilascio", yaxis_title=FEATURE_LABELS[ranked_by])
        col_a.plotly_chart(platform_line, use_container_width=True)

        platform_area = px.area(
            df_platform, 
            x='Year_of_Release', 
            y=ranked_by,
            color='Platform',
            title="Impilati")
        platform_area.update_layout(xaxis_title="Anno di rilascio", yaxis_title=FEATURE_LABELS[ranked_by])
        col_b.plotly_chart(platform_area, use_container_width=True)



        st.subheader("Trend di vendite per generi")
        col_c, col_d = st.columns(2)

        genre_line = px.line(
            df_genre, 
            x='Year_of_Release', 
            y=ranked_by,
            color='Genre',
            title="Singoli")
        genre_line.update_layout(xaxis_title="Anno di rilascio", yaxis_title=FEATURE_LABELS[ranked_by])
        col_c.plotly_chart(genre_line, use_container_width=True)

        genre_area = px.area(
            df_genre, 
            x='Year_of_Release', 
            y=ranked_by,
            color='Genre',
            title="Impilati")
        genre_area.update_layout(xaxis_title="Anno di rilascio", yaxis_title=FEATURE_LABELS[ranked_by])
        col_d.plotly_chart(genre_area, use_container_width=True)

    # Contenuto Tab 2
    with tab_modello:
        st.subheader("Metriche del modello")

        col1, col2 = st.columns(2)
        col1.metric("Accuracy su train", f"{metrics['acc_train']:.2%}")
        col2.metric("Accuracy su test", f"{metrics['acc_test']:.2%}")

else:
    st.write("Impossibile caricare i dati.")