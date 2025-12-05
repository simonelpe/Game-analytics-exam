from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Games Analysis", layout="wide")

DATA_DIR = Path("vgsales_clean.csv")

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





# ---------------- MAIN APP ---------------- #

df = load_data()

st.title("VideoGays")





# ---- FILTRI SIDEBAR ---- #
st.sidebar.header("Filtri base")

year_min = int(df["Year_of_Release"].min())
year_max = int(df["Year_of_Release"].max())
platform_list = sorted(df["Platform"].unique())
genre_list = sorted(df["Genre"].dropna().unique())
ranked_by = ['EU_Sales','NA_Sales','JP_Sales','Other_Sales','Global_Sales','Critic_Score','User_Score']

year_start, year_end = st.sidebar.slider(
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

selected_genre = st.sidebar.multiselect(
    "Generi (opzionale)",
    genre_list,
    default=[],
)

ranked_by = st.sidebar.selectbox('Ordianto per', ranked_by)


df = apply_filters(
    df=df,
    year_start=year_start,
    year_end=year_end,
    platform=selected_platform, 
    genre=selected_genre,
    ranked_by=ranked_by
)

st.sidebar.markdown("-----")
st.sidebar.write(f"**Righe filtrate:** {len(df)}")
st.sidebar.write(f"**COnsole distinte:** {df['Platform'].nunique()}")
st.sidebar.write(f"**Generi distinti:** {df['Genre'].nunique()}")





# ---- FILTRI SIDEBAR ---- #
first_bar = px.bar(
    df.head(10),
    x="Name",
    y=ranked_by,
    title="Top 10 piloti per numero di vittorie",
)

col_a, col_b = st.columns(2)
col_a.plotly_chart(first_bar, use_container_width=True)