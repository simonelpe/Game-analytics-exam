from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Games Analysis", layout="wide")

DATA_DIR = Path("vgsales_clean.csv")

ranked_by_list = [
    'EU_Sales',
    'NA_Sales',
    'JP_Sales',
    'Other_Sales',
    'Global_Sales',
    'Critic_Score',
    'User_Score'
]
ranked_by_dic = {
    'EU_Sales': 'Vendite in Europa',
    'NA_Sales': 'Vendite in Nord America',
    'JP_Sales': 'Vendite in Giappone',
    'Other_Sales': 'Vendite nel resto del mondo',
    'Global_Sales': 'Vendite globali',
    'Critic_Score': 'Punteggio critica',
    'User_Score': 'Punteggio utenti'
}

@st.cache_data
def load_data() -> pd.DataFrame:
    game_sales = pd.read_csv(DATA_DIR)
    return game_sales

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





# ---------------- MAIN APP ---------------- #

df = load_data()

st.title("VideoGays")





# ---- FILTRI SIDEBAR ---- #
with st.sidebar:
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

    ranked_by = st.selectbox('Ordianto per', ranked_by_list)

    df_filtered = apply_filters(
        df=df,
        year_start=year_start,
        year_end=year_end,
        platform=selected_platform, 
        genre=selected_genre,
        ranked_by=ranked_by
    )

    st.markdown("-----")
    st.write(f"**Righe filtrate:** {len(df_filtered)}")
    st.write(f"**COnsole distinte:** {df_filtered['Platform'].nunique()}")
    st.write(f"**Generi distinti:** {df_filtered['Genre'].nunique()}")

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





# ---- GRAFICI ---- #
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
platform_line.update_layout(xaxis_title="Anno di rilascio", yaxis_title=ranked_by_dic[ranked_by])
col_a.plotly_chart(platform_line, use_container_width=True)

platform_area = px.area(
    df_platform, 
    x='Year_of_Release', 
    y=ranked_by,
    color='Platform',
    title="Impilati")
platform_area.update_layout(xaxis_title="Anno di rilascio", yaxis_title=ranked_by_dic[ranked_by])
col_b.plotly_chart(platform_area, use_container_width=True)



st.subheader("Trend di vendite per generi")
col_c, col_d = st.columns(2)

genre_line = px.line(
    df_genre, 
    x='Year_of_Release', 
    y=ranked_by,
    color='Genre',
    title="Singoli")
genre_line.update_layout(xaxis_title="Anno di rilascio", yaxis_title=ranked_by_dic[ranked_by])
col_c.plotly_chart(genre_line, use_container_width=True)

genre_area = px.area(
    df_genre, 
    x='Year_of_Release', 
    y=ranked_by,
    color='Genre',
    title="Impilati")
genre_area.update_layout(xaxis_title="Anno di rilascio", yaxis_title=ranked_by_dic[ranked_by])
col_d.plotly_chart(genre_area, use_container_width=True)