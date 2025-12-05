from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Games Analysis", layout="wide")

DATA_DIR = Path("vgsales_clean.csv")

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR)

    # Pulizia Dati
    # Gestione 'tbd' in User_Score e conversione a numerico
    df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
    
    # Rimuoviamo righe senza anno per i grafici temporali
    df = df.dropna(subset=['Year_of_Release'])
    df['Year_of_Release'] = df['Year_of_Release'].astype(int)
    
    # Creiamo colonna Target per ML: HIT se vendite >= 1M
    df['is_HIT'] = (df['Global_Sales'] >= 1.0).astype(int)
    
    return df

df = load_data()

# --- SIDEBAR: FILTRI GLOBALI ---
st.sidebar.header("🔍 Filtri Dashboard")
st.sidebar.write("Definisci il contesto dell'analisi.")

min_year = int(df['Year_of_Release'].min())
max_year = int(df['Year_of_Release'].max())

selected_years = st.sidebar.slider("Periodo Temporale", min_year, max_year, (min_year, max_year))
selected_platform = st.sidebar.multiselect("Piattaforma", df['Platform'].unique(), default=df['Platform'].unique()[:5])
selected_genre = st.sidebar.multiselect("Genere", df['Genre'].unique(), default=df['Genre'].unique())

# Filtro del dataset principale
df_filtered = df[
    (df['Year_of_Release'] >= selected_years[0]) &
    (df['Year_of_Release'] <= selected_years[1]) &
    (df['Platform'].isin(selected_platform)) &
    (df['Genre'].isin(selected_genre))
]

# --- LAYOUT PRINCIPALE ---
st.title("🎮 Executive Dashboard: Video Game Sales Analysis")
st.markdown("Analisi strategica per supportare decisioni su investimenti e sviluppo prodotti.")

# --- SEZIONE KPI ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Vendite Totali (M)", f"{df_filtered['Global_Sales'].sum():,.1f} M")
with col2:
    st.metric("Giochi Analizzati", f"{len(df_filtered)}")
with col3:
    avg_critic = df_filtered['Critic_Score'].mean()
    st.metric("Media Critic Score", f"{avg_critic:.1f}" if not pd.isna(avg_critic) else "N/A")
with col4:
    hit_rate = (df_filtered['is_HIT'].sum() / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
    st.metric("% HIT (>1M Copie)", f"{hit_rate:.1f}%")

st.markdown("---")

# --- TABS PER LE VARIE ANALISI ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trend Mercato", "⭐ Recensioni vs Vendite", "🤖 Predittore HIT (ML)", "💬 Smart Query"])

# TAB 1: PIATTAFORME E GENERI NEL TEMPO
with tab1:
    st.subheader("Performance Catalogo: Generi e Piattaforme")
    
    # Grafico 1: Trend temporale per Genere
    sales_over_time = df_filtered.groupby(['Year_of_Release', 'Genre'])['Global_Sales'].sum().reset_index()
    fig_trend = px.area(sales_over_time, x='Year_of_Release', y='Global_Sales', color='Genre',
                        title="Evoluzione delle Vendite Globali per Genere",
                        labels={'Global_Sales': 'Vendite (M)', 'Year_of_Release': 'Anno'})
    st.plotly_chart(fig_trend, use_container_width=True)
    st.caption(f"💡 **Insight:** Mostra quali generi hanno dominato il mercato negli anni selezionati ({selected_years[0]}-{selected_years[1]}). Le aree più ampie indicano i generi trainanti.")

    col_a, col_b = st.columns(2)
    with col_a:
        # Grafico 2: Top Piattaforme
        platform_sales = df_filtered.groupby('Platform')['Global_Sales'].sum().reset_index().sort_values('Global_Sales', ascending=False)
        fig_bar = px.bar(platform_sales.head(10), x='Global_Sales', y='Platform', orientation='h',
                         title="Top 10 Piattaforme per Vendite Totali",
                         color='Global_Sales', color_continuous_scale='Viridis')
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption("💡 **Insight:** Identifica le piattaforme storicamente più redditizie nel periodo selezionato.")
    
    with col_b:
        # Note sui dati
        st.info("**Nota sui dati:** Le vendite sono espresse in Milioni di copie. I dati mancanti sugli anni di rilascio sono stati esclusi dai grafici temporali.")

# TAB 2: RECENSIONI VS SUCCESSO
with tab2:
    st.subheader("Impatto delle Recensioni sul Successo Commerciale")
    
    # Preparazione dati (rimozione NaN per questo plot)
    df_reviews = df_filtered.dropna(subset=['Critic_Score', 'Global_Sales'])
    
    if len(df_reviews) > 0:
        fig_scatter = px.scatter(df_reviews, x='Critic_Score', y='Global_Sales', 
                                 size='Global_Sales', color='Genre', hover_name='Name',
                                 log_y=True, trendline="ols",
                                 title="Correlazione: Punteggio Critica vs Vendite Globali (Scala Log)",
                                 labels={'Critic_Score': 'Punteggio Metacritic (0-100)', 'Global_Sales': 'Vendite Globali (Scala Log)'})
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("💡 **Insight:** Ogni punto è un gioco. La linea di tendenza mostra se voti più alti corrispondono a maggiori vendite. La scala Logaritmica (asse Y) è usata per gestire la grande disparità tra mega-hit e giochi di nicchia.")
    else:
        st.warning("Dati insufficienti su Critic Score per il periodo/filtri selezionati.")

# TAB 3: MODELLO ML
with tab3:
    st.subheader("Prototipo ML: Previsione Probabilità 'HIT'")
    st.markdown("""
    **Obiettivo Business:** Stimare la probabilità che un *nuovo concept* venda **≥ 1 Milione di copie** (HIT).
    
    Questo modello usa dati storici (Genere, Piattaforma, Punteggio stimato) per supportare il Green-light process.
    """)
    
    # Training del modello "al volo" (per semplicità del prototipo)
    # 1. Prep Dati ML
    ml_data = df.dropna(subset=['Critic_Score', 'Genre', 'Platform']).copy()
    le_genre = LabelEncoder()
    le_platform = LabelEncoder()
    ml_data['Genre_Code'] = le_genre.fit_transform(ml_data['Genre'])
    ml_data['Platform_Code'] = le_platform.fit_transform(ml_data['Platform'])
    
    X = ml_data[['Genre_Code', 'Platform_Code', 'Critic_Score']]
    y = ml_data['is_HIT']
    
    # Modello semplice
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # Form Input Utente
    with st.form("prediction_form"):
        st.write("### 📝 Scheda Nuovo Prodotto")
        c1, c2, c3 = st.columns(3)
        with c1:
            in_genre = st.selectbox("Genere Previsto", le_genre.classes_)
        with c2:
            in_platform = st.selectbox("Piattaforma Target", le_platform.classes_)
        with c3:
            in_score = st.slider("Target Metacritic Score (Atteso)", 0, 100, 75)
            
        submitted = st.form_submit_button("Calcola Probabilità di Successo")
        
        if submitted:
            # Encoding input
            try:
                g_code = le_genre.transform([in_genre])[0]
                p_code = le_platform.transform([in_platform])[0]
                
                # Predizione
                proba = model.predict_proba([[g_code, p_code, in_score]])[0][1] # Probabilità classe 1 (HIT)
                
                st.markdown("### Risultato Predizione")
                col_res1, col_res2 = st.columns([1, 3])
                
                with col_res1:
                    if proba >= 0.7:
                        st.success(f"Probabilità: {proba:.0%}")
                        st.markdown("🟢 **HIGH POTENTIAL**")
                    elif proba >= 0.4:
                        st.warning(f"Probabilità: {proba:.0%}")
                        st.markdown("🟡 **MEDIUM RISK**")
                    else:
                        st.error(f"Probabilità: {proba:.0%}")
                        st.markdown("🔴 **LOW POTENTIAL**")
                
                with col_res2:
                    st.progress(proba)
                    st.info(f"Un gioco '{in_genre}' su '{in_platform}' con score {in_score} ha il {proba:.0%} di chance di superare 1M copie basandosi sullo storico.")
                    
            except Exception as e:
                st.error("Errore nella codifica dei dati (valore mai visto nel training set).")

# TAB 4: NATURAL LANGUAGE QUERY (Simulato)
with tab4:
    st.subheader("Interrogazione Rapida (NLQ)")
    st.markdown("Fai domande semplici al dataset (es. *'Nintendo Sports sales'*, *'Shooter 2010'*).")
    
    query = st.text_input("Cosa vuoi sapere?", placeholder="Es: Platform Action Nintendo...")
    
    if query:
        # Logica "Naive" di ricerca keywords (senza API costose)
        keywords = query.lower().split()
        mask = pd.Series([True] * len(df))
        
        for word in keywords:
            # Cerca la parola in colonne stringa o anni
            mask = mask & (
                df['Name'].str.lower().str.contains(word) |
                df['Platform'].str.lower().str.contains(word) |
                df['Genre'].str.lower().str.contains(word) |
                df['Publisher'].str.lower().str.contains(word) |
                df['Year_of_Release'].astype(str).str.contains(word)
            )
            
        res = df[mask]
        
        if not res.empty:
            st.write(f"Trovati **{len(res)}** risultati:")
            st.dataframe(res[['Name', 'Platform', 'Year_of_Release', 'Global_Sales', 'Publisher']].head(10))
            st.caption("Mostrati i primi 10 risultati per rilevanza testuale.")
        else:
            st.warning("Nessun risultato trovato. Prova con parole chiave più generiche.")