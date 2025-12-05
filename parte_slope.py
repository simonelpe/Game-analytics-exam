import streamlit as st
import pandas as pd
import numpy as np

# 1. Configurazione della pagina
st.set_page_config(page_title="Dashboard KPI Videogiochi", layout="wide")
st.title("🎮 Dashboard Analisi Videogiochi")

# 2. Funzione per caricare i dati (o generarli se manca il file)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data.csv')
        # Controllo colonna necessaria
        if 'Global_Sales' not in df.columns:
            st.error("Manca la colonna 'Global_Sales' nel dataset.")
            return pd.DataFrame()
    except FileNotFoundError:
        # Generazione dati finti se non hai il file
        st.warning("File 'data.csv' non trovato. Uso dati di esempio.")
        data = {
            'Name': [f'Game {i}' for i in range(100)],
            'Platform': np.random.choice(['PS4', 'XOne', 'PC', 'Switch'], 100),
            'Global_Sales': np.random.exponential(scale=0.8, size=100)
        }
        df = pd.DataFrame(data)
    return df

df = load_data()

if not df.empty:
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
        st.header("Sezione Modello ML")
        # Placeholder modello
        st.success("Qui verranno visualizzati i parametri del modello (Placeholder)")
        
        model_placeholder = st.container()
        with model_placeholder:
            st.write("Output del modello...")

else:
    st.write("Impossibile caricare i dati.")