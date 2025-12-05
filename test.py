import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Game Sales Executive Dashboard", layout="wide", page_icon="🎮")

# --- UTILS & DATA LOADING ---

@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('data.csv')

    # --- CORREZIONE GESTIONE DATE ---
    # 1. Convertiamo la colonna in oggetti datetime (gestisce sia '2006' che '2006-01-01' che errori)
    # errors='coerce' trasforma valori come 'tbd' o 'N/A' in NaT (Not a Time)
    df['Year_of_Release'] = pd.to_datetime(df['Year_of_Release'], errors='coerce')
    
    # 2. Estraiamo solo l'anno (es. 2006)
    df['Year_of_Release'] = df['Year_of_Release'].dt.year

    # 3. Rimuoviamo le righe dove l'anno o il nome non sono validi
    df = df.dropna(subset=['Year_of_Release', 'Name'])
    
    # 4. Ora è sicuro convertire in intero (rimuove i decimali come 2006.0)
    df['Year_of_Release'] = df['Year_of_Release'].astype(int)
    
    # --- FINE CORREZIONE ---

    # Conversione User_Score in numerico (gestione 'tbd')
    df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')

    # 2. IMPUTAZIONE INTELLIGENTE
    numeric_cols_to_impute = ['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']
    for col in numeric_cols_to_impute:
        # Verifica se la colonna esiste per evitare errori se il CSV cambia
        if col in df.columns:
            df[col] = df.groupby('Genre')[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median())

    cat_cols_to_impute = ['Developer', 'Rating']
    for col in cat_cols_to_impute:
        if col in df.columns:
            def fill_mode(x):
                m = x.mode()
                return m[0] if not m.empty else "Unknown"
            
            df[col] = df.groupby('Genre')[col].transform(lambda x: x.fillna(fill_mode(x)))
            df[col] = df[col].fillna("Unknown")

    # 3. Target e Feature Engineering
    df['is_HIT'] = (df['Global_Sales'] >= 1.0).astype(int)
    
    le_genre = LabelEncoder()
    le_platform = LabelEncoder()
    # Convertiamo in stringa per sicurezza prima dell'encoding
    df['Genre_Code'] = le_genre.fit_transform(df['Genre'].astype(str))
    df['Platform_Code'] = le_platform.fit_transform(df['Platform'].astype(str))
    
    return df, le_genre, le_platform

@st.cache_resource
def train_model(df, feature_cols, target_col, max_depth, n_estimators, min_samples_leaf):
    """Allena il modello Random Forest e calcola metriche."""
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    # Baseline: predire sempre la classe maggioritaria
    majority_class = y_test.value_counts().idxmax()
    baseline_acc = accuracy_score(y_test, [majority_class]*len(y_test))

    metrics = {
        "acc_train": acc_train,
        "acc_test": acc_test,
        "baseline_acc": baseline_acc
    }
    
    feature_importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=True)

    return model, metrics, feature_importances

# --- CARICAMENTO ---
df, le_genre, le_platform = load_and_clean_data()

# --- SIDEBAR: SETTINGS & FILTRI ---
with st.sidebar:
    st.header("⚙️ Configurazione")
    
    st.subheader("Iperparametri ML")
    hyp_depth = st.slider("Max Depth", 1, 20, 10)
    hyp_est = st.slider("N Estimators", 10, 200, 100)
    hyp_leaf = st.slider("Min Samples Leaf", 1, 10, 2)
    
    st.markdown("---")
    st.subheader("Filtri Dati (Dashboard)")
    all_genres = df['Genre'].unique()
    sel_genres = st.multiselect("Generi", all_genres, default=all_genres[:5])
    
    min_y, max_y = int(df['Year_of_Release'].min()), int(df['Year_of_Release'].max())
    sel_years = st.slider("Anni", min_y, max_y, (2000, max_y))

# Filtro dati per la dashboard business (non per l'ML training che usa tutto lo storico)
df_filtered = df[
    (df['Genre'].isin(sel_genres if sel_genres else all_genres)) & 
    (df['Year_of_Release'] >= sel_years[0]) & 
    (df['Year_of_Release'] <= sel_years[1])
]

# --- MAIN PAGE STRUCTURE ---
st.title("🎮 Executive Dashboard & ML Analysis")
st.markdown("Analisi strategica delle vendite videogiochi e modello predittivo di successo commerciale.")

tab_biz, tab_ml, tab_nlq = st.tabs(["📊 Business Dashboard", "🤖 ML & Analisi Profonda", "💬 Smart Query"])

# ==============================================================================
# TAB 1: BUSINESS DASHBOARD (Richieste 1 e 2)
# ==============================================================================
with tab_biz:
    # KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vendite Totali", f"{df_filtered['Global_Sales'].sum():,.1f} M")
    c2.metric("Titoli Analizzati", len(df_filtered))
    c3.metric("% HIT (>1M)", f"{(df_filtered['is_HIT'].mean()*100):.1f}%")
    c4.metric("Avg Critic Score", f"{df_filtered['Critic_Score'].mean():.1f}")
    
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Performance Catalogo")
        # Trend temporale (Area chart)
        sales_trend = df_filtered.groupby(['Year_of_Release', 'Genre'])['Global_Sales'].sum().reset_index()
        fig_trend = px.area(sales_trend, x='Year_of_Release', y='Global_Sales', color='Genre',
                            title="Trend Vendite per Genere (Stacked)")
        st.plotly_chart(fig_trend, use_container_width=True)
        st.caption("Mostra l'evoluzione dei gusti di mercato nel tempo. Aree più grandi = generi dominanti.")

    with col_chart2:
        st.subheader("Recensioni vs Vendite")
        # Scatter plot (Log scale)
        fig_scatter = px.scatter(df_filtered, x='Critic_Score', y='Global_Sales', color='Genre',
                                 log_y=True, hover_name='Name', title="Critic Score vs Global Sales (Log Scale)")
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Relazione tra qualità percepita e successo. Asse Y logaritmico per evidenziare le differenze tra HIT e nicchia.")

# ==============================================================================
# TAB 2: ML DEMO (Ispirato a heart_ml_app.py)
# ==============================================================================
with tab_ml:
    st.header("🧠 Modello Predittivo: Probabilità di HIT")
    st.caption("Il modello stima se un gioco venderà ≥ 1M di copie basandosi su caratteristiche note e punteggi attesi.")
    
    # --- PANORAMICA DATASET ML ---
    st.subheader("🔍 Panoramica Dataset ML")
    col_a, col_b, col_c = st.columns(3)
    n_games = len(df)
    hit_rate = df['is_HIT'].mean()
    
    col_a.metric("Totale Giochi (Dataset Completo)", n_games)
    col_b.metric("Hit Rate (Vendite ≥ 1M)", f"{hit_rate:.1%}")
    col_c.metric("Rapporto Classi", f"{(1-hit_rate):.1%} Flop / {hit_rate:.1%} Hit")
    
    with st.expander("Mostra prime righe del dataset (con imputazione applicata)"):
        st.dataframe(df[['Name', 'Genre', 'Platform', 'Critic_Score', 'User_Score', 'is_HIT']].head())

    # --- CHALLENGE SECTION ---
    st.markdown("---")
    with st.expander("🎯 Challenge: Esplora le Feature", expanded=True):
        feature_options = ['Critic_Score', 'User_Score', 'Year_of_Release', 'Critic_Count', 'User_Count']
        selected_feat = st.selectbox("Seleziona variabile da analizzare", feature_options)
        
        c_d, c_e = st.columns(2)
        
        # Plot 1: Istogramma
        fig_hist = px.histogram(df, x=selected_feat, color="is_HIT", barmode="overlay",
                                title=f"Distribuzione di {selected_feat}",
                                labels={'is_HIT': 'Successo (1=Hit)'})
        c_d.plotly_chart(fig_hist, use_container_width=True)
        
        # Plot 2: Violin Plot
        fig_violin = px.violin(df, y=selected_feat, x="is_HIT", color="is_HIT", box=True,
                               title=f"{selected_feat} vs Target Hit",
                               labels={'is_HIT': 'Classe (0=No, 1=Hit)'})
        c_e.plotly_chart(fig_violin, use_container_width=True)
        
        # Stats
        st.markdown("#### Statistiche Descrittive")
        st.dataframe(df.groupby('is_HIT')[selected_feat].describe(), use_container_width=True)
        st.caption("Osserva se media e distribuzione cambiano significativamente tra giochi 'Hit' (1) e non (0).")

    # --- TRAINING & PERFORMANCE ---
    st.subheader("📏 Performance del Modello")
    
    # Feature usate per il training
    ML_FEATURES = ['Genre_Code', 'Platform_Code', 'Critic_Score', 'User_Score', 'Year_of_Release']
    FEATURE_LABELS = {
        'Genre_Code': 'Genere', 'Platform_Code': 'Piattaforma', 
        'Critic_Score': 'Punteggio Critica', 'User_Score': 'Punteggio Utenti', 
        'Year_of_Release': 'Anno Uscita'
    }
    
    model, metrics, feat_imp = train_model(df, ML_FEATURES, 'is_HIT', hyp_depth, hyp_est, hyp_leaf)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy Train", f"{metrics['acc_train']:.2%}")
    col2.metric("Accuracy Test", f"{metrics['acc_test']:.2%}")
    col3.metric("Baseline (Moda)", f"{metrics['baseline_acc']:.2%}")
    
    # Check Overfitting
    gap = metrics['acc_train'] - metrics['acc_test']
    if gap > 0.15:
        st.warning(f"⚠️ Possibile Overfitting (Gap: {gap:.1%}). Prova a ridurre Max Depth o aumentare Min Samples Leaf.")
    else:
        st.success("✅ Il modello generalizza bene (Gap contenuto tra Train e Test).")

    # --- CORRELAZIONI E IMPORTANZA ---
    st.subheader("📈 Analisi Fattori di Successo")
    col_corr, col_imp = st.columns(2)
    
    with col_corr:
        st.markdown("**Matrice di Correlazione (Feature Numeriche)**")
        # Usiamo solo feature numeriche vere + target
        corr_cols = ['Critic_Score', 'User_Score', 'Critic_Count', 'User_Count', 'Global_Sales', 'is_HIT']
        corr = df[corr_cols].corr()
        
        fig_corr, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig_corr)
        
    with col_imp:
        st.markdown("**Importanza Variabili (Random Forest)**")
        # Mappiamo i nomi tecnici a label leggibili per il grafico
        feat_imp_renamed = feat_imp.rename(index=FEATURE_LABELS)
        
        fig_imp, ax_imp = plt.subplots(figsize=(5,4))
        feat_imp_renamed.plot(kind="barh", color="#ff4b4b", ax=ax_imp)
        ax_imp.set_xlabel("Importanza (Gini Impurity)")
        st.pyplot(fig_imp)

    # --- FOCUS BOXPLOT ---
    top_feature_tech = feat_imp.idxmax()
    top_feature_label = FEATURE_LABELS[top_feature_tech]
    
    st.markdown("---")
    st.markdown(f"##### 📦 Focus Feature Più Importante: {top_feature_label}")
    
    fig_box, ax_box = plt.subplots(figsize=(8, 3))
    sns.boxplot(data=df, x='is_HIT', y=top_feature_tech, hue='is_HIT', palette="Set2", ax=ax_box)
    ax_box.set_xticklabels(["Non Hit (<1M)", "Hit (≥1M)"])
    ax_box.set_xlabel("Successo Commerciale")
    ax_box.set_ylabel(top_feature_label)
    st.pyplot(fig_box)

    # --- PREDIZIONE UTENTE ---
    st.subheader("🧪 Simulatore Lancio Prodotto")
    st.markdown("Inserisci le caratteristiche del gioco ipotetico:")
    
    with st.form("sim_form"):
        c_f1, c_f2, c_f3 = st.columns(3)
        with c_f1:
            in_genre = st.selectbox("Genere", le_genre.classes_)
            in_plat = st.selectbox("Piattaforma", le_platform.classes_)
        with c_f2:
            in_critic = st.slider("Critic Score (Atteso)", 0, 100, 75)
            in_user = st.slider("User Score (Atteso)", 0.0, 10.0, 7.5)
        with c_f3:
            in_year = st.number_input("Anno Lancio", 2000, 2030, 2024)
            
        submitted = st.form_submit_button("Predici Successo")
        
        if submitted:
            # Encoding input
            g_code = le_genre.transform([in_genre])[0]
            p_code = le_platform.transform([in_plat])[0]
            
            input_data = pd.DataFrame([[g_code, p_code, in_critic, in_user, in_year]], columns=ML_FEATURES)
            proba = model.predict_proba(input_data)[0][1] # Prob classe 1
            
            st.markdown("### Risultato Previsione")
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                if proba > 0.5:
                    st.success("🟢 PROBABILE HIT")
                else:
                    st.error("🔴 RISCHIO FLOP")
                st.metric("Probabilità >1M Copie", f"{proba:.1%}")
                
            with col_res2:
                st.progress(proba)
                st.info(f"Un gioco **{in_genre}** su **{in_plat}** con score **{in_critic}** ha il {proba:.0%} di chance di diventare un Hit.")

# ==============================================================================
# TAB 3: NLQ (Richiesta 4)
# ==============================================================================
with tab_nlq:
    st.subheader("Interrogazione in Linguaggio Naturale (Keywords)")
    query = st.text_input("Cerca nel catalogo (es. 'Nintendo Racing 2008')", "")
    
    if query:
        keywords = query.lower().split()
        mask = pd.Series([True]*len(df))
        for k in keywords:
            mask = mask & (
                df['Name'].str.lower().str.contains(k) | 
                df['Platform'].str.lower().str.contains(k) |
                df['Genre'].str.lower().str.contains(k) |
                df['Publisher'].astype(str).str.lower().str.contains(k) |
                df['Year_of_Release'].astype(str).str.contains(k)
            )
        res = df[mask]
        st.write(f"Trovati {len(res)} risultati:")
        st.dataframe(res.head(20))