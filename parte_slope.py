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

'''
Capire su quali piattaforme e generi ha funzionato meglio il catalogo nel tempo.
Valutare quanto contano recensioni e rating sul successo commerciale di un gioco.
Avere un prototipo di modello ML che aiuti a stimare la probabilità che un nuovo gioco diventi un “HIT” (es. ≥ 1M copie).
(Nice to have) Un modo rapido per interrogare i dati in linguaggio naturale.
'''