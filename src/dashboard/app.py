import streamlit as st
import requests
import pandas as pd
import random

st.set_page_config(page_title="SAFER-6G Real-Time Monitor", layout="wide")

st.title("🛡️ SAFER-6G : Network Intrusion Detection")

@st.cache_data 
def load_sample_data():
    return pd.read_csv('data/processed/clean_dataset.csv')

try:
    df_samples = load_sample_data()
    X_samples = df_samples.drop(columns=['label'])
except:
    st.error("⚠️ Fichier 'clean_dataset.csv' introuvable. Lance le preprocessor d'abord.")
    st.stop()

st.sidebar.header("Contrôle du Simulateur")

if st.sidebar.button("Piger un échantillon aléatoire"):
    random_idx = random.randint(0, len(X_samples) - 1)
    sample_row = X_samples.iloc[random_idx]
    
    st.session_state['current_sample'] = sample_row.to_dict()
    st.session_state['real_label'] = df_samples.iloc[random_idx]['label']

if 'current_sample' in st.session_state:
    sample = st.session_state['current_sample']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Données de l'échantillon")
        st.json({k: sample[k] for k in list(sample.keys())[:10]}) 
        
        if st.button("🔍 Lancer l'Analyse IA"):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/predict", 
                    json=sample, 
                    timeout=5
                )
                
                if response.status_code == 200:
                    res = response.json()
                    pred = res['prediction']
                    
                    real = st.session_state['real_label']
                    
                    if pred == 1:
                        st.error(f"ALERTE : Attaque détectée !")
                    else:
                        st.success(f"Trafic Normal.")
                        
                    st.info(f"VÉRITÉ TERRAIN : {'Attaque' if real == 1 else 'Normal'}")
                else:
                    st.warning("Erreur API : Vérifie la structure NetworkData.")
            except:
                st.error("L'API est-elle lancée ? (uvicorn)")

    with col2:
        st.subheader("Visualisation des Features")
        st.bar_chart(pd.Series(sample).head(15))

else:
    st.info("Clique sur 'Piger un échantillon' dans le menu à gauche pour commencer.")