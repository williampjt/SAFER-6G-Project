import pandas as pd
file_path = "C:\Users\willi\Engenering Project\SAFER-6G-Project\data\raw\UNSW_NB15_testing-set.parquet"

def load_and_preprocess_data(file_path: str):
    print(f"Chargement des données depuis {file_path}...")
    
    # --- Changement clé ici : utiliser read_parquet ---
    try:
        data = pd.read_parquet(file_path)
    except FileNotFoundError:
        print("Erreur : Fichier Parquet non trouvé. Vérifiez le chemin.")
        return None

    # --- Étape de Prétraitement ---
    # 1. Gestion des valeurs manquantes, etc.
    # 2. Encodage des caractéristiques catégorielles
    # ...
    
    return data

# Utilisation:
raw_file_path = 'data/raw/mon_dataset.parquet'
clean_df = load_and_preprocess_data(raw_file_path)

if clean_df is not None:
    # Sauvegarde des données nettoyées en format CSV ou Parquet compressé
    clean_df.to_csv('data/processed/clean_dataset.csv', index=False)
    print("Prétraitement terminé. Données sauvegardées.")