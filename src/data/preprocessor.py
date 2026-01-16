import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

file_path = r"C:\Users\willi\Engenering Project\SAFER-6G-Project\data\raw\UNSW_NB15_testing-set.parquet"

print("Chargement des données")
df = pd.read_parquet(file_path)

df = df.dropna()

print("Conversion des colonnes non-numériques")
le = LabelEncoder()

for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        print(f"   > Encodage de la colonne : {col}")
        df[col] = le.fit_transform(df[col].astype(str))

target_col = 'label' if 'label' in df.columns else df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

print("Normalisation des données")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Sauvegarde des fichiers")
os.makedirs('data/processed', exist_ok=True)
joblib.dump(scaler, 'data/processed/scaler.pkl')

clean_data = pd.DataFrame(X_scaled, columns=X.columns)
clean_data['label'] = y.values
clean_data.to_csv('data/processed/clean_dataset.csv', index=False)

print("Terminé ! Le fichier est prêt dans data/processed/clean_dataset.csv")