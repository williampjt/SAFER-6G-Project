import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def train_model(data_path):
    if not os.path.exists(data_path):
        print(f"Erreur : Le fichier {data_path} est introuvable")
        return

    print("Chargement des données")
    df = pd.read_csv(data_path)
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Début de l'entraînement sur {len(X_train)} lignes")
    model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        use_label_encoder=False, 
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("\nRapport de performance :")
    print(classification_report(y_test, y_pred))
    
    os.makedirs('models', exist_ok=True)
    model_path = 'models/final_model.pkl'
    joblib.dump(model, model_path)
    
    print(f"\nModèle sauvegardé avec succès dans : {model_path}")

if __name__ == "__main__":
    train_model('data/processed/clean_dataset.csv')