from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import os

app = FastAPI(title="SAFER-6G Detection API")

MODEL_PATH = 'models/final_model.pkl'
SCALER_PATH = 'data/processed/scaler.pkl'

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Modèle et Scaler chargés avec succès.")
else:
    print("Erreur : Modèle ou Scaler introuvable. Lancez d'abord l'entraînement.")

class NetworkData(BaseModel):
    dur: float
    proto: float
    service: float
    state: float
    spkts: float
    dpkts: float
    sbytes: float
    dbytes: float
    rate: float
    sload: float
    dload: float
    sloss: float
    dloss: float
    sinpkt: float
    dinpkt: float
    sjit: float
    djit: float
    swin: float
    stcpb: float
    dtcpb: float
    dwin: float
    tcprtt: float
    synack: float
    ackdat: float
    smean: float
    dmean: float
    trans_depth: float
    response_body_len: float
    ct_src_dport_ltm: float
    ct_dst_sport_ltm: float
    is_ftp_login: float
    ct_ftp_cmd: float
    ct_flw_http_mthd: float
    is_sm_ips_ports: float
    attack_cat: float

@app.get("/")
def home():
    return {"status": "online", "message": "SAFER-6G API is running"}

@app.post("/predict")
def predict(data: NetworkData):

    input_df = pd.DataFrame([data.dict()])
    
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)
    
    return {
        "prediction": int(prediction[0]),
        "label": "ATTACK" if prediction[0] == 1 else "NORMAL"
    }