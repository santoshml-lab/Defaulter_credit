from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# LOAD MODEL + COLUMNS
model = joblib.load("xgb_model_v2.pkl")
cols = joblib.load("columns_v2.pkl")

app = FastAPI()

# INPUT STRUCTURE
class InputData(BaseModel):
    amt: float
    category: str
    gender: str
    city_pop: int
    lat: float
    long: float
    merch_lat: float
    merch_long: float

# HOME ROUTE
@app.get("/")
def home():
    return {"message": "Fraud Detection API is Live 🚀"}

# PREDICTION ROUTE
@app.post("/predict")
def predict(data: InputData):

    # Convert to DataFrame
    df = pd.DataFrame([data.dict()])

    # 🔥 CREATE DISTANCE
    df["distance"] = np.sqrt(
        (df["lat"] - df["merch_lat"])**2 +
        (df["long"] - df["merch_long"])**2
    )

    # DROP RAW LAT/LONG
    df = df.drop(columns=["lat","long","merch_lat","merch_long"])

    # ONE HOT ENCODING
    df = pd.get_dummies(df)

    # ALIGN WITH TRAINING COLUMNS
    df = df.reindex(columns=cols, fill_value=0)

    # PREDICT
    prob = model.predict_proba(df)[0][1]
    pred = model.predict(df)[0]

    # 🎯 RISK LOGIC
    if prob > 0.4:
        risk = "HIGH"
    elif prob > 0.15:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "prediction": int(pred),
        "probability": float(prob),
        "risk": risk
    }
     
