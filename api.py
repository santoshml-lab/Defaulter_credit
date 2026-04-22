
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# LOAD MODEL + COLUMNS
model = joblib.load("xgb_model.pkl")
cols = joblib.load("columns.pkl")

app = FastAPI()

# INPUT SCHEMA (UI/POSTMAN)
class InputData(BaseModel):
    amt: float
    category: str
    gender: str
    city_pop: int

@app.get("/")
def home():
    return {"message": "Fraud Detection API is Live 🚀"}

@app.post("/predict")
def predict(data: InputData):

    # convert input to dataframe
    df = pd.DataFrame([data.dict()])

    # one-hot encoding (same as training)
    df = pd.get_dummies(df)

    # align with training columns (🔥 IMPORTANT FIX)
    df = df.reindex(columns=cols, fill_value=0)

    # prediction
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "prediction": int(pred),
        "probability": float(prob),
        "risk": "HIGH" if prob > 0.5 else "LOW"
    }
