from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

model = joblib.load("xgb_model.pkl")
cols = joblib.load("columns.pkl")

app = FastAPI()

class InputData(BaseModel):
    amt: float
    category: str
    gender: str
    city_pop: int
    lat: float
    long: float
    merch_lat: float
    merch_long: float

@app.post("/predict")
def predict(data: InputData):

    df = pd.DataFrame([data.dict()])

    # 🔥 create distance same as training
    df["distance"] = np.sqrt(
        (df["lat"] - df["merch_lat"])**2 +
        (df["long"] - df["merch_long"])**2
    )

    df = df.drop(columns=["lat","long","merch_lat","merch_long"])

    # encoding
    df = pd.get_dummies(df)

    # align
    df = df.reindex(columns=cols, fill_value=0)

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "prediction": int(pred),
        "probability": float(prob),
        "risk": "HIGH" if prob > 0.4 else "LOW"
    }
     
