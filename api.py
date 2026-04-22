from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# LOAD MODEL + COLUMNS
model = joblib.load("xgb_model.pkl")
cols = joblib.load("columns.pkl")

@app.get("/")
def home():
    return {"message": "Credit Default API Running 🚀"}

@app.post("/predict")
def predict(data: dict):
    try:
        # input → dataframe
        df = pd.DataFrame([data])

        # encoding (same as training)
        df = pd.get_dummies(df)

        # align columns
        df = df.reindex(columns=cols, fill_value=0)

        # prediction
        prob = model.predict_proba(df)[0][1]

        return {
            "probability": float(prob),
            "decision": "High Risk ⚠️" if prob > 0.5 else "Low Risk ✅"
        }

    except Exception as e:
        return {"error": str(e)}
