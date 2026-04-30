from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Healthcare Readmission Risk API")

# Load model
model = joblib.load("models/readmission_risk_model.joblib")

@app.get("/")
def home():
    return {"message": "Healthcare Readmission Risk API is running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    # One-hot encode same as training
    df = pd.get_dummies(df)

    # Align columns (important)
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in df:
            df[col] = 0

    df = df[model_features]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "readmission_risk_probability": round(float(probability), 4)
    }