from typing import Optional
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Titanic Survival Prediction API")
model = joblib.load("model.joblib")


class Passenger(BaseModel):
    Pclass: int                      # 1, 2 ya 3
    Sex: int                         # 0 = male, 1 = female
    Age: Optional[float] = None
    sibsp: int
    Parch: int
    Fare: float
    Embarked: Optional[int] = None   # 0, 1 ya 2


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(p: Passenger):
    df = pd.DataFrame([p.model_dump()]).astype(float)
    df = df[["Pclass", "Sex", "Age", "sibsp", "Parch", "Fare", "Embarked"]]
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return {"survived": pred, "probability": round(prob, 3)}