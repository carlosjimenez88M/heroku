from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Inicializar la API
app = FastAPI()

# Cargar el modelo y los encoders
model = joblib.load("starter/model/model.pkl")
encoder = joblib.load("starter/model/encoder.pkl")
lb = joblib.load("starter/model/lb.pkl")

# Definir el esquema de entrada con Pydantic
class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de predicción de ingresos"}

@app.post("/predict")
def predict(data: InputData):
    # Convertir el input en un DataFrame
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    # Definir columnas categóricas
    categorical_features = [
        "workclass", "education", "marital_status", "occupation",
        "relationship", "race", "sex", "native_country"
    ]

    # Procesar datos para predicción
    from data import process_data
    X, _, _, _ = process_data(df, categorical_features, training=False, encoder=encoder, lb=lb)

    # Hacer la predicción
    prediction = model.predict(X)

    # Convertir la predicción a la clase original
    prediction_label = lb.inverse_transform(prediction)[0]

    return {"prediction": prediction_label}
