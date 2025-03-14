# =====================#
# ---- Libraries ---- #
# =====================#
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ===============#
# ---- APP ---- #
# ===============#

app = FastAPI()


model = joblib.load("starter/model/model.pkl")
encoder = joblib.load("starter/model/encoder.pkl")
lb = joblib.load("starter/model/lb.pkl")

# ===========================#
# ---- Class Functions ---- #
# ===========================#


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
    return {"message": "Welcome to the Income Prediction API."}


@app.post("/predict")
def predict(data: InputData):
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    categorical_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country"
    ]

    from data import process_data
    X, _, _, _ = process_data(
        df, categorical_features,
        training=False,
        encoder=encoder, lb=lb)

    prediction = model.predict(X)
    prediction_label = lb.inverse_transform(prediction)[0]
    return {"prediction": prediction_label}
