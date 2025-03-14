import joblib
import pandas as pd
from starter.ml.model import train_model
from data import process_data


def test_train_model():
    df = pd.DataFrame(
        {"age": [25, 32, 40], "salary": [">50K", "<=50K", ">50K"]})
    X, y, _, _ = process_data(df, ["age"], "salary", True)
    model = train_model(X, y)
    assert model is not None


def test_inference():
    df = pd.DataFrame({"age": [25, 32], "salary": [">50K", "<=50K"]})
    X, y, _, _ = process_data(df, ["age"], "salary", True)
    model = train_model(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_saved_model():
    model_path = "model/model.pkl"
    m = joblib.load(model_path)
    assert m is not None
