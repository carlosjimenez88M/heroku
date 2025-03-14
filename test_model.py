import os
import joblib
import pandas as pd
from starter.ml.model import train_model, inference
from starter.ml.data import process_data

model_dir = os.path.join(os.path.dirname(__file__), "starter/model/")
model_path = os.path.join(model_dir, "model.pkl")
encoder_path = os.path.join(model_dir, "encoder.pkl")
lb_path = os.path.join(model_dir, "lb.pkl")


def test_train_model():
    """Test if the training function returns a valid model."""
    df = pd.DataFrame({"age": [25, 32, 40],
                       "salary": [">50K", "<=50K", ">50K"]})
    X, y, _, _ = process_data(df, ["age"], "salary", True)
    model = train_model(X, y)
    assert model is not None, "Trained model is None."


def test_inference():
    """Test if inference returns predictions of expected size."""
    df = pd.DataFrame({"age": [25, 32], "salary": [">50K", "<=50K"]})
    X, y, _, _ = process_data(df, ["age"], "salary", True)
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == len(y), "Predictions size mismatch."


def test_saved_model():
    """Test if the saved model, encoder, and label binarizer load correctly."""
    assert os.path.exists(model_path), f"Model not found at {model_path}"
    assert os.path.exists(encoder_path), f"Encoder not found at {encoder_path}"
    assert os.path.exists(lb_path), f"Label binarizer not found at {lb_path}"

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    lb = joblib.load(lb_path)

    assert model is not None, "Loaded model is None."
    assert encoder is not None, "Loaded encoder is None."
    assert lb is not None, "Loaded label binarizer is None."
