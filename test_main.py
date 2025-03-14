from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Income Prediction API."}


def test_post_predict():
    input_data = {
        "age": 35,
        "workclass": "Private",
        "fnlgt": 215646,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Never-married",
        "occupation": "Exec-managerial",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
