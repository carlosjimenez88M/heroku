import requests


HEROKU_API_URL = "https://super-cool-demo-app-aba1064edd33.herokuapp.com/"


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


response = requests.post(HEROKU_API_URL, json=input_data)


print("Status Code:", response.status_code)
print("Response JSON:", response.json())
