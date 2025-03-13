# Script para entrenar el modelo de Machine Learning.

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from data import process_data  

# 1️⃣ Cargar los datos
df = pd.read_csv("/Users/danieljimenez/Desktop/MLops_certification/heroku/nd0821-c3-starter-code/starter/data/census.csv")

# 🔹 Limpiar nombres de columnas eliminando espacios en blanco
df.columns = df.columns.str.strip()

# 🔹 Verificar que "salary" está correctamente escrito
if "salary" not in df.columns:
    raise KeyError("⚠️ ERROR: La columna 'salary' no está en el dataset. Revisa los nombres de las columnas.")

# 2️⃣ Definir las columnas categóricas
categorical_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

# 3️⃣ Dividir datos en entrenamiento y prueba
train, test = train_test_split(df, test_size=0.20, random_state=42)

# 4️⃣ Procesar datos (Entrenamiento)
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=categorical_features, label="salary", training=True
)

# 5️⃣ Procesar datos (Prueba)
X_test, y_test, _, _ = process_data(
    test, categorical_features=categorical_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# 6️⃣ Entrenar el modelo
def train_model(X_train, y_train):
    """
    Entrena un modelo de Machine Learning y lo devuelve.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
def compute_slice_metrics(df, feature, model, encoder, lb):
    with open("slice_output.txt", "w") as s:
        for val in df[feature].unique():
            temp = df[df[feature] == val]
            X, y, _, _ = process_data(
                temp, categorical_features=categorical_features, label="salary",
                training=False, encoder=encoder, lb=lb
            )
            preds = inference(model, X)
            p, r, f = compute_model_metrics(y, preds)
            s.write(f"{feature}={val} Precision={p} Recall={r} F1={f}\n")

model = train_model(X_train, y_train)

# 7️⃣ Guardar el modelo y los encoders
joblib.dump(model, "model/model.pkl")
joblib.dump(encoder, "model/encoder.pkl")
joblib.dump(lb, "model/lb.pkl")

print("✅ Modelo entrenado y guardado exitosamente.")
compute_slice_metrics(df, "education", model, encoder, lb)
print("slice_output.txt creado")