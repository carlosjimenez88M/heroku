import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from data import process_data

# Cargar los datos
df = pd.read_csv("/Users/danieljimenez/Desktop/MLops_certification/heroku/nd0821-c3-starter-code/starter/data/census.csv")
df = df.columns = df.columns.str.strip()
# Definir las columnas categóricas
categorical_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

# Procesar los datos (Entrenamiento)
X, y, encoder, lb = process_data(df, categorical_features, label="salary", training=True)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
def train_model(X_train, y_train):
    """
    Entrena un modelo de Machine Learning y lo devuelve.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# Guardar el modelo y los encoders
joblib.dump(model, "starter/model/model.pkl")
joblib.dump(encoder, "starter/model/encoder.pkl")
joblib.dump(lb, "starter/model/lb.pkl")

print("✅ Modelo entrenado y guardado exitosamente.")

# Función para evaluar el modelo
def compute_model_metrics(y, preds):
    """
    Evalúa el modelo con métricas de precisión, recall y F1-score.
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

# Hacer predicciones con el modelo
def inference(model, X):
    """
    Ejecuta inferencias con el modelo y devuelve las predicciones.
    """
    preds = model.predict(X)
    return preds

# Evaluar el modelo
y_preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_preds)

print(f"📊 Precisión: {precision:.4f}, Recall: {recall:.4f}, F1-score: {fbeta:.4f}")
