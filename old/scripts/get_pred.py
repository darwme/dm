import os
import pandas as pd
import joblib
import json
from prophet.serialize import model_from_json
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# Ruta del archivo .pkl
sarima_path = "E:\\Code\\PowerBi\\dm\\crime-dm\\modelos\\sarima_model.pkl"
prophet_path = "E:\\Code\\PowerBi\\dm\\crime-dm\\modelos\\prophet_model.json"

# Verificar existencia y tamaño
for path in [sarima_path, prophet_path]:
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    print(f"🔎 Archivo: {path}")
    print(f"   ➤ Existe: {exists}")
    print(f"   ➤ Tamaño: {size} bytes")
    print("-" * 50)

# --- CONFIGURACIÓN ---
SERVER = 'localhost,1433'  # o la IP/nombre de tu servidor
DATABASE = 'FlujoVehicular'
USERNAME = 'sa'
PASSWORD = 'StrongPassw0rd!'

# Cadena de conexión usando pyodbc y SQLAlchemy
conn_str = (
    f"mssql+pyodbc://{USERNAME}:{PASSWORD}@{SERVER}/{DATABASE}"
    "?driver=ODBC+Driver+17+for+SQL+Server"
)
engine = create_engine(conn_str)

# --- CARGAR MODELOS ---
sarima_model = joblib.load("E:\\Code\\PowerBi\\dm\\crime-dm\\modelos\\sarima_model.pkl")

with open("E:\\Code\\PowerBi\\dm\\crime-dm\\modelos\\prophet_model.json", "r") as f:
    prophet_model = model_from_json(json.load(f))

# --- GENERAR PREDICCIONES ---
now = datetime.now().replace(minute=0, second=0, microsecond=0)
future_dates = pd.date_range(start=now + timedelta(hours=1), periods=48, freq='H')

# SARIMA
sarima_pred = sarima_model.forecast(steps=48)
sarima_df = pd.DataFrame({
    'modelo': 'SARIMA',
    'datetime': future_dates,
    'prediccion': sarima_pred.values
})

# Prophet
future_df = pd.DataFrame({'ds': future_dates})
prophet_pred = prophet_model.predict(future_df)[['ds', 'yhat']]
prophet_df = prophet_pred.rename(columns={'ds': 'datetime', 'yhat': 'prediccion'})
prophet_df['modelo'] = 'PROPHET'

# --- COMBINAR E INSERTAR ---
final_df = pd.concat([sarima_df, prophet_df[['modelo', 'datetime', 'prediccion']]])
final_df.to_sql('predicciones_forecast', engine, if_exists='append', index=False)

print("✅ Predicciones insertadas exitosamente en la base de datos.")
