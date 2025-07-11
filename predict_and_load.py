import pandas as pd
import numpy as np
import joblib
import pyodbc
from datetime import datetime, timedelta

# Variables categóricas derivadas del tiempo
variables_categoricas = [
    'hour_sin', 'hour_cos', 'period_AM', 'period_PM',
    'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
    'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12',
    'weekday_0', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6'
]

modelos = {
    'XGB': 'models/v3/modelo_forecaster_XGB.pkl',
    'LGBM': 'models/v3/modelo_forecaster_LGBM.pkl',
    'CatBoost': 'models/v3/modelo_forecaster_CatBoost.pkl'
}

def get_connection_string(
    server='localhost,1433',
    database='FlujoVehicular',
    username='sa',
    password='StrongPassw0rd!',
    driver='{ODBC Driver 17 for SQL Server}'
):
    return f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};'

cnn = get_connection_string()

# Obtener fechas mínimas y máximas reales desde fact_pasos
with pyodbc.connect(cnn, autocommit=True) as conn:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT MIN(CONVERT(DATETIME, CONVERT(VARCHAR, fecha, 23) + ' ' + RIGHT('0' + CAST(hora_id AS VARCHAR), 2) + ':00:00')), \
                MAX(CONVERT(DATETIME, CONVERT(VARCHAR, fecha, 23) + ' ' + RIGHT('0' + CAST(hora_id AS VARCHAR), 2) + ':00:00')) FROM fact_pasos"
    )
    min_date, max_date = cursor.fetchone()
    print(f"Primer registro: {min_date} / Último registro: {max_date}")

# Generar fechas futuras hasta 1 año después del último dato
future_dates = pd.date_range(start=min_date, end=max_date + timedelta(days=365), freq='h')
df_futuro = pd.DataFrame({'date_time': future_dates})

# Crear variables exógenas de tiempo
df_futuro['hour'] = df_futuro['date_time'].dt.hour
df_futuro['month'] = df_futuro['date_time'].dt.month
df_futuro['weekday'] = df_futuro['date_time'].dt.weekday

for m in range(1, 13):
    df_futuro[f'month_{m}'] = (df_futuro['month'] == m).astype(float)
for w in range(7):
    df_futuro[f'weekday_{w}'] = (df_futuro['weekday'] == w).astype(float)

df_futuro['hour_sin'] = np.sin(2 * np.pi * df_futuro['hour'] / 24)
df_futuro['hour_cos'] = np.cos(2 * np.pi * df_futuro['hour'] / 24)
df_futuro['period_AM'] = (df_futuro['hour'] < 12).astype(float)
df_futuro['period_PM'] = (df_futuro['hour'] >= 12).astype(float)
df_futuro = df_futuro.drop(['hour', 'month', 'weekday'], axis=1)

# Limpiar tabla forecasting_pasos
with pyodbc.connect(cnn, autocommit=True) as conn:
    cursor = conn.cursor()
    cursor.execute("TRUNCATE TABLE forecasting_pasos")

batch_size = 50_000
corte_fecha = pd.Timestamp('2020-12-31 23:00:00')

for nombre, modelo_path in modelos.items():
    print(f"\nProcesando modelo {nombre}...")
    data = joblib.load(modelo_path)
    forecaster = data['model']
    last_fecha = forecaster.last_window_.index[-1]
    print(f"Última fecha entrenada: {last_fecha}")

    # Filtrar fechas futuras que empiezan justo después del entrenamiento
    df_futuro_modelo = df_futuro[df_futuro['date_time'] > last_fecha].copy()
    X_pred = df_futuro_modelo[variables_categoricas]
    fechas_pred = df_futuro_modelo['date_time']

    preds = []
    errores_sql = 0
    errores_pred = 0

    for start in range(0, len(X_pred), batch_size):
        end = min(start + batch_size, len(X_pred))
        X_batch = X_pred.iloc[start:end].copy()
        fechas_batch = fechas_pred.iloc[start:end]
        X_batch.index = pd.DatetimeIndex(fechas_batch.values).to_period('h').to_timestamp()

        try:
            y_batch = forecaster.predict(steps=len(X_batch), exog=X_batch)
        except Exception as ex:
            print(f"Error al predecir lote {start}-{end} para modelo {nombre}: {ex}")
            errores_pred += 1
            y_batch = [np.nan] * len(X_batch)

        arr = np.array(y_batch)
        arr[np.isinf(arr)] = np.nan
        arr = [None if np.isnan(x) else float(x) for x in arr]
        preds.extend(arr)

    # Crear DataFrame con resultados
    df_out = pd.DataFrame({
        'date_time': fechas_pred,
        'prediccion': preds,
        'modelo': f"MODEL_EXOGENEAS_{nombre}"
    })

    # Separar en forecasting (hasta 2020) y predicción futura (2021+)
    df_forecasting = df_out[df_out['date_time'] <= corte_fecha].copy()
    df_predicciones = df_out[df_out['date_time'] > corte_fecha].copy()
    df_predicciones['execution_date'] = datetime.now()

    # Insertar en forecasting_pasos
    with pyodbc.connect(cnn, autocommit=True) as conn:
        cursor = conn.cursor()
        for i in range(0, len(df_forecasting), batch_size):
            batch = df_forecasting.iloc[i:i+batch_size]
            values = [tuple(row) for row in batch.itertuples(index=False, name=None)]
            try:
                cursor.fast_executemany = True
                cursor.executemany(
                    "INSERT INTO forecasting_pasos (date_time, prediccion, modelo) VALUES (?, ?, ?)",
                    values
                )
            except Exception as ex:
                print(f"Error al insertar lote forecasting {i}-{i+batch_size} para modelo {nombre}: {ex}")
                errores_sql += 1

    # Insertar en prediccion_pasos
    with pyodbc.connect(cnn, autocommit=True) as conn:
        cursor = conn.cursor()
        for i in range(0, len(df_predicciones), batch_size):
            batch = df_predicciones.iloc[i:i+batch_size]
            values = [tuple(row) for row in batch.itertuples(index=False, name=None)]
            try:
                cursor.fast_executemany = True
                cursor.executemany(
                    "INSERT INTO prediccion_pasos (date_time, prediccion, modelo, execution_date) VALUES (?, ?, ?, ?)",
                    values
                )
            except Exception as ex:
                print(f"Error al insertar lote predicciones {i}-{i+batch_size} para modelo {nombre}: {ex}")
                errores_sql += 1

    print(f"{nombre}: forecasting = {len(df_forecasting)}, predicciones = {len(df_predicciones)}, errores_pred = {errores_pred}, errores_sql = {errores_sql}")

print("\n✅ ¡Listo! Todas las predicciones han sido generadas y cargadas.")
