# Tratamiento de datos
# ==============================================================================
import numpy as np
import nbformat
print(nbformat.__version__)  # Debe ser >= 4.2.0
import pandas as pd
import pyodbc
from datetime import datetime
from dotenv import load_dotenv
import os
# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly.express as px
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
%matplotlib inline

# Modelado y Forecasting
# ==============================================================================
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

from skforecast.ForecasterAutoreg import ForecasterAutoreg
# from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from skforecast.model_selection import backtesting_forecaster, grid_search_forecaster

from joblib import dump, load

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')
%config Completer.use_jedi = False

import pyodbc
import pandas as pd
from datetime import datetime

# 1. Obtener cadena de conexión sin usar os ni dotenv
def get_connection_string(server: str, database: str, username: str, password: str, driver: str = '{ODBC Driver 17 for SQL Server}'):
    return f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};'


# 2. Cargar datos de una tabla a DataFrame
def get_blob_to_df(table_name: str, connection_str: str):
    try:
        with pyodbc.connect(connection_str, autocommit=True) as conn:
            query = f"SELECT * FROM {table_name};"
            df = pd.read_sql(query, conn)
        return True, df
    except Exception as e:
        return False, str(e)


# 3. Guardar resultados con fecha de ejecución
def guardar_resultados(datos: pd.DataFrame, modelo: str):
    fecha_ejecucion = datetime.now()
    resultados = pd.DataFrame({
        'DATE_TIME': datos.index,
        'PREDICCION': datos['pred'],
        'MODELO': modelo,
        'EXECUTION_DATE': fecha_ejecucion
    })
    return resultados


# 4. Guardar pronóstico sin fecha de ejecución
def guardar_pronostico(datos: pd.DataFrame, modelo: str):
    resultados = pd.DataFrame({
        'DATE_TIME': datos.index,
        'PREDICCION': datos['pred'],
        'MODELO': modelo
    })
    return resultados


# 5. Subir DataFrame a SQL Server (sin optimización por lotes)
def upload_df_to_blob(df: pd.DataFrame, table_name: str, connection_str: str):
    try:
        df = df.where(pd.notnull(df), None)
        with pyodbc.connect(connection_str, autocommit=True) as conn:
            cursor = conn.cursor()
            columns = ','.join(df.columns)
            placeholders = ','.join(['?'] * len(df.columns))
            cmd_insert = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            cursor.fast_executemany = True
            cursor.executemany(cmd_insert, df.values.tolist())
        return True, ""
    except Exception as e:
        return False, str(e)


# 6. Subir DataFrame a SQL Server en lotes (mejor rendimiento)
def upload_df_to_blob_with_steroids(df: pd.DataFrame, table_name: str, connection_str: str, batch_size: int = 5000):
    try:
        df = df.where(pd.notnull(df), None)
        with pyodbc.connect(connection_str, autocommit=False) as conn:
            cursor = conn.cursor()
            columns = ','.join(df.columns)
            placeholders = ','.join(['?'] * len(df.columns))
            cmd_insert = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            cursor.fast_executemany = True

            for start in range(0, len(df), batch_size):
                end = start + batch_size
                batch = df.iloc[start:end].values.tolist()
                cursor.executemany(cmd_insert, batch)
                conn.commit()

        return True, ""
    except Exception as e:
        return False, str(e)


# 7. Crear tabla SQL
def crear_tabla_sql(connection_str: str, sql_create: str):
    try:
        with pyodbc.connect(connection_str, autocommit=True) as conn:
            cursor = conn.cursor()
            cursor.execute(sql_create)
        print("Tabla creada correctamente.")
    except Exception as e:
        print(f"Error al crear la tabla: {e}")

# Configura la conexión de base de datos SQL Server
cnn = get_connection_string('LAPTOP-9JDBI17R', 'master', 'sa', 'm@rk3t2o15')
# Query SQL para extraer datos
_, datos = get_blob_to_df(table_name = 'MD_FLUJO_VEHICULAR', connection_str = cnn)

# Eliminar la parte de la hora si existe
datos['fecha'] = datos['fecha'].str.extract(r'^(\d{2}/\d{2}/\d{4})')[0]

# Convertir las fechas a formato Y-M-D como texto
datos['fecha'] = pd.to_datetime(datos['fecha'], dayfirst=True).dt.strftime('%Y-%m-%d')
datos['fecha'] = pd.to_datetime(datos['fecha'])

datos['fecha']

<!-- 0         2017-01-01
1         2017-01-01
2         2017-01-01
3         2017-01-01
4         2017-01-01
             ...    
1048570   2020-12-29
1048571   2020-12-29
1048572   2020-12-29
1048573   2020-12-29
1048574   2020-12-29
Name: fecha, Length: 1048575, dtype: datetime64[ns] -->

**Descripción del conjunto de datos de flujo vehicular**

Este conjunto de datos recoge información sobre el tránsito vehicular registrado en distintas estaciones de monitoreo a 
partir del año 2017, en una ciudad específica. Los datos reflejan el volumen de vehículos que circularon por cada estación, desglosado por tipo de vehículo, método de pago y dirección del flujo. Esta información es clave para estudiar patrones de movilidad urbana y apoyar decisiones en la gestión del transporte.

El dataset resultante contiene las siguientes columnas:

| **Columna**        | **Descripción**                                              |
|--------------------|--------------------------------------------------------------|
| `periodo`          | Año al que pertenece cada registro.                                 |
| `fecha`            | Fecha de la medición (formato día/mes/año).                             |
| `hora_inicio`      | Hora exacta en que comienza el conteo.                             |
| `estacion`         | nombre de la estación donde se realizó el conteo vehicular                  |
| `sentido`          | Sentido del flujo vehicular (por ejemplo, "Centro").       |
| `tipo_vehiculo`    | Clasificación del vehículo (Liviano, Pesado, etc.).          |
| `forma_pago`       | Medio de pago utilizado (efectivo, telepase, no cobrado).    |
| `cantidad_pasos`   | Total de vehículos contabilizados en ese periodo. |

Este dataset es especialmente útil para analizar la dinámica del tránsito en distintos puntos de la ciudad, evaluar el uso de medios de pago y planificar mejoras en infraestructura vial o sistemas de control de tráfico.

#encontrar fecha minima y maxima
fecha_min = datos['fecha'].min()
fecha_max = datos['fecha'].max()
print(f"Fecha mínima: {fecha_min}")
print(f"Fecha máxima: {fecha_max}")

# Crea un dataframe que contenga las fechas desde la fecha mínima hasta la fecha máxima
df_fechas = pd.date_range(start=fecha_min, end=fecha_max, freq='D')

# En df_fechas agregar las columnas
df_fechas = pd.DataFrame(df_fechas, columns=['FECHA'])
df_fechas['id_fecha'] = df_fechas['FECHA'].dt.strftime('%Y%m%d')
df_fechas['DIA_SEMANA'] = df_fechas['FECHA'].dt.dayofweek
df_fechas['NOMBRE_DIA_SEMANA'] = df_fechas['FECHA'].dt.day_name()
df_fechas['DIA_MES'] = df_fechas['FECHA'].dt.day
df_fechas['DIA_ANIO'] = df_fechas['FECHA'].dt.dayofyear
df_fechas['NOMBRE_MES'] = df_fechas['FECHA'].dt.month_name()
df_fechas['MES_ANIO'] = df_fechas['FECHA'].dt.month
df_fechas['SEMANA_ANIO'] = df_fechas['FECHA'].dt.isocalendar().week
df_fechas['ANIO'] = df_fechas['FECHA'].dt.year
# ADICIONAL
df_fechas['BIMESTRE'] = ((df_fechas['FECHA'].dt.month - 1) // 2) + 1
df_fechas['TRIMESTRE'] = ((df_fechas['FECHA'].dt.month - 1) // 3) + 1
df_fechas['SEMESTRE'] = ((df_fechas['FECHA'].dt.month - 1) // 6) + 1
# FERIADOS
import holidays
pe = holidays.PE()
df_fechas['FERIADO'] = df_fechas['FECHA'].apply(lambda x: 1 if x in pe else 0)

# Intercambiar el orden de las 2 primeras columnas de df_fechas
df_fechas = df_fechas[['id_fecha', 'FECHA'] + [col for col in df_fechas.columns if col not in ['id_fecha', 'FECHA']]]
df_fechas

Fecha mínima: 2017-01-01 00:00:00
Fecha máxima: 2020-12-31 00:00:00


sql_create_tiempo = """
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='dim_tiempo' AND xtype='U')
BEGIN
    CREATE TABLE dbo.dim_tiempo (
        id_fecha VARCHAR(8) PRIMARY KEY,
        FECHA DATE,
        DIA_SEMANA INT,
        NOMBRE_DIA_SEMANA VARCHAR(20),
        DIA_MES INT,
        DIA_ANIO INT,
        NOMBRE_MES VARCHAR(20),
        MES_ANIO INT,
        SEMANA_ANIO INT,
        ANIO INT,
        BIMESTRE INT,
        TRIMESTRE INT,
        SEMESTRE INT,
        FERIADO BIT
    )
END
"""
crear_tabla_sql(cnn, sql_create_tiempo)

response, msg = upload_df_to_blob(df=df_fechas, table_name = 'dbo.dim_tiempo', connection_str = cnn)
print(f"Response: {response}, Message: {msg}")

# DataFrame con las columnas id_hora (incremental desde 1), HORA (de 0 a 23 tipo Time), PERIDO (AM/PM) y HORA_INT (de 0 a 23)
df_horas = pd.DataFrame({
    'id_hora': range(1, 25),
    'HORA': pd.date_range('00:00', '23:00', freq='H').time,
    'PERIODO': ['AM'] * 12 + ['PM'] * 12,
    'HORA_INT': list(range(0, 24))
})
df_horas.head(30)

sql_create_hora = """
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='dim_hora' AND xtype='U')
BEGIN
    CREATE TABLE dbo.dim_hora (
        id_hora INT PRIMARY KEY,
        HORA TIME,
        PERIODO VARCHAR(2),
        HORA_INT INT
    )
END
"""
crear_tabla_sql(cnn, sql_create_hora)

response, msg = upload_df_to_blob(df=df_horas, table_name = 'dbo.dim_hora', connection_str = cnn)
print(f"Response: {response}, Message: {msg}")

#Dataframe df_sentido con las columnas id_sentido (incremental desde 1), SENTIDO (valores unicos de la columna 'sentido' del dataframe 'datos')
df_sentido = pd.DataFrame({
    'id_sentido': range(1, len(datos['sentido'].unique()) + 1),
    'SENTIDO': datos['sentido'].unique()
})
df_sentido.head() 

sql_create_sentido = """
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='dim_sentido' AND xtype='U')
BEGIN
    CREATE TABLE dbo.dim_sentido (
        id_sentido INT PRIMARY KEY,
        SENTIDO VARCHAR(50)
    )
END
"""
crear_tabla_sql(cnn, sql_create_sentido)

response, msg = upload_df_to_blob(df=df_sentido, table_name = 'dbo.dim_sentido', connection_str = cnn)
print(f"Response: {response}, Message: {msg}")

#Dataframe df_tipo_vehiculo con las columnas id_tipo_vehiculo (incremental desde 1), TIPO_VEHICULO (valores unicos de la columna 'tipo_vehiculo' del dataframe 'datos')
df_tipo_vehiculo = pd.DataFrame({
    'id_tipo_vehiculo': range(1, len(datos['tipo_vehiculo'].unique()) + 1),
    'TIPO_VEHICULO': datos['tipo_vehiculo'].unique()
})
df_tipo_vehiculo.head(20)

<!-- id_tipo_vehiculo	TIPO_VEHICULO
0	1	Liviano
1	2	Pesado
2	3	Auto
3	4	Auto con trailer
4	5	Moto
5	6	N/D
6	7	Pago Doble Auto
7	8	Pesados 2 Ejes
8	9	Pesados 3 Ejes
9	10	Pesados 4 Ejes
10	11	Pesados 5 Ejes
11	12	Pesados 6 Ejes
12	13	Pago doble Moto
13	14	Pago Doble Auto con trailer / Pesado 2 Ejes -->

sql_create_tipo_vehiculo = """
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='dim_tipo_vehiculo' AND xtype='U')
BEGIN
    CREATE TABLE dbo.dim_tipo_vehiculo (
        id_tipo_vehiculo INT PRIMARY KEY,
        TIPO_VEHICULO VARCHAR(50)
    )
END
"""
crear_tabla_sql(cnn, sql_create_tipo_vehiculo)

response, msg = upload_df_to_blob(df=df_tipo_vehiculo, table_name = 'dbo.dim_tipo_vehiculo', connection_str = cnn)
print(f"Response: {response}, Message: {msg}")

#Dataframe df_forma_pago con las columnas id_forma_pago , FORMA_PAGO (valores unicos de la columna 'forma_pago' del dataframe 'datos')
df_forma_pago = pd.DataFrame({
    'id_forma_pago': range(1, len(datos['forma_pago'].unique()) + 1),
    'FORMA_PAGO': datos['forma_pago'].unique()
})
df_forma_pago.head(20)

<!-- id_forma_pago	FORMA_PAGO
0	1	no cobrado
1	2	efectivo
2	3	telepase
3	4	exento
4	5	tarjeta discapacidad
5	6	infraccion
6	7	monedero
7	8	rec.deuda
8	9	tarjeta magnÃ©tica
9	10	tag
10	11	violaciÃ³n
11	12	cpp
12	13	mercado pago -->

sql_create_forma_pago = """
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='dim_forma_pago' AND xtype='U')
BEGIN
    CREATE TABLE dbo.dim_forma_pago (
        id_forma_pago INT PRIMARY KEY,
        FORMA_PAGO VARCHAR(50)
    )
END
"""
crear_tabla_sql(cnn, sql_create_forma_pago)

response, msg = upload_df_to_blob(df=df_forma_pago, table_name = 'dbo.dim_forma_pago', connection_str = cnn)
print(f"Response: {response}, Message: {msg}")

# DataFrame de la tabla de hechos
datos['hora_inicio'] = datos['hora_inicio'].astype(int)
df_fact_flujo = pd.DataFrame({
    'id_flujo': range(1, len(datos) + 1),
    'CANTIDAD_PASOS': datos['cantidad_pasos'],
    'fk_hora': datos['hora_inicio'].map(df_horas.set_index('HORA_INT')['id_hora'].to_dict()),
    'fk_sentido': datos['sentido'].map(df_sentido.set_index('SENTIDO')['id_sentido'].to_dict()),
    'fk_tipo_vehiculo': datos['tipo_vehiculo'].map(df_tipo_vehiculo.set_index('TIPO_VEHICULO')['id_tipo_vehiculo'].to_dict()),
    'fk_forma_pago': datos['forma_pago'].map(df_forma_pago.set_index('FORMA_PAGO')['id_forma_pago'].to_dict()),
    'fk_tiempo': datos['fecha'].dt.date.map(df_fechas.set_index('FECHA')['id_fecha'].to_dict())
})
df_fact_flujo.head(20)

print(datos['fecha'].head())
print(datos['fecha'].dtypes)

sql_create_fact_flujo = """
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='fact_flujo' AND xtype='U')
BEGIN
    CREATE TABLE dbo.fact_flujo (
        id_flujo INT PRIMARY KEY,
        CANTIDAD_PASOS INT,
        fk_hora INT,
        fk_sentido INT,
        fk_tipo_vehiculo INT,
        fk_forma_pago INT,
        fk_tiempo VARCHAR(8),
        FOREIGN KEY (fk_hora) REFERENCES dim_hora(id_hora),
        FOREIGN KEY (fk_sentido) REFERENCES dim_sentido(id_sentido),
        FOREIGN KEY (fk_tipo_vehiculo) REFERENCES dim_tipo_vehiculo(id_tipo_vehiculo),
        FOREIGN KEY (fk_forma_pago) REFERENCES dim_forma_pago(id_forma_pago),
        FOREIGN KEY (fk_tiempo) REFERENCES dim_tiempo(id_fecha)
    )
END
"""
crear_tabla_sql(cnn, sql_create_fact_flujo)

response, msg = upload_df_to_blob_with_steroids(df=df_fact_flujo, table_name = 'dbo.fact_flujo', connection_str = cnn)
print(f"Response: {response}, Message: {msg}")

import matplotlib.pyplot as plt
# Gráfico de barras del promedio de flujo vehicular por día de la semana
# ==============================================================================    

# Convertir 'cantidad_pasos' a numérico
datos['cantidad_pasos'] = pd.to_numeric(datos['cantidad_pasos'], errors='coerce')

# Agrupar por día de la semana y calcular el promedio
datos['DIA_SEMANA'] = datos['fecha'].dt.dayofweek  # 0: lunes, ..., 6: domingo
promedio_dia = datos.groupby('DIA_SEMANA')['cantidad_pasos'].mean()

# Nombres de días
dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

# Gráfico de barras
plt.figure(figsize=(10, 5))
plt.bar(dias_semana, promedio_dia, color='green')
plt.title('Promedio de Flujo Vehicular por Día de la Semana')
plt.xlabel('Día de la Semana')
plt.ylabel('Promedio de Vehículos')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Query SQL para extraer datos
_, datos = get_blob_to_df(table_name = 'dbo.vw_FACT_TRAFFIC_FLOW_FORECASTING', connection_str = cnn)

# weekday a objeto
datos['weekday'] = datos['weekday'].astype('object')
datos['hour'] = datos['hour'].astype('object')

# Agrupar los datos por fecha, día de la semana y hora, y sumar el tráfico
# ==============================================================================
datos_grouped_by_datetime = datos.groupby(['date_time', 'weekday', 'hour']).agg({'traffic_count': 'sum'}).reset_index()

# Preprocesamiento de datos
# ==============================================================================
# Convertir las columnas 'fecha' y 'hora_inicio' a objetos date_time
datos_grouped_by_datetime['date_time'] = pd.to_datetime(datos_grouped_by_datetime['date_time'], format='%Y-%m-%d %H:%M:%S')

# Establecer 'date_time' como índice del DataFrame
datos_grouped_by_datetime = datos_grouped_by_datetime.set_index('date_time')

# Ajustar la frecuencia del DataFrame a intervalos horarios
datos_grouped_by_datetime = datos_grouped_by_datetime.asfreq('H')

# Ordenar el DataFrame por el índice (date_time)
datos_grouped_by_datetime = datos_grouped_by_datetime.sort_index()

# Asignar el DataFrame preprocesado a la variable 'datos'
datos = datos_grouped_by_datetime

# Separación datos train-val-test
# ==============================================================================
fin_train = '2019-06-30 23:59:00'
fin_validacion = '2020-06-30 23:59:00'
datos_train = datos.loc[: fin_train, :]
datos_val   = datos.loc[fin_train:fin_validacion, :]
datos_test  = datos.loc[fin_validacion:, :]

print(f"Fechas train      : {datos_train.index.min()} --- {datos_train.index.max()}  (n={len(datos_train)})")
print(f"Fechas validacion : {datos_val.index.min()} --- {datos_val.index.max()}  (n={len(datos_val)})")
print(f"Fechas test       : {datos_test.index.min()} --- {datos_test.index.max()}  (n={len(datos_test)})")

print(datos_train['traffic_count'].dtype)
print(datos_val['traffic_count'].dtype)
print(datos_test['traffic_count'].dtype)

# Contar nulos
print(f"Datos train nulos: {datos_train.isnull().sum()}")
print(f"Datos validacion nulos: {datos_val.isnull().sum()}") 
print(f"Datos test nulos: {datos_test.isnull().sum()}")

<!-- Datos train nulos: weekday          11
hour             11
traffic_count    11
dtype: int64
Datos validacion nulos: weekday          0
hour             0
traffic_count    0
dtype: int64
Datos test nulos: weekday          0
hour             0
traffic_count    0
dtype: int64 -->

# Mostrar registros con nulos
print("Datos train con nulos:")
print(datos_train[datos_train.isnull().any(axis=1)])
print("Datos validacion con nulos:")
print(datos_val[datos_val.isnull().any(axis=1)])
print("Datos test con nulos:")
print(datos_test[datos_test.isnull().any(axis=1)])

# Rellenar nulos en datos_train
datos_train['traffic_count'] = datos_train['traffic_count'].fillna(
    datos_train.groupby([datos_train.index.hour])['traffic_count'].transform('mean')
)

# Rellenar nulos en la columna 'weekday' de datos_train
datos_train['weekday'] = datos_train['weekday'].fillna(
    datos_train.groupby([datos_train.index.hour])['weekday'].transform('mean')
)

# Rellenar nulos en la columna 'weekday' de datos_train
datos_train['hour'] = datos_train['hour'].fillna(
    datos_train.groupby([datos_train.index.hour])['hour'].transform('mean')
)

print(f"Datos train nulos: {datos_train.isnull().sum()}")

# Gráfico serie temporal
# ==============================================================================
fig, ax = plt.subplots(figsize=(11, 4))
datos_train['traffic_count'].plot(ax=ax, label='entrenamiento')
datos_val['traffic_count'].plot(ax=ax, label='validación')
datos_test['traffic_count'].plot(ax=ax, label='test')
ax.set_title('Número de vehículos')
ax.legend();

# Gráfico serie temporal con zoom
# ==============================================================================
zoom = ('2018-11-20 00:00:00','2018-12-05 00:00:00')

fig = plt.figure(figsize=(11, 6))
grid = plt.GridSpec(nrows=8, ncols=1, hspace=0.6, wspace=0)

main_ax = fig.add_subplot(grid[1:3, :])
zoom_ax = fig.add_subplot(grid[5:, :])

datos_train['traffic_count'].plot(ax=main_ax, label='entrenamiento', alpha=0.5)
datos_val['traffic_count'].plot(ax=main_ax, label='validación', alpha=0.5)
datos_test['traffic_count'].plot(ax=main_ax, label='test', alpha=0.5)
min_y = min(datos['traffic_count'])
max_y = max(datos['traffic_count'])
main_ax.fill_between(zoom, min_y, max_y, facecolor='blue', alpha=0.5, zorder=0)
main_ax.set_xlabel('')
main_ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.8))

datos.loc[zoom[0]: zoom[1]]['traffic_count'].plot(ax=zoom_ax, color='blue', linewidth=2)

main_ax.set_title(f'Número de vehículos: {datos.index.min()}, {datos.index.max()}', fontsize=14)
zoom_ax.set_title(f'Número de vehículos: {zoom}', fontsize=14)
plt.subplots_adjust(hspace=1)

# Gráfico interactivo serie temporal
# ==============================================================================
try:
    datos.loc[:fin_train, 'particion'] = 'entrenamiento'
    datos.loc[fin_train:fin_validacion, 'particion'] = 'validación'
    datos.loc[fin_validacion:, 'particion'] = 'test'

    fig = px.line(
        data_frame = datos.reset_index(),
        x = 'date_time',
        y = 'traffic_count',
        color = 'particion',
        title = 'Número de vehículos',
        width = 900,
        height = 500
    )

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    # Intento principal con renderizador forzado
    try:
        fig.show(renderer="notebook")
    except:
        # Fallback 1: Browser
        import webbrowser
        fig.write_html("temp_plot.html")
        webbrowser.open("temp_plot.html")
        
        # Fallback 2: IFrame
        from IPython.display import IFrame
        display(IFrame(src="temp_plot.html", width=1000, height=600))

finally:
    # Limpieza
    datos = datos.drop(columns='particion', errors='ignore')

# Gráfico boxplot para estacionalidad diaria
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 3))
promedio_dia_hora = datos.groupby(["weekday", "hour"])["traffic_count"].mean()
q25_dia_hora = datos.groupby(["weekday", "hour"])["traffic_count"].quantile(0.25)
q75_dia_hora = datos.groupby(["weekday", "hour"])["traffic_count"].quantile(0.75)

promedio_dia_hora.plot(ax=ax, label='promedio')
q25_dia_hora.plot(ax=ax, linestyle='dashed', color='gray', label='')
q75_dia_hora.plot(ax=ax, linestyle='dashed', color='gray', label='cuantil 25 y 75')


ax.set(
    title="Promedio de autos a largo de la semana",
    xticks=[i * 24 for i in range(7)],
    xticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    xlabel="Día y hora de la semana",
    ylabel="Número de autos"
)

ax.legend();

# Gráfico autocorrelación
# ==============================================================================
fig, ax = plt.subplots(figsize=(13, 7))
plot_acf(
  datos['traffic_count'].fillna(
    datos.groupby([datos.index.hour])['traffic_count'].transform('mean')
  )  # Rellenar nulos hacia adelante
  , ax=ax
  , lags=72
  )
#plot_acf(data_a, lags=90)
plt.show()

# Gráfico autocorrelación parcial
# ==============================================================================
fig, ax = plt.subplots(figsize=(13, 7))
plot_pacf(
  datos['traffic_count'].fillna(
    datos.groupby([datos.index.hour])['traffic_count'].transform('mean')
  ) 
  , ax=ax, lags=72
  , method='ywm'
)
plt.show()

# Crear forecaster
forecaster = ForecasterAutoreg(
                regressor = XGBRegressor(random_state=123),
                lags = 24
             )

forecaster

# Grid search de hiperparámetros
# Hiperparámetros del regresor
param_grid = {
    'n_estimators': [100, 500],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1]
}

# Rellenar valores nulos con el promedio por hora
datos.loc[:fin_validacion, 'traffic_count'] = datos.loc[:fin_validacion, 'traffic_count'].fillna(
    datos.loc[:fin_validacion, 'traffic_count'].groupby(datos.loc[:fin_validacion].index.hour).transform('mean')
)

# Lags utilizados como predictores
lags_grid = [24, 48, 72, [1, 2, 3, 23, 24, 25, 71, 72, 73]]

# Grid search de hiperparámetros
resultados_grid = grid_search_forecaster(
        forecaster         = forecaster,
        y                  = datos.loc[:fin_validacion, 'traffic_count'], # conjunto de train y validación
        param_grid         = param_grid,
        lags_grid          = lags_grid,
        steps              = 36,
        refit              = False,
        metric             = 'mean_squared_error',
        initial_train_size = int(len(datos_train)), # El modelo se entrena con los datos de entrenamiento
        return_best        = True,
        verbose            = False
)

# Mostrar resultados del grid search
len(datos.loc[:fin_validacion])

# Backtesting
metric, predicciones = backtesting_forecaster(
    forecaster = forecaster,
    y          = datos['traffic_count'],
    initial_train_size = len(datos.loc[:fin_validacion]),
    steps      = 60,
    refit      = False,
    metric     = 'mean_squared_error',
    verbose    = False
)

print(f"Error de backtest: {metric}")

# Guardar predicciones test
# ==============================================================================
def guardar_resultados(datos, modelo):
    fecha_ejecucion = datetime.now()
    resultados = pd.DataFrame({
        'DATE_TIME': datos.index,
        'PREDICCION': datos['pred'],
        'MODELO': modelo,
        'EXECUTION_DATE': fecha_ejecucion
    })
    return resultados

def upload_df_to_blob(df: pd.DataFrame, table_name: str, connection_str: str):
    try:
        df = df.where(pd.notnull(df), None)
        with pyodbc.connect(connection_str, autocommit=True) as conn:
            cursor = conn.cursor()
            columns = ','.join(df.columns)
            cmd_insert = f"INSERT INTO {table_name} ({columns}) VALUES ({','.join(['?'] * len(df.columns))})"
            cursor.fast_executemany = True
            cursor.executemany(cmd_insert, df.values.tolist())
        return True, ""
    except Exception as e:
        return False, str(e)

def guardar_pronostico(datos, modelo):
    # Obtener la fecha de ejecución actual
    fecha_ejecucion = datetime.now()
    
    # Crear un nuevo DataFrame con las columnas requeridas
    resultados = pd.DataFrame({
        'DATE_TIME': datos.index,        # Fecha (index)
        'PREDICCION': datos['pred'],       # Columna pred
        'MODELO': modelo            # Nombre del modelo
    })
    
    return resultados

# Guardar predicciones test
# ==============================================================================
rs1 = guardar_resultados(predicciones, 'MODEL_FORECASTER')
upload_df_to_blob(df = rs1, table_name = 'PREDICCION_TRAFFIC_FLOW', connection_str = cnn)

# Guardar pronóstico
forecaster.fit(y=datos_train['traffic_count'])

# Realizar predicción para los próximos 2160 pasos (90 días con frecuencia horaria)
data1 = forecaster.predict(steps=2160, last_window=datos['traffic_count'])
data1 = pd.DataFrame(data1)
data1

# Guardar forecasting
rs1 = guardar_pronostico(data1, 'MODEL_FORECASTER')
upload_df_to_blob(df = rs1, table_name = 'FORECASTING_TRAFFIC_FLOW', connection_str = cnn)

# Gráfico estatico predicciones test
fig, ax = plt.subplots(figsize=(11, 4))
datos_test['traffic_count'].plot(ax=ax, label='test')
predicciones['pred'].plot(ax=ax, label='predicciones')
ax.legend();

# Gráfico interactivo serie temporal
datos_plot = pd.DataFrame({
                'test': datos_test['traffic_count'],
                'prediccion': predicciones['pred'],

                 })
datos_plot.index.name = 'date_time'

fig = px.line(
    data_frame = datos_plot.reset_index(),
    x      = 'date_time',
    y      = datos_plot.columns,
    title  = 'Número de autos: test vs predicciones',
    width  = 900,
    height = 500
)

fig.update_xaxes(rangeslider_visible=True)
fig.show()

# Query SQL para extraer datos
_, datos = get_blob_to_df(table_name = 'dbo.vw_FACT_TRAFFIC_FLOW_FORECASTING', connection_str = cnn)

import numpy as np

# Crear columnas precomputadas de forma vectorizada
for sentido in datos['sentido'].unique():
    datos[f'Sentido_{sentido}'] = np.where(datos['sentido'] == sentido, datos['traffic_count'], 0)

for tipo_vehiculo in datos['tipo_vehiculo'].unique():
    datos[f'TV_{tipo_vehiculo}'] = np.where(datos['tipo_vehiculo'] == tipo_vehiculo, datos['traffic_count'], 0)

for forma_pago in datos['forma_pago'].unique():
    datos[f'Forma_Pago_{forma_pago}'] = np.where(datos['forma_pago'] == forma_pago, datos['traffic_count'], 0)

# Agrupar y sumar
datos_pivot = datos.groupby(['date_time', 'hour', 'weekday', 'month', 'period', 'TIPO_CONJUNTO']).sum()

# Eliminamos columnas originales
datos_pivot = datos_pivot.drop(columns=['sentido', 'tipo_vehiculo', 'forma_pago'])

# Crear columnas de porcentajes de forma vectorizada
for col in [col for col in datos_pivot.columns if col.startswith('Sentido_') or col.startswith('TV_') or col.startswith('Forma_Pago_')]:
    datos_pivot[f'{col}_pct'] = datos_pivot[col] / datos_pivot['traffic_count']

# Resetear el índice para convertirlo en un DataFrame normal
datos_pivot = datos_pivot.reset_index()

# Index a datetime
datos_pivot['date_time'] = pd.to_datetime(datos_pivot['date_time'], format='%Y-%m-%d %H:%M:%S')

# Establecer 'date_time' como índice del DataFrame
datos_pivot = datos_pivot.set_index('date_time')

# Ajustar la frecuencia del DataFrame a intervalos horarios
datos_pivot = datos_pivot.asfreq('H')

# Ordenar el DataFrame por el índice (date_time)
datos_pivot = datos_pivot.sort_index()
# Mostrar el DataFrame resultante
datos_pivot

# Rellenar nulos en la columna 'traffic_count' de datos_pivot
datos_pivot['traffic_count'] = datos_pivot['traffic_count'].fillna(
    datos_pivot.groupby([datos_pivot.index.hour])['traffic_count'].transform('mean')
)

# Rellenar nulos en la columna 'weekday' de datos_pivot
datos_pivot['weekday'] = datos_pivot['weekday'].fillna(
    datos_pivot.groupby([datos_pivot.index.hour])['weekday'].transform('mean')
)

# Rellenar nulos en la columna 'hour' de datos_pivot
datos_pivot['hour'] = datos_pivot['hour'].fillna(
    datos_pivot.groupby([datos_pivot.index.hour])['hour'].transform('mean')
)

# Cambio de las variables categóricas a tipo category
# ==============================================================================
datos['period'] = datos['period'].astype('category')
datos['month']   = datos['month'].astype('category')
datos['weekday'] = datos['weekday'].astype('category')

# Transformación seno-coseno de la variable hora
# ==============================================================================
datos['hour_sin'] = np.sin(datos['hour'] / 23 * 2 * np.pi)
datos['hour_cos'] = np.cos(datos['hour'] / 23 * 2 * np.pi)

# Representación de la transformación seno-coseno de la variable hora
# ==============================================================================
fig, ax = plt.subplots(figsize=(4, 4))
sp = ax.scatter(datos["hour_sin"], datos["hour_cos"], c=datos["hour"])
ax.set(
    xlabel="sin(hour)",
    ylabel="cos(hour)",
)
_ = fig.colorbar(sp)

datos = datos.drop(columns='hour')

# One hot encoding
# ==============================================================================
datos = pd.get_dummies(datos, columns=['period', 'month', 'weekday'])
datos.head(3)

# Se seleccionan todas las variables exógenas, incluidas las obtenidas al hacer
# el one hot encoding.
variables_exogenas = [column for column in datos.columns
                      if column.startswith(('period', 'month', 'hour', 'weekday'))]
print(variables_exogenas)

['hour_sin', 'hour_cos', 'period_AM', 'period_PM', 'month_1.0', 'month_2.0', 'month_3.0', 'month_4.0', 'month_5.0', 'month_6.0', 'month_7.0', 'month_8.0', 'month_9.0', 'month_10.0', 'month_11.0', 'month_12.0', 'weekday_0.0', 'weekday_1.0', 'weekday_2.0', 'weekday_2.9958875942426317', 'weekday_2.995890410958904', 'weekday_2.9972602739726026', 'weekday_2.997945205479452', 'weekday_3.0', 'weekday_4.0', 'weekday_5.0', 'weekday_6.0']

# Como los datos han sido modificados, se repite el reparto en train, val y test.
fin_train = '2019-06-30 23:59:00'
fin_validacion = '2020-06-30 23:59:00'
datos_train = datos.loc[: fin_train, :]
datos_val   = datos.loc[fin_train:fin_validacion, :]
datos_test  = datos.loc[fin_validacion:, :]

print(f"Fechas train      : {datos_train.index.min()} --- {datos_train.index.max()}  (n={len(datos_train)})")
print(f"Fechas validacion : {datos_val.index.min()} --- {datos_val.index.max()}  (n={len(datos_val)})")
print(f"Fechas test       : {datos_test.index.min()} --- {datos_test.index.max()}  (n={len(datos_test)})")

# Crear forecaster
forecaster = ForecasterAutoreg(
                regressor = XGBRegressor(random_state=123),
                lags = 24
             )

# Grid search de hiperparámetros
# Hiperparámetros del regresor
param_grid = {
    'n_estimators': [100, 500],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1]
}

# Lags utilizados como predictores
lags_grid = [72, [1, 2, 3, 23, 24, 25, 71, 72, 73]]

resultados_grid = grid_search_forecaster(
                        forecaster         = forecaster,
                        y                  = datos.loc[:fin_validacion, 'traffic_count'],
                        exog               = datos.loc[:fin_validacion, variables_exogenas],
                        param_grid         = param_grid,
                        lags_grid          = lags_grid,
                        steps              = 36,
                        refit              = False,
                        metric             = 'mean_squared_error',
                        initial_train_size = int(len(datos_train)),
                        return_best        = True,
                        verbose            = False
                   )

# Backtesting
# ==============================================================================
metric, predicciones = backtesting_forecaster(
    forecaster         = forecaster,
    y                  = datos['traffic_count'],
    exog               = datos[variables_exogenas],
    initial_train_size = len(datos.loc[:fin_validacion]),
    steps              = 36,
    refit              = False,
    metric             = 'mean_squared_error',
    verbose            = False
)

print(f"Error de backtest: {metric}")

# Guardar predicciones test
rs1 = guardar_resultados(predicciones, 'MODEL_EXOGENEAS')
upload_df_to_blob(df = rs1, table_name = 'PREDICCION_TRAFFIC_FLOW', connection_str = cnn)

# Gráfico estatico predicciones test
# ==============================================================================
fig, ax = plt.subplots(figsize=(11, 4))
datos_test['traffic_count'].plot(ax=ax, label='test')
predicciones['pred'].plot(ax=ax, label='predicciones')
ax.legend();

# Gráfico interactivo serie temporal
# ==============================================================================
datos_plot = pd.DataFrame({
                'test': datos_test['traffic_count'],
                'prediccion': predicciones['pred'],

                 })
datos_plot.index.name = 'date_time'

fig = px.line(
    data_frame = datos_plot.reset_index(),
    x      = 'date_time',
    y      = datos_plot.columns,
    title  = 'Número de autos: test vs predicciones',
    width  = 900,
    height = 500
)

fig.update_xaxes(rangeslider_visible=True)
fig.show()

# Importancia predictores
# ==============================================================================
importancia = forecaster.regressor.get_booster().get_score(importance_type='weight')
importancia

# Convertir la importancia a un DataFrame
last_window = datos['traffic_count'][-forecaster.window_size:]

# Asegurarse de que las variables exógenas estén alineadas temporalmente
exog_future = datos[variables_exogenas].iloc[-2160:]
exog_future.index = pd.date_range(
    start=last_window.index[-1] + pd.Timedelta(hours=1),  # Comienza justo después de last_window
    periods=2160,
    freq='H'  # Ajusta la frecuencia según tus datos (por ejemplo, 'H' para horas)
)

predicciones = forecaster.predict(
    steps=2160,
    last_window=last_window,
    exog=exog_future
)
predicciones = pd.DataFrame(predicciones)
predicciones

# Save forecasting
# ==============================================================================
rs1 = guardar_pronostico(predicciones, 'MODEL_EXOGENEAS')
upload_df_to_blob(df = rs1, table_name = 'FORECASTING_TRAFFIC_FLOW', connection_str = cnn)

# Crear forecaster
forecaster = ForecasterAutoreg(
                regressor = LGBMRegressor(random_state=123, device='gpu'),
                lags = 24
             )

forecaster

print(datos[variables_exogenas].dtypes)

# Para variables categóricas (one-hot encoded)
datos[variables_exogenas] = datos[variables_exogenas].astype(float)

# O si son categóricas no numéricas:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in datos[variables_exogenas].select_dtypes(include=['object']).columns:
    datos[col] = le.fit_transform(datos[col].astype(str))

# Grid search de hiperparámetros
param_grid = {
    'n_estimators': [100, 500],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1]
}

lags_grid = [24, 48, 72, [1, 2, 3, 23, 24, 25, 71, 72, 73]]

resultados_grid = grid_search_forecaster(
                        forecaster         = forecaster,
                        y                  = datos.loc[:fin_validacion, 'traffic_count'],
                        exog               = datos.loc[:fin_validacion, variables_exogenas],
                        param_grid         = param_grid,
                        lags_grid          = lags_grid,
                        steps              = 36,
                        refit              = False,
                        metric             = 'mean_squared_error',
                        initial_train_size = int(len(datos_train)),
                        return_best        = True,
                        verbose            = False
                   )

# Backtesting
metric, predicciones = backtesting_forecaster(
    forecaster         = forecaster,
    y                  = datos['traffic_count'],
    exog               = datos[variables_exogenas],
    initial_train_size = len(datos.loc[:fin_validacion]),
    steps              = 36,
    refit              = False,
    metric             = 'mean_squared_error',
    verbose            = False
)

print(f"Error de backtest: {metric}")

# Gráfico estatico predicciones test
# ==============================================================================
fig, ax = plt.subplots(figsize=(11, 4))
datos_test['traffic_count'].plot(ax=ax, label='test')
predicciones['pred'].plot(ax=ax, label='predicciones')
ax.legend();

# Gráfico interactivo serie temporal
# ==============================================================================
datos_plot = pd.DataFrame({
                'test': datos_test['traffic_count'],
                'prediccion': predicciones['pred'],

                 })
datos_plot.index.name = 'date_time'

fig = px.line(
    data_frame = datos_plot.reset_index(),
    x      = 'date_time',
    y      = datos_plot.columns,
    title  = 'Número de autos: test vs predicciones',
    width  = 900,
    height = 500
)

fig.update_xaxes(rangeslider_visible=True)
fig.show()

# Save predicciones test
# ==============================================================================
rs1 = guardar_resultados(predicciones, 'MODEL_EXOGENEAS_LGBM')
upload_df_to_blob(df = rs1, table_name = 'PREDICCION_TRAFFIC_FLOW', connection_str = cnn)

# Importancia predictores
# ==============================================================================
importancia = forecaster.regressor.feature_importances_
importancia

last_window = datos['traffic_count'][-forecaster.window_size:]

# Asegurarse de que las variables exógenas estén alineadas temporalmente
exog_future = datos[variables_exogenas].iloc[-2160:]
exog_future.index = pd.date_range(
    start=last_window.index[-1] + pd.Timedelta(hours=1),  # Comienza justo después de last_window
    periods=2160,
    freq='H'  # Ajusta la frecuencia según tus datos (por ejemplo, 'H' para horas)
)

predicciones = forecaster.predict(
    steps=2160,
    last_window=last_window,
    exog=exog_future
)
predicciones = pd.DataFrame(predicciones)
predicciones

# Save forecasting
# ==============================================================================
rs1 = guardar_pronostico(predicciones, 'MODEL_EXOGENEAS_LGBM')
upload_df_to_blob(df = rs1, table_name = 'FORECASTING_TRAFFIC_FLOW', connection_str = cnn)

# Crear forecaster
# ==============================================================================
forecaster = ForecasterAutoreg(
                regressor = CatBoostRegressor(random_state=123, silent=True, task_type='GPU'),
                lags = 24
             )

forecaster

# Grid search de hiperparámetros
# ==============================================================================
# Hiperparámetros del regresor
from catboost import CatBoostRegressor

 

param_grid = {
    'n_estimators': [100, 500],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1]
}

# Lags utilizados como predictores
lags_grid = [72, [1, 2, 3, 23, 24, 25, 71, 72, 73]]

resultados_grid = grid_search_forecaster(
                        forecaster         = forecaster,
                        y                  = datos.loc[:fin_validacion, 'traffic_count'],
                        exog               = datos.loc[:fin_validacion, variables_exogenas],
                        param_grid         = param_grid,
                        lags_grid          = lags_grid,
                        steps              = 36,
                        refit              = False,
                        metric             = 'mean_squared_error',
                        initial_train_size = int(len(datos_train)),
                        return_best        = True,
                        verbose            = False
                   )

# Backtesting
# ==============================================================================
metric, predicciones = backtesting_forecaster(
    forecaster         = forecaster,
    y                  = datos['traffic_count'],
    exog               = datos[variables_exogenas],
    initial_train_size = len(datos.loc[:fin_validacion]),
    steps              = 36,
    refit              = False,
    metric             = 'mean_squared_error',
    verbose            = False
)

print(f"Error de backtest: {metric}")

# Gráfico estatico predicciones test
# ==============================================================================
fig, ax = plt.subplots(figsize=(11, 4))
datos_test['traffic_count'].plot(ax=ax, label='test')
predicciones['pred'].plot(ax=ax, label='predicciones')
ax.legend();

# Gráfico interactivo serie temporal
# ==============================================================================
datos_plot = pd.DataFrame({
                'test': datos_test['traffic_count'],
                'prediccion': predicciones['pred'],

                 })
datos_plot.index.name = 'date_time'

fig = px.line(
    data_frame = datos_plot.reset_index(),
    x      = 'date_time',
    y      = datos_plot.columns,
    title  = 'Número de autos: test vs predicciones',
    width  = 900,
    height = 500
)

fig.update_xaxes(rangeslider_visible=True)
fig.show()

# Save predicciones test
# ==============================================================================
rs1 = guardar_resultados(predicciones, 'MODEL_EXOGENEAS_CatBoost')
upload_df_to_blob(df = rs1, table_name = 'PREDICCION_TRAFFIC_FLOW', connection_str = cnn)

last_window = datos['traffic_count'][-forecaster.window_size:]

# Asegurarse de que las variables exógenas estén alineadas temporalmente
exog_future = datos[variables_exogenas].iloc[-2160:]
exog_future.index = pd.date_range(
    start=last_window.index[-1] + pd.Timedelta(hours=1),  # Comienza justo después de last_window
    periods=2160,
    freq='H'  # Ajusta la frecuencia según tus datos (por ejemplo, 'H' para horas)
)

predicciones = forecaster.predict(
    steps=2160,
    last_window=last_window,
    exog=exog_future
)
predicciones = pd.DataFrame(predicciones)
predicciones

# Save forecasting
# ==============================================================================
rs1 = guardar_pronostico(predicciones, 'MODEL_EXOGENEAS_CatBoost')
upload_df_to_blob(df = rs1, table_name = 'FORECASTING_TRAFFIC_FLOW', connection_str = cnn)

pip install keras

from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Convertir los datos a numpy array
train_data = datos_train['traffic_count'].values.reshape(-1, 1)
val_data = datos_val['traffic_count'].values.reshape(-1, 1)
test_data = datos_test['traffic_count'].values.reshape(-1, 1)

# Escalar los datos entre 0 y 1
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data)
val_data_scaled = scaler.transform(val_data)
test_data_scaled = scaler.transform(test_data)

# Función para crear secuencias temporales
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# Definir el número de pasos temporales
time_steps = 24

# Crear secuencias temporales para entrenamiento, validación y prueba
X_train, y_train = create_sequences(train_data_scaled, time_steps)
X_val, y_val = create_sequences(val_data_scaled, time_steps)
X_test, y_test = create_sequences(test_data_scaled, time_steps)

# Reshape de los datos para que sean aceptados por la red LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Definir el modelo LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Evaluar el modelo en el conjunto de prueba
loss = model.evaluate(X_test, y_test)
print("Loss en el conjunto de prueba:", loss)

# Predecir en el conjunto de prueba
predictions = model.predict(X_test)

# Desescalar las predicciones
predictions_descaled = scaler.inverse_transform(predictions)

# Mostrar algunas predicciones
for i in range(5):
    print("Predicción:", predictions_descaled[i][0])

# Crear un DataFrame con las fechas y las predicciones del modelo LSTM
predicciones_df = pd.DataFrame(index=datos_test.index[time_steps:], columns=['pred'])
predicciones_df['pred'] = predictions_descaled
predicciones_df

# Gráfico estatico predicciones test
# ==============================================================================
fig, ax = plt.subplots(figsize=(11, 4))
datos_test['traffic_count'].plot(ax=ax, label='test')
predicciones_df['pred'].plot(ax=ax, label='predicciones')
ax.legend();

# Gráfico interactivo serie temporal
# ==============================================================================
datos_plot = pd.DataFrame({
                'test': datos_test['traffic_count'],
                'prediccion': predicciones_df['pred'],

                 })
datos_plot.index.name = 'date_time'

fig = px.line(
    data_frame = datos_plot.reset_index(),
    x      = 'date_time',
    y      = datos_plot.columns,
    title  = 'Número de autos: test vs predicciones',
    width  = 900,
    height = 500
)

fig.update_xaxes(rangeslider_visible=True)
fig.show()

# Save predicciones test
# ==============================================================================
rs1 = guardar_resultados(predicciones_df, 'MODEL_EXOGENEAS_LSTM')
upload_df_to_blob(df = rs1, table_name = 'PREDICCION_TRAFFIC_FLOW', connection_str = cnn)

from datetime import timedelta
# Crear un rango de fechas futuro
start_date = pd.to_datetime('2021-01-01 00:00:00')
future_dates = [start_date + timedelta(hours=i) for i in range(90 * 24)]

# Utilizar los últimos datos conocidos para iniciar las predicciones
last_known_data = test_data_scaled[-time_steps:]

# Almacenar predicciones
future_predictions = []

for i in range(len(future_dates)):
    # Preparar el input para el modelo LSTM
    input_data = last_known_data.reshape((1, time_steps, 1))
    
    # Realizar la predicción
    prediction = model.predict(input_data)
    
    # Desescalar la predicción
    prediction_descaled = scaler.inverse_transform(prediction)
    
    # Almacenar la predicción
    future_predictions.append(prediction_descaled[0, 0])
    
    # Actualizar last_known_data para incluir la nueva predicción
    last_known_data = np.append(last_known_data[1:], prediction, axis=0)

# Crear un DataFrame con las fechas y las predicciones
future_predictions_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['pred'])

# Save forecasting
# ==============================================================================
rs1 = guardar_pronostico(future_predictions_df, 'MODEL_EXOGENEAS_LSTM')
upload_df_to_blob(df = rs1, table_name = 'FORECASTING_TRAFFIC_FLOW', connection_str = cnn)

