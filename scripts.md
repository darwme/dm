```bash
# Instalar SQL Server 2022 en Docker
docker run -e "ACCEPT_EULA=Y" -e "SA_PASSWORD=StrongPassw0rd!" -p 1433:1433 --name sqlserver -d mcr.microsoft.com/mssql/server:2022-latest

# Copiar el archivo CSV al contenedor de SQL Server
docker cp "C:\Users\DGEP\Downloads\dm\data\flujo-vehicular-2017_2021_illia.csv" sqlserver:/var/opt/mssql/flujo.csv

# Instalar las herramientas de línea de comandos de SQL Server
docker run -it --rm mcr.microsoft.com/mssql-tools /bin/bash -c "/opt/mssql-tools/bin/sqlcmd -S host.docker.internal -U sa -P StrongPassw0rd!"
```

```sql
CREATE DATABASE FlujoVehicular;
GO

USE FlujoVehicular;
GO

CREATE TABLE flujo_vehicular (
    periodo INT,
    fecha VARCHAR(20), -- ¡cambiado de DATE a VARCHAR!
    hora_inicio INT,
    estacion VARCHAR(50),
    sentido VARCHAR(50),
    tipo_vehiculo VARCHAR(50),
    forma_pago VARCHAR(50),
    cantidad_pasos INT
);
GO

BULK INSERT flujo_vehicular
FROM '/var/opt/mssql/flujo.csv'
WITH (
    FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '\n',
    TABLOCK
);
GO

-- Verificar la estructura de la tabla
SELECT TOP 10 * FROM flujo_vehicular;
GO

-- Verificar si hay filas con fecha inválida
SELECT fecha FROM flujo_vehicular WHERE TRY_CONVERT(DATE, fecha, 103) IS NULL;
GO

-- Convertir la columna fecha a DATE y crear una nueva columna
SELECT *,
       TRY_CONVERT(DATE, fecha, 103) AS fecha_convertida
FROM flujo_vehicular;
GO

-- Crear una nueva tabla con la fecha convertida
SELECT 
    periodo,
    TRY_CONVERT(DATE, fecha, 103) AS fecha,
    hora_inicio,
    estacion,
    sentido,
    tipo_vehiculo,
    forma_pago,
    cantidad_pasos
INTO flujo_vehicular_limpio
FROM flujo_vehicular
WHERE TRY_CONVERT(DATE, fecha, 103) IS NOT NULL;
GO

-- Eliminar la tabla original y renombrar la nueva tabla
DROP TABLE flujo_vehicular;
EXEC sp_rename 'flujo_vehicular_limpio', 'flujo_vehicular';
GO

-- Verificar la estructura de la nueva tabla
SELECT TOP 10 * FROM flujo_vehicular;
GO
```

(En el entorno seleccionado desde PowerBI)
Instalar las dependencias de Python necesarias para el análisis de datos
```bash
pip install pandas matplotlib statsmodels
```

Grafico de la tendencia del flujo vehicular por hora
```python
import pandas as pd
import matplotlib.pyplot as plt

# Parsear fechas correctamente
dataset['DATE_TIME'] = pd.to_datetime(dataset['DATE_TIME'], format='%d/%m/%Y %H:%M:%S', dayfirst=True)

# Agrupar por hora (ya que hay duplicados)
dataset = dataset.groupby('DATE_TIME').agg({'cantidad_pasos': 'sum'}).sort_index()

# Reindexar asegurando frecuencia horaria
dataset = dataset.asfreq('h')

# Graficar
fig, ax = plt.subplots(figsize=(12, 4))
dataset['cantidad_pasos'].plot(ax=ax, label='Flujo Vehicular', color='steelblue')

# Formato gráfico
ax.set(title='Tendencia del flujo vehicular por hora', ylabel='Cantidad de pasos', xlabel='Fecha y hora')
ax.legend()
plt.tight_layout()
plt.show()
```