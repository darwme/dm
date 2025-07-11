-- 1. Crear base de datos y tabla origen
CREATE DATABASE FlujoVehicular;
GO

USE FlujoVehicular;
GO

CREATE TABLE flujo_vehicular (
    periodo INT,
    fecha VARCHAR(20),
    hora_inicio INT,
    estacion VARCHAR(50),
    sentido VARCHAR(50),
    tipo_vehiculo VARCHAR(50),
    forma_pago VARCHAR(50),
    cantidad_pasos INT
);
GO

-- 2. Cargar el CSV (ajusta la ruta según corresponda)
SET DATEFORMAT dmy;
BULK INSERT flujo_vehicular
FROM '/data/flujo-vehicular-2017_2021_illia.csv'
WITH (
    FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '\n',
    TABLOCK
);
GO

-- 3. Limpiar y convertir fecha
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

DROP TABLE flujo_vehicular;
EXEC sp_rename 'flujo_vehicular_limpio', 'flujo_vehicular';
GO

-- 4. Crear dimensiones
CREATE TABLE dim_hora (
    hora_id INT PRIMARY KEY,
    hora TIME(0) 
);
GO

INSERT INTO dim_hora (hora_id, hora)
SELECT 
    ROW_NUMBER() OVER (ORDER BY hora_inicio) AS hora_id,
    CAST(RIGHT('00' + CAST(hora_inicio AS VARCHAR), 2) + ':00' AS TIME(0)) AS hora
FROM flujo_vehicular
WHERE ISNUMERIC(hora_inicio) = 1
  AND hora_inicio IS NOT NULL;
GO

CREATE TABLE dim_estacion (
    estacion_id INT PRIMARY KEY,
    estacion VARCHAR(50)
);
GO
INSERT INTO dim_estacion (estacion_id, estacion)
SELECT ROW_NUMBER() OVER (ORDER BY estacion), estacion
FROM (SELECT DISTINCT estacion FROM flujo_vehicular) AS t;
GO

CREATE TABLE dim_sentido (
    sentido_id INT PRIMARY KEY,
    sentido VARCHAR(50)
);
GO
INSERT INTO dim_sentido (sentido_id, sentido)
SELECT ROW_NUMBER() OVER (ORDER BY sentido), sentido
FROM (SELECT DISTINCT sentido FROM flujo_vehicular) AS t;
GO

CREATE TABLE dim_forma_pago (
    forma_pago_id INT PRIMARY KEY,
    forma_pago VARCHAR(50)
);
GO
INSERT INTO dim_forma_pago (forma_pago_id, forma_pago)
SELECT ROW_NUMBER() OVER (ORDER BY forma_pago), forma_pago
FROM (SELECT DISTINCT forma_pago FROM flujo_vehicular) AS t;
GO

CREATE TABLE dim_tipo_vehiculo (
    tipo_vehiculo_id INT PRIMARY KEY,
    tipo_vehiculo VARCHAR(50)
);
GO
INSERT INTO dim_tipo_vehiculo (tipo_vehiculo_id, tipo_vehiculo)
SELECT ROW_NUMBER() OVER (ORDER BY tipo_vehiculo), tipo_vehiculo
FROM (SELECT DISTINCT tipo_vehiculo FROM flujo_vehicular) AS t;
GO

-- 5. Crear tabla de hechos
CREATE TABLE fact_pasos (
    periodo INT NOT NULL,
    fecha DATE NOT NULL,
    hora_id INT NOT NULL,
    estacion_id INT NOT NULL,
    sentido_id INT NOT NULL,
    tipo_vehiculo_id INT NOT NULL,
    forma_pago_id INT NOT NULL,
    cantidad_pasos INT,
    CONSTRAINT fk_fact_hora FOREIGN KEY (hora_id) REFERENCES dim_hora(hora_id),
    CONSTRAINT fk_fact_estacion FOREIGN KEY (estacion_id) REFERENCES dim_estacion(estacion_id),
    CONSTRAINT fk_fact_sentido FOREIGN KEY (sentido_id) REFERENCES dim_sentido(sentido_id),
    CONSTRAINT fk_fact_tipo_vehiculo FOREIGN KEY (tipo_vehiculo_id) REFERENCES dim_tipo_vehiculo(tipo_vehiculo_id),
    CONSTRAINT fk_fact_forma_pago FOREIGN KEY (forma_pago_id) REFERENCES dim_forma_pago(forma_pago_id)
);
GO

INSERT INTO fact_pasos (
    periodo, fecha, hora_id, estacion_id, sentido_id, tipo_vehiculo_id, forma_pago_id, cantidad_pasos
)
SELECT 
    f.periodo,
    f.fecha,
    h.hora_id,
    e.estacion_id,
    s.sentido_id,
    v.tipo_vehiculo_id,
    p.forma_pago_id,
    f.cantidad_pasos
FROM flujo_vehicular f
INNER JOIN dim_hora h ON f.hora_inicio = h.hora_id
INNER JOIN dim_estacion e ON f.estacion = e.estacion
INNER JOIN dim_sentido s ON f.sentido = s.sentido
INNER JOIN dim_tipo_vehiculo v ON f.tipo_vehiculo = v.tipo_vehiculo
INNER JOIN dim_forma_pago p ON f.forma_pago = p.forma_pago
WHERE ISNUMERIC(f.hora_inicio) = 1;
GO

-- 6. Crear dim_fecha
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='dim_fecha' AND xtype='U')
BEGIN
    CREATE TABLE dim_fecha (
        fecha DATE PRIMARY KEY,
        fecha_id VARCHAR(8),
        dia_semana INT,
        nombre_dia_semana VARCHAR(20),
        dia_mes INT,
        dia_anio INT,
        nombre_mes VARCHAR(20),
        mes_anio INT,
        semana_anio INT,
        anio INT,
        bimestre INT,
        trimestre INT,
        semestre INT,
        feriado BIT
    );
END
GO

INSERT INTO dim_fecha (fecha)
SELECT DISTINCT fecha FROM fact_pasos
WHERE fecha IS NOT NULL
  AND fecha NOT IN (SELECT fecha FROM dim_fecha);
GO

UPDATE dim_fecha
SET 
    fecha_id = CONVERT(VARCHAR(8), fecha, 112),
    dia_semana = DATEPART(WEEKDAY, fecha),
    nombre_dia_semana = DATENAME(WEEKDAY, fecha),
    dia_mes = DAY(fecha),
    dia_anio = DATEPART(DAYOFYEAR, fecha),
    nombre_mes = DATENAME(MONTH, fecha),
    mes_anio = MONTH(fecha),
    semana_anio = DATEPART(ISO_WEEK, fecha),
    anio = YEAR(fecha),
    bimestre = ((MONTH(fecha) - 1) / 2) + 1,
    trimestre = ((MONTH(fecha) - 1) / 3) + 1,
    semestre = ((MONTH(fecha) - 1) / 6) + 1,
    feriado = 0;
GO

ALTER TABLE fact_pasos
WITH CHECK ADD CONSTRAINT fk_fact_dim_fecha
FOREIGN KEY (fecha) REFERENCES dim_fecha (fecha);
GO

-- 7. Tablas de forecasting y predicción
CREATE TABLE forecasting_pasos (
    date_time DATETIME2(0) NOT NULL,
    prediccion FLOAT NULL,
    modelo VARCHAR(50) NULL
);
GO

CREATE TABLE prediccion_pasos (
    date_time DATETIME2(0) NOT NULL,
    prediccion FLOAT NULL,
    modelo VARCHAR(50) NULL,
    execution_date DATETIME2(0) NULL
);
GO

-- 8. Vistas con tipos corregidos

CREATE OR ALTER VIEW vw_fact_pasos AS
SELECT
    CONCAT(CONVERT(CHAR(10), fecha, 120), ' ', RIGHT('00' + CAST(hora_id-1 AS VARCHAR), 2), ':00') AS date_time,
    SUM(cantidad_pasos) AS cantidad_pasos_real
FROM fact_pasos
GROUP BY fecha, hora_id;


CREATE OR ALTER VIEW vw_prediccion_pasos_pivot AS
SELECT
    fp.date_time,
    fp.cantidad_pasos_real,
    MAX(CASE WHEN f.modelo = 'MODEL_EXOGENEAS_XGB' THEN f.prediccion END) AS prediccion_XGBoost,
    MAX(CASE WHEN f.modelo = 'MODEL_EXOGENEAS_LGBM' THEN f.prediccion END) AS prediccion_LGBM,
    MAX(CASE WHEN f.modelo = 'MODEL_EXOGENEAS_CatBoost' THEN f.prediccion END) AS prediccion_CatBoost
FROM
    (SELECT 
        CONCAT(CONVERT(CHAR(10), fecha, 120), ' ', RIGHT('00' + CAST(hora_id-1 AS VARCHAR), 2), ':00') AS date_time,
        SUM(cantidad_pasos) AS cantidad_pasos_real
     FROM fact_pasos
     WHERE CONCAT(CONVERT(CHAR(10), fecha, 120), ' ', RIGHT('00' + CAST(hora_id-1 AS VARCHAR), 2), ':00') >= '2019-09-13 12:00:00'
     GROUP BY fecha, hora_id) fp
LEFT JOIN forecasting_pasos f
    ON fp.date_time = f.date_time
GROUP BY fp.date_time, fp.cantidad_pasos_real;


CREATE OR ALTER VIEW vw_forecasting_pasos_pivot AS
SELECT
    fp.date_time,
    fp.cantidad_pasos_real,
    MAX(CASE WHEN f.modelo = 'MODEL_EXOGENEAS_XGB' THEN f.prediccion END) AS forecast_XGBoost,
    MAX(CASE WHEN f.modelo = 'MODEL_EXOGENEAS_LGBM' THEN f.prediccion END) AS forecast_LGBM,
    MAX(CASE WHEN f.modelo = 'MODEL_EXOGENEAS_CatBoost' THEN f.prediccion END) AS forecast_CatBoost
FROM
    (SELECT 
        CONCAT(CONVERT(CHAR(10), fecha, 120), ' ', RIGHT('00' + CAST(hora_id-1 AS VARCHAR), 2), ':00') AS date_time,
        SUM(cantidad_pasos) AS cantidad_pasos_real
     FROM fact_pasos
     WHERE CONCAT(CONVERT(CHAR(10), fecha, 120), ' ', RIGHT('00' + CAST(hora_id-1 AS VARCHAR), 2), ':00') >= '2019-09-13 12:00:00'
     GROUP BY fecha, hora_id) fp
LEFT JOIN forecasting_pasos f
    ON fp.date_time = f.date_time
GROUP BY fp.date_time, fp.cantidad_pasos_real;

CREATE OR ALTER VIEW vw_predicciones_futuras_pivot AS
SELECT
    p.date_time,
    MAX(CASE WHEN p.modelo = 'MODEL_EXOGENEAS_XGB' THEN p.prediccion END) AS prediccion_XGBoost,
    MAX(CASE WHEN p.modelo = 'MODEL_EXOGENEAS_LGBM' THEN p.prediccion END) AS prediccion_LGBM,
    MAX(CASE WHEN p.modelo = 'MODEL_EXOGENEAS_CatBoost' THEN p.prediccion END) AS prediccion_CatBoost
FROM prediccion_pasos p
GROUP BY p.date_time;
GO

CREATE OR ALTER VIEW vw_comparativo_pasos_pivot AS
WITH all_datetimes AS (
    SELECT DISTINCT
        CONCAT(CONVERT(CHAR(10), fecha, 120), ' ', RIGHT('00' + CAST(hora_id-1 AS VARCHAR), 2), ':00') AS date_time
    FROM fact_pasos
    UNION
    SELECT DISTINCT date_time FROM prediccion_pasos
    UNION
    SELECT DISTINCT date_time FROM forecasting_pasos
),
realidad AS (
    SELECT
        CONCAT(CONVERT(CHAR(10), fecha, 120), ' ', RIGHT('00' + CAST(hora_id-1 AS VARCHAR), 2), ':00') AS date_time,
        SUM(cantidad_pasos) AS cantidad_pasos_real
    FROM fact_pasos
    GROUP BY fecha, hora_id
),
predicciones AS (
    SELECT
        date_time,
        MAX(CASE WHEN modelo = 'MODEL_EXOGENEAS_XGB' THEN prediccion END) AS prediccion_XGBoost,
        MAX(CASE WHEN modelo = 'MODEL_EXOGENEAS_LGBM' THEN prediccion END) AS prediccion_LGBM,
        MAX(CASE WHEN modelo = 'MODEL_EXOGENEAS_CatBoost' THEN prediccion END) AS prediccion_CatBoost
    FROM prediccion_pasos
    GROUP BY date_time
),
forecastings AS (
    SELECT
        date_time,
        MAX(CASE WHEN modelo = 'MODEL_EXOGENEAS_XGB' THEN prediccion END) AS forecast_XGBoost,
        MAX(CASE WHEN modelo = 'MODEL_EXOGENEAS_LGBM' THEN prediccion END) AS forecast_LGBM,
        MAX(CASE WHEN modelo = 'MODEL_EXOGENEAS_CatBoost' THEN prediccion END) AS forecast_CatBoost
    FROM forecasting_pasos
    GROUP BY date_time
)
SELECT
    ad.date_time,
    r.cantidad_pasos_real,
    p.prediccion_XGBoost,
    p.prediccion_LGBM,
    p.prediccion_CatBoost,
    f.forecast_XGBoost,
    f.forecast_LGBM,
    f.forecast_CatBoost
FROM
    all_datetimes ad
    LEFT JOIN realidad r ON ad.date_time = r.date_time
    LEFT JOIN predicciones p ON ad.date_time = p.date_time
    LEFT JOIN forecastings f ON ad.date_time = f.date_time;




-- Limpieza de tablas de forecast
TRUNCATE TABLE forecasting_pasos;
TRUNCATE TABLE prediccion_pasos;
GO
