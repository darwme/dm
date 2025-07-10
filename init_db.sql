-- 1. Crear base de datos y tabla origen (igual a tu ejemplo)
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
-- Hora
CREATE TABLE dim_hora (
    hora_id INT PRIMARY KEY,
    hora TIME(0)
);
GO
INSERT INTO dim_hora (hora_id, hora)
SELECT DISTINCT hora_inicio, 
    CAST(hora_inicio AS VARCHAR) + ':00' 
FROM flujo_vehicular
WHERE ISNUMERIC(hora_inicio) = 1 AND hora_inicio IS NOT NULL;
GO

-- Estacion
CREATE TABLE dim_estacion (
    estacion_id INT PRIMARY KEY,
    estacion VARCHAR(50)
);
GO
INSERT INTO dim_estacion (estacion_id, estacion)
SELECT ROW_NUMBER() OVER (ORDER BY estacion), estacion
FROM (SELECT DISTINCT estacion FROM flujo_vehicular) AS t;
GO

-- Sentido
CREATE TABLE dim_sentido (
    sentido_id INT PRIMARY KEY,
    sentido VARCHAR(50)
);
GO
INSERT INTO dim_sentido (sentido_id, sentido)
SELECT ROW_NUMBER() OVER (ORDER BY sentido), sentido
FROM (SELECT DISTINCT sentido FROM flujo_vehicular) AS t;
GO

-- Forma de pago
CREATE TABLE dim_forma_pago (
    forma_pago_id INT PRIMARY KEY,
    forma_pago VARCHAR(50)
);
GO
INSERT INTO dim_forma_pago (forma_pago_id, forma_pago)
SELECT ROW_NUMBER() OVER (ORDER BY forma_pago), forma_pago
FROM (SELECT DISTINCT forma_pago FROM flujo_vehicular) AS t;
GO

-- Tipo de Vehiculo
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

-- 6. Crear dim_fecha (igual a tu script)
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='dim_fecha' AND xtype='U')
BEGIN
    CREATE TABLE dim_fecha (
        fecha DATE PRIMARY KEY,
        id_fecha VARCHAR(8),
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

-- Insertar fechas únicas
INSERT INTO dim_fecha (fecha)
SELECT DISTINCT fecha FROM fact_pasos
WHERE fecha IS NOT NULL
  AND fecha NOT IN (SELECT fecha FROM dim_fecha);
GO

-- Actualizar columnas calculadas
UPDATE dim_fecha
SET 
    id_fecha = CONVERT(VARCHAR(8), fecha, 112),
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

-- 8. Vistas de conjuntos y forecasting
CREATE VIEW vw_fact_pasos AS
SELECT 
    fecha,
    hora_id,
    estacion_id,
    sentido_id,
    tipo_vehiculo_id,
    forma_pago_id,
    cantidad_pasos,
    CASE WHEN fecha <= '2017-08-31' THEN 'TRAIN' ELSE 'TEST' END AS tipo_conjunto,
    CASE WHEN fecha > '2017-08-31' THEN CONCAT(FORMAT(fecha, 'yyyyMMdd'), hora_id) ELSE NULL END AS id_prediccion
FROM fact_pasos;
GO

CREATE VIEW vw_fact_pasos_forecasting AS
SELECT  
    CONVERT(DATETIME, CONVERT(VARCHAR, f.fecha, 23) + ' ' + CONVERT(VARCHAR, h.hora, 8)) AS date_time,
    h.hora_id AS hour,
    d.mes_anio AS month,
    d.dia_semana AS weekday,
    f.estacion_id,
    f.sentido_id,
    f.tipo_vehiculo_id,
    f.forma_pago_id,
    f.cantidad_pasos,
    CAST(IIF(CASE WHEN f.fecha <= '2017-08-31' THEN 'TRAIN' ELSE 'TEST' END = 'TRAIN', 0, 1) AS INT) AS tipo_conjunto
FROM vw_fact_pasos f
INNER JOIN dim_fecha d ON FORMAT(f.fecha, 'yyyyMMdd') = d.id_fecha
INNER JOIN dim_hora h ON f.hora_id = h.hora_id;
GO

-- Vista pivote para forecasting
CREATE VIEW vw_forecasting_pasos AS
SELECT 
    FORMAT(date_time, 'yyyyMMdd') AS fecha_id,
    DATEPART(HOUR, date_time) AS hora_id,
    [MODEL_FORECASTER],
    [MODEL_EXOGENEAS],
    [MODEL_EXOGENEAS_LGBM],
    [MODEL_EXOGENEAS_CatBoost],
    [MODEL_EXOGENEAS_LSTM]
FROM (
    SELECT 
        date_time,
        prediccion,
        modelo
    FROM forecasting_pasos
) src
PIVOT (
    MAX(prediccion) FOR modelo IN (
        [MODEL_FORECASTER],
        [MODEL_EXOGENEAS],
        [MODEL_EXOGENEAS_LGBM],
        [MODEL_EXOGENEAS_CatBoost],
        [MODEL_EXOGENEAS_LSTM]
    )
) AS p;
GO

-- Vista pivote para predicción
CREATE VIEW vw_prediccion_pasos AS
WITH cte_pred AS (
    SELECT date_time, prediccion, modelo
    FROM prediccion_pasos
)
SELECT 
    CONCAT(FORMAT(date_time, 'yyyyMMdd'), DATEPART(HOUR, date_time)) AS id_prediccion,
    [MODEL_FORECASTER],
    [MODEL_EXOGENEAS],
    [MODEL_EXOGENEAS_CatBoost],
    [MODEL_EXOGENEAS_LGBM],
    [MODEL_EXOGENEAS_LSTM]
FROM (
    SELECT date_time, prediccion, modelo
    FROM cte_pred
) src
PIVOT (
    MAX(prediccion) FOR modelo IN (
        [MODEL_FORECASTER],
        [MODEL_EXOGENEAS],
        [MODEL_EXOGENEAS_CatBoost],
        [MODEL_EXOGENEAS_LGBM],
        [MODEL_EXOGENEAS_LSTM]
    )
) AS p;
GO

-- Limpieza de tablas de forecast
TRUNCATE TABLE forecasting_pasos;
TRUNCATE TABLE prediccion_pasos;
GO