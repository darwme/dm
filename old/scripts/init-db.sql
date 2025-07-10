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

BULK INSERT flujo_vehicular
FROM '/data/flujo-vehicular-2017_2021_illia.csv'
WITH (
    FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '\n',
    TABLOCK
);
GO

-- Convertir la columna fecha a DATE y crear una nueva tabla
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