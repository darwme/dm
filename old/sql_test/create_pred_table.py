import pyodbc

# Configuración de conexión (ya verificada)
server = 'sqlserver,1433'
database = 'FlujoVehicular'
username = 'sa'
password = 'StrongPassw0rd!'  # Asegúrate que coincida

# Cadena de conexión
conn_str = (
    f'DRIVER={{ODBC Driver 17 for SQL Server}};'
    f'SERVER={server};DATABASE={database};UID={username};PWD={password};'
    f'TrustServerCertificate=yes;Encrypt=no'
)

try:
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    # Verificar si la tabla ya existe
    check_table_sql = """
    SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_NAME = 'predicciones_forecast';
    """
    cursor.execute(check_table_sql)
    exists = cursor.fetchone()[0]

    if exists:
        print("⚠️ La tabla 'predicciones_forecast' ya existe.")
    else:
        # Crear tabla
        create_table_sql = """
        CREATE TABLE predicciones_forecast (
            modelo VARCHAR(20),
            datetime DATETIME,
            prediccion FLOAT
        );
        """
        cursor.execute(create_table_sql)
        conn.commit()
        print("✅ Tabla 'predicciones_forecast' creada correctamente.")

    cursor.close()
    conn.close()

except Exception as e:
    print("❌ Error al crear la tabla:", e)
