import pyodbc

server = 'sqlserver,1433'
database = 'FlujoVehicular'
username = 'sa'
password = 'StrongPassw0rd!'

conn_str = (
    f'DRIVER={{ODBC Driver 17 for SQL Server}};'
    f'SERVER={server};DATABASE={database};UID={username};PWD={password};'
    f'TrustServerCertificate=yes;Encrypt=no'
)

try:
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    # 🔍 1. Listar todas las tablas
    print("📂 Tablas en la base de datos:")
    cursor.execute("""
        SELECT TABLE_SCHEMA, TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
    """)
    for schema, table in cursor.fetchall():
        print(f"  - {schema}.{table}")

    # 🧪 2. Ejecutar SELECT TOP 100 en flujo_vehicular
    print("\n🔍 Mostrando TOP 100 de predicciones_forecast...")
    cursor.execute("SELECT TOP 100 * FROM predicciones_forecast;")
    columns = [column[0] for column in cursor.description]
    print("📋 Columnas:", columns)

    row_count = 0
    for row in cursor.fetchall():
        print(row)
        row_count += 1

    print(f"\n✅ Total de filas mostradas: {row_count}")

    cursor.close()
    conn.close()

except Exception as e:
    print("❌ Error:", e)
