import pyodbc

server = '172.17.0.1,1433'
database = 'FlujoVehicular'
username = 'sa'
password = 'StrongPassw0rd!'  # Cambia si usaste otra

conn_str = (
    f'DRIVER={{ODBC Driver 17 for SQL Server}};'
    f'SERVER={server};DATABASE={database};UID={username};PWD={password}'
)

try:
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM flujo_vehicular;")
    total = cursor.fetchone()[0]
    print(f"✅ Total de filas en 'flujo_vehicular': {total}")

    cursor.execute("SELECT TOP 5 * FROM flujo_vehicular;")
    rows = cursor.fetchall()
    for row in rows:
        print(row)

    cursor.close()
    conn.close()

except Exception as e:
    print("❌ Error:", e)
