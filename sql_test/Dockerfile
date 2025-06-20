FROM python:3.11-slim

# Instalar dependencias de ODBC y herramientas básicas
RUN apt-get update && apt-get install -y \
    gnupg curl unixodbc-dev gcc g++ \
    && curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && pip install pyodbc

# Copiar el script
COPY verificar_sql.py /verificar_sql.py

CMD ["python", "/verificar_sql.py"]
