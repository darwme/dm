# Eliminar cualquier contenedor existente
docker compose down -v

# Iniciar los contenedores en segundo plano
docker-compose up -d

# Esperar a que los contenedores estén listos
En docker revisa que el contenedor init-db y haya terminado, deberá salir algo así:

```
2025-07-11 07:09:34 Waiting for SQL Server to be ready...
2025-07-11 07:09:44 Initializing database...
2025-07-11 07:09:45 Changed database context to 'FlujoVehicular'.
2025-07-11 07:09:47 
2025-07-11 07:09:47 (1048575 rows affected)
2025-07-11 07:09:49 
2025-07-11 07:09:49 (1048575 rows affected)
2025-07-11 07:09:49 Caution: Changing any part of an object name could break scripts and stored procedures.
2025-07-11 07:09:51 
2025-07-11 07:09:51 (1048575 rows affected)
2025-07-11 07:09:52 
2025-07-11 07:09:52 (1 rows affected)
2025-07-11 07:09:52 
2025-07-11 07:09:52 (2 rows affected)
2025-07-11 07:09:52 
2025-07-11 07:09:52 (13 rows affected)
2025-07-11 07:09:52 
2025-07-11 07:09:52 (14 rows affected)
2025-07-11 07:10:00 
2025-07-11 07:10:00 (1013234 rows affected)
2025-07-11 07:10:01 
2025-07-11 07:10:01 (1461 rows affected)
2025-07-11 07:10:01 
2025-07-11 07:10:01 (1461 rows affected)
```

# Instalar python venv
python -m venv venv

# Activar el entorno virtual
source venv/bin/activate

# Instalar las dependencias del proyecto
pip install -r requirements.txt

# Ejecutar script para predecir y cargar a la base de datos el flujo vehicular
python predict_and_load.py

# Abrir el dashboard en Power BI Desktop

TODO List:

- [X] Corporativo
- [X] Mapa de Calor
- [X] Tendencia
- [X] Correlación
- [ ] Modelos: Los modelos ya están entrenados y guardados en la carpeta `models/v3`, y las predicciones se cargan en la base de datos al ejecutar el script `predict_and_load.py`. Solo falta poner las vistas en power bi.
- [ ] Predicción: Lo mismo que los modelos, ya están entrenados y guardados en la carpeta `models/v3`, y las predicciones se cargan en la base de datos al ejecutar el script `predict_and_load.py`. Solo falta poner las vistas en power bi.

Notas: Se tomo la data hasta finales de 2019 dado a que la data de 2020 en adelante tiene un comportamiento anómalo debido a la pandemia de COVID-19. Por lo tanto, no se considera para el entrenamiento de los modelos.
Sin embargo, se pueden hacer predicciones para el año 2020 y 2021, para el año 2020 se va a ver que obviamente no se va a cumplir la predicción, pero para el año 2021 se espera una predicción más cercana a la realidad, aunque no se tienen datos reales para comparar.
