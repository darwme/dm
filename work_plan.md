## 📝 TO-DO LIST: PLAN DE TRABAJO (Flujo Vehicular)

### 1. **Conjunto de datos**
- [x] Cargar datos desde CSV o SQL Server
- [x] Crear columna `DATE_TIME` combinando `fecha` y `hora_inicio`

### 2. **Limpieza, Transformación y Modelado**
- [x] Convertir `DATE_TIME` a datetime real
- [ ] Agrupar por hora (`groupby('DATE_TIME').sum()`)

### 3. **Visualización y DAX**

#### a) Corporativo
- [X] Crear página “Corporativo” en Power BI
- [X] Agregar filtros por año, mes (desde `DATE_TIME`)
- [X] Crear métricas:
  - [X] Total de pasos por mes
  - [X] Promedio móvil diario
  - [X] % de variación respecto al mes anterior

#### b) Mapa de Calor
- [ ] Crear mapa de calor con seaborn: tráfico promedio por hora y día de la semana

#### c) Tendencia
- [ ] Graficar flujo por hora
- [ ] Añadir tendencia mensual o semanal

#### d) Correlación
- [ ] Realizar `plot_acf` y `plot_pacf`

#### e) Modelos
- [ ] Crear modelo predictivo simple (ARIMA, regresión lineal, media móvil)
- [ ] Visualizar predicciones vs valores reales
- [ ] Agregar parámetro tipo “modelo A / modelo B” (para comparar varios modelos)

#### f) Predicción
- [ ] Realizar las predicciones
- [ ] Exportarlas a la db
- [ ] Crear página de predicción en Power BI
- [ ] Mostrar predicciones futuras
- [ ] Mostrar tabla con valores predichos por hora
