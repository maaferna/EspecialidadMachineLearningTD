# Proyecto: Comparación de Técnicas Avanzadas de Regresión

Este proyecto implementa tres enfoques distintos:

1. **Elastic Net** para predicción de precios de viviendas.
2. **Regresión Cuantílica** para estimar percentiles de ingresos.
3. **VAR** para proyección de indicadores macroeconómicos.

La estructura está modularizada con scripts de carga, modelamiento y visualización, facilitando su mantenimiento y escalabilidad.

## Estructura del Proyecto

- `src/`: Módulos reutilizables.
- `scripts/`: Ejecución principal y generación de notebooks.
- `outputs/`: Resultados, métricas y visualizaciones.
- `notebooks/`: Versión interactiva del flujo.

## Activación del entorno

```bash
conda env create -f environment.yml
conda activate regresion_avanzada
```

## Ejecución principal

```bash
python -m scripts.main
```

# 📘 Análisis Comparativo de Modelos de Regresión

## 🧠 Resumen Ejecutivo

Este proyecto tiene como objetivo evaluar tres enfoques de modelado estadístico en distintos contextos de datos:

- **ElasticNet** aplicado al conjunto *California Housing* para estimar precios de vivienda.
- **Regresión Cuantílica** sobre el conjunto *Adult Income* para estimar ingresos en distintos percentiles.
- **Modelo VAR (Vector Autoregressive)** en *indicadores macroeconómicos* para realizar pronósticos multivariados.

Cada modelo fue evaluado con métricas específicas, y se generaron gráficos y archivos `.csv` para visualizar y respaldar los resultados. Finalmente, se realiza un análisis comparativo y se concluye cuál fue la técnica más robusta en su contexto.

---

## 1️⃣ ElasticNet – California Housing 🏠

📥 Dataset: 20.640 filas – 9 variables  
🎯 Objetivo: Predecir `MedHouseVal` usando regresión penalizada.

### 🔢 Resultados (MSE):

| alpha | l1_ratio | MSE     |
|-------|----------|---------|
| 0.001 | 0.1      | 0.5092  |
| 0.01  | 0.5      | 0.5100  |
| 0.1   | 0.7      | 0.6305  |
| 1.0   | 0.9      | 1.3107  |

🏆 **Mejor configuración:** `alpha=0.001`, `l1_ratio=0.1` → **MSE = 0.5092**

### 📊 Coeficientes

![Coeficientes ElasticNet](outputs/coeficientes_elasticnet.png)

🔗 [📄 Ver CSV resultados](outputs/elasticnet_resultados.csv)  
🔗 [🏆 Ver CSV mejor modelo](outputs/elasticnet_mejor.csv)

---

## 2️⃣ Regresión Cuantílica – Adult Income 💼

📥 Dataset: 45.222 filas – 15 columnas  
🎯 Objetivo: Estimar el ingreso (<=50K / >50K) por cuantiles.

### 📈 Pinball Loss por Cuantil:

| Cuantil | Pinball Loss |
|---------|---------------|
| 0.1     | 0.0244        |
| 0.2     | 0.0487        |
| 0.3     | 0.0731        |
| 0.4     | 0.0974        |
| 0.5     | 0.1218        |
| 0.6     | 0.1461        |
| 0.7     | 0.1705        |
| 0.8     | 0.1513        |
| 0.9     | 0.0756        |

🏆 **Mejor cuantil:** `q = 0.1` → **Pinball Loss = 0.0244**

### 📊 Gráfico:

![Pinball Loss](outputs/quantile_pinball_loss.png)

🔗 [📄 Ver CSV resultados](outputs/quantile_resultados.csv)  
🔗 [🏆 Ver CSV mejor cuantil](outputs/quantile_mejor.csv)

---

## 3️⃣ VAR – Indicadores Macroeconómicos 📉

📥 Dataset: 202 observaciones – 3 variables (`realgdp`, `realcons`, `realinv`)  
🎯 Objetivo: Predecir valores futuros de la serie con modelo VAR.

### ⚙️ Configuración y Resultados

- ✅ Series estacionarias según ADF test.
- 📌 Lag óptimo (AIC): 1
- 📈 Pronóstico a 5 pasos:

| Paso | realgdp | realcons | realinv |
|------|---------|----------|---------|
| 1    | 0.00785 | 0.00861  | 0.00835 |
| 2    | 0.00783 | 0.00841  | 0.00873 |
| 3    | 0.00771 | 0.00836  | 0.00799 |
| 4    | 0.00767 | 0.00833  | 0.00782 |
| 5    | 0.00765 | 0.00832  | 0.00771 |

### 📊 Gráfico Forecast:

![Forecast VAR](outputs/var_forecast_plot.png)

🔗 [📄 Ver CSV pronóstico](outputs/var_forecast.csv)

---

## 🧩 Comparación de Técnicas

| Modelo              | Dataset           | Métrica       | Resultado      | Fortalezas                            | Limitaciones                     |
|---------------------|-------------------|---------------|----------------|----------------------------------------|----------------------------------|
| ElasticNet          | CaliforniaHousing | MSE           | **0.5092**     | Interpretabilidad, buen ajuste         | Menos flexible en datos no lineales |
| Quantile Regression | Adult Income      | Pinball Loss  | **0.0244 (q=0.1)** | Modela diferentes cuantiles         | Costoso computacionalmente        |
| VAR                 | Macroeconómicos   | Forecast      | 5 pasos estables | Captura dependencia temporal múltiple | Solo para series estacionarias   |

---

## ✅ Conclusiones

- **ElasticNet** fue la técnica más robusta para datos continuos como los precios de viviendas, por su capacidad de regularización y simplicidad.
- **Regresión Cuantílica** permitió modelar con alta precisión los extremos de ingreso en `Adult Income`, especialmente en cuantiles bajos.
- **VAR** entregó pronósticos razonables para un conjunto estacionario, siendo adecuado para escenarios macroeconómicos.

📌 **Cada técnica fue la más robusta dentro de su dominio de aplicación.**  
No sería apropiado comparar directamente VAR con ElasticNet en un mismo dataset, pero sí se puede destacar su idoneidad en su contexto.

---

## 📂 Archivos generados

```bash
outputs/
├── elasticnet_resultados.csv
├── elasticnet_mejor.csv
├── coeficientes_elasticnet.png
├── quantile_resultados.csv
├── quantile_mejor.csv
├── quantile_pinball_loss.png
├── var_forecast.csv
├── var_forecast_plot.png
