# 🫀 SHAZAM ECG: Enseñando a la IA a Entender el Corazón como un Médico

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)]()
[![Databricks](https://img.shields.io/badge/Databricks-PySpark-FF3621.svg)]()
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20EC2-232F3E.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B.svg)]()
[![FAISS](https://img.shields.io/badge/Meta%20FAISS-Vector%20Search-0467DF.svg)]()

## 📖 Descripción General
**SHAZAM para el Corazón** es un Sistema de Soporte a la Decisión Clínica (CDSS) híbrido para la clasificación y explicación visual de señales electrocardiográficas (ECG) de 12 derivaciones estándar. 

La adopción de la Inteligencia Artificial en entornos críticos, como las Unidades de Cuidados Intensivos (UCI), ha estado históricamente limitada por el problema de la **"caja negra"**. Los médicos no confían en máquinas que entregan un porcentaje de probabilidad sin justificar su razonamiento. 

Este proyecto resuelve ese paradigma integrando *Deep Learning* con un motor de Recuperación de Información Basado en Contenido (CBIR). Funciona como la famosa aplicación de música: cuando detecta una anomalía cardíaca, busca en fracciones de segundo en un índice vectorial y **le muestra al cardiólogo los trazados de pacientes históricos que sufrieron exactamente la misma deformidad**. 

## ✨ Innovaciones Principales y Aportes de Ingeniería

* **Desacoplamiento Polimórfico Absoluto:** El sistema divide el análisis en dos motores independientes. El primer cerebro mira ventanas largas de 8 segundos (2000 muestras) para entender el ritmo global (ej. Fibrilación Auricular). El segundo cerebro analiza pedacitos minúsculos de 0.8 segundos (200 muestras) para evaluar la morfología latido a latido (ej. Bloqueos de Rama y Extrasístoles).
* **Aprendizaje Contrastivo Supervisado:** En lugar de memorizar etiquetas médicas, la IA juega a "encuentra las similitudes". Agrupa los latidos con la misma enfermedad en un mapa topográfico virtual (espacio latente) y aleja los diferentes. 
* **Data Lake (AWS S3) y Mapeo en Memoria:** Transición de un DWH tradicional a un Data Lake científico capaz de almacenar tensores tridimensionales masivos `.npz`. El uso de `mmap_mode` permite que el servidor asíncrono lea matrices gigantes directamente desde el disco sin desbordar la memoria RAM (OOM).

## 🚀 Arquitectura y Tech Stack

El pipeline orquesta el flujo de datos bajo la Arquitectura Medallón (Bronze, Silver, Gold), asegurando alta resiliencia desde la ingesta hasta el despliegue.

### 🛠️ Ecosistema Tecnológico
* **Data Engineering (ETL):** Orquestación de clústeres masivos sobre registros médicos (PhysioNet, MIMIC-IV, CPSC) utilizando **PySpark** en **Databricks**. 
* **Procesamiento Digital de Señales (DSP):** Remoción de tendencia lineal, filtro Butterworth pasa-banda de 4° orden (0.5 Hz - 45.0 Hz) de fase cero, y detección de complejos QRS con el algoritmo Pan-Tompkins.
* **Modelado de Machine Learning:** Arquitecturas de Redes Neuronales Convolucionales 1D desarrolladas en **PyTorch**, entrenadas en **Google Colab Pro** (GPUs T4/A100).
* **Motor de Similitud Vectorial:** Indexación de vectores de 128 dimensiones utilizando **FAISS** para cálculos de vecinos más cercanos (KNN) en submilisegundos.
* **Despliegue Full-Stack:** Servidor Ubuntu en **AWS EC2** (instancia `t3.large`), impulsado por un backend asíncrono en **FastAPI** y una interfaz interactiva de usuario en **Streamlit**.

## 📊 Rendimiento Clínico en Producción

Sometido a una prueba ciega de estrés y validación externa cruzada con **57,064 registros médicos reales**, el sistema logró una mitigación drástica de falsas alarmas ("fatiga de alarmas"). 

| Condición Médica Detectada | Precisión | Sensibilidad (Recall) | Casos Evaluados |
| :--- | :--- | :--- | :--- |
| **Latido Sano (Normal)** | 99.68% | 99.98% | 53,201 |
| **Fibrilación Auricular (AFIB)** | 98.54% | 97.35% | 415 |
| **Bloqueo de Rama Izq. (LBBB)** | 99.74% | 87.17% | 865 |
| **Bloqueo de Rama Der. (RBBB)** | 99.54% | 90.48% | 483 |
| **Extrasístole Ventricular (PVC)** | 99.95% | 99.95% | 2,100 |

*Precisión Global del Sistema: **99.68%***.

## 🖥️ Experiencia de Usuario: Monitor UCI Interactivo

El frontend sustituye el tradicional reporte estático de papel por un entorno visual avanzado:
* **Osciloscopio Interactivo (60 FPS):** Uso de `components.html` para inyectar *HTML5 Canvas* y *JavaScript* puro, animando el trazado de las 12 derivaciones sin bloquear el hilo principal de Python.
* **Herramientas de Precisión:** El médico cuenta con una *Regla de Medición (Caliper)* y *Zoom Magnificador* controlados por el mouse para analizar milivoltios y milisegundos directamente sobre la onda.
* **Alertas Sensoriales:** Integración de la *Web Audio API* para emitir el característico pitido analógico de 800Hz sincronizado con cada latido.
* **Generación de Reportes PDF:** Procesamiento HTML/CSS en memoria RAM utilizando *WeasyPrint* para entregar reportes exportables al instante.

## 📸 Galería y Capturas de Pantalla

<div align="center">
  <img width="1916" height="992" alt="image" src="https://github.com/user-attachments/assets/55b330bc-6f6e-4bf4-bda5-3fdad6e6418a" />
  <p><i>Vista general del sistema de monitorización en tiempo real.</i></p>
</div>

<br>

<div align="center">
  <img width="1913" height="995" alt="image" src="https://github.com/user-attachments/assets/381b9d40-3fe9-4e9c-8d75-2aac33731f0a" />
  <p><i>Osciloscopio Interactivo.</i></p>
  <img width="1911" height="990" alt="image" src="https://github.com/user-attachments/assets/d3dba122-23d0-4483-8dd6-888ce2ea7905" />
  <p><i>Diagnostico IA</i></p>
 <img width="1915" height="995" alt="image" src="https://github.com/user-attachments/assets/4e0ee899-5ff5-4ddb-8fb9-062c2c1e79c1" />
  <p><i>Motor de busqueda vectorial</i></p>
  <img width="1913" height="992" alt="image" src="https://github.com/user-attachments/assets/aa13623c-430c-4b24-a935-2d8b316548f6" />
   <p><i>Latido recuperado</i></p>
<img width="1913" height="986" alt="image" src="https://github.com/user-attachments/assets/c638de63-67bf-4436-8d9d-a85c1389ab5e" />
   <p><i>Reporte Clinico en Formato PDF</i></p>
</div>
