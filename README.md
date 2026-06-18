# [cite_start]🫀 SHAZAM ECG: Enseñando a la IA a Entender el Corazón como un Médico [cite: 1]

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)]()
[![Databricks](https://img.shields.io/badge/Databricks-PySpark-FF3621.svg)]()
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20EC2-232F3E.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B.svg)]()
[![FAISS](https://img.shields.io/badge/Meta%20FAISS-Vector%20Search-0467DF.svg)]()

## 📖 Descripción General
[cite_start]**SHAZAM para el Corazón** es un Sistema de Soporte a la Decisión Clínica (CDSS) híbrido para la clasificación y explicación visual de señales electrocardiográficas (ECG) de 12 derivaciones estándar. 

[cite_start]La adopción de la Inteligencia Artificial en entornos críticos, como las Unidades de Cuidados Intensivos (UCI), ha estado históricamente limitada por el problema de la **"caja negra"**[cite: 191]. [cite_start]Los médicos no confían en máquinas que entregan un porcentaje de probabilidad sin justificar su razonamiento[cite: 7]. 

[cite_start]Este proyecto resuelve ese paradigma integrando *Deep Learning* con un motor de Recuperación de Información Basado en Contenido (CBIR)[cite: 172]. [cite_start]Funciona como la famosa aplicación de música: cuando detecta una anomalía cardíaca, busca en fracciones de segundo en un índice vectorial y **le muestra al cardiólogo los trazados de pacientes históricos que sufrieron exactamente la misma deformidad**[cite: 9, 339]. 

## ✨ Innovaciones Principales y Aportes de Ingeniería

* [cite_start]**Desacoplamiento Polimórfico Absoluto:** El sistema divide el análisis en dos motores independientes[cite: 32, 33]. [cite_start]El primer cerebro mira ventanas largas de 8 segundos (2000 muestras) para entender el ritmo global (ej. Fibrilación Auricular)[cite: 34]. [cite_start]El segundo cerebro analiza pedacitos minúsculos de 0.8 segundos (200 muestras) para evaluar la morfología latido a latido (ej. Bloqueos de Rama y Extrasístoles)[cite: 35, 193].
* [cite_start]**Aprendizaje Contrastivo Supervisado:** En lugar de memorizar etiquetas médicas, la IA juega a "encuentra las similitudes"[cite: 39, 40]. [cite_start]Agrupa los latidos con la misma enfermedad en un mapa topográfico virtual (espacio latente) y aleja los diferentes[cite: 79, 81]. 
* [cite_start]**Data Lake (AWS S3) y Mapeo en Memoria:** Transición de un DWH tradicional a un Data Lake científico capaz de almacenar tensores tridimensionales masivos `.npz`[cite: 365, 366]. [cite_start]El uso de `mmap_mode` permite que el servidor asíncrono lea matrices gigantes directamente desde el disco sin desbordar la memoria RAM (OOM)[cite: 566].

## 🚀 Arquitectura y Tech Stack

[cite_start]El pipeline orquesta el flujo de datos bajo la Arquitectura Medallón (Bronze, Silver, Gold), asegurando alta resiliencia desde la ingesta hasta el despliegue[cite: 443].

### 🛠️ Ecosistema Tecnológico
* [cite_start]**Data Engineering (ETL):** Orquestación de clústeres masivos sobre registros médicos (PhysioNet, MIMIC-IV, CPSC) utilizando **PySpark** en **Databricks**[cite: 344, 353]. 
* [cite_start]**Procesamiento Digital de Señales (DSP):** Remoción de tendencia lineal, filtro Butterworth pasa-banda de 4° orden (0.5 Hz - 45.0 Hz) de fase cero, y detección de complejos QRS con el algoritmo Pan-Tompkins[cite: 343, 490, 491, 492].
* [cite_start]**Modelado de Machine Learning:** Arquitecturas de Redes Neuronales Convolucionales 1D desarrolladas en **PyTorch**, entrenadas en **Google Colab Pro** (GPUs T4/A100)[cite: 344].
* [cite_start]**Motor de Similitud Vectorial:** Indexación de vectores de 128 dimensiones utilizando **FAISS** para cálculos de vecinos más cercanos (KNN) en submilisegundos[cite: 355, 498].
* [cite_start]**Despliegue Full-Stack:** Servidor Ubuntu en **AWS EC2** (instancia `t3.large`), impulsado por un backend asíncrono en **FastAPI** y una interfaz interactiva de usuario en **Streamlit**[cite: 344, 452].

## 📊 Rendimiento Clínico en Producción

[cite_start]Sometido a una prueba ciega de estrés y validación externa cruzada con **57,064 registros médicos reales**, el sistema logró una mitigación drástica de falsas alarmas ("fatiga de alarmas")[cite: 16, 94]. 

| Condición Médica Detectada | Precisión | Sensibilidad (Recall) | Casos Evaluados |
| :--- | :--- | :--- | :--- |
| **Latido Sano (Normal)** | 99.68% | 99.98% | 53,201 |
| **Fibrilación Auricular (AFIB)** | 98.54% | 97.35% | 415 |
| **Bloqueo de Rama Izq. (LBBB)** | 99.74% | 87.17% | 865 |
| **Bloqueo de Rama Der. (RBBB)** | 99.54% | 90.48% | 483 |
| **Extrasístole Ventricular (PVC)** | 99.95% | 99.95% | 2,100 |

[cite_start]*Precisión Global del Sistema: **99.68%***[cite: 99, 100].

## 🖥️ Experiencia de Usuario: Monitor UCI Interactivo

El frontend sustituye el tradicional reporte estático de papel por un entorno visual avanzado:
* [cite_start]**Osciloscopio Interactivo (60 FPS):** Uso de `components.html` para inyectar *HTML5 Canvas* y *JavaScript* puro, animando el trazado de las 12 derivaciones sin bloquear el hilo principal de Python[cite: 370, 956, 957].
* [cite_start]**Herramientas de Precisión:** El médico cuenta con una *Regla de Medición (Caliper)* y *Zoom Magnificador* controlados por el mouse para analizar milivoltios y milisegundos directamente sobre la onda[cite: 115, 130].
* [cite_start]**Alertas Sensoriales:** Integración de la *Web Audio API* para emitir el característico pitido analógico de 800Hz sincronizado con cada latido[cite: 961, 963].
* [cite_start]**Generación de Reportes PDF:** Procesamiento HTML/CSS en memoria RAM utilizando *WeasyPrint* para entregar reportes exportables al instante[cite: 979, 982, 983].
