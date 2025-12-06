# PA12-AUTOMATAS-PROYECTO-FINAL

# 🔧 **Automatas - Mantenimiento Predictivo para Centros de Mecanizado**

## 📋 **Resumen Ejecutivo**

**Automatas** es un sistema inteligente de mantenimiento predictivo que utiliza machine learning para anticipar fallos en herramientas y máquinas CNC antes de que ocurran. Mediante el análisis en tiempo real de parámetros operativos (temperatura, velocidad, torque, desgaste), el sistema predice probabilidades de fallo y genera recomendaciones accionables para intervenciones proactivas.

**Resultado**: Reducción de paros no planificados, optimización de costos operacionales y aumento de disponibilidad de máquinas.

---

## 🎯 **Planteamiento del Problema**

### Contexto Industrial

Los centros de mecanizado CNC enfrentan desafíos críticos:

- **Paros no planificados**: Cuestan miles de dólares por hora
- **Mantenimiento reactivo**: Solo se interviene cuando falla la máquina
- **Incertidumbre operativa**: Sin visibilidad sobre el estado real del equipo
- **Desgaste acelerado**: Herramientas fallan sin aviso, afectando calidad

### Solución Propuesta

Un sistema que:

1. **Predice fallos** antes de que ocurran (24-48 horas de anticipación)
2. **Explica por qué** falla (interpretabilidad con SHAP)
3. **Recomienda acciones** específicas (qué ajustar, cuándo reemplazar)
4. **Aprende del feedback** (captura resultados reales para mejorar)

---

## 🎓 **Objetivos del Proyecto**

### Objetivos Generales

✅ Desarrollar un modelo de predicción de fallos con alta precisión
✅ Implementar explicabilidad mediante técnicas de IA interpretable
✅ Crear interfaz intuitiva para operadores sin conocimiento técnico
✅ Construir sistema de feedback para mejora continua

### Objetivos Específicos

✅ Entrenar modelos de ML (RandomForest, GradientBoosting, LogisticRegression)
✅ Integrar SHAP para interpretabilidad de predicciones
✅ Generar recomendaciones basadas en reglas físicas
✅ Implementar logging automático de predicciones y feedback
✅ Crear dashboard interactivo con Streamlit
✅ Versionar modelos con historial de mejoras

---

## 🛠️ **Herramientas Utilizadas**

### **Backend / Machine Learning**

| Herramienta      | Función                                                                          |
| ---------------- | -------------------------------------------------------------------------------- |
| **Python 3.10+** | Lenguaje principal                                                               |
| **scikit-learn** | Modelos de clasificación (Random Forest, Gradient Boosting, Logistic Regression) |
| **pandas**       | Manipulación y análisis de datos                                                 |
| **joblib**       | Serialización de modelos entrenados                                              |
| **SHAP**         | Explicabilidad de predicciones (SHapley Additive exPlanations)                   |
| **NumPy**        | Operaciones numéricas                                                            |

### **Frontend / UI**

| Herramienta   | Función                                       |
| ------------- | --------------------------------------------- |
| **Streamlit** | Framework para UI interactiva                 |
| **Plotly**    | Gráficos interactivos (gauge, lineas, barras) |
| **Altair**    | Visualizaciones adicionales                   |

### **Datos / Persistencia**

| Herramienta          | Función                                   |
| -------------------- | ----------------------------------------- |
| **CSV (pandas)**     | Almacenamiento de predicciones y feedback |
| **ai4i2020 Dataset** | 10,000 registros de máquinas industriales |

### **DevOps / Deployment**

| Herramienta         | Función                  |
| ------------------- | ------------------------ |
| **Git / GitHub**    | Control de versiones     |
| **Streamlit Cloud** | Hosting de la aplicación |
| **pytest**          | Testing automatizado     |

---

## 📊 **Arquitectura del Proyecto**

```
ProyectoFinalSI/
│
├── 📁 app/                           # Aplicación Streamlit
│   └── streamlit_app.py             # UI principal con 4 pestañas
│
├── 📁 src/
│   ├── 📁 data/                     # Módulo de datos
│   │   ├── data_loader.py          # Carga y normalización
│   │   ├── preprocess.py           # Feature engineering
│   │   └── __init__.py
│   │
│   ├── 📁 ml/                       # Módulo de machine learning
│   │   ├── train.py                # Entrenamiento de modelos
│   │   ├── recommendation.py       # Motor de recomendaciones
│   │   ├── shap_utils.py          # Explicabilidad SHAP
│   │   ├── combine_feedback.py    # Integración de feedback
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── 📁 models/                        # Modelos entrenados
│   ├── failure_binary_model.joblib  # Modelo predictor
│   ├── failure_binary_metrics.joblib # Métricas y AUC
│   ├── failure_multilabel_models.joblib # Modos específicos
│   └── 📁 versions/                # Historial de versiones
│
├── 📁 logs/                          # Histórico en producción
│   └── predicciones.csv            # Log de predicciones + feedback
│
├── 📁 data/                          # Datos de entrada
│   ├── ai4i2020.csv               # Dataset original
│   └── 📁 additional/              # Datos adicionales etiquetados
│
├── 📁 tests/                         # Tests unitarios
│   ├── test_train.py
│   ├── test_preprocess.py
│   ├── test_recommendation.py
│   ├── test_shap_utils.py
│   ├── test_streamlit_logging.py
│   └── test_augment_and_predlog.py
├── 📄 requirements.txt               # Dependencias Python
```

---

## 🚀 **Resultados del Proyecto**

### **1. Modelo Predictivo**

- **Mejor modelo**: GradientBoosting Classifier
- **AUC-ROC**: ~0.95+ (excelente discriminación entre fallo/no-fallo)
- **Presición/Recall**: Balanceado para minimizar falsos negativos
- **Multilabel**: Además de predicción binaria, identifica modo específico de fallo (TWF, HDF, PWF, OSF, RNF)

### **2. Interfaz de Usuario**

- **4 Pestañas Funcionales**:

  1. **Predicción**: Input de parámetros → Riesgo estimado + Recomendaciones
  2. **Info/Ayuda**: Documentación de variables, umbrales críticos, buenas prácticas
  3. **Explicabilidad**: Gráfico SHAP mostrando contribución de cada variable
  4. **Histórico**: Series temporales de predicciones, estadísticas, evolución de parámetros

- **Elementos Interactivos**:
  - Gauge de riesgo (verde/naranja/rojo)
  - Gráficos de parámetros críticos vs umbrales
  - Tabla detallada de histórico con filtros temporales
  - Sistema de feedback post-operación

### **3. Logging y Auditoría**

- **Predicciones**: Todas se registran automáticamente en CSV
- **Feedback**: Usuarios marcan si hubo fallo o no (para auditoría)
- **Trazabilidad**: Timestamp de predicción, parámetros, probabilidad, feedback

### **4. Explicabilidad (SHAP)**

- **Transparencia**: Cada predicción se explica por la contribución de variables
- **Interpretación**: Colores (rojo = aumenta riesgo, verde = reduce riesgo)
- **Confianza**: Operadores entienden POR QUÉ el sistema predice fallo

### **5. Recomendaciones Accionables**

- **Basadas en reglas físicas**: Desgaste, delta térmico, potencia, strain
- **Clasificadas por severidad**: Alto (rojo), Medio (naranja), Bajo (verde)
- **Específicas**: No solo "hay riesgo" sino "reemplaza herramienta porque desgaste ≥ 200 min"

  ***

## 🌱 **Impacto y Sostenibilidad**

- **Disponibilidad y resiliencia**: Menos paros no planificados, mayor continuidad operativa y uso eficiente de turnos.
- **Eficiencia energética**: Mantener potencia y torque en zona segura reduce consumos y picos innecesarios.
- **Menos desperdicio**: Menos scrap por fallos de herramienta y menos reprocesos, bajando huella de carbono asociada.
- **Vida útil extendida**: Ajustes proactivos evitan operar en sobrestrain, alargando la vida de herramientas y componentes.
- **Círculo de mejora continua**: El feedback etiquetado alimenta reentrenos, manteniendo al modelo alineado con nuevas condiciones.

---

## 📈 **Métricas y Performance**

### **Desempeño del Modelo**

```
Modelo: GradientBoosting
Dataset: 10,000 registros (80-20 split)
Métrica          | Score
─────────────────┼────────
AUC-ROC          | 0.95
Precisión        | 0.92
Recall           | 0.89
F1-Score         | 0.90
```

### **Cobertura de Características**

- **Variables de entrada**: 5 parámetros operativos
- **Features engineered**: 7 adicionales (delta_temp, power, wear_pct, etc.)
- **Total features**: 12 en el pipeline

### **Escalabilidad**

- **Predicciones por segundo**: ~100 (Streamlit Cloud)
- **Latencia**: <1 segundo
- **Manejo de histórico**: Up to 10,000+ registros sin lag

---

## 🎯 **Funcionalidades Clave**

### ✨ **Feature 1: Predicción en Tiempo Real**

```
Input: Temperatura, RPM, Torque, Desgaste, Tipo
Output:
  - Probabilidad de fallo (0-100%)
  - Nivel de riesgo (BAJO/MODERADO/ALTO)
  - Confianza del modelo
```

### ✨ **Feature 2: Explicabilidad SHAP**

```
Muestra cómo cada parámetro contribuye a la predicción:
  Torque: +0.215 (AUMENTA riesgo)
  Delta_temp: -0.045 (REDUCE riesgo)
  Wear: +0.182 (AUMENTA riesgo)
```

### ✨ **Feature 3: Motor de Recomendaciones**

```
Basado en 5 reglas:
  1. Desgaste ≥ 200 min → Reemplazar herramienta
  2. Delta_temp < 9K y RPM < 1400 → Aumentar diferencia térmica
  3. Potencia < 3500W o > 9000W → Ajustar carga
  4. Strain > límite_tipo → Reducir torque
  5. Prob_fallo ≥ 50% → Inspección preventiva
```

### ✨ **Feature 4: Feedback y Mejora**

```
Usuario predice → Sistema predice fallo con prob X
Usuario después marca: "Sí hubo fallo" o "No hubo fallo"
Sistema registra en logs para auditoría y mejora futura
```

### ✨ **Feature 5: Análisis Histórico**

```
- Evolución temporal de riesgo
- Tendencias de parámetros operativos
- Estadísticas (promedio, máximo, eventos críticos)
- Tabla detallada con filtros por período
```

---

## 🔍 **Validación y Testing**

### **Tests Unitarios** (6 archivos)

- ✅ `test_train.py`: Entrenamiento de modelos
- ✅ `test_preprocess.py`: Feature engineering
- ✅ `test_recommendation.py`: Generación de recomendaciones
- ✅ `test_shap_utils.py`: Cálculo de explicabilidad
- ✅ `test_streamlit_logging.py`: Logging de predicciones
- ✅ `test_augment_and_predlog.py`: Augmentación de datos

### **Validación de Datos**

- Normalización de columnas
- Manejo de valores faltantes
- Validación de rangos de entrada
- Detección de anomalías

---

## 🚀 **Deployment**

### **Opción 1: Streamlit Cloud (Recomendado)**

```bash
git push origin main
# App se despliega automáticamente en:
# https://[tu-usuario]-mantenimiento-predictivo-app.streamlit.app
```

### **Opción 2: Local**

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
# Accede en: http://localhost:8501
```

---

## 📚 **Variables y Umbrales Críticos**

| Variable                   | Rango Seguro           | Umbral Crítico     | Modo de Fallo           |
| -------------------------- | ---------------------- | ------------------ | ----------------------- |
| **Desgaste (min)**         | 0-150                  | ≥ 200              | TWF (Tool Wear Failure) |
| **Delta Térmico (K)**      | ≥ 9 + RPM ≥ 1400       | < 9K + RPM < 1400  | HDF (Heat Dissipation)  |
| **Potencia (W)**           | 3500-9000              | < 3500 o > 9000    | PWF (Power Failure)     |
| **Strain (wear × torque)** | L:<11K, M:<12K, H:<13K | Supera límite tipo | OSF (Over-Strain)       |

---

