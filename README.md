# TRIMAX API - Sistema Predictivo de Retrasos con Machine Learning

Sistema web completo para predecir retrasos en órdenes de fabricación utilizando Machine Learning. Permite entrenar modelos de Random Forest a partir de datos históricos y obtener predicciones, análisis y visualizaciones.

## 🚀 Características

- **Interfaz Web Moderna**: Diseño futurista con tema oscuro y animaciones
- **Entrenamiento de Modelos ML**: Random Forest Classifier para predicción de retrasos
- **Análisis Completo**: Gráficos de variables importantes, matriz de confusión, análisis temporal
- **Subida de Archivos**: Soporte para archivos Excel (.xlsx) y CSV
- **Procesamiento Asíncrono**: Entrenamiento en segundo plano con seguimiento de estado
- **Autenticación**: Sistema de login con tokens de sesión
- **Deployment**: Desplegado en Render.com con Docker

## 🛠️ Tecnologías Utilizadas

### Backend
- **FastAPI**: Framework web moderno y rápido
- **Python 3.11**: Lenguaje de programación
- **scikit-learn**: Machine Learning (Random Forest)
- **Pandas**: Manipulación y análisis de datos
- **Matplotlib/Seaborn**: Visualización de datos
- **Uvicorn**: Servidor ASGI

### Frontend
- **HTML5/CSS3**: Estructura y estilos
- **JavaScript**: Lógica del cliente
- **Chart.js**: Gráficos interactivos
- **Google Fonts**: Tipografías (Orbitron, Rajdhani)

### Deployment
- **Docker**: Contenedorización
- **Render.com**: Plataforma de hosting (PaaS)
- **GitHub**: Control de versiones

## 📋 Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

## 🔧 Instalación Local

1. **Clonar o descargar el repositorio**
   ```bash
   git clone https://github.com/hiroshihv89/PROYECTO_TRIMAX_ML.git
   cd PROYECTO_TRIMAX_ML
   ```

2. **Crear entorno virtual**
   ```bash
   python -m venv venv
   ```

3. **Activar entorno virtual**
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

4. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

5. **Ejecutar el servidor**
   ```bash
   python app.py
   ```

6. **Abrir en el navegador**
   ```
   http://localhost:8000
   ```

7. **Credenciales de acceso**
   - Usuario: `admin`
   - Contraseña: `trimax2025`

## 📁 Estructura del Proyecto

```
trimax_api/
│
├── app.py                    # API principal (FastAPI)
├── requirements.txt         # Dependencias Python
├── Dockerfile              # Configuración Docker
├── .gitignore              # Archivos a ignorar
│
├── model/
│   └── trainer.py          # Entrenamiento del modelo ML
│
├── frontend/
│   ├── index.html          # Interfaz de usuario
│   └── img/                # Imágenes y recursos
│
├── uploads/                 # Archivos Excel subidos
├── results/                 # Modelos y resultados generados
└── logs/                    # Archivos de log
```

## 🎯 Uso

1. **Iniciar sesión** con las credenciales proporcionadas
2. **Subir archivo Excel** con datos históricos de órdenes de fabricación
3. **Iniciar entrenamiento** haciendo clic en el botón "Entrenar Modelo"
4. **Esperar procesamiento** (2-5 minutos aproximadamente)
5. **Visualizar resultados**: accuracy, variables importantes, gráficos
6. **Descargar resultados** en formato ZIP (modelo, gráficos, predicciones)

## 📊 Formato de Datos

El archivo Excel debe contener las siguientes columnas:

- `FECHA_INICIO`: Fecha de inicio de la orden
- `FECHA_TERMINO`: Fecha de término de la orden
- `PLANTA`: Nombre de la planta
- `SEDE`: Nombre de la sede
- `TIPO`: Tipo de orden (FABRICACION, BISELADO, etc.)
- `SIMTIPO`: Subtipo
- `PRODUCTO`: Nombre del producto
- `TIPO TRATAMIENTO`: Tipo de tratamiento

## 🔍 Endpoints de la API

- `POST /login` - Iniciar sesión
- `POST /logout` - Cerrar sesión
- `POST /train-retrasos` - Subir archivo e iniciar entrenamiento
- `GET /train-status/{job_id}` - Consultar estado del entrenamiento
- `GET /download/{filename}` - Descargar resultados
- `GET /docs` - Documentación interactiva (Swagger UI)
- `GET /health` - Estado del servidor

## 🌐 Deployment

El proyecto está desplegado en Render.com:

**URL:** https://proyecto-trimax-ml.onrender.com

## 📈 Modelo de Machine Learning

- **Algoritmo**: Random Forest Classifier
- **Parámetros**:
  - `n_estimators=200`: Número de árboles
  - `max_depth=15`: Profundidad máxima
  - `class_weight='balanced'`: Balanceo de clases
- **Feature Engineering**: Extracción de características temporales (año, mes, día, día de semana, trimestre)
- **Evaluación**: Accuracy, matriz de confusión, feature importance

## 📝 Notas

- El plan gratuito de Render puede "dormir" el servicio tras 15 minutos de inactividad. La primera carga puede tardar 30-60 segundos.
- Los archivos subidos deben tener un tamaño máximo de 100MB.
- El entrenamiento se ejecuta en segundo plano para no bloquear la interfaz.

## 📄 Licencia

Este proyecto fue desarrollado como parte de un trabajo académico de SENATI.

---

## 👥 Integrantes

**GRUPO 5:**

- ESPINOZA SAAVEDRA, DAVID ANTONIO
- LUPACA AGUILAR, HULK KING
- HERNÁNDEZ VICENTE, EFRÉN HIROSHI
- ROBLES CASTRO, JEAN CESAR
- GUTIERREZ RODRIGUEZ, SHIRLEY CAROLINA
- CLEMENTE RAMOS, JHORDAN MICHAEL

**Instructor:**

- MORALES CARLOS, ALDO OMAR
