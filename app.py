from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Header
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import logging
from typing import Optional, Dict, Any
from model.trainer import train_model_retrasos
import uvicorn
from datetime import datetime
import zipfile
from pathlib import Path
import secrets
import pandas as pd

# Configuracion de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trimax_api")

# Carpetas de trabajo
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
LOG_FOLDER = "logs"

# Crear carpetas si no existen
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, LOG_FOLDER]:
    Path(folder).mkdir(exist_ok=True)

app = FastAPI(
    title="TRIMAX – Predicción de Retrasos API",
    version="2.0",
    description="API avanzada para entrenar modelo ML de predicción de retrasos",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuracion CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Para desarrollo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Diccionario para guardar el estado de los entrenamientos
training_jobs: Dict[str, Dict[str, Any]] = {}

# Credenciales de acceso (cambiar por las que quieras)
USUARIO = "admin"
PASSWORD = "trimax2025"

# Sesiones activas
active_sessions: Dict[str, datetime] = {}

@app.post("/login")
async def login(username: str, password: str):
    """Endpoint para iniciar sesion"""
    if username == USUARIO and password == PASSWORD:
        token = secrets.token_urlsafe(16)
        active_sessions[token] = datetime.now()
        return JSONResponse({
            "success": True,
            "token": token,
            "message": "Login exitoso"
        })
    raise HTTPException(status_code=401, detail="Credenciales incorrectas")

@app.post("/logout")
async def logout(token: str):
    """Endpoint para cerrar sesion"""
    if token in active_sessions:
        del active_sessions[token]
    return JSONResponse({"success": True, "message": "Sesion cerrada"})

class TrainingManager:
    @staticmethod
    def train_model_sync(job_id: str, file_location: str, results_folder: str):
        """Ejecuta el entrenamiento de forma sincrona"""
        try:
            # Actualizar estado a running
            training_jobs[job_id].update({
                "status": "running",
                "start_time": datetime.now().isoformat()
            })
            
            logger.info(f"Iniciando entrenamiento para job {job_id}")
            
            # Ejecutar entrenamiento
            results = train_model_retrasos(file_location, results_folder=results_folder)
            
            # Crear ZIP con resultados
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"modelo_retrasos_{job_id}_{timestamp}.zip"
            zip_path = os.path.join(results_folder, zip_filename)
            
            with zipfile.ZipFile(zip_path, "w") as zf:
                files_to_zip = [
                    "modelo_retrasos.pkl",
                    "label_encoders.pkl", 
                    "dataset_predicciones.xlsx",
                    "analisis_completo.png"
                ]
                
                for file in files_to_zip:
                    file_path = os.path.join(results_folder, file)
                    if os.path.exists(file_path):
                        zf.write(file_path, file)
            
            # Actualizar estado a completed con resultados
            training_jobs[job_id].update({
                "status": "completed",
                "end_time": datetime.now().isoformat(),
                "results": results,
                "zip_file": zip_filename,
                "download_url": f"/download/{zip_filename}"
            })
            
            logger.info(f"Entrenamiento completado para job {job_id}")
            logger.info(f"Resultados: {results}")
            
        except Exception as e:
            logger.error(f"Error en entrenamiento {job_id}: {str(e)}")
            # Actualizar estado a failed
            training_jobs[job_id].update({
                "status": "failed",
                "error": str(e),
                "end_time": datetime.now().isoformat()
            })


# Servir archivos estáticos del frontend
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_path):
    app.mount("/img", StaticFiles(directory=os.path.join(frontend_path, "img")), name="img")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Servir el frontend"""
    frontend_file = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if os.path.exists(frontend_file):
        with open(frontend_file, "r", encoding="utf-8") as f:
            content = f.read()
            # Reemplazar rutas de imágenes para Railway
            content = content.replace('src="img/', 'src="/img/')
            return content
    return HTMLResponse("""
    <html>
        <body>
            <h1>TRIMAX API funcionando correctamente</h1>
            <p>Frontend no encontrado. Ve a <a href="/docs">/docs</a> para la documentación.</p>
        </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "healthy", 
        "service": "trimax_api",
        "timestamp": datetime.now().isoformat()
    })

@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    """
    Analiza el archivo y devuelve estadisticas basicas sin entrenar el modelo.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Nombre de archivo no valido")
        
        if not (file.filename.lower().endswith(('.xlsx', '.csv'))):
            raise HTTPException(
                status_code=400, 
                detail="Formato de archivo no soportado. Use .xlsx o .csv"
            )
        
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > 100 * 1024 * 1024:
            raise HTTPException(
                status_code=400, 
                detail="Archivo demasiado grande. Maximo 100MB permitido."
            )
        
        # Guardar temporalmente para analizar
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx' if file.filename.endswith('.xlsx') else '.csv') as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        try:
            # Leer y analizar
            if file.filename.endswith('.xlsx'):
                df = pd.read_excel(tmp_path)
            else:
                df = pd.read_csv(tmp_path)
            
            # Calcular estadisticas
            stats = {
                "total_registros": int(len(df)),
                "total_columnas": int(len(df.columns)),
                "columnas": list(df.columns),
                "tamano_archivo_mb": round(file_size / (1024 * 1024), 2)
            }
            
            # Si tiene fechas, calcular rango
            if 'FECHA_INICIO' in df.columns:
                try:
                    df['FECHA_INICIO'] = pd.to_datetime(df['FECHA_INICIO'], errors='coerce')
                    fechas_validas = df['FECHA_INICIO'].dropna()
                    if len(fechas_validas) > 0:
                        stats["fecha_minima"] = fechas_validas.min().strftime("%Y-%m-%d")
                        stats["fecha_maxima"] = fechas_validas.max().strftime("%Y-%m-%d")
                except:
                    pass
            
            return JSONResponse(stats)
            
        finally:
            # Eliminar archivo temporal
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        logger.error(f"Error analizando archivo: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error al analizar archivo: {str(e)}"
        )

@app.post("/train-retrasos")
async def train_retrasos(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Endpoint para entrenar el modelo de predicción de retrasos.
    """
    try:
        # Validaciones del archivo
        if not file.filename:
            raise HTTPException(status_code=400, detail="Nombre de archivo no válido")
        
        if not (file.filename.lower().endswith(('.xlsx', '.csv'))):
            raise HTTPException(
                status_code=400, 
                detail="Formato de archivo no soportado. Use .xlsx o .csv"
            )
        
        # Leer archivo
        file_content = await file.read()
        file_size = len(file_content)
        
        # Validar tamaño (max 100MB)
        if file_size > 100 * 1024 * 1024:
            raise HTTPException(
                status_code=400, 
                detail="Archivo demasiado grande. Máximo 100MB permitido."
            )
        
        # Generar ID único para el job
        job_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename.replace(' ', '_')}"
        file_location = os.path.join(UPLOAD_FOLDER, safe_filename)
        
        # Guardar archivo
        with open(file_location, "wb") as f:
            f.write(file_content)
        
        # Inicializar job con estado pending
        training_jobs[job_id] = {
            "job_id": job_id,
            "filename": safe_filename,
            "status": "pending",  # Estado inicial
            "created_at": datetime.now().isoformat(),
            "file_size": file_size
        }
        
        # Ejecutar entrenamiento en segundo plano
        background_tasks.add_task(
            TrainingManager.train_model_sync,
            job_id,
            file_location,
            RESULTS_FOLDER
        )
        
        logger.info(f"Archivo recibido: {safe_filename} ({file_size} bytes)")
        
        return JSONResponse({
            "message": "Archivo recibido y entrenamiento iniciado",
            "job_id": job_id,
            "filename": safe_filename,
            "status": "pending",
            "check_status": f"/train-status/{job_id}",
            "estimated_time": "2-5 minutos"
        })
        
    except Exception as e:
        logger.error(f"Error procesando archivo: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error interno del servidor: {str(e)}"
        )

@app.get("/train-status/{job_id}")
async def get_training_status(job_id: str):
    """
    Consultar el estado de un job de entrenamiento
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    
    job = training_jobs[job_id].copy()  # Copia para no modificar el original
    
    # Calcular tiempo transcurrido si está en progreso
    if job["status"] in ["running", "completed", "failed"] and "start_time" in job:
        start_time = datetime.fromisoformat(job["start_time"])
        elapsed = datetime.now() - start_time
        job["elapsed_seconds"] = int(elapsed.total_seconds())
    
    return job

@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Descargar archivos de resultados
    """
    try:
        # Validar nombre de archivo por seguridad
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Nombre de archivo no válido")
        
        path = os.path.join(RESULTS_FOLDER, filename)
        
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        return FileResponse(
            path, 
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"Error descargando archivo {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al descargar archivo")

# Middleware para logging
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.2f}s"
    )
    
    return response

if __name__ == "__main__":
    import webbrowser
    import threading
    import sys
    
    try:
        # Obtener puerto de variable de entorno (Railway) o usar 8000 por defecto
        port = int(os.environ.get("PORT", 8000))
        
        # Detectar si estamos en un ejecutable de PyInstaller
        if getattr(sys, 'frozen', False):
            # En ejecutable, los archivos estan en sys._MEIPASS
            base_path = sys._MEIPASS
            # Pero el frontend debe estar junto al .exe
            exe_dir = os.path.dirname(sys.executable)
            frontend_path = os.path.join(exe_dir, "frontend", "index.html")
        else:
            # En desarrollo normal
            base_path = os.path.dirname(__file__)
            frontend_path = os.path.join(base_path, "frontend", "index.html")
        
        print("=" * 50)
        print("TRIMAX API - Iniciando Servidor")
        print("=" * 50)
        print(f"Servidor: http://0.0.0.0:{port}")
        print(f"Documentacion: http://0.0.0.0:{port}/docs")
        print("=" * 50)
        print()
        
        # Solo abrir navegador si estamos en local (no en Railway)
        if port == 8000 and not os.environ.get("RAILWAY_ENVIRONMENT"):
            def abrir_navegador():
                import time
                time.sleep(3)
                if os.path.exists(frontend_path):
                    path_formatted = os.path.abspath(frontend_path).replace("\\", "/")
                    webbrowser.open(f"file:///{path_formatted}")
                    print(f"Navegador abierto: {path_formatted}")
                else:
                    webbrowser.open("http://localhost:8000")
                    print("Navegador abierto: http://localhost:8000")
            
            threading.Thread(target=abrir_navegador, daemon=True).start()
            print("Presiona Ctrl+C para detener el servidor")
        else:
            print("Servidor listo en Railway")
        
        print()
        
        # Importar la app directamente para PyInstaller
        uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        if not os.environ.get("RAILWAY_ENVIRONMENT"):
            input("Presiona Enter para salir...")
