# Desplegar TRIMAX API en Railway

## Pasos para desplegar:

### 1. Crear cuenta en Railway
- Ve a https://railway.app
- Inicia sesión con GitHub

### 2. Crear nuevo proyecto
- Click en "New Project"
- Selecciona "Deploy from GitHub repo"
- Conecta tu repositorio

### 3. Railway detectará automáticamente:
- El `Dockerfile` que creamos
- Las dependencias de `requirements.txt`

### 4. Configurar variables (opcional)
- En Railway, ve a "Variables"
- No necesitas configurar nada por defecto

### 5. Desplegar
- Railway desplegará automáticamente
- Te dará una URL como: `https://tu-proyecto.railway.app`

### 6. Acceder al sistema
- Abre la URL que Railway te dio
- Ingresa credenciales:
  - Usuario: `admin`
  - Contraseña: `trimax2025`

## Notas importantes:

- El frontend se servirá desde la misma URL de Railway
- Los archivos se guardan temporalmente (se pierden al reiniciar)
- Para persistencia, necesitarías agregar una base de datos

## Cambiar credenciales:

Edita `app.py` líneas 57-58 y vuelve a desplegar:
```python
USUARIO = "tu_usuario"
PASSWORD = "tu_password"
```


