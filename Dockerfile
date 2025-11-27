FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install seaborn

# Copiar todo el proyecto
COPY . .

# Crear carpetas necesarias
RUN mkdir -p uploads results logs

# Exponer puerto (Railway asigna uno dinamicamente)
ENV PORT=8000
EXPOSE $PORT

# Comando para iniciar
CMD python app.py

