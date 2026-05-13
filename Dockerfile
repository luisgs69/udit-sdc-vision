# Utilizar la imagen oficial de Ultralytics
FROM ultralytics/ultralytics:latest

# Directorio de trabajo en el contenedor
WORKDIR /usr/src/app

# Copiamos el contenido del directorio actual (contexto de build = trabajo/)
COPY . .

# Instalar dependencias (ahora requirements.txt está en la raíz del WORKDIR)
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto de la API
EXPOSE 8000

# Por defecto arrancamos la API, pero el usuario puede sobrescribir esto para entrenar
CMD ["python3", "api/main.py"]
