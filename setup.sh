#!/bin/bash

# Script de instalación para el sistema de asignación de posiciones

echo "Instalando dependencias..."

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar entorno virtual
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Descargar modelo spaCy para español
python -m spacy download es_core_news_sm

# Crear directorios necesarios
mkdir -p data/raw/cvs data/raw/job_descriptions data/processed/cvs data/processed/job_descriptions temp database

echo "Instalación completada."
echo "Para ejecutar el sistema, use uno de los siguientes comandos:"
echo "  - Interfaz web: streamlit run run_app.py"
echo "  - API REST: python run_api.py"
echo "  - Línea de comandos: python main.py --mode predict --cv_path <ruta_cv> --job_desc_path <ruta_descripcion>"
