#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para ejecutar la aplicación Streamlit.
"""

import os
import sys
import logging
import streamlit as st
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Asegurarse de que el directorio del proyecto esté en el path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

# Desactivar el monitoreo de módulos de PyTorch para evitar errores
os.environ["STREAMLIT_WATCH_MODULES"] = "false"

# Importar componentes necesarios
from config.config import Config
from preprocessing.text_extractor import TextExtractor
from preprocessing.text_cleaner import TextCleaner
from preprocessing.feature_extractor import FeatureExtractor
from models.ensemble import EnsembleModel
from database.db_handler import DatabaseHandler
from ui.dashboard import show_compatibility_results

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Asignación de Posiciones",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Función principal de la aplicación Streamlit."""
    
    st.title("Sistema de Asignación de Posiciones")
    
    # Cargar configuración
    config = Config()
    
    # Inicializar componentes
    text_extractor = TextExtractor(config)
    text_cleaner = TextCleaner(config)
    feature_extractor = FeatureExtractor(config)
    model = EnsembleModel(config)
    db_handler = DatabaseHandler(config)
    
    # Crear pestañas
    tab1, tab2, tab3 = st.tabs(["Subir CV", "Subir Descripción", "Comparar"])
    
    with tab1:
        st.header("Subir CV")
        
        uploaded_file = st.file_uploader("Selecciona un archivo CV", 
                                         type=["pdf", "docx", "doc", "txt", "jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Crear directorio temporal
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            
            # Guardar archivo
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"Archivo guardado: {file_path}")
            
            # Extraer texto
            with st.spinner("Extrayendo texto..."):
                cv_text = text_extractor.extract(file_path)
            
            # Limpiar texto
            with st.spinner("Procesando texto..."):
                cleaned_text = text_cleaner.clean(cv_text)
            
            # Guardar en base de datos
            with st.spinner("Guardando en base de datos..."):
                # Guardar en la base de datos (ajustando según los parámetros que acepta la función)
                cv_id = db_handler.save_cv(uploaded_file.name, cv_text, cleaned_text)
                
                # Guardar la ruta del archivo en la sesión para visualización posterior
                if 'cv_files' not in st.session_state:
                    st.session_state.cv_files = {}
                st.session_state.cv_files[uploaded_file.name] = str(file_path)
            
            if cv_id != -1:
                st.success(f"CV guardado con ID: {cv_id}")
                
                # Mostrar texto extraído
                with st.expander("Ver texto extraído"):
                    st.text_area("Texto del CV", cv_text, height=300)
            else:
                st.error("Error al guardar CV en base de datos")
    
    with tab2:
        st.header("Subir Descripción de Puesto")
        
        job_id = st.text_input("ID del puesto")
        job_title = st.text_input("Título del puesto")
        job_content = st.text_area("Descripción del puesto", height=300)
        
        if st.button("Guardar Descripción"):
            if job_id and job_title and job_content:
                # Limpiar texto
                with st.spinner("Procesando texto..."):
                    cleaned_content = text_cleaner.clean(job_content)
                
                # Guardar en base de datos
                with st.spinner("Guardando en base de datos..."):
                    desc_id = db_handler.save_job_description(job_id, job_title, job_content, cleaned_content)
                
                if desc_id != -1:
                    st.success(f"Descripción guardada con ID: {desc_id}")
                else:
                    st.error("Error al guardar descripción en base de datos")
            else:
                st.warning("Por favor, completa todos los campos")
    
    with tab3:
        st.header("Comparar CV con Descripción de Puesto")
        
        # Obtener CVs y descripciones de la base de datos
        cvs = db_handler.get_all_cvs()
        jobs = db_handler.get_all_job_descriptions()
        
        # Crear selectores
        cv_options = [cv['filename'] for cv in cvs]
        job_options = [f"{job['job_id']} - {job['title']}" for job in jobs]
        
        selected_cv = st.selectbox("Selecciona un CV", cv_options if cv_options else ["No hay CVs disponibles"])
        selected_job = st.selectbox("Selecciona un puesto", job_options if job_options else ["No hay puestos disponibles"])
        
        if st.button("Comparar"):
            if cv_options and job_options:
                # Extraer job_id del formato "job_id - title"
                job_id = selected_job.split(" - ")[0]
                
                # Obtener datos
                cv_data = db_handler.get_cv(filename=selected_cv)
                job_data = db_handler.get_job_description(job_id)
                
                if cv_data and job_data:
                    # Usar contenido procesado si está disponible
                    cv_text = cv_data.get('processed_content') or cv_data.get('content')
                    job_text = job_data.get('processed_content') or job_data.get('content')
                    
                    # Predecir compatibilidad
                    with st.spinner("Analizando compatibilidad..."):
                        score = model.predict(cv_text, job_text)
                    
                    # Guardar resultado
                    result_id = db_handler.save_result(selected_cv, job_id, score)
                    
                    # Mostrar resultados usando el dashboard
                    show_compatibility_results(cv_data, job_data, score, selected_cv, job_id, temp_dir)
                else:
                    st.error("Error al obtener datos de CV o descripción de puesto")
            else:
                st.warning("No hay CVs o descripciones de puestos disponibles para comparar")

if __name__ == "__main__":
    main()
