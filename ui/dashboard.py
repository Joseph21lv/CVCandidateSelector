#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interfaz gráfica para el sistema de asignación de posiciones basado en CVs.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

import streamlit as st
from streamlit_option_menu import option_menu

# Agregar directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from preprocessing.text_extractor import TextExtractor
from preprocessing.text_cleaner import TextCleaner
from preprocessing.feature_extractor import FeatureExtractor
from models.ensemble import EnsembleModel
from database.db_handler import DatabaseHandler

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar configuración
config = Config()

# Inicializar componentes
text_extractor = TextExtractor(config)
text_cleaner = TextCleaner(config)
feature_extractor = FeatureExtractor(config)
model = EnsembleModel(config)
db_handler = DatabaseHandler(config)

# Eliminar la configuración de página duplicada

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #424242;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.25rem 0.75rem rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e3f2fd;
        box-shadow: 0 0.25rem 0.75rem rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1976D2;
    }
    .metric-label {
        font-size: 1rem;
        color: #424242;
    }
    .highlight {
        background-color: #ffff00;
        padding: 0.2rem;
        border-radius: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Función principal de la interfaz gráfica."""
    # Menú lateral
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/resume.png", width=100)
        st.markdown("## Sistema de Asignación de Posiciones")
        
        selected = option_menu(
            menu_title=None,
            options=["Inicio", "Cargar CV", "Gestionar Puestos", "Asignar Posiciones", "Resultados", "Configuración"],
            icons=["house", "file-earmark-text", "briefcase", "check2-square", "bar-chart", "gear"],
            menu_icon="cast",
            default_index=0
        )
    
    # Contenido principal según la opción seleccionada
    if selected == "Inicio":
        show_home()
    elif selected == "Cargar CV":
        show_upload_cv()
    elif selected == "Gestionar Puestos":
        show_manage_jobs()
    elif selected == "Asignar Posiciones":
        show_match_cv_job()
    elif selected == "Resultados":
        show_results()
    elif selected == "Configuración":
        show_settings()

def show_home():
    """Muestra la página de inicio."""
    st.markdown('<div class="main-header">Sistema de Asignación de Posiciones</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Este sistema utiliza técnicas avanzadas de Procesamiento de Lenguaje Natural (PLN) y aprendizaje automático
    para analizar currículums vitae (CVs) y asignarlos a posiciones laborales adecuadas.
    """)
    
    # Estadísticas generales
    st.markdown('<div class="sub-header">Estadísticas Generales</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # Obtener datos para estadísticas
    cvs = db_handler.get_all_cvs()
    jobs = db_handler.get_all_job_descriptions()
    results = db_handler.get_results()
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(cvs)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">CVs Procesados</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(jobs)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Posiciones Disponibles</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(results)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Asignaciones Realizadas</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Gráfico de puntuaciones recientes
    if results:
        st.markdown('<div class="sub-header">Puntuaciones Recientes</div>', unsafe_allow_html=True)
        
        # Convertir resultados a DataFrame
        df_results = pd.DataFrame(results)
        
        # Limitar a los 20 resultados más recientes
        df_recent = df_results.head(20)
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Obtener títulos de puestos
        job_titles = []
        for job_id in df_recent['job_id']:
            job_data = db_handler.get_job_description(job_id)
            job_titles.append(job_data.get('title', job_id) if job_data else job_id)
        
        # Crear etiquetas para el gráfico
        labels = [f"{cv[:10]}... - {job[:15]}..." for cv, job in zip(df_recent['cv_filename'], job_titles)]
        
        # Crear barras con colores según puntuación
        bars = ax.barh(labels, df_recent['score'], color=plt.cm.RdYlGn(df_recent['score']))
        
        # Configurar gráfico
        ax.set_xlabel('Puntuación de Compatibilidad')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Mostrar valores en las barras
        for i, v in enumerate(df_recent['score']):
            ax.text(v + 0.01, i, f"{v:.2f}", va='center')
        
        st.pyplot(fig)
    
    # Información del sistema
    st.markdown('<div class="sub-header">Información del Sistema</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Modelos Utilizados")
        st.markdown("""
        - **Naive Bayes**: Clasificación basada en palabras clave
        - **SBERT**: Comparación semántica avanzada
        - **BERT/RoBERTa**: Fine-tuning para clasificación específica
        - **Ensemble**: Combinación ponderada de modelos
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Características Principales")
        st.markdown("""
        - Procesamiento de CVs en múltiples formatos (PDF, DOCX, TXT, imágenes)
        - Extracción de entidades y habilidades relevantes
        - Comparación semántica con descripciones de puestos
        - Visualización de resultados y exportación de datos
        """)
        st.markdown('</div>', unsafe_allow_html=True)

def show_upload_cv():
    """Muestra la página para cargar CVs."""
    st.markdown('<div class="main-header">Cargar Currículum Vitae</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Cargue un CV en formato PDF, DOCX, TXT o imagen (JPG, PNG). El sistema extraerá automáticamente
    el texto y lo procesará para su análisis.
    """)
    
    # Crear directorio para almacenamiento permanente de PDFs
    pdf_storage_dir = Path("data/pdf_storage")
    pdf_storage_dir.mkdir(parents=True, exist_ok=True)
    
    # Formulario de carga
    with st.form("upload_cv_form"):
        uploaded_file = st.file_uploader("Seleccione un archivo", type=['pdf', 'docx', 'doc', 'txt', 'jpg', 'jpeg', 'png'])
        submit_button = st.form_submit_button("Procesar CV")
    
    if submit_button and uploaded_file is not None:
        with st.spinner("Procesando CV..."):
            # Guardar archivo temporalmente
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / 'cv_matcher_uploads'
            temp_dir.mkdir(exist_ok=True)
            
            file_path = temp_dir / uploaded_file.name
            
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Extraer texto
            cv_text = text_extractor.extract(file_path)
            
            # Limpiar texto
            cleaned_text = text_cleaner.clean(cv_text)
            
            # Guardar archivo en almacenamiento permanente si es PDF
            if uploaded_file.name.lower().endswith('.pdf'):
                storage_path = pdf_storage_dir / uploaded_file.name
                with open(storage_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                logger.info(f"PDF guardado en almacenamiento permanente: {storage_path}")
                
                # Guardar la ruta en la sesión
                if 'cv_files' not in st.session_state:
                    st.session_state.cv_files = {}
                st.session_state.cv_files[uploaded_file.name] = str(storage_path)
            
            # Guardar en base de datos
            cv_id = db_handler.save_cv(uploaded_file.name, cv_text, cleaned_text)
            
            if cv_id != -1:
                st.success(f"CV procesado correctamente (ID: {cv_id})")
                
                # Mostrar texto extraído
                with st.expander("Ver texto extraído"):
                    st.text_area("Texto original", cv_text, height=200)
                
                with st.expander("Ver texto procesado"):
                    st.text_area("Texto procesado", cleaned_text, height=200)
                
                # Extraer entidades
                entities = text_cleaner.extract_entities(cv_text)
                
                if entities:
                    with st.expander("Ver entidades detectadas"):
                        for entity_type, entity_list in entities.items():
                            st.markdown(f"**{entity_type}**: {', '.join(entity_list)}")
            else:
                st.error("Error al procesar el CV")
    
    # Mostrar CVs cargados
    st.markdown('<div class="sub-header">CVs Cargados</div>', unsafe_allow_html=True)
    
    cvs = db_handler.get_all_cvs()
    
    if cvs:
        # Convertir a DataFrame
        df_cvs = pd.DataFrame(cvs)
        
        # Formatear fecha
        if 'upload_date' in df_cvs.columns:
            df_cvs['upload_date'] = pd.to_datetime(df_cvs['upload_date']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Mostrar tabla
        st.dataframe(df_cvs[['id', 'filename', 'upload_date']], use_container_width=True)
        
        # Opción para eliminar CV
        with st.expander("Eliminar CV"):
            cv_to_delete = st.selectbox("Seleccione CV a eliminar", 
                                        options=[f"{cv['id']} - {cv['filename']}" for cv in cvs],
                                        format_func=lambda x: x)
            
            if st.button("Eliminar"):
                cv_id = int(cv_to_delete.split(" - ")[0])
                # Aquí iría la lógica para eliminar el CV
                st.warning(f"Funcionalidad de eliminación no implementada para CV ID: {cv_id}")
    else:
        st.info("No hay CVs cargados en el sistema")

def show_manage_jobs():
    """Muestra la página para gestionar puestos de trabajo."""
    st.markdown('<div class="main-header">Gestionar Posiciones</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Agregue, edite o elimine descripciones de puestos de trabajo. Estas descripciones se utilizarán
    para comparar con los CVs y determinar la compatibilidad.
    """)
    
    # Pestañas para diferentes acciones
    tab1, tab2, tab3 = st.tabs(["Agregar Posición", "Ver Posiciones", "Importar Posiciones"])
    
    with tab1:
        # Formulario para agregar posición
        with st.form("add_job_form"):
            job_id = st.text_input("ID de la Posición")
            job_title = st.text_input("Título de la Posición")
            job_description = st.text_area("Descripción de la Posición", height=300)
            
            submit_button = st.form_submit_button("Guardar Posición")
        
        if submit_button:
            if not job_id or not job_title or not job_description:
                st.error("Todos los campos son obligatorios")
            else:
                # Limpiar texto
                cleaned_description = text_cleaner.clean(job_description)
                
                # Guardar en base de datos
                desc_id = db_handler.save_job_description(job_id, job_title, job_description, cleaned_description)
                
                if desc_id != -1:
                    st.success(f"Posición guardada correctamente (ID: {desc_id})")
                else:
                    st.error("Error al guardar la posición")
    
    with tab2:
        # Mostrar posiciones existentes
        jobs = db_handler.get_all_job_descriptions()
        
        if jobs:
            # Convertir a DataFrame
            df_jobs = pd.DataFrame(jobs)
            
            # Formatear fecha
            if 'creation_date' in df_jobs.columns:
                df_jobs['creation_date'] = pd.to_datetime(df_jobs['creation_date']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Mostrar tabla
            st.dataframe(df_jobs[['id', 'job_id', 'title', 'creation_date']], use_container_width=True)
            
            # Ver detalles de posición
            job_to_view = st.selectbox("Seleccione posición para ver detalles", 
                                      options=[f"{job['id']} - {job['title']}" for job in jobs],
                                      format_func=lambda x: x)
            
            if job_to_view:
                job_id = job_to_view.split(" - ")[0]
                
                # Buscar posición seleccionada
                selected_job = None
                for job in jobs:
                    if str(job['id']) == job_id:
                        selected_job = job
                        break
                
                if selected_job:
                    # Obtener detalles completos
                    job_details = db_handler.get_job_description(selected_job['job_id'])
                    
                    if job_details:
                        st.markdown(f"### {job_details.get('title')}")
                        st.markdown(f"**ID de Posición:** {job_details.get('job_id')}")
                        
                        with st.expander("Ver descripción completa"):
                            st.markdown(job_details.get('content', ''))
                        
                        with st.expander("Ver texto procesado"):
                            st.text_area("Texto procesado", job_details.get('processed_content', ''), height=200)
        else:
            st.info("No hay posiciones registradas en el sistema")
    
    with tab3:
        # Importar posiciones desde CSV
        st.markdown("### Importar Posiciones desde CSV")
        
        st.markdown("""
        Cargue un archivo CSV con las siguientes columnas:
        - `job_id`: Identificador único de la posición
        - `title`: Título de la posición
        - `description`: Descripción completa de la posición
        """)
        
        uploaded_file = st.file_uploader("Seleccione archivo CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Verificar columnas requeridas
                required_columns = ['job_id', 'title', 'description']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Faltan columnas requeridas: {', '.join(missing_columns)}")
                else:
                    st.dataframe(df.head(), use_container_width=True)
                    
                    if st.button("Importar Posiciones"):
                        with st.spinner("Importando posiciones..."):
                            success_count = 0
                            error_count = 0
                            
                            for _, row in df.iterrows():
                                # Limpiar texto
                                cleaned_description = text_cleaner.clean(row['description'])
                                
                                # Guardar en base de datos
                                desc_id = db_handler.save_job_description(
                                    row['job_id'], row['title'], row['description'], cleaned_description
                                )
                                
                                if desc_id != -1:
                                    success_count += 1
                                else:
                                    error_count += 1
                            
                            st.success(f"Importación completada: {success_count} posiciones importadas correctamente, {error_count} errores")
            except Exception as e:
                st.error(f"Error al procesar el archivo CSV: {e}")

def show_match_cv_job():
    """Muestra la página para asignar CVs a posiciones."""
    st.markdown('<div class="main-header">Asignar CVs a Posiciones</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Compare CVs con descripciones de puestos para determinar la compatibilidad. El sistema
    analizará el contenido de ambos documentos y calculará una puntuación de compatibilidad.
    """)
    
    # Crear directorio para almacenar PDFs si no existe
    pdf_storage_dir = Path("data/pdf_storage")
    pdf_storage_dir.mkdir(parents=True, exist_ok=True)
    
    # Obtener CVs y posiciones disponibles
    cvs = db_handler.get_all_cvs()
    jobs = db_handler.get_all_job_descriptions()
    
    if not cvs:
        st.warning("No hay CVs cargados en el sistema. Por favor, cargue al menos un CV.")
        return
    
    if not jobs:
        st.warning("No hay posiciones registradas en el sistema. Por favor, agregue al menos una posición.")
        return
    
    # Formulario para asignar CV a posición
    col1, col2 = st.columns(2)
    
    with col1:
        selected_cv = st.selectbox("Seleccione CV", 
                                  options=[cv['filename'] for cv in cvs],
                                  format_func=lambda x: x)
    
    with col2:
        selected_job = st.selectbox("Seleccione Posición", 
                                   options=[f"{job['job_id']} - {job['title']}" for job in jobs],
                                   format_func=lambda x: x)
    
    if st.button("Analizar Compatibilidad"):
        if selected_cv and selected_job:
            with st.spinner("Analizando compatibilidad..."):
                # Extraer job_id
                job_id = selected_job.split(" - ")[0]
                
                # Obtener datos
                cv_data = db_handler.get_cv(filename=selected_cv)
                job_data = db_handler.get_job_description(job_id)
                
                if cv_data and job_data:
                    # Usar contenido procesado si está disponible
                    cv_text = cv_data.get('processed_content') or cv_data.get('content')
                    job_text = job_data.get('processed_content') or job_data.get('content')
                    
                    # Predecir compatibilidad
                    score = model.predict(cv_text, job_text)
                    
                    # Guardar resultado
                    result_id = db_handler.save_result(selected_cv, job_id, score)
                    
                    # Mostrar resultado
                    st.success("Análisis completado")
                    
                    # Crear medidor visual
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        # Determinar color según puntuación
                        if score >= 0.7:
                            color = "green"
                            message = "Alta compatibilidad"
                        elif score >= 0.4:
                            color = "orange"
                            message = "Compatibilidad media"
                        else:
                            color = "red"
                            message = "Baja compatibilidad"
                        
                        # Crear medidor
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <div style="font-size: 5rem; color: {color};">{score*100:.0f}%</div>
                            <div style="font-size: 1.5rem; color: {color};">{message}</div>
                            <div style="margin: 20px 0; background-color: #e0e0e0; border-radius: 10px; height: 20px;">
                                <div style="width: {score*100}%; background-color: {color}; height: 20px; border-radius: 10px;"></div>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <div>0.0</div>
                                <div>0.5</div>
                                <div>1.0</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Mostrar detalles
                    st.markdown("### Detalles del Análisis")
                    
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.markdown(f"**CV:** {selected_cv}")
                        with st.expander("Ver contenido del CV"):
                            st.text_area("Texto del CV", cv_text, height=200)
                    
                    with col2:
                        st.markdown(f"**Posición:** {job_data.get('title')}")
                        with st.expander("Ver descripción de la posición"):
                            st.text_area("Descripción", job_text, height=200)
                    
                    with col3:
                        st.markdown("**Acciones**")
                        
                        # Añadir botón para descargar el PDF sin procesar
                        if selected_cv.lower().endswith('.pdf'):
                            # Buscar el archivo PDF
                            pdf_path = None
                            possible_paths = [
                                Path("temp") / selected_cv,
                                Path("raw_cv_dir") / selected_cv,
                                Path("processed_cv_dir") / selected_cv,
                                Path("data/cvs") / selected_cv,
                                Path.cwd() / "temp" / selected_cv,
                                Path.cwd() / "raw_cv_dir" / selected_cv,
                                Path.cwd() / "processed_cv_dir" / selected_cv,
                                Path.cwd() / "data" / "cvs" / selected_cv
                            ]
                            
                            # Verificar si existe la ruta en la sesión
                            session_path = st.session_state.cv_files.get(selected_cv) if hasattr(st.session_state, 'cv_files') else None
                            if session_path and Path(session_path).exists():
                                pdf_path = session_path
                            else:
                                # Buscar en las rutas posibles
                                for path in possible_paths:
                                    if path.exists():
                                        pdf_path = str(path)
                                        break
                                
                                # Si no se encuentra, buscar en todo el proyecto
                                if not pdf_path:
                                    import glob
                                    project_dir = Path(__file__).parent.parent
                                    pdf_files = list(glob.glob(f"{project_dir}/**/{selected_cv}", recursive=True))
                                    if pdf_files:
                                        pdf_path = pdf_files[0]
                            
                            # Si se encontró el PDF, mostrar botón de descarga
                            if pdf_path:
                                # Copiar el PDF a la carpeta de almacenamiento permanente
                                import shutil
                                storage_path = pdf_storage_dir / selected_cv
                                try:
                                    shutil.copy2(pdf_path, storage_path)
                                    logger.info(f"PDF copiado a almacenamiento permanente: {storage_path}")
                                except Exception as e:
                                    logger.error(f"Error al copiar PDF: {e}")
                                
                                # Usar la ruta de almacenamiento permanente si existe
                                final_path = storage_path if storage_path.exists() else pdf_path
                                
                                with open(final_path, "rb") as file:
                                    pdf_bytes = file.read()
                                    st.download_button(
                                        label="Descargar PDF sin procesar",
                                        data=pdf_bytes,
                                        file_name=selected_cv,
                                        mime="application/pdf",
                                        key=f"download_pdf_match_{selected_cv}_{hash(pdf_bytes)}"
                                    )
                                    
                                    # Añadir botón para abrir el PDF en una nueva pestaña
                                    # pdf_url = f"file://{final_path}"
                                    # st.markdown(f'<a href="{pdf_url}" target="_blank">Abrir PDF en nueva pestaña</a>', unsafe_allow_html=True)
                    
                    # Mostrar puntuaciones por modelo (si es un ensamble)
                    if hasattr(model, 'models') and model.models:
                        st.markdown("### Puntuaciones por Modelo")
                        
                        model_scores = {}
                        for model_name, model_instance in model.models.items():
                            model_scores[model_name] = model_instance.predict(cv_text, job_text)
                        
                        # Crear gráfico de barras
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        bars = ax.bar(model_scores.keys(), model_scores.values(), color=plt.cm.viridis(np.linspace(0, 1, len(model_scores))))
                        
                        # Añadir etiquetas
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{height:.2f}', ha='center', va='bottom')
                        
                        ax.set_ylim(0, 1)
                        ax.set_ylabel('Puntuación')
                        ax.set_title('Puntuaciones por Modelo')
                        
                        st.pyplot(fig)
                else:
                    st.error("Error al obtener datos del CV o la posición")

def show_results():
    """Muestra la página de resultados y análisis."""
    st.markdown('<div class="main-header">Resultados y Análisis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Visualice y analice los resultados de las asignaciones de CVs a posiciones. Explore tendencias,
    compare puntuaciones y exporte datos para análisis adicionales.
    """)
    
    # Obtener todos los resultados
    results = db_handler.get_results()
    
    if not results:
        st.info("No hay resultados disponibles. Realice algunas asignaciones primero.")
        return
    
    # Convertir a DataFrame
    df_results = pd.DataFrame(results)
    
    # Formatear fecha
    if 'prediction_date' in df_results.columns:
        df_results['prediction_date'] = pd.to_datetime(df_results['prediction_date']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Agregar información de puestos
    job_titles = []
    for job_id in df_results['job_id']:
        job_data = db_handler.get_job_description(job_id)
        job_titles.append(job_data.get('title', job_id) if job_data else job_id)
    
    df_results['job_title'] = job_titles
    
    # Pestañas para diferentes vistas
    tab1, tab2, tab3 = st.tabs(["Tabla de Resultados", "Visualizaciones", "Exportar Datos"])
    
    with tab1:
        # Filtros
        col1, col2 = st.columns(2)
        
        with col1:
            min_score = st.slider("Puntuación mínima", 0.0, 1.0, 0.0, 0.05)
        
        with col2:
            selected_model = st.selectbox("Modelo", 
                                         options=["Todos"] + df_results['model_used'].unique().tolist())
        
        # Aplicar filtros
        filtered_df = df_results.copy()
        
        if min_score > 0:
            filtered_df = filtered_df[filtered_df['score'] >= min_score]
        
        if selected_model != "Todos":
            filtered_df = filtered_df[filtered_df['model_used'] == selected_model]
        
        # Mostrar tabla
        st.dataframe(filtered_df[['cv_filename', 'job_id', 'job_title', 'score', 'model_used', 'prediction_date']], 
                    use_container_width=True)
        
        # Estadísticas básicas
        st.markdown("### Estadísticas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Puntuación Media", f"{filtered_df['score'].mean():.2f}")
        
        with col2:
            st.metric("Puntuación Máxima", f"{filtered_df['score'].max():.2f}")
        
        with col3:
            st.metric("Puntuación Mínima", f"{filtered_df['score'].min():.2f}")
    
    with tab2:
        # Visualizaciones
        st.markdown("### Distribución de Puntuaciones")
        
        # Histograma de puntuaciones
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.histplot(df_results['score'], bins=20, kde=True, ax=ax)
        
        ax.set_xlabel('Puntuación')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Puntuaciones de Compatibilidad')
        
        st.pyplot(fig)
        
        # Puntuaciones por posición
        st.markdown("### Puntuaciones por Posición")
        
        # Agrupar por posición
        job_scores = df_results.groupby('job_title')['score'].agg(['mean', 'min', 'max', 'count']).reset_index()
        job_scores = job_scores.sort_values('mean', ascending=False)
        
        # Gráfico de barras
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(job_scores['job_title'], job_scores['mean'], 
                     yerr=[(job_scores['mean'] - job_scores['min']), (job_scores['max'] - job_scores['mean'])],
                     capsize=5, color=plt.cm.viridis(np.linspace(0, 1, len(job_scores))))
        
        # Rotar etiquetas
        plt.xticks(rotation=45, ha='right')
        
        # Ajustar diseño
        ax.set_xlabel('Posición')
        ax.set_ylabel('Puntuación Media')
        ax.set_title('Puntuación Media por Posición')
        ax.set_ylim(0, 1)
        
        # Añadir etiquetas de conteo
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width()/2, 0.05, 
                   f"n={job_scores['count'].iloc[i]}", 
                   ha='center', va='bottom', color='white', fontweight='bold')
        
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Tendencia temporal
        st.markdown("### Tendencia Temporal")
        
        # Convertir a datetime para gráfico temporal
        df_time = df_results.copy()
        df_time['prediction_date'] = pd.to_datetime(df_results['prediction_date'])
        
        # Agrupar por día
        df_time['date'] = df_time['prediction_date'].dt.date
        daily_scores = df_time.groupby('date')['score'].mean().reset_index()
        
        # Gráfico de línea
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(daily_scores['date'], daily_scores['score'], marker='o', linestyle='-', linewidth=2)
        
        # Ajustar diseño
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Puntuación Media')
        ax.set_title('Tendencia de Puntuaciones a lo Largo del Tiempo')
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Formatear eje x
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        st.pyplot(fig)
    
    with tab3:
        # Exportar datos
        st.markdown("### Exportar Resultados")
        
        export_format = st.selectbox("Formato de exportación", ["CSV", "Excel"])
        
        if st.button("Exportar Datos"):
            # Crear DataFrame para exportación
            export_df = df_results[['cv_filename', 'job_id', 'job_title', 'score', 'model_used', 'prediction_date']]
            
            # Generar nombre de archivo
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if export_format == "CSV":
                # Exportar a CSV
                csv = export_df.to_csv(index=False)
                
                # Crear botón de descarga
                st.download_button(
                    label="Descargar CSV",
                    data=csv,
                    file_name=f"resultados_cv_matcher_{timestamp}.csv",
                    mime="text/csv"
                )
            else:
                # Exportar a Excel
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    export_df.to_excel(writer, sheet_name='Resultados', index=False)
                
                # Crear botón de descarga
                st.download_button(
                    label="Descargar Excel",
                    data=buffer.getvalue(),
                    file_name=f"resultados_cv_matcher_{timestamp}.xlsx",
                    mime="application/vnd.ms-excel"
                )

def show_settings():
    """Muestra la página de configuración."""
    global model
    global model
    st.markdown('<div class="main-header">Configuración del Sistema</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Configure los parámetros del sistema, ajuste los modelos y gestione las opciones de procesamiento.
    """)
    
    # Pestañas para diferentes secciones de configuración
    tab1, tab2, tab3 = st.tabs(["Modelos", "Preprocesamiento", "Sistema"])
    
    with tab1:
        st.markdown("### Configuración de Modelos")
        
        # Obtener configuración actual
        model_config = config.get('models', {})
        ensemble_config = model_config.get('ensemble', {}) if model_config else {}
        weights = ensemble_config.get('weights', {
            'naive_bayes': 0.3,
            'sbert': 0.3,
            'bert': 0.3
        })
        
        st.info("Los pesos actuales del modelo de ensamble son: Naive Bayes: {}, SBERT: {}, BERT: {}".format(
            weights.get('naive_bayes', 0.3),
            weights.get('sbert', 0.3),
            weights.get('bert', 0.3)
        ))
        
        st.markdown("""
        ### Cómo cambiar los pesos del modelo de ensamble:
        
        1. Ajusta los deslizadores a continuación para modificar los pesos
        2. Haz clic en "Guardar Configuración de Modelos"
        3. Los nuevos pesos se aplicarán a las futuras predicciones
        
        Nota: La suma de los pesos se normalizará automáticamente.
        """)
        
        # Ajustar pesos del ensamble
        st.markdown("#### Pesos del Modelo de Ensamble")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            naive_bayes_weight = st.slider("Peso de Naive Bayes", 0.0, 1.0, weights.get('naive_bayes', 0.3), 0.05)
        
        with col2:
            sbert_weight = st.slider("Peso de SBERT", 0.0, 1.0, weights.get('sbert', 0.3), 0.05)
        
        with col3:
            bert_weight = st.slider("Peso de BERT", 0.0, 1.0, weights.get('bert', 0.3), 0.05)
        
        # Normalizar pesos
        total_weight = naive_bayes_weight + sbert_weight + bert_weight
        
        if total_weight > 0:
            normalized_weights = {
                'naive_bayes': naive_bayes_weight / total_weight,
                'sbert': sbert_weight / total_weight,
                'bert': bert_weight / total_weight
            }
            
            # Mostrar pesos normalizados
            st.markdown("#### Pesos Normalizados")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Naive Bayes", f"{normalized_weights['naive_bayes']:.2f}")
            
            with col2:
                st.metric("SBERT", f"{normalized_weights['sbert']:.2f}")
            
            with col3:
                st.metric("BERT", f"{normalized_weights['bert']:.2f}")
            
            # Gráfico de pesos
            fig, ax = plt.subplots(figsize=(8, 8))
            
            ax.pie(
                normalized_weights.values(), 
                labels=normalized_weights.keys(),
                autopct='%1.1f%%',
                startangle=90,
                colors=plt.cm.viridis(np.linspace(0, 0.8, len(normalized_weights)))
            )
            ax.axis('equal')
            
            st.pyplot(fig)
            
            # Botón para guardar configuración
            if st.button("Guardar Configuración de Modelos"):
                # Actualizar configuración
                try:
                    config.set_config('models', {'ensemble': {'weights': normalized_weights}})
                except AttributeError:
                    if hasattr(config, 'config_data'):
                        if 'models' not in config.config_data:
                            config.config_data['models'] = {}
                        config.config_data['models']['ensemble'] = {'weights': normalized_weights}
                
                # Guardar la configuración en el archivo
                try:
                    config.save_config()
                except AttributeError:
                    # Si el método save_config no existe, intentar guardar de otra manera
                    try:
                        config.save()
                    except AttributeError:
                        # Intentar escribir directamente en el archivo de configuración
                        try:
                            import yaml
                            # Asegurarse de que la estructura de datos sea correcta
                            if not hasattr(config, 'config'):
                                config.config = {}
                            
                            if 'models' not in config.config:
                                config.config['models'] = {}
                            
                            if 'ensemble' not in config.config['models']:
                                config.config['models']['ensemble'] = {}
                            
                            config.config['models']['ensemble']['weights'] = normalized_weights
                            
                            # Guardar en el archivo
                            with open('config/default.yaml', 'w') as f:
                                yaml.dump(config.config, f)
                            
                            # Mostrar mensaje de éxito con instrucciones
                            st.success("Configuración guardada en config/default.yaml")
                            st.info("Para aplicar los cambios en los pesos, reinicia la aplicación o ejecuta nuevamente el análisis de compatibilidad.")
                        except Exception as e:
                            st.warning(f"No se pudo guardar la configuración automáticamente: {str(e)}. Los cambios se aplicarán solo para esta sesión.")
                
                # Reinicializar modelo de ensamble
                try:
                    # Guardar los pesos directamente en el archivo de configuración
                    import yaml
                    config_path = Path(os.path.dirname(os.path.dirname(__file__))) / 'config' / 'default.yaml'
                    
                    # Leer el archivo de configuración actual
                    if config_path.exists():
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config_data = yaml.safe_load(f) or {}
                    else:
                        config_data = {}
                    
                    # Asegurarse de que la estructura exista
                    if 'models' not in config_data:
                        config_data['models'] = {}
                    if 'ensemble' not in config_data['models']:
                        config_data['models']['ensemble'] = {}
                    
                    # Actualizar los pesos
                    config_data['models']['ensemble']['weights'] = normalized_weights
                    
                    # Guardar el archivo
                    with open(config_path, 'w', encoding='utf-8') as f:
                        yaml.dump(config_data, f, default_flow_style=False)
                    
                    # Aplicar los pesos directamente al modelo actual
                    model.weights = normalized_weights
                    logger.info(f"Pesos actualizados manualmente: {normalized_weights}")
                    
                    # Reinicializar el modelo con los nuevos pesos
                    model = EnsembleModel(config)
                    
                    # Mostrar mensaje informativo
                    st.success("Los pesos se han actualizado correctamente y se han guardado en el archivo de configuración.")
                    st.info("""
                    Los pesos se han actualizado en memoria y se aplicarán a las predicciones actuales.
                    Para que los cambios sean permanentes en todas las sesiones, reinicia la aplicación.
                    """)
                    
                    # Mostrar los pesos actualizados
                    st.write("### Pesos actualizados:")
                    st.write(f"- Naive Bayes: {normalized_weights['naive_bayes']:.2f}")
                    st.write(f"- SBERT: {normalized_weights['sbert']:.2f}")
                    st.write(f"- BERT: {normalized_weights['bert']:.2f}")
                except Exception as e:
                    st.error(f"Error al actualizar los pesos: {str(e)}")
                
                st.success("Configuración guardada correctamente")
        else:
            st.error("La suma de los pesos debe ser mayor que cero")
    
    with tab2:
        st.markdown("### Configuración de Preprocesamiento")
        
        # Obtener configuración actual
        preprocessing_config = config.get('preprocessing', {})
        
        # Opciones de preprocesamiento
        language = st.selectbox("Idioma", ["es", "en"], 
                               index=0 if preprocessing_config and preprocessing_config.get('language') == 'es' else 1)
        
        min_token_length = st.slider("Longitud mínima de token", 1, 5, 
                                    preprocessing_config.get('min_token_length', 2) if preprocessing_config else 2)
        
        remove_stopwords = st.checkbox("Eliminar stopwords", 
                                      preprocessing_config.get('remove_stopwords', True) if preprocessing_config else True)
        
        lemmatize = st.checkbox("Lematizar", 
                               preprocessing_config.get('lemmatize', True) if preprocessing_config else True)
        
        remove_punctuation = st.checkbox("Eliminar puntuación", 
                                        preprocessing_config.get('remove_punctuation', True) if preprocessing_config else True)
        
        remove_numbers = st.checkbox("Eliminar números", 
                                    preprocessing_config.get('remove_numbers', False) if preprocessing_config else False)
        
        # Botón para guardar configuración
        if st.button("Guardar Configuración de Preprocesamiento"):
            # Actualizar configuración
            preprocessing_dict = {}
            preprocessing_dict['language'] = language
            preprocessing_dict['min_token_length'] = min_token_length
            preprocessing_dict['remove_stopwords'] = remove_stopwords
            preprocessing_dict['lemmatize'] = lemmatize
            preprocessing_dict['remove_punctuation'] = remove_punctuation
            preprocessing_dict['remove_numbers'] = remove_numbers
            
            # Usar el método set_config si está disponible, o actualizar directamente
            try:
                config.set_config('preprocessing', preprocessing_dict)
            except AttributeError:
                if hasattr(config, 'config_data'):
                    if 'preprocessing' not in config.config_data:
                        config.config_data['preprocessing'] = {}
                    for key, value in preprocessing_dict.items():
                        config.config_data['preprocessing'][key] = value
            
            # Guardar la configuración en el archivo
            try:
                config.save_config()
            except AttributeError:
                # Si el método save_config no existe, intentar guardar de otra manera
                try:
                    config.save()
                except AttributeError:
                    st.warning("No se pudo guardar la configuración automáticamente. Los cambios se aplicarán solo para esta sesión.")
            
            # Reinicializar limpiador de texto
            text_cleaner.__init__(config)
            
            st.success("Configuración guardada correctamente")
    
    with tab3:
        st.markdown("### Configuración del Sistema")
        
        # Obtener configuración actual
        evaluation_config = config.get('evaluation')
        
        # Umbral de clasificación
        threshold = st.slider("Umbral de clasificación", 0.0, 1.0, 
                             0.5 if not evaluation_config else evaluation_config.get('threshold', 0.5), 0.05)
        
        # Botón para guardar configuración
        if st.button("Guardar Configuración del Sistema"):
            # Actualizar configuración
            try:
                config.set_config('evaluation', {'threshold': threshold})
            except AttributeError:
                if hasattr(config, 'config_data'):
                    if 'evaluation' not in config.config_data:
                        config.config_data['evaluation'] = {}
                    config.config_data['evaluation']['threshold'] = threshold
            
            # Guardar la configuración en el archivo
            try:
                config.save_config()
            except AttributeError:
                # Si el método save_config no existe, intentar guardar de otra manera
                try:
                    config.save()
                except AttributeError:
                    st.warning("No se pudo guardar la configuración automáticamente. Los cambios se aplicarán solo para esta sesión.")
            
            st.success("Configuración guardada correctamente")
        
        # Exportar/Importar configuración
        st.markdown("### Exportar/Importar Configuración")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Exportar Configuración"):
                # Convertir configuración a JSON
                import json
                
                config_json = json.dumps(config.config_data, indent=2)
                
                # Crear botón de descarga
                st.download_button(
                    label="Descargar Configuración",
                    data=config_json,
                    file_name="cv_matcher_config.json",
                    mime="application/json"
                )
        
        with col2:
            uploaded_file = st.file_uploader("Importar Configuración", type=['json'])
            
            if uploaded_file is not None:
                try:
                    import json
                    
                    # Cargar configuración
                    config_data = json.load(uploaded_file)
                    
                    # Actualizar configuración
                    for section, values in config_data.items():
                        if isinstance(values, dict):
                            try:
                                config.set_config(section, values)
                            except AttributeError:
                                if hasattr(config, 'config_data'):
                                    if section not in config.config_data:
                                        config.config_data[section] = {}
                                    for key, value in values.items():
                                        config.config_data[section][key] = value
                    
                    # Guardar la configuración en el archivo
                    try:
                        config.save_config()
                    except AttributeError:
                        # Si el método save_config no existe, intentar guardar de otra manera
                        try:
                            config.save()
                        except AttributeError:
                            st.warning("No se pudo guardar la configuración automáticamente. Los cambios se aplicarán solo para esta sesión.")
                    
                    st.success("Configuración importada correctamente")
                except Exception as e:
                    st.error(f"Error al importar configuración: {e}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dashboard para visualización de CVs y descripciones de puestos.
"""

import streamlit as st
import base64
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def display_pdf(file_path):
    """
    Muestra un PDF en Streamlit usando un iframe con base64.
    
    Args:
        file_path: Ruta al archivo PDF
        
    Returns:
        String con el HTML para mostrar el PDF
    """
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
        return pdf_display
    except Exception as e:
        return f"Error al mostrar PDF: {str(e)}"

def show_compatibility_results(cv_data, job_data, score, selected_cv, job_id, temp_dir):
    """
    Muestra los resultados de compatibilidad entre un CV y una descripción de puesto.
    
    Args:
        cv_data: Datos del CV
        job_data: Datos de la descripción del puesto
        score: Puntuación de compatibilidad
        selected_cv: Nombre del archivo CV seleccionado
        job_id: ID del puesto
        temp_dir: Directorio temporal donde se guardan los archivos
    """
    # Directorio de almacenamiento permanente
    pdf_storage_dir = Path("data/pdf_storage")
    pdf_storage_dir.mkdir(parents=True, exist_ok=True)
    # Mostrar resultado
    st.success(f"Compatibilidad: {score * 100:.2f}%")
    
    # Crear gráfico de compatibilidad
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Crear barra de progreso
    ax.barh(0, score, color='green')
    ax.barh(0, 1 - score, left=score, color='lightgray')
    
    # Configurar gráfico
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels([f"{x*100:.0f}%" for x in np.arange(0, 1.1, 0.1)])
    ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.7)
    ax.text(0.7, 0.3, "Umbral de compatibilidad", color='red', alpha=0.7)
    
    # Mostrar gráfico
    st.pyplot(fig)
    
    # Mostrar detalles
    col1, col2, col3 = st.columns([2, 2, 1])
    
    # Obtener texto del CV y del puesto
    cv_text = cv_data.get('processed_content') or cv_data.get('content')
    job_text = job_data.get('processed_content') or job_data.get('content')
    
    with col1:
        st.subheader("Detalles del CV")
        st.write(f"Nombre: {selected_cv}")
        
        # Verificar si el CV es un PDF para mostrarlo
        cv_file_path = temp_dir / selected_cv
        
        # Intentar obtener la ruta del archivo desde la sesión
        session_path = st.session_state.cv_files.get(selected_cv) if hasattr(st.session_state, 'cv_files') else None
        
        if selected_cv.lower().endswith('.pdf'):
            # Mostrar PDF sin procesar directamente
            st.subheader("Vista previa del PDF")
            
            pdf_path = None
            
            # Primero intentar con la ruta guardada en la sesión
            if session_path and Path(session_path).exists():
                pdf_path = session_path
                st.markdown(display_pdf(session_path), unsafe_allow_html=True)
                st.success(f"Mostrando PDF desde: {session_path}")
            # Si no, intentar con la ruta temporal
            elif cv_file_path.exists():
                pdf_path = str(cv_file_path)
                st.markdown(display_pdf(cv_file_path), unsafe_allow_html=True)
                st.success(f"Mostrando PDF desde: {cv_file_path}")
            
            # Mover el botón de descarga a la columna 3
            if pdf_path:
                # Copiar el PDF a la carpeta de almacenamiento permanente
                import shutil
                storage_path = pdf_storage_dir / selected_cv
                try:
                    shutil.copy2(pdf_path, storage_path)
                    logger.info(f"PDF copiado a almacenamiento permanente: {storage_path}")
                except Exception as e:
                    logger.error(f"Error al copiar PDF: {e}")
                
                # Usar la ruta de almacenamiento permanente si existe
                final_path = storage_path if storage_path.exists() else pdf_path
                
                # Mover estos botones a la columna 3
                with col3:
                    with open(final_path, "rb") as file:
                        pdf_bytes = file.read()
                        st.download_button(
                            label="Descargar PDF sin procesar",
                            data=pdf_bytes,
                            file_name=selected_cv,
                            mime="application/pdf",
                            key=f"download_pdf_compatibility_{selected_cv}_{hash(pdf_bytes)}"
                        )
                        
                    # Añadir botón para abrir el PDF en una nueva pestaña
                    pdf_url = f"file://{final_path}"
                    st.markdown(f'<a href="{pdf_url}" target="_blank">Abrir PDF en nueva pestaña</a>', unsafe_allow_html=True)
                
            else:
                # Buscar en directorios comunes
                possible_paths = [
                    Path("temp") / selected_cv,
                    Path("raw_cv_dir") / selected_cv,
                    Path("processed_cv_dir") / selected_cv,
                    Path("data/cvs") / selected_cv,
                    Path.cwd() / "temp" / selected_cv,
                    Path.cwd() / "raw_cv_dir" / selected_cv,
                    Path.cwd() / "processed_cv_dir" / selected_cv,
                    Path.cwd() / "data" / "cvs" / selected_cv
                ]
                
                pdf_found = False
                for path in possible_paths:
                    if path.exists():
                        pdf_path = str(path)
                        st.markdown(display_pdf(pdf_path), unsafe_allow_html=True)
                        st.success(f"Mostrando PDF desde: {path}")
                        
                        # Añadir botón para descargar el PDF
                        with open(pdf_path, "rb") as file:
                            pdf_bytes = file.read()
                            # Copiar el PDF a la carpeta de almacenamiento permanente
                            import shutil
                            storage_path = pdf_storage_dir / selected_cv
                            try:
                                shutil.copy2(path, storage_path)
                                logger.info(f"PDF copiado a almacenamiento permanente: {storage_path}")
                            except Exception as e:
                                logger.error(f"Error al copiar PDF: {e}")
                            
                            # Usar la ruta de almacenamiento permanente si existe
                            final_path = storage_path if storage_path.exists() else path
                            
                            with open(final_path, "rb") as file:
                                pdf_bytes = file.read()
                                st.download_button(
                                    label="Descargar PDF sin procesar",
                                    data=pdf_bytes,
                                    file_name=selected_cv,
                                    mime="application/pdf",
                                    key=f"download_pdf_path_{selected_cv}_{hash(pdf_bytes)}"
                                )
                        
                        # Añadir botón para abrir el PDF en una nueva pestaña
                        pdf_url = f"file://{pdf_path}"
                        st.markdown(f'<a href="{pdf_url}" target="_blank">Abrir PDF en nueva pestaña</a>', unsafe_allow_html=True)
                        st.markdown(f'<a href="{pdf_url}" target="_blank">Abrir PDF en nueva pestaña</a>', unsafe_allow_html=True)
                      
                        
                        pdf_found = True
                        break
                
                if not pdf_found:
                    st.warning("No se puede mostrar el PDF. Archivo no encontrado.")
                    
                    # Intentar buscar en todo el sistema de archivos del proyecto
                    import glob
                    project_dir = Path(__file__).parent.parent
                    pdf_files = list(glob.glob(f"{project_dir}/**/{selected_cv}", recursive=True))
                    
                    if pdf_files:
                        pdf_path = pdf_files[0]
                        st.success(f"¡PDF encontrado en otra ubicación: {pdf_path}!")
                        st.markdown(display_pdf(pdf_path), unsafe_allow_html=True)
                        
                        # Añadir botón para descargar el PDF
                        with open(pdf_path, "rb") as file:
                            pdf_bytes = file.read()
                            # Copiar el PDF a la carpeta de almacenamiento permanente
                            import shutil
                            storage_path = pdf_storage_dir / selected_cv
                            try:
                                shutil.copy2(pdf_path, storage_path)
                                logger.info(f"PDF copiado a almacenamiento permanente: {storage_path}")
                            except Exception as e:
                                logger.error(f"Error al copiar PDF: {e}")
                            
                            # Usar la ruta de almacenamiento permanente si existe
                            final_path = storage_path if storage_path.exists() else pdf_path
                            
                            with open(final_path, "rb") as file:
                                pdf_bytes = file.read()
                                st.download_button(
                                    label="Descargar PDF sin procesar",
                                    data=pdf_bytes,
                                    file_name=selected_cv,
                                    mime="application/pdf",
                                    key=f"download_pdf_glob_{selected_cv}_{hash(pdf_bytes)}"
                                )
                        
                        # Añadir botón para abrir el PDF en una nueva pestaña
                        pdf_url = f"file://{pdf_path}"

                    else:
                        # Mostrar todos los archivos PDF disponibles
                        all_pdfs = list(glob.glob(f"{project_dir}/**/*.pdf", recursive=True))
                        if all_pdfs:
                            st.info(f"Se encontraron {len(all_pdfs)} archivos PDF en el proyecto.")
                            pdf_to_show = st.selectbox("Seleccionar un PDF para mostrar:", all_pdfs)
                            if pdf_to_show:
                                st.markdown(display_pdf(pdf_to_show), unsafe_allow_html=True)
        
        # Siempre mostrar el contenido extraído
        with st.expander("Ver contenido extraído"):
            st.text_area("Texto del CV", cv_text, height=200)
            
            # Añadir botón para descargar el PDF después del contenido
            if pdf_path and selected_cv.lower().endswith('.pdf'):
                with open(final_path, "rb") as file:
                    pdf_bytes = file.read()
                    st.download_button(
                        label="Descargar PDF sin procesar",
                        data=pdf_bytes,
                        file_name=selected_cv,
                        mime="application/pdf",
                        key=f"download_pdf_compatibility_{selected_cv}_{hash(pdf_bytes)}"
                    )
    
    with col2:
        st.subheader("Detalles del Puesto")
        st.write(f"ID: {job_id}")
        st.write(f"Título: {job_data.get('title')}")
        with st.expander("Ver contenido"):
            st.text_area("Descripción del puesto", job_text, height=200)
            
    with col3:
        st.subheader("Acciones")
