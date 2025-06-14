#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API REST para el sistema de asignación de posiciones basado en CVs.
"""

import os
import logging
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

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

# Crear directorio temporal para archivos subidos
UPLOAD_FOLDER = Path(tempfile.gettempdir()) / 'cv_matcher_uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Inicializar aplicación Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB máximo

# Extensiones permitidas
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt', 'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    """Verifica si la extensión del archivo está permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint para verificar el estado de la API."""
    return jsonify({'status': 'ok'})

@app.route('/api/upload-cv', methods=['POST'])
def upload_cv():
    """Endpoint para subir un CV."""
    if 'file' not in request.files:
        return jsonify({'error': 'No se proporcionó archivo'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Tipo de archivo no permitido'}), 400
    
    try:
        # Guardar archivo temporalmente
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extraer texto
        cv_text = text_extractor.extract(file_path)
        
        # Limpiar texto
        cleaned_text = text_cleaner.clean(cv_text)
        
        # Guardar en base de datos
        cv_id = db_handler.save_cv(filename, cv_text, cleaned_text)
        
        if cv_id == -1:
            return jsonify({'error': 'Error al guardar CV en base de datos'}), 500
        
        return jsonify({
            'success': True,
            'cv_id': cv_id,
            'filename': filename,
            'text_length': len(cv_text)
        })
    except Exception as e:
        logger.error(f"Error al procesar CV: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload-job', methods=['POST'])
def upload_job():
    """Endpoint para subir una descripción de puesto."""
    data = request.json
    
    if not data:
        return jsonify({'error': 'No se proporcionaron datos'}), 400
    
    job_id = data.get('job_id')
    title = data.get('title')
    content = data.get('content')
    
    if not job_id or not title or not content:
        return jsonify({'error': 'Faltan campos requeridos (job_id, title, content)'}), 400
    
    try:
        # Limpiar texto
        cleaned_content = text_cleaner.clean(content)
        
        # Guardar en base de datos
        desc_id = db_handler.save_job_description(job_id, title, content, cleaned_content)
        
        if desc_id == -1:
            return jsonify({'error': 'Error al guardar descripción en base de datos'}), 500
        
        return jsonify({
            'success': True,
            'desc_id': desc_id,
            'job_id': job_id
        })
    except Exception as e:
        logger.error(f"Error al procesar descripción de puesto: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/match', methods=['POST'])
def match_cv_job():
    """Endpoint para comparar un CV con una descripción de puesto."""
    data = request.json
    
    if not data:
        return jsonify({'error': 'No se proporcionaron datos'}), 400
    
    cv_filename = data.get('cv_filename')
    job_id = data.get('job_id')
    
    if not cv_filename or not job_id:
        return jsonify({'error': 'Faltan campos requeridos (cv_filename, job_id)'}), 400
    
    try:
        # Obtener CV y descripción de puesto de la base de datos
        cv_data = db_handler.get_cv(filename=cv_filename)
        job_data = db_handler.get_job_description(job_id)
        
        if not cv_data:
            return jsonify({'error': f'CV no encontrado: {cv_filename}'}), 404
        
        if not job_data:
            return jsonify({'error': f'Descripción de puesto no encontrada: {job_id}'}), 404
        
        # Usar contenido procesado si está disponible
        cv_text = cv_data.get('processed_content') or cv_data.get('content')
        job_text = job_data.get('processed_content') or job_data.get('content')
        
        # Predecir compatibilidad
        score = model.predict(cv_text, job_text)
        
        # Guardar resultado en base de datos
        result_id = db_handler.save_result(cv_filename, job_id, score)
        
        return jsonify({
            'success': True,
            'result_id': result_id,
            'cv_filename': cv_filename,
            'job_id': job_id,
            'job_title': job_data.get('title'),
            'score': score,
            'percentage': f"{score * 100:.2f}%"
        })
    except Exception as e:
        logger.error(f"Error al comparar CV con puesto: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/results', methods=['GET'])
def get_results():
    """Endpoint para obtener resultados de comparaciones."""
    cv_filename = request.args.get('cv_filename')
    job_id = request.args.get('job_id')
    
    try:
        results = db_handler.get_results(cv_filename, job_id)
        
        # Enriquecer resultados con información adicional
        enriched_results = []
        
        for result in results:
            job_data = db_handler.get_job_description(result['job_id'])
            
            enriched_result = {
                **result,
                'job_title': job_data.get('title') if job_data else 'Desconocido',
                'percentage': f"{result['score'] * 100:.2f}%"
            }
            
            enriched_results.append(enriched_result)
        
        return jsonify({
            'success': True,
            'count': len(enriched_results),
            'results': enriched_results
        })
    except Exception as e:
        logger.error(f"Error al obtener resultados: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cvs', methods=['GET'])
def get_cvs():
    """Endpoint para obtener lista de CVs."""
    try:
        cvs = db_handler.get_all_cvs()
        
        return jsonify({
            'success': True,
            'count': len(cvs),
            'cvs': cvs
        })
    except Exception as e:
        logger.error(f"Error al obtener CVs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/jobs', methods=['GET'])
def get_jobs():
    """Endpoint para obtener lista de descripciones de puestos."""
    try:
        jobs = db_handler.get_all_job_descriptions()
        
        return jsonify({
            'success': True,
            'count': len(jobs),
            'jobs': jobs
        })
    except Exception as e:
        logger.error(f"Error al obtener descripciones de puestos: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-results', methods=['GET'])
def export_results():
    """Endpoint para exportar resultados a CSV."""
    try:
        # Crear directorio temporal para el archivo CSV
        export_dir = Path(tempfile.gettempdir()) / 'cv_matcher_exports'
        export_dir.mkdir(exist_ok=True)
        
        # Generar nombre de archivo
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results_export_{timestamp}.csv"
        filepath = export_dir / filename
        
        # Exportar resultados
        success = db_handler.export_results_to_csv(filepath)
        
        if not success:
            return jsonify({'error': 'Error al exportar resultados'}), 500
        
        # Devolver archivo
        return send_from_directory(
            directory=export_dir,
            path=filename,
            as_attachment=True
        )
    except Exception as e:
        logger.error(f"Error al exportar resultados: {e}")
        return jsonify({'error': str(e)}), 500

def start_api():
    """Inicia la API REST."""
    api_config = config.get('api')
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 5000)
    debug = api_config.get('debug', False)
    
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    start_api()
