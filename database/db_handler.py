#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para manejar la base de datos del sistema.
"""

import os
import logging
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseHandler:
    """Clase para manejar la base de datos del sistema."""
    
    def __init__(self, config):
        """
        Inicializa el manejador de base de datos.
        
        Args:
            config (Config): Objeto de configuración.
        """
        self.config = config
        self.db_type = self.config.get('database.type', 'sqlite')
        self.db_path = Path(self.config.get('database.path', 'data/cv_matcher.db'))
        
        # Crear directorio para la base de datos si no existe
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Inicializar base de datos
        self._initialize_database()
    
    def _initialize_database(self):
        """Inicializa la base de datos y crea las tablas necesarias."""
        if self.db_type == 'sqlite':
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Crear tabla de CVs
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS cvs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    content TEXT,
                    processed_content TEXT,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Crear tabla de descripciones de puestos
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS job_descriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    title TEXT,
                    content TEXT,
                    processed_content TEXT,
                    creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Crear tabla de resultados
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cv_filename TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    score REAL,
                    model_used TEXT,
                    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                conn.commit()
                conn.close()
                
                logger.info("Base de datos inicializada correctamente")
            except Exception as e:
                logger.error(f"Error al inicializar la base de datos: {e}")
    
    def save_cv(self, filename, content, processed_content=None):
        """
        Guarda un CV en la base de datos.
        
        Args:
            filename (str): Nombre del archivo del CV.
            content (str): Contenido del CV.
            processed_content (str, optional): Contenido procesado del CV.
            
        Returns:
            int: ID del CV guardado, o -1 si hubo un error.
        """
        if self.db_type == 'sqlite':
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO cvs (filename, content, processed_content, upload_date)
                VALUES (?, ?, ?, ?)
                ''', (filename, content, processed_content, datetime.now()))
                
                cv_id = cursor.lastrowid
                
                conn.commit()
                conn.close()
                
                logger.info(f"CV guardado con ID: {cv_id}")
                return cv_id
            except Exception as e:
                logger.error(f"Error al guardar CV: {e}")
                return -1
    
    def save_job_description(self, job_id, title, content, processed_content=None):
        """
        Guarda una descripción de puesto en la base de datos.
        
        Args:
            job_id (str): ID del puesto.
            title (str): Título del puesto.
            content (str): Contenido de la descripción.
            processed_content (str, optional): Contenido procesado de la descripción.
            
        Returns:
            int: ID de la descripción guardada, o -1 si hubo un error.
        """
        if self.db_type == 'sqlite':
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO job_descriptions (job_id, title, content, processed_content, creation_date)
                VALUES (?, ?, ?, ?, ?)
                ''', (job_id, title, content, processed_content, datetime.now()))
                
                desc_id = cursor.lastrowid
                
                conn.commit()
                conn.close()
                
                logger.info(f"Descripción de puesto guardada con ID: {desc_id}")
                return desc_id
            except Exception as e:
                logger.error(f"Error al guardar descripción de puesto: {e}")
                return -1
    
    def save_result(self, cv_filename, job_id, score, model_used='ensemble'):
        """
        Guarda un resultado de predicción en la base de datos.
        
        Args:
            cv_filename (str): Nombre del archivo del CV.
            job_id (str): ID del puesto.
            score (float): Puntuación de compatibilidad.
            model_used (str): Modelo utilizado para la predicción.
            
        Returns:
            int: ID del resultado guardado, o -1 si hubo un error.
        """
        if self.db_type == 'sqlite':
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO results (cv_filename, job_id, score, model_used, prediction_date)
                VALUES (?, ?, ?, ?, ?)
                ''', (cv_filename, job_id, score, model_used, datetime.now()))
                
                result_id = cursor.lastrowid
                
                conn.commit()
                conn.close()
                
                logger.info(f"Resultado guardado con ID: {result_id}")
                return result_id
            except Exception as e:
                logger.error(f"Error al guardar resultado: {e}")
                return -1
    
    def get_cv(self, cv_id=None, filename=None):
        """
        Obtiene un CV de la base de datos.
        
        Args:
            cv_id (int, optional): ID del CV.
            filename (str, optional): Nombre del archivo del CV.
            
        Returns:
            dict: Datos del CV, o None si no se encontró.
        """
        if self.db_type == 'sqlite':
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if cv_id is not None:
                    cursor.execute('SELECT * FROM cvs WHERE id = ?', (cv_id,))
                elif filename is not None:
                    cursor.execute('SELECT * FROM cvs WHERE filename = ?', (filename,))
                else:
                    logger.error("Se debe proporcionar cv_id o filename")
                    return None
                
                row = cursor.fetchone()
                
                conn.close()
                
                if row:
                    return dict(row)
                else:
                    return None
            except Exception as e:
                logger.error(f"Error al obtener CV: {e}")
                return None
    
    def get_job_description(self, job_id):
        """
        Obtiene una descripción de puesto de la base de datos.
        
        Args:
            job_id (str): ID del puesto.
            
        Returns:
            dict: Datos de la descripción, o None si no se encontró.
        """
        if self.db_type == 'sqlite':
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM job_descriptions WHERE job_id = ?', (job_id,))
                row = cursor.fetchone()
                
                conn.close()
                
                if row:
                    return dict(row)
                else:
                    return None
            except Exception as e:
                logger.error(f"Error al obtener descripción de puesto: {e}")
                return None
    
    def get_results(self, cv_filename=None, job_id=None):
        """
        Obtiene resultados de predicción de la base de datos.
        
        Args:
            cv_filename (str, optional): Nombre del archivo del CV.
            job_id (str, optional): ID del puesto.
            
        Returns:
            list: Lista de resultados, o lista vacía si no se encontraron.
        """
        if self.db_type == 'sqlite':
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = 'SELECT * FROM results'
                params = []
                
                if cv_filename is not None and job_id is not None:
                    query += ' WHERE cv_filename = ? AND job_id = ?'
                    params = [cv_filename, job_id]
                elif cv_filename is not None:
                    query += ' WHERE cv_filename = ?'
                    params = [cv_filename]
                elif job_id is not None:
                    query += ' WHERE job_id = ?'
                    params = [job_id]
                
                query += ' ORDER BY prediction_date DESC'
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                conn.close()
                
                return [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Error al obtener resultados: {e}")
                return []
    
    def get_all_cvs(self):
        """
        Obtiene todos los CVs de la base de datos.
        
        Returns:
            list: Lista de CVs, o lista vacía si no se encontraron.
        """
        if self.db_type == 'sqlite':
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('SELECT id, filename, upload_date FROM cvs ORDER BY upload_date DESC')
                rows = cursor.fetchall()
                
                conn.close()
                
                return [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Error al obtener todos los CVs: {e}")
                return []
    
    def get_all_job_descriptions(self):
        """
        Obtiene todas las descripciones de puestos de la base de datos.
        
        Returns:
            list: Lista de descripciones, o lista vacía si no se encontraron.
        """
        if self.db_type == 'sqlite':
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('SELECT id, job_id, title, creation_date FROM job_descriptions ORDER BY creation_date DESC')
                rows = cursor.fetchall()
                
                conn.close()
                
                return [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Error al obtener todas las descripciones de puestos: {e}")
                return []
    
    def export_results_to_csv(self, output_path):
        """
        Exporta los resultados a un archivo CSV.
        
        Args:
            output_path (str): Ruta donde guardar el archivo CSV.
            
        Returns:
            bool: True si se exportó correctamente, False en caso contrario.
        """
        if self.db_type == 'sqlite':
            try:
                conn = sqlite3.connect(self.db_path)
                
                # Consulta para obtener resultados con información adicional
                query = '''
                SELECT r.id, r.cv_filename, r.job_id, j.title as job_title, 
                       r.score, r.model_used, r.prediction_date
                FROM results r
                LEFT JOIN job_descriptions j ON r.job_id = j.job_id
                ORDER BY r.prediction_date DESC
                '''
                
                df = pd.read_sql_query(query, conn)
                
                conn.close()
                
                # Guardar a CSV
                df.to_csv(output_path, index=False)
                
                logger.info(f"Resultados exportados a {output_path}")
                return True
            except Exception as e:
                logger.error(f"Error al exportar resultados a CSV: {e}")
                return False
