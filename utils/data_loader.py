#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para cargar datos de CVs y descripciones de puestos.
"""

import os
import logging
import random
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class DataLoader:
    """Clase para cargar datos de CVs y descripciones de puestos."""
    
    def __init__(self, config):
        """
        Inicializa el cargador de datos.
        
        Args:
            config (Config): Objeto de configuración.
        """
        self.config = config
        self.raw_cv_dir = Path(self.config.get('data_paths', 'raw_cv_dir'))
        self.raw_job_desc_dir = Path(self.config.get('data_paths', 'raw_job_desc_dir'))
        self.processed_cv_dir = Path(self.config.get('data_paths', 'processed_cv_dir'))
        self.processed_job_desc_dir = Path(self.config.get('data_paths', 'processed_job_desc_dir'))
        
        # Crear directorios si no existen
        self._create_directories()
    
    def _create_directories(self):
        """Crea los directorios necesarios si no existen."""
        for directory in [self.raw_cv_dir, self.raw_job_desc_dir, 
                         self.processed_cv_dir, self.processed_job_desc_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_text(self, file_path):
        """
        Carga texto de un archivo.
        
        Args:
            file_path (str or Path): Ruta al archivo.
            
        Returns:
            str: Texto cargado del archivo.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"El archivo {file_path} no existe.")
            return ""
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error al cargar texto de {file_path}: {e}")
            return ""
    
    def load_raw_cvs(self):
        """
        Carga los CVs sin procesar.
        
        Returns:
            list: Lista de rutas a archivos de CVs.
        """
        try:
            cv_files = list(self.raw_cv_dir.glob('*'))
            logger.info(f"Cargados {len(cv_files)} CVs sin procesar")
            return cv_files
        except Exception as e:
            logger.error(f"Error al cargar CVs sin procesar: {e}")
            return []
    
    def load_raw_job_descriptions(self):
        """
        Carga las descripciones de puestos sin procesar.
        
        Returns:
            list: Lista de rutas a archivos de descripciones de puestos.
        """
        try:
            job_desc_files = list(self.raw_job_desc_dir.glob('*'))
            logger.info(f"Cargadas {len(job_desc_files)} descripciones de puestos sin procesar")
            return job_desc_files
        except Exception as e:
            logger.error(f"Error al cargar descripciones de puestos sin procesar: {e}")
            return []
    
    def load_processed_cvs(self):
        """
        Carga los CVs procesados.
        
        Returns:
            list: Lista de textos de CVs procesados.
        """
        try:
            cv_files = list(self.processed_cv_dir.glob('*.txt'))
            cv_texts = []
            
            for file_path in cv_files:
                cv_texts.append(self.load_text(file_path))
            
            logger.info(f"Cargados {len(cv_texts)} CVs procesados")
            return cv_texts
        except Exception as e:
            logger.error(f"Error al cargar CVs procesados: {e}")
            return []
    
    def load_processed_job_descriptions(self):
        """
        Carga las descripciones de puestos procesadas.
        
        Returns:
            list: Lista de textos de descripciones de puestos procesados.
        """
        try:
            job_desc_files = list(self.processed_job_desc_dir.glob('*.txt'))
            job_desc_texts = []
            
            for file_path in job_desc_files:
                job_desc_texts.append(self.load_text(file_path))
            
            logger.info(f"Cargadas {len(job_desc_texts)} descripciones de puestos procesadas")
            return job_desc_texts
        except Exception as e:
            logger.error(f"Error al cargar descripciones de puestos procesadas: {e}")
            return []
    
    def load_training_data(self):
        """
        Carga datos de entrenamiento.
        
        Returns:
            tuple: (cv_texts, job_desc_texts)
        """
        cv_texts = self.load_processed_cvs()
        job_desc_texts = self.load_processed_job_descriptions()
        
        return cv_texts, job_desc_texts
    
    def load_evaluation_data(self):
        """
        Carga datos de evaluación con etiquetas.
        
        Returns:
            tuple: (cv_texts, job_desc_texts, labels)
        """
        try:
            # Intentar cargar desde archivo CSV si existe
            eval_file = Path('data/evaluation_data.csv')
            
            if eval_file.exists():
                df = pd.read_csv(eval_file)
                cv_texts = df['cv_text'].tolist()
                job_desc_texts = df['job_desc_text'].tolist()
                labels = df['label'].tolist()
                
                logger.info(f"Cargados {len(cv_texts)} ejemplos de evaluación desde CSV")
                return cv_texts, job_desc_texts, labels
            
            # Si no existe, crear datos de evaluación sintéticos
            cv_texts = self.load_processed_cvs()
            job_desc_texts = self.load_processed_job_descriptions()
            
            if not cv_texts or not job_desc_texts:
                logger.error("No hay datos procesados disponibles para evaluación")
                return [], [], []
            
            # Crear pares positivos (coincidentes)
            positive_pairs = min(len(cv_texts), len(job_desc_texts))
            positive_cv_texts = cv_texts[:positive_pairs]
            positive_job_texts = job_desc_texts[:positive_pairs]
            positive_labels = [1] * positive_pairs
            
            # Crear pares negativos (no coincidentes)
            negative_cv_texts = []
            negative_job_texts = []
            negative_labels = []
            
            for i in range(positive_pairs):
                # Seleccionar una descripción de puesto diferente para cada CV
                j = (i + 1) % positive_pairs  # Asegura que j ≠ i
                negative_cv_texts.append(positive_cv_texts[i])
                negative_job_texts.append(positive_job_texts[j])
                negative_labels.append(0)
            
            # Combinar pares positivos y negativos
            all_cv_texts = positive_cv_texts + negative_cv_texts
            all_job_texts = positive_job_texts + negative_job_texts
            all_labels = positive_labels + negative_labels
            
            # Mezclar datos
            combined = list(zip(all_cv_texts, all_job_texts, all_labels))
            random.shuffle(combined)
            all_cv_texts, all_job_texts, all_labels = zip(*combined)
            
            logger.info(f"Creados {len(all_cv_texts)} ejemplos de evaluación sintéticos")
            return list(all_cv_texts), list(all_job_texts), list(all_labels)
        except Exception as e:
            logger.error(f"Error al cargar datos de evaluación: {e}")
            return [], [], []
    
    def split_data(self, cv_texts, job_texts, labels=None, test_size=0.2, random_state=42):
        """
        Divide los datos en conjuntos de entrenamiento y prueba.
        
        Args:
            cv_texts (list): Textos de CVs.
            job_texts (list): Textos de descripciones de puestos.
            labels (list, optional): Etiquetas.
            test_size (float): Proporción del conjunto de prueba.
            random_state (int): Semilla aleatoria.
            
        Returns:
            tuple: (cv_train, cv_test, job_train, job_test, labels_train, labels_test)
        """
        if labels is None:
            # Si no hay etiquetas, asumimos que cada CV coincide con su descripción de puesto
            cv_train, cv_test, job_train, job_test = train_test_split(
                cv_texts, job_texts, test_size=test_size, random_state=random_state
            )
            return cv_train, cv_test, job_train, job_test, None, None
        else:
            cv_train, cv_test, job_train, job_test, labels_train, labels_test = train_test_split(
                cv_texts, job_texts, labels, test_size=test_size, random_state=random_state
            )
            return cv_train, cv_test, job_train, job_test, labels_train, labels_test
    
    def save_processed_text(self, text, file_path):
        """
        Guarda texto procesado en un archivo.
        
        Args:
            text (str): Texto a guardar.
            file_path (str or Path): Ruta donde guardar el archivo.
            
        Returns:
            bool: True si se guardó correctamente, False en caso contrario.
        """
        file_path = Path(file_path)
        
        try:
            # Crear directorio si no existe
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(text)
            
            return True
        except Exception as e:
            logger.error(f"Error al guardar texto en {file_path}: {e}")
            return False
    
    def load_csv_dataset(self, file_path):
        """
        Carga un dataset desde un archivo CSV.
        
        Args:
            file_path (str or Path): Ruta al archivo CSV.
            
        Returns:
            pandas.DataFrame: DataFrame con los datos cargados.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"El archivo {file_path} no existe.")
            return None
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Cargado dataset con {len(df)} filas desde {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error al cargar dataset desde {file_path}: {e}")
            return None
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para cargar datos de entrenamiento y evaluación.
"""

import os
import logging
from pathlib import Path
import random

logger = logging.getLogger(__name__)

class DataLoader:
    """Clase para cargar datos de entrenamiento y evaluación."""
    
    def __init__(self, config):
        """
        Inicializa el cargador de datos.
        
        Args:
            config (Config): Objeto de configuración.
        """
        self.config = config
        
        # Obtener rutas de datos
        data_paths = self.config.get('data_paths', {})
        
        if isinstance(data_paths, dict):
            self.raw_cv_dir = Path(data_paths.get('raw_cv_dir', 'data/raw/cvs'))
            self.raw_job_desc_dir = Path(data_paths.get('raw_job_desc_dir', 'data/raw/job_descriptions'))
            self.processed_cv_dir = Path(data_paths.get('processed_cv_dir', 'data/processed/cvs'))
            self.processed_job_desc_dir = Path(data_paths.get('processed_job_desc_dir', 'data/processed/job_descriptions'))
        else:
            self.raw_cv_dir = Path('data/raw/cvs')
            self.raw_job_desc_dir = Path('data/raw/job_descriptions')
            self.processed_cv_dir = Path('data/processed/cvs')
            self.processed_job_desc_dir = Path('data/processed/job_descriptions')
        
        # Crear directorios si no existen
        self.raw_cv_dir.mkdir(parents=True, exist_ok=True)
        self.raw_job_desc_dir.mkdir(parents=True, exist_ok=True)
        self.processed_cv_dir.mkdir(parents=True, exist_ok=True)
        self.processed_job_desc_dir.mkdir(parents=True, exist_ok=True)
    
    def load_text(self, file_path):
        """
        Carga texto de un archivo.
        
        Args:
            file_path (Path): Ruta al archivo.
            
        Returns:
            str: Contenido del archivo.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error al cargar texto de {file_path}: {e}")
            return ""
    
    def load_training_data(self):
        """
        Carga datos de entrenamiento.
        
        Returns:
            tuple: (cv_data, job_desc_data)
        """
        try:
            # Cargar CVs
            cv_files = list(self.raw_cv_dir.glob('*.txt'))
            cv_data = [self.load_text(file) for file in cv_files]
            
            # Cargar descripciones de puestos
            job_desc_files = list(self.raw_job_desc_dir.glob('*.txt'))
            job_desc_data = [self.load_text(file) for file in job_desc_files]
            
            logger.info(f"Datos de entrenamiento cargados: {len(cv_data)} CVs, {len(job_desc_data)} descripciones")
            return cv_data, job_desc_data
        except Exception as e:
            logger.error(f"Error al cargar datos de entrenamiento: {e}")
            return [], []
    
    def load_evaluation_data(self):
        """
        Carga datos de evaluación.
        
        Returns:
            tuple: (cv_data, job_desc_data, labels)
        """
        try:
            # Cargar CVs y descripciones de puestos
            cv_data, job_desc_data = self.load_training_data()
            
            # Generar pares de evaluación
            pairs = []
            labels = []
            
            # Pares positivos (compatibles)
            for i in range(min(len(cv_data), len(job_desc_data))):
                pairs.append((cv_data[i], job_desc_data[i]))
                labels.append(1)
            
            # Pares negativos (no compatibles)
            for i in range(min(len(cv_data), len(job_desc_data))):
                j = (i + 1) % len(job_desc_data)
                pairs.append((cv_data[i], job_desc_data[j]))
                labels.append(0)
            
            # Mezclar datos
            combined = list(zip(pairs, labels))
            random.shuffle(combined)
            pairs, labels = zip(*combined)
            
            # Separar CVs y descripciones
            cv_eval_data = [pair[0] for pair in pairs]
            job_desc_eval_data = [pair[1] for pair in pairs]
            
            logger.info(f"Datos de evaluación cargados: {len(pairs)} pares")
            return cv_eval_data, job_desc_eval_data, labels
        except Exception as e:
            logger.error(f"Error al cargar datos de evaluación: {e}")
            return [], [], []
