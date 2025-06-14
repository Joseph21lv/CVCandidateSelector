#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo de configuración para el sistema de asignación de posiciones.
"""

import yaml
import os
from pathlib import Path

class Config:
    """Clase para manejar la configuración del sistema."""
    
    def __init__(self, config_path=None):
        """
        Inicializa la configuración del sistema.
        
        Args:
            config_path (str, optional): Ruta al archivo de configuración YAML.
        """
        self.config_data = {
            # Rutas de datos
            'data_paths': {
                'raw_cv_dir': 'data/raw/cvs',
                'raw_job_desc_dir': 'data/raw/job_descriptions',
                'processed_cv_dir': 'data/processed/cvs',
                'processed_job_desc_dir': 'data/processed/job_descriptions',
            },
            
            # Parámetros de preprocesamiento
            'preprocessing': {
                'language': 'es',
                'min_token_length': 3,
                'remove_stopwords': True,
                'lemmatize': True,
                'remove_punctuation': True,
                'remove_numbers': False,
            },
            
            # Parámetros de extracción de características
            'feature_extraction': {
                'tfidf': {
                    'max_features': 5000,
                    'ngram_range': (1, 2),
                    'min_df': 2,
                },
                'word2vec': {
                    'vector_size': 300,
                    'window': 5,
                    'min_count': 1,
                },
                'bert': {
                    'model_name': 'dccuchile/bert-base-spanish-wwm-cased',
                    'max_length': 512,
                },
                'sbert': {
                    'model_name': 'hiiamsid/sentence_similarity_spanish_es',
                },
            },
            
            # Parámetros de modelos
            'models': {
                'naive_bayes': {
                    'alpha': 1.0,
                    'fit_prior': True,
                },
                'bert': {
                    'learning_rate': 2e-5,
                    'epochs': 4,
                    'batch_size': 16,
                    'max_length': 512,
                },
                'ensemble': {
                    'weights': {
                        'naive_bayes': 0.2,  # Cambia este valor para ajustar el peso de Naive Bayes
                        'sbert': 0.3,        # Cambia este valor para ajustar el peso de SBERT
                        'bert': 0.5,         # Cambia este valor para ajustar el peso de BERT
                    },
                },
            },
            
            # Parámetros de evaluación
            'evaluation': {
                'test_size': 0.2,
                'random_state': 42,
                'threshold': 0.5,
            },
            
            # Parámetros de base de datos
            'database': {
                'type': 'sqlite',
                'path': 'database/cv_matcher.db',
            },
            
            # Parámetros de API
            'api': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': True,
            },
        }
        
        # Cargar configuración desde archivo si se proporciona
        if config_path:
            self._load_from_file(config_path)
    
    def _load_from_file(self, config_path):
        """
        Carga la configuración desde un archivo YAML.
        
        Args:
            config_path (str): Ruta al archivo de configuración.
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                file_config = yaml.safe_load(file)
                
                # Actualizar configuración con valores del archivo
                for section, values in file_config.items():
                    if section in self.config_data:
                        if isinstance(values, dict) and isinstance(self.config_data[section], dict):
                            self.config_data[section].update(values)
                        else:
                            self.config_data[section] = values
        except Exception as e:
            print(f"Error al cargar la configuración: {e}")
    
    def get(self, section, key=None):
        """
        Obtiene un valor de configuración.
        
        Args:
            section (str): Sección de configuración.
            key (str, optional): Clave específica dentro de la sección.
            
        Returns:
            El valor de configuración solicitado.
        """
        if section not in self.config_data:
            return None
        
        if key is None:
            return self.config_data[section]
        
        if key not in self.config_data[section]:
            return None
            
        return self.config_data[section][key]
    
    def reload(self):
        """Recarga la configuración desde el archivo."""
        try:
            # Cargar configuración predeterminada
            default_config_path = os.path.join(os.path.dirname(__file__), 'default.yaml')
            with open(default_config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Configuración predeterminada recargada desde: {default_config_path}")
            
            # Cargar configuración personalizada si existe
            if hasattr(self, 'config_path') and self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    custom_config = yaml.safe_load(f)
                    if custom_config:
                        # Actualizar configuración con valores personalizados
                        self._update_dict(self.config, custom_config)
                logger.info(f"Configuración personalizada recargada desde: {self.config_path}")
        except Exception as e:
            logger.error(f"Error al recargar la configuración: {e}")
    
    def set(self, section, key, value):
        """
        Establece un valor de configuración.
        
        Args:
            section (str): Sección de configuración.
            key (str): Clave dentro de la sección.
            value: Valor a establecer.
        """
        if section not in self.config_data:
            self.config_data[section] = {}
            
        self.config_data[section][key] = value
    
    def save(self, config_path):
        """
        Guarda la configuración actual en un archivo YAML.
        
        Args:
            config_path (str): Ruta donde guardar el archivo.
        """
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config_data, file, default_flow_style=False)
        except Exception as e:
            print(f"Error al guardar la configuración: {e}")
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para gestionar la configuración del sistema.
"""

import os
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Config:
    """Clase para gestionar la configuración del sistema."""
    
    def __init__(self, config_path=None):
        """
        Inicializa la configuración.
        
        Args:
            config_path (str, optional): Ruta al archivo de configuración. 
                                         Si es None, se usa la configuración predeterminada.
        """
        self.config = {}
        
        # Directorio base del proyecto
        self.base_dir = Path(__file__).parent.parent
        
        # Cargar configuración predeterminada
        default_config_path = self.base_dir / 'config' / 'default.yaml'
        
        try:
            with open(default_config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Configuración predeterminada cargada desde: {default_config_path}")
        except Exception as e:
            logger.error(f"Error al cargar la configuración predeterminada: {e}")
            # Crear configuración mínima para evitar errores
            self.config = {
                'api': {'host': '0.0.0.0', 'port': 5000, 'debug': False},
                'models': {
                    'ensemble': {'weights': {'naive_bayes': 0.3, 'sbert': 0.3, 'bert': 0.3}},
                    'bert': {'model_name': 'dccuchile/bert-base-spanish-wwm-cased', 'max_length': 512},
                    'sbert': {'model_name': 'hiiamsid/sentence_similarity_spanish_es'},
                    'naive_bayes': {'min_df': 2, 'max_features': 10000}
                },
                'preprocessing': {
                    'text_cleaner': {'remove_stopwords': True, 'remove_punctuation': True, 'lemmatize': True},
                    'feature_extractor': {
                        'bert_model': 'dccuchile/bert-base-spanish-wwm-cased',
                        'sbert_model': 'hiiamsid/sentence_similarity_spanish_es'
                    }
                },
                'database': {'path': 'data/cv_matcher.db'}
            }
        
        # Cargar configuración personalizada si se proporciona
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    custom_config = yaml.safe_load(f)
                    # Actualizar configuración con valores personalizados
                    self._update_dict(self.config, custom_config)
                logger.info(f"Configuración personalizada cargada desde: {config_path}")
            except Exception as e:
                logger.error(f"Error al cargar la configuración personalizada: {e}")
    
    def _update_dict(self, d, u):
        """
        Actualiza un diccionario de forma recursiva.
        
        Args:
            d (dict): Diccionario a actualizar
            u (dict): Diccionario con nuevos valores
        
        Returns:
            dict: Diccionario actualizado
        """
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._update_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    def get(self, key, default=None):
        """
        Obtiene un valor de la configuración.
        
        Args:
            key (str): Clave a buscar
            default: Valor predeterminado si la clave no existe
        
        Returns:
            Valor asociado a la clave o el valor predeterminado
        """
        if '.' in key:
            # Acceso a claves anidadas (ej: 'models.bert.model_name')
            parts = key.split('.')
            value = self.config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        else:
            # Acceso a claves de primer nivel
            if isinstance(self.config, dict) and key in self.config:
                return self.config[key]
            return default
    
    def set(self, key, value):
        """
        Establece un valor en la configuración.
        
        Args:
            key (str): Clave a establecer
            value: Valor a asignar
        """
        if '.' in key:
            # Acceso a claves anidadas (ej: 'models.bert.model_name')
            parts = key.split('.')
            config = self.config
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                config = config[part]
            config[parts[-1]] = value
        else:
            # Acceso a claves de primer nivel
            self.config[key] = value
