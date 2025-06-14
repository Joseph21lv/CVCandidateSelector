#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementación de un modelo de ensamble que combina los resultados de múltiples modelos.
"""

import logging
import numpy as np
from models.naive_bayes import NaiveBayesClassifier
from models.sbert_model import SBERTModel
from models.bert_model import BERTModel

logger = logging.getLogger(__name__)

class EnsembleModel:
    """Modelo de ensamble que combina múltiples modelos."""
    
    def __init__(self, config):
        """
        Inicializa el modelo de ensamble.
        
        Args:
            config (Config): Objeto de configuración.
        """
        self.config = config
        self.models = {}
        self.weights = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Inicializa los modelos individuales y sus pesos."""
        try:
            # Obtener configuración del ensamble
            ensemble_config = self.config.get('models', 'ensemble')
            
            if not ensemble_config:
                logger.warning("No se encontró configuración para el ensamble, usando valores predeterminados")
                self.weights = {
                    'naive_bayes': 0.3,
                    'sbert': 0.3,
                    'bert': 0.3
                }
            else:
                self.weights = ensemble_config.get('weights', {
                    'naive_bayes': 0.3,
                    'sbert': 0.3,
                    'bert': 0.3
                })
            
            # Inicializar modelos individuales
            if 'naive_bayes' in self.weights and self.weights['naive_bayes'] > 0:
                self.models['naive_bayes'] = NaiveBayesClassifier(self.config)
            
            if 'sbert' in self.weights and self.weights['sbert'] > 0:
                self.models['sbert'] = SBERTModel(self.config)
            
            if 'bert' in self.weights and self.weights['bert'] > 0:
                self.models['bert'] = BERTModel(self.config)
            
            # Normalizar pesos
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                for model_name in self.weights:
                    self.weights[model_name] /= total_weight
            
            logger.info(f"Modelo de ensamble inicializado con pesos: {self.weights}")
        except Exception as e:
            logger.error(f"Error al inicializar el modelo de ensamble: {e}")
    
    def train(self, cv_texts, job_texts, labels=None):
        """
        Entrena los modelos individuales del ensamble.
        
        Args:
            cv_texts (list): Lista de textos de CVs.
            job_texts (list): Lista de textos de descripciones de puestos.
            labels (list, optional): Etiquetas de entrenamiento.
        
        Returns:
            bool: True si el entrenamiento fue exitoso, False en caso contrario.
        """
        success = True
        
        for model_name, model in self.models.items():
            logger.info(f"Entrenando modelo {model_name}...")
            model_success = model.train(cv_texts, job_texts, labels)
            
            if not model_success:
                logger.warning(f"Error al entrenar el modelo {model_name}")
                success = False
        
        return success
    
    def predict(self, cv_text, job_text):
        """
        Predice la compatibilidad entre un CV y una descripción de puesto.
        
        Args:
            cv_text (str): Texto del CV.
            job_text (str): Texto de la descripción del puesto.
            
        Returns:
            float: Puntuación de compatibilidad ponderada (0-1).
        """
        if not self.models:
            logger.error("No hay modelos inicializados en el ensamble")
            return 0.0
        
        try:
            # Obtener predicciones de cada modelo
            predictions = {}
            for model_name, model in self.models.items():
                predictions[model_name] = model.predict(cv_text, job_text)
            
            # Calcular predicción ponderada
            weighted_prediction = 0.0
            total_weight = 0.0
            
            for model_name, prediction in predictions.items():
                if model_name in self.weights:
                    weighted_prediction += prediction * self.weights[model_name]
                    total_weight += self.weights[model_name]
            
            # Normalizar si es necesario
            if total_weight > 0:
                weighted_prediction /= total_weight
            
            return weighted_prediction
        except Exception as e:
            logger.error(f"Error al predecir con el ensamble: {e}")
            return 0.0
    
    def predict_batch(self, cv_texts, job_texts):
        """
        Predice la compatibilidad para múltiples pares de CV-puesto.
        
        Args:
            cv_texts (list): Lista de textos de CVs.
            job_texts (list): Lista de textos de descripciones de puestos.
            
        Returns:
            list: Lista de puntuaciones de compatibilidad.
        """
        if not self.models:
            logger.error("No hay modelos inicializados en el ensamble")
            return [0.0] * len(cv_texts)
        
        try:
            # Obtener predicciones de cada modelo
            model_predictions = {}
            
            for model_name, model in self.models.items():
                if hasattr(model, 'predict_batch'):
                    model_predictions[model_name] = model.predict_batch(cv_texts, job_texts)
                else:
                    # Fallback para modelos sin método predict_batch
                    predictions = []
                    for i in range(len(cv_texts)):
                        predictions.append(model.predict(cv_texts[i], job_texts[i]))
                    model_predictions[model_name] = predictions
            
            # Calcular predicciones ponderadas
            weighted_predictions = []
            
            for i in range(len(cv_texts)):
                weighted_prediction = 0.0
                total_weight = 0.0
                
                for model_name, predictions in model_predictions.items():
                    if model_name in self.weights:
                        weighted_prediction += predictions[i] * self.weights[model_name]
                        total_weight += self.weights[model_name]
                
                # Normalizar si es necesario
                if total_weight > 0:
                    weighted_prediction /= total_weight
                
                weighted_predictions.append(weighted_prediction)
            
            return weighted_predictions
        except Exception as e:
            logger.error(f"Error al predecir lote con el ensamble: {e}")
            return [0.0] * len(cv_texts)
    
    def evaluate(self, cv_texts, job_texts, labels):
        """
        Evalúa el rendimiento del modelo de ensamble.
        
        Args:
            cv_texts (list): Lista de textos de CVs.
            job_texts (list): Lista de textos de descripciones de puestos.
            labels (list): Etiquetas reales.
            
        Returns:
            dict: Métricas de evaluación.
        """
        try:
            # Evaluar cada modelo individual
            model_metrics = {}
            
            for model_name, model in self.models.items():
                model_metrics[model_name] = model.evaluate(cv_texts, job_texts, labels)
            
            # Obtener predicciones del ensamble
            ensemble_predictions = self.predict_batch(cv_texts, job_texts)
            
            # Convertir predicciones a binario usando umbral
            threshold = self.config.get('evaluation', 'threshold') or 0.5
            binary_predictions = [1 if pred >= threshold else 0 for pred in ensemble_predictions]
            
            # Calcular métricas del ensamble
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            ensemble_metrics = {
                'accuracy': accuracy_score(labels, binary_predictions),
                'precision': precision_score(labels, binary_predictions),
                'recall': recall_score(labels, binary_predictions),
                'f1_score': f1_score(labels, binary_predictions),
                'roc_auc': roc_auc_score(labels, ensemble_predictions),
                'predictions': ensemble_predictions
            }
            
            # Combinar métricas
            all_metrics = {
                'ensemble': ensemble_metrics
            }
            
            for model_name, metrics in model_metrics.items():
                all_metrics[model_name] = metrics
            
            return all_metrics
        except Exception as e:
            logger.error(f"Error al evaluar el modelo de ensamble: {e}")
            return {}
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modelo de ensamble para el sistema de asignación de posiciones.
"""

import logging
from models.naive_bayes import NaiveBayesClassifier
from models.sbert_model import SBERTModel
from models.bert_model import BERTModel

logger = logging.getLogger(__name__)

class EnsembleModel:
    """Modelo de ensamble que combina varios modelos para mejorar la predicción."""
    
    def __init__(self, config):
        """
        Inicializa el modelo de ensamble.
        
        Args:
            config (Config): Objeto de configuración.
        """
        self.config = config
        self.models = {}
        self.weights = {}
        self._initialize_models()
        
    def _initialize_models(self):
        """Inicializa los modelos individuales y sus pesos."""
        # Cargar pesos directamente del archivo de configuración
        try:
            # Intentar cargar desde config/default.yaml
            import yaml
            import os
            from pathlib import Path
            
            config_path = Path(os.path.dirname(os.path.dirname(__file__))) / 'config' / 'default.yaml'
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                    if config_data and 'models' in config_data and 'ensemble' in config_data['models'] and 'weights' in config_data['models']['ensemble']:
                        self.weights = config_data['models']['ensemble']['weights']
                        logger.info(f"Pesos cargados directamente del archivo de configuración: {self.weights}")
        except Exception as e:
            logger.error(f"Error al cargar pesos desde archivo: {e}")
        
        # Si no se pudieron cargar los pesos o no hay pesos definidos, usar los predeterminados
        if not self.weights:
            # Obtener configuración del modelo
            ensemble_config = self.config.get('models.ensemble', {})
            
            # Obtener pesos de los modelos
            if isinstance(ensemble_config, dict) and 'weights' in ensemble_config:
                self.weights = ensemble_config['weights']
            else:
                self.weights = {
                    'naive_bayes': 0.45,
                    'sbert': 0.4,
                    'bert': 0.15
                }
        
        # Inicializar modelos individuales
        self.models['naive_bayes'] = NaiveBayesClassifier(self.config)
        self.models['sbert'] = SBERTModel(self.config)
        self.models['bert'] = BERTModel(self.config)
        
        # Mostrar los pesos de forma más clara
        logger.info("Modelo de ensamble inicializado con pesos:")
        logger.info(f"  - Naive Bayes: {self.weights.get('naive_bayes', 0):.2f}")
        logger.info(f"  - SBERT: {self.weights.get('sbert', 0):.2f}")
        logger.info(f"  - BERT: {self.weights.get('bert', 0):.2f}")
    
    def train(self, cv_texts, job_texts):
        """
        Entrena todos los modelos del ensamble.
        
        Args:
            cv_texts (list): Lista de textos de CVs.
            job_texts (list): Lista de textos de descripciones de puestos.
        """
        for name, model in self.models.items():
            logger.info(f"Entrenando modelo {name}...")
            model.train(cv_texts, job_texts)
    
    def predict(self, cv_text, job_text):
        """
        Predice la compatibilidad entre un CV y una descripción de puesto
        combinando las predicciones de varios modelos.
        
        Args:
            cv_text (str): Texto del CV.
            job_text (str): Texto de la descripción del puesto.
            
        Returns:
            float: Puntuación de compatibilidad entre 0 y 1.
        """
        # Recargar los pesos antes de predecir para asegurar que se usen los más recientes
        try:
            # Intentar cargar desde config/default.yaml
            import yaml
            import os
            from pathlib import Path
            
            config_path = Path(os.path.dirname(os.path.dirname(__file__))) / 'config' / 'default.yaml'
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                    if config_data and 'models' in config_data and 'ensemble' in config_data['models'] and 'weights' in config_data['models']['ensemble']:
                        self.weights = config_data['models']['ensemble']['weights']
                        logger.info(f"Pesos actualizados para predicción: {self.weights}")
        except Exception as e:
            logger.error(f"Error al recargar configuración: {e}")
        
        predictions = {}
        weighted_sum = 0
        total_weight = 0
        
        # Obtener predicciones de cada modelo
        for name, model in self.models.items():
            try:
                predictions[name] = model.predict(cv_text, job_text)
                weight = self.weights.get(name, 0)
                weighted_sum += predictions[name] * weight
                total_weight += weight
            except Exception as e:
                logger.error(f"Error al predecir con el modelo {name}: {e}")
        
        # Calcular predicción ponderada
        if total_weight > 0:
            final_prediction = weighted_sum / total_weight
        else:
            final_prediction = 0.5  # Valor predeterminado si no hay pesos
        
        logger.info(f"Predicciones individuales: {predictions}")
        logger.info(f"Predicción final: {final_prediction*100:.0f}%")
        
        return final_prediction
