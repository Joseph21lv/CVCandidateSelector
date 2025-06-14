#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementación del modelo SBERT (Sentence-BERT) para la comparación semántica de CVs y descripciones de puestos.
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

logger = logging.getLogger(__name__)

class SBERTModel:
    """Modelo SBERT para comparación semántica de textos."""
    
    def __init__(self, config):
        """
        Inicializa el modelo SBERT.
        
        Args:
            config (Config): Objeto de configuración.
        """
        self.config = config
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa el modelo SBERT."""
        try:
            # Obtener configuración del modelo
            sbert_config = self.config.get('feature_extraction', 'sbert')
            model_name = sbert_config.get('model_name', 'hiiamsid/sentence_similarity_spanish_es')
            
            # Cargar modelo preentrenado
            self.model = SentenceTransformer(model_name)
            
            logger.info(f"Modelo SBERT inicializado correctamente: {model_name}")
        except Exception as e:
            logger.error(f"Error al inicializar el modelo SBERT: {e}")
    
    def train(self, cv_texts, job_texts, labels=None):
        """
        Entrena el modelo SBERT (en este caso, no se entrena ya que usamos un modelo preentrenado).
        
        Args:
            cv_texts (list): Lista de textos de CVs.
            job_texts (list): Lista de textos de descripciones de puestos.
            labels (list, optional): Etiquetas de entrenamiento (no utilizadas).
        
        Returns:
            bool: True si la inicialización fue exitosa, False en caso contrario.
        """
        # SBERT es un modelo preentrenado, no necesitamos entrenarlo
        # Solo verificamos que esté inicializado correctamente
        return self.model is not None
    
    def predict(self, cv_text, job_text):
        """
        Predice la similitud semántica entre un CV y una descripción de puesto.
        
        Args:
            cv_text (str): Texto del CV.
            job_text (str): Texto de la descripción del puesto.
            
        Returns:
            float: Puntuación de similitud (0-1).
        """
        if not self.model:
            logger.error("El modelo SBERT no ha sido inicializado")
            return 0.0
        
        try:
            # Codificar textos
            cv_embedding = self.model.encode(cv_text, convert_to_tensor=True)
            job_embedding = self.model.encode(job_text, convert_to_tensor=True)
            
            # Calcular similitud de coseno
            cosine_score = util.pytorch_cos_sim(cv_embedding, job_embedding).item()
            
            # Normalizar a rango 0-1
            similarity_score = (cosine_score + 1) / 2
            
            return similarity_score
        except Exception as e:
            logger.error(f"Error al predecir con SBERT: {e}")
            return 0.0
    
    def predict_batch(self, cv_texts, job_texts):
        """
        Predice la similitud semántica para múltiples pares de CV-puesto.
        
        Args:
            cv_texts (list): Lista de textos de CVs.
            job_texts (list): Lista de textos de descripciones de puestos.
            
        Returns:
            list: Lista de puntuaciones de similitud.
        """
        if not self.model:
            logger.error("El modelo SBERT no ha sido inicializado")
            return [0.0] * len(cv_texts)
        
        try:
            # Codificar todos los textos
            cv_embeddings = self.model.encode(cv_texts, convert_to_tensor=True)
            job_embeddings = self.model.encode(job_texts, convert_to_tensor=True)
            
            # Calcular similitud para cada par
            scores = []
            for i in range(len(cv_texts)):
                cosine_score = util.pytorch_cos_sim(cv_embeddings[i], job_embeddings[i]).item()
                similarity_score = (cosine_score + 1) / 2
                scores.append(similarity_score)
            
            return scores
        except Exception as e:
            logger.error(f"Error al predecir lote con SBERT: {e}")
            return [0.0] * len(cv_texts)
    
    def evaluate(self, cv_texts, job_texts, labels):
        """
        Evalúa el rendimiento del modelo.
        
        Args:
            cv_texts (list): Lista de textos de CVs.
            job_texts (list): Lista de textos de descripciones de puestos.
            labels (list): Etiquetas reales (1 para coincidencia, 0 para no coincidencia).
            
        Returns:
            dict: Métricas de evaluación.
        """
        try:
            # Predecir similitudes
            similarity_scores = self.predict_batch(cv_texts, job_texts)
            
            # Convertir similitudes a predicciones binarias usando umbral
            threshold = self.config.get('evaluation', 'threshold') or 0.5
            predictions = [1 if score >= threshold else 0 for score in similarity_scores]
            
            # Calcular métricas
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                'accuracy': accuracy_score(labels, predictions),
                'precision': precision_score(labels, predictions),
                'recall': recall_score(labels, predictions),
                'f1_score': f1_score(labels, predictions),
                'roc_auc': roc_auc_score(labels, similarity_scores),
                'similarity_scores': similarity_scores
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error al evaluar el modelo SBERT: {e}")
            return {}
    
    def get_top_matches(self, cv_text, job_texts, top_n=5):
        """
        Obtiene las mejores coincidencias para un CV entre varias descripciones de puestos.
        
        Args:
            cv_text (str): Texto del CV.
            job_texts (list): Lista de textos de descripciones de puestos.
            top_n (int): Número de mejores coincidencias a devolver.
            
        Returns:
            list: Lista de tuplas (índice, puntuación) de las mejores coincidencias.
        """
        if not self.model:
            logger.error("El modelo SBERT no ha sido inicializado")
            return []
        
        try:
            # Codificar CV
            cv_embedding = self.model.encode(cv_text, convert_to_tensor=True)
            
            # Codificar descripciones de puestos
            job_embeddings = self.model.encode(job_texts, convert_to_tensor=True)
            
            # Calcular similitudes
            similarities = util.pytorch_cos_sim(cv_embedding, job_embeddings)[0]
            
            # Normalizar a rango 0-1
            similarities = (similarities + 1) / 2
            
            # Obtener índices de las mejores coincidencias
            top_indices = torch.topk(similarities, min(top_n, len(job_texts))).indices.tolist()
            
            # Crear lista de tuplas (índice, puntuación)
            top_matches = [(idx, similarities[idx].item()) for idx in top_indices]
            
            return top_matches
        except Exception as e:
            logger.error(f"Error al obtener mejores coincidencias con SBERT: {e}")
            return []
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modelo SBERT para el sistema de asignación de posiciones.
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

class SBERTModel:
    """Modelo SBERT para comparar CVs con descripciones de puestos."""
    
    def __init__(self, config):
        """
        Inicializa el modelo SBERT.
        
        Args:
            config (Config): Objeto de configuración.
        """
        self.config = config
        self.model = None
        
        try:
            # Obtener configuración del modelo
            model_config = self.config.get('models.sbert', {})
            
            # Inicializar modelo SBERT
            if isinstance(model_config, dict):
                model_name = model_config.get('model_name', 'hiiamsid/sentence_similarity_spanish_es')
            else:
                model_name = 'hiiamsid/sentence_similarity_spanish_es'
                
            self.model = SentenceTransformer(model_name)
            
            logger.info(f"Modelo SBERT inicializado correctamente: {model_name}")
        except Exception as e:
            logger.error(f"Error al inicializar el modelo SBERT: {e}")
    
    def train(self, cv_texts, job_texts):
        """
        No es necesario entrenar el modelo SBERT preentrenado.
        
        Args:
            cv_texts (list): Lista de textos de CVs.
            job_texts (list): Lista de textos de descripciones de puestos.
        """
        logger.info("El modelo SBERT preentrenado no requiere entrenamiento adicional")
    
    def predict(self, cv_text, job_text):
        """
        Predice la compatibilidad entre un CV y una descripción de puesto.
        
        Args:
            cv_text (str): Texto del CV.
            job_text (str): Texto de la descripción del puesto.
            
        Returns:
            float: Puntuación de compatibilidad entre 0 y 1.
        """
        try:
            if not self.model:
                logger.error("El modelo SBERT no está inicializado")
                return 0.5
            
            # Codificar textos
            cv_embedding = self.model.encode(cv_text)
            job_embedding = self.model.encode(job_text)
            
            # Calcular similitud coseno
            similarity = util.cos_sim(cv_embedding, job_embedding).item()
            
            # Normalizar a [0, 1]
            similarity = (similarity + 1) / 2
            
            return similarity
        except Exception as e:
            logger.error(f"Error al predecir con el modelo SBERT: {e}")
            return 0.5
