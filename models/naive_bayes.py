#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementación del clasificador Naive Bayes para la asignación de CVs a posiciones.
"""

import logging
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

class NaiveBayesClassifier:
    """Clasificador Naive Bayes para CVs."""
    
    def __init__(self, config):
        """
        Inicializa el clasificador Naive Bayes.
        
        Args:
            config (Config): Objeto de configuración.
        """
        self.config = config
        self.model = None
        self.vectorizer = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa el modelo Naive Bayes y el vectorizador TF-IDF."""
        try:
            # Obtener configuración del modelo
            model_config = self.config.get('models', 'naive_bayes')
            
            # Configurar vectorizador TF-IDF
            tfidf_config = self.config.get('feature_extraction', 'tfidf')
            self.vectorizer = TfidfVectorizer(
                max_features=tfidf_config.get('max_features', 5000),
                ngram_range=tfidf_config.get('ngram_range', (1, 2)),
                min_df=tfidf_config.get('min_df', 2)
            )
            
            # Configurar modelo Naive Bayes
            self.model = MultinomialNB(
                alpha=model_config.get('alpha', 1.0),
                fit_prior=model_config.get('fit_prior', True)
            )
            
            logger.info("Modelo Naive Bayes inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar el modelo Naive Bayes: {e}")
    
    def train(self, cv_texts, job_texts, labels=None):
        """
        Entrena el modelo Naive Bayes.
        
        Args:
            cv_texts (list): Lista de textos de CVs.
            job_texts (list): Lista de textos de descripciones de puestos.
            labels (list, optional): Etiquetas de entrenamiento (1 para coincidencia, 0 para no coincidencia).
                Si no se proporcionan, se asume que cada CV coincide con su descripción de puesto correspondiente.
        
        Returns:
            bool: True si el entrenamiento fue exitoso, False en caso contrario.
        """
        try:
            # Si no se proporcionan etiquetas, crear etiquetas sintéticas
            if labels is None:
                # Asumimos que cada CV coincide con su descripción de puesto correspondiente
                positive_samples = len(cv_texts)
                
                # Crear muestras negativas aleatorias
                negative_samples = []
                negative_labels = []
                
                for i in range(len(cv_texts)):
                    for j in range(len(job_texts)):
                        if i != j:  # No coincide
                            negative_samples.append((cv_texts[i], job_texts[j]))
                            negative_labels.append(0)
                
                # Limitar el número de muestras negativas para equilibrar el conjunto
                if len(negative_samples) > positive_samples * 3:
                    import random
                    random.seed(42)
                    indices = random.sample(range(len(negative_samples)), positive_samples * 3)
                    negative_samples = [negative_samples[i] for i in indices]
                    negative_labels = [negative_labels[i] for i in indices]
                
                # Combinar muestras positivas y negativas
                X = []
                y = []
                
                # Agregar muestras positivas
                for i in range(len(cv_texts)):
                    X.append(self._combine_texts(cv_texts[i], job_texts[i]))
                    y.append(1)
                
                # Agregar muestras negativas
                for i in range(len(negative_samples)):
                    X.append(self._combine_texts(negative_samples[i][0], negative_samples[i][1]))
                    y.append(negative_labels[i])
            else:
                # Usar etiquetas proporcionadas
                X = []
                y = labels
                
                for i in range(len(cv_texts)):
                    X.append(self._combine_texts(cv_texts[i], job_texts[i]))
            
            # Vectorizar textos
            X_tfidf = self.vectorizer.fit_transform(X)
            
            # Entrenar modelo
            self.model.fit(X_tfidf, y)
            
            logger.info("Modelo Naive Bayes entrenado exitosamente")
            return True
        except Exception as e:
            logger.error(f"Error al entrenar el modelo Naive Bayes: {e}")
            return False
    
    def predict(self, cv_text, job_text):
        """
        Predice la compatibilidad entre un CV y una descripción de puesto.
        
        Args:
            cv_text (str): Texto del CV.
            job_text (str): Texto de la descripción del puesto.
            
        Returns:
            float: Puntuación de compatibilidad (probabilidad de clase positiva).
        """
        if not self.model or not self.vectorizer:
            logger.error("El modelo no ha sido entrenado")
            return 0.0
        
        try:
            # Combinar textos
            combined_text = self._combine_texts(cv_text, job_text)
            
            # Vectorizar texto
            X_tfidf = self.vectorizer.transform([combined_text])
            
            # Predecir probabilidad de clase positiva
            proba = self.model.predict_proba(X_tfidf)[0, 1]
            
            return proba
        except Exception as e:
            logger.error(f"Error al predecir con Naive Bayes: {e}")
            return 0.0
    
    def evaluate(self, cv_texts, job_texts, labels):
        """
        Evalúa el rendimiento del modelo.
        
        Args:
            cv_texts (list): Lista de textos de CVs.
            job_texts (list): Lista de textos de descripciones de puestos.
            labels (list): Etiquetas reales.
            
        Returns:
            dict: Métricas de evaluación.
        """
        try:
            # Preparar datos
            X = []
            for i in range(len(cv_texts)):
                X.append(self._combine_texts(cv_texts[i], job_texts[i]))
            
            # Vectorizar textos
            X_tfidf = self.vectorizer.transform(X)
            
            # Predecir
            y_pred = self.model.predict(X_tfidf)
            y_proba = self.model.predict_proba(X_tfidf)[:, 1]
            
            # Calcular métricas
            metrics = {
                'accuracy': accuracy_score(labels, y_pred),
                'precision': precision_score(labels, y_pred),
                'recall': recall_score(labels, y_pred),
                'f1_score': f1_score(labels, y_pred),
                'probabilities': y_proba.tolist()
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error al evaluar el modelo Naive Bayes: {e}")
            return {}
    
    def _combine_texts(self, cv_text, job_text):
        """
        Combina el texto del CV y la descripción del puesto.
        
        Args:
            cv_text (str): Texto del CV.
            job_text (str): Texto de la descripción del puesto.
            
        Returns:
            str: Texto combinado.
        """
        return f"CV: {cv_text} PUESTO: {job_text}"
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modelo de clasificación Naive Bayes para el sistema de asignación de posiciones.
"""

import logging
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

class NaiveBayesClassifier:
    """Clasificador Naive Bayes para comparar CVs con descripciones de puestos."""
    
    def __init__(self, config):
        """
        Inicializa el clasificador Naive Bayes.
        
        Args:
            config (Config): Objeto de configuración.
        """
        self.config = config
        self.model = None
        self.vectorizer = None
        
        try:
            # Obtener configuración del modelo
            model_config = self.config.get('models.naive_bayes', {})
            
            # Inicializar vectorizador TF-IDF
            if isinstance(model_config, dict):
                self.vectorizer = TfidfVectorizer(
                    max_features=model_config.get('max_features', 10000),
                    min_df=model_config.get('min_df', 2)
                )
            else:
                self.vectorizer = TfidfVectorizer(max_features=10000, min_df=2)
            
            # Inicializar modelo Naive Bayes
            if isinstance(model_config, dict):
                self.model = MultinomialNB(
                    alpha=model_config.get('alpha', 1.0),
                    fit_prior=model_config.get('fit_prior', True)
                )
            else:
                self.model = MultinomialNB(alpha=1.0, fit_prior=True)
            
            logger.info("Modelo Naive Bayes inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar el modelo Naive Bayes: {e}")
    
    def train(self, cv_texts, job_texts):
        """
        Entrena el modelo con textos de CVs y descripciones de puestos.
        
        Args:
            cv_texts (list): Lista de textos de CVs.
            job_texts (list): Lista de textos de descripciones de puestos.
        """
        try:
            # Preparar datos de entrenamiento
            X = cv_texts + job_texts
            y = [0] * len(cv_texts) + [1] * len(job_texts)  # 0 para CVs, 1 para descripciones
            
            # Vectorizar textos
            X_tfidf = self.vectorizer.fit_transform(X)
            
            # Entrenar modelo
            self.model.fit(X_tfidf, y)
            
            logger.info("Modelo Naive Bayes entrenado correctamente")
        except Exception as e:
            logger.error(f"Error al entrenar el modelo Naive Bayes: {e}")
    
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
            if not self.model or not self.vectorizer:
                logger.error("El modelo no ha sido entrenado")
                return 0.5
            
            # Vectorizar textos
            cv_vector = self.vectorizer.transform([cv_text])
            job_vector = self.vectorizer.transform([job_text])
            
            # Calcular similitud coseno
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(cv_vector, job_vector)[0][0]
            
            # Normalizar a [0, 1]
            similarity = (similarity + 1) / 2
            
            return similarity
        except Exception as e:
            logger.error(f"Error al predecir con el modelo Naive Bayes: {e}")
            return 0.5
