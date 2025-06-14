#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para evaluar el rendimiento de los modelos de clasificación.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)

logger = logging.getLogger(__name__)

class Evaluator:
    """Clase para evaluar modelos de clasificación."""
    
    def __init__(self, config):
        """
        Inicializa el evaluador.
        
        Args:
            config (Config): Objeto de configuración.
        """
        self.config = config
        self.threshold = self.config.get('evaluation', 'threshold') or 0.5
    
    def evaluate(self, model, cv_features, job_features, labels):
        """
        Evalúa el rendimiento de un modelo.
        
        Args:
            model: Modelo a evaluar.
            cv_features (list): Características de los CVs.
            job_features (list): Características de las descripciones de puestos.
            labels (list): Etiquetas reales.
            
        Returns:
            dict: Métricas de evaluación.
        """
        try:
            # Verificar si el modelo tiene método evaluate
            if hasattr(model, 'evaluate'):
                return model.evaluate(cv_features, job_features, labels)
            
            # Obtener predicciones
            if hasattr(model, 'predict_batch'):
                scores = model.predict_batch(cv_features, job_features)
            else:
                scores = []
                for i in range(len(cv_features)):
                    scores.append(model.predict(cv_features[i], job_features[i]))
            
            # Convertir scores a predicciones binarias
            predictions = [1 if score >= self.threshold else 0 for score in scores]
            
            # Calcular métricas
            metrics = self._calculate_metrics(labels, predictions, scores)
            
            return metrics
        except Exception as e:
            logger.error(f"Error al evaluar el modelo: {e}")
            return {}
    
    def _calculate_metrics(self, labels, predictions, scores):
        """
        Calcula métricas de evaluación.
        
        Args:
            labels (list): Etiquetas reales.
            predictions (list): Predicciones binarias.
            scores (list): Puntuaciones de confianza.
            
        Returns:
            dict: Métricas calculadas.
        """
        metrics = {}
        
        # Métricas básicas
        metrics['accuracy'] = accuracy_score(labels, predictions)
        metrics['precision'] = precision_score(labels, predictions)
        metrics['recall'] = recall_score(labels, predictions)
        metrics['f1_score'] = f1_score(labels, predictions)
        
        # Matriz de confusión
        cm = confusion_matrix(labels, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        # AUC-ROC
        if len(set(labels)) > 1:  # Solo si hay más de una clase
            metrics['roc_auc'] = roc_auc_score(labels, scores)
        
        # Informe de clasificación
        report = classification_report(labels, predictions, output_dict=True)
        metrics['classification_report'] = report
        
        return metrics
    
    def plot_confusion_matrix(self, labels, predictions, output_path=None):
        """
        Genera una visualización de la matriz de confusión.
        
        Args:
            labels (list): Etiquetas reales.
            predictions (list): Predicciones binarias.
            output_path (str, optional): Ruta donde guardar la visualización.
            
        Returns:
            matplotlib.figure.Figure: Figura generada.
        """
        try:
            import seaborn as sns
            
            cm = confusion_matrix(labels, predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['No Coincide', 'Coincide'],
                        yticklabels=['No Coincide', 'Coincide'])
            plt.xlabel('Predicción')
            plt.ylabel('Real')
            plt.title('Matriz de Confusión')
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight')
            
            return plt.gcf()
        except Exception as e:
            logger.error(f"Error al generar matriz de confusión: {e}")
            return None
    
    def plot_roc_curve(self, labels, scores, output_path=None):
        """
        Genera una visualización de la curva ROC.
        
        Args:
            labels (list): Etiquetas reales.
            scores (list): Puntuaciones de confianza.
            output_path (str, optional): Ruta donde guardar la visualización.
            
        Returns:
            matplotlib.figure.Figure: Figura generada.
        """
        try:
            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = roc_auc_score(labels, scores)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'AUC = {roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title('Curva ROC')
            plt.legend(loc='lower right')
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight')
            
            return plt.gcf()
        except Exception as e:
            logger.error(f"Error al generar curva ROC: {e}")
            return None
    
    def plot_precision_recall_curve(self, labels, scores, output_path=None):
        """
        Genera una visualización de la curva Precision-Recall.
        
        Args:
            labels (list): Etiquetas reales.
            scores (list): Puntuaciones de confianza.
            output_path (str, optional): Ruta donde guardar la visualización.
            
        Returns:
            matplotlib.figure.Figure: Figura generada.
        """
        try:
            precision, recall, _ = precision_recall_curve(labels, scores)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Curva Precision-Recall')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight')
            
            return plt.gcf()
        except Exception as e:
            logger.error(f"Error al generar curva Precision-Recall: {e}")
            return None
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para evaluar el rendimiento de los modelos.
"""

import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)

class Evaluator:
    """Clase para evaluar el rendimiento de los modelos."""
    
    def __init__(self, config):
        """
        Inicializa el evaluador.
        
        Args:
            config (Config): Objeto de configuración.
        """
        self.config = config
        self.threshold = self.config.get('evaluation.threshold', 0.5)
    
    def evaluate(self, model, cv_texts, job_texts, labels):
        """
        Evalúa el rendimiento del modelo.
        
        Args:
            model: Modelo a evaluar.
            cv_texts (list): Lista de textos de CVs.
            job_texts (list): Lista de textos de descripciones de puestos.
            labels (list): Etiquetas reales (1 para compatibles, 0 para no compatibles).
            
        Returns:
            dict: Métricas de evaluación.
        """
        try:
            # Obtener predicciones
            predictions = []
            for cv, job in zip(cv_texts, job_texts):
                score = model.predict(cv, job)
                predictions.append(score)
            
            # Convertir predicciones a etiquetas binarias
            binary_predictions = [1 if p >= self.threshold else 0 for p in predictions]
            
            # Calcular métricas
            metrics = {
                'accuracy': accuracy_score(labels, binary_predictions),
                'precision': precision_score(labels, binary_predictions, zero_division=0),
                'recall': recall_score(labels, binary_predictions, zero_division=0),
                'f1': f1_score(labels, binary_predictions, zero_division=0),
                'auc': roc_auc_score(labels, predictions) if len(set(labels)) > 1 else 0.5
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error al evaluar el modelo: {e}")
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'auc': 0.5
            }
