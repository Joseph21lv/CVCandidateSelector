#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementación del modelo BERT/RoBERTa con fine-tuning para la clasificación de CVs.
"""

import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)

class CVJobDataset(Dataset):
    """Dataset para pares de CV-puesto."""
    
    def __init__(self, cv_texts, job_texts, labels=None, tokenizer=None, max_length=512):
        """
        Inicializa el dataset.
        
        Args:
            cv_texts (list): Lista de textos de CVs.
            job_texts (list): Lista de textos de descripciones de puestos.
            labels (list, optional): Etiquetas (1 para coincidencia, 0 para no coincidencia).
            tokenizer: Tokenizador de BERT.
            max_length (int): Longitud máxima de secuencia.
        """
        self.cv_texts = cv_texts
        self.job_texts = job_texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.cv_texts)
    
    def __getitem__(self, idx):
        cv_text = self.cv_texts[idx]
        job_text = self.job_texts[idx]
        
        # Tokenizar textos
        encoding = self.tokenizer(
            cv_text,
            job_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Eliminar dimensión extra
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # Agregar etiqueta si está disponible
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return item

class BERTModel:
    """Modelo BERT/RoBERTa para clasificación de CVs."""
    
    def __init__(self, config):
        """
        Inicializa el modelo BERT.
        
        Args:
            config (Config): Objeto de configuración.
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa el modelo BERT y su tokenizador."""
        try:
            # Obtener configuración del modelo
            bert_config = self.config.get('feature_extraction', 'bert')
            model_name = bert_config.get('model_name', 'dccuchile/bert-base-spanish-wwm-cased')
            
            # Cargar tokenizador
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Cargar modelo preentrenado para clasificación
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1  # Regresión para puntuación de similitud
            )
            
            # Mover modelo a GPU si está disponible
            self.model.to(self.device)
            
            logger.info(f"Modelo BERT inicializado correctamente: {model_name}")
        except Exception as e:
            logger.error(f"Error al inicializar el modelo BERT: {e}")
    
    def train(self, cv_texts, job_texts, labels=None):
        """
        Entrena el modelo BERT con fine-tuning.
        
        Args:
            cv_texts (list): Lista de textos de CVs.
            job_texts (list): Lista de textos de descripciones de puestos.
            labels (list, optional): Etiquetas de entrenamiento (1 para coincidencia, 0 para no coincidencia).
                Si no se proporcionan, se asume que cada CV coincide con su descripción de puesto correspondiente.
        
        Returns:
            bool: True si el entrenamiento fue exitoso, False en caso contrario.
        """
        if not self.model or not self.tokenizer:
            logger.error("El modelo BERT no ha sido inicializado")
            return False
        
        try:
            # Si no se proporcionan etiquetas, crear etiquetas sintéticas
            if labels is None:
                # Asumimos que cada CV coincide con su descripción de puesto correspondiente
                positive_samples = len(cv_texts)
                positive_cv_texts = cv_texts.copy()
                positive_job_texts = job_texts.copy()
                positive_labels = [1.0] * positive_samples
                
                # Crear muestras negativas aleatorias
                negative_cv_texts = []
                negative_job_texts = []
                negative_labels = []
                
                for i in range(len(cv_texts)):
                    for j in range(len(job_texts)):
                        if i != j:  # No coincide
                            negative_cv_texts.append(cv_texts[i])
                            negative_job_texts.append(job_texts[j])
                            negative_labels.append(0.0)
                
                # Limitar el número de muestras negativas para equilibrar el conjunto
                if len(negative_labels) > positive_samples * 3:
                    import random
                    random.seed(42)
                    indices = random.sample(range(len(negative_labels)), positive_samples * 3)
                    negative_cv_texts = [negative_cv_texts[i] for i in indices]
                    negative_job_texts = [negative_job_texts[i] for i in indices]
                    negative_labels = [negative_labels[i] for i in indices]
                
                # Combinar muestras positivas y negativas
                all_cv_texts = positive_cv_texts + negative_cv_texts
                all_job_texts = positive_job_texts + negative_job_texts
                all_labels = positive_labels + negative_labels
            else:
                # Usar etiquetas proporcionadas
                all_cv_texts = cv_texts
                all_job_texts = job_texts
                all_labels = labels
            
            # Crear dataset
            bert_config = self.config.get('feature_extraction', 'bert')
            max_length = bert_config.get('max_length', 512)
            
            dataset = CVJobDataset(
                all_cv_texts,
                all_job_texts,
                all_labels,
                self.tokenizer,
                max_length
            )
            
            # Configurar parámetros de entrenamiento
            model_config = self.config.get('models', 'bert')
            batch_size = model_config.get('batch_size', 16)
            learning_rate = model_config.get('learning_rate', 2e-5)
            epochs = model_config.get('epochs', 4)
            
            # Crear dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True
            )
            
            # Configurar optimizador
            optimizer = AdamW(
                self.model.parameters(),
                lr=learning_rate,
                eps=1e-8
            )
            
            # Configurar scheduler
            total_steps = len(dataloader) * epochs
            scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
            )
            
            # Entrenar modelo
            self.model.train()
            
            for epoch in range(epochs):
                total_loss = 0
                
                for batch in dataloader:
                    # Mover batch a GPU/CPU
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Limpiar gradientes
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Backward pass
                    loss.backward()
                    
                    # Actualizar parámetros
                    optimizer.step()
                    scheduler.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(dataloader)
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
            # Guardar modelo entrenado
            self.model.eval()
            logger.info("Modelo BERT entrenado exitosamente")
            
            return True
        except Exception as e:
            logger.error(f"Error al entrenar el modelo BERT: {e}")
            return False
    
    def predict(self, cv_text, job_text):
        """
        Predice la compatibilidad entre un CV y una descripción de puesto.
        
        Args:
            cv_text (str): Texto del CV.
            job_text (str): Texto de la descripción del puesto.
            
        Returns:
            float: Puntuación de compatibilidad (0-1).
        """
        if not self.model or not self.tokenizer:
            logger.error("El modelo BERT no ha sido inicializado")
            return 0.0
        
        try:
            # Tokenizar textos
            bert_config = self.config.get('feature_extraction', 'bert')
            max_length = bert_config.get('max_length', 512)
            
            inputs = self.tokenizer(
                cv_text,
                job_text,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Mover inputs a GPU/CPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predecir
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Convertir a puntuación (0-1) usando sigmoide
            score = torch.sigmoid(logits).item()
            
            return score
        except Exception as e:
            logger.error(f"Error al predecir con BERT: {e}")
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
        if not self.model or not self.tokenizer:
            logger.error("El modelo BERT no ha sido inicializado")
            return [0.0] * len(cv_texts)
        
        try:
            # Crear dataset
            bert_config = self.config.get('feature_extraction', 'bert')
            max_length = bert_config.get('max_length', 512)
            
            dataset = CVJobDataset(
                cv_texts,
                job_texts,
                None,
                self.tokenizer,
                max_length
            )
            
            # Crear dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=8,
                shuffle=False
            )
            
            # Predecir
            self.model.eval()
            scores = []
            
            with torch.no_grad():
                for batch in dataloader:
                    # Mover batch a GPU/CPU
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    logits = outputs.logits
                    
                    # Convertir a puntuaciones (0-1) usando sigmoide
                    batch_scores = torch.sigmoid(logits).cpu().numpy().flatten().tolist()
                    scores.extend(batch_scores)
            
            return scores
        except Exception as e:
            logger.error(f"Error al predecir lote con BERT: {e}")
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
            # Predecir puntuaciones
            scores = self.predict_batch(cv_texts, job_texts)
            
            # Convertir puntuaciones a predicciones binarias usando umbral
            threshold = self.config.get('evaluation', 'threshold') or 0.5
            predictions = [1 if score >= threshold else 0 for score in scores]
            
            # Calcular métricas
            metrics = {
                'accuracy': accuracy_score(labels, predictions),
                'precision': precision_score(labels, predictions),
                'recall': recall_score(labels, predictions),
                'f1_score': f1_score(labels, predictions),
                'roc_auc': roc_auc_score(labels, scores),
                'scores': scores
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error al evaluar el modelo BERT: {e}")
            return {}
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modelo BERT para el sistema de asignación de posiciones.
"""

import logging
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class BERTModel:
    """Modelo BERT para comparar CVs con descripciones de puestos."""
    
    def __init__(self, config):
        """
        Inicializa el modelo BERT.
        
        Args:
            config (Config): Objeto de configuración.
        """
        self.config = config
        self.tokenizer = None
        self.model = None
        
        try:
            # Obtener configuración del modelo
            model_config = self.config.get('models.bert', {})
            
            # Inicializar modelo BERT
            if isinstance(model_config, dict):
                model_name = model_config.get('model_name', 'dccuchile/bert-base-spanish-wwm-cased')
                self.max_length = model_config.get('max_length', 512)
            else:
                model_name = 'dccuchile/bert-base-spanish-wwm-cased'
                self.max_length = 512
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Poner el modelo en modo evaluación
            self.model.eval()
            
            logger.info(f"Modelo BERT inicializado correctamente: {model_name}")
        except Exception as e:
            logger.error(f"Error al inicializar el modelo BERT: {e}")
    
    def _get_embedding(self, text):
        """
        Obtiene el embedding de un texto usando BERT.
        
        Args:
            text (str): Texto a codificar.
            
        Returns:
            numpy.ndarray: Vector de embedding.
        """
        # Tokenizar texto
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Obtener embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Usar el embedding del token [CLS] como representación del documento
        return outputs.last_hidden_state[:, 0, :].numpy()
    
    def train(self, cv_texts, job_texts):
        """
        No es necesario entrenar el modelo BERT preentrenado.
        
        Args:
            cv_texts (list): Lista de textos de CVs.
            job_texts (list): Lista de textos de descripciones de puestos.
        """
        logger.info("El modelo BERT preentrenado no requiere entrenamiento adicional")
    
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
            if not self.model or not self.tokenizer:
                logger.error("El modelo BERT no está inicializado")
                return 0.5
            
            # Obtener embeddings
            cv_embedding = self._get_embedding(cv_text)
            job_embedding = self._get_embedding(job_text)
            
            # Calcular similitud coseno
            similarity = cosine_similarity(cv_embedding, job_embedding)[0][0]
            
            # Normalizar a [0, 1]
            similarity = (similarity + 1) / 2
            
            return similarity
        except Exception as e:
            logger.error(f"Error al predecir con el modelo BERT: {e}")
            return 0.5
