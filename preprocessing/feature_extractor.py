#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para extraer características de texto para su uso en modelos de clasificación.
"""

import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import torch
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Clase para extraer características de texto."""
    
    def __init__(self, config):
        """
        Inicializa el extractor de características.
        
        Args:
            config (Config): Objeto de configuración.
        """
        self.config = config
        self.tfidf_vectorizer = None
        self.word2vec_model = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.sbert_model = None
        
        # Inicializar vectorizador TF-IDF
        self._init_tfidf()
        
        # Inicializar modelo BERT si está configurado
        if self.config.get('feature_extraction', 'bert'):
            self._init_bert()
        
        # Inicializar modelo SBERT si está configurado
        if self.config.get('feature_extraction', 'sbert'):
            self._init_sbert()
    
    def _init_tfidf(self):
        """Inicializa el vectorizador TF-IDF."""
        tfidf_config = self.config.get('feature_extraction.tfidf')
        
        if tfidf_config:
            if isinstance(tfidf_config, dict):
                max_features = tfidf_config.get('max_features', 5000)
                ngram_range = tfidf_config.get('ngram_range', (1, 2))
                min_df = tfidf_config.get('min_df', 2)
            else:
                # Valores predeterminados si tfidf_config no es un diccionario
                max_features = 5000
                ngram_range = (1, 2)
                min_df = 2
                
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df
            )
    
    def _init_bert(self):
        """Inicializa el modelo BERT y su tokenizador."""
        try:
            bert_config = self.config.get('feature_extraction.bert')
            if isinstance(bert_config, dict):
                model_name = bert_config.get('model_name', 'dccuchile/bert-base-spanish-wwm-cased')
            else:
                model_name = 'dccuchile/bert-base-spanish-wwm-cased'
            
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            
            # Poner el modelo en modo evaluación
            self.bert_model.eval()
            
            logger.info(f"Modelo BERT inicializado: {model_name}")
        except Exception as e:
            logger.error(f"Error al inicializar BERT: {e}")
            self.bert_tokenizer = None
            self.bert_model = None
    
    def _init_sbert(self):
        """Inicializa el modelo Sentence-BERT."""
        try:
            from sentence_transformers import SentenceTransformer
            
            sbert_config = self.config.get('feature_extraction.sbert')
            if isinstance(sbert_config, dict):
                model_name = sbert_config.get('model_name', 'hiiamsid/sentence_similarity_spanish_es')
            else:
                model_name = 'hiiamsid/sentence_similarity_spanish_es'
            
            self.sbert_model = SentenceTransformer(model_name)
            
            logger.info(f"Modelo SBERT inicializado: {model_name}")
        except Exception as e:
            logger.error(f"Error al inicializar SBERT: {e}")
            self.sbert_model = None
    
    def train_word2vec(self, texts):
        """
        Entrena un modelo Word2Vec con los textos proporcionados.
        
        Args:
            texts (list): Lista de textos tokenizados.
        """
        try:
            word2vec_config = self.config.get('feature_extraction.word2vec')
            
            if word2vec_config and texts:
                # Tokenizar textos si no están tokenizados
                tokenized_texts = []
                for text in texts:
                    if isinstance(text, str):
                        tokenized_texts.append(text.split())
                    else:
                        tokenized_texts.append(text)
                
                # Configurar parámetros
                if isinstance(word2vec_config, dict):
                    vector_size = word2vec_config.get('vector_size', 300)
                    window = word2vec_config.get('window', 5)
                    min_count = word2vec_config.get('min_count', 1)
                else:
                    vector_size = 300
                    window = 5
                    min_count = 1
                
                # Entrenar modelo Word2Vec
                self.word2vec_model = Word2Vec(
                    sentences=tokenized_texts,
                    vector_size=vector_size,
                    window=window,
                    min_count=min_count,
                    workers=4
                )
                
                logger.info("Modelo Word2Vec entrenado exitosamente")
        except Exception as e:
            logger.error(f"Error al entrenar Word2Vec: {e}")
    
    def fit_tfidf(self, texts):
        """
        Ajusta el vectorizador TF-IDF con los textos proporcionados.
        
        Args:
            texts (list): Lista de textos.
        """
        if self.tfidf_vectorizer and texts:
            self.tfidf_vectorizer.fit(texts)
            logger.info("Vectorizador TF-IDF ajustado exitosamente")
    
    def extract_features(self, text, method='all'):
        """
        Extrae características de un texto.
        
        Args:
            text (str): Texto del que extraer características.
            method (str): Método de extracción ('tfidf', 'word2vec', 'bert', 'sbert', 'all').
            
        Returns:
            dict: Diccionario con las características extraídas.
        """
        features = {}
        
        if not text:
            return features
        
        # Extraer características TF-IDF
        if method in ['tfidf', 'all'] and self.tfidf_vectorizer:
            try:
                tfidf_features = self.tfidf_vectorizer.transform([text])
                features['tfidf'] = tfidf_features
            except Exception as e:
                logger.error(f"Error al extraer características TF-IDF: {e}")
        
        # Extraer características Word2Vec
        if method in ['word2vec', 'all'] and self.word2vec_model:
            try:
                tokens = text.split()
                word_vectors = []
                
                for token in tokens:
                    if token in self.word2vec_model.wv:
                        word_vectors.append(self.word2vec_model.wv[token])
                
                if word_vectors:
                    # Calcular vector promedio
                    avg_vector = np.mean(word_vectors, axis=0)
                    features['word2vec'] = avg_vector
            except Exception as e:
                logger.error(f"Error al extraer características Word2Vec: {e}")
        
        # Extraer características BERT
        if method in ['bert', 'all'] and self.bert_tokenizer and self.bert_model:
            try:
                # Tokenizar texto
                bert_config = self.config.get('feature_extraction.bert')
                if isinstance(bert_config, dict):
                    max_length = bert_config.get('max_length', 512)
                else:
                    max_length = 512
                
                inputs = self.bert_tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                )
                
                # Obtener embeddings
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                
                # Usar el embedding del token [CLS] como representación del documento
                cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
                features['bert'] = cls_embedding
            except Exception as e:
                logger.error(f"Error al extraer características BERT: {e}")
        
        # Extraer características SBERT
        if method in ['sbert', 'all'] and self.sbert_model:
            try:
                # Obtener embedding de la oración
                sentence_embedding = self.sbert_model.encode(text)
                features['sbert'] = sentence_embedding
            except Exception as e:
                logger.error(f"Error al extraer características SBERT: {e}")
        
        return features
    
    def extract_named_entities(self, text):
        """
        Extrae entidades nombradas del texto usando spaCy.
        
        Args:
            text (str): Texto del que extraer entidades.
            
        Returns:
            dict: Diccionario con entidades agrupadas por tipo.
        """
        from preprocessing.text_cleaner import TextCleaner
        
        text_cleaner = TextCleaner(self.config)
        return text_cleaner.extract_entities(text)
