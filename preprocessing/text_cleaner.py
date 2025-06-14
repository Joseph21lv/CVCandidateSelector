#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para limpiar y normalizar texto extraído de CVs y descripciones de puestos.
"""

import re
import unicodedata
import logging
import spacy
from spacy.lang.es.stop_words import STOP_WORDS as es_stop_words
from spacy.lang.en.stop_words import STOP_WORDS as en_stop_words

logger = logging.getLogger(__name__)

class TextCleaner:
    """Clase para limpiar y normalizar texto."""
    
    def __init__(self, config):
        """
        Inicializa el limpiador de texto.
        
        Args:
            config (Config): Objeto de configuración.
        """
        self.config = config
        self.language = self.config.get('preprocessing', 'language')
        self.min_token_length = self.config.get('preprocessing', 'min_token_length')
        self.remove_stopwords = self.config.get('preprocessing', 'remove_stopwords')
        self.lemmatize = self.config.get('preprocessing', 'lemmatize')
        self.remove_punctuation = self.config.get('preprocessing', 'remove_punctuation')
        self.remove_numbers = self.config.get('preprocessing', 'remove_numbers')
        
        # Cargar modelo de spaCy según el idioma
        try:
            if self.language == 'es':
                self.nlp = spacy.load('es_core_news_md')
                self.stop_words = es_stop_words
            else:  # default to English
                self.nlp = spacy.load('en_core_web_md')
                self.stop_words = en_stop_words
        except OSError:
            logger.warning(f"Modelo de spaCy para {self.language} no encontrado. Descargando...")
            if self.language == 'es':
                spacy.cli.download('es_core_news_md')
                self.nlp = spacy.load('es_core_news_md')
                self.stop_words = es_stop_words
            else:
                spacy.cli.download('en_core_web_md')
                self.nlp = spacy.load('en_core_web_md')
                self.stop_words = en_stop_words
    
    def clean(self, text):
        """
        Limpia y normaliza un texto.
        
        Args:
            text (str): Texto a limpiar.
            
        Returns:
            str: Texto limpio y normalizado.
        """
        if not text:
            return ""
        
        # Normalizar espacios en blanco
        text = self._normalize_whitespace(text)
        
        # Normalizar caracteres Unicode
        text = self._normalize_unicode(text)
        
        # Procesar con spaCy
        doc = self.nlp(text)
        
        # Filtrar tokens según configuración
        tokens = []
        for token in doc:
            # Verificar longitud mínima
            if len(token.text) < self.min_token_length:
                continue
                
            # Verificar si es stopword
            if self.remove_stopwords and token.is_stop:
                continue
                
            # Verificar si es puntuación
            if self.remove_punctuation and token.is_punct:
                continue
                
            # Verificar si es número
            if self.remove_numbers and token.like_num:
                continue
                
            # Lematizar si está configurado
            if self.lemmatize:
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)
        
        # Unir tokens en texto limpio
        clean_text = " ".join(tokens)
        
        return clean_text
    
    def _normalize_whitespace(self, text):
        """
        Normaliza espacios en blanco en el texto.
        
        Args:
            text (str): Texto a normalizar.
            
        Returns:
            str: Texto con espacios normalizados.
        """
        # Reemplazar múltiples espacios en blanco con uno solo
        text = re.sub(r'\s+', ' ', text)
        
        # Eliminar espacios al inicio y final
        text = text.strip()
        
        return text
    
    def _normalize_unicode(self, text):
        """
        Normaliza caracteres Unicode en el texto.
        
        Args:
            text (str): Texto a normalizar.
            
        Returns:
            str: Texto con caracteres Unicode normalizados.
        """
        # Normalizar a forma NFKD
        text = unicodedata.normalize('NFKD', text)
        
        # Eliminar caracteres de control
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        
        return text
    
    def extract_entities(self, text):
        """
        Extrae entidades nombradas del texto.
        
        Args:
            text (str): Texto del que extraer entidades.
            
        Returns:
            dict: Diccionario con entidades agrupadas por tipo.
        """
        if not text:
            return {}
        
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            
            entities[ent.label_].append(ent.text)
        
        return entities
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para limpiar y preprocesar texto.
"""

import re
import logging
import string
import spacy
from spacy.lang.es.stop_words import STOP_WORDS

logger = logging.getLogger(__name__)

class TextCleaner:
    """Clase para limpiar y preprocesar texto."""
    
    def __init__(self, config):
        """
        Inicializa el limpiador de texto.
        
        Args:
            config (Config): Objeto de configuración.
        """
        self.config = config
        self.nlp = None
        
        # Cargar configuración de preprocesamiento
        preprocessing_config = self.config.get('preprocessing', {})
        
        if isinstance(preprocessing_config, dict):
            self.remove_stopwords = preprocessing_config.get('remove_stopwords', True)
            self.remove_punctuation = preprocessing_config.get('remove_punctuation', True)
            self.lemmatize = preprocessing_config.get('lemmatize', True)
            self.remove_numbers = preprocessing_config.get('remove_numbers', False)
            self.language = preprocessing_config.get('language', 'es')
        else:
            self.remove_stopwords = True
            self.remove_punctuation = True
            self.lemmatize = True
            self.remove_numbers = False
            self.language = 'es'
        
        # Inicializar spaCy si se va a usar lematización
        if self.lemmatize:
            try:
                self.nlp = spacy.load(f"{self.language}_core_news_sm")
                logger.info(f"Modelo spaCy cargado: {self.language}_core_news_sm")
            except Exception as e:
                logger.warning(f"No se pudo cargar el modelo spaCy: {e}")
                logger.warning("Intente instalar el modelo con: python -m spacy download es_core_news_sm")
                self.nlp = None
                self.lemmatize = False
    
    def clean(self, text):
        """
        Limpia y preprocesa un texto.
        
        Args:
            text (str): Texto a limpiar.
            
        Returns:
            str: Texto limpio.
        """
        if not text:
            return ""
        
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar caracteres especiales y números
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        if self.remove_numbers:
            text = re.sub(r'\d+', ' ', text)
        
        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Lematizar y eliminar stopwords si se configuró
        if self.lemmatize and self.nlp:
            return self._lemmatize_and_remove_stopwords(text)
        elif self.remove_stopwords:
            return self._remove_stopwords(text)
        else:
            return text
    
    def _lemmatize_and_remove_stopwords(self, text):
        """
        Lematiza el texto y elimina stopwords usando spaCy.
        
        Args:
            text (str): Texto a procesar.
            
        Returns:
            str: Texto procesado.
        """
        doc = self.nlp(text)
        tokens = []
        
        for token in doc:
            if self.remove_stopwords and token.is_stop:
                continue
            if token.lemma_ == '-PRON-':
                tokens.append(token.text)
            else:
                tokens.append(token.lemma_)
        
        return ' '.join(tokens)
    
    def _remove_stopwords(self, text):
        """
        Elimina stopwords del texto sin lematizar.
        
        Args:
            text (str): Texto a procesar.
            
        Returns:
            str: Texto sin stopwords.
        """
        words = text.split()
        filtered_words = [word for word in words if word not in STOP_WORDS]
        return ' '.join(filtered_words)
    
    def extract_entities(self, text):
        """
        Extrae entidades nombradas del texto usando spaCy.
        
        Args:
            text (str): Texto del que extraer entidades.
            
        Returns:
            dict: Diccionario con entidades agrupadas por tipo.
        """
        if not self.nlp:
            logger.warning("No se puede extraer entidades sin un modelo spaCy")
            return {}
        
        entities = {}
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            
            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)
        
        return entities
