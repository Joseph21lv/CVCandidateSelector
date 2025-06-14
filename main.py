#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sistema de Asignación de Posiciones basado en CVs
-------------------------------------------------
Este sistema analiza CVs en diferentes formatos y los compara con descripciones
de puestos para determinar la idoneidad de los candidatos.

Autores:
- Rolando Adrián Palacios Guzmán
- Sebastián Astiazarán López
- José Pablo López Valdez
"""

import argparse
import logging
from pathlib import Path

from config.config import Config
from preprocessing.text_extractor import TextExtractor
from preprocessing.text_cleaner import TextCleaner
from preprocessing.feature_extractor import FeatureExtractor
from models.naive_bayes import NaiveBayesClassifier
from models.sbert_model import SBERTModel
from models.bert_model import BERTModel
from models.ensemble import EnsembleModel
from evaluation.metrics import Evaluator
from utils.data_loader import DataLoader
from database.db_handler import DatabaseHandler

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cv_matcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Sistema de Asignación de Posiciones basado en CVs')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'evaluate'], 
                        default='predict', help='Modo de operación')
    parser.add_argument('--cv_path', type=str, help='Ruta al CV o directorio de CVs')
    parser.add_argument('--job_desc_path', type=str, help='Ruta a la descripción del puesto')
    parser.add_argument('--model', type=str, choices=['naive_bayes', 'sbert', 'bert', 'ensemble'],
                        default='ensemble', help='Modelo a utilizar')
    parser.add_argument('--config', type=str, default='config/default.yaml', 
                        help='Ruta al archivo de configuración')
    
    return parser.parse_args()

def main():
    """Función principal del sistema."""
    args = parse_arguments()
    config = Config(args.config)
    logger.info(f"Iniciando sistema en modo: {args.mode}")
    
    # Inicializar componentes
    data_loader = DataLoader(config)
    text_extractor = TextExtractor(config)
    text_cleaner = TextCleaner(config)
    feature_extractor = FeatureExtractor(config)
    db_handler = DatabaseHandler(config)
    
    # Seleccionar modelo
    if args.model == 'naive_bayes':
        model = NaiveBayesClassifier(config)
    elif args.model == 'sbert':
        model = SBERTModel(config)
    elif args.model == 'bert':
        model = BERTModel(config)
    else:  # ensemble
        model = EnsembleModel(config)
    
    # Ejecutar según el modo
    if args.mode == 'train':
        # Cargar datos de entrenamiento
        cv_data, job_desc_data = data_loader.load_training_data()
        
        # Preprocesar datos
        cv_texts = [text_extractor.extract(cv) for cv in cv_data]
        cv_texts = [text_cleaner.clean(text) for text in cv_texts]
        
        job_texts = [text_cleaner.clean(desc) for desc in job_desc_data]
        
        # Extraer características
        cv_features = [feature_extractor.extract_features(text) for text in cv_texts]
        job_features = [feature_extractor.extract_features(text) for text in job_texts]
        
        # Entrenar modelo
        model.train(cv_features, job_features)
        logger.info(f"Modelo {args.model} entrenado exitosamente")
        
    elif args.mode == 'predict':
        # Cargar CV y descripción del puesto
        cv_path = Path(args.cv_path) if args.cv_path else None
        job_desc_path = Path(args.job_desc_path) if args.job_desc_path else None
    
        if not cv_path or not job_desc_path:
            logger.error("Se requieren las rutas al CV y a la descripción del puesto")
            return
        
        # Extraer y limpiar texto
        cv_text = text_extractor.extract(cv_path)
        cv_text = text_cleaner.clean(cv_text)
        
        job_text = text_cleaner.clean(data_loader.load_text(job_desc_path))
        
        # Extraer características
        cv_features = feature_extractor.extract_features(cv_text)
        job_features = feature_extractor.extract_features(job_text)
        
        # Predecir
        score = model.predict(cv_features, job_features)
        logger.info(f"Puntuación de compatibilidad: {score:.2f}")
        
        # Guardar resultado en la base de datos
        db_handler.save_result(cv_path.name, job_desc_path.name, score)
        
    elif args.mode == 'evaluate':
        # Cargar datos de evaluación
        cv_data, job_desc_data, labels = data_loader.load_evaluation_data()
        
        # Preprocesar datos
        cv_texts = [text_extractor.extract(cv) for cv in cv_data]
        cv_texts = [text_cleaner.clean(text) for text in cv_texts]
        
        job_texts = [text_cleaner.clean(desc) for desc in job_desc_data]
        
        # Extraer características
        cv_features = [feature_extractor.extract_features(text) for text in cv_texts]
        job_features = [feature_extractor.extract_features(text) for text in job_texts]
        
        # Evaluar modelo
        evaluator = Evaluator(config)
        metrics = evaluator.evaluate(model, cv_features, job_features, labels)
        
        logger.info(f"Resultados de evaluación: {metrics}")

if __name__ == "__main__":
    main()
