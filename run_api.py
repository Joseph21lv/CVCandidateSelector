#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para ejecutar la API REST.
"""

import os
import sys
from pathlib import Path

# Asegurarse de que el directorio del proyecto esté en el path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

# Importar la API
from api.app import start_api

if __name__ == "__main__":
    start_api()
