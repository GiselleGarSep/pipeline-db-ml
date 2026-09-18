from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.contexts.api.controllers import HealthCheckController
from src.contexts.api.controllers import TrainModelController


### 🎵 API de Analítica Predictiva y MLOps - Base de Datos Chinook

Esta API expone un modelo de Machine Learning (Random Forest) entrenado de forma directa 
con datos consolidados desde **Supabase**. Permite estimar las preferencias musicales de un cliente 
analizando sus variables demográficas y de contacto.

**Desarrollado con:** FastAPI, Docker, PostgreSQL (Supabase) y Scikit-Learn.
"""

class ApiApp:
    def __init__(self):
        # 🎨 Personalizamos FastAPI con títulos, descripción, contactos y licencias corporativas
        self.app = FastAPI(
            title="🎵 Chinook Music Predictor API",
            description=description,
            version="1.0.0",
            contact={
                "name": "Giselle Garcia",
                "email": "0151990@up.edu.mx",
            },
            license_info={
                "name": "Apache 2.0",
                "url": "https://apache.org",
            }
        )
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.setup_routes()

    def setup_routes(self):
        # 🏷️ El parámetro tags sirve para agrupar visualmente los endpoints en secciones
        self.app.add_api_route(
            "/api/health-check",
            HealthCheckController().execute, 
            methods=["GET"],
            tags=["Infraestructura & Monitoreo"]
        )
        
        self.app.add_api_route(
            "/api/model",
            TrainModelController().execute, 
            methods=["POST"],
            tags=["Predicciones de Inteligencia Artificial"]
        )

    def start(self):
        print(f"\n 🚀 init ApiApp")
        uvicorn.run(self.app, host="0.0.0.0", port=8000)
