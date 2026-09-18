import os
import io
import joblib
import boto3
import numpy as np
from fastapi import HTTPException
from src.contexts.api.models import PredictorRequest
from src.TrainModel import train_music_model

class TrainModelController:
    # 1. Proceso de Entrenamiento (Sin parámetros, procesa toda la vista de Supabase)
    def execute_train(self):
        try:
            accuracy = train_music_model()
            return {
                "status": "success",
                "message": "El modelo para Chinook se entrenó y guardó de manera exitosa.",
                "metrics": {
                    "accuracy": round(accuracy, 4)
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error en el entrenamiento del modelo: {str(e)}")

    # 2. Proceso de Predicción (SÍ recibe parámetros desde Swagger UI)
    def execute_predict(self, request: PredictorRequest):
        try:
            bucket_name = os.environ.get("AWS_S3_BUCKET_NAME")
            model_key = "models/music_predictor_model.pkl"
            
            # Descargar el modelo entrenado desde Amazon S3 (o usar local de respaldo)
            if bucket_name:
                s3_client = boto3.client('s3')
                response = s3_client.get_object(Bucket=bucket_name, Key=model_key)
                model_file = io.BytesIO(response['Body'].read())
                model = joblib.load(model_file)
            else:
                model = joblib.load("models/music_predictor_model.pkl")
                
            # Formatear la entrada para el pipeline
            input_data = [[request.tipo_correo, request.pais_origen, request.ciudad_origen]]
            
            # Realizar la predicción
            prediction = model.predict(input_data)
            
            return {
                "status": "success",
                "prediction": prediction[0]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al procesar la predicción: {str(e)}")
