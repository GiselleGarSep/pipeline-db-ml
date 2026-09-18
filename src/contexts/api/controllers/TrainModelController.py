from fastapi import APIRouter, HTTPException
from src.TrainModel import train_music_model

router = APIRouter()

@router.post("/execute_api_model")
def run_training():
    try:
        # Llama a tu función que descarga de Supabase y entrena el modelo
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
