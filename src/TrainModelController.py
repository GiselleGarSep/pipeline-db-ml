from fastapi import APIRouter, HTTPException
from src.TrainModel import train_music_model

router = APIRouter(prefix="/ml", tags=["Machine Learning"])

@router.post("/train")
def run_training():
    try:
        accuracy = train_music_model()
        return {
            "status": "success",
            "message": "Pipeline ejecutado correctamente.",
            "metrics": {
                "accuracy": round(accuracy, 4)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el pipeline de entrenamiento: {str(e)}")
