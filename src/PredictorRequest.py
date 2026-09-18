from pydantic import BaseModel

class PredictorRequest(BaseModel):
    tipo_correo: str
    pais_origen: str
    ciudad_origen: str

    class Config:
        json_schema_extra = {
            "example": {
                "tipo_correo": "gmail.com",
                "pais_origen": "Brazil",
                "ciudad_origen": "São Paulo"
            }
        }
