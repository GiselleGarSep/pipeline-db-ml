from pydantic import BaseModel, Field

class PredictorRequest(BaseModel):
    tipo_correo: str = Field(..., description="Dominio del correo, ej: gmail.com")
    pais_origen: str = Field(..., description="País del cliente, ej: Brazil")
    ciudad_origen: str = Field(..., description="Ciudad del cliente, ej: São Paulo")

    class Config:
        json_schema_extra = {
            "example": {
                "tipo_correo": "gmail.com",
                "pais_origen": "Brazil",
                "ciudad_origen": "São Paulo"
            }
        }
