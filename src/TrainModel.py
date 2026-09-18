import os
import io
import joblib
import boto3
import pandas as pd
from supabase import create_client
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def train_music_model():
    # 1. Conexión a Supabase y extracción de datos desde tu vista
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Faltan las variables de entorno SUPABASE_URL o SUPABASE_KEY")
        
    supabase = create_client(url, key)
    response = supabase.table("vista_clientes_generos").select("*").execute()
    df = pd.DataFrame(response.data)
    
    if df.empty:
        raise ValueError("No se encontraron registros en 'vista_clientes_generos'")

    # 2. Separación de Features (X) y Target (y)
    X = df[['tipo_correo', 'pais_origen', 'ciudad_origen']]
    y = df['genero_musical']
    
    # 3. División del Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Pipeline: Codificación de variables de texto + Modelo de Clasificación
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['tipo_correo', 'pais_origen', 'ciudad_origen'])
        ]
    )
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # 5. Entrenamiento
    model_pipeline.fit(X_train, y_train)
    accuracy = model_pipeline.score(X_test, y_test)
    
    # 6. Almacenamiento directo en AWS S3
    bucket_name = os.environ.get("AWS_S3_BUCKET_NAME")
    if bucket_name:
        buffer = io.BytesIO()
        joblib.dump(model_pipeline, buffer)
        buffer.seek(0)
        
        s3_client = boto3.client('s3')
        s3_client.upload_fileobj(buffer, bucket_name, "models/music_predictor_model.pkl")
        print("Modelo guardado exitosamente en Amazon S3.")
    else:
        os.makedirs("models", exist_ok=True)
        joblib.dump(model_pipeline, "models/music_predictor_model.pkl")
        print("Modelo guardado localmente (AWS_S3_BUCKET_NAME no configurado).")
        
    return accuracy
