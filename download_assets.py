import os
from dotenv import load_dotenv
import boto3

# 1. Cargar credenciales del archivo .env
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# 2. Inicializar cliente de S3
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION
)

# 3. Mapeo exacto de archivos (Ruta S3 -> Ruta Local en tu EC2)
archivos_a_descargar = {
    # Arquitectura común
    "gold_v4/encoders/encoder.py": "common/encoder.py",
    "gold_v4/encoders/enc_beat.pth": "backend/enc_beat.pth",
    "gold_v4/encoders/enc_rhythm.pth": "backend/enc_rhythm.pth",
    # Pesos de los clasificadores especializados (Cabezales)
    "gold_v4/heads/head_afib.pth": "backend/head_afib.pth",
    "gold_v4/heads/head_lbbb.pth": "backend/head_lbbb.pth",
    "gold_v4/heads/head_rbbb.pth": "backend/head_rbbb.pth",
    "gold_v4/heads/head_pvc.pth": "backend/head_pvc.pth",
    # Base de conocimiento vectorial (FAISS)
    "gold_v4/vector_db/base_conocimiento_mimic.index": "backend/base_conocimiento_mimic.index",
    "gold_v4/vector_db/metadata_faiss.json": "backend/metadata_faiss.json"
    # Base de conocimiento vectorial (FAISS)
    "gold_v4/vector_db/base_conocimiento_mimic.index": "backend/base_conocimiento_mimic.index",
    "gold_v4/vector_db/metadata_faiss.json": "backend/metadata_faiss.json"
}

print(f"⏳ Iniciando descarga de artefactos desde s3://{BUCKET_NAME}...\n")

for s3_key, local_path in archivos_a_descargar.items():
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"⬇️ Descargando {s3_key} -> {local_path}...")
    try:
        s3_client.download_file(BUCKET_NAME, s3_key, local_path)
        print(f"✅ ¡Completado!")
    except Exception as e:
        print(f"❌ Error al descargar {s3_key}: {e}")

print("\n🎉 ¡Todos los pesos, la arquitectura y el índice FAISS están sincronizados en tu instancia de AWS!")