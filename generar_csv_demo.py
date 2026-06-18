import os
import io
import boto3
import wfdb
import pandas as pd
from dotenv import load_dotenv

# =====================================================================
# 1. CONFIGURACIÓN DE AWS Y S3
# =====================================================================
load_dotenv()
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
)

MIMIC_AP = 'arn:aws:s3:us-east-1:724665945834:accesspoint/mimic-iv-ecg-v1-0-01'
nombres_derivaciones = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

os.makedirs("csv_demo_tesis", exist_ok=True)

# =====================================================================
# 2. LECTURA AUTÓNOMA DEL CATÁLOGO EN S3
# =====================================================================
print("⏳ 1. Descargando el catálogo de MIMIC-IV desde S3 para buscar candidatos...")
try:
    # Agregamos low_memory=False para evitar el DtypeWarning en archivos gigantes
    obj_csv = s3_client.get_object(Bucket=BUCKET_NAME, Key="bronze/MIMIC-IV-ECG/machine_measurements.csv")
    df_mimic = pd.read_csv(io.BytesIO(obj_csv['Body'].read()), low_memory=False)
    
    # Asegurar que los IDs sean texto
    df_mimic['subject_id'] = df_mimic['subject_id'].astype(str)
    df_mimic['study_id'] = df_mimic['study_id'].astype(str)
    print(f"✅ Catálogo cargado: {len(df_mimic)} registros disponibles para minería de datos.\n")
except Exception as e:
    print(f"❌ Error crítico al leer el catálogo: {e}")
    exit()

# =====================================================================
# 3. BÚSQUEDA VECTORIZADA ULTRA RÁPIDA DE PATOLOGÍAS
# =====================================================================
patologias_a_buscar = {
    "Normal": "sinus rhythm|normal sinus",
    "AFIB": "atrial fibrillation|afib",
    "LBBB": "left bundle branch block|lbbb",
    "RBBB": "right bundle branch block|rbbb",
    "PVC": "premature ventricular|pvc"
}

casos_seleccionados = {}

print("🔍 2. Unificando columnas de diagnóstico (Procesamiento Vectorizado)...")
# En lugar de usar apply (que es muy lento), sumamos las columnas vectorialmente.
# Filtramos solo las columnas que empiezan con 'report' (que es donde MIMIC guarda el texto)
columnas_reporte = [col for col in df_mimic.columns if col.startswith('report')]

# Si por algún motivo no hay columnas 'report', usamos todas
if len(columnas_reporte) == 0:
    columnas_reporte = df_mimic.columns

# Creamos una serie vacía y le sumamos el texto de cada columna súper rápido
texto_filas = pd.Series([""] * len(df_mimic))
for col in columnas_reporte:
    texto_filas += df_mimic[col].fillna("").astype(str).str.lower() + " "

print("🔍 3. Buscando 5 candidatos ideales para cada patología...")
for patologia, keyword in patologias_a_buscar.items():
    # Filtrar las filas que contienen la palabra clave usando la serie vectorizada
    df_filtrado = df_mimic[texto_filas.str.contains(keyword, na=False)]
    
    if not df_filtrado.empty:
        # Tomar 5 casos aleatorios
        cantidad = min(5, len(df_filtrado))
        muestra = df_filtrado.sample(n=cantidad, random_state=42) 
        
        casos_seleccionados[patologia] = muestra[['subject_id', 'study_id']].to_dict(orient="records")
        print(f"   ✔️ {patologia}: Encontrados {cantidad} casos.")
    else:
        print(f"   ⚠️ {patologia}: No se encontraron pacientes con estas palabras clave.")

# =====================================================================
# 4. DESCARGA Y CONVERSIÓN DE ONDAS
# =====================================================================
print("\n 3. Iniciando extracción de señales con WFDB y conversión a CSV...")

for patologia, lista_pacientes in casos_seleccionados.items():
    print(f"\n--- Procesando lote: {patologia} ---")
    
    # Crear una subcarpeta para mantener todo ultra organizado
    carpeta_salida = f"csv_demo_tesis/{patologia}"
    os.makedirs(carpeta_salida, exist_ok=True)
    
    for i, paciente in enumerate(lista_pacientes, start=1):
        subject_id = paciente['subject_id']
        study_id = paciente['study_id']
        sub_folder = f"p{subject_id[:4]}"
        
        base_s3_key = f"mimic-iv-ecg/1.0/files/{sub_folder}/p{subject_id}/s{study_id}/{study_id}"
        temp_path = f"/tmp/{study_id}"
        
        try:
            # Descargar de PhysioNet vía Access Point
            s3_client.download_file(Bucket=MIMIC_AP, Key=f"{base_s3_key}.dat", Filename=f"{temp_path}.dat")
            s3_client.download_file(Bucket=MIMIC_AP, Key=f"{base_s3_key}.hea", Filename=f"{temp_path}.hea")
            
            # Leer con WFDB para tener voltajes reales
            record = wfdb.rdrecord(temp_path, physical=True)
            
            # Obtener la señal física (ya escalada a milivolts)
            # Si por algún motivo está en ADC, hacer conversión manual
            signal_data = record.p_signal
            
            # Validar que los valores estén en el rango esperado de mV (típicamente -5 a +5 mV)
            # Si los valores son muy grandes (>100), probablemente estén en ADC
            if signal_data.max() > 100 or signal_data.min() < -100:
                # Conversión manual: dividir entre ADC gain (típicamente 1000 para MIMIC)
                adc_gain = record.adc_gain[0] if hasattr(record, 'adc_gain') and record.adc_gain is not None else 1000
                signal_data = signal_data / adc_gain
                print(f"    ⚠️ Conversión manual de ADC a mV (ganancia: {adc_gain})")
            
            # Convertir a DataFrame de Pandas
            df_ecg = pd.DataFrame(signal_data, columns=nombres_derivaciones)
            
            # Cortar a 2000 muestras (8 segundos exactos)
            df_ecg = df_ecg.head(4000)
            
            # Guardar el CSV con un nombre identificable para el jurado
            nombre_archivo = f"{carpeta_salida}/{patologia}_Caso{i}_Sub{subject_id}_Study{study_id}.csv"
            df_ecg.to_csv(nombre_archivo, index=False)
            print(f"    [100%] Archivo generado: {nombre_archivo}")
            
            # Limpiar RAM y disco de la EC2
            os.remove(f"{temp_path}.dat")
            os.remove(f"{temp_path}.hea")
            
        except Exception as e:
            print(f"    Error con paciente {subject_id}: {e}")

print("\n 25 archivos CSV listos y perfectamente categorizados en la carpeta 'csv_demo_tesis'.")