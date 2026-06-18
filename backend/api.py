import os
import io
import re
import json
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import boto3
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import numpy as np
import pandas as pd
import scipy.signal as signal
from rich import print
import wfdb
import numpy as np
from botocore.config import Config
import scipy.io as sio

# =====================================================================
# 1. CONFIGURACIÓN DE SEGURIDAD (.env) Y ENTORNO
# =====================================================================
ruta_env = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(ruta_env)

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "shazam-proyecto-ecg")

# Aseguramos que Python encuentre la arquitectura común en la carpeta /common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from common.encoder import BeatEncoder, RhythmEncoder
except ImportError:
    raise RuntimeError(" No se encontró 'encoder.py' en la carpeta 'common'. Asegúrate de haberlo creado allí.")

app = FastAPI(
    title=" Shazam ECG - Cerebro Clínico API",
    description="Backend de producción de alto rendimiento para el diagnóstico híbrido (IA Multi-Task + FAISS)",
    version="4.0"
)

DEVICE = "cpu"  # Forzado a CPU para optimizar la RAM de tu t3.large
TARGET_FS = 250
TARGET_LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
THRESHOLDS = {"AFIB": 0.85, "LBBB": 0.50, "RBBB": 0.50, "PVC": 0.70}

# Cabezal clasificador para las redes especializadas
class SpecialistHead(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(64, 2)
        )
    def forward(self, x): return self.net(x)

# =====================================================================
# 2. CARGA DE MODELOS Y FAISS EN MEMORIA (Precalentamiento)
# =====================================================================
print("\n[IA] Inicializando e instalando redes neuronales en la memoria RAM...")

enc_beat = BeatEncoder().to(DEVICE)
enc_rhythm = RhythmEncoder().to(DEVICE)
head_afib = SpecialistHead(0.3).to(DEVICE)
head_lbbb = SpecialistHead(0.2).to(DEVICE)
head_rbbb = SpecialistHead(0.2).to(DEVICE)
head_pvc = SpecialistHead(0.2).to(DEVICE)

# Se asume que arrastraste estos archivos a la carpeta 'backend/' en VS Code
try:
    enc_beat.load_state_dict(torch.load("enc_beat.pth", map_location=DEVICE, weights_only=True)); enc_beat.eval()
    enc_rhythm.load_state_dict(torch.load("enc_rhythm.pth", map_location=DEVICE, weights_only=True)); enc_rhythm.eval()
    head_afib.net.load_state_dict(torch.load("head_afib.pth", map_location=DEVICE, weights_only=True)); head_afib.eval()
    head_lbbb.net.load_state_dict(torch.load("head_lbbb.pth", map_location=DEVICE, weights_only=True)); head_lbbb.eval()
    head_rbbb.net.load_state_dict(torch.load("head_rbbb.pth", map_location=DEVICE, weights_only=True)); head_rbbb.eval()
    head_pvc.net.load_state_dict(torch.load("head_pvc.pth", map_location=DEVICE, weights_only=True)); head_pvc.eval()
    print("✅ Modelos de Deep Learning cargados exitosamente.")
except Exception as e:
    print(f"⚠️ Alerta: No se pudieron cargar los archivos de peso .pth locales: {e}")

# Inicializar motor de búsqueda FAISS
try:
    faiss_index = faiss.read_index("base_conocimiento_mimic.index")
    with open("metadata_faiss.json", "r") as f:
        base_conocimiento_metadata = {int(k): v for k, v in json.load(f).items()}
    print(f"✅ Motor Vectorial FAISS cargado con {faiss_index.ntotal} firmas clínicas.")
except Exception as e:
    print(f"⚠️ Alerta: No se pudo levantar el índice FAISS local: {e}")

# Inicializar cliente seguro de S3
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION,
    config=Config(s3={'use_arn_region': True})
)
print(f"🔒 Conectado de forma segura a AWS S3 (Bucket: {BUCKET_NAME})\n")

# Cargar el mapeador de MIMIC desde tu S3 Bronze
print("⏳ Cargando catálogo de sujetos MIMIC-IV desde S3 Bronze...")
try:
    obj_csv = s3_client.get_object(Bucket=BUCKET_NAME, Key="bronze/MIMIC-IV-ECG/machine_measurements.csv")
    df_mimic_catalog = pd.read_csv(io.BytesIO(obj_csv['Body'].read()))
    
    # Limpiamos y aseguramos que los IDs sean strings para evitar problemas de formato
    df_mimic_catalog['subject_id'] = df_mimic_catalog['subject_id'].astype(str)
    df_mimic_catalog['study_id'] = df_mimic_catalog['study_id'].astype(str)
    
    print(f"✅ Catálogo MIMIC cargado. {len(df_mimic_catalog)} registros indexados en RAM.")
except Exception as e:
    print(f"⚠️ Alerta: No se pudo cargar machine_measurements.csv desde S3: {e}")
    df_mimic_catalog = pd.DataFrame() # Fallback vacío para que no colapse el back

# Cargar el catálogo CPSC desde S3
print("⏳ Cargando catálogo CPSC desde S3 Bronze...")
try:
    obj_csv_cpsc = s3_client.get_object(Bucket=BUCKET_NAME, Key="bronze/cpsc/mapeo_patologias_cpsc.csv")
    df_cpsc_catalog = pd.read_csv(io.BytesIO(obj_csv_cpsc['Body'].read()))
    
    # Aseguramos que la columna Record sea string
    df_cpsc_catalog['Record'] = df_cpsc_catalog['Record'].astype(str)
    
    print(f"✅ Catálogo CPSC cargado. {len(df_cpsc_catalog)} registros indexados en RAM.")
except Exception as e:
    print(f"⚠️ Alerta: No se pudo cargar mapeo_patologias_cpsc.csv desde S3: {e}")
    df_cpsc_catalog = pd.DataFrame() # Fallback por si falla

# =====================================================================
# 3. MÓDULO MATEMÁTICO DE FILTRADO Y PROCESAMIENTO CLÍNICO
# =====================================================================
def filtrar_y_normalizar_senal(raw_signal, fs_original):
    """Aplica remoción de tendencia, filtro Butterworth y Z-score norm."""
    if fs_original != TARGET_FS:
        num = int(len(raw_signal) * TARGET_FS / fs_original)
        raw_signal = signal.resample(raw_signal, num)
    detrended = signal.detrend(raw_signal)
    nyq = 0.5 * TARGET_FS
    b, a = signal.butter(4, [0.5 / nyq, 45.0 / nyq], btype='band')
    filtered = signal.filtfilt(b, a, detrended)
    filtered_norm = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
    return filtered_norm

def detectar_picos_qrs_centrados(filt_sigs_3d, fs):
    """Detecta latidos en la derivación II usando operador de energía de Pan-Tompkins simplificado."""
    lead_ii = filt_sigs_3d[0, 1, :]
    diff = np.diff(lead_ii)
    sq = diff ** 2
    window_size = int(0.12 * fs)
    mavg = np.convolve(sq, np.ones(window_size)/window_size, mode='same')
    peaks_energy, _ = signal.find_peaks(mavg, distance=int(0.3 * fs), prominence=np.max(mavg)*0.10)
    
    centered_peaks = []
    total_len = filt_sigs_3d.shape[2]
    search_window = int(fs * 0.08)

    for p in peaks_energy:
        start_p = max(0, p - search_window)
        end_p = min(total_len, p + search_window)
        if start_p < end_p:
            abs_sum = np.sum(np.abs(filt_sigs_3d[0, :, start_p:end_p]), axis=0)
            true_peak = start_p + np.argmax(abs_sum)
            if true_peak >= 100 and true_peak + 100 <= total_len:
                centered_peaks.append(true_peak)
    return np.unique(centered_peaks)

def ejecutar_motor_hibrido_shazam(filt_sigs_3d, raw_sigs_3d):
    """
    Ejecuta inferencia multi-task y búsqueda morfológica indexada en FAISS
    a nivel granular (latido a latido), retornando trazas completas de las
    12 derivaciones (crudas/filtradas) y sus respectivas ventanas de análisis.
    """
    total_len_filt = filt_sigs_3d.shape[2]
    
    # REMUESTREO AUTOMÁTICO DE LA SEÑAL CRUDA
    # Si la señal cruda tiene una longitud distinta a la filtrada, la igualamos aquí mismo
    if raw_sigs_3d.shape[2] != total_len_filt:
        raw_sigs_3d = signal.resample(raw_sigs_3d, total_len_filt, axis=2)

    if total_len_filt < 2000:
        return {"error": "Señal demasiado corta. Se requiere una ventana válida para análisis."}

    # 1. Evaluación de Ritmo Macroscópico (Ventana global)
    # Tomamos las primeras 2000 muestras solo para la inferencia de AFIB (si tu red lo exige)
    t_2000 = torch.tensor(filt_sigs_3d[:, :, :2000], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        score_afib = F.softmax(head_afib(enc_rhythm(t_2000)), dim=1)[0][1].item()

    # 2. Segmentación de Complejos QRS (Morfología de Latidos a 0.8s)
    valid_peaks = detectar_picos_qrs_centrados(filt_sigs_3d, TARGET_FS)
    if len(valid_peaks) == 0:
        return {"error": "No se detectaron complejos QRS válidos en la Derivación II."}

    lbbb_scores, rbbb_scores, pvc_scores = [], [], []
    beat_embeddings = []
    lista_latidos_detallada = []

    with torch.no_grad():
        for idx, p in enumerate(valid_peaks):
            # Recorte estricto de ventanas de 200 muestras (0.8 segundos) centrado en el QRS
            w200_filt = filt_sigs_3d[:, :, p-100:p+100]
            
            # Inferencia profunda individual del latido
            feat_beat_tensor = torch.tensor(w200_filt, dtype=torch.float32).to(DEVICE)
            feat_beat = enc_beat(feat_beat_tensor)
            beat_embeddings.append(feat_beat)
            
            # Extraer porcentajes de predicción individuales para este latido
            lbbb_p = F.softmax(head_lbbb(feat_beat), dim=1)[0][1].item()
            rbbb_p = F.softmax(head_rbbb(feat_beat), dim=1)[0][1].item()
            pvc_p = F.softmax(head_pvc(feat_beat), dim=1)[0][1].item()
            
            lbbb_scores.append(lbbb_p)
            rbbb_scores.append(rbbb_p)
            pvc_scores.append(pvc_p)
            
            # SHAZAM ECG: Búsqueda Vectorial Individual de Vecinos Cercanos para este Latido
            signature_beat = F.normalize(feat_beat, dim=1).cpu().numpy()
            scores_coseno_b, indices_vecinos_b = faiss_index.search(signature_beat, 3)
            
            vecinos_shazam_beat = []
            for i in range(3):
                vecino_idx = int(indices_vecinos_b[0][i])
                similitud_pct = float(scores_coseno_b[0][i] * 100)
                if vecino_idx in base_conocimiento_metadata:
                    meta = base_conocimiento_metadata[vecino_idx]
                    vecinos_shazam_beat.append({
                        "id_caso": meta["id_caso"],
                        "diagnostico_base": meta["diagnostico"],
                        "similitud": round(similitud_pct, 2)
                    })
            
            # Empaquetamos la metadata microclínica detallada del latido
            lista_latidos_detallada.append({
                "num_latido": idx + 1,
                "pico_muestra": int(p),
                "tiempo_segundos": round(p / TARGET_FS, 2),
                "predicciones_morfologia": {
                    "LBBB": round(lbbb_p, 4),
                    "RBBB": round(rbbb_p, 4),
                    "PVC": round(pvc_p, 4)
                },
                "shazam_vecinos_latido": vecinos_shazam_beat
            })

        # 3. SHAZAM ECG GLOBAL (Consistencia del mapa promedio del paciente completo)
        patient_signature = torch.mean(torch.stack(beat_embeddings), dim=0)
        patient_signature = F.normalize(patient_signature, dim=1).cpu().numpy()
        scores_coseno, indices_vecinos = faiss_index.search(patient_signature, 3)
        
        vecinos_shazam_global = []
        for i in range(3):
            vecino_idx = int(indices_vecinos[0][i])
            similitud_pct = float(scores_coseno[0][i] * 100)
            if vecino_idx in base_conocimiento_metadata:
                meta = base_conocimiento_metadata[vecino_idx]
                vecinos_shazam_global.append({
                    "id_caso": meta["id_caso"],
                    "diagnostico_base": meta["diagnostico"],
                    "similitud": round(similitud_pct, 2)
                })

    # Calcular medias aritméticas globales para estabilidad diagnóstica macro
    avg_lbbb = np.mean(lbbb_scores)
    avg_rbbb = np.mean(rbbb_scores)
    avg_pvc = np.mean(pvc_scores)

    scores_dict = {"AFIB": score_afib, "LBBB": avg_lbbb, "RBBB": avg_rbbb, "PVC": avg_pvc}
    alarmas_activas = [k for k, v in scores_dict.items() if v >= THRESHOLDS[k]]
    
    #  SOLUCIÓN CONCILIACIÓN RITMO SINUSAL NORMAL
    if not alarmas_activas:
        diagnostico_clinico_final = "Ritmo Sinusal Normal"
    else:
        diagnostico_clinico_final = f"Anomalías Detectadas: {', '.join(alarmas_activas)}"

    return {
        "score_afib": round(score_afib, 4),
        "score_lbbb_promedio": round(avg_lbbb, 4),
        "score_rbbb_promedio": round(avg_rbbb, 4),
        "score_pvc_promedio": round(avg_pvc, 4),
        "alarmas_ia": alarmas_activas,
        "diagnostico_clinico_final": diagnostico_clinico_final,
        "num_latidos_detectados": len(valid_peaks),
        "shazam_global_vecinos": vecinos_shazam_global,
        "analisis_latido_a_latido": lista_latidos_detallada,
        "senales_completas_ UCI": {
            "filtradas_12_derivaciones": filt_sigs_3d[0].tolist(), 
            "crudas_12_derivaciones": raw_sigs_3d[0].tolist()       
        }
    }
# =====================================================================
# 4. PROTOCOLOS DE ENTRADA Y SALIDA (Pydantic Models)
# =====================================================================
class SolicitudID(BaseModel):
    subject_id: str
    study_id: str = None  # Opcional para CPSC
    fuente: str = "MIMIC" # Por defecto MIMIC, pero el Front puede enviar "CPSC"

# =====================================================================
# 5. ENDPOINTS DE LA API (Servicios HTTP)
# =====================================================================
@app.get("/")
def home():
    """Ruta de salud de la API."""
    return {
        "status": "online",
        "proyecto": "Shazam ECG V4 - Sistema de Soporte a Decisiones Médicas en la UCI",
        "faiss_status": f"Cargado con {faiss_index.ntotal} vectores."
    }

@app.get("/lista-sujetos")
def obtener_lista_sujetos():
    """Devuelve todos los sujetos para no saturar el menú desplegable del Front."""
    if df_mimic_catalog.empty:
        return {"sujetos": []}
    #Head de 100 pacientes
    muestras = df_mimic_catalog[['subject_id', 'study_id']].head(200).to_dict(orient="records")
    return {"sujetos": muestras}

@app.get("/lista-sujetos-cpsc")
def obtener_lista_sujetos_cpsc():
    """Devuelve una muestra de sujetos de CPSC para el menú desplegable."""
    if df_cpsc_catalog.empty:
        return {"sujetos": []}
    # Mandamos el Record y su patología base para que el menú del Front sea informativo
    muestras = df_cpsc_catalog[['Record', 'Todas_Patologias']].head(200).to_dict(orient="records")
    return {"sujetos": muestras}


@app.post("/diagnosticar-id")
async def diagnosticar_por_id(datos: SolicitudID):
    """Ingesta dinámica (Multi-Dataset): Lee de MIMIC (.dat) o CPSC (.mat) basado en la fuente."""
    fuente = datos.fuente.upper()

    try:
        # =================================================================
        # RAMA A: INGESTA DESDE MIMIC-IV-ECG
        # =================================================================
        if fuente == "MIMIC":
            if df_mimic_catalog.empty:
                raise HTTPException(status_code=503, detail="Catálogo MIMIC no disponible.")
            if not datos.study_id:
                raise HTTPException(status_code=400, detail="MIMIC requiere un 'study_id'.")
                
            temp_path = f"/tmp/{datos.study_id}"
            sub_folder = f"p{datos.subject_id[:4]}"
            base_s3_key = f"mimic-iv-ecg/1.0/files/{sub_folder}/p{datos.subject_id}/s{datos.study_id}/{datos.study_id}"
            MIMIC_AP = 'arn:aws:s3:us-east-1:724665945834:accesspoint/mimic-iv-ecg-v1-0-01'
            
            print(f"⬇️ Descargando MIMIC: {datos.study_id} ...")
            s3_client.download_file(Bucket=MIMIC_AP, Key=f"{base_s3_key}.dat", Filename=f"{temp_path}.dat")
            s3_client.download_file(Bucket=MIMIC_AP, Key=f"{base_s3_key}.hea", Filename=f"{temp_path}.hea")
            
            record = wfdb.rdrecord(temp_path)
            matriz_raw = record.p_signal.T  
            fs_original = record.fs
            
            os.remove(f"{temp_path}.dat")
            os.remove(f"{temp_path}.hea")

        # =================================================================
        # RAMA B: INGESTA DESDE CPSC (Formato .mat)
        # =================================================================
        elif fuente == "CPSC":
            if df_cpsc_catalog.empty:
                raise HTTPException(status_code=503, detail="Catálogo CPSC no disponible.")
            # Buscar la ruta S3 exacta en el CSV mapeador
            fila = df_cpsc_catalog[df_cpsc_catalog['Record'] == datos.subject_id]
            if fila.empty:
                raise HTTPException(status_code=404, detail="Registro CPSC no encontrado en el catálogo.")
                
            s3_path_crudo = fila.iloc[0]['S3_Path_MAT']
            
            # 🛠️ CORRECCIÓN: Limpiamos el prefijo s3:// y el nombre del bucket si vienen incluidos en el CSV
            s3_key = s3_path_crudo.replace(f"s3://{BUCKET_NAME}/", "")
            
            temp_path = f"/tmp/{datos.subject_id}.mat"
            
            print(f"⬇️ Descargando CPSC: {datos.subject_id} desde Key limpio: {s3_key} ...")
            s3_client.download_file(Bucket=BUCKET_NAME, Key=s3_key, Filename=temp_path)

            # Leer el archivo .mat
            mat_data = sio.loadmat(temp_path)
            
            # La señal en archivos CPSC .mat casi siempre está bajo la llave 'val'
            if 'val' in mat_data:
                matriz_raw = mat_data['val'].astype(np.float32)
            else:
                # Fallback: tomar la primera llave útil
                llaves = [k for k in mat_data.keys() if not k.startswith('__')]
                matriz_raw = mat_data[llaves[0]].astype(np.float32)
                
            # Asegurar dimensión (12, muestras)
            if matriz_raw.shape[0] != 12 and matriz_raw.shape[1] == 12:
                matriz_raw = matriz_raw.T
                
            # CPSC suele guardarse en unidades ADU (Analog-to-Digital Units) con ganancia de 1000
            # pasar a escala en mV aproximada
            matriz_raw = matriz_raw * 0.001 
            fs_original = 500  # CPSC fue muestreado nativamente a 500 Hz
            
            os.remove(temp_path)
            
        else:
            raise HTTPException(status_code=400, detail="Fuente desconocida. Usa 'MIMIC' o 'CPSC'.")
        
        # =================================================================
        # PROCESAMIENTO COMÚN (Shazam IA)
        # =================================================================
        raw_sigs_3d = matriz_raw[np.newaxis, ...] # Expandir dimensión (1, 12, muestras)
        
        filt_sigs = []
        for c in range(12):
            filt_lead = filtrar_y_normalizar_senal(matriz_raw[c, :], fs_original=fs_original)
            filt_sigs.append(filt_lead)
            
        filt_sigs_3d = np.stack(filt_sigs)[np.newaxis, ...]
        
        # Ejecutamos el cerebro IA
        analisis_clinico = ejecutar_motor_hibrido_shazam(filt_sigs_3d, raw_sigs_3d)
        
        if "error" in analisis_clinico:
            raise HTTPException(status_code=422, detail=analisis_clinico["error"])
            
        return analisis_clinico

    except HTTPException:
        raise 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo crítico en procesamiento ({fuente}): {str(e)}")

@app.post("/diagnosticar-externo")
async def diagnosticar_externo(data: dict):
    try:
        # 1. Validar manualmente que el JSON contenga las llaves que necesitamos
        if "signal" not in data or "fs" not in data:
            raise HTTPException(
                status_code=400,
                detail="El payload JSON debe contener obligatoriamente las llaves 'signal' y 'fs'."
            )
            
        signal_data = data["signal"]
        fs_original = int(data["fs"])
        
        # 2. Convertir la lista matemática 2D a un objeto NumPy
        raw_np = np.array(signal_data, dtype=np.float32) 
        
        # Control de calidad automático por si el CSV viene transpuesto
        if raw_np.shape[0] != 12 and raw_np.shape[1] == 12:
            raw_np = raw_np.T
            
        if raw_np.shape[0] != 12:
            raise HTTPException(
                status_code=400, 
                detail=f"Estructura inválida. El CSV debe mapear exactamente 12 derivaciones. Se detectaron: {raw_np.shape[0]}"
            )
        
        # 3. Filtrar y normalizar la señal derivación por derivación
        filt_leads = []
        for i in range(12):
            # Filtramos usando tu excelente función matemática de fase cero
            lead_filtrado = filtrar_y_normalizar_senal(raw_np[i, :], fs_original=fs_original)
            filt_leads.append(lead_filtrado)
        
        filt_np = np.stack(filt_leads)
        
        # 4. Expandir dimensiones a 3D para PyTorch (Batch, Leads, Samples)
        raw_sigs_3d = raw_np[np.newaxis, ...]
        filt_sigs_3d = filt_np[np.newaxis, ...]
        
        # 5. Enviar las matrices limpias al motor híbrido de Shazam
        resultado = ejecutar_motor_hibrido_shazam(filt_sigs_3d, raw_sigs_3d)
        
        return resultado

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"❌ Error crítico en /diagnosticar-externo: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Fallo en la segmentación o inferencia de la señal externa: {str(e)}"
        )