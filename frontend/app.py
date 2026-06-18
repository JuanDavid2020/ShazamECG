import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
import streamlit.components.v1 as components
import io
import base64
import matplotlib.pyplot as plt

try:
    from weasyprint import HTML
    HAS_WEASYPRINT = True
except ImportError:
    HAS_WEASYPRINT = False

# =====================================================================
# CONFIGURACIÓN Y ESTILOS
# =====================================================================
st.set_page_config(page_title="Shazam ECG", page_icon="🫀", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; color: #333333; }
    .titulo-principal { font-size: 3.2rem; color: #0056b3; font-weight: bold; text-align: center; margin-bottom: 20px; }
    .metric-card { background-color: #F8F9FA; padding: 20px; border-radius: 10px; border-top: 4px solid #0056b3; text-align: center; }
    .vecino-box { padding: 10px; border-radius: 8px; margin-bottom: 10px; border-left: 5px solid #ccc; background-color: #f9f9f9;}
    .vecino-box.top1 { border-left-color: #ffc107; background-color: #fffdf2; border-width: 6px;}
    .badge { background: #28a745; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; float: right; }
    </style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"

@st.cache_data(show_spinner=False)
def obtener_listas():
    try:
        mimic = requests.get(f"{API_URL}/lista-sujetos").json().get("sujetos", [])
        cpsc = requests.get(f"{API_URL}/lista-sujetos-cpsc").json().get("sujetos", [])
        return mimic, cpsc
    except:
        return [], []

@st.cache_data(show_spinner=False)
def generar_pdf_reporte(diagnostico, afib, lbbb, rbbb,pvc, latidos_data, id_paciente):
    """ Función Maestra que renderiza el PDF Clínico con Resumen y Tabla Latido a Latido """
    if not HAS_WEASYPRINT:
        return None
        
    # Construimos dinámicamente las filas de la tabla HTML
    filas_html = ""
    for lat in latidos_data:
        num = lat['num_latido']
        tiempo = lat['tiempo_segundos']
        l_p = lat['predicciones_morfologia']['LBBB'] * 100
        r_p = lat['predicciones_morfologia']['RBBB'] * 100
        p_p = lat['predicciones_morfologia']['PVC'] * 100
        
        vecinos_html = ""
        for idx, vec in enumerate(lat['shazam_vecinos_latido']):
            clase = "alt" if idx == 0 else ""
            color_borde = "#ffc107" if idx == 0 else "#ccc"
            background = "#fffdf2" if idx == 0 else "#f8f9fa"
            
            vecinos_html += f"""
            <div style="background: {background}; border-left: 4px solid {color_borde}; padding: 6px 8px; margin-bottom: 5px; border-radius: 0 4px 4px 0;">
                <strong>#{idx+1} {vec['diagnostico_base']}</strong> ({vec['similitud']}% Similitud)<br>
                <small style="color:#555;">ID: {vec['id_caso']}</small>
            </div>
            """
            
        filas_html += f"""
        <tr>
            <td style="text-align:center;"><strong>#{num}</strong><br><span style="color:#666; font-size:8.5pt;">{tiempo} s</span></td>
            <td>LBBB: {l_p:.2f}%<br>RBBB: {r_p:.2f}%<br>PVC: {p_p:.2f}%</td>
            <td>{vecinos_html}</td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{ size: A4; margin: 20mm 15mm; }}
            body {{ font-family: 'Arial', sans-serif; color: #333333; font-size: 10pt; line-height: 1.4; }}
            .header-title {{ font-size: 24pt; color: #0056b3; font-weight: bold; margin: 0; }}
            .header-subtitle {{ font-size: 11pt; color: #555555; margin: 4px 0 25px 0; border-bottom: 2px solid #ccc; padding-bottom: 10px; }}
            .meta-container {{ background-color: #f8f9fa; border-top: 4px solid #0056b3; padding: 15px; margin-bottom: 20px; }}
            .verdict-box {{ background-color: #fffdf2; border-left: 5px solid #ffc107; padding: 12px; font-weight: bold; margin-bottom: 20px; }}
            .t-clinica {{ width: 100%; border-collapse: collapse; font-size: 9.5pt; margin-top: 15px; }}
            .t-clinica th {{ background-color: #0056b3; color: white; padding: 10px; text-align: left; }}
            .t-clinica td {{ border-bottom: 1px solid #eee; padding: 10px; vertical-align: top; }}
        </style>
    </head>
    <body>
        <div class="header-title">SHAZAM para el Corazón</div>
        <div class="header-subtitle">Reporte Clínico de Inteligencia Artificial</div>
        
        <div class="meta-container">
            <strong>ID del Análisis / Paciente:</strong> {id_paciente}<br>
            <strong>Diagnóstico Final IA:</strong> <span style="color:#0056b3; font-weight:bold;">{diagnostico}</span><br>
            <strong>AFIB:</strong> {afib*100:.1f}% | <strong>LBBB:</strong> {lbbb*100:.1f}% | <strong>RBBB:</strong> {rbbb*100:.1f}% | <strong>PVC:</strong> {pvc*100:.1f}%
        </div>
        
        <div class="verdict-box">Reporte Morfológico Detallado Latido a Latido (FAISS)</div>
        
        <table class="t-clinica">
            <thead>
                <tr>
                    <th width="12%">Latido</th>
                    <th width="28%">Predicciones PyTorch</th>
                    <th width="60%">Búsqueda Indexada (S3)</th>
                </tr>
            </thead>
            <tbody>
                {filas_html}
            </tbody>
        </table>
    </body>
    </html>
    """
    pdf_buffer = io.BytesIO()
    HTML(string=html_content).write_pdf(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer

# =====================================================================
# BARRA LATERAL
# =====================================================================
try:
    st.sidebar.image("frontend/logooficial.png", use_container_width=True)
except:
    st.sidebar.markdown("<h1 style='color:#0056b3; text-align:center;'>🫀 SHAZAM</h1>", unsafe_allow_html=True)

st.sidebar.markdown("---")
fuente_datos = st.sidebar.radio("📡 Fuente:", ["MIMIC-IV (S3)", "CPSC (S3)", "Archivo Local"])
mimic_lista, cpsc_lista = obtener_listas()

payload, endpoint = None, None
id_identificador = "Archivo_Local"

if fuente_datos == "MIMIC-IV (S3)":
    opciones = [f"Sub: {r['subject_id']} | Study: {r['study_id']}" for r in mimic_lista] if mimic_lista else ["No hay datos"]
    seleccion = st.sidebar.selectbox("Pacientes MIMIC:", opciones)
    if seleccion != "No hay datos":
        partes = seleccion.split(" | ")
        payload = {"fuente": "MIMIC", "subject_id": partes[0].split(": ")[1], "study_id": partes[1].split(": ")[1]}
        endpoint = "/diagnosticar-id"
elif fuente_datos == "CPSC (S3)":
    opciones = [f"Record: {r['Record']} | Patología: {r['Todas_Patologias']}" for r in cpsc_lista] if cpsc_lista else ["No hay datos"]
    seleccion = st.sidebar.selectbox("Pacientes CPSC:", opciones)
    if seleccion != "No hay datos":
        payload = {"fuente": "CPSC", "subject_id": seleccion.split(" | ")[0].split(": ")[1], "study_id": ""}
        endpoint = "/diagnosticar-id"
else:
    archivo_csv = st.sidebar.file_uploader("Subir CSV", type=["csv"])
    if archivo_csv:
        df = pd.read_csv(archivo_csv)
        try:
            float(df.columns[0])
            df = pd.read_csv(archivo_csv, header=None)
        except ValueError:
            pass
            
        columnas_basura = [
            c for c in df.columns 
            if 'time' in str(c).lower() 
            or 'unnamed' in str(c).lower() 
            or 'index' in str(c).lower()
            or 'elapsed' in str(c).lower()
        ]
        if columnas_basura:
            df = df.drop(columns=columnas_basura)
        
        df_calibrado = df
            
        payload = {"signal": df_calibrado.values.T.tolist(), "fs": 500}
        endpoint = "/diagnosticar-externo"
        id_identificador = archivo_csv.name

btn_cargar = st.sidebar.button("🚀 CARGAR Y ANALIZAR", type="primary", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Puedes escribir el ID de tu paciente en la caja para buscarlo rápido.")

# =====================================================================
# ÁREA PRINCIPAL
# =====================================================================
st.markdown("<div class='titulo-principal'>MONITOR UCI MULTIPARAMÉTRICO</div>", unsafe_allow_html=True)


if btn_cargar and payload:
    with st.spinner("⏳ Diagnóstico en proceso... Evaluando redes neuronales e indexando con FAISS..."):
        res = requests.post(f"{API_URL}{endpoint}", json=payload)
        
        if res.status_code == 200:
            data = res.json()

            if "error" in data:
                st.error(f"⚠️ El Motor Shazam detectó un problema en el archivo: {data['error']}")
            else:
                key_raiz = "senales_completas_ UCI" if "senales_completas_ UCI" in data else "senales_completas_12_leads"
                
                sub_keys = data[key_raiz].keys()
                key_crudas = "crudas_12_derivaciones" if "crudas_12_derivaciones" in sub_keys else "crudas_12_leads"
                key_filtradas = "filtradas_12_derivaciones" if "filtradas_12_derivaciones" in sub_keys else "filtradas_12_leads"
                
                senales_crudas = data[key_raiz][key_crudas]
                senales_filt = data[key_raiz][key_filtradas]
                
                picos_muestras = [lat["pico_muestra"] for lat in data.get("analisis_latido_a_latido", [])]
            
            rr_intervals = np.diff([lat["tiempo_segundos"] for lat in data["analisis_latido_a_latido"]]) if len(picos_muestras)>1 else []
            bpm_real = int(60 / np.mean(rr_intervals)) if len(rr_intervals) > 0 else 0

            # --- BOTÓN DE EXPORTACIÓN A PDF MÁGICO ---
            if HAS_WEASYPRINT:
                diag = data['diagnostico_clinico_final']
                afib = data['score_afib']
                lbbb = data['score_lbbb_promedio']
                rbbb = data['score_rbbb_promedio']
                pvc= data['score_pvc_promedio']
                latidos_data = data.get("analisis_latido_a_latido", [])
                
                pdf_bytes = generar_pdf_reporte(diag, afib, lbbb, rbbb, pvc, latidos_data, id_identificador)
                if pdf_bytes:
                    st.download_button(
                        label="📄 Exportar Reporte Clínico (PDF)",
                        data=pdf_bytes,
                        file_name=f"Reporte_SHAZAM_{id_identificador.split('|')[0].replace(' ','')}.pdf",
                        mime="application/pdf",
                    )
            else:
                st.warning("⚠️ Instala WeasyPrint ('pip install weasyprint') en tu EC2 para habilitar la exportación a PDF.")

            # =====================================================================
            # MOTOR GRÁFICO (NUEVO ALTO 2400px + ESPACIADO MATEMÁTICO PERFECTO)
            # =====================================================================
            html_code = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ margin: 0; font-family: sans-serif; cursor: crosshair; user-select: none; }}
                    /* 🛠️ AUMENTAMOS A 2400px DE ALTO PARA 12 CARRILES AMPLIOS */
                    .monitor-container {{ position: relative; width: 100%; height: 2400px; background: white; border: 1px solid #ccc; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }}
                    canvas {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; }}
                    
                    .overlay-play {{ position: absolute; top:0; left:0; width:100%; height:100%; background: rgba(255,255,255,0.85); z-index: 40; display: flex; justify-content: center; align-items: center; flex-direction: column; }}
                    .play-btn {{ padding: 20px 40px; font-size: 22px; background: #0056b3; color: white; border: none; border-radius: 50px; cursor: pointer; box-shadow: 0 4px 15px rgba(0,86,179,0.4); transition: 0.3s; font-weight: bold; }}
                    .play-btn:hover {{ background: #004494; transform: scale(1.05); }}
                    
                    .bpm-panel {{ position: fixed; top: 100px; right: 40px; background: white; border: 3px solid #ff4b4b; border-radius: 10px; padding: 15px; text-align: center; z-index: 50; box-shadow: 0 4px 10px rgba(0,0,0,0.2); }}
                    .bpm-value {{ font-size: 40px; font-weight: bold; color: #ff4b4b; }}
                    .leyenda {{ position: fixed; bottom: 30px; right: 40px; background: rgba(255,255,255,0.95); padding: 10px; border: 1px solid #ccc; border-radius: 5px; font-size: 13px; font-weight: bold; z-index: 50; box-shadow: 0 4px 10px rgba(0,0,0,0.2); pointer-events: none; }}
                    
                    #btnRepetir {{ display: none; position: fixed; top: 100px; left: 50%; transform: translateX(-50%); padding: 15px 30px; font-size: 18px; background: #0056b3; color: white; border: none; border-radius: 8px; cursor: pointer; z-index: 50; box-shadow: 0 4px 10px rgba(0,0,0,0.3); font-weight: bold; }}
                    
                    #toolbar {{ display: none; position: fixed; top: 100px; left: 40px; background: rgba(255,255,255,0.98); border-radius: 8px; font-size: 14px; font-weight: bold; z-index: 60; box-shadow: 0 4px 15px rgba(0,0,0,0.3); border: 2px solid #0056b3; width: 230px; overflow: hidden; }}
                    #toolbar-header {{ background: #0056b3; color: white; padding: 10px; cursor: grab; text-align: center; font-size: 15px; }}
                    #toolbar-header:active {{ cursor: grabbing; }}
                    #toolbar-body {{ padding: 15px; }}
                    .tool-btn {{ margin-top: 8px; width: 100%; padding: 10px 15px; cursor: pointer; border: 1px solid #ccc; background: #f8f9fa; border-radius: 6px; font-weight: bold; text-align: left; transition: 0.2s; font-size: 14px; }}
                    .tool-btn:hover {{ background: #e2e6ea; }}
                    .tool-btn.active {{ background: #0056b3; color: white; border-color: #004494; }}
                    
                    #measureTip {{ display: none; position: fixed; top: 50px; left: 50%; transform: translateX(-50%); background: #17a2b8; color: white; padding: 10px 20px; border-radius: 8px; font-size: 14px; font-weight: bold; z-index: 50; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }}
                </style>
            </head>
            <body>
                <div class="monitor-container" id="container">
                    
                    <div class="overlay-play" id="overlay">
                        <button class="play-btn" id="btnStart">▶ INICIAR MONITOR (Activa Sonido)</button>
                    </div>

                    <div id="measureTip">📏 Herramienta Activa: Arrastra el mouse para medir</div>

                    <div id="toolbar">
                        <div id="toolbar-header">⚙️ Panel de Herramientas</div>
                        <div id="toolbar-body">
                            <button class="tool-btn active" id="btnMeasure">📏 Regla de Medición</button>
                            <button class="tool-btn" id="btnMagnify">🔍 Zoom por Selección</button>
                        </div>
                    </div>

                    <button id="btnRepetir">🔄 Repetir Trazado</button>
                    
                    <canvas id="gridCanvas"></canvas>
                    <canvas id="signalCanvas"></canvas>
                    <canvas id="interactCanvas"></canvas>
                    
                    <div class="bpm-panel">
                        <div id="heartIcon" style="font-size:24px; color:#ff4b4b; transition: transform 0.1s;">❤️</div>
                        <div class="bpm-value" id="bpmDisplay">{bpm_real}</div>
                        <div>BPM</div>
                    </div>
                    
                    <div class="leyenda">
                        <span style="color:#000000; margin-right:10px;">■ Arriba: Original (Negro)</span><br>
                        <span style="color:#0056b3; margin-right:10px;">■ Abajo: Filtrada IA (Azul)</span><br>
                        <span style="color:rgba(0,191,255,0.8);">┇ Ventana de 200 pts</span>
                    </div>
                </div>

                <script>
                    const rawData = {json.dumps(senales_crudas)};
                    const filtData = {json.dumps(senales_filt)};
                    const peaks = {json.dumps(picos_muestras)};
                    const leads = {json.dumps(['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])};
                    const numSamples = rawData[0].length;
                    const totalSeconds = 8.0;

                    const gridCanvas = document.getElementById('gridCanvas');
                    const signalCanvas = document.getElementById('signalCanvas');
                    const interactCanvas = document.getElementById('interactCanvas');
                    const overlay = document.getElementById('overlay');
                    const btnStart = document.getElementById('btnStart');
                    const toolbar = document.getElementById('toolbar');
                    const toolbarHeader = document.getElementById('toolbar-header');
                    
                    const ctxGrid = gridCanvas.getContext('2d', {{ alpha: false }});
                    const ctxSig = signalCanvas.getContext('2d');
                    const ctxInteract = interactCanvas.getContext('2d');
                    
                    const btnRepetir = document.getElementById('btnRepetir');
                    const btnMeasure = document.getElementById('btnMeasure');
                    const btnMagnify = document.getElementById('btnMagnify');
                    const measureTip = document.getElementById('measureTip');
                    const container = document.getElementById('container');
                    const heartIcon = document.getElementById('heartIcon');

                    let width = gridCanvas.clientWidth;
                    let height = gridCanvas.clientHeight;
                    
                    gridCanvas.width = width; gridCanvas.height = height;
                    signalCanvas.width = width; signalCanvas.height = height;
                    interactCanvas.width = width; interactCanvas.height = height;

                    const numLeads = 12;
                    const trackHeight = height / numLeads; // Ahora son 200px por derivación
                    const dx = width / numSamples;

                    let allRaw = [];
                    let allFilt = [];
                    for(let i=0; i<numLeads; i++) {{
                        for(let j=0; j<numSamples; j++) {{
                            allRaw.push(Math.abs(rawData[i][j]));
                            allFilt.push(Math.abs(filtData[i][j]));
                        }}
                    }}
                    allRaw.sort((a, b) => a - b);
                    allFilt.sort((a, b) => a - b);
                    
                    let maxRawVal = allRaw[Math.floor(allRaw.length * 0.99)] || 0.001;
                    let maxFiltVal = allFilt[Math.floor(allFilt.length * 0.99)] || 0.001;

                    // 🛠️ MATEMÁTICA DE SEPARACIÓN BLINDADA
                    // Crecen exactamente un 23% hacia arriba y 23% hacia abajo de su propio centro. Imposible solaparse.
                    const yScaleRaw = (trackHeight * 0.23) / maxRawVal;
                    const yScaleFilt = (trackHeight * 0.23) / maxFiltVal;

                    // 1. DIBUJAR GRILLA MILIMETRADA
                    ctxGrid.fillStyle = '#ffffff';
                    ctxGrid.fillRect(0, 0, width, height);
                    
                    const largeSq = width / (totalSeconds * 5); 
                    const smallSq = largeSq / 5;

                    ctxGrid.lineWidth = 0.5;
                    ctxGrid.strokeStyle = '#f0d0d0'; 
                    ctxGrid.beginPath();
                    for(let x=0; x<=width; x+=smallSq) {{ ctxGrid.moveTo(x, 0); ctxGrid.lineTo(x, height); }}
                    for(let y=0; y<=height; y+=smallSq) {{ ctxGrid.moveTo(0, y); ctxGrid.lineTo(width, y); }}
                    ctxGrid.stroke();

                    ctxGrid.lineWidth = 1;
                    ctxGrid.strokeStyle = '#e0a0a0'; 
                    ctxGrid.beginPath();
                    for(let x=0; x<=width; x+=largeSq) {{ ctxGrid.moveTo(x, 0); ctxGrid.lineTo(x, height); }}
                    for(let y=0; y<=height; y+=largeSq) {{ ctxGrid.moveTo(0, y); ctxGrid.lineTo(width, y); }}
                    ctxGrid.stroke();

                    ctxGrid.font = "bold 16px Arial";
                    ctxGrid.fillStyle = "#333";
                    for(let i=0; i<numLeads; i++) {{
                        ctxGrid.fillText(leads[i], 10, i * trackHeight + 25);
                        
                        ctxGrid.beginPath();
                        ctxGrid.moveTo(0, i * trackHeight); ctxGrid.lineTo(width, i * trackHeight);
                        ctxGrid.strokeStyle = '#666'; ctxGrid.lineWidth = 2; ctxGrid.stroke();
                        
                        ctxGrid.beginPath();
                        ctxGrid.moveTo(0, i * trackHeight + trackHeight/2); ctxGrid.lineTo(width, i * trackHeight + trackHeight/2);
                        ctxGrid.strokeStyle = 'rgba(0,0,0,0.15)'; ctxGrid.lineWidth = 1; ctxGrid.stroke();
                    }}

                    ctxGrid.fillStyle = "#000";
                    ctxGrid.font = "bold 14px Arial";
                    for (let s=0; s<=totalSeconds; s++) {{
                        let xPos = (s / totalSeconds) * width;
                        ctxGrid.fillText(s + "s", xPos + 5, height - 10);
                    }}

                    for (let i=0; i<peaks.length; i++) {{
                        let startX = (peaks[i] - 100) * dx;
                        let endX = (peaks[i] + 100) * dx;
                        
                        ctxGrid.strokeStyle = 'rgba(0, 150, 255, 0.7)';
                        ctxGrid.lineWidth = 2;
                        ctxGrid.setLineDash([6, 4]);
                        ctxGrid.beginPath();
                        ctxGrid.moveTo(startX, 0); ctxGrid.lineTo(startX, height);
                        ctxGrid.moveTo(endX, 0); ctxGrid.lineTo(endX, height);
                        ctxGrid.stroke();
                        ctxGrid.setLineDash([]);
                    }}

                    // 2. LÓGICA DE AUDIO
                    let audioCtx = null;
                    function playBeep() {{
                        if(!audioCtx || audioCtx.state !== 'running') return;
                        const osc = audioCtx.createOscillator();
                        const gain = audioCtx.createGain();
                        osc.type = 'sine'; osc.frequency.value = 800;
                        gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
                        gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.1);
                        osc.connect(gain); gain.connect(audioCtx.destination);
                        osc.start(); osc.stop(audioCtx.currentTime + 0.1);
                        
                        heartIcon.style.transform = 'scale(1.4)';
                        setTimeout(() => heartIcon.style.transform = 'scale(1)', 100);
                    }}

                    // 3. LÓGICA DRAGGABLE (Panel Arrastrable)
                    let isDraggingToolbar = false;
                    let dragOffsetX = 0;
                    let dragOffsetY = 0;

                    toolbarHeader.addEventListener('mousedown', (e) => {{
                        isDraggingToolbar = true;
                        const rect = toolbar.getBoundingClientRect();
                        dragOffsetX = e.clientX - rect.left;
                        dragOffsetY = e.clientY - rect.top;
                    }});

                    document.addEventListener('mousemove', (e) => {{
                        if (!isDraggingToolbar) return;
                        toolbar.style.left = (e.clientX - dragOffsetX) + 'px';
                        toolbar.style.top = (e.clientY - dragOffsetY) + 'px';
                        toolbar.style.right = 'auto'; 
                    }});

                    document.addEventListener('mouseup', () => {{
                        isDraggingToolbar = false;
                    }});

                    // 4. ANIMACIÓN Y TRAZADO
                    let isPlaying = false;
                    let startTime = null;
                    let currentSample = 0;

                    function animate(time) {{
                        if(!isPlaying) return;
                        if(!startTime) startTime = time;
                        
                        let elapsed = time - startTime;
                        let targetSample = Math.floor((elapsed / 8000) * numSamples);
                        
                        if (targetSample >= numSamples) {{
                            targetSample = numSamples;
                            isPlaying = false;
                            btnRepetir.style.display = 'block';
                            toolbar.style.display = 'block';
                            measureTip.style.display = 'block';
                        }}

                        for(let i=0; i<numLeads; i++) {{
                            // 🛠️ POSICIÓN BLINDADA: Ancladas exactamente al 26% y 76% del riel.
                            let topY = i * trackHeight + (trackHeight * 0.26);
                            let botY = i * trackHeight + (trackHeight * 0.76);

                            ctxSig.beginPath();
                            ctxSig.strokeStyle = '#000000'; 
                            ctxSig.lineWidth = 2;
                            ctxSig.moveTo(currentSample * dx, topY - rawData[i][currentSample] * yScaleRaw);
                            for(let s = currentSample + 1; s <= targetSample && s < numSamples; s++) {{
                                ctxSig.lineTo(s * dx, topY - rawData[i][s] * yScaleRaw);
                            }}
                            ctxSig.stroke();

                            ctxSig.beginPath();
                            ctxSig.strokeStyle = '#0056b3';
                            ctxSig.lineWidth = 2;
                            ctxSig.moveTo(currentSample * dx, botY - filtData[i][currentSample] * yScaleFilt);
                            for(let s = currentSample + 1; s <= targetSample && s < numSamples; s++) {{
                                ctxSig.lineTo(s * dx, botY - filtData[i][s] * yScaleFilt);
                            }}
                            ctxSig.stroke();
                        }}

                        for (let p = 0; p < peaks.length; p++) {{
                            if (peaks[p] >= currentSample && peaks[p] < targetSample) {{
                                playBeep();
                            }}
                        }}

                        currentSample = targetSample;
                        if (isPlaying) requestAnimationFrame(animate);
                    }}

                    function iniciarTrazado() {{
                        ctxSig.clearRect(0, 0, width, height);
                        ctxInteract.clearRect(0, 0, width, height);
                        currentSample = 0;
                        startTime = null;
                        isPlaying = true;
                        btnRepetir.style.display = 'none';
                        toolbar.style.display = 'none';
                        measureTip.style.display = 'none';
                        isZoomed = false;
                        
                        if(!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                        if(audioCtx.state === 'suspended') audioCtx.resume();
                        
                        requestAnimationFrame(animate);
                    }}

                    btnStart.addEventListener('click', () => {{ overlay.style.display = 'none'; iniciarTrazado(); }});
                    btnRepetir.addEventListener('click', (e) => {{ e.stopPropagation(); iniciarTrazado(); }});

                    // 5. HERRAMIENTAS INTERACTIVAS (REGLA Y ZOOM POR SELECCIÓN)
                    let currentTool = 'measure';
                    let isInteracting = false;
                    let isZoomed = false; 
                    let startMx = 0, startMy = 0;

                    btnMeasure.addEventListener('click', () => {{
                        currentTool = 'measure';
                        btnMeasure.classList.add('active');
                        btnMagnify.classList.remove('active');
                        ctxInteract.clearRect(0, 0, width, height);
                        isZoomed = false;
                    }});

                    btnMagnify.addEventListener('click', () => {{
                        currentTool = 'magnify';
                        btnMagnify.classList.add('active');
                        btnMeasure.classList.remove('active');
                        ctxInteract.clearRect(0, 0, width, height);
                        isZoomed = false;
                    }});

                    container.addEventListener('mousedown', (e) => {{
                        if (isPlaying) return; 
                        
                        if (isZoomed) {{
                            ctxInteract.clearRect(0, 0, width, height);
                            isZoomed = false;
                            return;
                        }}

                        const rect = container.getBoundingClientRect();
                        startMx = e.clientX - rect.left;
                        startMy = e.clientY - rect.top;
                        isInteracting = true;
                    }});

                    container.addEventListener('mousemove', (e) => {{
                        if (isPlaying || !isInteracting || isZoomed) return;
                        
                        const rect = container.getBoundingClientRect();
                        let currMx = e.clientX - rect.left;
                        let currMy = e.clientY - rect.top;

                        ctxInteract.clearRect(0, 0, width, height);

                        if (currentTool === 'measure') {{
                            ctxInteract.beginPath();
                            ctxInteract.strokeStyle = '#E0115F';
                            ctxInteract.lineWidth = 2;
                            ctxInteract.moveTo(startMx, startMy);
                            ctxInteract.lineTo(currMx, currMy);
                            ctxInteract.stroke();

                            let dt = Math.abs(currMx - startMx) / width * totalSeconds;
                            let dv = Math.abs(currMy - startMy) / yScaleFilt; 

                            let textoMedida = "Delta t: " + dt.toFixed(3) + "s  |  Delta V: " + dv.toFixed(2) + "mV";
                            
                            ctxInteract.font = "bold 14px Arial";
                            let textW = ctxInteract.measureText(textoMedida).width;
                            
                            ctxInteract.fillStyle = "rgba(255, 255, 255, 0.9)";
                            ctxInteract.fillRect(currMx + 10, currMy - 25, textW + 20, 25);
                            ctxInteract.fillStyle = "#E0115F";
                            ctxInteract.fillText(textoMedida, currMx + 20, currMy - 8);
                            
                        }} else if (currentTool === 'magnify') {{
                            ctxInteract.fillStyle = 'rgba(0, 150, 255, 0.15)';
                            ctxInteract.strokeStyle = '#0056b3';
                            ctxInteract.lineWidth = 2;
                            ctxInteract.setLineDash([5, 5]);
                            
                            let w = currMx - startMx;
                            let h = currMy - startMy;
                            
                            ctxInteract.fillRect(startMx, startMy, w, h);
                            ctxInteract.strokeRect(startMx, startMy, w, h);
                            ctxInteract.setLineDash([]);
                        }}
                    }});

                    container.addEventListener('mouseup', (e) => {{
                        if (!isInteracting || isZoomed) return;
                        isInteracting = false;
                        
                        if (currentTool === 'magnify') {{
                            const rect = container.getBoundingClientRect();
                            let endMx = e.clientX - rect.left;
                            let endMy = e.clientY - rect.top;
                            
                            let sw = Math.abs(endMx - startMx);
                            let sh = Math.abs(endMy - startMy);
                            let sx = Math.min(startMx, endMx);
                            let sy = Math.min(startMy, endMy);
                            
                            ctxInteract.clearRect(0, 0, width, height);
                            
                            if (sw > 30 && sh > 30) {{
                                let zoomFactor = 2.8; 
                                
                                if (sw * zoomFactor > width * 0.9) zoomFactor = (width * 0.9) / sw;
                                
                                let tw = sw * zoomFactor;
                                let th = sh * zoomFactor;
                                
                                let tx = sx - (tw - sw)/2;
                                let ty = sy - (th - sh)/2;
                                
                                if (tx < 10) tx = 10;
                                if (ty < 10) ty = 10;
                                if (tx + tw > width - 10) tx = width - tw - 10;
                                if (ty + th > height - 10) ty = height - th - 10;
                                
                                ctxInteract.save();
                                ctxInteract.shadowColor = "rgba(0,0,0,0.6)";
                                ctxInteract.shadowBlur = 20;
                                
                                ctxInteract.fillStyle = "white";
                                ctxInteract.fillRect(tx, ty, tw, th);
                                ctxInteract.shadowBlur = 0; 
                                
                                ctxInteract.drawImage(gridCanvas, sx, sy, sw, sh, tx, ty, tw, th);
                                ctxInteract.drawImage(signalCanvas, sx, sy, sw, sh, tx, ty, tw, th);
                                
                                ctxInteract.strokeStyle = "#0056b3";
                                ctxInteract.lineWidth = 4;
                                ctxInteract.strokeRect(tx, ty, tw, th);
                                
                                ctxInteract.fillStyle = "rgba(0, 86, 179, 0.95)";
                                ctxInteract.fillRect(tx, ty, tw, 35);
                                ctxInteract.fillStyle = "white";
                                ctxInteract.font = "bold 15px Arial";
                                ctxInteract.fillText("🔍 Vista Ampliada (Haz clic fuera para cerrar)", tx + 15, ty + 23);
                                
                                ctxInteract.restore();
                                isZoomed = true; 
                            }}
                        }}
                    }});
                </script>
            </body>
            </html>
            """
            # 🛠️ AJUSTADO A 2450px PARA ACOMODAR EL NUEVO ALTO
            components.html(html_code, height=2450)

            # =====================================================================
            # MÉTRICAS GLOBALES Y TABLA
            # =====================================================================
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.markdown(f"<div class='metric-card'><h4>Veredicto IA</h4><h3 style='color:#0056b3;'>{data['diagnostico_clinico_final']}</h3></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'><h4>Probabilidad AFIB</h4><h2>{data['score_afib']*100:.1f}%</h2></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-card'><h4>Promedio LBBB</h4><h2>{data['score_lbbb_promedio']*100:.1f}%</h2></div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='metric-card'><h4>Promedio RBBB</h4><h2>{data['score_rbbb_promedio']*100:.1f}%</h2></div>", unsafe_allow_html=True)
            c5.markdown(f"<div class='metric-card'><h4>Promedio PVC</h4><h2>{data['score_pvc_promedio']*100:.1f}%</h2></div>", unsafe_allow_html=True)
           
            st.markdown("<br><h2>🔬 Reporte Clínico: Morfología Latido a Latido</h2><hr>", unsafe_allow_html=True)
            
            for lat in data.get("analisis_latido_a_latido", []):
                with st.container():
                    col_info, col_red, col_faiss = st.columns([1, 2, 4])
                    with col_info:
                        st.markdown(f"<h1 style='color:#0056b3; margin-bottom:0;'>#{lat['num_latido']}</h1>", unsafe_allow_html=True)
                        st.caption(f"⏱️ {lat['tiempo_segundos']} s")
                    with col_red:
                        st.markdown("**Predicción Red Neuronal:**")
                        st.write(f"🩸 **LBBB:** {lat['predicciones_morfologia']['LBBB']*100:.2f}%")
                        st.write(f"🩸 **RBBB:** {lat['predicciones_morfologia']['RBBB']*100:.2f}%")
                        st.write(f"🩸 **PVC:** {lat['predicciones_morfologia']['PVC']*100:.2f}%")
                    with col_faiss:
                        html_vecinos = ""
                        for idx, vec in enumerate(lat["shazam_vecinos_latido"]):
                            clase = "top1" if idx == 0 else ""
                            html_vecinos += f"""
                            <div class="vecino-box {clase}">
                                <span class="badge">{vec['similitud']}% Similitud</span>
                                <b>#{idx+1} {vec['diagnostico_base']}</b><br>
                                <small style="color:#666;">ID: {vec['id_caso']}</small>
                            </div>
                            """
                        st.markdown(html_vecinos, unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.error(f"Error en el servidor: {res.status_code}")