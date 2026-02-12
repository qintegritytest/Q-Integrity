'''
Q-INTEGRITY – APP COMPLETA (PC)
✅ 1) DENSIDADES – Pantalla 1 (Ingreso/Editar/Eliminar/Export)
✅ 2) DENSIDADES – Pantalla 2 (KPIs/Gráficos/Eliminar/Export)
✅ 3) CONTROL PIE m² – Ingreso/Editar/Eliminar/Export
✅ 4) CONTROL PIE m² – KPIs/Gráficos/Export
✅ 5) CONTROL PIE m³ – Ingreso/Editar/Eliminar/Export
✅ 6) CONTROL PIE m³ – KPIs/Gráficos/Export
✅ TODO en MENÚ LATERAL (sin tabs arriba)
✅ Sin rutas fijas tipo C:/... (solo archivos locales al lado del app.py)
PEGAR COMPLETO EN app.py
'''
# ======================= INICIO IMPORTS ==================================
import os
import io
import uuid
import time
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from api_ia import ApiIa
import fitz  # PyMuPDF
import cv2
from rapidocr_onnxruntime import RapidOCR

################################# OPTIMIZACIONES MOTOR OCR ################################
# Instancia OCR en cache para optimizacion
@st.cache_resource
def get_rapidocr_engine():
    return RapidOCR()


def _is_mostly_blank(gray: np.ndarray) -> bool:
    """
    Detecta páginas casi blancas para saltarlas y ahorrar tiempo.
    Ajusta el umbral si tus escaneos son amarillentos/oscuros.
    """
    return (gray > 245).mean() > 0.995


def _preprocess(gray: np.ndarray, max_w: int = 1200) -> np.ndarray:
    """
    Optimizado para scans claros (carta/oficio):
    - Binariza (Otsu)
    - Auto-crop de márgenes (reduce pixeles => acelera MUCHO)
    - Downscale opcional si queda muy grande
    """
    # Otsu: fondo blanco, texto negro (rápido y efectivo en impresos)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Auto-crop márgenes ---
    inv = 255 - th  # texto/blobs quedan >0
    coords = cv2.findNonZero(inv)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)

        pad = 10  # margen extra para no cortar letras
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(th.shape[1], x + w + pad)
        y1 = min(th.shape[0], y + h + pad)

        # Evita recortes absurdos por ruido
        if (x1 - x0) > 200 and (y1 - y0) > 200:
            th = th[y0:y1, x0:x1]

    # --- Downscale por ancho máximo (a 90 DPI carta/oficio normalmente no aplica, pero tras crop puede ayudar) ---
    h, w = th.shape
    if w > max_w:
        scale = max_w / w
        th = cv2.resize(th, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)

    return th


def ocr_pdf_rapidocr(abs_path: str, cache_path: str, diag: List[str], dpi: int = 90) -> Tuple[str, str]:
    """
    OCR de PDF escaneado usando PyMuPDF + RapidOCR (ONNXRuntime).
    - Sin poppler/tesseract
    - Optimizado para scans claros (carta/oficio)
    """
    try:
        diag.append("Usando OCR: PyMuPDF + RapidOCR (onnxruntime, pip-only)")
        diag.append(f"Configuración: dpi={dpi}, colorspace=GRAY, auto-crop=ON, blank-skip=ON")

        ocr_engine = get_rapidocr_engine()

        doc = fitz.open(abs_path)
        diag.append(f"PDF abierto: {doc.page_count} páginas")

        prog = st.progress(0, text="OCR en progreso...")
        info = st.empty()

        ocr_pages: List[str] = []
        t_all0 = time.perf_counter()

        for idx in range(doc.page_count):
            t0 = time.perf_counter()
            try:
                page = doc.load_page(idx)

                # Render liviano: gris, sin alpha
                pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY, alpha=False)
                gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

                # Saltar páginas casi vacías
                if _is_mostly_blank(gray):
                    ocr_pages.append("")
                    diag.append(f"Página {idx + 1}: saltada (casi en blanco)")
                    prog.progress((idx + 1) / doc.page_count, text=f"OCR {idx+1}/{doc.page_count}")
                    continue

                # Preproceso (binariza + recorta márgenes + downscale opcional)
                img = _preprocess(gray, max_w=1200)

                # RapidOCR suele esperar BGR (OpenCV)
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                # Ejecutar OCR
                out = ocr_engine(bgr)

                # RapidOCR puede devolver (result, elapse) o solo result según versión
                result = out[0] if isinstance(out, tuple) and len(out) >= 1 else out

                # result: lista de [box, text, score]
                if result:
                    lines = [r[1] for r in result if len(r) >= 2 and r[1]]
                    ptext = "\n".join(lines).strip()
                else:
                    ptext = ""

                ocr_pages.append(ptext)

                t1 = time.perf_counter()
                diag.append(f"Página {idx + 1}: {len(ptext)} chars | {t1 - t0:.2f}s")

            except Exception as e:
                diag.append(f"Página {idx + 1}: error RapidOCR: {e}")
                ocr_pages.append("")

            # Progreso UI liviano (sin toast)
            prog.progress((idx + 1) / doc.page_count, text=f"OCR {idx+1}/{doc.page_count}")
            if (idx + 1) % 5 == 0 or (idx + 1) == doc.page_count:
                info.caption(f"Procesadas {idx+1}/{doc.page_count} páginas")

        doc.close()
        info.empty()

        text = "\n".join(ocr_pages).strip()

        t_all1 = time.perf_counter()
        diag.append(f"Tiempo total OCR: {t_all1 - t_all0:.2f}s")

        if text:
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(text)
            prog.progress(1.0, text="OCR listo ✅")
            return text, "\n".join(diag)

        prog.empty()
        return "", "\n".join(diag)

    except Exception as e:
        diag.append(f"Fallback RapidOCR falló: {e}")
        return "", "\n".join(diag)

################################# FIN OPTIMIZACIONES MOTOR OCR ################################
# IMPORT OPENPYXL OPCIONAL
try:
    import openpyxl  # noqa: F401
    HAS_OPENPYXL = True
except Exception:
    HAS_OPENPYXL = False

# IMPORT GROQ SEGURO PARA PANTALLA 8, ANTES DE USO
# Cerca de la línea 190, donde está tu bloque de inicialización


# En la parte superior de app.py, después de los imports
from api_ia import ApiIa 
HAS_GROQ = False

# No definas HAS_GROQ = False aquí si luego usas el de session_state, 
# puede crear confusión de "ámbito" (scope).
# --- INICIALIZACIÓN ROBUSTA DE IA ---
if "HAS_GROQ" not in st.session_state:
    st.session_state["HAS_GROQ"] = False

if "API_IA" not in st.session_state or st.session_state["API_IA"] is None:
    try:
        from groq import Groq
        api_key = st.secrets.get("groq_api_key")
        
        if api_key:
            client_groq = Groq(api_key=api_key)
            # Asegúrate que ApiIa esté importado arriba: from api_ia import ApiIa
            st.session_state["API_IA"] = ApiIa(client_groq)
            st.session_state["HAS_GROQ"] = True
        else:
            st.warning("⚠️ No se encontró la clave 'groq_api_key' en st.secrets.")
    except Exception as e:
        st.session_state["HAS_GROQ"] = False
        st.error(f"❌ Error al conectar con Groq: {e}")

# Esto crea un alias para que tus funciones viejas que usan HAS_GROQ no mueran
HAS_GROQ = st.session_state["HAS_GROQ"]


# ======================== FIN IMPORTS ===========================================
# --- 🕵️ BLOQUE DE DIAGNÓSTICO (BORRAR AL FINAL) ---
def verificar_groq():
    import streamlit as st
    try:
        st.markdown("### 🕵️ Diagnóstico de Conexión")
        
        # 1. Probar si la librería existe
        import groq
        st.success("1. ✅ Librería `groq` instalada correctamente.")
        
        # 2. Probar si lee los secretos
        if "groq_api_key" in st.secrets:
            clave = st.secrets["groq_api_key"]
            # Mostrar solo los primeros 4 caracteres para verificar
            st.success(f"2. ✅ Secreto encontrado. Empieza con: `{clave[:4]}...`")
            
            # 3. Probar conexión real
            try:
                client_test = groq.Groq(api_key=clave)
                st.success("3. ✅ Cliente Groq inicializado sin errores.")
            except Exception as e:
                st.error(f"3. ❌ Error al iniciar cliente Groq: {e}")
                
        else:
            st.error(f"2. ❌ NO se encuentra la clave 'groq_api_key' en los secretos. Claves disponibles: {list(st.secrets.keys())}")
            
    except ImportError:
        st.error("1. ❌ La librería `groq` NO está instalada (revisa requirements.txt).")
    except Exception as e:
        st.error(f"❌ Error general en diagnóstico: {e}")

    st.divider()
# ----------------------------------------------------

# ======================== INICIO DE UTILIDADES  =================================
def _alt_csv_path(path: str) -> str:
    if not path:
        return path
    p = str(path)
    if p.lower().endswith(".xlsx"):
        return p[:-5] + ".csv"
    if p.lower().endswith(".xls"):
        return p[:-4] + ".csv"
    return p + ".csv"

def safe_read_excel(path: str, columns=None, sheet_name=0) -> pd.DataFrame:
    """Lee una tabla sin botar la app.
    - Si hay openpyxl: lee .xlsx normal.
    - Si NO hay openpyxl: intenta leer el .csv alternativo (mismo nombre).
    - Si nada existe: devuelve DF vacío con columnas esperadas.
    """
    if columns is None:
        columns = []
    if not path:
        return pd.DataFrame(columns=columns)

    # Preferir CSV alternativo si no hay openpyxl o si se pide csv directamente
    if (not HAS_OPENPYXL) or str(path).lower().endswith(".csv"):
        csv_path = path if str(path).lower().endswith(".csv") else _alt_csv_path(path)
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, dtype=str, encoding="utf-8", sep=",")
            except Exception:
                df = pd.read_csv(csv_path, dtype=str, encoding="latin-1", sep=",")
            return df if (not columns) else df.reindex(columns=columns)
        return pd.DataFrame(columns=columns)

    # openpyxl disponible
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns)

    try:
        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        # si sheet_name es lista/dict, pd.read_excel puede devolver dict
        if isinstance(df, dict):
            # tomar la primera hoja
            df = next(iter(df.values())) if df else pd.DataFrame()
        return df if (not columns) else df.reindex(columns=columns)
    except Exception:
        # último recurso: intentar CSV alternativo
        csv_path = _alt_csv_path(path)
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, dtype=str, encoding="utf-8", sep=",")
            except Exception:
                df = pd.read_csv(csv_path, dtype=str, encoding="latin-1", sep=",")
            return df if (not columns) else df.reindex(columns=columns)
        return pd.DataFrame(columns=columns)


def safe_write_excel_malo(df: pd.DataFrame, path: str) -> str:
    """Escribe una tabla sin botar la app.
    Retorna la ruta efectivamente escrita (xlsx o csv alternativo).
    """
    if not path:
        return ""

    out = df.copy()

    # Si no hay openpyxl (o se pide csv), escribir CSV alternativo
    if (not HAS_OPENPYXL) or str(path).lower().endswith(".csv"):
        csv_path = path if str(path).lower().endswith(".csv") else _alt_csv_path(path)
        out.to_csv(csv_path, index=False, encoding="utf-8")
        return csv_path

    # Escritura atómica: escribir a temp y luego reemplazar (evita archivos corruptos)
    tmp_path = str(path) + ".tmp"
    try:
        with pd.ExcelWriter(tmp_path, engine="openpyxl") as w:
            (out if not out.empty else pd.DataFrame({"INFO": ["Sin datos"]})).to_excel(w, index=False, sheet_name="Datos")
        os.replace(tmp_path, path)
        return path
    except PermissionError:
        # archivo abierto/bloqueado: degradar a CSV alternativo
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        csv_path = _alt_csv_path(path)
        out.to_csv(csv_path, index=False, encoding="utf-8")
        return csv_path
    except Exception:
        # cualquier otro error: degradar a CSV alternativo
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        csv_path = _alt_csv_path(path)
        out.to_csv(csv_path, index=False, encoding="utf-8")
        return csv_path


def safe_write_excel(df: pd.DataFrame, path: str) -> str:
    if not path:
        return ""

    out = df.copy()
    p = Path(path)

    # CSV si no hay openpyxl o si el path pide csv
    if (not HAS_OPENPYXL) or p.suffix.lower() == ".csv":
        csv_path = str(p) if p.suffix.lower() == ".csv" else _alt_csv_path(str(p))
        out.to_csv(csv_path, index=False, encoding="utf-8")
        return csv_path

    # temp conservando extensión .xlsx/.xlsm
    tmp_p = p.with_name(p.stem + ".__tmp" + p.suffix)  # ej: reporte.__tmp.xlsx

    try:
        with pd.ExcelWriter(tmp_p, engine="openpyxl") as w:
            (out if not out.empty else pd.DataFrame({"INFO": ["Sin datos"]})).to_excel(
                w, index=False, sheet_name="Datos"
            )
        os.replace(tmp_p, p)  # reemplazo atómico
        return str(p)

    except PermissionError:
        try:
            if tmp_p.exists():
                tmp_p.unlink()
        except Exception:
            pass
        csv_path = _alt_csv_path(str(p))
        out.to_csv(csv_path, index=False, encoding="utf-8")
        return csv_path

    except Exception as e:
        try:
            if tmp_p.exists():
                tmp_p.unlink()
        except Exception:
            pass
        csv_path = _alt_csv_path(str(p))
        out.to_csv(csv_path, index=False, encoding="utf-8")
        return csv_path

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception as e:
    HAS_MPL = False
    import streamlit as st
    st.error("Matplotlib no está disponible. Instálalo en el mismo entorno donde corres Streamlit:  pip install matplotlib")
    st.caption(f"Detalle: {e}")
    st.stop()
# ======================== FIN DE UTILIDADES  =================================

# ======================== INICIO DE CONFIGURACION Y RESETS SEGUROS =================================
st.set_page_config(page_title="Q-INTEGRITY", layout="wide")

if st.session_state.get("DEN_RESET_REQUESTED"):
    _clear_last = bool(st.session_state.get("DEN_RESET_CLEAR_LAST", True))
    for _k in [
        "den_fecha_ctrl","den_cod_proy","den_proyecto","den_n_reg","den_n_ctrl","den_n_acta",
        "den_sector_txt","den_tramo_txt","den_frente","den_capa_txt","den_esp_txt",
        "den_dm_ini_txt","den_dm_ter_txt","den_dm_ctrl_txt","den_coord_n_txt","den_coord_e_txt","den_cota_txt",
        "den_operador","den_met_sel","den_met_otro","den_prof_txt","den_obs",
        "den_dh_num","den_h_num","den_hopt_num","den_dmcs_num",
        "den_docid_eett","DEN_DOC_EETT_IDX",
        "DEN_EDIT_ID","DEN_EDIT_ROWKEY","DEN_EDIT_PICK",
        "DEN_LAST_SUBMIT_TS","DEN_FORCE_RECALC_TS",
    ]:
        st.session_state.pop(_k, None)
    if _clear_last:
        st.session_state.pop("DEN_LAST_SAVED", None)
    st.session_state.pop("DEN_RESET_REQUESTED", None)
    st.session_state.pop("DEN_RESET_CLEAR_LAST", None)

# Archivos locales (MISMA carpeta del app.py)
DATA_FILE_DEN = "qintegrity_densidades.xlsx"
CONFIG_FILE_DEN = "qintegrity_config.xlsx"
TEMPLATE_FILE_DEN = "QI-DEN-PLT_FINAL_CORREGIDO_v12.xlsx"  # opcional (si existe se usa)

DATA_FILE_PIE_M2 = "qintegrity_control_pie_m2.xlsx"
DATA_FILE_PIE_M3 = "qintegrity_control_pie_m3.xlsx"

# ✅ Índice de Biblioteca EETT (cuando exista Pantalla 7)
EETT_INDEX_FILE = "qintegrity_biblioteca.xlsx"

# Indice de checklist aprobados (Pantalla 8)
DATA_FILE_REVISIONES = Path("qintegrity_revisiones.txt")

FIG_W = 5.8
FIG_H = 3.2

DEFAULT_TOL_HUM_OPT = 2.0
DEFAULT_OBS_BAND = 2.0
DEFAULT_KEEP_VALUES = False

ANTI_DOUBLECLICK_SECONDS = 1.2
ANTI_DUPLICATE_WINDOW_SECONDS = 5

# ======================== FIN DE CONFIGURACION Y RESETS SEGUROS =================================

# ======================== INICIO DE ESTILOS PERSONALIZADOS =================================
st.markdown(
    """
<style>
.stApp { background:#f4f6fb; }
.qi-topbar{ background: #0f2f4f; padding: 10px 14px; border-radius: 14px; margin-bottom: 12px; }
.qi-title{ color:#ffffff; font-size:22px; font-weight:900; margin:0; line-height:1.1; }
.qi-subtitle{ color:#cfe0ee; font-size:13px; margin:0; }

.qi-card{ background:#ffffff; border:1px solid #c7d3e4; border-radius:14px; padding:12px 12px;
  box-shadow: 0 1px 0 rgba(0,0,0,0.02); }
.hr { height:1px; background:#d6deea; margin: 12px 0; }

.qi-section{ background:#ffffff; border:1px solid #c7d3e4; border-radius:14px; padding:12px 12px;
  box-shadow: 0 1px 0 rgba(0,0,0,0.02); margin-bottom: 10px; }
.qi-h3{ font-size:16px; font-weight:900; margin:0 0 8px 0; color:#0f172a; }

div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, div[data-baseweb="textarea"] > div, div[data-baseweb="datepicker"] > div{
  background:#eaf1fb !important; border:1px solid #7fa0d2 !important; border-radius:12px !important;
}
div[data-baseweb="input"] input{ color:#0f172a !important; font-weight:800 !important; }
div[data-baseweb="textarea"] textarea{ color:#0f172a !important; font-weight:800 !important; }
div[data-baseweb="select"] span{ color:#0f172a !important; font-weight:800 !important; }
label { font-weight: 900 !important; color:#0f172a !important; }

div[data-testid="stDataFrame"] div[role="grid"]{ border: 2px solid #aabbd6 !important; border-radius: 12px !important; }
div[data-testid="stDataFrame"] div[role="columnheader"]{ background: #dfe8f7 !important; font-weight: 900 !important; color:#0f172a !important; }

.qi-chip{ display:inline-block; padding:4px 10px; border-radius:999px; font-weight:900; font-size:12px; margin-right:8px; }
.qi-green{ background:#e7f6ea; color:#1b5e20; border:1px solid #bfe8c6; }
.qi-amber{ background:#fff4db; color:#7a4f00; border:1px solid #ffd68a; }
.qi-red{ background:#fde7ea; color:#8a1c1c; border:1px solid #f6b9c1; }
.qi-muted{ color:#475569; }

button[kind="primary"] { border-radius: 12px !important; font-weight: 900 !important; }
button { border-radius: 12px !important; font-weight: 800 !important; color:black important!;}

/* Number input del sidebar */
section[data-testid="stSidebar"] button[data-testid="stNumberInputStepDown"],
section[data-testid="stSidebar"] button[data-testid="stNumberInputStepUp"]{
    background-color: #262730 !important;
    color: #fafafa !important;
    border-radius: 6px !important;
    border: 1px solid #444 !important;
    padding: 0.4em !important;
}

/* Number input de densidades */
section[data-testid="stMain"] button[data-testid="stNumberInputStepDown"],
section[data-testid="stMain"] button[data-testid="stNumberInputStepUp"]{
    background-color:white !important;
    color: black !important;
    border-radius: 6px !important;
    border: 1px solid #444 !important;
    padding: 0.4em !important;
}

/* Botones del navbar */
section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] div[data-testid="stMarkdownContainer"] p{
    color: black !important;
}

/* Seccion dropzone */
section[data-testid="stFileUploaderDropzone"]{
    background-color:#e6e6e6 !important;
}

div[data-testid="stToastContainer"]{
    top: auto !important;
    bottom: 20px !important;
    right: auto !important;
    left: 100px !important;
}

section[data-testid="stSidebar"]{ background:#0f2f4f !important; }
section[data-testid="stSidebar"] *{ color:#e5e7eb !important; }
section[data-testid="stSidebar"] label{ color:#e5e7eb !important; }
section[data-testid="stSidebar"] .qi-card, section[data-testid="stSidebar"] .qi-card *{ color:#0f172a !important; }
section[data-testid="stSidebar"] .stNumberInput input, section[data-testid="stSidebar"] .stTextInput input, section[data-testid="stSidebar"] .stTextArea textarea{ color:#0f172a !important; }
</style>
""",
    unsafe_allow_html=True,
)

# HEADER
colA, colB = st.columns([1, 12])
with colA:
    st.markdown("## 🛡️")
with colB:
    st.markdown(
        """
    <div class="qi-topbar">
        <p class="qi-title">Q-INTEGRITY</p>
        <p class="qi-subtitle">Densidades + Control PIE (m² / m³) · App PC</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ======================== FIN DE ESTILOS PERSONALIZADOS =================================

# ======================== INICIO DE HELPERS GENERALES =================================

def _safe_uuid() -> str:
    return str(uuid.uuid4())


def kpi_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="qi-card">
            <div style="color:#64748b;font-size:0.90rem;font-weight:900">{label}</div>
            <div style="color:#0f172a;font-size:2.0rem;font-weight:900;margin-top:4px">{value}</div>
            <div style="color:#475569;font-size:0.95rem;margin-top:2px">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def parse_int(txt: str) -> Optional[int]:
    if txt is None:
        return None
    s = str(txt).strip().replace(",", ".")
    if s == "":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def parse_float_loose(txt: str) -> Optional[float]:
    if txt is None:
        return None
    s = str(txt).strip().replace(",", ".")
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def is_invalid_number_if_filled(label: str, raw: str) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "":
        return None
    try:
        float(str(raw).replace(",", "."))
        return None
    except Exception:
        return f"⚠️ {label}: debe ser NUMÉRICO (no letras)."


def export_excel_bytes(df_data: pd.DataFrame, df_kpi: pd.DataFrame) -> bytes:
    """Exporta Datos + KPIs.
    - No revienta si df_data/df_kpi vienen None.
    - Si falta openpyxl, exporta CSV concatenado (degradación controlada).
    """
    if df_data is None:
        df_data = pd.DataFrame({"INFO":["Sin datos (df_data=None)"]})
    if df_kpi is None:
        df_kpi = pd.DataFrame({"INFO":["Sin KPIs (df_kpi=None)"]})
    if not HAS_OPENPYXL:
        # Degradación: un solo CSV (no Excel) para no botar la app
        out = io.StringIO()
        out.write("# DATOS\n")
        df_data.to_csv(out, index=False)
        out.write("\n# KPIs\n")
        df_kpi.to_csv(out, index=False)
        return out.getvalue().encode("utf-8")
    out = io.BytesIO()
    try:
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            # Siempre escribir al menos 1 hoja visible
            (df_data if not df_data.empty else pd.DataFrame({"INFO":["Sin datos"]})).to_excel(
                writer, index=False, sheet_name="Datos"
            )
            (df_kpi if not df_kpi.empty else pd.DataFrame({"INFO":["Sin KPIs"]})).to_excel(
                writer, index=False, sheet_name="KPIs"
            )
        return out.getvalue()

    except Exception:
        #Último recurso: CSV (evita 'At least one sheet must be visible')
        out2 = io.StringIO()
        out2.write("# DATOS\n")
        df_data.to_csv(out2, index=False)
        out2.write("\n# KPIs\n")
        df_kpi.to_csv(out2, index=False)
        return out2.getvalue().encode("utf-8")

def clear_widget_key(key: str):
    try:
        if key in st.session_state:
            del st.session_state[key]
    except Exception:
        pass

def safe_date_bounds(series_dt: pd.Series) -> Tuple[date, date]:
    try:
        s = pd.to_datetime(series_dt, errors="coerce").dropna()
        if s.empty:
            return date.today(), date.today()
        return s.dt.date.min(), s.dt.date.max()
    except Exception:
        return date.today(), date.today()


# ---------------------------------------------------------
# ✅ HELPERS EETT (selector o texto, sin romper si no existe archivo)
# ---------------------------------------------------------
def _load_eett_index_df() -> pd.DataFrame:
    if not os.path.exists(EETT_INDEX_FILE):
        return pd.DataFrame()
    try:
        df = safe_read_excel(EETT_INDEX_FILE)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def get_eett_options() -> Tuple[List[str], List[str]]:
    """Devuelve SIEMPRE (values, labels) para el selector de Documento Técnico (EETT).

    - values: lista de DocID (values[0] = "")
    - labels: lista de etiquetas visibles (labels[0] = "— Sin documento asociado —")

    Nunca retorna None (evita: TypeError cannot unpack non-iterable NoneType).
    """
    values: List[str] = [""]
    labels: List[str] = ["— Sin documento asociado —"]

    # Si no existe biblioteca, devolvemos opción vacía
    if not os.path.exists(EETT_INDEX_FILE):
        return values, labels

    # Cargar biblioteca (tolerante a fallas)
    try:
        df = pd.read_excel(EETT_INDEX_FILE)
    except Exception:
        return values, labels

    if df is None or df.empty:
        return values, labels

    # Columnas tolerantes
    col_docid = "DocID" if "DocID" in df.columns else ("ID" if "ID" in df.columns else ("DocID_EETT" if "DocID_EETT" in df.columns else None))
    col_nombre = "Nombre_Original" if "Nombre_Original" in df.columns else ("Nombre" if "Nombre" in df.columns else ("Documento" if "Documento" in df.columns else None))
    col_rev = "REV" if "REV" in df.columns else ("Rev" if "Rev" in df.columns else None)
    col_ext = "Ext" if "Ext" in df.columns else ("Extension" if "Extension" in df.columns else None)
    col_estado = "Estado" if "Estado" in df.columns else None

    if not col_docid:
        return values, labels

    # Filtrar vigentes/activos si existe columna Estado
    try:
        if col_estado:
            est = df[col_estado].astype(str).str.strip().str.lower()
            df = df[est.isin(["vigente", "activo", "activa", "ok", "si", "sí", "true", "1"])].copy()
            if df.empty:
                return values, labels
    except Exception:
        pass

    # Normalizar y ordenar suave por nombre si existe
    try:
        df[col_docid] = df[col_docid].astype(str).str.strip()
        df = df[df[col_docid] != ""].copy()
        if col_nombre and col_nombre in df.columns:
            df[col_nombre] = df[col_nombre].astype(str)
            df = df.sort_values(col_nombre, ascending=True)
    except Exception:
        pass

    for _, r in df.iterrows():
        docid = str(r.get(col_docid, "")).strip()
        if not docid or docid.lower() == "nan":
            continue

        nombre = str(r.get(col_nombre, "")).strip() if col_nombre else ""
        rev = str(r.get(col_rev, "")).strip() if col_rev else ""
        ext = str(r.get(col_ext, "")).strip() if col_ext else ""

        parts = []
        parts.append(nombre if (nombre and nombre.lower() != "nan") else docid)
        if rev and rev.lower() != "nan":
            parts.append(f"REV:{rev}")
        if ext and ext.lower() != "nan":
            parts.append(f"Ext:{ext}")

        label = " · ".join(parts) if parts else docid

        values.append(docid)
        labels.append(label)

    return values, labels


def eett_doc_selector(module_key: str, state_key: str, label: str = "Documento Técnico (EETT)") -> str:
    """Selector de EETT por módulo.
    Importante: NO escribir st.session_state[state_key] manualmente después de instanciar el widget.
    El widget con key=state_key es el dueño del valor (evita StreamlitAPIException).
    """
    values, labels = get_eett_options()
    has_catalog = (len(values) > 1)

    cur = str(st.session_state.get(state_key, "") or "").strip()

    if has_catalog:
        # calcular índice inicial
        try:
            default_idx = values.index(cur) if cur in values else 0
        except Exception:
            default_idx = 0

        # map value -> label
        label_map = {v: labels[i] for i, v in enumerate(values)}

        selected = st.selectbox(
            label,
            options=values,
            index=default_idx,
            format_func=lambda v: label_map.get(v, str(v)),
            key=state_key,
        )
        return str(selected or "").strip()

    # fallback sin catálogo: input libre (también con key=state_key)
    val = st.text_input("DocID EETT", value=cur, key=state_key).strip()
    return val


def den_docid_eett_input() -> str:
    """
    ÚNICA función que debes usar en Pantalla 1 (Densidades):
    - Si existe biblioteca => selectbox
    - Si no existe => text_input
    Retorna DocID_EETT (string; "" si ninguno).
    """
    values, labels = get_eett_options()
    has_catalog = (len(values) > 1)

    # valor actual (al editar viene precargado por load_record_into_form_den)
    cur = str(st.session_state.get("den_docid_eett", "") or "").strip()

    if has_catalog:
        # setear índice por defecto antes del widget
        if "DEN_DOC_EETT_IDX" not in st.session_state:
            st.session_state["DEN_DOC_EETT_IDX"] = values.index(cur) if cur in values else 0

        idx = st.selectbox(
            "Documento Técnico (EETT)",
            options=list(range(len(values))),
            format_func=lambda i: labels[i],
            key="DEN_DOC_EETT_IDX",
        )
        val = values[int(idx)] if idx is not None else ""
        st.session_state["den_docid_eett"] = val
        return val

    # fallback sin biblioteca
    val = st.text_input("DocID EETT", value=cur, key="den_docid_eett").strip()
    st.session_state["den_docid_eett"] = val
    return val

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

# ======================== FIN DE HELPERS GENERALES =================================


# =========================================================
# =====================  DENSIDADES  ======================
# =========================================================
COLUMNS_DEN = [
    "RowKey",
    "ID_Registro",
    "Codigo_Proyecto",
    "Proyecto",
    "N_Registro",
    "N_Control",
    "N_Acta",
    "Fecha_control",
    "Sector_Zona",
    "Tramo",
    "Frente_Tramo",
    "Capa_N",
    "Espesor_capa_cm",
    "Dm_inicio",
    "Dm_termino",
    "Dm_Control",
    "Coordenada_Norte",
    "Coordenada_Este",
    "Cota",
    "Operador",
    "Metodo",
    "Profundidad_cm",
    "Densidad_Humeda_gcm3",
    "Humedad_medida_pct",
    "Humedad_Optima_pct",
    "Delta_Humedad_pct",
    "Ventana_Humedad",
    "DocID_EETT",
    "Densidad_Seca_gcm3",
    "DMCS_Proctor_gcm3",
    "pct_Compactacion",
    "Umbral_Cumple_pct",
    "Umbral_Observado_pct",
    "Estado_QAQC",
    "Observacion",
    "Timestamp",
]


def ensure_data_file_den(path: str) -> None:
    if not os.path.exists(path):
        safe_write_excel(pd.DataFrame(columns=COLUMNS_DEN), path)


def ensure_config_file_den(path: str) -> None:
    if os.path.exists(path):
        return
    metodos = ["Cono de Arena", "Densímetro Nuclear", "Corte y Pesada", "Balón de caucho"]
    df = pd.DataFrame({"Metodos": metodos})
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Listas")


def load_config_lists_den(path: str) -> Dict[str, List[str]]:
    ensure_config_file_den(path)
    try:
        df = safe_read_excel(path, sheet_name="Listas")
        metodos = df.get("Metodos", pd.Series([], dtype=str)).dropna().astype(str).tolist()
        metodos = [m.strip() for m in metodos if str(m).strip()]
        return {"metodos": metodos}
    except Exception:
        return {"metodos": ["Cono de Arena", "Densímetro Nuclear"]}


def save_config_lists_den(path: str, metodos: List[str]) -> None:
    metodos = [m.strip() for m in metodos if str(m).strip()]
    df = pd.DataFrame({"Metodos": metodos})
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Listas")


def load_lists_from_template_den(template_path: str) -> Dict:
    defaults = {"metodos": [], "umbral_cumple": 92.0, "umbral_obs": 90.0}
    if (not template_path) or (not os.path.exists(template_path)):
        return defaults
    try:
        df_l = safe_read_excel(template_path, sheet_name="Listas")

        def pull_any(possible_cols: List[str]) -> List[str]:
            for c in possible_cols:
                if c in df_l.columns:
                    vals = df_l[c].dropna().astype(str).tolist()
                    vals = [v.strip() for v in vals if v.strip()]
                    if vals:
                        return vals
            return []

        metodos = pull_any(["Metodo", "Método", "Metodos", "Métodos", "Columna5"])

        umbral_cumple = defaults["umbral_cumple"]
        umbral_obs = defaults["umbral_obs"]
        if "Columna7" in df_l.columns and "Columna8" in df_l.columns:
            params = pd.DataFrame({"k": df_l["Columna7"], "v": df_l["Columna8"]}).dropna()
            params["k"] = params["k"].astype(str).str.strip()
            for _, r in params.iterrows():
                try:
                    k = str(r["k"])
                    v = float(r["v"])
                    if "Umbral_A" in k or "UMBRAL_A" in k:
                        umbral_cumple = v
                    if "Umbral_O" in k or "UMBRAL_O" in k:
                        umbral_obs = v
                except Exception:
                    pass

        return {"metodos": metodos, "umbral_cumple": float(umbral_cumple), "umbral_obs": float(umbral_obs)}
    except Exception:
        return defaults


def save_data_den(df: pd.DataFrame, path: str) -> None:
    out = df.copy()
    for c in COLUMNS_DEN:
        if c not in out.columns:
            out[c] = np.nan
    out = out[COLUMNS_DEN]
    safe_write_excel(out, path)


def load_data_den(path: str) -> pd.DataFrame:
    df = safe_read_excel(path, columns=COLUMNS_DEN)
    rename_map = {"Observación": "Observacion", "Método": "Metodo", "Fecha": "Fecha_control", "_RowKey": "RowKey"}
    df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)

    for c in COLUMNS_DEN:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLUMNS_DEN].copy()

    df["ID_Registro"] = pd.to_numeric(df["ID_Registro"], errors="coerce")
    df["Fecha_control"] = pd.to_datetime(df["Fecha_control"], errors="coerce")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    num_cols = [
        "Capa_N",
        "Espesor_capa_cm",
        "Dm_inicio",
        "Dm_termino",
        "Dm_Control",
        "Coordenada_Norte",
        "Coordenada_Este",
        "Cota",
        "Profundidad_cm",
        "Densidad_Humeda_gcm3",
        "Humedad_medida_pct",
        "Humedad_Optima_pct",
        "Delta_Humedad_pct",
        "Densidad_Seca_gcm3",
        "DMCS_Proctor_gcm3",
        "pct_Compactacion",
        "Umbral_Cumple_pct",
        "Umbral_Observado_pct",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["RowKey"] = df["RowKey"].astype(str)
    needs_key = df["RowKey"].isna() | (df["RowKey"].str.strip() == "") | (df["RowKey"].str.lower() == "nan")
    if needs_key.any():
        df.loc[needs_key, "RowKey"] = [_safe_uuid() for _ in range(int(needs_key.sum()))]
        save_data_den(df, path)

    if df["ID_Registro"].notna().any():
        df.loc[df["ID_Registro"].notna(), "ID_Registro"] = df.loc[df["ID_Registro"].notna(), "ID_Registro"].astype(int)

    return df


def next_id_den(df: pd.DataFrame) -> int:
    if df.empty or df["ID_Registro"].dropna().empty:
        return 1
    return int(df["ID_Registro"].dropna().max()) + 1


def calc_densidad_seca(dh: float, w_pct: float) -> float:
    return float(dh) / (1.0 + float(w_pct) / 100.0)


def calc_pct_comp(ds: float, dmcs: float) -> float:
    return (float(ds) / float(dmcs)) * 100.0 if float(dmcs) else np.nan


def adjust_umbral_obs(umbral_a: float, umbral_o_raw: float, band: float = DEFAULT_OBS_BAND) -> float:
    o_min = max(0.0, float(umbral_a) - float(band))
    return max(float(umbral_o_raw), o_min)


def estado_qaqc_den(pct: float, umbral_a: float, umbral_o: float) -> str:
    if pd.isna(pct):
        return "—"
    if pct >= float(umbral_a):
        return "CUMPLE"
    if pct >= float(umbral_o):
        return "OBSERVADO"
    return "NO CUMPLE"


def diagnostico_den(pct: float, umbral_a: float, umbral_o: float, delta_h: float, tol_h: float) -> List[str]:
    """Explica por qué el resultado quedó OBSERVADO o NO CUMPLE."""
    reasons: List[str] = []
    try:
        if pd.isna(pct):
            reasons.append("Compactación no calculada (faltan datos o DMCS=0).")
        else:
            if float(pct) < float(umbral_o):
                reasons.append(f"Compactación {float(pct):.2f}% < Umbral OBSERVADO {float(umbral_o):.1f}% (rechazo).")
            elif float(pct) < float(umbral_a):
                reasons.append(f"Compactación {float(pct):.2f}% < Umbral CUMPLE {float(umbral_a):.1f}% (observado).")
    except Exception:
        pass
    try:
        if (delta_h is not None) and (tol_h is not None) and (not pd.isna(delta_h)) and (not pd.isna(tol_h)):
            if abs(float(delta_h)) > float(tol_h):
                reasons.append(f"ΔH={float(delta_h):+.2f}% fuera de ventana ±{float(tol_h):.1f}%.")
    except Exception:
        pass
    return reasons


def compute_kpis_den(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    d = df_in.copy()
    if d.empty:
        df_kpi = pd.DataFrame({"Metrica": [], "Valor": []})
        return df_kpi, {"total": 0, "a": 0, "o": 0, "r": 0, "pct_cumple": 0.0, "prom": np.nan, "mx": np.nan, "mn": np.nan}

    d["Estado_QAQC"] = d["Estado_QAQC"].astype(str).str.upper().str.strip()

    total = int(len(d))
    a = int((d["Estado_QAQC"] == "CUMPLE").sum())
    o = int((d["Estado_QAQC"] == "OBSERVADO").sum())
    r = int((d["Estado_QAQC"] == "NO CUMPLE").sum())

    prom = float(np.nanmean(d["pct_Compactacion"])) if d["pct_Compactacion"].notna().any() else np.nan
    mx = float(np.nanmax(d["pct_Compactacion"])) if d["pct_Compactacion"].notna().any() else np.nan
    mn = float(np.nanmin(d["pct_Compactacion"])) if d["pct_Compactacion"].notna().any() else np.nan

    pct_cumple = (a / total * 100.0) if total else 0.0

    df_kpi = pd.DataFrame(
        {
            "Metrica": [
                "Total_muestras",
                "Cant_Aprobacion",
                "Cant_Observacion",
                "Cant_Rechazo",
                "Pct_Cumple",
                "Promedio_Compactacion",
                "Max_Compactacion",
                "Min_Compactacion",
            ],
            "Valor": [
                total,
                a,
                o,
                r,
                round(pct_cumple, 2),
                round(prom, 2) if not np.isnan(prom) else np.nan,
                round(mx, 2) if not np.isnan(mx) else np.nan,
                round(mn, 2) if not np.isnan(mn) else np.nan,
            ],
        }
    )
    return df_kpi, {"total": total, "a": a, "o": o, "r": r, "pct_cumple": pct_cumple, "prom": prom, "mx": mx, "mn": mn}


def style_table_den(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    def row_bg(row):
        stt = str(row.get("Estado_QAQC", "")).upper().strip()
        if stt == "CUMPLE":
            return ["background-color: #eef9f0"] * len(row)
        if stt == "OBSERVADO":
            return ["background-color: #fff7e6"] * len(row)
        if stt == "NO CUMPLE":
            return ["background-color: #fdeff1"] * len(row)
        return [""] * len(row)

    return (
        df.style.apply(row_bg, axis=1)
        .set_table_styles(
            [
                {"selector": "th", "props": "background-color:#dfe8f7; color:#0f172a; font-weight:900;"},
                {"selector": "td", "props": "border:1px solid #d7e1f0;"},
                {"selector": "table", "props": "border-collapse:collapse; width:100%;"},
            ]
        )
    )


def delete_by_ids_den(df_all: pd.DataFrame, ids_to_delete: List[int]) -> Tuple[pd.DataFrame, int]:
    if df_all.empty or not ids_to_delete:
        return df_all, 0
    ids_to_delete = [int(x) for x in ids_to_delete]
    before = len(df_all)
    df_new = df_all.copy()
    df_new["ID_Registro"] = pd.to_numeric(df_new["ID_Registro"], errors="coerce")
    df_new = df_new[~df_new["ID_Registro"].isin(ids_to_delete)].copy()
    return df_new, (before - len(df_new))


def record_signature_den(d: Dict) -> str:
    parts = [
        str(d.get("Codigo_Proyecto", "")).strip(),
        str(d.get("Proyecto", "")).strip(),
        str(d.get("Fecha_control", "")).strip(),
        str(d.get("Sector_Zona", "")).strip(),
        str(d.get("Tramo", "")).strip(),
        str(d.get("Operador", "")).strip(),
        str(d.get("Metodo", "")).strip(),
        str(d.get("Densidad_Humeda_gcm3", "")).strip(),
        str(d.get("Humedad_medida_pct", "")).strip(),
        str(d.get("Humedad_Optima_pct", "")).strip(),
        str(d.get("DMCS_Proctor_gcm3", "")).strip(),
        str(d.get("N_Control", "")).strip(),
        str(d.get("N_Registro", "")).strip(),
        str(d.get("N_Acta", "")).strip(),
    ]
    return "|".join(parts)


def is_duplicate_recent_den(df: pd.DataFrame, sig: str, seconds: int = ANTI_DUPLICATE_WINDOW_SECONDS) -> bool:
    if df.empty or "Timestamp" not in df.columns:
        return False
    try:
        now = datetime.now()
        d2 = df.dropna(subset=["Timestamp"]).copy()
        if d2.empty:
            return False
        d2 = d2.sort_values("Timestamp", ascending=False).head(80)
        d2["__sig"] = d2.apply(lambda r: record_signature_den(r.to_dict()), axis=1)
        d2["__dt"] = pd.to_datetime(d2["Timestamp"], errors="coerce")
        d2 = d2.dropna(subset=["__dt"])
        if d2.empty:
            return False
        recent = d2[(now - d2["__dt"]).dt.total_seconds() <= float(seconds)]
        return bool((recent["__sig"] == sig).any())
    except Exception:
        return False


def get_record_by_id_den(df: pd.DataFrame, rid: int) -> Optional[pd.Series]:
    if df.empty:
        return None
    d = df.copy()
    d["ID_Registro"] = pd.to_numeric(d["ID_Registro"], errors="coerce")
    d = d[d["ID_Registro"] == int(rid)]
    if d.empty:
        return None
    d = d.sort_values("Timestamp", ascending=False)
    return d.iloc[0]


def apply_update_by_rowkey_den(df: pd.DataFrame, rowkey: str, new_values: Dict) -> Tuple[pd.DataFrame, bool]:
    if df.empty:
        return df, False
    d = df.copy()
    mask = d["RowKey"].astype(str) == str(rowkey)
    if not mask.any():
        return df, False
    for k, v in new_values.items():
        if k in d.columns:
            d.loc[mask, k] = v
    return d, True



def reset_form_den(clear_last_saved: bool = True):
  '''  """Solicita reset sin modificar keys de widgets en el mismo ciclo."""
    st.session_state["DEN_RESET_REQUESTED"] = True
    st.session_state["DEN_RESET_CLEAR_LAST"] = bool(clear_last_saved)
    _clear_last = bool(st.session_state.get("DEN_RESET_CLEAR_LAST", True))
    for _k in [
        "den_fecha_ctrl", "den_cod_proy", "den_proyecto", "den_n_reg", "den_n_ctrl", "den_n_acta",
        "den_sector_txt", "den_tramo_txt", "den_frente", "den_capa_txt", "den_esp_txt",
        "den_dm_ini_txt", "den_dm_ter_txt", "den_dm_ctrl_txt", "den_coord_n_txt", "den_coord_e_txt", "den_cota_txt",
        "den_operador", "den_met_sel", "den_met_otro", "den_prof_txt", "den_obs",
        "den_dh_num", "den_h_num", "den_hopt_num", "den_dmcs_num",
        "den_docid_eett", "DEN_DOC_EETT_IDX",
        "DEN_EDIT_ID", "DEN_EDIT_ROWKEY", "DEN_EDIT_PICK",
        "DEN_LAST_SUBMIT_TS", "DEN_FORCE_RECALC_TS",
    ]:
        st.session_state.pop(_k, None)
    if _clear_last:
        st.session_state.pop("DEN_LAST_SAVED", None)
        st.session_state.pop("DEN_RESET_REQUESTED", None)
        st.session_state.pop("DEN_RESET_CLEAR_LAST", None)
    st.rerun()'''

def check_reset_form_den():
    if "RESET_DEN_PENDING" in st.session_state:
        den_fields = [
        "den_cod_proy", "den_proyecto", "den_n_reg", "den_n_ctrl", "den_n_acta",
        "den_sector_txt", "den_tramo_txt", "den_frente", "den_capa_txt", "den_esp_txt",
        "den_dm_ini_txt", "den_dm_ter_txt", "den_dm_ctrl_txt", "den_coord_n_txt", "den_coord_e_txt", "den_cota_txt",
        "den_operador", "den_met_sel", "den_met_otro", "den_prof_txt", "den_obs",
        "den_dh_num", "den_h_num", "den_hopt_num", "den_dmcs_num",
        "den_docid_eett", "DEN_DOC_EETT_IDX",
        "DEN_EDIT_ID", "DEN_EDIT_ROWKEY", "DEN_EDIT_PICK",
        "DEN_LAST_SUBMIT_TS", "DEN_FORCE_RECALC_TS",
         ]
        for field in den_fields:
            if field in st.session_state:
                st.session_state[field] = "" if "num" not in field else 0.0
        del st.session_state["RESET_DEN_PENDING"]
        st.rerun()




def set_last_saved_den(calc: Optional[Dict]):
    if calc is None:
        st.session_state.pop("DEN_LAST_SAVED", None)
    else:
        st.session_state["DEN_LAST_SAVED"] = calc


def get_last_saved_den() -> Optional[Dict]:
    d = st.session_state.get("DEN_LAST_SAVED", None)
    return d if isinstance(d, dict) else None


def load_record_into_form_den(row: pd.Series):
    st.session_state["DEN_EDIT_ID"] = int(row.get("ID_Registro"))
    st.session_state["DEN_EDIT_ROWKEY"] = str(row.get("RowKey"))

    st.session_state["den_fecha_ctrl"] = row.get("Fecha_control").date() if pd.notna(row.get("Fecha_control")) else date.today()
    st.session_state["den_cod_proy"] = str(row.get("Codigo_Proyecto") or "")
    st.session_state["den_proyecto"] = str(row.get("Proyecto") or "")
    st.session_state["den_n_reg"] = str(row.get("N_Registro") or "")
    st.session_state["den_n_ctrl"] = str(row.get("N_Control") or "")
    st.session_state["den_n_acta"] = str(row.get("N_Acta") or "")
    st.session_state["den_docid_eett"] = str(row.get("DocID_EETT") or "")  # ✅ carga doc

    # fuerza recálculo del índice del selectbox en el próximo render (seguro)
    if "DEN_DOC_EETT_IDX" in st.session_state:
        del st.session_state["DEN_DOC_EETT_IDX"]

    st.session_state["den_sector_txt"] = str(row.get("Sector_Zona") or "")
    st.session_state["den_tramo_txt"] = str(row.get("Tramo") or "")
    st.session_state["den_frente"] = str(row.get("Frente_Tramo") or "")

    st.session_state["den_capa_txt"] = "" if pd.isna(row.get("Capa_N")) else str(int(row.get("Capa_N")))
    st.session_state["den_esp_txt"] = "" if pd.isna(row.get("Espesor_capa_cm")) else str(float(row.get("Espesor_capa_cm")))
    st.session_state["den_dm_ini_txt"] = "" if pd.isna(row.get("Dm_inicio")) else str(float(row.get("Dm_inicio")))
    st.session_state["den_dm_ter_txt"] = "" if pd.isna(row.get("Dm_termino")) else str(float(row.get("Dm_termino")))
    st.session_state["den_dm_ctrl_txt"] = "" if pd.isna(row.get("Dm_Control")) else str(float(row.get("Dm_Control")))

    st.session_state["den_coord_n_txt"] = "" if pd.isna(row.get("Coordenada_Norte")) else str(float(row.get("Coordenada_Norte")))
    st.session_state["den_coord_e_txt"] = "" if pd.isna(row.get("Coordenada_Este")) else str(float(row.get("Coordenada_Este")))
    st.session_state["den_cota_txt"] = "" if pd.isna(row.get("Cota")) else str(float(row.get("Cota")))

    st.session_state["den_operador"] = str(row.get("Operador") or "")
    st.session_state["den_prof_txt"] = "" if pd.isna(row.get("Profundidad_cm")) else str(float(row.get("Profundidad_cm")))

    metodo_val = str(row.get("Metodo") or "").strip()
    st.session_state["den_met_sel"] = metodo_val if metodo_val else "— Seleccionar —"
    st.session_state["den_met_otro"] = ""

    st.session_state["den_dh_num"] = float(row.get("Densidad_Humeda_gcm3")) if pd.notna(row.get("Densidad_Humeda_gcm3")) else 0.0
    st.session_state["den_h_num"] = float(row.get("Humedad_medida_pct")) if pd.notna(row.get("Humedad_medida_pct")) else 0.0
    st.session_state["den_hopt_num"] = float(row.get("Humedad_Optima_pct")) if pd.notna(row.get("Humedad_Optima_pct")) else 0.0
    st.session_state["den_dmcs_num"] = float(row.get("DMCS_Proctor_gcm3")) if pd.notna(row.get("DMCS_Proctor_gcm3")) else 0.0
    st.session_state["den_obs"] = str(row.get("Observacion") or "")


# INIT DENSIDADES
ensure_data_file_den(DATA_FILE_DEN)
ensure_config_file_den(CONFIG_FILE_DEN)
tpl_den = load_lists_from_template_den(TEMPLATE_FILE_DEN)
cfg_den = load_config_lists_den(CONFIG_FILE_DEN)
metodos_den = list(dict.fromkeys([*cfg_den.get("metodos", []), *tpl_den.get("metodos", [])])) or ["Cono de Arena", "Densímetro Nuclear"]

# =========================================================
# =================  CONTROL PIE (m² y m³)  ================
# =========================================================
COLUMNS_PIE_M2 = [
    "RowKey",
    "ID_Registro",
    "Fecha",
    "COD_PROYECTO",
     "DocID_EETT",
    "Sector_Zona",
    "Frente_Tramo",
    "DM_inicio",
    "DM_termino",
    "Largo_Tramo_m",
    "Ancho_m",
    "Area_m2",
    "PIE_VALOR_m2_por_ensayo",
    "Requeridas",
    "Ejecutadas",
    "Brecha",
    "Pct_Cumpl",
    "Estado",
    "Timestamp",
]

COLUMNS_PIE_M3 = [
    "RowKey",
    "ID_Registro",
    "Fecha",
    "COD_PROYECTO",
    "DocID_EETT",
    "Sector_Zona",
    "Frente_Tramo",
    "DM_inicio",
    "DM_termino",
    "Largo_Tramo_m",
    "Ancho_m",
    "Espesor_m",
    "Volumen_m3",
    "PIE_VALOR_m3_por_ensayo",
    "Requeridas",
    "Ejecutadas",
    "Brecha",
    "Pct_Cumpl",
    "Estado",
    "Timestamp",
]


def ensure_data_file(path: str, columns: List[str]) -> None:
    if not os.path.exists(path):
        safe_write_excel(pd.DataFrame(columns=columns), path)


def save_data_generic(df: pd.DataFrame, path: str, columns: List[str]) -> None:
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = np.nan
    out = out[columns]
    safe_write_excel(out, path)


def load_data_generic(path: str, columns: List[str], date_cols: List[str], num_cols: List[str]) -> pd.DataFrame:
    df = safe_read_excel(path, columns=columns)
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    df = df[columns].copy()

    if "ID_Registro" in df.columns:
        df["ID_Registro"] = pd.to_numeric(df["ID_Registro"], errors="coerce")

    for dc in date_cols:
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["RowKey"] = df["RowKey"].astype(str)
    needs_key = df["RowKey"].isna() | (df["RowKey"].str.strip() == "") | (df["RowKey"].str.lower() == "nan")
    if needs_key.any():
        df.loc[needs_key, "RowKey"] = [_safe_uuid() for _ in range(int(needs_key.sum()))]
        save_data_generic(df, path, columns)

    if df["ID_Registro"].notna().any():
        df.loc[df["ID_Registro"].notna(), "ID_Registro"] = df.loc[df["ID_Registro"].notna(), "ID_Registro"].astype(int)

    return df


def next_id_generic(df: pd.DataFrame) -> int:
    if df.empty or df["ID_Registro"].dropna().empty:
        return 1
    return int(df["ID_Registro"].dropna().max()) + 1


def get_record_by_id_generic(df: pd.DataFrame, rid: int) -> Optional[pd.Series]:
    if df.empty:
        return None
    d = df.copy()
    d["ID_Registro"] = pd.to_numeric(d["ID_Registro"], errors="coerce")
    d = d[d["ID_Registro"] == int(rid)]
    if d.empty:
        return None
    d = d.sort_values("Timestamp", ascending=False)
    return d.iloc[0]


def apply_update_by_rowkey_generic(df: pd.DataFrame, rowkey: str, new_values: Dict) -> Tuple[pd.DataFrame, bool]:
    if df.empty:
        return df, False
    d = df.copy()
    mask = d["RowKey"].astype(str) == str(rowkey)
    if not mask.any():
        return df, False
    for k, v in new_values.items():
        if k in d.columns:
            d.loc[mask, k] = v
    return d, True


def delete_by_ids_generic(df_all: pd.DataFrame, ids_to_delete: List[int]) -> Tuple[pd.DataFrame, int]:
    if df_all.empty or not ids_to_delete:
        return df_all, 0
    ids_to_delete = [int(x) for x in ids_to_delete]
    before = len(df_all)
    df_new = df_all.copy()
    df_new["ID_Registro"] = pd.to_numeric(df_new["ID_Registro"], errors="coerce")
    df_new = df_new[~df_new["ID_Registro"].isin(ids_to_delete)].copy()
    return df_new, (before - len(df_new))


def style_table_pie(df: pd.DataFrame, estado_col: str = "Estado") -> "pd.io.formats.style.Styler":
    def row_bg(row):
        stt = str(row.get(estado_col, "")).upper().strip()
        if stt == "CUMPLE":
            return ["background-color: #eef9f0"] * len(row)
        if stt in ["PENDIENTE", "OBSERVADO"]:
            return ["background-color: #fff7e6"] * len(row)
        return [""] * len(row)

    return (
        df.style.apply(row_bg, axis=1)
        .set_table_styles(
            [
                {"selector": "th", "props": "background-color:#dfe8f7; color:#0f172a; font-weight:900;"},
                {"selector": "td", "props": "border:1px solid #d7e1f0;"},
                {"selector": "table", "props": "border-collapse:collapse; width:100%;"},
            ]
        )
    )


def pie_calc_m2(largo: Optional[float], ancho: Optional[float], pie_valor: Optional[float], ejecutadas: Optional[float]) -> Dict:
    largo = float(largo) if largo is not None and not np.isnan(largo) else np.nan
    ancho = float(ancho) if ancho is not None and not np.isnan(ancho) else np.nan
    pie_valor = float(pie_valor) if pie_valor is not None and not np.isnan(pie_valor) else np.nan
    ejecutadas = float(ejecutadas) if ejecutadas is not None and not np.isnan(ejecutadas) else 0.0

    area = np.nan
    if not np.isnan(largo) and not np.isnan(ancho):
        area = float(largo) * float(ancho)

    requeridas = np.nan
    if not np.isnan(area) and not np.isnan(pie_valor) and pie_valor > 0:
        requeridas = float(int(np.ceil(area / pie_valor)))

    brecha = np.nan
    pct = np.nan
    estado = "—"
    if not np.isnan(requeridas) and requeridas >= 0:
        brecha = float(requeridas) - float(ejecutadas)
        pct = (float(ejecutadas) / float(requeridas) * 100.0) if requeridas > 0 else (100.0 if ejecutadas > 0 else 0.0)
        if requeridas == 0:
            estado = "—"
        elif ejecutadas >= requeridas:
            estado = "CUMPLE"
        else:
            estado = "PENDIENTE"

    return {"base": area, "requeridas": requeridas, "brecha": brecha, "pct": pct, "estado": estado}


def pie_calc_m3(largo: Optional[float], ancho: Optional[float], espesor: Optional[float], pie_valor: Optional[float], ejecutadas: Optional[float]) -> Dict:
    largo = float(largo) if largo is not None and not np.isnan(largo) else np.nan
    ancho = float(ancho) if ancho is not None and not np.isnan(ancho) else np.nan
    espesor = float(espesor) if espesor is not None and not np.isnan(espesor) else np.nan
    pie_valor = float(pie_valor) if pie_valor is not None and not np.isnan(pie_valor) else np.nan
    ejecutadas = float(ejecutadas) if ejecutadas is not None and not np.isnan(ejecutadas) else 0.0

    vol = np.nan
    if not np.isnan(largo) and not np.isnan(ancho) and not np.isnan(espesor):
        vol = float(largo) * float(ancho) * float(espesor)

    requeridas = np.nan
    if not np.isnan(vol) and not np.isnan(pie_valor) and pie_valor > 0:
        requeridas = float(int(np.ceil(vol / pie_valor)))

    brecha = np.nan
    pct = np.nan
    estado = "—"
    if not np.isnan(requeridas) and requeridas >= 0:
        brecha = float(requeridas) - float(ejecutadas)
        pct = (float(ejecutadas) / float(requeridas) * 100.0) if requeridas > 0 else (100.0 if ejecutadas > 0 else 0.0)
        if requeridas == 0:
            estado = "—"
        elif ejecutadas >= requeridas:
            estado = "CUMPLE"
        else:
            estado = "PENDIENTE"

    return {"base": vol, "requeridas": requeridas, "brecha": brecha, "pct": pct, "estado": estado}


def compute_kpis_pie(df_in: pd.DataFrame, base_col: str) -> Tuple[pd.DataFrame, Dict]:
    d = df_in.copy()
    if d.empty:
        dfk = pd.DataFrame({"Metrica": [], "Valor": []})
        return dfk, {"total": 0}

    d["Estado"] = d["Estado"].astype(str).str.upper().str.strip()

    total = int(len(d))
    cumple = int((d["Estado"] == "CUMPLE").sum())
    pend = int((d["Estado"] == "PENDIENTE").sum())

    base_total = float(np.nansum(d[base_col].values)) if base_col in d.columns and d[base_col].notna().any() else 0.0
    req_total = float(np.nansum(d["Requeridas"].values)) if d["Requeridas"].notna().any() else 0.0
    eje_total = float(np.nansum(d["Ejecutadas"].values)) if d["Ejecutadas"].notna().any() else 0.0
    brecha_total = float(req_total - eje_total)

    pct_global = (eje_total / req_total * 100.0) if req_total > 0 else (100.0 if eje_total > 0 else 0.0)

    df_kpi = pd.DataFrame(
        {
            "Metrica": [
                "Total_tramos",
                "Cant_Cumple",
                "Cant_Pendiente",
                f"{base_col}_total",
                "Ensayos_requeridos_total",
                "Ensayos_ejecutados_total",
                "Brecha_total",
                "Cumplimiento_global_pct",
            ],
            "Valor": [
                total,
                cumple,
                pend,
                round(base_total, 2),
                round(req_total, 2),
                round(eje_total, 2),
                round(brecha_total, 2),
                round(pct_global, 2),
            ],
        }
    )
    return df_kpi, {
        "total": total,
        "cumple": cumple,
        "pend": pend,
        "base_total": base_total,
        "req_total": req_total,
        "eje_total": eje_total,
        "brecha_total": brecha_total,
        "pct_global": pct_global,
    }


# INIT PIE FILES
ensure_data_file(DATA_FILE_PIE_M2, COLUMNS_PIE_M2)
ensure_data_file(DATA_FILE_PIE_M3, COLUMNS_PIE_M3)

# ---------------------------------------------------------
# SIDEBAR: NAVEGACIÓN (6 PANTALLAS)
# ---------------------------------------------------------
st.sidebar.markdown("### 🧭 Navegación")

if "APP_PAGE" not in st.session_state:
    st.session_state["APP_PAGE"] = "DEN_P1"

st.sidebar.markdown("#### 🧾 Densidades")
if st.sidebar.button("🧾 1) Ingreso Densidades", width='stretch'):
    st.session_state["APP_PAGE"] = "DEN_P1"

if st.sidebar.button("📊 2) KPIs Densidades", width='stretch'):
    st.session_state["APP_PAGE"] = "DEN_P2"


st.sidebar.markdown("<div class='hr'></div>", unsafe_allow_html=True)
st.sidebar.markdown("#### 📐 Control PIE m²")
if st.sidebar.button("🧾 3) Ingreso PIE m²", width='stretch'):
    st.session_state["APP_PAGE"] = "PIE_M2_P1"

if st.sidebar.button("📊 4) KPIs PIE m²", width='stretch'):
    st.session_state["APP_PAGE"] = "PIE_M2_P2"


st.sidebar.markdown("<div class='hr'></div>", unsafe_allow_html=True)
st.sidebar.markdown("#### 🧱 Control PIE m³")
if st.sidebar.button("🧾 5) Ingreso PIE m³", width='stretch'):
    st.session_state["APP_PAGE"] = "PIE_M3_P1"

if st.sidebar.button("📊 6) KPIs PIE m³", width='stretch'):
    st.session_state["APP_PAGE"] = "PIE_M3_P2"


# =========================================================
# ===============  DENSIDADES – SIDEBAR QA/QC  =============
# =========================================================
if st.session_state["APP_PAGE"] in ["DEN_P1", "DEN_P2"]:
    st.sidebar.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("### Parámetros QA/QC (Densidades)")

    tol_hum_opt = st.sidebar.number_input(
        "Tolerancia Humedad Óptima (±%)",
        value=float(st.session_state.get("TOL_HUM_OPT", DEFAULT_TOL_HUM_OPT)),
        step=0.5,
        format="%.1f",
    )
    st.session_state["TOL_HUM_OPT"] = float(tol_hum_opt)

    if "UMBRAL_A" not in st.session_state:
        st.session_state["UMBRAL_A"] = float(tpl_den.get("umbral_cumple", 92.0) or 92.0)
    if "UMBRAL_O_RAW" not in st.session_state:
        st.session_state["UMBRAL_O_RAW"] = float(tpl_den.get("umbral_obs", 90.0) or 90.0)

    UMBRAL_A = st.sidebar.number_input("Umbral A (CUMPLE ≥ %)", value=float(st.session_state["UMBRAL_A"]), step=0.5, format="%.1f")
    UMBRAL_O_RAW = st.sidebar.number_input("Umbral O (OBSERVADO ≥ %)", value=float(st.session_state["UMBRAL_O_RAW"]), step=0.5, format="%.1f")
    UMBRAL_O = adjust_umbral_obs(float(UMBRAL_A), float(UMBRAL_O_RAW), band=float(DEFAULT_OBS_BAND))

    st.session_state["UMBRAL_A"] = float(UMBRAL_A)
    st.session_state["UMBRAL_O_RAW"] = float(UMBRAL_O_RAW)

    st.sidebar.markdown(
        f"""
<div class="qi-card">
  <div style="font-weight:900;color:#0f172a;margin-bottom:6px">Leyenda A/O/R</div>
  <span class="qi-chip qi-green">A · CUMPLE</span>
  <span class="qi-chip qi-amber">O · OBSERVADO</span>
  <span class="qi-chip qi-red">R · NO CUMPLE</span>
  <div class="qi-muted" style="margin-top:8px;font-size:0.95rem">
    <b>O auto-ajustado</b>: A={float(UMBRAL_A):.1f}% · O(usado)={float(UMBRAL_O):.1f}%
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("⚙️ Administrar lista (Métodos)", expanded=False):
        txt_met = st.text_area("Métodos (1 por línea)", value="\n".join(metodos_den), height=160, key="cfg_met_den")
        if st.button("💾 Guardar lista", width='stretch'):
            new_met = [x.strip() for x in txt_met.splitlines() if x.strip()]
            save_config_lists_den(CONFIG_FILE_DEN, new_met)
            st.success("Lista guardada.")
            st.rerun()


# =========================================================
# ===================  PANTALLA 1 DENSIDADES  ==============
# =========================================================
if st.session_state["APP_PAGE"] == "DEN_P1":
    check_reset_form_den()
    if "RESET_DEN_PENDING" not in st.session_state and "guardado_exitoso" in st.session_state:
        st.toast(st.session_state.get("guardado_exitoso"),icon="✅")
        del st.session_state["guardado_exitoso"]

    st.caption("Densidades · Pantalla 1 · Ingreso + Cálculos + Ver/Editar/Eliminar + Export")
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    df_all0 = load_data_den(DATA_FILE_DEN)
    ids_all0 = sorted(df_all0["ID_Registro"].dropna().astype(int).unique().tolist()) if not df_all0.empty else []

    # MANTENER VALORES DESPUES DEL GUARDADO INICIALMENTE FALSO
    st.session_state["den_keep"] = DEFAULT_KEEP_VALUES

    topb1, topb2, topb3, topb4 = st.columns([2.4, 1.1, 1.3, 3.2])

    with topb1:
        keep_values = st.checkbox(
            "Mantener valores después de guardar (si lo desmarcas, queda todo en blanco)",
            key="den_keep_chk",
        )

    # SI CAMBIA EL CHECKBOX CAMBIA A TRUE
    if keep_values:
        st.session_state["den_keep"] = True

    with topb2:
        limpiar= st.button("🧹 LIMPIAR", width='stretch')


    with topb3:
        if st.button("🔄 RECALCULAR", width='stretch'):
            st.session_state["DEN_FORCE_RECALC_TS"] = time.time()
            st.rerun()

    with topb4:
        edit_id = st.selectbox("✏️ Editar ID", options=[None] + ids_all0, index=0, key="DEN_EDIT_PICK")
        if edit_id is not None:
            if st.button("✏️ Cargar ID seleccionado", width='stretch'):
                df_temp = load_data_den(DATA_FILE_DEN)
                row = get_record_by_id_den(df_temp, int(edit_id))
                if row is None:
                    st.error("No encontré ese ID en la base.")
                else:
                    load_record_into_form_den(row)
                    st.success(f"ID {int(edit_id)} cargado. Modifica y presiona **Guardar cambios**.")
                    st.rerun()

    # ---------- SECCIÓN 1 ----------
    st.markdown("<div class='qi-section'><div class='qi-h3'>Identificación y Control</div>", unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        fecha_ctrl = st.date_input("Fecha control", key="den_fecha_ctrl")
        codigo_proy = st.text_input("Código de Proyecto", key="den_cod_proy").strip()
    with a2:
        proyecto = st.text_input("Proyecto (DIGITAR)", key="den_proyecto").strip()
        n_registro = st.text_input("N° Registro",  key="den_n_reg").strip()
    with a3:
        n_control = st.text_input("N° Control", key="den_n_ctrl").strip()
        n_acta = st.text_input("N° Acta", key="den_n_acta").strip()
    with a4:
        sector_final = st.text_input("Sector/Zona (DIGITAR)", key="den_sector_txt").strip()
        tramo_final = st.text_input("Tramo (DIGITAR)",  key="den_tramo_txt").strip()
        den_docid_eett = eett_doc_selector("DEN", "den_docid_eett", label="Documento Técnico (EETT)")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- SECCIÓN 2 ----------
    st.markdown("<div class='qi-section'><div class='qi-h3'>Ubicación / Geometría</div>", unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        capa_txt = st.text_input("N° Capa", key="den_capa_txt")
        esp_txt = st.text_input("Espesor capa (cm)",  key="den_esp_txt")
    with b2:
        dm_ini_txt = st.text_input("Dm inicio", key="den_dm_ini_txt")
        dm_ter_txt = st.text_input("Dm término",key="den_dm_ter_txt")
    with b3:
        dm_ctrl_txt = st.text_input("Dm Control",key="den_dm_ctrl_txt")
        cota_txt = st.text_input("Cota", key="den_cota_txt")
    with b4:
        coord_n_txt = st.text_input("Coordenada Norte", key="den_coord_n_txt")
        coord_e_txt = st.text_input("Coordenada Este",  key="den_coord_e_txt")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- SECCIÓN 3 ----------
    st.markdown("<div class='qi-section'><div class='qi-h3'>Operación / Ensayo</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        operador = st.text_input("Operador (DIGITAR)",  key="den_operador").strip()
        prof_txt = st.text_input("Profundidad (cm)",  key="den_prof_txt")
    with c2:
        met_opts = ["— Seleccionar —", *metodos_den, "Otro (digitar)"]
        met_sel = st.selectbox("Método", met_opts, index=0, key="den_met_sel")
        met_otro = ""
        if met_sel == "Otro (digitar)":
            met_otro = st.text_input("Método (otro)",  key="den_met_otro")
        frente = st.text_input("Frente / Detalle", key="den_frente").strip()

    metodo_final = (met_otro.strip() if met_sel == "Otro (digitar)" else ("" if met_sel == "— Seleccionar —" else met_sel)).strip()

    with c3:
        dh_num = st.number_input("Densidad Húmeda (g/cm³)", min_value=0.0, step=0.001, format="%.3f", key="den_dh_num")
        h_num = st.number_input("Humedad medida (%)", min_value=0.0, step=0.1, format="%.1f", key="den_h_num")
    with c4:
        hopt_num = st.number_input("Humedad óptima Proctor (%)", min_value=0.0, step=0.1, format="%.1f", key="den_hopt_num")
        dmcs_num = st.number_input("DMCS Proctor (g/cm³)", min_value=0.0, step=0.001, format="%.3f", key="den_dmcs_num")

    observacion = st.text_area("Observación", key="den_obs")
    st.markdown("</div>", unsafe_allow_html=True)

    # CALC LIVE
    dens_h_v = safe_float(dh_num) if safe_float(dh_num) > 0 else None
    hum_v = safe_float(h_num) if safe_float(h_num) >= 0 else None
    hopt_v = safe_float(hopt_num ) if safe_float(hopt_num) >= 0 else None
    dmcs_v = safe_float(dmcs_num) if safe_float(dmcs_num) > 0 else None

    has_live = (dens_h_v is not None) and (dmcs_v is not None) and (hum_v is not None) and (hopt_v is not None)

    dens_s_disp, pct_disp = np.nan, np.nan
    delta_disp, vent_disp = np.nan, "—"
    estado_disp = "—"

    if has_live:
        dens_s_disp = calc_densidad_seca(dens_h_v, hum_v)
        pct_disp = calc_pct_comp(dens_s_disp, dmcs_v)
        UMBRAL_A_eff = float(st.session_state["UMBRAL_A"])
        UMBRAL_O_eff = adjust_umbral_obs(float(st.session_state["UMBRAL_A"]), float(st.session_state["UMBRAL_O_RAW"]))
        estado_disp = estado_qaqc_den(float(pct_disp), float(UMBRAL_A_eff), float(UMBRAL_O_eff)) if pd.notna(pct_disp) else "—"
        delta_disp = hum_v - hopt_v
        vent_disp = "OK" if abs(delta_disp) <= float(st.session_state["TOL_HUM_OPT"]) else "OBSERVADO"
    else:
        last = get_last_saved_den()
        if last:
            dens_s_disp = last.get("dens_s", np.nan)
            pct_disp = last.get("pct", np.nan)
            delta_disp = last.get("delta", np.nan)
            vent_disp = last.get("ventana", "—")
            estado_disp = last.get("estado", "—")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        kpi_card("Densidad Seca (g/cm³)", f"{dens_s_disp:.3f}" if pd.notna(dens_s_disp) else "—")
    with r2:
        UMBRAL_A_eff = float(st.session_state["UMBRAL_A"])
        UMBRAL_O_eff = adjust_umbral_obs(float(st.session_state["UMBRAL_A"]), float(st.session_state["UMBRAL_O_RAW"]))
        kpi_card("% Compactación", f"{pct_disp:.1f}%" if pd.notna(pct_disp) else "—", f"A={UMBRAL_A_eff:.1f}% · O={UMBRAL_O_eff:.1f}%")
    with r3:
        kpi_card("Δ Humedad (Terreno-Proctor)", f"{delta_disp:+.2f}%" if pd.notna(delta_disp) else "—", f"Ventana ±{float(st.session_state['TOL_HUM_OPT']):.1f}% → {vent_disp}")
    with r4:
        color = "#1b5e20" if estado_disp == "CUMPLE" else ("#7a4f00" if estado_disp == "OBSERVADO" else "#8a1c1c")
        st.markdown(
            f"""
            <div class="qi-card">
                <div style="color:#64748b;font-size:0.90rem;font-weight:900">Estado QA/QC</div>
                <div style="color:{color};font-size:1.8rem;font-weight:900;margin-top:6px">{estado_disp}</div>
                <div class="qi-muted" style="margin-top:6px">
                    <span class="qi-chip qi-green">A · CUMPLE</span>
                    <span class="qi-chip qi-amber">O · OBSERVADO</span>
                    <span class="qi-chip qi-red">R · NO CUMPLE</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    
    # Diagnóstico cuando no cumple (o queda observado)
    if str(estado_disp).upper().strip() != "CUMPLE":
        try:
            reasons = diagnostico_den(
                float(pct_disp) if pd.notna(pct_disp) else np.nan,
                float(UMBRAL_A_eff),
                float(UMBRAL_O_eff),
                float(delta_disp) if pd.notna(delta_disp) else np.nan,
                float(st.session_state.get("TOL_HUM_OPT", DEFAULT_TOL_HUM_OPT)),
            )
        except Exception:
            reasons = []
        if reasons:
            with st.expander("🩺 Diagnóstico (por qué no cumple)", expanded=True):
                for r in reasons:
                    st.write(f"- {r}")
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    left_save, mid_save, right_save = st.columns([1.3, 1.3, 3.4])
    with left_save:
        guardar = st.button("💾 Guardar registro", type="primary", width='stretch')
    with mid_save:
        guardar_cambios = st.button("💾 Guardar cambios (EDIT)", width='stretch')
    with right_save:
        if st.session_state.get("DEN_EDIT_ID") and st.session_state.get("DEN_EDIT_ROWKEY"):
            st.info(f"Editando ID={st.session_state['DEN_EDIT_ID']} ✅")
        else:
            st.info("Modo nuevo registro ✅")

    def validate_den_common() -> List[str]:
        errs = []
        for label, raw in [
            ("Espesor capa (cm)", esp_txt),
            ("Dm inicio", dm_ini_txt),
            ("Dm término", dm_ter_txt),
            ("Dm Control", dm_ctrl_txt),
            ("Coordenada Norte", coord_n_txt),
            ("Coordenada Este", coord_e_txt),
            ("Cota", cota_txt),
            ("Profundidad (cm)", prof_txt),
            ("N° Capa", capa_txt),
        ]:
            e = is_invalid_number_if_filled(label, raw)
            if e:
                errs.append(e)

        if not codigo_proy:
            errs.append("⚠️ Falta Código de Proyecto.")
        if not proyecto:
            errs.append("⚠️ Falta Proyecto.")
        if not operador:
            errs.append("⚠️ Falta Operador.")
        if not sector_final:
            errs.append("⚠️ Falta Sector/Zona.")
        if not metodo_final:
            errs.append("⚠️ Falta Método.")

        if dens_h_v is None:
            errs.append("⚠️ Densidad Húmeda inválida (debe ser > 0).")
        if dmcs_v is None or dmcs_v <= 0:
            errs.append("⚠️ DMCS Proctor inválido (debe ser > 0).")
        if hum_v is None or hum_v < 0:
            errs.append("⚠️ Humedad medida inválida (debe ser ≥ 0).")
        if hopt_v is None or hopt_v < 0:
            errs.append("⚠️ Humedad óptima inválida (debe ser ≥ 0).")
        return errs


    if limpiar:
        reset_form_den(clear_last_saved=True)
        st.session_state["RESET_DEN_PENDING"] = True
        st.rerun()

    if guardar:
        now_ts = time.time()
        last_ts = 0.0
        if (now_ts - last_ts) < ANTI_DOUBLECLICK_SECONDS:
            st.warning("⚠️ Botón presionado muy rápido. Se bloqueó para evitar duplicidad.")
            st.stop()
        st.session_state["DEN_LAST_SUBMIT_TS"] = now_ts

        errs = validate_den_common()
        if errs:
            for e in errs:
                st.error(e)
            st.stop()

        capa = parse_int(capa_txt)
        espesor_cm = parse_float_loose(esp_txt)
        dm_ini = parse_float_loose(dm_ini_txt)
        dm_ter = parse_float_loose(dm_ter_txt)
        dm_control = parse_float_loose(dm_ctrl_txt)
        cota = parse_float_loose(cota_txt)
        coord_n = parse_float_loose(coord_n_txt)
        coord_e = parse_float_loose(coord_e_txt)
        prof_cm = parse_float_loose(prof_txt)

        dens_s = calc_densidad_seca(float(dens_h_v), float(hum_v))
        pct_comp = calc_pct_comp(float(dens_s), float(dmcs_v))
        delta_h = float(hum_v) - float(hopt_v)
        ventana = "OK" if abs(delta_h) <= float(st.session_state["TOL_HUM_OPT"]) else "OBSERVADO"
        UMBRAL_A_eff = float(st.session_state["UMBRAL_A"])
        UMBRAL_O_eff = adjust_umbral_obs(float(st.session_state["UMBRAL_A"]), float(st.session_state["UMBRAL_O_RAW"]))
        estado = estado_qaqc_den(float(pct_comp), float(UMBRAL_A_eff), float(UMBRAL_O_eff))

        df_now = load_data_den(DATA_FILE_DEN)
        new_id = int(next_id_den(df_now))

        nuevo = {
            "RowKey": _safe_uuid(),
            "ID_Registro": new_id,
            "Codigo_Proyecto": codigo_proy,
            "Proyecto": proyecto,
            "N_Registro": n_registro,
            "N_Control": n_control,
            "N_Acta": n_acta,
            "Fecha_control": pd.to_datetime(fecha_ctrl),
            "Sector_Zona": sector_final,
            "Tramo": tramo_final,
            "Frente_Tramo": frente,
            "Capa_N": float(capa) if capa is not None else np.nan,
            "Espesor_capa_cm": float(espesor_cm) if espesor_cm is not None else np.nan,
            "Dm_inicio": float(dm_ini) if dm_ini is not None else np.nan,
            "Dm_termino": float(dm_ter) if dm_ter is not None else np.nan,
            "Dm_Control": float(dm_control) if dm_control is not None else np.nan,
            "Coordenada_Norte": float(coord_n) if coord_n is not None else np.nan,
            "Coordenada_Este": float(coord_e) if coord_e is not None else np.nan,
            "Cota": float(cota) if cota is not None else np.nan,
            "Operador": operador,
            "Metodo": metodo_final,
            "Profundidad_cm": float(prof_cm) if prof_cm is not None else np.nan,
            "Densidad_Humeda_gcm3": float(dens_h_v),
            "Humedad_medida_pct": float(hum_v),
            "Humedad_Optima_pct": float(hopt_v),
            "Delta_Humedad_pct": float(delta_h),
            "Ventana_Humedad": str(ventana),
            "Densidad_Seca_gcm3": float(dens_s),
            "DMCS_Proctor_gcm3": float(dmcs_v),
            "pct_Compactacion": float(pct_comp),
            "Umbral_Cumple_pct": float(UMBRAL_A_eff),
            "Umbral_Observado_pct": float(UMBRAL_O_eff),
            "Estado_QAQC": estado,
            "Observacion": str(observacion).strip(),
            "Timestamp": pd.to_datetime(datetime.now()),
        }

        sig = record_signature_den(nuevo)
        if is_duplicate_recent_den(df_now, sig, seconds=ANTI_DUPLICATE_WINDOW_SECONDS):
            st.warning("⚠️ Se detectó un duplicado reciente. No se guardó.")
            st.stop()
        df2 = pd.concat([df_now, pd.DataFrame([nuevo])], ignore_index=True)
        save_data_den(df2, DATA_FILE_DEN)
        set_last_saved_den({"dens_s": float(dens_s), "pct": float(pct_comp), "delta": float(delta_h), "ventana": str(ventana), "estado": str(estado)})
        st.session_state["guardado_exitoso"] = "El registro se guardo exitosamente"
        if  not bool(st.session_state.get("den_keep", DEFAULT_KEEP_VALUES)):
            st.session_state["DEN_EDIT_ID"] = None
            st.session_state["DEN_EDIT_ROWKEY"] = None
            reset_form_den(clear_last_saved=False)
            st.session_state["RESET_DEN_PENDING"] = True
        st.rerun()





# ====================================== GUARDAR CAMBIOS ==============================================================
    if guardar_cambios:
        rowkey = st.session_state.get("DEN_EDIT_ROWKEY")
        rid = st.session_state.get("DEN_EDIT_ID")
        if not rowkey or not rid:
            st.warning("Primero carga un ID para editar (arriba: Editar ID → Cargar).")
            st.stop()

        errs = validate_den_common()
        if errs:
            for e in errs:
                st.error(e)
            st.stop()

        capa = parse_int(capa_txt)
        espesor_cm = parse_float_loose(esp_txt)
        dm_ini = parse_float_loose(dm_ini_txt)
        dm_ter = parse_float_loose(dm_ter_txt)
        dm_control = parse_float_loose(dm_ctrl_txt)
        cota = parse_float_loose(cota_txt)
        coord_n = parse_float_loose(coord_n_txt)
        coord_e = parse_float_loose(coord_e_txt)
        prof_cm = parse_float_loose(prof_txt)

        dens_s = calc_densidad_seca(float(dens_h_v), float(hum_v))
        pct_comp = calc_pct_comp(float(dens_s), float(dmcs_v))
        delta_h = float(hum_v) - float(hopt_v)
        ventana = "OK" if abs(delta_h) <= float(st.session_state["TOL_HUM_OPT"]) else "OBSERVADO"
        UMBRAL_A_eff = float(st.session_state["UMBRAL_A"])
        UMBRAL_O_eff = adjust_umbral_obs(float(st.session_state["UMBRAL_A"]), float(st.session_state["UMBRAL_O_RAW"]))
        estado = estado_qaqc_den(float(pct_comp), float(UMBRAL_A_eff), float(UMBRAL_O_eff))

        newvals = {
            "Codigo_Proyecto": codigo_proy,
            "Proyecto": proyecto,
            "N_Registro": n_registro,
            "N_Control": n_control,
            "N_Acta": n_acta,
            "Fecha_control": pd.to_datetime(fecha_ctrl),
            "Sector_Zona": sector_final,
            "Tramo": tramo_final,
            "Frente_Tramo": frente,
            "Capa_N": float(capa) if capa is not None else np.nan,
            "Espesor_capa_cm": float(espesor_cm) if espesor_cm is not None else np.nan,
            "Dm_inicio": float(dm_ini) if dm_ini is not None else np.nan,
            "Dm_termino": float(dm_ter) if dm_ter is not None else np.nan,
            "Dm_Control": float(dm_control) if dm_control is not None else np.nan,
            "Coordenada_Norte": float(coord_n) if coord_n is not None else np.nan,
            "Coordenada_Este": float(coord_e) if coord_e is not None else np.nan,
            "Cota": float(cota) if cota is not None else np.nan,
            "Operador": operador,
            "Metodo": metodo_final,
            "Profundidad_cm": float(prof_cm) if prof_cm is not None else np.nan,
            "Densidad_Humeda_gcm3": float(dens_h_v),
            "Humedad_medida_pct": float(hum_v),
            "Humedad_Optima_pct": float(hopt_v),
            "Delta_Humedad_pct": float(delta_h),
            "Ventana_Humedad": str(ventana),
            "Densidad_Seca_gcm3": float(dens_s),
            "DMCS_Proctor_gcm3": float(dmcs_v),
            "pct_Compactacion": float(pct_comp),
            "Umbral_Cumple_pct": float(UMBRAL_A_eff),
            "Umbral_Observado_pct": float(UMBRAL_O_eff),
            "Estado_QAQC": str(estado),
            "Observacion": str(observacion).strip(),
            "Timestamp": pd.to_datetime(datetime.now()),
        }


        df_now = load_data_den(DATA_FILE_DEN)
        df_upd, ok = apply_update_by_rowkey_den(df_now, str(rowkey), newvals)
        if not ok:
            st.error("No pude actualizar: RowKey no encontrado.")
            st.stop()

        save_data_den(df_upd, DATA_FILE_DEN)
        st.session_state["guardado_exitoso"] = f"ID {int(rid)} actualizado"
        if not bool(st.session_state.get("den_keep", DEFAULT_KEEP_VALUES)):
            st.session_state["DEN_EDIT_ID"] = None
            st.session_state["DEN_EDIT_ROWKEY"] = None
            reset_form_den(clear_last_saved=False)
            st.session_state["RESET_DEN_PENDING"] = True
        st.rerun()

    # ELIMINAR (P1)
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("### 🗑️ Eliminar registros (Densidades)")
    df_all_del = load_data_den(DATA_FILE_DEN)
    ids_all_del = sorted(df_all_del["ID_Registro"].dropna().astype(int).unique().tolist()) if not df_all_del.empty else []
    del_ids = st.multiselect("IDs a eliminar", options=ids_all_del, default=[], key="den_del_ids")
    cdel1, cdel2 = st.columns([1, 3])
    with cdel1:
        do_del = st.button("🗑️ Eliminar seleccionados", width='stretch')
    with cdel2:
        st.caption("Borra por ID_Registro (visible).")

    if do_del:
        df_now = load_data_den(DATA_FILE_DEN)
        df_new, n = delete_by_ids_den(df_now, del_ids)
        save_data_den(df_new, DATA_FILE_DEN)
        clear_widget_key("den_del_ids")
        st.session_state["guardado_exitoso"] = f"Eliminados: {n}"
        st.session_state["DEN_EDIT_ID"] = None
        st.session_state["DEN_EDIT_ROWKEY"] = None
        reset_form_den(clear_last_saved=False)
        st.session_state["RESET_DEN_PENDING"] = True
        st.rerun()

    # TABLA + EXPORT (P1)
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("### 📋 Base de datos (Densidades)")
    df_show = load_data_den(DATA_FILE_DEN).sort_values(["Fecha_control", "ID_Registro"], ascending=[False, False])
    show_cols = [
        "ID_Registro", "Fecha_control", "Codigo_Proyecto", "Proyecto", "Sector_Zona", "Tramo", "Operador",
        "Metodo", "Densidad_Seca_gcm3", "pct_Compactacion", "Estado_QAQC", "Humedad_medida_pct", "Humedad_Optima_pct",
        "Delta_Humedad_pct", "DMCS_Proctor_gcm3", "Observacion"
    ]
    df_view = df_show[show_cols].copy() if not df_show.empty else pd.DataFrame(columns=show_cols)
    st.dataframe(style_table_den(df_view), width='stretch', height=360)

    df_kpi, _ = compute_kpis_den(df_show if not df_show.empty else pd.DataFrame(columns=COLUMNS_DEN))
    xbytes = export_excel_bytes(df_show, df_kpi) # puede venir en CSV o En xlsx
    st.download_button(
     "⬇️ Exportar Excel (Datos+KPIs)",
       data=xbytes,
       file_name=f"QINTEGRITY_DENSIDADES.{'xlsx' if HAS_OPENPYXL else 'csv'}", # A
       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# =========================================================
# ===================  PANTALLA 2 DENSIDADES  ==============
# =========================================================
if st.session_state["APP_PAGE"] == "DEN_P2":
    st.caption("Densidades · Pantalla 2 · KPIs + Gráficos + Eliminar + Export")
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    # MOSTRAR MENSAJE DE BORRADO PENDIENTE
    if "eliminado_exitoso" in st.session_state:
        st.toast(st.session_state.get("eliminado_exitoso"),icon="✅")
        del st.session_state["eliminado_exitoso"]


    df_all = load_data_den(DATA_FILE_DEN)
    if df_all.empty:
        st.warning("No hay registros aún.")
    else:
        dmin0, dmax0 = safe_date_bounds(df_all["Fecha_control"])

        f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.6, 1.6])
        with f1:
            dmin = st.date_input("Desde", value=dmin0, key="den2_dmin")
        with f2:
            dmax = st.date_input("Hasta", value=dmax0, key="den2_dmax")
        with f3:
            cods = sorted(df_all["Codigo_Proyecto"].dropna().astype(str).unique().tolist())
            cod_sel = st.multiselect("Código Proyecto", options=cods, default=cods[:1] if len(cods) else [], key="den2_cod")
        with f4:
            ops = sorted(df_all["Operador"].dropna().astype(str).unique().tolist())
            op_sel = st.multiselect("Operador", options=ops, default=[], key="den2_op")

        df_f = df_all.copy()
        df_f = df_f[df_f["Fecha_control"].notna()].copy()
        df_f = df_f[(df_f["Fecha_control"].dt.date >= dmin) & (df_f["Fecha_control"].dt.date <= dmax)].copy()
        if cod_sel:
            df_f = df_f[df_f["Codigo_Proyecto"].astype(str).isin([str(x) for x in cod_sel])].copy()
        if op_sel:
            df_f = df_f[df_f["Operador"].astype(str).isin([str(x) for x in op_sel])].copy()

        df_kpi, k = compute_kpis_den(df_f)
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi_card("Total", str(k.get("total", 0)))
        with k2:
            kpi_card("Cumple", str(k.get("a", 0)))
        with k3:
            kpi_card("Observado", str(k.get("o", 0)))
        with k4:
            kpi_card("No cumple", str(k.get("r", 0)))

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("#### % Compactación (histograma)")
            fig = plt.figure(figsize=(FIG_W, FIG_H))
            ax = fig.add_subplot(111)
            vals = df_f["pct_Compactacion"].dropna().astype(float).values
            if len(vals):
                ax.hist(vals, bins=12)
                ax.set_xlabel("% Compactación")
                ax.set_ylabel("Frecuencia")
            st.pyplot(fig, width='stretch')
        with g2:
            st.markdown("#### Estados QA/QC (conteo)")
            fig = plt.figure(figsize=(FIG_W, FIG_H))
            ax = fig.add_subplot(111)
            c = df_f["Estado_QAQC"].astype(str).str.upper().value_counts()
            if len(c):
                ax.bar(c.index.tolist(), c.values.tolist())
                ax.set_xlabel("Estado")
                ax.set_ylabel("Cantidad")
            st.pyplot(fig, width='stretch')

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("### 🗑️ Eliminar (Densidades)")
        ids = sorted(df_f["ID_Registro"].dropna().astype(int).unique().tolist())
        del_ids2 = st.multiselect("IDs filtrados a eliminar", options=ids, default=[], key="den2_del")
        if st.button("🗑️ Eliminar seleccionados (de la base)", width='stretch'):
            df_now = load_data_den(DATA_FILE_DEN)
            df_new, n = delete_by_ids_den(df_now, del_ids2)
            save_data_den(df_new, DATA_FILE_DEN)
            clear_widget_key("den2_del")
            st.session_state["eliminado_exitoso"] = f"Registros eliminados: {n}"
            st.rerun()

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("### 📥 Export")
        xbytes = export_excel_bytes(df_f, df_kpi)
        st.download_button(
            "⬇️ Exportar Excel filtrado (Datos+KPIs)",
            data=xbytes,
            file_name=f"QINTEGRITY_DENSIDADES_FILTRADO.{'xlsx' if HAS_OPENPYXL else 'csv'}",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("### 📋 Tabla filtrada")
        show_cols = ["ID_Registro", "Fecha_control", "Codigo_Proyecto", "Proyecto", "Sector_Zona", "Tramo", "Operador", "pct_Compactacion", "Estado_QAQC", "Observacion"]
        st.dataframe(style_table_den(df_f[show_cols].copy()), width='stretch', height=360)


# =========================================================
# ===================  PIE m² – INGRESO  ===================
# =========================================================

def _pie_m2_apply_pending_reset():
    """Reset DIFERIDO para no disparar: cannot be modified after widget is instantiated."""
    if st.session_state.get("_PIE2_PENDING_RESET", False):
        defaults = {
            "pie2_fecha": date.today(),
            "pie2_cod": "",
            "pie2_docid_eett": "",
            "pie2_sector": "",
            "pie2_frente": "",
            "pie2_dm_ini": "",
            "pie2_dm_ter": "",
            "pie2_largo": "",
            "pie2_ancho": "",
            "pie2_valor": "",
            "pie2_ejec": "",
            "PIE2_EDIT_ID": None,
            "PIE2_EDIT_ROWKEY": None,
            "PIE2_EDIT_PICK": None,
        }
        for k, v in defaults.items():
            st.session_state[k] = v
        st.session_state["_PIE2_PENDING_RESET"] = False


def pie_m2_reset_form():
    """Marca reset para el próximo run (seguro con widgets)."""
    st.session_state["_PIE2_PENDING_RESET"] = True
    st.rerun()


def pie_m2_load_record_into_form(row: pd.Series):
    st.session_state["PIE2_EDIT_ID"] = int(row.get("ID_Registro"))
    st.session_state["PIE2_EDIT_ROWKEY"] = str(row.get("RowKey"))

    st.session_state["pie2_fecha"] = row.get("Fecha").date() if pd.notna(row.get("Fecha")) else date.today()
    st.session_state["pie2_cod"] = str(row.get("COD_PROYECTO") or "")
    st.session_state["pie2_docid_eett"] = str(row.get("DocID_EETT") or "")
    st.session_state["pie2_sector"] = str(row.get("Sector_Zona") or "")
    st.session_state["pie2_frente"] = str(row.get("Frente_Tramo") or "")

    st.session_state["pie2_dm_ini"] = "" if pd.isna(row.get("DM_inicio")) else str(float(row.get("DM_inicio")))
    st.session_state["pie2_dm_ter"] = "" if pd.isna(row.get("DM_termino")) else str(float(row.get("DM_termino")))
    st.session_state["pie2_largo"] = "" if pd.isna(row.get("Largo_Tramo_m")) else str(float(row.get("Largo_Tramo_m")))
    st.session_state["pie2_ancho"] = "" if pd.isna(row.get("Ancho_m")) else str(float(row.get("Ancho_m")))
    st.session_state["pie2_valor"] = "" if pd.isna(row.get("PIE_VALOR_m2_por_ensayo")) else str(float(row.get("PIE_VALOR_m2_por_ensayo")))
    st.session_state["pie2_ejec"] = "" if pd.isna(row.get("Ejecutadas")) else str(float(row.get("Ejecutadas")))


#================================================                ================================
#============================================       PANTALLA 3     ================================
#================================================                ================================
if st.session_state["APP_PAGE"] == "PIE_M2_P1":
    _pie_m2_apply_pending_reset()

    st.caption("Control PIE m² · Pantalla 3 · Ingreso / Editar / Eliminar + Export")
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    # MOSTRAR MENSAJES PENDIENTES (GUARDADO,MODIFICADO,BORRADO)
    if "operacion_exitosa" in st.session_state:
        st.toast(st.session_state.get("operacion_exitosa"), icon="✅")
        del st.session_state["operacion_exitosa"]

    df_all0 = load_data_generic(
        DATA_FILE_PIE_M2,
        COLUMNS_PIE_M2,
        date_cols=["Fecha"],
        num_cols=[
            "DM_inicio", "DM_termino", "Largo_Tramo_m", "Ancho_m",
            "Area_m2", "PIE_VALOR_m2_por_ensayo",
            "Requeridas", "Ejecutadas", "Brecha", "Pct_Cumpl",
        ],
    )
    ids_all0 = sorted(df_all0["ID_Registro"].dropna().astype(int).unique().tolist()) if not df_all0.empty else []

    t1, t2, t3 = st.columns([1.2, 1.2, 2.6])
    with t1:
        if st.button("🧹 LIMPIAR", width='stretch'):
            pie_m2_reset_form()

    with t2:
        edit_id = st.selectbox("✏️ Editar ID", options=[None] + ids_all0, index=0, key="PIE2_EDIT_PICK")
        if edit_id is not None:
            if st.button("✏️ Cargar ID", width='stretch'):
                row = get_record_by_id_generic(df_all0, int(edit_id))
                if row is None:
                    st.error("No encontré ese ID.")
                else:
                    pie_m2_load_record_into_form(row)
                    st.success(f"ID {int(edit_id)} cargado.")
                    st.rerun()

    with t3:
        if st.session_state.get("PIE2_EDIT_ID") and st.session_state.get("PIE2_EDIT_ROWKEY"):
            st.info(f"Editando ID={st.session_state['PIE2_EDIT_ID']} ✅")
        else:
            st.info("Modo nuevo registro ✅")

    st.markdown("<div class='qi-section'><div class='qi-h3'>Ingreso Control PIE m²</div>", unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        pie_fecha = st.date_input("Fecha",  key="pie2_fecha")
        pie_cod = st.text_input("COD_PROYECTO",  key="pie2_cod").strip()
        pie_docid_eett = eett_doc_selector("PIE2", "pie2_docid_eett", label="Documento Técnico (EETT)")

    with p2:
        pie_sector = st.text_input("Sector/Zona", key="pie2_sector").strip()
        pie_frente = st.text_input("Frente/Tramo",  key="pie2_frente").strip()

    with p3:
        pie_dm_ini = st.text_input("DM inicio",  key="pie2_dm_ini")
        pie_dm_ter = st.text_input("DM término",  key="pie2_dm_ter")

    with p4:
        pie_largo = st.text_input("Largo tramo (m)",  key="pie2_largo")
        pie_ancho = st.text_input("Ancho (m)",  key="pie2_ancho")

    q1, q2 = st.columns(2)
    with q1:
        pie_valor = st.text_input("PIE (m²/ensayo)",  key="pie2_valor")
    with q2:
        pie_ejec = st.text_input("Ensayos ejecutados",  key="pie2_ejec")

    largo_v = parse_float_loose(pie_largo)
    ancho_v = parse_float_loose(pie_ancho)
    pie_v = parse_float_loose(pie_valor)
    eje_v = parse_float_loose(pie_ejec)
    calc = pie_calc_m2(largo_v, ancho_v, pie_v, eje_v)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Área (m²)", f"{calc['base']:.2f}" if pd.notna(calc["base"]) else "—")
    with k2:
        kpi_card("Requeridas", f"{calc['requeridas']:.0f}" if pd.notna(calc["requeridas"]) else "—")
    with k3:
        kpi_card("Ejecutadas", f"{float(eje_v):.0f}" if eje_v is not None else "0")
    with k4:
        kpi_card("Estado", str(calc["estado"]))

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    bsave1, bsave2, bsave3 = st.columns([1.3, 1.3, 3.4])
    with bsave1:
        pie_save = st.button("💾 Guardar PIE m²", type="primary", width='stretch')
    with bsave2:
        pie_save_edit = st.button("💾 Guardar cambios PIE m²", width='stretch')
    with bsave3:
        st.caption(f"Guardado en {DATA_FILE_PIE_M2}")

    def validate_pie_m2() -> List[str]:
        errs = []
        if not pie_cod:
            errs.append("⚠️ Falta COD_PROYECTO.")
        if not pie_sector:
            errs.append("⚠️ Falta Sector/Zona.")
        if not pie_frente:
            errs.append("⚠️ Falta Frente/Tramo.")
        if largo_v is None:
            errs.append("⚠️ Largo inválido.")
        if ancho_v is None:
            errs.append("⚠️ Ancho inválido.")
        if pie_v is None or pie_v <= 0:
            errs.append("⚠️ PIE inválido (>0).")
        if eje_v is None:
            errs.append("⚠️ Ejecutadas inválido.")
        return errs

    if pie_save:
        errs = validate_pie_m2()
        if errs:
            for e in errs:
                st.error(e)
            st.stop()

        df_now = df_all0
        new_id = int(next_id_generic(df_now))

        dm_ini_v = parse_float_loose(pie_dm_ini)
        dm_ter_v = parse_float_loose(pie_dm_ter)

        nuevo = {
            "RowKey": _safe_uuid(),
            "ID_Registro": new_id,
            "Fecha": pd.to_datetime(pie_fecha),
            "COD_PROYECTO": pie_cod,
            "DocID_EETT": pie_docid_eett,  # ✅ NUEVO
            "Sector_Zona": pie_sector,
            "Frente_Tramo": pie_frente,
            "DM_inicio": float(dm_ini_v) if dm_ini_v is not None else np.nan,
            "DM_termino": float(dm_ter_v) if dm_ter_v is not None else np.nan,
            "Largo_Tramo_m": float(largo_v),
            "Ancho_m": float(ancho_v),
            "Area_m2": float(calc["base"]) if pd.notna(calc["base"]) else np.nan,
            "PIE_VALOR_m2_por_ensayo": float(pie_v),
            "Requeridas": float(calc["requeridas"]) if pd.notna(calc["requeridas"]) else np.nan,
            "Ejecutadas": float(eje_v),
            "Brecha": float(calc["brecha"]) if pd.notna(calc["brecha"]) else np.nan,
            "Pct_Cumpl": float(calc["pct"]) if pd.notna(calc["pct"]) else np.nan,
            "Estado": str(calc["estado"]),
            "Timestamp": pd.to_datetime(datetime.now()),
        }

        df2 = pd.concat([df_now, pd.DataFrame([nuevo])], ignore_index=True)
        save_data_generic(df2, DATA_FILE_PIE_M2, COLUMNS_PIE_M2)
        st.session_state["operacion_exitosa"] = "PIE m² guardado"
        pie_m2_reset_form()
        st.rerun()

    if pie_save_edit:
        rowkey = st.session_state.get("PIE2_EDIT_ROWKEY")
        rid = st.session_state.get("PIE2_EDIT_ID")
        if not rowkey or not rid:
            st.warning("Primero carga un ID para editar.")
            st.stop()

        errs = validate_pie_m2()
        if errs:
            for e in errs:
                st.error(e)
            st.stop()

        dm_ini_v = parse_float_loose(pie_dm_ini)
        dm_ter_v = parse_float_loose(pie_dm_ter)

        newvals = {
            "Fecha": pd.to_datetime(pie_fecha),
            "COD_PROYECTO": pie_cod,
            "DocID_EETT": pie_docid_eett,  # ✅ NUEVO
            "Sector_Zona": pie_sector,
            "Frente_Tramo": pie_frente,
            "DM_inicio": float(dm_ini_v) if dm_ini_v is not None else np.nan,
            "DM_termino": float(dm_ter_v) if dm_ter_v is not None else np.nan,
            "Largo_Tramo_m": float(largo_v),
            "Ancho_m": float(ancho_v),
            "Area_m2": float(calc["base"]) if pd.notna(calc["base"]) else np.nan,
            "PIE_VALOR_m2_por_ensayo": float(pie_v),
            "Requeridas": float(calc["requeridas"]) if pd.notna(calc["requeridas"]) else np.nan,
            "Ejecutadas": float(eje_v),
            "Brecha": float(calc["brecha"]) if pd.notna(calc["brecha"]) else np.nan,
            "Pct_Cumpl": float(calc["pct"]) if pd.notna(calc["pct"]) else np.nan,
            "Estado": str(calc["estado"]),
            "Timestamp": pd.to_datetime(datetime.now()),
        }

        # Recargar para editar sobre lo último
        df_now = load_data_generic(
            DATA_FILE_PIE_M2,
            COLUMNS_PIE_M2,
            date_cols=["Fecha"],
            num_cols=[
                "DM_inicio", "DM_termino", "Largo_Tramo_m", "Ancho_m",
                "Area_m2", "PIE_VALOR_m2_por_ensayo",
                "Requeridas", "Ejecutadas", "Brecha", "Pct_Cumpl",
            ],
        )
        df_upd, ok = apply_update_by_rowkey_generic(df_now, str(rowkey), newvals)
        if not ok:
            st.error("No pude actualizar: RowKey no encontrado.")
            st.stop()

        save_data_generic(df_upd, DATA_FILE_PIE_M2, COLUMNS_PIE_M2)
        st.session_state["operacion_exitosa"] = f"PIE m² ID {int(rid)} actualizado ✅"
        st.rerun()

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("### 🗑️ Eliminar PIE m²")
    df_now = load_data_generic(
        DATA_FILE_PIE_M2,
        COLUMNS_PIE_M2,
        date_cols=["Fecha"],
        num_cols=[
            "DM_inicio", "DM_termino", "Largo_Tramo_m", "Ancho_m",
            "Area_m2", "PIE_VALOR_m2_por_ensayo",
            "Requeridas", "Ejecutadas", "Brecha", "Pct_Cumpl",
        ],
    )
    ids = sorted(df_now["ID_Registro"].dropna().astype(int).unique().tolist()) if not df_now.empty else []
    del_ids = st.multiselect("IDs a eliminar", options=ids, default=[], key="pie2_del_ids")
    if st.button("🗑️ Eliminar PIE m² seleccionados", width='stretch'):
        df_new, n = delete_by_ids_generic(df_now, del_ids)
        save_data_generic(df_new, DATA_FILE_PIE_M2, COLUMNS_PIE_M2)
        clear_widget_key("pie2_del_ids")
        st.success(f"Eliminados: {n}")
        st.session_state["operacion_exitosa"] = f"Registros eliminados: {int(n)}"
        st.rerun()

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("### 📋 Base PIE m²")
    df_show = df_now.sort_values(["Fecha", "ID_Registro"], ascending=[False, False])

    # 👇 Mostrar DocID_EETT en la tabla (si quieres, lo sacas después)
    view_cols = [
        "ID_Registro", "Fecha", "COD_PROYECTO", "DocID_EETT",
        "Sector_Zona", "Frente_Tramo",
        "Area_m2", "PIE_VALOR_m2_por_ensayo", "Requeridas", "Ejecutadas",
        "Brecha", "Pct_Cumpl", "Estado",
    ]
    df_view = df_show[view_cols].copy() if not df_show.empty else pd.DataFrame(columns=view_cols)
    st.dataframe(style_table_pie(df_view), width='stretch', height=360)

    df_kpi, _ = compute_kpis_pie(df_show, base_col="Area_m2")
    xbytes = export_excel_bytes(df_show, df_kpi)
    st.download_button(
        "⬇️ Exportar Excel PIE m² (Datos+KPIs)",
        data=xbytes,
        file_name=f"QINTEGRITY_PIE_M2.{'xlsx' if HAS_OPENPYXL else 'csv'}",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# =========================================================
# ===================  PIE m² – KPIs  ======================
# =========================================================
if st.session_state["APP_PAGE"] == "PIE_M2_P2":
    st.caption("Control PIE m² · Pantalla 4 · KPIs + Gráficos + Export")
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    df_all = load_data_generic(
        DATA_FILE_PIE_M2,
        COLUMNS_PIE_M2,
        date_cols=["Fecha"],
        num_cols=["DM_inicio", "DM_termino", "Largo_Tramo_m", "Ancho_m", "Area_m2", "PIE_VALOR_m2_por_ensayo", "Requeridas", "Ejecutadas", "Brecha", "Pct_Cumpl"],
    )
    if df_all.empty:
        st.warning("No hay registros PIE m² aún.")
    else:
        dmin0, dmax0 = safe_date_bounds(df_all["Fecha"])

        f1, f2, f3 = st.columns([1.2, 1.2, 2.6])
        with f1:
            dmin = st.date_input("Desde", value=dmin0, key="pie2_dmin")
        with f2:
            dmax = st.date_input("Hasta", value=dmax0, key="pie2_dmax")
        with f3:
            cods = sorted(df_all["COD_PROYECTO"].dropna().astype(str).unique().tolist())
            cod_sel = st.multiselect("COD_PROYECTO", options=cods, default=cods[:1] if len(cods) else [], key="pie2_codf")

        df_f = df_all.copy()
        df_f = df_f[df_f["Fecha"].notna()].copy()
        df_f = df_f[(df_f["Fecha"].dt.date >= dmin) & (df_f["Fecha"].dt.date <= dmax)].copy()
        if cod_sel:
            df_f = df_f[df_f["COD_PROYECTO"].astype(str).isin([str(x) for x in cod_sel])].copy()

        df_kpi, k = compute_kpis_pie(df_f, base_col="Area_m2")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi_card("Tramos", str(k.get("total", 0)))
        with k2:
            kpi_card("Área total (m²)", f"{k.get('base_total', 0.0):.2f}")
        with k3:
            kpi_card("Req total", f"{k.get('req_total', 0.0):.0f}")
        with k4:
            kpi_card("Cumpl (%)", f"{k.get('pct_global', 0.0):.1f}%")

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("#### Estado (conteo)")
            fig = plt.figure(figsize=(FIG_W, FIG_H))
            ax = fig.add_subplot(111)
            c = df_f["Estado"].astype(str).str.upper().value_counts()
            if len(c):
                ax.bar(c.index.tolist(), c.values.tolist())
                ax.set_xlabel("Estado")
                ax.set_ylabel("Cantidad")
            st.pyplot(fig, width='stretch')

        with g2:
            st.markdown("#### Cumplimiento % (hist)")
            fig = plt.figure(figsize=(FIG_W, FIG_H))
            ax = fig.add_subplot(111)
            vals = df_f["Pct_Cumpl"].dropna().astype(float).values
            if len(vals):
                ax.hist(vals, bins=12)
                ax.set_xlabel("% Cumpl")
                ax.set_ylabel("Frecuencia")
            st.pyplot(fig, width='stretch')

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("### 📥 Export KPI PIE m²")
        xbytes = export_excel_bytes(df_f, df_kpi)
        st.download_button(
            "⬇️ Exportar Excel PIE m² filtrado",
            data=xbytes,
            file_name=f"QINTEGRITY_PIE_M2_FILTRADO.{'xlsx' if HAS_OPENPYXL else 'csv'}",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("### 📋 Tabla filtrada")
        view_cols = ["ID_Registro", "Fecha", "COD_PROYECTO", "Sector_Zona", "Frente_Tramo", "Area_m2", "Requeridas", "Ejecutadas", "Brecha", "Pct_Cumpl", "Estado"]
        st.dataframe(style_table_pie(df_f[view_cols].copy()), width='stretch', height=360)


# =========================================================
# ===================  PIE m³ – INGRESO  ===================
# =========================================================
def _pie_m3_apply_pending_reset():
    """Reset DIFERIDO para no disparar: cannot be modified after widget is instantiated."""
    if st.session_state.get("_PIE3_PENDING_RESET", False):
        defaults = {
            "pie3_fecha": date.today(),
            "pie3_cod": "",
            "pie3_docid_eett": "",
            "pie3_sector": "",
            "pie3_frente": "",
            "pie3_dm_ini": "",
            "pie3_dm_ter": "",
            "pie3_largo": "",
            "pie3_ancho": "",
            "pie3_esp": "",
            "pie3_valor": "",
            "pie3_ejec": "",
            "PIE3_EDIT_ID": None,
            "PIE3_EDIT_ROWKEY": None,
        }
        for k, v in defaults.items():
            st.session_state[k] = v
        st.session_state["_PIE3_PENDING_RESET"] = False

    # MOSTRAR MENSAJES DE OPERACIONES PENDIENTES
    if "operacion_p3" in st.session_state and not  st.session_state.get("_PIE3_PENDING_RESET", False):
        st.toast(st.session_state["operacion_p3"],icon="✅")
        del st.session_state["operacion_p3"]



def pie_m3_reset_form():
    """Marca reset para el próximo run (seguro con widgets)."""
    st.session_state["_PIE3_PENDING_RESET"] = True
    st.rerun()


def pie_m3_load_record_into_form(row: pd.Series):
    st.session_state["PIE3_EDIT_ID"] = int(row.get("ID_Registro"))
    st.session_state["PIE3_EDIT_ROWKEY"] = str(row.get("RowKey"))

    st.session_state["pie3_fecha"] = row.get("Fecha").date() if pd.notna(row.get("Fecha")) else date.today()
    st.session_state["pie3_cod"] = str(row.get("COD_PROYECTO") or "")
    st.session_state["pie3_sector"] = str(row.get("Sector_Zona") or "")
    st.session_state["pie3_frente"] = str(row.get("Frente_Tramo") or "")

    st.session_state["pie3_dm_ini"] = "" if pd.isna(row.get("DM_inicio")) else str(float(row.get("DM_inicio")))
    st.session_state["pie3_dm_ter"] = "" if pd.isna(row.get("DM_termino")) else str(float(row.get("DM_termino")))
    st.session_state["pie3_largo"] = "" if pd.isna(row.get("Largo_Tramo_m")) else str(float(row.get("Largo_Tramo_m")))
    st.session_state["pie3_ancho"] = "" if pd.isna(row.get("Ancho_m")) else str(float(row.get("Ancho_m")))
    st.session_state["pie3_esp"] = "" if pd.isna(row.get("Espesor_m")) else str(float(row.get("Espesor_m")))
    st.session_state["pie3_valor"] = "" if pd.isna(row.get("PIE_VALOR_m3_por_ensayo")) else str(float(row.get("PIE_VALOR_m3_por_ensayo")))
    st.session_state["pie3_ejec"] = "" if pd.isna(row.get("Ejecutadas")) else str(float(row.get("Ejecutadas")))


if st.session_state["APP_PAGE"] == "PIE_M3_P1":
    _pie_m3_apply_pending_reset()

    st.caption("Control PIE m³ · Pantalla 5 · Ingreso/Editar/Eliminar + Export")
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    df_all0 = load_data_generic(
        DATA_FILE_PIE_M3,
        COLUMNS_PIE_M3,
        date_cols=["Fecha"],
        num_cols=["DM_inicio", "DM_termino", "Largo_Tramo_m", "Ancho_m", "Espesor_m", "Volumen_m3", "PIE_VALOR_m3_por_ensayo", "Requeridas", "Ejecutadas", "Brecha", "Pct_Cumpl"],
    )
    ids_all0 = sorted(df_all0["ID_Registro"].dropna().astype(int).unique().tolist()) if not df_all0.empty else []

    t1, t2, t3 = st.columns([1.2, 1.2, 2.6])
    with t1:
        if st.button("🧹 LIMPIAR", width='stretch'):
            pie_m3_reset_form()
            st.rerun()
    with t2:
        edit_id = st.selectbox("✏️ Editar ID", options=[None] + ids_all0, index=0, key="PIE3_EDIT_PICK")
        if edit_id is not None:
            if st.button("✏️ Cargar ID", width='stretch'):
                row = get_record_by_id_generic(df_all0, int(edit_id))
                if row is None:
                    st.error("No encontré ese ID.")
                else:
                    pie_m3_load_record_into_form(row)
                    st.success(f"ID {int(edit_id)} cargado.")
                    st.rerun()
    with t3:
        if st.session_state.get("PIE3_EDIT_ID") and st.session_state.get("PIE3_EDIT_ROWKEY"):
            st.info(f"Editando ID={st.session_state['PIE3_EDIT_ID']} ✅")
        else:
            st.info("Modo nuevo registro ✅")

    st.markdown("<div class='qi-section'><div class='qi-h3'>Ingreso Control PIE m³</div>", unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        pie_fecha = st.date_input("Fecha",  key="pie3_fecha")
        pie_cod = st.text_input("COD_PROYECTO", key="pie3_cod").strip()
    with p2:
        pie_sector = st.text_input("Sector/Zona", key="pie3_sector").strip()
        pie_frente = st.text_input("Frente/Tramo", key="pie3_frente").strip()
    with p3:
        pie_dm_ini = st.text_input("DM inicio",  key="pie3_dm_ini")
        pie_dm_ter = st.text_input("DM término", key="pie3_dm_ter")
    with p4:
        pie_largo = st.text_input("Largo tramo (m)",  key="pie3_largo")
        pie_ancho = st.text_input("Ancho (m)",  key="pie3_ancho")

    q1, q2, q3 = st.columns(3)
    with q1:
        pie_esp = st.text_input("Espesor (m)",  key="pie3_esp")
    with q2:
        pie_valor = st.text_input("PIE (m³/ensayo)", key="pie3_valor")
    with q3:
        pie_ejec = st.text_input("Ensayos ejecutados", key="pie3_ejec")

    largo_v = parse_float_loose(pie_largo)
    ancho_v = parse_float_loose(pie_ancho)
    esp_v = parse_float_loose(pie_esp)
    pie_v = parse_float_loose(pie_valor)
    eje_v = parse_float_loose(pie_ejec)

    calc = pie_calc_m3(largo_v, ancho_v, esp_v, pie_v, eje_v)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Volumen (m³)", f"{calc['base']:.3f}" if pd.notna(calc["base"]) else "—")
    with k2:
        kpi_card("Requeridas", f"{calc['requeridas']:.0f}" if pd.notna(calc["requeridas"]) else "—")
    with k3:
        kpi_card("Ejecutadas", f"{float(eje_v):.0f}" if eje_v is not None else "0")
    with k4:
        kpi_card("Estado", str(calc["estado"]))

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    bsave1, bsave2, bsave3 = st.columns([1.3, 1.3, 3.4])
    with bsave1:
        pie_save = st.button("💾 Guardar PIE m³", type="primary", width='stretch')
    with bsave2:
        pie_save_edit = st.button("💾 Guardar cambios PIE m³", width='stretch')
    with bsave3:
        st.caption(f"Guardado en {DATA_FILE_PIE_M3}")

    def validate_pie_m3() -> List[str]:
        errs = []
        if not pie_cod:
            errs.append("⚠️ Falta COD_PROYECTO.")
        if not pie_sector:
            errs.append("⚠️ Falta Sector/Zona.")
        if not pie_frente:
            errs.append("⚠️ Falta Frente/Tramo.")
        if largo_v is None:
            errs.append("⚠️ Largo inválido.")
        if ancho_v is None:
            errs.append("⚠️ Ancho inválido.")
        if esp_v is None:
            errs.append("⚠️ Espesor inválido.")
        if pie_v is None or pie_v <= 0:
            errs.append("⚠️ PIE inválido (>0).")
        if eje_v is None:
            errs.append("⚠️ Ejecutadas inválido.")
        return errs

    if pie_save:
        errs = validate_pie_m3()
        if errs:
            for e in errs:
                st.error(e)
            st.stop()

        df_now = df_all0
        new_id = int(next_id_generic(df_now))

        dm_ini_v = parse_float_loose(pie_dm_ini)
        dm_ter_v = parse_float_loose(pie_dm_ter)

        nuevo = {
            "RowKey": _safe_uuid(),
            "ID_Registro": new_id,
            "Fecha": pd.to_datetime(pie_fecha),
            "COD_PROYECTO": pie_cod,
            "Sector_Zona": pie_sector,
            "Frente_Tramo": pie_frente,
            "DM_inicio": float(dm_ini_v) if dm_ini_v is not None else np.nan,
            "DM_termino": float(dm_ter_v) if dm_ter_v is not None else np.nan,
            "Largo_Tramo_m": float(largo_v),
            "Ancho_m": float(ancho_v),
            "Espesor_m": float(esp_v),
            "Volumen_m3": float(calc["base"]) if pd.notna(calc["base"]) else np.nan,
            "PIE_VALOR_m3_por_ensayo": float(pie_v),
            "Requeridas": float(calc["requeridas"]) if pd.notna(calc["requeridas"]) else np.nan,
            "Ejecutadas": float(eje_v),
            "Brecha": float(calc["brecha"]) if pd.notna(calc["brecha"]) else np.nan,
            "Pct_Cumpl": float(calc["pct"]) if pd.notna(calc["pct"]) else np.nan,
            "Estado": str(calc["estado"]),
            "Timestamp": pd.to_datetime(datetime.now()),
        }

        df2 = pd.concat([df_now, pd.DataFrame([nuevo])], ignore_index=True)
        save_data_generic(df2, DATA_FILE_PIE_M3, COLUMNS_PIE_M3)
        st.session_state["operacion_p3"] = "Pie m³ guardado con exito"
        pie_m3_reset_form()
        st.rerun()

    if pie_save_edit:
        rowkey = st.session_state.get("PIE3_EDIT_ROWKEY")
        rid = st.session_state.get("PIE3_EDIT_ID")
        if not rowkey or not rid:
            st.warning("Primero carga un ID para editar.")
            st.stop()

        errs = validate_pie_m3()
        if errs:
            for e in errs:
                st.error(e)
            st.stop()

        dm_ini_v = parse_float_loose(pie_dm_ini)
        dm_ter_v = parse_float_loose(pie_dm_ter)

        newvals = {
            "Fecha": pd.to_datetime(pie_fecha),
            "COD_PROYECTO": pie_cod,
            "Sector_Zona": pie_sector,
            "Frente_Tramo": pie_frente,
            "DM_inicio": float(dm_ini_v) if dm_ini_v is not None else np.nan,
            "DM_termino": float(dm_ter_v) if dm_ter_v is not None else np.nan,
            "Largo_Tramo_m": float(largo_v),
            "Ancho_m": float(ancho_v),
            "Espesor_m": float(esp_v),
            "Volumen_m3": float(calc["base"]) if pd.notna(calc["base"]) else np.nan,
            "PIE_VALOR_m3_por_ensayo": float(pie_v),
            "Requeridas": float(calc["requeridas"]) if pd.notna(calc["requeridas"]) else np.nan,
            "Ejecutadas": float(eje_v),
            "Brecha": float(calc["brecha"]) if pd.notna(calc["brecha"]) else np.nan,
            "Pct_Cumpl": float(calc["pct"]) if pd.notna(calc["pct"]) else np.nan,
            "Estado": str(calc["estado"]),
            "Timestamp": pd.to_datetime(datetime.now()),
        }

        df_now = load_data_generic(
            DATA_FILE_PIE_M3,
            COLUMNS_PIE_M3,
            date_cols=["Fecha"],
            num_cols=["DM_inicio", "DM_termino", "Largo_Tramo_m", "Ancho_m", "Espesor_m", "Volumen_m3", "PIE_VALOR_m3_por_ensayo", "Requeridas", "Ejecutadas", "Brecha", "Pct_Cumpl"],
        )
        df_upd, ok = apply_update_by_rowkey_generic(df_now, str(rowkey), newvals)
        if not ok:
            st.error("No pude actualizar: RowKey no encontrado.")
            st.stop()

        save_data_generic(df_upd, DATA_FILE_PIE_M3, COLUMNS_PIE_M3)
        st.session_state["operacion_p3"] = f"PIE m³ ID {int(rid)} actualizado"
        st.rerun()

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("### 🗑️ Eliminar PIE m³")
    df_now = load_data_generic(
        DATA_FILE_PIE_M3,
        COLUMNS_PIE_M3,
        date_cols=["Fecha"],
        num_cols=["DM_inicio", "DM_termino", "Largo_Tramo_m", "Ancho_m", "Espesor_m", "Volumen_m3", "PIE_VALOR_m3_por_ensayo", "Requeridas", "Ejecutadas", "Brecha", "Pct_Cumpl"],
    )
    ids = sorted(df_now["ID_Registro"].dropna().astype(int).unique().tolist()) if not df_now.empty else []
    del_ids = st.multiselect("IDs a eliminar", options=ids, default=[], key="pie3_del_ids")
    if st.button("🗑️ Eliminar PIE m³ seleccionados", width='stretch'):
        df_new, n = delete_by_ids_generic(df_now, del_ids)
        save_data_generic(df_new, DATA_FILE_PIE_M3, COLUMNS_PIE_M3)
        clear_widget_key("pie3_del_ids")
        st.session_state["operacion_p3"] =f"Eliminados: {n}"
        st.rerun()

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("### 📋 Base PIE m³")
    df_show = df_now.sort_values(["Fecha", "ID_Registro"], ascending=[False, False])
    view_cols = ["ID_Registro", "Fecha", "COD_PROYECTO", "Sector_Zona", "Frente_Tramo", "Volumen_m3", "PIE_VALOR_m3_por_ensayo", "Requeridas", "Ejecutadas", "Brecha", "Pct_Cumpl", "Estado"]
    df_view = df_show[view_cols].copy() if not df_show.empty else pd.DataFrame(columns=view_cols)
    st.dataframe(style_table_pie(df_view), width='stretch', height=360)

    df_kpi, _ = compute_kpis_pie(df_show, base_col="Volumen_m3")
    xbytes = export_excel_bytes(df_show, df_kpi)
    st.download_button(
        "⬇️ Exportar Excel PIE m³ (Datos+KPIs)",
        data=xbytes,
        file_name=f"QINTEGRITY_PIE_M3.{'xlsx' if HAS_OPENPYXL else 'csv'}",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# =========================================================
# ===================  PIE m³ – KPIs  ======================
# =========================================================
if st.session_state["APP_PAGE"] == "PIE_M3_P2":
    st.caption("Control PIE m³ · Pantalla 6 · KPIs + Gráficos + Export")
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    df_all = load_data_generic(
        DATA_FILE_PIE_M3,
        COLUMNS_PIE_M3,
        date_cols=["Fecha"],
        num_cols=["DM_inicio", "DM_termino", "Largo_Tramo_m", "Ancho_m", "Espesor_m", "Volumen_m3", "PIE_VALOR_m3_por_ensayo", "Requeridas", "Ejecutadas", "Brecha", "Pct_Cumpl"],
    )
    if df_all.empty:
        st.warning("No hay registros PIE m³ aún.")
    else:
        dmin0, dmax0 = safe_date_bounds(df_all["Fecha"])

        f1, f2, f3 = st.columns([1.2, 1.2, 2.6])
        with f1:
            dmin = st.date_input("Desde", value=dmin0, key="pie3_dmin")
        with f2:
            dmax = st.date_input("Hasta", value=dmax0, key="pie3_dmax")
        with f3:
            cods = sorted(df_all["COD_PROYECTO"].dropna().astype(str).unique().tolist())
            cod_sel = st.multiselect("COD_PROYECTO", options=cods, default=cods[:1] if len(cods) else [], key="pie3_codf")

        df_f = df_all.copy()
        df_f = df_f[df_f["Fecha"].notna()].copy()
        df_f = df_f[(df_f["Fecha"].dt.date >= dmin) & (df_f["Fecha"].dt.date <= dmax)].copy()
        if cod_sel:
            df_f = df_f[df_f["COD_PROYECTO"].astype(str).isin([str(x) for x in cod_sel])].copy()

        df_kpi, k = compute_kpis_pie(df_f, base_col="Volumen_m3")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi_card("Tramos", str(k.get("total", 0)))
        with k2:
            kpi_card("Volumen total (m³)", f"{k.get('base_total', 0.0):.3f}")
        with k3:
            kpi_card("Req total", f"{k.get('req_total', 0.0):.0f}")
        with k4:
            kpi_card("Cumpl (%)", f"{k.get('pct_global', 0.0):.1f}%")

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("#### Estado (conteo)")
            fig = plt.figure(figsize=(FIG_W, FIG_H))
            ax = fig.add_subplot(111)
            c = df_f["Estado"].astype(str).str.upper().value_counts()
            if len(c):
                ax.bar(c.index.tolist(), c.values.tolist())
                ax.set_xlabel("Estado")
                ax.set_ylabel("Cantidad")
            st.pyplot(fig, width='stretch')

        with g2:
            st.markdown("#### Cumplimiento % (hist)")
            fig = plt.figure(figsize=(FIG_W, FIG_H))
            ax = fig.add_subplot(111)
            vals = df_f["Pct_Cumpl"].dropna().astype(float).values
            if len(vals):
                ax.hist(vals, bins=12)
                ax.set_xlabel("% Cumpl")
                ax.set_ylabel("Frecuencia")
            st.pyplot(fig, width='stretch')

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("### 📥 Export KPI PIE m³")
        xbytes = export_excel_bytes(df_f, df_kpi)
        st.download_button(
            "⬇️ Exportar Excel PIE m³ filtrado",
            data=xbytes,
            file_name=f"QINTEGRITY_PIE_M3_FILTRADO.{'xlsx' if HAS_OPENPYXL else 'csv'}",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("### 📋 Tabla filtrada")
        view_cols = ["ID_Registro", "Fecha", "COD_PROYECTO", "Sector_Zona", "Frente_Tramo", "Volumen_m3", "Requeridas", "Ejecutadas", "Brecha", "Pct_Cumpl", "Estado"]
        st.dataframe(style_table_pie(df_f[view_cols].copy()), width='stretch', height=360)
st.sidebar.markdown("<div class='hr'></div>", unsafe_allow_html=True)
st.sidebar.markdown("#### 📚 Biblioteca / IA")
if st.sidebar.button("📚 7) Biblioteca EETT", width='stretch'):
    st.session_state["APP_PAGE"] = "EETT_P7"
    st.rerun()
if st.sidebar.button("🤖 8) IA sobre EETT", width='stretch'):
    st.session_state["APP_PAGE"] = "IA_P8"
    st.rerun()
# =========================================================
# ============  PANTALLA 7 – BIBLIOTECA EETT  =============
# =========================================================

import re
import hashlib
from pathlib import Path

EETT_DIR = "biblioteca_eett"
EETT_INDEX_FILE = "qintegrity_biblioteca.xlsx"  # ya lo tienes en tu set
EETT_INDEX_COLS = [
    "DocID",
    "Nombre_Original",
    "Nombre_Fisico",
    "REV",
    "Disciplina",
    "Tags",
    "Estado",
    "Fecha_Carga",
    "Hash_SHA256",
    "Ruta_Relativa",
    "Tamano_Bytes",
]

def _eett_ensure_index() -> pd.DataFrame:
    """Carga el índice de biblioteca EETT de forma robusta.
    Fuente de verdad: CSV (evita perder registros cuando el XLSX está desactualizado o bloqueado).
    """
    csv_path = os.path.splitext(EETT_INDEX_FILE)[0] + ".csv"

    df = None

    # 1) CSV (source of truth)
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            df = None

    # 2) XLSX solo si no hay CSV
    if df is None and os.path.exists(EETT_INDEX_FILE) and HAS_OPENPYXL:
        try:
            df = safe_read_excel(EETT_INDEX_FILE, engine="openpyxl")
        except Exception:
            df = None

    if df is None:
        df = pd.DataFrame(columns=EETT_INDEX_COLS)

    for c in EETT_INDEX_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[EETT_INDEX_COLS].copy()

    df["Fecha_Carga"] = pd.to_datetime(df["Fecha_Carga"], errors="coerce")
    df["Tamano_Bytes"] = pd.to_numeric(df["Tamano_Bytes"], errors="coerce")
    df["Estado"] = df["Estado"].fillna("Vigente").astype(str)

    return df



def _eett_save_index(df: pd.DataFrame) -> None:
    """Guarda índice EETT.
    - Siempre guarda un CSV espejo para que funcione sin openpyxl.
    - Si openpyxl existe, además guarda XLSX (compatibilidad con tu archivo histórico).
    """
    out = df.copy()
    for c in EETT_INDEX_COLS:
        if c not in out.columns:
            out[c] = np.nan
    out = out[EETT_INDEX_COLS]

    csv_path = os.path.splitext(EETT_INDEX_FILE)[0] + ".csv"
    try:
        out.to_csv(csv_path, index=False, encoding="utf-8")
    except Exception:
        pass

    if HAS_OPENPYXL:
        try:
            out.to_excel(EETT_INDEX_FILE, index=False, engine="openpyxl")
        except Exception:
            # si falla xlsx, igual queda el csv
            pass

def _eett_sanitize(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^\w\s\.-]", "", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    return name[:140] if len(name) > 140 else name

def _eett_sha256(file_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(file_bytes)
    return h.hexdigest()

def _eett_upsert_document(file_name: str, file_bytes: bytes, disciplina: str, rev: str, tags: str) -> Tuple[bool, str]:
    """Inserta/actualiza un documento en Biblioteca EETT.

    - Guarda físico en ./biblioteca_eett/
    - Índice con SHA256 anti-duplicado
    - NO duplica si el contenido ya existe (mismo hash); solo actualiza metadatos.
    """
    Path(EETT_DIR).mkdir(parents=True, exist_ok=True)
    df = _eett_ensure_index()

    sha = _eett_sha256(file_bytes)

    # anti-duplicado por hash
    exists_mask = df["Hash_SHA256"].astype(str).str.lower().eq(str(sha).lower())
    if exists_mask.any():
        idx = df[exists_mask].index[0]
        df.at[idx, "Disciplina"] = disciplina
        df.at[idx, "REV"] = rev
        df.at[idx, "Tags"] = tags
        df.at[idx, "Estado"] = "Vigente"
        df.at[idx, "Fecha_Carga"] = datetime.now()
        _eett_save_index(df)
        return (False, "Documento ya existía (mismo contenido / hash). No se duplicó. Metadatos actualizados.")

    docid = _safe_uuid()

    suffix = (Path(file_name).suffix or "").lower()
    ext = suffix.lstrip(".") if suffix else "bin"

    clean_stem = _eett_sanitize(Path(file_name).stem)
    clean_rev = _eett_sanitize(rev) if rev else "NA"

    nombre_fisico = f"{docid}__{clean_stem}__REV{clean_rev}.{ext}"
    rel_path = str(Path(EETT_DIR) / nombre_fisico)
    abs_path = str(Path(rel_path).resolve())

    with open(abs_path, "wb") as f:
        f.write(file_bytes)

    row = {
        "DocID": docid,
        "Nombre_Original": str(file_name),
        "Nombre_Fisico": nombre_fisico,
        "REV": rev,
        "Disciplina": disciplina,
        "Tags": tags,
        "Estado": "Vigente",
        "Fecha_Carga": datetime.now(),
        "Hash_SHA256": sha,
        "Ruta_Relativa": rel_path.replace("\\", "/"),
        "Tamano_Bytes": len(file_bytes),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _eett_save_index(df)
    return (True, "Documento cargado y indexado OK (con hash anti-duplicado).")



def _eett_get_catalog(vigentes_only: bool = True) -> pd.DataFrame:
    df = _eett_ensure_index()
    if vigentes_only:
        df = df[df["Estado"].astype(str).str.lower().eq("vigente")].copy()
    return df

def _eett_mark_obsolete(docid: str) -> None:
    df = _eett_ensure_index()
    m = df["DocID"].astype(str).eq(str(docid))
    if m.any():
        df.loc[m, "Estado"] = "Obsoleta"
        _eett_save_index(df)

def _eett_delete(docid: str) -> Tuple[bool, str]:
    df = _eett_ensure_index()
    m = df["DocID"].astype(str).eq(str(docid))
    if not m.any():
        return (False, "No encontré ese DocID en el índice.")

    row = df[m].iloc[0]
    rel = str(row.get("Ruta_Relativa", "") or "")
    abs_path = str(Path(rel).resolve()) if rel else ""

    # Borra físico si existe
    try:
        if abs_path and os.path.exists(abs_path):
            os.remove(abs_path)
    except Exception as e:
        return (False, f"No pude borrar el PDF físico: {e}")

    # Borra del índice
    df = df[~m].copy()
    _eett_save_index(df)
    return (True, "Eliminado definitivo: archivo + registro índice.")

def render_pantalla_7_eett():
    st.subheader("📚 Biblioteca EETT")
    st.caption("Pantalla 7 · Carga documentos + Índice con trazabilidad (hash anti-duplicado) + acciones")
    if "mensaje" in st.session_state:
        st.toast(st.session_state["mensaje"],icon="ℹ️")
        del st.session_state["mensaje"]
    c1, c2 = st.columns([1.2, 1.8])

    with c1:
        st.markdown("### Cargar EETT (cualquier formato)")
        up = st.file_uploader("Subir documento (PDF/DOCX/otros)", type=None, key="EETT_UPLOADER")
        disciplina = st.text_input("Disciplina / Tipo", value="Movimiento de Tierra", key="EETT_DISC")
        rev = st.text_input("REV", value="A", key="EETT_REV")
        tags = st.text_input("Tags (separados por coma)", value="", key="EETT_TAGS")

        if up is not None:
            file_bytes = up.getvalue()
            if st.button("⬆️ Guardar en Biblioteca", width='stretch'):
                created, msg = _eett_upsert_document(
                    file_name=up.name,
                    file_bytes=file_bytes,
                    disciplina=str(disciplina).strip(),
                    rev=str(rev).strip(),
                    tags=str(tags).strip(),
                )
                st.session_state["mensaje"] = "Documento subido correctamente" if created else msg
                st.rerun()

    with c2:
        st.markdown("### Catálogo / Acciones")
        df = _eett_ensure_index()

        # Filtros
        f1, f2, f3, f4 = st.columns([1.3, 1.2, 1.2, 1.3])
        with f1:
            q = st.text_input("Buscar", value="", key="EETT_Q").strip().lower()
        with f2:
            estados = ["(Todos)"] + sorted(df["Estado"].dropna().astype(str).unique().tolist())
            est = st.selectbox("Estado", options=estados, index=0, key="EETT_EST")
        with f3:
            discs = ["(Todas)"] + sorted(df["Disciplina"].dropna().astype(str).unique().tolist())
            disc_sel = st.selectbox("Disciplina", options=discs, index=0, key="EETT_DISC_F")
        with f4:
            revs = ["(Todas)"] + sorted(df["REV"].dropna().astype(str).unique().tolist())
            rev_sel = st.selectbox("REV", options=revs, index=0, key="EETT_REV_F")

        dfv = df.copy()
        if est != "(Todos)":
            dfv = dfv[dfv["Estado"].astype(str).eq(est)].copy()
        if disc_sel != "(Todas)":
            dfv = dfv[dfv["Disciplina"].astype(str).eq(disc_sel)].copy()
        if rev_sel != "(Todas)":
            dfv = dfv[dfv["REV"].astype(str).eq(rev_sel)].copy()
        if q:
            blob = (
                dfv["Nombre_Original"].fillna("").astype(str)
                + " "
                + dfv["Disciplina"].fillna("").astype(str)
                + " "
                + dfv["Tags"].fillna("").astype(str)
                + " "
                + dfv["REV"].fillna("").astype(str)
            ).str.lower()
            dfv = dfv[blob.str.contains(q, na=False)].copy()

        df_show = dfv.sort_values(["Estado", "Fecha_Carga"], ascending=[True, False]).copy()

        st.dataframe(
            df_show[["DocID", "Nombre_Original", "Disciplina", "REV", "Estado", "Fecha_Carga"]]
            if not df_show.empty
            else df_show,
            width='stretch',
            height=360,
        )

        if df_show.empty:
            st.info("No hay documentos con esos filtros.")
            return

        docids = df_show["DocID"].astype(str).tolist()
        pick = st.selectbox("Seleccionar DocID", options=[None] + docids, index=0, key="EETT_PICK")

        if pick:
            row = df_show[df_show["DocID"].astype(str).eq(str(pick))].iloc[0]
            rel = str(row.get("Ruta_Relativa", "") or "")
            abs_path = str(Path(rel).resolve()) if rel else ""
            nombre = str(row.get("Nombre_Original", "") or "EETT.pdf")

            a1, a2, a3 = st.columns(3)
            with a1:
                # Descargar
                try:
                    if abs_path and os.path.exists(abs_path):
                        with open(abs_path, "rb") as f:
                            dwld = st.download_button("⬇️ Descargar PDF", data=f.read(), file_name=nombre, mime="application/pdf", width='stretch')
                    else:
                        st.warning("Archivo físico no encontrado.")
                except Exception as e:
                    st.error(f"Error descarga: {e}")

            with a2:
                if st.button("🗂️ Marcar Obsoleta", width='stretch'):
                    _eett_mark_obsolete(str(pick))
                    st.session_state["mensaje"]= "Marcada como Obsoleta (no borré el PDF físico)."
                    st.rerun()

            with a3:
                if st.button("🧨 Eliminar definitivo", width='stretch'):
                    ok, msg = _eett_delete(str(pick))
                    st.session_state["mensaje"] = "Se elimino correctamente" if ok else msg
                    st.rerun()
            if dwld:
                st.session_state["mensaje"] = "Descargando...."



# =========================================================
# ================  PANTALLA 8 – IA EETT  =================
# =========================================================

def _pdf_extract_text(path_pdf: str, max_pages: int = 25) -> Tuple[bool, str]:
    """
    Extrae texto. Si no puede (PDF escaneado/imagen), devuelve (False, motivo).
    """
    if not path_pdf or (not os.path.exists(path_pdf)):
        return (False, "No existe el archivo PDF seleccionado.")
    text = ""
    # Intento 1: PyPDF2
    try:
        import PyPDF2  # type: ignore
        with open(path_pdf, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            n = min(len(reader.pages), int(max_pages))
            for i in range(n):
                t = reader.pages[i].extract_text() or ""
                text += "\n" + t
        text = text.strip()
        if len(text) >= 200:
            return (True, text)
    except Exception:
        pass

    # Intento 2: pdfplumber
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(path_pdf) as pdf:
            n = min(len(pdf.pages), int(max_pages))
            for i in range(n):
                t = pdf.pages[i].extract_text() or ""
                text += "\n" + t
        text = text.strip()
        if len(text) >= 200:
            return (True, text)
    except Exception:
        pass

    return (False, "No pude extraer texto (posible PDF escaneado/imágenes). No invento contenido.")

def _ia_rule_based_summary(text: str) -> Dict[str, str]:
    """
    IA local (sin API): resumen y checklist por heurística.
    NO inventa: usa solo lo encontrado en el texto.
    """
    t = re.sub(r"\s+", " ", text).strip()
    # troceo básico
    head = t[:3000]

    # keywords QA/QC típicas
    keys = [
        "alcance", "objetivo", "especific", "requisito", "criterio", "tolerancia",
        "ensayo", "frecuencia", "muestreo", "compact", "densidad", "proctor",
        "control", "acept", "rechaz", "observ", "norma", "astm", "nch", "mop", "serviu",
        "proced", "registro", "trazabil", "subrasante", "subbase", "base", "hormigon", "asfalto",
    ]

    hits = []
    low = t.lower()
    for k in keys:
        if k in low:
            hits.append(k)

    # "Resumen" = primeras frases recortadas
    resumen = head
    if len(resumen) > 900:
        resumen = resumen[:900] + "..."

    checklist = []
    if "ensayo" in low:
        checklist.append("Verificar lista de ensayos exigidos + frecuencia + criterio de aceptación.")
    if "norma" in low or "astm" in low or "nch" in low or "mop" in low:
        checklist.append("Verificar normas citadas y su versión aplicable (ASTM/NCh/MOP).")
    if "trazabil" in low or "registro" in low:
        checklist.append("Confirmar trazabilidad documental: códigos, revisiones, registros y control de cambios.")
    if "compact" in low or "densidad" in low or "proctor" in low:
        checklist.append("Revisar parámetros de compactación: %comp, humedad, Proctor, tolerancias y criterio A/O/R.")
    if "acept" in low or "rechaz" in low or "observ" in low:
        checklist.append("Validar procedimiento de aceptación/rechazo/observación y gestión de no conformidades.")
    if not checklist:
        checklist.append("No detecté keywords suficientes para checklist robusto (texto muy corto o escaneado).")

    return {
        "resumen": resumen,
        "hallazgos": ", ".join(sorted(set(hits))) if hits else "(sin keywords QA/QC detectables en texto extraído)",
        "checklist": "\n".join([f"- {x}" for x in checklist]),

    }

def _read_docx_text_robust(abs_path: str) -> str:
    """Lee texto desde .docx de forma robusta.
    1) Intenta python-docx (si está instalado)
    2) Fallback: abre el .docx como ZIP y extrae word/document.xml
    """
    try:
        from docx import Document  # type: ignore
        doc = Document(abs_path)
        parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        text = "\n".join(parts).strip()
        if text:
            return text
    except Exception:
        pass

    try:
        import zipfile
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(abs_path) as z:
            xml_bytes = z.read("word/document.xml")
        root = ET.fromstring(xml_bytes)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        texts = []
        for node in root.findall(".//w:t", ns):
            if node.text and node.text.strip():
                texts.append(node.text.strip())
        return "\n".join(texts).strip()
    except Exception:
        return ""

# --- OCR Y DETECCIÓN DE IDIOMA  ---
def _detect_language(text: str) -> str:
    """Detecta el idioma del texto usando langdetect (español por prioridad)."""
    try:
        from langdetect import detect  # type: ignore
        return detect(text)
    except Exception:
        return "unknown"


def _find_poppler_path() -> Optional[str]:

    try:
        # 1) variable de entorno explícita
        poppler_path = os.environ.get("POPPLER_PATH")
        if poppler_path and os.path.exists(poppler_path):
            return poppler_path

        import shutil, glob
        # 2) pdftoppm en PATH
        pdftoppm_exe = shutil.which("pdftoppm")
        if pdftoppm_exe:
            return os.path.dirname(pdftoppm_exe)

        # 3) buscar en WinGet local appdata (pattern común en instalaciones por winget)
        local = os.environ.get("LOCALAPPDATA")
        if local:
            pattern = os.path.join(local, "Microsoft", "WinGet", "Packages", "**", "poppler-*")
            candidates = glob.glob(pattern, recursive=True)
            for c in candidates:
                bin_path = os.path.join(c, "Library", "bin")
                if os.path.exists(os.path.join(bin_path, "pdftoppm.exe")):
                    return bin_path
    except Exception:
        pass
    return None


def _read_pdf_text_robust(abs_path: str, force_ocr: bool = False) -> Tuple[str, str]:
    """Extrae texto de un PDF de forma robusta y devuelve (texto, diagnostico).

    Retorna:
      - texto extraído ("" si falla)
      - diagnóstico/string con mensajes de log de lo ocurrido
    """
    cache_path = abs_path + ".ocr.txt"
    diag: List[str] = []

    try:
        if os.path.exists(cache_path) and not force_ocr:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = f.read()
            if cached.strip():
                diag.append("Texto leído desde cache")
                return cached, "\n".join(diag)
    except Exception as e:
        diag.append(f"No pude leer cache: {e}")

    text = ""

    # 1) Extracción nativa con PyMuPDF
    if not force_ocr:
        try:
            import fitz  # type: ignore
            doc = fitz.open(abs_path)
            pages = [p.get_text("text") for p in doc]
            text = "\n".join(pages).strip()
            diag.append(f"Extracción nativa: {len(pages)} páginas, {len(text)} chars")
            if len(text) > 200:
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text)
                return text, "\n".join(diag)
        except Exception as e:
            diag.append(f"Error extracción nativa: {e}")

    # 2) Intentar ocrmypdf -> produce PDF con texto extraíble
    try:
        import subprocess
        out_pdf = abs_path + ".ocr.pdf"
        # Si se fuerza OCR, usar --force-ocr para reprocesar páginas que ya tienen texto
        if force_ocr:
            cmd = ["ocrmypdf", "-l", "spa", "--force-ocr", abs_path, out_pdf]
        else:
            cmd = ["ocrmypdf", "-l", "spa", abs_path, out_pdf, "--skip-text"]
        diag.append(f"Ejecutando: {' '.join(cmd)}")
        # Timeout aumentado para PDFs grandes
        proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
        diag.append(f"ocrmypdf returncode={proc.returncode}")
        if proc.stdout:
            diag.append(f"ocrmypdf stdout: {proc.stdout[:1000]}")
        if proc.stderr:
            diag.append(f"ocrmypdf stderr: {proc.stderr[:1000]}")
        if proc.returncode == 0 and os.path.exists(out_pdf):
            try:
                import fitz  # type: ignore
                doc = fitz.open(out_pdf)
                pages = [p.get_text("text") for p in doc]
                text = "\n".join(pages).strip()
                diag.append(f"ocrmypdf produjo {len(pages)} páginas, {len(text)} chars")
                if text:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    return text, "\n".join(diag)
            except Exception as e:
                diag.append(f"Error leyendo out_pdf: {e}")
        else:
            # Diagnosticar errores comunes (qpdf/ghostscript faltantes)
            stderr = (proc.stderr or "")
            if "qpdf" in stderr.lower():
                diag.append("ocrmypdf: parece faltar 'qpdf' (choco install qpdf)")
            if "ghostscript" in stderr.lower() or "gswin" in stderr.lower():
                diag.append("ocrmypdf: parece faltar 'ghostscript' (choco install ghostscript)")
    except Exception as e:
        diag.append(f"Error ejecutando ocrmypdf: {e}")

    # 3) Fallback: PyMuPDF (pymupdf) + RapidOCR (sin poppler/tesseract)
    try:
        diag.append("Usando fallback OCR: PyMuPDF + RapidOCR (onnxruntime, sin binarios externos)")

        # Llama a la función que pegaste (ya abre el PDF, procesa páginas, cachea motor y guarda progreso)
        text, debug = ocr_pdf_rapidocr(abs_path, cache_path, diag, dpi=90)

        # debug ya trae el diag unido; pero como estamos pasando diag, puedes retornar el debug directamente
        if text:
            return text, debug

    except Exception as e:
        diag.append(f"Fallback RapidOCR falló: {e}")
# --- MOTOR OCR Y DETECCIÓN DE IDIOMA ---


def _simple_audit_summarize(text: str) -> dict:
    """Resumen auditor simple (sin inventar)."""
    import re
    t = (text or "").strip()
    if not t:
        return {"summary": "", "requirements": [], "checklist": []}

    t_norm = re.sub(r"\s+", " ", t)
    sentences = re.split(r"(?<=[\.!\?])\s+", t_norm)
    summary = " ".join([s for s in sentences[:5] if s]).strip()

    req_lines = []
    for line in text.splitlines():
        l = line.strip()
        if not l:
            continue
        if re.search(r"\b(requisit|debe|deberá|shall|must|criterio|ensayo|tolerancia|control|verific|QA|QC)\b", l, re.IGNORECASE):
            req_lines.append(l)
        if len(req_lines) >= 12:
            break

    checklist = [
        "Documento asociado al registro correcto (DocID)",
        "Revisión (REV) coherente con el documento",
        "Campos obligatorios completos",
        "Criterios / tolerancias aplicados según documento",
        "Evidencia QA/QC adjunta cuando corresponda",
    ]
    for l in req_lines[:10]:
        checklist.append(f"Verificar: {l[:120]}")

    return {"summary": summary, "requirements": req_lines, "checklist": checklist}

def render_pantalla_8_ia():
    #verificar_groq() #se uso para verificar si funciona el groq y mostrar el problema
    st.subheader("🤖 IA sobre EETT (modo auditor)")
    # --- MODIFICADO: caption ampliado para mencionar PDF/OCR ---
    st.caption("Pantalla 8 · Selecciona EETT de Biblioteca · Lee WORD (.docx) o PDF (.pdf) y genera resumen + checklist QA/QC (sin inventar).")
    # --- FIN MODIFICACIÓN: caption ---

    #REVISAR MENSAJES DE LISTA DE PENDIENTES EN COLA
    if "revision" in st.session_state:
        st.toast(st.session_state["revision"])
        del st.session_state["revision"]
    df = _eett_ensure_index()
    if df is None or df.empty:
        st.info("No hay documentos en la biblioteca EETT. Sube uno en Pantalla 7.")
        return
    if not st.session_state.get("HAS_GROQ", False):
        st.error(
            "Groq no está instalado en el proyecto, es necesario para este modulo de analisis mediante IA")
        st.error(
            "Instálalo en el mismo entorno donde corres Streamlit:  ```pip install groq``` e intentalo nuevamente")
        st.stop()

    dff = df.copy()
    dff["Estado"] = dff["Estado"].fillna("Vigente").astype(str)
    dff = dff[dff["Estado"].str.lower().ne("obsoleta")].copy()
    if dff.empty:
        st.info("No hay documentos vigentes en la biblioteca.")
        return

    def _label(r):
        return f"{r.get('DocID','')} | {r.get('Nombre_Original','')} | REV {r.get('REV','')}"

    options = dff.to_dict("records")
    labels = [_label(r) for r in options]
    sel = st.selectbox("Selecciona documento", labels, index=0)
    

    row = options[labels.index(sel)]
    docid = str(row.get("DocID", ""))
    rev = str(row.get("REV", ""))
    rel = str(row.get("Ruta_Relativa", "")).replace("\\", "/")
    abs_path = str(Path(rel).resolve())


    ext = (Path(abs_path).suffix or Path(str(row.get("Nombre_Original", ""))).suffix or "").lower().lstrip(".")
    st.markdown(f"-**DocID:** {docid}")
    st.markdown(f"-**REV:** {rev}")
    st.markdown(f"-**Ext:** .{ext}")

    if not os.path.exists(abs_path):
        st.error("Archivo físico no encontrado en biblioteca_eett/.")
        return

    # --- MODIFICADO: Extracción y OCR ---
    text = ""
    if ext == "docx":
        text = _read_docx_text_robust(abs_path)
        if not text.strip():
            st.error("No pude extraer texto desde el Word. (El archivo en disco no parece ser un DOCX válido o está vacío).")
            return
    elif ext == "pdf":
        # Extracción nativa, si falla o es insuficiente, OCR automático
        text, diag = _read_pdf_text_robust(abs_path, force_ocr=False)
        if not text or len(text.strip()) < 200:
            text, diag = _read_pdf_text_robust(abs_path, force_ocr=True)
        if not text.strip():
            st.error(diag)
            return
    else:
        st.warning("Tipo de archivo no soportado. Esta demo analiza DOCX y PDF (con OCR).")
        return


    # --- FIN MODIFICACIÓN: extracción y OCR ---

    # ANALISIS DE IA Y GENERACION DE CHECKLISTS QA/QC
    # Leer las revisiones de checkboxes previas existentes
    def obtener_revisiones():
        try:
            revisiones = DATA_FILE_REVISIONES.read_text()
            revisiones = revisiones.replace("'", "")
        except FileNotFoundError:
            return []
        except PermissionError:
            return []
        except UnicodeDecodeError:
            return []
        except Exception as e:
            return []
        else:
            return revisiones.split("\n")

    # Aprobar una revision de acuerdo al estado del checkbox
    def aprobar_revision(id_revision):
        try:
            revisiones = obtener_revisiones()
            if id_revision not in revisiones:
                with open(DATA_FILE_REVISIONES, 'a') as f:
                    f.write(f"\n{id_revision.strip()}")
        except FileNotFoundError:
            return ("❌ El archivo no existe")
        except PermissionError:
            return ("❌ No tienes permisos para leer el archivo")
        except UnicodeDecodeError:
            return ("❌ Problema de codificación del archivo")
        except Exception as e:
            return (f"❌ Error inesperado: {e}")
        else:
            return f"✅ Aprobada"

    # Revertir check de aprobacion
    def revertir_desicion(id_revision):
        try:
            revisiones = obtener_revisiones()
            if id_revision in revisiones:
                revisiones.remove(id_revision)
                DATA_FILE_REVISIONES.write_text(data="\n".join(revisiones).strip())
                return "✅ Revertiste la aprobacion"
            else:
                return "✅ Revertiste la aprobacion"

        except FileNotFoundError:
            return "❌ El archivo no existe"
        except PermissionError:
            return "❌ No tienes permisos para leer el archivo"
        except UnicodeDecodeError:
            return "❌ Problema de codificación del archivo"
        except Exception as e:
            return f"❌ Error inesperado: {e}"

    def create_checkboxes(id_generated,checkboxes):
        # Metodo al cambiar estado del checkbox
        revisiones = obtener_revisiones()
        def callback_chk_box(chk_key):
            if st.session_state.get(chk_key, False) and chk_key not in revisiones:
                st.session_state["revision"] = aprobar_revision(chk_key)

            elif not st.session_state.get(chk_key, False) and chk_key in revisiones:
                st.session_state["revision"] = revertir_desicion(chk_key)

        for index,chk_text in enumerate(checkboxes):
            chk_key = f"{id_generated}_chk_{index}"
            st.checkbox(chk_text if chk_key not in revisiones else f"~~{chk_text.strip()}~~",key=chk_key,on_change=callback_chk_box,args=(chk_key,),value=True if chk_key in revisiones else False)


    id_generated = Path(abs_path).name

    if "current_doc_id" not in st.session_state:
        st.session_state["current_doc_id"] = id_generated

    # Si el ID guardado es diferente al actual, limpiamos el historial
    if st.session_state["current_doc_id"] != id_generated:
        st.session_state["chat_history"] = []
        st.session_state["current_doc_id"] = id_generated
        # Opcional: Mostrar un aviso rápido
        st.toast(f"Cambiando contexto a: {id_generated}", icon="🔄")

    # 1. RECUPERAR INSTANCIA DE IA
    API_IA_INSTANCIA = st.session_state.get("API_IA")

    if API_IA_INSTANCIA is None:
        st.error("❌ El motor de IA no está inicializado. Revisa la configuración al inicio de la app.")
        st.stop()

    # 2. INTENTO DE RESUMEN Y CHECKLIST
    # CACHED EN SESSION_STATE para no regenerar en cada st.rerun()
    cache_key = f"chat_ia_{id_generated}"
    
    if cache_key not in st.session_state:

        ia_content = API_IA_INSTANCIA.check_resumen_ia(id_generated)
        
        if ia_content == "":
            # Si no existe, lo generamos una sola vez 
            with st.spinner("Generando análisis inicial con IA..."):
                chat_ia = API_IA_INSTANCIA.generate_ia_resume(text)
        else:
            
            # Reutilizamos el existente
            chat_ia = ia_content
        # GUARDAR EN SESSION_STATE para evitar regeneraciones
        st.session_state[cache_key] = chat_ia
    else:
        # Ya estaba en cache: lo reutilizamos directamente
        chat_ia = st.session_state[cache_key]

    # Mostrar el resumen
    clean_resume = API_IA_INSTANCIA.clean_checkboxes(chat_ia)
    st.markdown(clean_resume)
    
    # Generamos los checkboxes interactivos
    checkboxes = API_IA_INSTANCIA.generate_checkboxes(chat_ia)
    create_checkboxes(id_generated, checkboxes)

    # --- 3. CHAT INTERACTIVO SOBRE EL DOCUMENTO ---
    st.markdown("---")
    st.subheader("💬 Chat Consultor de EETT")
    st.caption("Pregunta sobre tolerancias, materiales o normativas específicas de este documento.")

    # Inicializar historial de chat si no existe
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Mostrar los mensajes que ya existen en el historial
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada de nueva pregunta del usuario
    if pregunta := st.chat_input("Escribe tu duda aquí..."):
        # Se agrega la pregunta al historial y la mostramos
        st.session_state.chat_history.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        # Generamos la respuesta de la IA
        with st.chat_message("assistant"):
            with st.spinner("Revisando el documento..."):
                respuesta = API_IA_INSTANCIA.chat_interactivo(
                    pregunta,
                    st.session_state.chat_history[:-1],
                    chat_ia  
                )
                st.markdown(respuesta)
        
        # Guardamos la respuesta y refrescamos para mantener el orden
        st.session_state.chat_history.append({"role": "assistant", "content": respuesta})
        st.rerun()

# =========================================================
# ==============  ROUTER NUEVAS PANTALLAS 7/8  =============
# =========================================================
if st.session_state.get("APP_PAGE") == "EETT_P7":
    st.caption("Biblioteca EETT · Pantalla 7 · Carga + Catálogo + Acciones")
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    render_pantalla_7_eett()

if st.session_state.get("APP_PAGE") == "IA_P8":
    st.caption("IA sobre EETT · Pantalla 8 · Resumen + Checklist QA/QC")
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    render_pantalla_8_ia()
# =========================================================
# BLOQUE EXTRA – ASOCIACIÓN EETT A DENSIDADES (NO ROMPE BASE)
# =========================================================

def get_eett_catalog_vigente():
    """
    Devuelve lista de documentos EETT vigentes
    Formato: [(DocID, 'DocID | Nombre | REV')]
    """
    try:
        df = safe_read_excel("qintegrity_biblioteca.xlsx")
        df = df[df["Estado"].str.upper() == "VIGENTE"]
        opciones = []
        for _, r in df.iterrows():
            label = f'{r["DocID"]} | {r["Nombre"]} | REV {r.get("REV","")}'
            opciones.append((r["DocID"], label))
        return opciones
    except Exception:
        return []

def densidades_doc_eett_selector():
    """
    Selector opcional de Documento Técnico (EETT)
    """
    opciones = get_eett_catalog_vigente()

    if not opciones:
        st.info("No hay documentos EETT vigentes cargados")
        return None

    labels = ["— Seleccione un documento —"] + [o[1] for o in opciones]
    values = [None] + [o[0] for o in opciones]
    
    seleccion = st.selectbox(
        "Documento Técnico Asociado (EETT)",
        options=list(range(len(values))),
        format_func=lambda i: labels[i],
        key="DEN_DOC_EETT_SELECT"
    )

    return values[seleccion]


def densidades_attach_doc_eett(df):
    """
    Inserta columna DocID_EETT si no existe
    """
    if "DocID_EETT" not in df.columns:
        df["DocID_EETT"] = None
    return df
