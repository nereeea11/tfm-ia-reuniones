import streamlit as st
import tempfile
import os, re

from datetime import datetime
from io import BytesIO
from transcribir import transcribir_audio
from resumen_transformers import resumir_texto_transformers  # resumidor (EN/ES)

IS_CLOUD = os.environ.get("CLOUD_DEPLOY", "0") == "1"


# ========= utilidades de guardado =========
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _slugify(name: str) -> str:
    name = re.sub(r"[^\w\s.-]", "", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "-", name.strip())
    return name.lower() or "audio"

def guardar_resultados(base_filename: str, texto: str, resumen: str) -> None:
    _ensure_dir(os.path.join("data", "processed"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slugify(os.path.splitext(base_filename)[0])

    trans_path = os.path.join("data", "processed", f"{stamp}_{slug}_transcripcion.txt")
    resum_path = os.path.join("data", "processed", f"{stamp}_{slug}_resumen.txt")
    both_path  = os.path.join("data", "processed", f"{stamp}_{slug}_transcripcion+resumen.md")

    with open(trans_path, "w", encoding="utf-8") as f:
        f.write(texto or "")
    with open(resum_path, "w", encoding="utf-8") as f:
        f.write(resumen or "")
    with open(both_path, "w", encoding="utf-8") as f:
        f.write("# Transcripción\n\n")
        f.write((texto or "") + "\n\n")
        f.write("# Resumen\n\n")
        f.write(resumen or "")

def _to_bytes(s: str, filename: str) -> BytesIO:
    bio = BytesIO()
    bio.write((s or "").encode("utf-8"))
    bio.seek(0)
    bio.name = filename
    return bio
# ==========================================

# ========= subtítulos SRT =========
def _format_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def to_srt(segments):
    lines = []
    for i, seg in enumerate(segments, 1):
        text = seg["text"] if isinstance(seg.get("text"), str) else str(seg.get("text", ""))
        lines.append(str(i))
        lines.append(f"{_format_ts(seg['start'])} --> {_format_ts(seg['end'])}")
        lines.append(text.strip())
        lines.append("")  # línea en blanco
    return "\n".join(lines)
# ===================================

st.set_page_config(page_title="Asistente de Reuniones – TFM IA", layout="wide")

# Estilos: cajas claras legibles siempre
st.markdown("""
<style>
.big-title { font-size: 32px; font-weight: 700; margin-bottom: 0.3rem;}
.subtle { opacity: 0.8; }
.box {
  padding: 1rem; border-radius: 14px;
  border:1px solid #e5e7eb;
  background: #f8fafc;
  color:#111827;
  line-height: 1.5;
  white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🤖 Asistente de Reuniones</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Sube un audio para obtener la transcripción y resumen </div>', unsafe_allow_html=True)

# Sidebar (solo lo imprescindible)
st.sidebar.header("Opciones")

# Calidad de ASR
modo = st.sidebar.selectbox(
    "Modo de calidad de transcripción",
    ["Rápido", "Equilibrado (recomendado)"],
    index=1,
    help="Rápido = más veloz. Equilibrado = mejor precisión (puede tardar un poco más)."
)
os.environ["ASR_MODEL_SIZE"] = "base" if modo.startswith("Rápido") else "small"

# Idioma del audio (forzado si lo sabes; si no, auto)
idioma_ui = st.sidebar.selectbox(
    "Idioma del audio",
    ["Detectar automáticamente", "Español", "Inglés"],
    index=0
)
lang_map = {"Detectar automáticamente": "auto", "Español": "es", "Inglés": "en"}
language_mode = lang_map[idioma_ui]

# Uploader
audio = st.file_uploader(
    "📂 Arrastra aquí tu archivo de audio o haz clic en *Examinar*",
    type=(["wav"] if IS_CLOUD else ["mp3", "wav", "m4a"]),
    help=("En la nube: sube WAV mono 16 kHz." if IS_CLOUD else "Formatos permitidos: MP3, WAV o M4A.")
)


with st.container():
    if audio:
        # --- Procesar audio ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio.name.split('.')[-1]}") as tmp:
            tmp.write(audio.read())
            tmp_path = tmp.name

        with st.spinner("Transcribiendo…"):
            texto, segmentos, meta = transcribir_audio(
                tmp_path,
                language_mode=language_mode
            )

        # --- Resumen con Transformers (elige modelo según idioma detectado) ---
        detected = (meta.get("detected_language") or "").lower() if isinstance(meta, dict) else ""
        fallback_lang = language_mode if language_mode in ("es", "en") else "en"
        lang_for_summary = detected if detected in ("es", "en") else fallback_lang

        with st.spinner("Resumiendo…"):
            resumen = resumir_texto_transformers(texto, lang=lang_for_summary)

        # --- UI principal ---
        st.subheader("Transcripción")
        st.write(f'<div class="box">{texto}</div>', unsafe_allow_html=True)

        st.subheader("Resumen")
        st.write(f'<div class="box">{resumen}</div>', unsafe_allow_html=True)

        # Guardado automático (txt/md en data/processed)
        guardar_resultados(audio.name, texto, resumen)

        # Separador elegante antes de botones
        st.divider()

        # === Solo 3 botones de descarga ===
        colA, colB, colC = st.columns(3)
        with colA:
            st.download_button(
                "⬇️ Descargar transcripción",
                data=_to_bytes(texto, "transcripcion.txt"),
                file_name="transcripcion.txt",
                mime="text/plain"
            )
        with colB:
            st.download_button(
                "⬇️ Descargar resumen",
                data=_to_bytes(resumen, "resumen.txt"),
                file_name="resumen.txt",
                mime="text/plain"
            )
        with colC:
            srt_text = to_srt(segmentos)
            st.download_button(
                "⬇️ Descargar subtítulos",
                data=_to_bytes(srt_text, "subtitulos.srt"),
                file_name="subtitulos.srt",
                mime="text/plain"
            )

    else:
        st.info("Sube un archivo de audio ")
