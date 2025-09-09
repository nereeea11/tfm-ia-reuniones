import os
import re
import subprocess
import shutil
from typing import List, Dict, Tuple, Optional
from faster_whisper import WhisperModel

# ------------------------------------------------------
# Configuración del modelo
#  - "base"  = más rápido
#  - "small" = mejor precisión (recomendado)
# Se puede cambiar desde la app con os.environ["ASR_MODEL_SIZE"]
# ------------------------------------------------------
MODEL_SIZE = os.environ.get("ASR_MODEL_SIZE", "small")

_model: Optional[WhisperModel] = None


def _get_model() -> WhisperModel:
    """Carga perezosa del modelo en CPU con cuantización int8."""
    global _model
    if _model is None:
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def _to_wav_16k_mono(input_path: str, output_path: str) -> None:
    """
    Si hay FFmpeg, convierte/limpia audio.
    Si NO hay FFmpeg, solo acepta WAV (idealmente mono 16 kHz) y lo copia.
    """
    if _has_ffmpeg():
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ac", "1", "-ar", "16000",
            "-af", "afftdn=nr=12:nt=w,highpass=f=60,lowpass=f=7800,volume=1.5,dynaudnorm",
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    else:
        if not input_path.lower().endswith(".wav"):
            raise RuntimeError(
                "FFmpeg no está disponible en este entorno. "
                "En la nube, sube un archivo WAV (mono, 16 kHz)."
            )
        shutil.copyfile(input_path, output_path)


def _postprocess_text(t: str) -> str:
    """Limpieza ligera: espacios, puntuación, duplicados y capitalización básica."""
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)          # quita espacio antes de signos
    t = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", t)   # asegura espacio tras signo
    t = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", t, flags=re.I)  # elimina duplicados inmediatos
    # capitaliza frases separadas por punto
    frases = [s.strip() for s in re.split(r"\.\s+", t) if s.strip()]
    if frases:
        t = ". ".join(s[0].upper() + s[1:] if len(s) > 1 else s.upper() for s in frases)
        if not t.endswith("."):
            t += "."
    return t


def transcribir_audio(ruta_audio: str, language_mode: str = "auto"):
    """
    Transcribe un audio con faster-whisper.

    Parámetros:
      - language_mode: "auto" | "es" | "en"
        * "auto" -> detección automática de idioma
        * "es"   -> fuerza español
        * "en"   -> fuerza inglés

    Devuelve:
      (texto_completo, segmentos, meta)
        - texto_completo: str
        - segmentos: List[{'start': float, 'end': float, 'text': str}]
        - meta: {'detected_language': str|None, 'language_probability': float|None,
                 'language_param': str, 'model_size': str}
    """
    if not os.path.exists(ruta_audio):
        raise FileNotFoundError(f"No existe el archivo: {ruta_audio}")

    # Preprocesa a wav mono 16k (mejora robustez en CPU)
    tmp_wav = ruta_audio + ".16k.wav"
    _to_wav_16k_mono(ruta_audio, tmp_wav)

    model = _get_model()

    # None = autodetección; "es"/"en" fuerza idioma
    language = None if language_mode == "auto" else language_mode

    segments, info = model.transcribe(
        tmp_wav,
        language=language,
        task="transcribe",
        vad_filter=True,
        vad_parameters={"min_speech_duration_ms": 250, "max_speech_duration_s": 14},
        beam_size=5,
        best_of=5,
        temperature=0.0,
        no_speech_threshold=0.45,
        condition_on_previous_text=False,
        initial_prompt=(
            "Reunión de trabajo. Términos: ventas, marketing, producción, logística, cadena de suministro. "
            "Nombres propios en español o inglés. Evitar inventar nombres."
        ),
    )

    textos: List[str] = []
    segs: List[Dict[str, float | str]] = []

    for s in segments:
        t = (s.text or "").strip()
        if t:
            textos.append(t)
            segs.append({"start": float(s.start), "end": float(s.end), "text": t})

    full_text = _postprocess_text(" ".join(textos).strip()) if textos else ""

    meta = {
        "detected_language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "language_param": language_mode,
        "model_size": MODEL_SIZE,
    }
    return full_text, segs, meta


if __name__ == "__main__":
    # Prueba rápida (ajusta la ruta a un audio tuyo)
    ruta = "data/raw/prueba.mp3"
    try:
        texto, segmentos, meta = transcribir_audio(ruta, language_mode="auto")
        print("Idioma detectado:", meta.get("detected_language"), meta.get("language_probability"))
        print("Modelo:", meta.get("model_size"))
        print("Transcripción (primeros 300 chars):", texto[:300], "...")
        print("Segmentos:", segmentos[:3], "...")
    except Exception as e:
        print("Error:", e)
