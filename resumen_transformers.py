# resumen_transformers.py
import re
from typing import List, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# Modelos
MODEL_EN = "facebook/bart-large-cnn"
MODEL_ES_CANDIDATES = [
    "PlanTL-GOB-ES/t5-base-spanish-summarizer",
    "mrm8488/t5-base-spanish-summarization",
    "mrm8488/bert2bert_shared-spanish-finetuned-summarization",
]

# Caches
_tok_en = None
_mod_en = None
_tok_es = None
_mod_es = None
_mod_es_id: Optional[str] = None  # cuál cargó

DEVICE = torch.device("cpu")  # fuerza CPU; evita 'meta'

def _load_pair(model_id: str):
    """Carga tokenizer+modelo en CPU, sin device_map ni low_mem, y asegura pad_token."""
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    mod = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=None,      # float32 por defecto en CPU
        device_map=None,       # evita 'meta'
    )
    # Asegurar pad_token
    if tok.pad_token is None:
        # usa eos como pad si no existe
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            # último recurso
            tok.add_special_tokens({'pad_token': '[PAD]'})
            mod.resize_token_embeddings(len(tok))
    if getattr(mod.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        mod.config.pad_token_id = tok.pad_token_id

    mod.to(DEVICE)
    return tok, mod

def _load_en():
    global _tok_en, _mod_en
    if _tok_en is None or _mod_en is None:
        _tok_en, _mod_en = _load_pair(MODEL_EN)
    return _tok_en, _mod_en

def _load_es():
    """Carga el primer modelo ES disponible; si ninguno, devuelve (None, None)."""
    global _tok_es, _mod_es, _mod_es_id
    if _tok_es is not None and _mod_es is not None:
        return _tok_es, _mod_es
    last_err = None
    for cand in MODEL_ES_CANDIDATES:
        try:
            tok, mod = _load_pair(cand)
            _tok_es, _mod_es, _mod_es_id = tok, mod, cand
            return _tok_es, _mod_es
        except Exception as e:
            last_err = e
            continue
    # print(f"[WARN] No se pudo cargar un modelo ES. Último error: {last_err}")
    return None, None

def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()

def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\?\!])\s+", (text or "").strip())
    return [p for p in parts if p]

def _chunk(text: str, max_chars=2500) -> List[str]:
    sents = _split_sentences(text)
    chunks, cur, size = [], [], 0
    for s in sents:
        if size + len(s) + 1 > max_chars and cur:
            chunks.append(" ".join(cur))
            cur, size = [s], len(s) + 1
        else:
            cur.append(s)
            size += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks

@torch.no_grad()
def _summ_block(text: str, lang: str, max_len=160, min_len=60) -> str:
    text = _clean(text)
    if not text:
        return ""
    if lang.startswith("es"):
        tok, mod = _load_es()
        if tok is None or mod is None:
            tok, mod = _load_en()  # fallback
    else:
        tok, mod = _load_en()

    inputs = tok(text, return_tensors="pt", truncation=True, max_length=2048)
    # mover a CPU explícitamente
    for k in inputs:
        inputs[k] = inputs[k].to(DEVICE)

    outputs = mod.generate(
        **inputs,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    out = tok.decode(outputs[0], skip_special_tokens=True)
    return _clean(out)

def resumir_texto_transformers(texto: str, lang: Optional[str] = None) -> str:
    """
    1) Limpia
    2) Trocea si es largo
    3) Resume cada trozo
    4) Fusiona y segunda pasada para compactar
    Devuelve en viñetas para claridad.
    """
    if not texto or len(texto.split()) < 40:
        return _clean(texto)

    lang = (lang or "en").lower()
    texto = _clean(texto)
    chunks = _chunk(texto, max_chars=2500)

    parciales = []
    for c in chunks:
        parciales.append(_summ_block(c, lang, max_len=160, min_len=60))

    fusion = " ".join(parciales).strip()

    if len(fusion.split()) > 180:
        fusion = _summ_block(" ".join(_split_sentences(fusion)), lang, max_len=140, min_len=50)

    bullets = []
    for line in re.split(r"[\.\n]\s+", fusion):
        line = _clean(line).lstrip("-•–— ").strip()
        if len(line) >= 4:
            bullets.append("• " + line[0].upper() + line[1:])

    if not bullets:
        bullets = ["• " + fusion] if fusion else []

    return "\n".join(bullets)
