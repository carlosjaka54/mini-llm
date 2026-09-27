"""Etapa 3a: limpieza y filtrado de textos.

Un modelo pequeño aprende lo que ve. Si el corpus trae caracteres rotos,
comillas de cinco tipos distintos o textos duplicados, el modelo gasta
parámetros y vocabulario en ruido. Aquí normalizamos y filtramos.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# Caracteres esperados en cuentos en español
ALLOWED_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzáéíóúüñ"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ"
    "0123456789 \n.,;:¿?¡!\"'()-"
)
MIN_ALLOWED_RATIO = 0.97  # al menos 97 % de caracteres "normales"

# Unificamos variantes tipográficas: menos variantes = vocabulario más útil
_CHAR_MAP = str.maketrans({
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u00ab": '"', "\u00bb": '"',
    "\u2018": "'", "\u2019": "'", "\u00b4": "'", "`": "'",
    "\u2013": "-", "\u2014": "-", "\u2011": "-",
    "\u2026": "...",
    "\u00a0": " ", "\t": " ",
    "\u200b": "", "\ufeff": "",
})

# Señales de "mojibake": UTF-8 leído por error como Windows-1252 (p. ej. "Ã©" en vez de "é")
_MOJIBAKE_MARKERS = ("Ã", "â€", "Â")


def fix_mojibake(text: str) -> str:
    """Repara textos con codificación rota; si no puede, los deja igual."""
    if not any(marker in text for marker in _MOJIBAKE_MARKERS):
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def normalize_text(text: str) -> str:
    """Normaliza Unicode, tipografía y espacios sin cambiar el contenido."""
    text = fix_mojibake(text)
    text = unicodedata.normalize("NFC", text)       # "é" siempre como un solo carácter
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_CHAR_MAP)
    text = text.replace("**", "").replace("__", "")  # restos de formato Markdown
    text = "".join(ch for ch in text if ch == "\n" or not unicodedata.category(ch).startswith("C"))
    text = re.sub(r"[ ]{2,}", " ", text)            # espacios repetidos
    text = re.sub(r" *\n *", "\n", text)            # espacios alrededor de saltos de línea
    text = re.sub(r"\n{3,}", "\n\n", text)          # máximo una línea en blanco
    return text.strip()


def allowed_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(ch in ALLOWED_CHARS for ch in text) / len(text)


def clean_document(text: str, min_words: int, max_words: int) -> tuple[str | None, str]:
    """Devuelve (texto_limpio, "ok") o (None, motivo_de_descarte)."""
    text = normalize_text(text or "")
    if not text:
        return None, "vacío"
    n_words = len(text.split())
    if n_words < min_words:
        return None, "muy corto"
    if n_words > max_words:
        return None, "muy largo"
    if allowed_ratio(text) < MIN_ALLOWED_RATIO:
        return None, "caracteres extraños"
    return text, "ok"


def text_fingerprint(text: str) -> str:
    """Huella para detectar duplicados (ignora mayúsculas y espacios)."""
    canonical = " ".join(text.lower().split())
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()
