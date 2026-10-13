"""Mejor de N: escribir varios borradores y quedarse con el más coherente.

Un modelo pequeño se equivoca a veces (cambia el nombre del protagonista, olvida lo pedido,
se repite). En vez de reentrenar, se generan N borradores y se elige el de mejor puntaje:

    puntaje = log-probabilidad media                (fluidez: menos perplejidad, mejor)
              - 0.6 x fracción de lo pedido que no aparece  (¿habla del dragón que pediste?)
              - 0.25 x personajes nuevos más allá de uno    (¿Sofía se volvió Lucy?)
              - 1.0 x fracción de trigramas repetidos       (¿"Él amaba su hueso" dos veces?)

La idea se llama *reranking* o *best-of-N*: es la misma que usan muchos sistemas reales
(p. ej., con un modelo que puntúa respuestas). Aquí el juez son reglas simples y explicables.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

_UP = "A-ZÁÉÍÓÚÑ"
_LOW = "a-záéíóúüñ"
# Nombre propio: mayúscula a mitad de frase ("dijo Leo", "llamado Spot")
_MID_NAME = re.compile(rf"(?<=[{_LOW},;:]\s)([{_UP}][{_LOW}]+)")
_ANY_CAP = re.compile(rf"\b([{_UP}][{_LOW}]+)\b")
_WORD = re.compile(rf"[{_LOW}{_UP}]+")
# Lo pedido: "un perro", "una niña", "un dragón", "un conejo"...
_ASKED_NOUN = re.compile(rf"\b(?:un|una|el|la|del|al)\s+([{_LOW}]{{3,}})")
_NOT_TOPIC = {"cuento", "historia", "moraleja", "vez", "día", "poco", "rato", "llamado", "llamada",
              "parque", "playa", "bosque", "escuela", "jardín", "casa"}  # lugares: opcionales
# Palabras que pueden ir con mayúscula a mitad de frase sin ser personajes
_NOT_NAME = {"Él", "Ella", "Ellos", "Ellas", "Mamá", "Papá", "Sí", "No", "Gracias", "Hola"}

W_MISSING, W_NEW_NAMES, W_REPEAT = 0.6, 0.25, 1.0
FREE_NEW_NAMES = 1  # un amigo nuevo en el cuento está bien


def character_names(text: str) -> list[str]:
    """Nombres propios en orden de aparición (sin repetir)."""
    mid = {m for m in _MID_NAME.findall(text) if m not in _NOT_NAME}
    seen: list[str] = []
    for m in _ANY_CAP.findall(text):  # incluye sus apariciones a inicio de frase, para el orden
        if m in mid and m not in seen:
            seen.append(m)
    return seen


def requested(message: str) -> tuple[list[str], list[str]]:
    """(nombres, sustantivos) que pide el mensaje: 'un perro llamado Max' -> (['Max'], ['perro'])."""
    names = [m for m in _MID_NAME.findall(message) if m not in _NOT_NAME]
    nouns = [n for n in _ASKED_NOUN.findall(message.lower()) if n not in _NOT_TOPIC]
    return names, nouns


def repeated_trigrams(text: str) -> float:
    words = [w.lower() for w in _WORD.findall(text)]
    grams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


@dataclass
class Score:
    score: float
    perplexity: float | None
    missing: list[str]      # cosas pedidas que el cuento no menciona
    new_names: list[str]    # personajes que no estaban en la petición
    repetition: float       # fracción de trigramas repetidos

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def score_story(message: str, story: str, perplexity: float | None) -> Score:
    names, nouns = requested(message)
    low = story.lower()
    asked = names + nouns
    missing = [w for w in names if w not in story] + [w for w in nouns if w not in low]
    new_names = [n for n in character_names(story) if n not in names]
    rep = repeated_trigrams(story)
    fluency = -math.log(perplexity) if perplexity else -5.0
    score = (fluency
             - W_MISSING * (len(missing) / len(asked) if asked else 0.0)
             - W_NEW_NAMES * max(0, len(new_names) - FREE_NEW_NAMES)
             - W_REPEAT * rep)
    return Score(score, perplexity, missing, new_names, rep)
