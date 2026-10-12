"""Estrategias de muestreo: cómo elegir el siguiente token a partir de los logits.

El modelo devuelve una puntuación (logit) por cada token del vocabulario. Antes de
sortear el siguiente, se pueden aplicar varios filtros, en este orden:

  1. Penalización de repetición: baja la puntuación de los tokens que ya aparecieron.
     Ataca los bucles del tipo "El oso dijo: ¡Hola! El oso dijo: ¡Hola!...".
  2. Temperatura: divide los logits. <1 concentra la probabilidad, >1 la reparte.
  3. Top-k: solo quedan los k tokens con mayor puntuación.
  4. Top-p (núcleo): solo quedan los tokens más probables cuya suma alcanza p.
     A diferencia de top-k, se adapta: si el modelo está seguro, quedan pocos;
     si duda, quedan muchos.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class SamplingParams:
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float | None = None          # None o 1.0 = sin filtro de núcleo
    repetition_penalty: float = 1.0     # 1.0 = sin penalización; 1.1-1.3 es lo habitual


def apply_repetition_penalty(logits: torch.Tensor, context: torch.Tensor, penalty: float) -> torch.Tensor:
    """Penaliza los tokens que ya están en el contexto (método de CTRL, Keskar et al. 2019).

    Un logit positivo se divide entre `penalty` y uno negativo se multiplica: en ambos
    casos el token se vuelve menos probable.
    """
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    for b in range(logits.size(0)):
        seen = torch.unique(context[b])
        scores = logits[b, seen]
        logits[b, seen] = torch.where(scores > 0, scores / penalty, scores * penalty)
    return logits


def filter_top_k(logits: torch.Tensor, k: int | None) -> torch.Tensor:
    if k is None or k <= 0 or k >= logits.size(-1):
        return logits
    kth = torch.topk(logits, k).values[:, [-1]]
    return logits.masked_fill(logits < kth, float("-inf"))


def filter_top_p(logits: torch.Tensor, p: float | None) -> torch.Tensor:
    """Conserva el menor grupo de tokens cuya probabilidad acumulada llega a `p`."""
    if p is None or p >= 1.0:
        return logits
    sorted_logits, order = torch.sort(logits, descending=True)
    cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    # Se elimina un token si la suma de los ANTERIORES ya alcanzó p (el más probable siempre queda)
    remove_sorted = (cumulative - F.softmax(sorted_logits, dim=-1)) >= p
    remove = remove_sorted.scatter(-1, order, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def next_token_distribution(logits: torch.Tensor, context: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """Probabilidades finales tras aplicar todos los filtros (B, V). Temperatura 0 = argmax."""
    logits = apply_repetition_penalty(logits, context, params.repetition_penalty)
    if params.temperature <= 0:
        probs = torch.zeros_like(logits)
        probs.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return probs
    logits = logits / params.temperature
    logits = filter_top_k(logits, params.top_k)
    logits = filter_top_p(logits, params.top_p)
    return F.softmax(logits, dim=-1)


def sample(probs: torch.Tensor) -> torch.Tensor:
    """Sortea un token por fila según `probs`. Devuelve (B, 1)."""
    return torch.multinomial(probs, num_samples=1)
