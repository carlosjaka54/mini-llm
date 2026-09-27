"""Lectura de los tokens en disco y construcción de lotes (batches).

Un lote son `batch_size` ventanas aleatorias de `block_size` tokens:

    tokens:  [Había] [una] [vez] [un] [gato] [que] ...
    x     :  [Había] [una] [vez] [un]            <- lo que el modelo lee
    y     :          [una] [vez] [un] [gato]     <- lo que debe predecir

`y` es `x` desplazado una posición: en CADA posición el modelo aprende a
predecir el token siguiente. Una ventana de 128 tokens son 128 ejercicios.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from mini_llm.utils import PROCESSED_DIR


def load_meta() -> dict:
    path = PROCESSED_DIR / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero: mini-llm-prepare")
    return json.loads(path.read_text(encoding="utf-8"))


def load_tokens(split: str) -> np.memmap:
    """Abre data/processed/{split}.bin sin cargarlo en RAM (memoria mapeada)."""
    path = PROCESSED_DIR / f"{split}.bin"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero: mini-llm-prepare")
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Devuelve (x, y), ambos de forma (batch_size, block_size) y tipo int64."""
    if len(data) <= block_size + 1:
        raise ValueError(f"Hay muy pocos tokens ({len(data)}) para block_size={block_size}")
    starts = torch.randint(len(data) - block_size, (batch_size,), generator=generator).tolist()
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in starts])
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in starts]
    )
    if device == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)
