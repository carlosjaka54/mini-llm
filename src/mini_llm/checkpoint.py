"""Etapa 6: guardar y cargar checkpoints.

Un checkpoint es una "foto" del entrenamiento. Guardamos dos tipos:

  last.pt  -> pesos + estado del optimizador + iteración + RNG. Sirve para REANUDAR.
              (~3x el tamaño del modelo: AdamW guarda 2 valores extra por parámetro)
  best.pt  -> solo los pesos con la mejor pérdida de validación. Sirve para USAR el modelo.

Ver los checkpoints existentes:
    mini-llm-ckpt
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from mini_llm.config import Config
from mini_llm.model import GPT, GPTConfig
from mini_llm.utils import CHECKPOINTS_DIR

FORMAT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: GPT,
    optimizer: torch.optim.Optimizer | None,
    cfg: Config,
    iter_num: int,
    best_val_loss: float,
    tokenizer_sha256: str = "",
) -> None:
    """Guarda de forma atómica (.tmp -> .pt): un corte de luz nunca deja un archivo corrupto."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "model": model.state_dict(),
        "model_args": asdict(model.config),
        "config": cfg.to_dict(),
        "iter": iter_num,
        "best_val_loss": float(best_val_loss),
        "tokenizer_sha256": tokenizer_sha256,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rng_state": torch.get_rng_state(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict:
    """Carga con weights_only=True: no ejecuta código arbitrario escondido en el archivo."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el checkpoint {path}")
    return torch.load(path, map_location=device, weights_only=True)


def load_model(path: str | Path, device: str = "cpu") -> tuple[GPT, dict]:
    """Reconstruye el modelo a partir del checkpoint y lo deja en modo evaluación."""
    ckpt = load_checkpoint(path, device)
    model = GPT(GPTConfig(**ckpt["model_args"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lista los checkpoints guardados")
    parser.parse_args(argv)
    files = sorted(CHECKPOINTS_DIR.glob("*/*.pt"))
    if not files:
        print("No hay checkpoints todavía. Entrena con: mini-llm-train")
        return 0
    print(f"\n{'Archivo':<32} {'Tamaño':>9} {'Iter':>6} {'Mejor val':>10} {'Optimizador':>12}  Fecha (UTC)")
    print("-" * 95)
    for f in files:
        ck = load_checkpoint(f)
        print(
            f"{f.relative_to(CHECKPOINTS_DIR)!s:<32} {f.stat().st_size / 1e6:>7.1f}MB "
            f"{ck['iter']:>6} {ck['best_val_loss']:>10.4f} "
            f"{'sí' if 'optimizer' in ck else 'no':>12}  {ck['created_at']}"
        )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
