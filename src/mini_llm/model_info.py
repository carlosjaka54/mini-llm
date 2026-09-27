"""Etapa 4: inspección del modelo antes de entrenar.

Construye el GPT con la configuración elegida y muestra:
  - cuántos parámetros tiene y dónde están,
  - cuánta memoria necesitará el entrenamiento,
  - la pérdida inicial (debe rondar ln(V): el modelo "adivina al azar"),
  - la velocidad real de un paso de entrenamiento en tu PC,
  - un texto generado SIN entrenar (para comparar después).

Uso:
    mini-llm-model
    mini-llm-model --config configs/gpu.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import torch

from mini_llm.config import Config
from mini_llm.dataset import get_batch, load_meta, load_tokens
from mini_llm.model import GPT, GPTConfig
from mini_llm.tokenizer import load_tokenizer
from mini_llm.utils import CONFIGS_DIR, configure_threads, get_logger, resolve_device, set_seed

PROMPT = "Había una vez"


def time_train_step(model: GPT, x: torch.Tensor, y: torch.Tensor, device: str, repeats: int = 5) -> float:
    """Segundos por paso de entrenamiento (forward + backward), medido de verdad."""
    model.train()

    def step() -> None:
        model.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()

    for _ in range(2):  # calentamiento
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    return (time.perf_counter() - t0) / repeats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Etapa 4: inspeccionar el modelo")
    parser.add_argument("--config", default=str(CONFIGS_DIR / "cpu.yaml"))
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    log = get_logger("model")
    set_seed(cfg.seed)
    configure_threads(cfg.train.num_threads)
    device = resolve_device(cfg.device)

    meta = load_meta()
    vocab_size = meta["vocab_size"]
    if vocab_size != cfg.data.vocab_size:
        log.warning("El tokenizador tiene %d tokens y la config dice %d. Se usa %d.",
                    vocab_size, cfg.data.vocab_size, vocab_size)

    gcfg = GPTConfig.from_config(cfg, vocab_size)
    model = GPT(gcfg).to(device)
    counts = model.num_parameters()
    total = counts["total"]

    print(model)
    log.info("-" * 60)
    log.info("Parámetros del modelo (%s)", cfg.experiment_name)
    for name, n in counts.items():
        if name != "total":
            log.info("   %-22s %10s  (%4.1f %%)", name, f"{n:,}", 100 * n / total)
    log.info("   %-22s %10s", "TOTAL", f"{total:,}")
    log.info("Memoria de los pesos   : %.1f MB (float32)", total * 4 / 1e6)
    log.info("Memoria al entrenar    : ~%.0f MB (pesos + gradientes + 2 estados de AdamW)",
             total * 4 * 4 / 1e6)

    # Pérdida inicial: con pesos aleatorios, todas las opciones son ~igual de probables
    x, y = get_batch(load_tokens("val"), cfg.train.batch_size, cfg.data.block_size, device)
    model.eval()
    with torch.no_grad():
        _, loss = model(x, y)
    expected = math.log(vocab_size)
    log.info("Pérdida inicial        : %.3f (esperada ≈ ln(%d) = %.3f)", loss.item(), vocab_size, expected)
    if abs(loss.item() - expected) > 0.5:
        log.warning("La pérdida inicial se aleja de lo esperado: revisa la inicialización.")

    # Velocidad real en este PC
    xb, yb = get_batch(load_tokens("train"), cfg.train.batch_size, cfg.data.block_size, device)
    sec = time_train_step(model, xb, yb, device)
    log.info("Paso de entrenamiento  : %.2f s (batch=%d x %d tokens) en %s",
             sec, cfg.train.batch_size, cfg.data.block_size, device)
    log.info("Estimación total       : ~%.0f min para %d iteraciones",
             sec * cfg.train.max_iters * 1.05 / 60, cfg.train.max_iters)

    # Texto generado por el modelo SIN entrenar
    tok = load_tokenizer()
    idx = torch.tensor([tok.encode(PROMPT).ids], dtype=torch.long, device=device)
    model.eval()
    out = model.generate(idx, max_new_tokens=40, temperature=1.0)
    log.info("Texto SIN entrenar     : %s", tok.decode(out[0].tolist()))
    log.info("(Es ruido: el modelo aún no sabe nada. Tras entrenar compararemos.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
