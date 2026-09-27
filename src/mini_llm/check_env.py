"""Etapa 1: diagnóstico del entorno y estimación del tiempo de entrenamiento.

Uso:
    mini-llm-check
    mini-llm-check --config configs/gpu.yaml
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
import time

import psutil
import torch

from mini_llm import __version__
from mini_llm.config import Config
from mini_llm.utils import CONFIGS_DIR, PROJECT_ROOT, get_logger, resolve_device

MIN_RAM_GB = 8
MIN_DISK_GB = 15
MATMUL_EFFICIENCY = 0.3  # un Transformer pequeño aprovecha solo parte del pico medido


def benchmark_gflops(device: str, size: int = 1024, repeats: int = 20) -> float:
    """Mide GFLOPS con multiplicaciones de matrices, la operación central de un Transformer."""
    x = torch.randn(size, size, device=device)
    _ = x @ x  # calentamiento

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    sync()
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = x @ x
    sync()
    elapsed = time.perf_counter() - t0
    return repeats * 2 * size**3 / elapsed / 1e9


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnóstico del entorno para mini-llm")
    parser.add_argument("--config", default=str(CONFIGS_DIR / "cpu.yaml"))
    args = parser.parse_args(argv)

    log = get_logger("check_env")
    cfg = Config.from_yaml(args.config)
    device = resolve_device(cfg.device)
    ok = True

    log.info("mini-llm v%s | raíz del proyecto: %s", __version__, PROJECT_ROOT)
    log.info("Python  : %s (%s %s)", platform.python_version(), platform.system(), platform.release())
    log.info("PyTorch : %s", torch.__version__)

    if sys.version_info < (3, 10):  # noqa: UP036
        log.error("Se requiere Python >= 3.10")
        ok = False

    ram_gb = psutil.virtual_memory().total / 1e9
    log.info("CPU     : %s núcleos lógicos | hilos de PyTorch: %s", os.cpu_count(), torch.get_num_threads())
    log.info("RAM     : %.1f GB", ram_gb)
    if ram_gb < MIN_RAM_GB:
        log.warning("Menos de %d GB de RAM: reduce batch_size si hay problemas.", MIN_RAM_GB)

    disk_gb = shutil.disk_usage(PROJECT_ROOT).free / 1e9
    log.info("Disco   : %.1f GB libres en %s", disk_gb, PROJECT_ROOT.anchor)
    if disk_gb < MIN_DISK_GB:
        log.error("Se recomiendan al menos %d GB libres.", MIN_DISK_GB)
        ok = False

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        log.info("GPU     : %s (%.1f GB VRAM)", props.name, props.total_memory / 1e9)
    elif device == "mps":
        log.info("GPU     : Apple Silicon (MPS)")
    else:
        log.info("GPU     : no se usará GPU; entrenamiento en CPU")

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        log.info("HF_HOME : %s", hf_home)
    else:
        log.warning("HF_HOME no definido: los datasets se descargarán en la carpeta de usuario (C:).")

    log.info("-" * 60)
    log.info("Experimento : %s  (config: %s)", cfg.experiment_name, args.config)
    log.info(
        "Modelo      : %d capas, %d cabezas, n_embd=%d, contexto=%d, vocab=%d",
        cfg.model.n_layer, cfg.model.n_head, cfg.model.n_embd,
        cfg.data.block_size, cfg.data.vocab_size,
    )
    log.info("Parámetros  : ~%.1f M", cfg.approx_params / 1e6)

    gflops = benchmark_gflops(device)
    flops_per_iter = 6 * cfg.approx_params * cfg.tokens_per_iter  # forward + backward
    sec_per_iter = flops_per_iter / (gflops * 1e9 * MATMUL_EFFICIENCY)
    total_min = sec_per_iter * cfg.train.max_iters / 60
    log.info("Rendimiento : ~%.0f GFLOPS en %s", gflops, device)
    log.info("Estimación  : ~%.2f s/iter -> ~%.0f min para %d iteraciones (orientativo)",
             sec_per_iter, total_min, cfg.train.max_iters)

    log.info("-" * 60)
    if ok:
        log.info("Entorno listo para continuar con la etapa 2.")
    else:
        log.error("Hay problemas que resolver antes de continuar.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
