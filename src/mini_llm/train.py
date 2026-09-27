"""Etapa 5: entrenamiento (pretraining) del modelo.

El bucle repite miles de veces:
  1. Tomar un lote de ventanas de texto (x) y sus siguientes tokens (y).
  2. Forward: el modelo predice y se mide el error (entropía cruzada).
  3. Backward: se calcula cómo cambiar cada peso para reducir el error (gradientes).
  4. El optimizador AdamW ajusta los pesos un poquito en esa dirección.

Cada `eval_interval` iteraciones medimos la pérdida en train y en validación,
guardamos el mejor modelo y generamos un texto de muestra para ver el progreso.

Uso:
    mini-llm-train                               # entrenamiento completo
    mini-llm-train --max-iters 30 --run-name prueba   # prueba rápida
    mini-llm-train --resume                      # continuar tras interrumpir (Ctrl+C)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mini_llm.checkpoint import load_checkpoint, save_checkpoint
from mini_llm.config import Config, TrainConfig
from mini_llm.dataset import get_batch, load_meta, load_tokens
from mini_llm.model import GPT, GPTConfig
from mini_llm.tokenizer import EOT, load_tokenizer
from mini_llm.utils import (
    CHECKPOINTS_DIR,
    CONFIGS_DIR,
    LOGS_DIR,
    configure_threads,
    get_logger,
    resolve_device,
    set_seed,
)

SAMPLE_PROMPT = "Había una vez"
METRICS_FIELDS = ["iter", "train_loss", "val_loss", "val_perplexity", "lr", "elapsed_min"]
EVAL_SEED = 1234  # mismos lotes de evaluación siempre: comparaciones justas


def get_lr(it: int, t: TrainConfig) -> float:
    """Calentamiento lineal y luego descenso en coseno hasta min_lr.

    Al principio los pesos son aleatorios y pasos grandes desestabilizan: por eso
    se sube poco a poco. Al final, pasos pequeños afinan el resultado.
    """
    if it < t.warmup_iters:
        return t.learning_rate * (it + 1) / t.warmup_iters
    if it >= t.max_iters:
        return t.min_lr
    progress = (it - t.warmup_iters) / (t.max_iters - t.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # de 1 a 0
    return t.min_lr + coeff * (t.learning_rate - t.min_lr)


def configure_optimizer(model: GPT, weight_decay: float, lr: float) -> torch.optim.AdamW:
    """AdamW con weight decay solo en matrices (2D); no en LayerNorm ni sesgos."""
    decay, no_decay = [], []
    for p in model.parameters():  # parameters() no repite los pesos compartidos
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


@torch.no_grad()
def estimate_loss(model: GPT, splits: dict[str, np.ndarray], cfg: Config, device: str) -> dict[str, float]:
    """Promedio de la pérdida sobre `eval_iters` lotes (siempre los mismos) por split."""
    model.eval()
    out = {}
    for name, data in splits.items():
        gen = torch.Generator().manual_seed(EVAL_SEED)
        losses = []
        for _ in range(cfg.train.eval_iters):
            x, y = get_batch(data, cfg.train.batch_size, cfg.data.block_size, device, gen)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def sample_text(model: GPT, tokenizer, device: str, max_new_tokens: int = 40) -> str:
    """Texto de muestra, sin alterar la aleatoriedad del entrenamiento."""
    idx = torch.tensor([tokenizer.encode(SAMPLE_PROMPT).ids], dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    with torch.random.fork_rng(devices=[]):
        out = model.generate(idx, max_new_tokens, temperature=0.8, top_k=50,
                             eot_id=tokenizer.token_to_id(EOT))
    model.train(was_training)
    return tokenizer.decode(out[0].tolist()).replace("\n", " ")


def plot_metrics(metrics_path: Path, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # sin ventana: solo guarda la imagen
    import matplotlib.pyplot as plt

    with metrics_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    its = [int(r["iter"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(its, [float(r["train_loss"]) for r in rows], marker="o", label="entrenamiento")
    ax.plot(its, [float(r["val_loss"]) for r in rows], marker="o", label="validación")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Pérdida (entropía cruzada)")
    ax.set_title("Curva de aprendizaje")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def train(
    cfg: Config,
    train_data: np.ndarray,
    val_data: np.ndarray,
    run_dir: Path,
    log,
    vocab_size: int,
    resume: bool = False,
    tokenizer=None,
    tokenizer_sha256: str = "",
) -> dict:
    t = cfg.train
    device = resolve_device(cfg.device)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")  # la receta exacta queda junto a los pesos
    last_path, best_path, metrics_path = run_dir / "last.pt", run_dir / "best.pt", run_dir / "metrics.csv"

    model = GPT(GPTConfig.from_config(cfg, vocab_size)).to(device)
    optimizer = configure_optimizer(model, t.weight_decay, t.learning_rate)
    start_iter, best_val = 0, float("inf")

    if resume:
        ckpt = load_checkpoint(last_path, device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        torch.set_rng_state(ckpt["rng_state"].cpu())
        start_iter, best_val = ckpt["iter"] + 1, ckpt["best_val_loss"]
        log.info("Reanudando desde la iteración %d (mejor val hasta ahora: %.4f)", start_iter, best_val)
    else:
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(METRICS_FIELDS)

    def save(path: Path, it: int, with_optimizer: bool) -> None:
        save_checkpoint(path, model, optimizer if with_optimizer else None, cfg, it, best_val,
                        tokenizer_sha256)

    log.info("Entrenando en %s | %d iteraciones | %s tokens por iteración",
             device, t.max_iters, f"{cfg.tokens_per_iter:,}")
    model.train()
    t_start = time.perf_counter()
    train_time, steps_done, running, it = 0.0, 0, None, start_iter

    try:
        for it in range(start_iter, t.max_iters + 1):
            lr = get_lr(it, t)
            for group in optimizer.param_groups:
                group["lr"] = lr

            # --- Evaluación periódica -------------------------------------------------
            if it % t.eval_interval == 0 or it == t.max_iters:
                losses = estimate_loss(model, {"train": train_data, "val": val_data}, cfg, device)
                ppl = math.exp(losses["val"])
                elapsed = (time.perf_counter() - t_start) / 60
                log.info("[eval] iter %5d | train %.4f | val %.4f | perplejidad %.1f",
                         it, losses["train"], losses["val"], ppl)
                with metrics_path.open("a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([it, f"{losses['train']:.5f}", f"{losses['val']:.5f}",
                                            f"{ppl:.3f}", f"{lr:.3e}", f"{elapsed:.2f}"])
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    save(best_path, it, with_optimizer=False)
                    log.info("[ckpt] nuevo mejor modelo -> %s", best_path.name)
                if tokenizer is not None:
                    log.info("[muestra] %s", sample_text(model, tokenizer, device))

            if it == t.max_iters:
                break  # la última vuelta solo evalúa

            # --- Paso de entrenamiento ------------------------------------------------
            t0 = time.perf_counter()
            x, y = get_batch(train_data, t.batch_size, cfg.data.block_size, device)
            _, loss = model(x, y)                      # 1-2. forward + pérdida
            optimizer.zero_grad(set_to_none=True)
            loss.backward()                            # 3. gradientes
            if t.grad_clip > 0:                        # evita pasos gigantes por un lote raro
                torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            optimizer.step()                           # 4. actualizar pesos
            train_time += time.perf_counter() - t0
            steps_done += 1

            value = loss.item()
            running = value if running is None else 0.9 * running + 0.1 * value  # media suavizada
            if it % t.log_interval == 0:
                sec_it = train_time / steps_done
                eta_min = sec_it * (t.max_iters - it - 1) / 60
                log.info("iter %5d/%d | loss %.4f | lr %.2e | %.2f s/iter | %s tok/s | ETA %.0f min",
                         it, t.max_iters, running, lr, sec_it,
                         f"{cfg.tokens_per_iter / sec_it:,.0f}", eta_min)
            if it > 0 and it % t.checkpoint_interval == 0:
                save(last_path, it, with_optimizer=True)
                log.info("[ckpt] guardado %s (iter %d)", last_path.name, it)

    except KeyboardInterrupt:
        it = max(it - 1, start_iter - 1)
        save(last_path, it, with_optimizer=True)
        log.warning("Interrumpido. Estado guardado en iter %d. Continúa con: mini-llm-train --resume", it)
        plot_metrics(metrics_path, run_dir / "loss.png")
        return {"best_val_loss": best_val, "iters": it, "interrupted": True}

    save(last_path, t.max_iters, with_optimizer=True)
    plot_metrics(metrics_path, run_dir / "loss.png")
    minutes = (time.perf_counter() - t_start) / 60
    return {"best_val_loss": best_val, "iters": t.max_iters, "minutes": minutes, "interrupted": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Etapa 5: entrenar el modelo")
    parser.add_argument("--config", default=str(CONFIGS_DIR / "cpu.yaml"))
    parser.add_argument("--resume", action="store_true", help="continuar desde last.pt")
    parser.add_argument("--overwrite", action="store_true", help="empezar de cero aunque exista")
    parser.add_argument("--max-iters", type=int, help="sobrescribe max_iters (mínimo 10)")
    parser.add_argument("--run-name", help="nombre del experimento (carpeta de checkpoints)")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    if args.run_name:
        cfg.experiment_name = args.run_name
    if args.max_iters:
        if args.max_iters < 10:
            parser.error("--max-iters debe ser al menos 10")
        t = cfg.train
        t.max_iters = args.max_iters
        t.warmup_iters = min(t.warmup_iters, max(1, args.max_iters // 10))
        t.eval_interval = min(t.eval_interval, args.max_iters)
        t.checkpoint_interval = min(t.checkpoint_interval, args.max_iters)
        t.log_interval = min(t.log_interval, max(1, args.max_iters // 10))
    cfg.validate()

    run_dir = CHECKPOINTS_DIR / cfg.experiment_name
    log = get_logger("train", LOGS_DIR / f"train_{cfg.experiment_name}.log")
    if args.resume and not (run_dir / "last.pt").exists():
        log.error("No hay nada que reanudar en %s", run_dir)
        return 1
    if not args.resume and (run_dir / "last.pt").exists() and not args.overwrite:
        log.error("Ya existe un entrenamiento en %s.", run_dir)
        log.error("Usa --resume para continuarlo, --overwrite para empezar de cero "
                  "o --run-name para otro nombre.")
        return 1

    set_seed(cfg.seed)
    configure_threads(cfg.train.num_threads)
    meta = load_meta()
    result = train(
        cfg,
        load_tokens("train"),
        load_tokens("val"),
        run_dir,
        log,
        vocab_size=meta["vocab_size"],
        resume=args.resume,
        tokenizer=load_tokenizer(),
        tokenizer_sha256=meta["tokenizer_sha256"],
    )

    if result["interrupted"]:
        return 130
    log.info("=" * 60)
    log.info("Entrenamiento terminado en %.1f min", result["minutes"])
    log.info("Mejor pérdida de validación: %.4f (perplejidad %.1f)",
             result["best_val_loss"], math.exp(result["best_val_loss"]))
    log.info("Resultados en %s: best.pt, last.pt, metrics.csv, loss.png, config.yaml", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
