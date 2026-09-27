"""Etapa 7: evaluación del modelo entrenado.

Medimos el modelo de varias formas complementarias:

  1. Pérdida y perplejidad sobre TODO el conjunto de validación (no solo unos lotes).
  2. Precisión top-1 / top-5: ¿el token correcto está entre las primeras opciones?
  3. Comparación con modelos de referencia (baselines):
       - uniforme : todos los tokens igual de probables          -> ln(V)
       - unigrama : solo conoce la frecuencia de cada token
       - bigrama  : predice mirando únicamente el token anterior
     Si el Transformer no supera al bigrama, la atención no está aportando nada.
  4. Pérdida según la posición en el contexto: ¿predice mejor con más texto previo?
  5. Muestras de texto y su diversidad (distinct-2): detecta repeticiones.

Uso:
    mini-llm-eval
    mini-llm-eval --checkpoint artifacts/checkpoints/prueba/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mini_llm.checkpoint import load_model
from mini_llm.config import Config
from mini_llm.dataset import load_meta, load_tokens
from mini_llm.model import GPT
from mini_llm.tokenizer import EOT, load_tokenizer
from mini_llm.utils import (
    ARTIFACTS_DIR,
    CHECKPOINTS_DIR,
    CONFIGS_DIR,
    configure_threads,
    get_logger,
    resolve_device,
)

PROMPTS = [
    "Había una vez",
    "Un día, un niño llamado Tomás",
    "La niña tenía un gato que",
    "En el bosque vivía un oso",
    "Mamá dijo:",
]


# --- Modelos de referencia -----------------------------------------------------------
def unigram_loss(train: np.ndarray, val: np.ndarray, vocab_size: int) -> float:
    """Pérdida de un modelo que solo conoce la frecuencia de cada token (suavizado Laplace)."""
    counts = np.bincount(train, minlength=vocab_size).astype(np.float64) + 1.0
    probs = counts / counts.sum()
    return float(-np.log(probs[val]).mean())


def bigram_loss(train: np.ndarray, val: np.ndarray, vocab_size: int, alpha: float = 0.1) -> float:
    """Pérdida de un modelo que predice mirando solo el token anterior (suavizado aditivo)."""
    a = train[:-1].astype(np.int64)
    b = train[1:].astype(np.int64)
    pairs = np.bincount(a * vocab_size + b, minlength=vocab_size * vocab_size)
    pairs = pairs.reshape(vocab_size, vocab_size)
    rows = pairs.sum(axis=1)
    va = val[:-1].astype(np.int64)
    vb = val[1:].astype(np.int64)
    p = (pairs[va, vb] + alpha) / (rows[va] + alpha * vocab_size)
    return float(-np.log(p).mean())


# --- Evaluación del Transformer --------------------------------------------------------
@torch.no_grad()
def evaluate_model(
    model: GPT,
    data: np.ndarray,
    batch_size: int = 64,
    device: str = "cpu",
    max_tokens: int | None = None,
) -> dict:
    """Recorre los datos en ventanas consecutivas sin solaparse (cobertura completa)."""
    model.eval()
    T = model.config.block_size
    V = model.config.vocab_size
    if max_tokens is not None:
        data = data[: max_tokens + 1]
    n_windows = (len(data) - 1) // T
    if n_windows == 0:
        raise ValueError("No hay suficientes tokens para evaluar")

    loss_sum, count, top1, top5 = 0.0, 0, 0, 0
    pos_loss = torch.zeros(T, dtype=torch.float64)
    for start in range(0, n_windows, batch_size):
        windows = range(start, min(start + batch_size, n_windows))
        x = torch.stack([torch.from_numpy(data[i * T : i * T + T].astype(np.int64)) for i in windows])
        y = torch.stack(
            [torch.from_numpy(data[i * T + 1 : i * T + T + 1].astype(np.int64)) for i in windows]
        )
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="none").view_as(y)

        loss_sum += loss.sum().item()
        count += loss.numel()
        pos_loss += loss.sum(dim=0).double().cpu()
        best5 = logits.topk(5, dim=-1).indices              # (B, T, 5)
        top1 += (best5[..., 0] == y).sum().item()
        top5 += (best5 == y.unsqueeze(-1)).any(dim=-1).sum().item()

    mean = loss_sum / count
    return {
        "loss": mean,
        "perplexity": math.exp(mean),
        "top1_accuracy": top1 / count,
        "top5_accuracy": top5 / count,
        "tokens": count,
        "loss_by_position": (pos_loss / n_windows).tolist(),
    }


def distinct_n(texts: list[str], n: int = 2) -> float:
    """Proporción de n-gramas de palabras distintos. 1.0 = nada repetido; bajo = repetitivo."""
    total, unique = 0, set()
    for text in texts:
        words = text.split()
        grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


def generate_samples(model: GPT, tokenizer, device: str, max_new_tokens: int = 120) -> list[dict]:
    eot = tokenizer.token_to_id(EOT)
    samples = []
    for i, prompt in enumerate(PROMPTS):
        torch.manual_seed(1000 + i)  # reproducible
        idx = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens, temperature=0.8, top_k=50, eot_id=eot)
        samples.append({"prompt": prompt, "text": tokenizer.decode(out[0].tolist()).strip()})
    return samples


# --- Reporte ------------------------------------------------------------------------------
def plot_loss_by_position(values: list[float], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(values) + 1), values)
    ax.set_xlabel("Tokens de contexto disponibles")
    ax.set_ylabel("Pérdida media")
    ax.set_title("¿Predice mejor con más contexto?")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_report(path: Path, r: dict) -> None:
    m, b = r["model_metrics"], r["baselines"]
    lines = [
        f"# Evaluación: {r['experiment']}",
        "",
        f"- Checkpoint: `{r['checkpoint']}` (iteración {r['iter']})",
        f"- Parámetros: {r['parameters']:,}",
        f"- Tokens evaluados: {m['tokens']:,} (conjunto de validación completo)",
        f"- Fecha: {r['date']}",
        "",
        "## Métricas",
        "",
        "| Modelo | Pérdida | Perplejidad |",
        "|---|---|---|",
        f"| Uniforme (azar) | {b['uniform']:.3f} | {math.exp(b['uniform']):,.0f} |",
        f"| Unigrama (frecuencias) | {b['unigram']:.3f} | {math.exp(b['unigram']):,.1f} |",
        f"| Bigrama (token anterior) | {b['bigram']:.3f} | {math.exp(b['bigram']):,.1f} |",
        f"| **mini-GPT** | **{m['loss']:.3f}** | **{m['perplexity']:.1f}** |",
        "",
        f"- Precisión top-1: **{100 * m['top1_accuracy']:.1f} %** del siguiente token acertado",
        f"- Precisión top-5: **{100 * m['top5_accuracy']:.1f} %** con el correcto entre los 5 primeros",
        (
            f"- Pérdida con 1 token de contexto: {m['loss_by_position'][0]:.3f}; "
            f"con {len(m['loss_by_position'])}: {m['loss_by_position'][-1]:.3f}"
        ),
        f"- Diversidad de las muestras (distinct-2): {r['distinct_2']:.2f}",
        "",
        "![Pérdida por posición](loss_by_position.png)",
        "",
        "## Muestras (temperatura 0.8, top-k 50)",
        "",
    ]
    for s in r["samples"]:
        lines += [f"**{s['prompt']}**", "", f"> {s['text'].replace(chr(10), ' ')}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Etapa 7: evaluar el modelo")
    parser.add_argument("--config", default=str(CONFIGS_DIR / "cpu.yaml"))
    parser.add_argument("--checkpoint", help="por defecto: best.pt del experimento")
    parser.add_argument("--max-tokens", type=int, help="limitar tokens de validación (más rápido)")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    log = get_logger("eval")
    configure_threads(cfg.train.num_threads)
    device = resolve_device(cfg.device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else CHECKPOINTS_DIR / cfg.experiment_name / "best.pt"
    out_dir = ARTIFACTS_DIR / "eval" / ckpt_path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_model(ckpt_path, device)
    meta = load_meta()
    if ckpt.get("tokenizer_sha256") and ckpt["tokenizer_sha256"] != meta["tokenizer_sha256"]:
        log.warning("El tokenizador actual NO es el usado al entrenar este checkpoint.")
    tokenizer = load_tokenizer()
    V = model.config.vocab_size
    log.info("Checkpoint %s | iteración %d | %s parámetros",
             ckpt_path, ckpt["iter"], f"{model.num_parameters()['total']:,}")

    train, val = load_tokens("train"), load_tokens("val")
    if args.max_tokens:
        val = val[: args.max_tokens + 1]

    t0 = time.perf_counter()
    log.info("[1/3] Modelos de referencia (uniforme, unigrama, bigrama)...")
    baselines = {
        "uniform": math.log(V),
        "unigram": unigram_loss(np.asarray(train), np.asarray(val), V),
        "bigram": bigram_loss(np.asarray(train), np.asarray(val), V),
    }

    log.info("[2/3] Evaluando el Transformer sobre %s tokens de validación...", f"{len(val):,}")
    metrics = evaluate_model(model, val, batch_size=64, device=device)

    log.info("[3/3] Generando muestras...")
    samples = generate_samples(model, tokenizer, device)
    distinct2 = distinct_n([s["text"] for s in samples], 2)

    results = {
        "experiment": ckpt_path.parent.name,
        "checkpoint": str(ckpt_path),
        "iter": ckpt["iter"],
        "parameters": model.num_parameters()["total"],
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "model_metrics": metrics,
        "baselines": baselines,
        "distinct_2": distinct2,
        "samples": samples,
    }
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    plot_loss_by_position(metrics["loss_by_position"], out_dir / "loss_by_position.png")
    write_report(out_dir / "report.md", results)

    log.info("-" * 60)
    log.info("%-26s %8s %12s", "Modelo", "Pérdida", "Perplejidad")
    for name, value in [("Uniforme (azar)", baselines["uniform"]),
                        ("Unigrama (frecuencias)", baselines["unigram"]),
                        ("Bigrama (token anterior)", baselines["bigram"]),
                        ("mini-GPT", metrics["loss"])]:
        log.info("%-26s %8.3f %12.1f", name, value, math.exp(value))
    log.info("Precisión top-1 / top-5  : %.1f %% / %.1f %%",
             100 * metrics["top1_accuracy"], 100 * metrics["top5_accuracy"])
    lp = metrics["loss_by_position"]
    log.info("Pérdida según contexto   : %.3f (1 token) -> %.3f (%d tokens)", lp[0], lp[-1], len(lp))
    log.info("Diversidad distinct-2    : %.2f", distinct2)
    for s in samples:
        log.info("[muestra] %s", s["text"].replace("\n", " "))
    log.info("Reporte en %s (%.0f s)", out_dir / "report.md", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
