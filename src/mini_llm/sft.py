"""Etapa 9: ajuste fino supervisado (SFT) para convertir el modelo base en un chatbot.

El modelo base solo sabe CONTINUAR texto. Para que responda peticiones le mostramos
miles de conversaciones con este formato:

    <|usuario|>Cuéntame un cuento sobre un perro llamado Spot.<|asistente|>Había una vez...<|endoftext|>

1. Dataset de instrucciones: a partir de cada cuento se genera automáticamente la petición
   que le correspondería ("un perro llamado Spot" -> "Cuéntame un cuento sobre...").
2. Tokens especiales nuevos: <|usuario|> y <|asistente|> marcan quién habla.
3. Ajuste fino: se continúa entrenando desde el modelo base (no desde cero) y la pérdida
   se calcula SOLO sobre la respuesta: el modelo aprende a responder, no a pedir.
4. Medición: porcentaje de respuestas que mencionan lo pedido, antes y después del ajuste.

Uso:
    mini-llm-sft                  # construye el dataset y entrena (~30-40 min en CPU)
    mini-llm-chat                 # conversar en la terminal
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from mini_llm.checkpoint import load_model, save_checkpoint
from mini_llm.config import Config
from mini_llm.generate import StreamDecoder
from mini_llm.model import GPT
from mini_llm.tokenizer import EOT, load_tokenizer
from mini_llm.train import METRICS_FIELDS, configure_optimizer, get_lr, plot_metrics
from mini_llm.utils import (
    CHECKPOINTS_DIR,
    CONFIGS_DIR,
    DATA_DIR,
    LOGS_DIR,
    PROJECT_ROOT,
    configure_threads,
    get_logger,
    processed_dir,
    resolve_device,
    set_seed,
    tokenizer_file,
)

USER = "<|usuario|>"
ASSISTANT = "<|asistente|>"
IGNORE = -100            # posiciones que no cuentan en la pérdida (F.cross_entropy las ignora)
CHAT_TOKENIZER = "chat"  # artifacts/tokenizer/chat/tokenizer.json (perfil por defecto)
SFT_DATA_DIR = DATA_DIR / "sft"


def chat_tokenizer_name(cfg: Config) -> str:
    """Cada corpus tiene su tokenizador de chat: chat/ (por defecto) o chat_<data.name>/."""
    return f"{CHAT_TOKENIZER}_{cfg.data.name}" if cfg.data.name else CHAT_TOKENIZER


def sft_data_dir(cfg: Config) -> Path:
    return SFT_DATA_DIR / cfg.data.name if cfg.data.name else SFT_DATA_DIR
MIN_RESPONSE_TOKENS = 20

# --- 1. Dataset de instrucciones ------------------------------------------------------------
_W = r"[a-záéíóúüñ]+"
_NAME = r"[A-ZÁÉÍÓÚÑ][a-záéíóúüñ]+"
NAMED_RE = re.compile(rf"\b([Uu]na?)\s+((?:{_W}\s+){{0,2}}?{_W})\s+(llamad[oa])\s+({_NAME})")
ONCE_RE = re.compile(rf"Había una vez,?\s+(un|una)\s+({_W})(?:\s+({_W}))?")
PLACE_RE = re.compile(r"\b(el parque|la playa|el bosque|la escuela|el jardín|el mar|el cielo|"
                      r"la granja|la selva|el río)\b")
STOP = {"que", "en", "y", "con", "muy", "a", "de", "la", "el", "los", "las", "se", "del", "al",
        "por", "para", "su", "sus", "un", "una", "era", "no", "le", "lo"}
NOT_SUBJECT = {"día", "vez", "momento", "lugar", "tiempo", "poco", "rato", "año", "mañana", "noche"}
TEMPLATES = [
    "Cuéntame un cuento sobre {s}.",
    "Escribe una historia corta sobre {s}.",
    "¿Me cuentas un cuento de {s}?",
    "Quiero un cuento sobre {s}.",
    "Inventa una historia con {s}.",
    "Escribe un cuento infantil sobre {s}.",
]


@dataclass
class Example:
    instruction: str
    response: str
    key: str  # palabra que la respuesta debería mencionar (para medir si obedece)


def make_instruction(story: str, rng: random.Random) -> tuple[str, str] | None:
    """Deduce la petición que le correspondería a un cuento. Devuelve (petición, palabra clave)."""
    m = NAMED_RE.search(story)
    if m:
        det, noun, called, name = m.groups()
        words = noun.split()
        # "una vez un perro llamado Spot": nos quedamos con lo que sigue al último artículo
        for i in range(len(words) - 1, -1, -1):
            if words[i] in {"un", "una"}:
                det, words = words[i], words[i + 1 :]
                break
        if not words or words[0] in STOP or words[0] in NOT_SUBJECT:
            return None
        subject, key = f"{det.lower()} {' '.join(words)} {called} {name}", name
    else:
        m = ONCE_RE.search(story)
        if not m:
            return None
        det, w1, w2 = m.groups()
        if w1 in STOP or w1 in NOT_SUBJECT:
            return None
        words = [w1] + ([w2] if w2 and w2 not in STOP else [])
        subject, key = f"{det} {' '.join(words)}", words[-1]
    extra = ""
    place = PLACE_RE.search(story)
    if place and rng.random() < 0.4:
        extra = f" en {place.group(1)}"
    elif "aprendi" in story and rng.random() < 0.25:
        extra = " que tenga una moraleja"
    return rng.choice(TEMPLATES).format(s=subject + extra), key


def add_chat_tokens(tokenizer) -> None:
    """Añade <|usuario|> y <|asistente|> al final del vocabulario (IDs nuevos)."""
    tokenizer.add_special_tokens([USER, ASSISTANT])


def chat_prompt_ids(tokenizer, message: str) -> list[int]:
    return tokenizer.encode(f"{USER}{message.strip()}{ASSISTANT}").ids


def plain_prompt_ids(tokenizer, message: str) -> list[int]:
    """La petición como texto normal (así la "ve" el modelo base, sin tokens de chat)."""
    return tokenizer.encode(message.strip() + "\n").ids


def encode_example(tokenizer, ex: Example, chat: bool = True) -> tuple[list[int], int]:
    """IDs de la conversación completa y la posición donde empieza la respuesta."""
    prompt = chat_prompt_ids(tokenizer, ex.instruction) if chat else plain_prompt_ids(tokenizer, ex.instruction)
    response = tokenizer.encode(ex.response).ids + [tokenizer.token_to_id(EOT)]
    return prompt + response, len(prompt)


_SENTENCE_END = re.compile(r'(?<=[.!?"])(\s+)')


def fit_response(tokenizer, story: str, budget: int) -> str:
    """El cuento completo si cabe en `budget` tokens; si no, sus primeras oraciones completas."""
    if len(tokenizer.encode(story).ids) <= budget:
        return story
    parts = _SENTENCE_END.split(story)  # [oración, espacio, oración, espacio, ...]
    text = ""
    for i in range(0, len(parts), 2):
        candidate = text + (parts[i - 1] if i else "") + parts[i]
        if len(tokenizer.encode(candidate).ids) > budget:
            break
        text = candidate
    return text


def collect_examples(tokenizer, stories: list[str], rng: random.Random, limit: int,
                     block_size: int, min_response: int = MIN_RESPONSE_TOKENS) -> list[Example]:
    """Crea ejemplos que caben en el contexto del modelo (block_size + 1 tokens).

    Los cuentos largos se recortan en una oración completa: el modelo aprende a contar
    cuentos cortos que caben en su ventana de contexto (block_size tokens).
    """
    out: list[Example] = []
    for story in stories:
        made = make_instruction(story, rng)
        if made is None:
            continue
        instruction, key = made
        budget = block_size - len(chat_prompt_ids(tokenizer, instruction))  # deja sitio para EOT
        response = fit_response(tokenizer, story, budget)
        if key.lower() not in response.lower():
            continue  # tras recortar, el cuento debe seguir tratando de lo pedido
        ex = Example(instruction, response, key)
        ids, start = encode_example(tokenizer, ex)
        if len(ids) > block_size + 1 or len(ids) - start < min_response:
            continue
        out.append(ex)
        if len(out) >= limit:
            break
    return out


def build_tensors(tokenizer, examples: list[Example], block_size: int,
                  chat: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """X: entradas (rellenadas). Y: objetivos con la petición y el relleno en IGNORE."""
    pad = tokenizer.token_to_id(EOT)
    xs, ys = [], []
    for ex in examples:
        ids, start = encode_example(tokenizer, ex, chat)
        if len(ids) > block_size + 1:
            ids = ids[: block_size + 1]
        x, y = ids[:-1], ids[1:]
        y = [IGNORE] * (start - 1) + y[start - 1 :]  # solo se aprende la respuesta
        fill = block_size - len(x)
        xs.append(x + [pad] * fill)
        ys.append(y + [IGNORE] * fill)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


# --- 2. Ampliar el vocabulario del modelo ------------------------------------------------------
@torch.no_grad()
def expand_vocab(model: GPT, new_vocab: int) -> None:
    """Añade filas al embedding para los tokens nuevos, iniciadas con el promedio de las demás."""
    old = model.wte.weight
    V, C = old.shape
    if new_vocab <= V:
        return
    wte = nn.Embedding(new_vocab, C).to(old.device)
    wte.weight[:V] = old
    wte.weight[V:] = old.mean(dim=0, keepdim=True) + 0.02 * torch.randn(new_vocab - V, C, device=old.device)
    model.wte = wte
    model.lm_head = nn.Linear(C, new_vocab, bias=False).to(old.device)
    model.lm_head.weight = model.wte.weight  # se mantiene el weight tying
    model.config.vocab_size = new_vocab


# --- 3. Entrenamiento ---------------------------------------------------------------------------
def get_sft_batch(X: torch.Tensor, Y: torch.Tensor, batch_size: int, device: str,
                  generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(len(X), (batch_size,), generator=generator)
    return X[idx].to(device), Y[idx].to(device)


@torch.no_grad()
def sft_loss(model: GPT, X: torch.Tensor, Y: torch.Tensor, batch_size: int, iters: int, device: str) -> float:
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    losses = []
    for _ in range(iters):
        x, y = get_sft_batch(X, Y, batch_size, device, gen)
        losses.append(model(x, y)[1].item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def response_loss(model: GPT, X: torch.Tensor, Y: torch.Tensor, device: str, batch_size: int = 64) -> float:
    """Pérdida media sobre los tokens de la respuesta correcta (menor = la predice mejor)."""
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(X), batch_size):
        x, y = X[i : i + batch_size].to(device), Y[i : i + batch_size].to(device)
        logits, _ = model(x)
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                 ignore_index=IGNORE, reduction="sum").item()
        count += int((y != IGNORE).sum())
    return total / max(count, 1)


def stream_reply(model: GPT, tokenizer, prompt_ids: list[int], max_new_tokens: int = 110,
                 temperature: float = 0.7, top_k: int | None = 50, device: str = "cpu") -> Iterator[str]:
    """Genera la respuesta fragmento a fragmento; se detiene en fin de texto o de turno."""
    eot = tokenizer.token_to_id(EOT)
    stops = {i for i in (tokenizer.token_to_id(USER), tokenizer.token_to_id(ASSISTANT)) if i is not None}
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    decoder = StreamDecoder(tokenizer)
    model.eval()
    for next_id in model.generate_iter(idx, max_new_tokens, temperature, top_k, eot_id=eot):
        token = int(next_id[0, 0])
        if token in stops:
            return
        piece = decoder.push(token)
        if piece:
            yield piece


def reply(model: GPT, tokenizer, message: str, device: str = "cpu", chat: bool = True, **kw) -> str:
    ids = chat_prompt_ids(tokenizer, message) if chat else plain_prompt_ids(tokenizer, message)
    return "".join(stream_reply(model, tokenizer, ids, device=device, **kw))


def adherence(model: GPT, tokenizer, examples: list[Example], device: str, chat: bool) -> float:
    """% de respuestas cuyas primeras 12 palabras mencionan lo pedido (p. ej. el personaje).

    Es una medida aproximada: el nombre ya está en la petición y el modelo base tiende a
    copiarlo, así que la métrica principal es la pérdida de la respuesta correcta.
    """
    hits = 0
    for i, ex in enumerate(examples):
        torch.manual_seed(i)
        text = reply(model, tokenizer, ex.instruction, device, chat=chat, temperature=0.7)
        start = " ".join(text.split()[:12]).lower()
        hits += ex.key.lower() in start
    return hits / max(len(examples), 1)


def sft_train(cfg: Config, model: GPT, tensors: dict, run_dir: Path, log, tokenizer=None,
              tokenizer_sha256: str = "", sample_request: str = "") -> dict:
    s = cfg.sft
    device = next(model.parameters()).device.type
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")
    X, Y, Xv, Yv = tensors["X"], tensors["Y"], tensors["Xv"], tensors["Yv"]
    optimizer = configure_optimizer(model, s.weight_decay, s.learning_rate)
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(METRICS_FIELDS)
    best_val, t_start, running, it = float("inf"), time.perf_counter(), None, 0

    def save(path: Path, step: int, with_opt: bool) -> None:
        save_checkpoint(path, model, optimizer if with_opt else None, cfg, step, best_val, tokenizer_sha256)

    model.train()
    try:
        for it in range(s.max_iters + 1):
            lr = get_lr(it, s)
            for group in optimizer.param_groups:
                group["lr"] = lr
            if it % s.eval_interval == 0 or it == s.max_iters:
                tr = sft_loss(model, X, Y, s.batch_size, s.eval_iters, device)
                va = sft_loss(model, Xv, Yv, s.batch_size, s.eval_iters, device)
                elapsed = (time.perf_counter() - t_start) / 60
                log.info("[eval] iter %5d | train %.4f | val %.4f | perplejidad %.1f", it, tr, va, math.exp(va))
                with metrics_path.open("a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([it, f"{tr:.5f}", f"{va:.5f}", f"{math.exp(va):.3f}",
                                            f"{lr:.3e}", f"{elapsed:.2f}"])
                if va < best_val:
                    best_val = va
                    save(run_dir / "best.pt", it, False)
                    log.info("[ckpt] nuevo mejor modelo -> best.pt")
                if tokenizer is not None and sample_request:
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(0)
                        answer = reply(model, tokenizer, sample_request, device, max_new_tokens=60)
                    log.info("[chat] %s -> %s", sample_request, answer.replace("\n", " "))
                model.train()
            if it == s.max_iters:
                break
            x, y = get_sft_batch(X, Y, s.batch_size, device)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if s.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), s.grad_clip)
            optimizer.step()
            running = loss.item() if running is None else 0.9 * running + 0.1 * loss.item()
            if it % s.log_interval == 0:
                sec = (time.perf_counter() - t_start) / (it + 1)
                log.info("iter %5d/%d | loss %.4f | lr %.2e | ETA %.0f min",
                         it, s.max_iters, running, lr, sec * (s.max_iters - it) / 60)
    except KeyboardInterrupt:
        log.warning("Interrumpido en la iteración %d. best.pt conserva el mejor modelo hasta ahora.", it)
        save(run_dir / "last.pt", it, True)
        plot_metrics(metrics_path, run_dir / "loss.png")
        return {"best_val_loss": best_val, "interrupted": True}
    save(run_dir / "last.pt", s.max_iters, True)
    plot_metrics(metrics_path, run_dir / "loss.png")
    return {"best_val_loss": best_val, "interrupted": False,
            "minutes": (time.perf_counter() - t_start) / 60}


def _read_stories(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero: mini-llm-prepare")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def _write_examples(examples: list[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Etapa 9: ajuste fino (SFT) para chat")
    parser.add_argument("--config", default=str(CONFIGS_DIR / "chat.yaml"))
    parser.add_argument("--overwrite", action="store_true", help="rehacer aunque ya exista")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    s = cfg.sft
    log = get_logger("sft", LOGS_DIR / "sft.log")
    set_seed(cfg.seed)
    configure_threads(cfg.train.num_threads)
    device = resolve_device(cfg.device)
    run_dir = CHECKPOINTS_DIR / cfg.experiment_name
    if (run_dir / "best.pt").exists() and not args.overwrite:
        log.error("Ya existe %s. Usa --overwrite para repetir el ajuste fino.", run_dir / "best.pt")
        return 1

    # Modelo base y tokenizador ampliado
    base_path = PROJECT_ROOT / s.base_checkpoint
    model, _ = load_model(base_path, device)
    tokenizer = load_tokenizer(tokenizer_file(cfg.data.name))
    base_vocab = tokenizer.get_vocab_size()
    add_chat_tokens(tokenizer)
    chat_tok_path = tokenizer_file(chat_tokenizer_name(cfg))
    chat_tok_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(chat_tok_path))
    expand_vocab(model, tokenizer.get_vocab_size())
    log.info("Modelo base: %s | vocabulario %d -> %d (<|usuario|>=%d, <|asistente|>=%d)",
             base_path, base_vocab, tokenizer.get_vocab_size(),
             tokenizer.token_to_id(USER), tokenizer.token_to_id(ASSISTANT))

    # Dataset de instrucciones
    rng = random.Random(cfg.seed)
    proc = processed_dir(cfg.data.name)
    T = model.config.block_size
    log.info("[1/4] Creando el dataset de instrucciones a partir de los cuentos...")
    train_ex = collect_examples(tokenizer, _read_stories(proc / "train_clean.jsonl"), rng, s.train_examples, T)
    val_ex = collect_examples(tokenizer, _read_stories(proc / "val_clean.jsonl"), rng, s.val_examples, T)
    sft_dir = sft_data_dir(cfg)
    _write_examples(train_ex, sft_dir / "train.jsonl")
    _write_examples(val_ex, sft_dir / "val.jsonl")
    log.info("Ejemplos: %s de entrenamiento, %s de validación (guardados en %s)",
             f"{len(train_ex):,}", f"{len(val_ex):,}", sft_dir)
    for ex in train_ex[:3]:
        log.info("   petición: %s  | respuesta: %s...", ex.instruction, ex.response[:70].replace("\n", " "))
    X, Y = build_tensors(tokenizer, train_ex, T)
    Xv, Yv = build_tensors(tokenizer, val_ex, T)

    probe = val_ex[: s.adherence_samples]
    log.info("[2/4] Midiendo el modelo base ANTES del ajuste...")
    Xp, Yp = build_tensors(tokenizer, val_ex, T, chat=False)  # la petición como texto normal
    loss_before = response_loss(model, Xp, Yp, device)
    before = adherence(model, tokenizer, probe, device, chat=False)
    log.info("Antes: pérdida de la respuesta correcta %.3f | menciona lo pedido al inicio: %.0f %%",
             loss_before, 100 * before)

    log.info("[3/4] Ajuste fino: %d iteraciones", s.max_iters)
    result = sft_train(cfg, model, {"X": X, "Y": Y, "Xv": Xv, "Yv": Yv}, run_dir, log, tokenizer,
                       hashlib.sha256(chat_tok_path.read_bytes()).hexdigest(), cfg.demo.prompt)
    if result["interrupted"]:
        return 130

    log.info("[4/4] Midiendo el modelo DESPUÉS del ajuste...")
    tuned, ckpt = load_model(run_dir / "best.pt", device)
    loss_after = response_loss(tuned, Xv, Yv, device)
    after = adherence(tuned, tokenizer, probe, device, chat=True)
    samples = []
    for i, request in enumerate(cfg.demo.eval_prompts):
        torch.manual_seed(100 + i)
        samples.append({"request": request, "reply": reply(tuned, tokenizer, request, device)})
    report = {
        "base_checkpoint": s.base_checkpoint,
        "iter": ckpt["iter"],
        "best_val_loss": result["best_val_loss"],
        "minutes": result["minutes"],
        "train_examples": len(train_ex),
        "val_examples": len(val_ex),
        "loss_before": loss_before,
        "loss_after": loss_after,
        "adherence_before": before,
        "adherence_after": after,
        "adherence_samples": len(probe),
        "samples": samples,
    }
    (run_dir / "sft_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("=" * 60)
    log.info("Ajuste fino terminado en %.1f min", result["minutes"])
    log.info("Pérdida de la respuesta correcta: %.3f antes -> %.3f después (perplejidad %.1f -> %.1f)",
             loss_before, loss_after, math.exp(loss_before), math.exp(loss_after))
    log.info("Menciona lo pedido al inicio: %.0f %% antes -> %.0f %% después", 100 * before, 100 * after)
    for smp in samples:
        log.info("[chat] %s\n        %s", smp["request"], smp["reply"].replace("\n", " "))
    log.info("Prueba el chat: mini-llm-chat   o la pestaña Chat de mini-llm-web")
    return 0


def chat_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Conversar con el modelo ajustado (SFT)")
    parser.add_argument("--config", default=str(CONFIGS_DIR / "chat.yaml"))
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    configure_threads(cfg.train.num_threads)
    device = resolve_device(cfg.device)
    model, ckpt = load_model(CHECKPOINTS_DIR / cfg.experiment_name / "best.pt", device)
    tokenizer = load_tokenizer(tokenizer_file(chat_tokenizer_name(cfg)))
    print(f"Chat de cuentos (iteración {ckpt['iter']}). Pide un cuento; /salir para terminar.")
    print(f"Cada petición es independiente: el modelo solo ve {model.config.block_size} tokens.\n")
    while True:
        try:
            message = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if message in {"/salir", "/exit"}:
            return 0
        if not message:
            continue
        print("Modelo: ", end="", flush=True)
        try:
            for piece in stream_reply(model, tokenizer, chat_prompt_ids(tokenizer, message),
                                      temperature=args.temperature, device=device):
                print(piece, end="", flush=True)
        except KeyboardInterrupt:
            print(" [detenido]", end="")
        print("\n")


if __name__ == "__main__":
    sys.exit(main())
