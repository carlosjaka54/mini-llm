"""Interfaz web para demostrar y sustentar el proyecto.

Un servidor local (FastAPI) carga el modelo entrenado y expone una API que usa la
página web `static/index.html`. Funciona sin internet.

Rutas principales:
    GET  /api/info           datos del modelo, entrenamiento, evaluación y chat
    POST /api/generate       texto en streaming, con la confianza de cada token
    POST /api/chat           respuesta del chatbot (SFT) o del modelo base
    POST /api/probs          los tokens más probables para continuar un texto
    POST /api/score          qué tan "sorprendente" es cada token de un texto
    POST /api/attention      pesos de atención por capa y cabeza
    POST /api/tokenize       tokenización de un texto
    POST /api/neighbors      palabras más cercanas en el espacio de embeddings
    POST /api/embedding_map  mapa 2D (PCA) de los embeddings de varias palabras

Uso:
    mini-llm-web                 # abre http://127.0.0.1:8000 (o el siguiente puerto libre)
    mini-llm-web --port 8080 --no-browser
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import sys
import threading
import time
import webbrowser
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from torch.nn import functional as F

from mini_llm import __version__
from mini_llm.checkpoint import load_model
from mini_llm.config import Config
from mini_llm.dataset import load_meta
from mini_llm.generate import StreamDecoder, prompt_ids
from mini_llm.model import GPT
from mini_llm.sampling import SamplingParams
from mini_llm.sft import ASSISTANT, USER, chat_prompt_ids, chat_tokenizer_name, plain_prompt_ids
from mini_llm.tokenizer import EOT, load_tokenizer
from mini_llm.utils import (
    ARTIFACTS_DIR,
    CHECKPOINTS_DIR,
    CONFIGS_DIR,
    configure_threads,
    resolve_device,
    tokenizer_file,
)

STATIC_DIR = Path(__file__).parent / "static"
N_ALTERNATIVES = 4


# --- Peticiones -----------------------------------------------------------------------------
class SamplingRequest(BaseModel):
    temperature: float = Field(0.8, ge=0.0, le=3.0)
    top_k: int = Field(50, ge=0, le=10_000)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=1.0, le=3.0)
    max_new_tokens: int = Field(150, ge=1, le=600)
    seed: int | None = None

    def params(self) -> SamplingParams:
        return SamplingParams(self.temperature, self.top_k or None,
                              self.top_p if self.top_p < 1.0 else None, self.repetition_penalty)


class GenerateRequest(SamplingRequest):
    prompt: str = Field("", max_length=2000)
    trained: bool = True


class ChatRequest(SamplingRequest):
    message: str = Field(..., min_length=1, max_length=500)
    temperature: float = Field(0.7, ge=0.0, le=3.0)
    max_new_tokens: int = Field(110, ge=1, le=400)
    mode: Literal["chat", "base"] = "chat"  # base = el modelo sin ajuste, que solo continúa texto


class TextRequest(BaseModel):
    text: str = Field("", max_length=2000)
    temperature: float = Field(1.0, ge=0.0, le=3.0)
    k: int = Field(10, ge=1, le=50)


class WordsRequest(BaseModel):
    words: list[str] = Field(..., min_length=1, max_length=40)


class NeighborsRequest(BaseModel):
    word: str = Field(..., min_length=1, max_length=40)
    k: int = Field(12, ge=1, le=40)


# --- Lógica (independiente de la web, fácil de probar) --------------------------------------
def token_pieces(tokenizer, text: str) -> list[dict]:
    """Un elemento por token. Si un carácter ("é") ocupa 2 tokens, ambos lo muestran."""
    enc = tokenizer.encode(text)
    return [{"id": i, "piece": text[a:b]} for i, (a, b) in zip(enc.ids, enc.offsets, strict=True)]


def tokenize_grouped(tokenizer, text: str) -> list[dict]:
    """Como token_pieces, pero agrupa los tokens que forman un mismo carácter (sus bytes)."""
    enc = tokenizer.encode(text)
    groups: list[dict] = []
    last_span = None
    for i, span in zip(enc.ids, enc.offsets, strict=True):
        if span == last_span:
            groups[-1]["ids"].append(i)
        else:
            groups.append({"ids": [i], "piece": text[span[0] : span[1]]})
        last_span = span
    return groups


def _token_label(tokenizer, token_id: int, eot: int | None) -> dict:
    if token_id == eot:
        return {"id": token_id, "text": "", "eot": True, "partial": False}
    piece = tokenizer.decode([token_id])
    return {"id": token_id, "text": piece, "eot": False, "partial": piece.endswith("�")}


def _alternatives(tokenizer, probs: torch.Tensor, eot: int | None, n: int) -> list[dict]:
    top = torch.topk(probs, min(n, probs.numel()))
    return [{**_token_label(tokenizer, i, eot), "p": round(p, 5)}
            for p, i in zip(top.values.tolist(), top.indices.tolist(), strict=True)]


def stream_events(
    model: GPT,
    tokenizer,
    ids: list[int],
    params: SamplingParams,
    max_new_tokens: int,
    device: str = "cpu",
    stop_tokens: tuple[str, ...] = (),
) -> Iterator[dict]:
    """Genera texto y entrega un evento por fragmento visible, con la confianza del modelo.

    Cada evento: {"type": "token", "text", "p", "rank", "alts"}. Al final: {"type": "done", ...}.
    `p` es la probabilidad que el modelo daba al token elegido; `alts`, sus opciones favoritas.
    Si un carácter ocupa varios tokens (sus bytes), se agrupan y `p` es la menor del grupo.
    """
    eot = tokenizer.token_to_id(EOT)
    stops = {i for i in (tokenizer.token_to_id(t) for t in stop_tokens) if i is not None}
    idx = torch.tensor([ids[-model.config.block_size :] or [eot]], dtype=torch.long, device=device)
    decoder = StreamDecoder(tokenizer)
    model.eval()
    t0 = time.perf_counter()
    n_tokens, sum_logp, pending = 0, 0.0, None
    for next_id, raw in model.generate_steps(idx, max_new_tokens, params, eot, stops):
        token = int(next_id[0, 0])
        probs = raw[0]
        p = float(probs[token])
        n_tokens += 1
        sum_logp += math.log(max(p, 1e-12))
        if pending is None:
            rank = int((probs > probs[token]).sum()) + 1
            pending = {"p": p, "rank": rank, "alts": _alternatives(tokenizer, probs, eot, N_ALTERNATIVES)}
        else:
            pending["p"] = min(pending["p"], p)
        if token == eot:
            break
        piece = decoder.push(token)
        if piece:
            yield {"type": "token", "text": piece, "p": round(pending["p"], 5), "rank": pending["rank"],
                   "alts": pending["alts"]}
            pending = None
    seconds = time.perf_counter() - t0
    yield {
        "type": "done",
        "tokens": n_tokens,
        "seconds": round(seconds, 3),
        "tokens_per_second": round(n_tokens / seconds, 1) if seconds > 0 else None,
        "perplexity": round(math.exp(-sum_logp / n_tokens), 2) if n_tokens else None,
    }


def ndjson(events: Iterator[dict]) -> Iterator[str]:
    for event in events:
        yield json.dumps(event, ensure_ascii=False) + "\n"


@torch.no_grad()
def top_next_tokens(model: GPT, tokenizer, text: str, temperature: float, k: int, device: str) -> list[dict]:
    ids = prompt_ids(tokenizer, text)[-model.config.block_size :]
    logits, _ = model(torch.tensor([ids], dtype=torch.long, device=device))
    probs = F.softmax(logits[0, -1] / max(temperature, 1e-6), dim=-1)
    eot = tokenizer.token_to_id(EOT)
    return [{**alt, "prob": alt["p"]} for alt in _alternatives(tokenizer, probs, eot, k)]


@torch.no_grad()
def score_text(model: GPT, tokenizer, text: str, device: str) -> dict:
    """Probabilidad que el modelo asignaba a cada token real del texto (desde el 2.º token)."""
    eot = tokenizer.token_to_id(EOT)
    groups = tokenize_grouped(tokenizer, text)
    ids = [eot] + [i for g in groups for i in g["ids"]]
    ids = ids[: model.config.block_size + 1]
    if len(ids) < 2:
        return {"tokens": [], "perplexity": None}
    logits, _ = model(torch.tensor([ids[:-1]], dtype=torch.long, device=device))
    probs = F.softmax(logits[0], dim=-1)
    out, pos, logps = [], 0, []
    for g in groups:
        p_group, top = 1.0, None
        for _ in g["ids"]:
            if pos >= probs.size(0):
                break
            real = ids[pos + 1]
            p = float(probs[pos, real])
            p_group = min(p_group, p)
            logps.append(math.log(max(p, 1e-12)))
            if top is None:
                top = _alternatives(tokenizer, probs[pos], eot, N_ALTERNATIVES)
            pos += 1
        if top is None:
            break
        out.append({"text": g["piece"], "p": round(p_group, 5), "alts": top})
    ppl = math.exp(-sum(logps) / len(logps)) if logps else None
    return {"tokens": out, "perplexity": round(ppl, 2) if ppl else None, "n_tokens": len(logps)}


@torch.no_grad()
def attention_maps(model: GPT, tokenizer, text: str, device: str) -> dict:
    """Pesos de atención de cada capa y cabeza: layers[capa][cabeza][i][j] = cuánto mira i a j."""
    pieces = token_pieces(tokenizer, text)[: model.config.block_size]
    if not pieces:
        return {"tokens": [], "layers": []}
    idx = torch.tensor([[p["id"] for p in pieces]], dtype=torch.long, device=device)
    for block in model.blocks:
        block.attn.store_attention = True
    try:
        model(idx)
        maps = [block.attn.last_attention[0].float().cpu() for block in model.blocks]
    finally:
        for block in model.blocks:
            block.attn.store_attention = False
            block.attn.last_attention = None
    layers = [(torch.round(m * 10_000) / 10_000).tolist() for m in maps]
    return {"tokens": pieces, "layers": layers}


# --- Embeddings ---------------------------------------------------------------------------------
def _word_vector(model: GPT, tokenizer, word: str) -> tuple[torch.Tensor, list[int]]:
    """Vector de una palabra: su token si es uno solo, o el promedio de sus tokens."""
    word = word.strip()
    ids = tokenizer.encode(" " + word).ids  # en medio de una frase, las palabras llevan espacio
    if len(ids) > 1:
        alone = tokenizer.encode(word).ids
        ids = alone if len(alone) < len(ids) else ids
    emb = model.wte.weight.detach()
    return emb[ids].mean(dim=0), ids


def _is_word_token(text: str) -> bool:
    """Un token que es una palabra completa: empieza con espacio y solo tiene letras."""
    core = text.strip()
    return text.startswith(" ") and len(core) >= 3 and core.isalpha()


@torch.no_grad()
def embedding_neighbors(model: GPT, tokenizer, word: str, k: int) -> dict:
    """Tokens cuyo embedding apunta en la dirección más parecida (similitud coseno)."""
    vec, ids = _word_vector(model, tokenizer, word)
    emb = F.normalize(model.wte.weight.detach(), dim=-1)
    sims = emb @ F.normalize(vec, dim=0)
    order = torch.argsort(sims, descending=True).tolist()
    special = {tokenizer.token_to_id(t) for t in (EOT, USER, ASSISTANT)}
    seen, out = {word.strip().lower()}, []
    for i in order:
        if i in ids or i in special:
            continue
        text = tokenizer.decode([i])
        key = text.strip().lower()
        if not _is_word_token(text) or key in seen:
            continue
        seen.add(key)
        out.append({"text": text.strip(), "similarity": round(float(sims[i]), 4)})
        if len(out) >= k:
            break
    return {"word": word.strip(), "tokens": [tokenizer.decode([i]) for i in ids], "neighbors": out}


@torch.no_grad()
def embedding_map(model: GPT, tokenizer, words: list[str]) -> dict:
    """Proyecta los embeddings de las palabras a 2D con PCA (las 2 direcciones de más varianza)."""
    words = [w.strip() for w in words if w.strip()]
    if len(words) < 3:
        raise ValueError("Se necesitan al menos 3 palabras")
    X = torch.stack([_word_vector(model, tokenizer, w)[0] for w in words]).float()
    X = X - X.mean(dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
    coords = X @ Vh[:2].T
    var = (S**2) / (S**2).sum()
    return {
        "points": [{"word": w, "x": round(float(c[0]), 4), "y": round(float(c[1]), 4)}
                   for w, c in zip(words, coords, strict=True)],
        "explained_variance": [round(float(v), 4) for v in var[:2]],
    }


# --- Información para la página -----------------------------------------------------------------
def _read_training_curve(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [
            {"iter": int(r["iter"]), "train": float(r["train_loss"]), "val": float(r["val_loss"]),
             "elapsed_min": float(r["elapsed_min"])}
            for r in csv.DictReader(f)
        ]


def build_info(cfg: Config, ckpt_path: Path, model: GPT, ckpt: dict, tokenizer, device: str) -> dict:
    """Todo lo que la página muestra en Resumen y Métricas (tolera archivos ausentes)."""
    run_dir = ckpt_path.parent
    eval_path = ARTIFACTS_DIR / "eval" / run_dir.name / "metrics.json"
    evaluation = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else None
    try:
        meta = load_meta(cfg.data.name)
    except FileNotFoundError:
        meta = None
    saved = ckpt.get("config", {})
    data_cfg = saved.get("data", {})
    train_cfg = saved.get("train", {})
    return {
        "version": __version__,
        "experiment": run_dir.name,
        "checkpoint": ckpt_path.name,
        "iter": ckpt.get("iter"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "device": device,
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "parameters": model.num_parameters(),
        "model_config": asdict(model.config),
        "vocab_size": tokenizer.get_vocab_size(),
        "dataset": f"{data_cfg.get('dataset', '?')} [{data_cfg.get('subset', '')}]",
        "train_config": {k: train_cfg.get(k) for k in ("batch_size", "max_iters", "learning_rate",
                                                        "warmup_iters", "weight_decay")},
        "data": None if meta is None else {
            "train_tokens": meta["splits"]["train"]["tokens"],
            "val_tokens": meta["splits"]["val"]["tokens"],
            "train_documents": meta["splits"]["train"]["documents"],
            "chars_per_token": meta["chars_per_token"],
        },
        "demo": asdict(cfg.demo),
        "training_curve": _read_training_curve(run_dir / "metrics.csv"),
        "evaluation": None if evaluation is None else {
            "baselines": evaluation["baselines"],
            "model_metrics": evaluation["model_metrics"],
            "distinct_2": evaluation.get("distinct_2"),
        },
    }


def load_chat(config_path: str, device: str) -> tuple[dict | None, dict]:
    """Carga el modelo de chat si ya se entrenó con mini-llm-sft."""
    path = Path(config_path)
    if not path.exists():
        return None, {"available": False}
    chat_cfg = Config.from_yaml(path)
    run_dir = CHECKPOINTS_DIR / chat_cfg.experiment_name
    tok_path = tokenizer_file(chat_tokenizer_name(chat_cfg))
    if not (run_dir / "best.pt").exists() or not tok_path.exists():
        return None, {"available": False}
    chat_model, ckpt = load_model(run_dir / "best.pt", device)
    report_path = run_dir / "sft_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    info = {
        "available": True,
        "iter": ckpt.get("iter"),
        "block_size": chat_model.config.block_size,
        "loss_before": report.get("loss_before"),
        "loss_after": report.get("loss_after"),
        "adherence_before": report.get("adherence_before"),
        "adherence_after": report.get("adherence_after"),
        "adherence_samples": report.get("adherence_samples"),
        "train_examples": report.get("train_examples"),
        "minutes": report.get("minutes"),
        "training_curve": _read_training_curve(run_dir / "metrics.csv"),
    }
    return {"model": chat_model, "tokenizer": load_tokenizer(tok_path)}, info


# --- Aplicación -------------------------------------------------------------------------------
def create_app(
    model: GPT, untrained: GPT, tokenizer, info: dict, device: str = "cpu", chat: dict | None = None
) -> FastAPI:
    """`chat` = {"model": GPT ajustado, "tokenizer": tokenizador con tokens de chat} o None."""
    app = FastAPI(title="mini-LLM", version=__version__, docs_url="/api/docs")
    attention_lock = threading.Lock()
    info.setdefault("chat", {"available": chat is not None})

    def streaming(events: Iterator[dict], seed: int | None) -> StreamingResponse:
        def run():
            if seed is not None:
                torch.manual_seed(seed)
            yield from ndjson(events)
        return StreamingResponse(run(), media_type="application/x-ndjson; charset=utf-8")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/info")
    def get_info() -> dict:
        return info

    @app.post("/api/generate")
    def generate(req: GenerateRequest) -> StreamingResponse:
        m = model if req.trained else untrained
        ids = prompt_ids(tokenizer, req.prompt)
        return streaming(stream_events(m, tokenizer, ids, req.params(), req.max_new_tokens, device), req.seed)

    @app.post("/api/chat")
    def chat_endpoint(req: ChatRequest) -> StreamingResponse:
        if req.mode == "chat" and chat is None:
            raise HTTPException(409, "El modelo de chat no está entrenado. Ejecuta: mini-llm-sft")
        if req.mode == "chat":
            m, tok = chat["model"], chat["tokenizer"]
            ids = chat_prompt_ids(tok, req.message)
        else:
            m, tok = model, tokenizer
            ids = plain_prompt_ids(tok, req.message)
        events = stream_events(m, tok, ids, req.params(), req.max_new_tokens, device, (USER, ASSISTANT))
        return streaming(events, req.seed)

    @app.post("/api/probs")
    def probs(req: TextRequest) -> list[dict]:
        return top_next_tokens(model, tokenizer, req.text, req.temperature, req.k, device)

    @app.post("/api/score")
    def score(req: TextRequest) -> dict:
        return score_text(model, tokenizer, req.text, device)

    @app.post("/api/tokenize")
    def tokenize(req: TextRequest) -> list[dict]:
        return tokenize_grouped(tokenizer, req.text)

    @app.post("/api/attention")
    def attention(req: TextRequest) -> dict:
        with attention_lock:
            return attention_maps(model, tokenizer, req.text, device)

    @app.post("/api/neighbors")
    def neighbors(req: NeighborsRequest) -> dict:
        return embedding_neighbors(model, tokenizer, req.word, req.k)

    @app.post("/api/embedding_map")
    def emb_map(req: WordsRequest) -> dict:
        try:
            return embedding_map(model, tokenizer, req.words)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


def find_free_port(host: str, preferred: int, attempts: int = 20) -> int:
    """El puerto pedido si está libre; si no, el siguiente libre (evita el error 10048 de Windows)."""
    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise OSError(f"No hay puertos libres entre {preferred} y {preferred + attempts - 1}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interfaz web del mini-LLM")
    parser.add_argument("--config", default=str(CONFIGS_DIR / "cpu.yaml"))
    parser.add_argument("--checkpoint", help="por defecto: best.pt del experimento")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="no abrir el navegador")
    parser.add_argument("--chat-config", default=str(CONFIGS_DIR / "chat.yaml"),
                        help="configuración del modelo de chat (etapa 9)")
    args = parser.parse_args(argv)

    import uvicorn

    cfg = Config.from_yaml(args.config)
    configure_threads(cfg.train.num_threads)
    device = resolve_device(cfg.device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else CHECKPOINTS_DIR / cfg.experiment_name / "best.pt"
    if not ckpt_path.exists():
        print(f"\n  No existe {ckpt_path}. Entrena primero con: mini-llm-train\n")
        return 1
    model, ckpt = load_model(ckpt_path, device)
    torch.manual_seed(0)
    untrained = GPT(model.config).to(device).eval()  # misma arquitectura, pesos aleatorios
    tokenizer = load_tokenizer(tokenizer_file(cfg.data.name))
    info = build_info(cfg, ckpt_path, model, ckpt, tokenizer, device)
    chat, info["chat"] = load_chat(args.chat_config, device)
    app = create_app(model, untrained, tokenizer, info, device, chat)

    port = find_free_port(args.host, args.port)
    if port != args.port:
        print(f"\n  El puerto {args.port} está ocupado por otro programa; se usa el {port}.")
    url = f"http://{args.host}:{port}"
    print(f"\n  mini-LLM listo en {url}   (Ctrl+C para cerrar)\n")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
