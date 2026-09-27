"""Interfaz web para demostrar y sustentar el proyecto.

Un servidor local (FastAPI) carga el modelo entrenado y expone una API que usa
una página web con seis secciones: resumen, generación, probabilidades,
atención, tokenizador y métricas. Funciona sin internet.

Uso:
    mini-llm-web                 # abre http://127.0.0.1:8000 en el navegador
    mini-llm-web --port 8080 --no-browser
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import webbrowser
from dataclasses import asdict
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from torch.nn import functional as F

from mini_llm.checkpoint import load_model
from mini_llm.config import Config
from mini_llm.dataset import load_meta
from mini_llm.generate import StreamDecoder, prompt_ids
from mini_llm.model import GPT
from mini_llm.tokenizer import EOT, load_tokenizer
from mini_llm.utils import (
    ARTIFACTS_DIR,
    CHECKPOINTS_DIR,
    CONFIGS_DIR,
    configure_threads,
    resolve_device,
)

STATIC_DIR = Path(__file__).parent / "static"


# --- Peticiones -----------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str = Field("", max_length=2000)
    temperature: float = Field(0.8, ge=0.0, le=3.0)
    top_k: int = Field(50, ge=0, le=10_000)
    max_new_tokens: int = Field(150, ge=1, le=600)
    seed: int | None = None
    trained: bool = True


class TextRequest(BaseModel):
    text: str = Field("", max_length=2000)
    temperature: float = Field(1.0, ge=0.0, le=3.0)
    k: int = Field(10, ge=1, le=50)


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


@torch.no_grad()
def top_next_tokens(model: GPT, tokenizer, text: str, temperature: float, k: int, device: str) -> list[dict]:
    ids = prompt_ids(tokenizer, text)[-model.config.block_size :]
    logits, _ = model(torch.tensor([ids], dtype=torch.long, device=device))
    probs = F.softmax(logits[0, -1] / max(temperature, 1e-6), dim=-1)
    top = torch.topk(probs, min(k, probs.numel()))
    eot = tokenizer.token_to_id(EOT)
    rows = []
    for p, i in zip(top.values.tolist(), top.indices.tolist(), strict=True):
        piece = "" if i == eot else tokenizer.decode([i])
        rows.append({
            "id": i,
            "text": piece,
            "prob": p,
            "eot": i == eot,
            "partial": piece.endswith("\ufffd"),  # medio carácter (p. ej. 1 de los 2 bytes de "á")
        })
    return rows


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


def _read_training_curve(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [
            {"iter": int(r["iter"]), "train": float(r["train_loss"]), "val": float(r["val_loss"]),
             "elapsed_min": float(r["elapsed_min"])}
            for r in csv.DictReader(f)
        ]


def build_info(ckpt_path: Path, model: GPT, ckpt: dict, tokenizer, device: str) -> dict:
    """Todo lo que la página muestra en Resumen y Métricas (tolera archivos ausentes)."""
    run_dir = ckpt_path.parent
    eval_path = ARTIFACTS_DIR / "eval" / run_dir.name / "metrics.json"
    evaluation = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else None
    try:
        meta = load_meta()
    except FileNotFoundError:
        meta = None
    cfg = ckpt.get("config", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    return {
        "experiment": run_dir.name,
        "checkpoint": ckpt_path.name,
        "iter": ckpt.get("iter"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "device": device,
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
        "training_curve": _read_training_curve(run_dir / "metrics.csv"),
        "evaluation": None if evaluation is None else {
            "baselines": evaluation["baselines"],
            "model_metrics": evaluation["model_metrics"],
            "distinct_2": evaluation.get("distinct_2"),
        },
    }


# --- Aplicación -------------------------------------------------------------------------------
def create_app(model: GPT, untrained: GPT, tokenizer, info: dict, device: str = "cpu") -> FastAPI:
    app = FastAPI(title="mini-LLM", docs_url="/api/docs")
    eot = tokenizer.token_to_id(EOT)
    attention_lock = threading.Lock()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/info")
    def get_info() -> dict:
        return info

    @app.post("/api/generate")
    def generate(req: GenerateRequest) -> StreamingResponse:
        m = model if req.trained else untrained

        def stream():
            if req.seed is not None:
                torch.manual_seed(req.seed)
            idx = torch.tensor([prompt_ids(tokenizer, req.prompt)], dtype=torch.long, device=device)
            decoder = StreamDecoder(tokenizer)
            for next_id in m.generate_iter(idx, req.max_new_tokens, req.temperature,
                                           req.top_k or None, eot_id=eot):
                piece = decoder.push(int(next_id[0, 0]))
                if piece:
                    yield piece

        return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")

    @app.post("/api/probs")
    def probs(req: TextRequest) -> list[dict]:
        return top_next_tokens(model, tokenizer, req.text, req.temperature, req.k, device)

    @app.post("/api/tokenize")
    def tokenize(req: TextRequest) -> list[dict]:
        return tokenize_grouped(tokenizer, req.text)

    @app.post("/api/attention")
    def attention(req: TextRequest) -> dict:
        with attention_lock:
            return attention_maps(model, tokenizer, req.text, device)

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interfaz web del mini-LLM")
    parser.add_argument("--config", default=str(CONFIGS_DIR / "cpu.yaml"))
    parser.add_argument("--checkpoint", help="por defecto: best.pt del experimento")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="no abrir el navegador")
    args = parser.parse_args(argv)

    import uvicorn

    cfg = Config.from_yaml(args.config)
    configure_threads(cfg.train.num_threads)
    device = resolve_device(cfg.device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else CHECKPOINTS_DIR / cfg.experiment_name / "best.pt"
    model, ckpt = load_model(ckpt_path, device)
    torch.manual_seed(0)
    untrained = GPT(model.config).to(device).eval()  # misma arquitectura, pesos aleatorios
    tokenizer = load_tokenizer()
    app = create_app(model, untrained, tokenizer, build_info(ckpt_path, model, ckpt, tokenizer, device),
                     device)

    url = f"http://{args.host}:{args.port}"
    print(f"\n  mini-LLM listo en {url}   (Ctrl+C para cerrar)\n")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
