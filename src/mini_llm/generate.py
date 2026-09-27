"""Etapa 8: generar texto con el modelo entrenado.

El modelo produce UN token cada vez: mira el texto, calcula la probabilidad de cada
token del vocabulario, sortea uno, lo añade al texto y repite.

Dos perillas controlan ese sorteo:
  temperatura : divide los logits antes del softmax.
                baja (0.3) -> elige casi siempre lo más probable: seguro pero repetitivo.
                alta (1.5) -> reparte la probabilidad: creativo pero incoherente.
                0          -> siempre el más probable (determinista).
  top-k       : solo sortea entre los k tokens más probables; evita rarezas.

Uso:
    mini-llm-generate                                  # modo interactivo
    mini-llm-generate "Había una vez"
    mini-llm-generate "Había una vez" --temperature 1.2 --num-samples 3 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO

import torch
from torch.nn import functional as F

from mini_llm.checkpoint import load_model
from mini_llm.config import Config
from mini_llm.model import GPT
from mini_llm.tokenizer import EOT, load_tokenizer
from mini_llm.utils import CHECKPOINTS_DIR, CONFIGS_DIR, configure_threads, resolve_device

DEFAULT_PROMPT = "Había una vez"

HELP = """
Escribe el inicio de un cuento y pulsa Enter (o Enter vacío para un cuento desde cero).
Comandos:
  /temp 1.2          temperatura (0 = determinista, 0.8 = equilibrado, 1.5 = alocado)
  /topk 20           top-k (0 = sin límite)
  /largo 150         máximo de tokens nuevos
  /semilla 42        fija la aleatoriedad (/semilla sin número = aleatorio)
  /probs <texto>     muestra los 10 tokens más probables para continuar <texto>
  /comparar <texto>  genera <texto> con temperaturas 0.3, 0.8 y 1.5
  /config            muestra la configuración actual
  /ayuda  /salir
Ctrl+C detiene la generación en curso."""


@dataclass
class GenSettings:
    temperature: float = 0.8
    top_k: int | None = 50
    max_new_tokens: int = 200
    seed: int | None = None


class StreamDecoder:
    """Convierte tokens en texto a medida que llegan.

    Un carácter como "á" ocupa 2 bytes y puede llegar en 2 tokens: mientras está
    incompleto, el decodificador produce "�", así que esperamos antes de mostrarlo.
    """

    def __init__(self, tokenizer) -> None:
        self.tok = tokenizer
        self.ids: list[int] = []
        self.emitted = 0

    def push(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = self.tok.decode(self.ids)
        if text.endswith("\ufffd"):
            return ""
        new, self.emitted = text[self.emitted :], len(text)
        return new


def prompt_ids(tokenizer, text: str) -> list[int]:
    # Sin texto: empezamos desde <|endoftext|>, es decir, "comienza un cuento nuevo"
    return tokenizer.encode(text).ids or [tokenizer.token_to_id(EOT)]


def stream_generate(
    model: GPT, tokenizer, prompt: str, s: GenSettings, device: str = "cpu", out: TextIO = sys.stdout
) -> str:
    """Genera mostrando cada fragmento en cuanto se produce. Devuelve el texto completo."""
    if s.seed is not None:
        torch.manual_seed(s.seed)
    idx = torch.tensor([prompt_ids(tokenizer, prompt)], dtype=torch.long, device=device)
    decoder = StreamDecoder(tokenizer)
    pieces: list[str] = []
    out.write(prompt)
    out.flush()
    for next_id in model.generate_iter(idx, s.max_new_tokens, s.temperature, s.top_k,
                                       eot_id=tokenizer.token_to_id(EOT)):
        piece = decoder.push(int(next_id[0, 0]))
        pieces.append(piece)
        out.write(piece)
        out.flush()
    out.write("\n")
    return prompt + "".join(pieces)


@torch.no_grad()
def next_token_table(
    model: GPT, tokenizer, text: str, temperature: float = 1.0, k: int = 10, device: str = "cpu"
) -> list[tuple[str, float]]:
    """Los k tokens más probables para continuar `text`, con su probabilidad."""
    model.eval()
    ids = prompt_ids(tokenizer, text)[-model.config.block_size :]
    logits, _ = model(torch.tensor([ids], dtype=torch.long, device=device))
    probs = F.softmax(logits[0, -1] / max(temperature, 1e-6), dim=-1)
    top = torch.topk(probs, min(k, probs.numel()))
    eot = tokenizer.token_to_id(EOT)
    rows = []
    for p, i in zip(top.values.tolist(), top.indices.tolist(), strict=True):
        shown = "<fin del cuento>" if i == eot else tokenizer.decode([i]).replace(" ", "·")
        rows.append((shown, p))
    return rows


def show_probs(model: GPT, tokenizer, text: str, temperature: float, device: str) -> None:
    print(f'\nSiguiente token después de "{text}" (temperatura {temperature}):')
    for shown, p in next_token_table(model, tokenizer, text, temperature, 10, device):
        print(f"  {shown:<18} {100 * p:5.1f} % {'█' * round(40 * p)}")


def interactive(model: GPT, tokenizer, device: str, s: GenSettings) -> None:
    print(HELP)
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        try:
            if not line.startswith("/"):
                print()
                stream_generate(model, tokenizer, line, s, device)
                continue
            cmd, _, arg = line.partition(" ")
            arg = arg.strip()
            if cmd in {"/salir", "/exit", "/quit"}:
                return
            if cmd == "/ayuda":
                print(HELP)
            elif cmd == "/temp":
                s.temperature = max(0.0, float(arg))
                print(f"Temperatura = {s.temperature}")
            elif cmd == "/topk":
                s.top_k = int(arg) if int(arg) > 0 else None
                print(f"Top-k = {s.top_k or 'sin límite'}")
            elif cmd == "/largo":
                s.max_new_tokens = max(1, int(arg))
                print(f"Máximo de tokens nuevos = {s.max_new_tokens}")
            elif cmd == "/semilla":
                s.seed = int(arg) if arg else None
                print(f"Semilla = {s.seed if s.seed is not None else 'aleatoria'}")
            elif cmd == "/config":
                print(s)
            elif cmd == "/probs":
                show_probs(model, tokenizer, arg or DEFAULT_PROMPT, s.temperature, device)
            elif cmd == "/comparar":
                for t in (0.3, 0.8, 1.5):
                    print(f"\n--- temperatura {t} ---")
                    short = replace(s, temperature=t, max_new_tokens=min(s.max_new_tokens, 80))
                    stream_generate(model, tokenizer, arg or DEFAULT_PROMPT, short, device)
            else:
                print("Comando desconocido. Escribe /ayuda")
        except ValueError:
            print("Valor no válido. Ejemplo: /temp 0.8")
        except KeyboardInterrupt:
            print("\n[generación detenida]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Etapa 8: generar texto")
    parser.add_argument("prompt", nargs="*", help="inicio del texto (vacío = modo interactivo)")
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("--config", default=str(CONFIGS_DIR / "cpu.yaml"))
    parser.add_argument("--checkpoint", help="por defecto: best.pt del experimento")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50, help="0 = sin límite")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    configure_threads(cfg.train.num_threads)
    device = resolve_device(cfg.device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else CHECKPOINTS_DIR / cfg.experiment_name / "best.pt"
    model, ckpt = load_model(ckpt_path, device)
    tokenizer = load_tokenizer()
    print(f"Modelo: {ckpt_path.parent.name}/{ckpt_path.name} | iteración {ckpt['iter']} | "
          f"pérdida val {ckpt['best_val_loss']:.3f} | {model.num_parameters()['total']:,} parámetros")

    settings = GenSettings(args.temperature, args.top_k if args.top_k > 0 else None,
                           args.max_new_tokens, args.seed)
    prompt = " ".join(args.prompt)
    if args.interactive or not prompt:
        interactive(model, tokenizer, device, settings)
        return 0
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n--- Muestra {i + 1} ---")
        sample = replace(settings, seed=None if args.seed is None else args.seed + i)
        try:
            stream_generate(model, tokenizer, prompt, sample, device)
        except KeyboardInterrupt:
            print("\n[generación detenida]")
            return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
