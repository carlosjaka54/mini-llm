"""Etapa 3b: tokenizador BPE entrenado desde cero sobre nuestro corpus.

BPE (Byte Pair Encoding) empieza con los 256 bytes posibles y va fusionando
los pares más frecuentes: "q"+"u" -> "qu", "qu"+"e" -> "que"... hasta llegar
a `vocab_size` tokens. Al ser a nivel de bytes, puede representar cualquier
texto (tildes, ñ, emojis) sin tokens "desconocidos".

Demo:
    mini-llm-tokens "Había una vez un gato que quería volar."
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from mini_llm.utils import TOKENIZER_DIR

EOT = "<|endoftext|>"          # separa documentos: "aquí termina un cuento"
SPECIAL_TOKENS = [EOT]
TOKENIZER_FILE = TOKENIZER_DIR / "tokenizer.json"


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(
    texts: Iterable[str], vocab_size: int, min_frequency: int = 2, length: int | None = None
) -> Tokenizer:
    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer, length=length)
    return tok


def load_tokenizer(path: str | Path = TOKENIZER_FILE) -> Tokenizer:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero: mini-llm-prepare")
    return Tokenizer.from_file(str(path))


def eot_id(tok: Tokenizer) -> int:
    idx = tok.token_to_id(EOT)
    if idx is None:
        raise ValueError(f"El tokenizador no contiene el token especial {EOT}")
    return idx


def iter_token_ids(tok: Tokenizer, texts: list[str], batch_size: int = 1000) -> Iterator[list[int]]:
    """Codifica en lotes (mucho más rápido) y añade EOT al final de cada documento."""
    end = eot_id(tok)
    for i in range(0, len(texts), batch_size):
        for enc in tok.encode_batch(texts[i : i + batch_size]):
            yield enc.ids + [end]


def demo_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Muestra cómo se tokeniza un texto")
    parser.add_argument("text", nargs="*", help="texto a tokenizar")
    args = parser.parse_args(argv)
    text = " ".join(args.text) or "Había una vez un pequeño gato que quería volar hasta la luna."

    tok = load_tokenizer()
    enc = tok.encode(text)
    # Usamos los offsets (posición en el texto original) para mostrar cada token legible;
    # "·" marca los espacios, que forman parte del token siguiente.
    pieces = [text[a:b].replace(" ", "·") for a, b in enc.offsets]

    print(f"\nTexto    : {text}")
    print(f"Tokens   : {' | '.join(pieces)}")
    print(f"IDs      : {enc.ids}")
    print(f"Resumen  : {len(text)} caracteres -> {len(enc.ids)} tokens "
          f"({len(text) / max(len(enc.ids), 1):.2f} caracteres/token)")
    print(f"Decodif. : {tok.decode(enc.ids)}")
    print(f"Vocabulario: {tok.get_vocab_size()} tokens\n")
    return 0


if __name__ == "__main__":
    sys.exit(demo_main())
