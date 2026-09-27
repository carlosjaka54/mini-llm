"""Etapa 3: limpiar, entrenar el tokenizador y convertir el texto en tokens.

Flujo:
    data/raw/{train,val}.jsonl
      -> limpieza y filtrado     -> data/processed/{train,val}_clean.jsonl
      -> tokenizador BPE          -> artifacts/tokenizer/tokenizer.json
      -> codificación a tokens    -> data/processed/{train,val}.bin  (+ meta.json)

Los .bin son arrays de enteros uint16 (2 bytes por token) que el entrenamiento
lee con numpy.memmap sin cargar todo en memoria.

Uso:
    mini-llm-prepare
    mini-llm-prepare --config configs/gpu.yaml --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mini_llm.cleaning import clean_document, text_fingerprint
from mini_llm.config import Config
from mini_llm.tokenizer import TOKENIZER_FILE, iter_token_ids, train_tokenizer
from mini_llm.utils import CONFIGS_DIR, LOGS_DIR, PROCESSED_DIR, RAW_DIR, get_logger

TOKEN_DTYPE = np.uint16
META_FILE = PROCESSED_DIR / "meta.json"


def read_jsonl(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def write_jsonl(texts: list[str], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    tmp.replace(path)


def clean_split(
    texts: list[str], min_words: int, max_words: int, seen: set[str], dup_reason: str
) -> tuple[list[str], Counter]:
    """Limpia una lista de textos. `seen` acumula huellas para eliminar duplicados."""
    kept: list[str] = []
    stats: Counter = Counter()
    for raw in texts:
        text, reason = clean_document(raw, min_words, max_words)
        if text is not None:
            fp = text_fingerprint(text)
            if fp in seen:
                text, reason = None, dup_reason
            else:
                seen.add(fp)
        stats[reason] += 1
        if text is not None:
            kept.append(text)
    return kept, stats


def encode_to_bin(tok, texts: list[str], path: Path) -> int:
    """Codifica los textos y los escribe como un único array uint16. Devuelve nº de tokens."""
    tmp = path.with_suffix(".bin.tmp")
    total = 0
    with tmp.open("wb") as f:
        for ids in tqdm(iter_token_ids(tok, texts), total=len(texts), desc=f"Tokenizando {path.stem}",
                        unit="doc"):
            arr = np.asarray(ids, dtype=TOKEN_DTYPE)
            arr.tofile(f)
            total += arr.size
    tmp.replace(path)
    return total


def fingerprint(cfg: Config) -> dict:
    """Todo lo que, si cambia, obliga a regenerar los datos procesados."""
    manifest = json.loads((RAW_DIR / "manifest.json").read_text(encoding="utf-8"))
    return {
        "raw_sha256": {k: v["sha256"] for k, v in manifest["splits"].items()},
        "min_words": cfg.data.min_words,
        "max_words": cfg.data.max_words,
        "vocab_size": cfg.data.vocab_size,
        "tokenizer_min_frequency": cfg.data.tokenizer_min_frequency,
    }


def log_cleaning(log, split: str, stats: Counter) -> None:
    total = sum(stats.values())
    log.info("Limpieza de '%s': %s documentos", split, f"{total:,}")
    for reason, count in stats.most_common():
        log.info("   %-22s %8s  (%5.1f %%)", reason, f"{count:,}", 100 * count / total)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Etapa 3: limpiar y tokenizar")
    parser.add_argument("--config", default=str(CONFIGS_DIR / "cpu.yaml"))
    parser.add_argument("--force", action="store_true", help="regenerar aunque ya exista")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    log = get_logger("prepare", LOGS_DIR / "prepare.log")

    if not (RAW_DIR / "manifest.json").exists():
        log.error("No hay datos descargados. Ejecuta primero: mini-llm-data")
        return 1

    fp = fingerprint(cfg)
    if not args.force and META_FILE.exists():
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fp and TOKENIZER_FILE.exists():
            log.info("Los datos ya están procesados con esta configuración. Usa --force para rehacerlos.")
            return 0

    t0 = time.perf_counter()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 1) Limpieza --------------------------------------------------------------
    log.info("[1/3] Limpieza y filtrado")
    seen: set[str] = set()
    train, train_stats = clean_split(
        read_jsonl(RAW_DIR / "train.jsonl"), cfg.data.min_words, cfg.data.max_words, seen, "duplicado"
    )
    # Un cuento de validación idéntico a uno de entrenamiento inflaría la nota: se elimina
    val, val_stats = clean_split(
        read_jsonl(RAW_DIR / "val.jsonl"), cfg.data.min_words, cfg.data.max_words, seen,
        "repetido en train",
    )
    log_cleaning(log, "train", train_stats)
    log_cleaning(log, "val", val_stats)
    write_jsonl(train, PROCESSED_DIR / "train_clean.jsonl")
    write_jsonl(val, PROCESSED_DIR / "val_clean.jsonl")

    # 2) Tokenizador (solo con train, para no "espiar" la validación) ----------
    log.info("[2/3] Entrenando tokenizador BPE (vocab=%d)", cfg.data.vocab_size)
    tok = train_tokenizer(train, cfg.data.vocab_size, cfg.data.tokenizer_min_frequency, len(train))
    tok.save(str(TOKENIZER_FILE))
    log.info("Tokenizador guardado en %s", TOKENIZER_FILE)

    # 3) Codificación ------------------------------------------------------------
    log.info("[3/3] Convirtiendo texto en tokens")
    n_train = encode_to_bin(tok, train, PROCESSED_DIR / "train.bin")
    n_val = encode_to_bin(tok, val, PROCESSED_DIR / "val.bin")

    train_chars = sum(len(t) for t in train)
    tokens_needed = cfg.train.max_iters * cfg.tokens_per_iter
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_file": str(args.config),
        "fingerprint": fp,
        "dtype": "uint16",
        "vocab_size": tok.get_vocab_size(),
        "eot_token": "<|endoftext|>",
        "eot_id": tok.token_to_id("<|endoftext|>"),
        "tokenizer_sha256": hashlib.sha256(TOKENIZER_FILE.read_bytes()).hexdigest(),
        "splits": {
            "train": {"documents": len(train), "tokens": n_train},
            "val": {"documents": len(val), "tokens": n_val},
        },
        "chars_per_token": round(train_chars / max(n_train - len(train), 1), 2),
    }
    META_FILE.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("-" * 60)
    log.info("Vocabulario        : %d tokens", meta["vocab_size"])
    log.info("Tokens train / val : %s / %s", f"{n_train:,}", f"{n_val:,}")
    log.info("Compresión         : %.2f caracteres por token", meta["chars_per_token"])
    log.info("Tamaño en disco    : %.1f MB + %.1f MB", n_train * 2 / 1e6, n_val * 2 / 1e6)
    log.info("El entrenamiento usará ~%s tokens (%.2f épocas sobre train)",
             f"{tokens_needed:,}", tokens_needed / max(n_train, 1))
    log.info("Etapa 3 completada en %.0f s. Prueba: mini-llm-tokens \"Había una vez...\"",
             time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
