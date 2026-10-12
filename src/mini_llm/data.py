"""Etapa 2: descarga de un subconjunto del dataset desde Hugging Face.

- Usa *streaming*: solo descarga lo que necesitamos, no el corpus entero.
- Filtro temático opcional: con `topic_keywords` en el YAML solo se guardan los
  documentos que hablan del tema (p. ej. tecnología dentro de toda Wikipedia).
- Si el dataset tiene un único split (como Wikipedia), train y val se separan
  durante la descarga: nunca comparten documentos.
- Es tolerante a cortes de red: reintenta y continúa donde iba.
- Escribe de forma atómica (.tmp -> .jsonl) y genera un manifiesto con estadísticas.

Uso:
    mini-llm-data
    mini-llm-data --config configs/tech.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

# Tiempos de espera más generosos para conexiones lentas (antes de importar datasets)
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

from tqdm import tqdm

from mini_llm.config import Config, DataConfig
from mini_llm.utils import CONFIGS_DIR, LOGS_DIR, get_logger, raw_dir

MAX_RETRIES = 8
TOPIC_WINDOW = 3000  # el tema se juzga por el inicio del documento (donde se define de qué trata)

StreamFn = Callable[[DataConfig, str, int], Iterator[str]]


def stream_texts(data_cfg: DataConfig, split: str, skip: int = 0) -> Iterator[str]:
    """Recorre el dataset en streaming, saltando los primeros `skip` documentos."""
    from datasets import load_dataset

    ds = load_dataset(data_cfg.dataset, name=data_cfg.subset or None, split=split, streaming=True)
    if skip:
        ds = ds.skip(skip)
    for row in ds:
        if data_cfg.text_field not in row:
            raise KeyError(f"La columna '{data_cfg.text_field}' no existe. Columnas: {list(row)}")
        yield row[data_cfg.text_field]


# --- Filtro temático ---------------------------------------------------------------------
class TopicFilter:
    """Acepta un texto si menciona suficientes términos del tema.

    Cuenta palabras completas (con plural opcional): "datos" no coincide con "candidatos".
    """

    def __init__(self, data_cfg: DataConfig) -> None:
        self.enabled = bool(data_cfg.topic_keywords)
        self.min_terms = data_cfg.min_topic_terms
        self.min_hits = data_cfg.min_topic_hits
        if self.enabled:
            terms = sorted({k.lower() for k in data_cfg.topic_keywords}, key=len, reverse=True)
            alternatives = "|".join(re.escape(t) for t in terms)
            self.pattern = re.compile(rf"\b({alternatives})(?:s|es)?\b")

    def score(self, text: str) -> tuple[int, int]:
        """(términos distintos, apariciones totales) en el inicio del texto."""
        found = self.pattern.findall(text[:TOPIC_WINDOW].lower())
        return len(set(found)), len(found)

    def accepts(self, text: str) -> bool:
        if not self.enabled:
            return True
        distinct, hits = self.score(text)
        return distinct >= self.min_terms and hits >= self.min_hits


# --- Descarga ---------------------------------------------------------------------------
def download(
    cfg: Config,
    targets: dict[str, int],
    files: dict[str, TextIO],
    log: logging.Logger,
    stream_fn: StreamFn = stream_texts,
) -> dict[str, int]:
    """Llena cada archivo con `targets[key]` documentos. Devuelve cuántos escribió de cada uno."""
    d = cfg.data
    if d.train_split == d.val_split:
        plans = [(d.train_split, ["train", "val"])]  # un solo split: se reparte al vuelo
    else:
        plans = [(d.train_split, ["train"]), (d.val_split, ["val"])]
    written = dict.fromkeys(targets, 0)
    for source, keys in plans:
        _consume(cfg, source, keys, targets, written, files, log, stream_fn)
    return written


def _route(accepted: int, keys: list[str], targets: dict[str, int], written: dict[str, int]) -> str | None:
    """Decide a qué conjunto va el documento aceptado número `accepted`."""
    if len(keys) == 1:
        return keys[0] if written[keys[0]] < targets[keys[0]] else None
    every = max(2, round((targets["train"] + targets["val"]) / max(targets["val"], 1)))
    if accepted % every == every - 1 and written["val"] < targets["val"]:
        return "val"
    if written["train"] < targets["train"]:
        return "train"
    return "val" if written["val"] < targets["val"] else None


def _consume(cfg, source, keys, targets, written, files, log, stream_fn) -> None:
    topic = TopicFilter(cfg.data)
    consumed = accepted = attempt = 0
    total = sum(targets[k] for k in keys)
    with tqdm(total=total, desc=f"Descargando {source}", unit="doc") as bar:
        while any(written[k] < targets[k] for k in keys):
            try:
                for text in stream_fn(cfg.data, source, consumed):
                    consumed += 1
                    if topic.enabled and consumed % 500 == 0:
                        bar.set_postfix(revisados=f"{consumed:,}", aceptados=f"{100 * accepted / consumed:.1f}%")
                    text = (text or "").strip()
                    if not text or not topic.accepts(text):
                        continue
                    key = _route(accepted, keys, targets, written)
                    accepted += 1
                    if key is None:
                        continue
                    files[key].write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    written[key] += 1
                    bar.update(1)
                    if all(written[k] >= targets[k] for k in keys):
                        break
                else:
                    log.warning("El split '%s' se agotó: %s", source, {k: written[k] for k in keys})
                    break
            except (KeyError, ValueError):
                raise  # errores de configuración: reintentar no sirve
            except Exception as exc:  # errores de red u otros transitorios
                attempt += 1
                if attempt > MAX_RETRIES:
                    raise RuntimeError(f"Falló la descarga tras {MAX_RETRIES} reintentos") from exc
                wait = min(2**attempt, 60)
                log.warning("Error de red (%s). Reintento %d/%d en %d s, desde el documento %d.",
                            type(exc).__name__, attempt, MAX_RETRIES, wait, consumed)
                for f in files.values():
                    f.flush()
                time.sleep(wait)
    if topic.enabled:
        log.info("Filtro temático: %s documentos revisados, %s aceptados (%.1f %%)",
                 f"{consumed:,}", f"{accepted:,}", 100 * accepted / max(consumed, 1))


def file_stats(path: Path) -> dict:
    """Estadísticas básicas del archivo JSONL y su huella SHA-256."""
    sha = hashlib.sha256()
    with path.open("rb") as fb:
        for chunk in iter(lambda: fb.read(1 << 20), b""):
            sha.update(chunk)
    docs = chars = words = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            docs += 1
            chars += len(text)
            words += len(text.split())
    return {
        "file": path.name,
        "documents": docs,
        "characters": chars,
        "words": words,
        "avg_words_per_doc": round(words / max(docs, 1), 1),
        "size_mb": round(path.stat().st_size / 1e6, 2),
        "sha256": sha.hexdigest(),
    }


def source_signature(cfg: Config) -> dict:
    """Todo lo que define QUÉ se descarga: si cambia, hay que volver a descargar."""
    d = cfg.data
    return {
        "dataset": d.dataset,
        "subset": d.subset,
        "text_field": d.text_field,
        "source_splits": {"train": d.train_split, "val": d.val_split},
        "topic_keywords": sorted(d.topic_keywords),
        "min_topic_terms": d.min_topic_terms,
        "min_topic_hits": d.min_topic_hits,
    }


def is_up_to_date(manifest_path: Path, cfg: Config, targets: dict[str, int]) -> bool:
    """True si ya existe una descarga completa que coincide con la configuración."""
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source") != source_signature(cfg):
        return False
    for key, limit in targets.items():
        info = manifest.get("splits", {}).get(key)
        if not info or info["documents"] != limit or not (manifest_path.parent / info["file"]).exists():
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Etapa 2: descargar el dataset")
    parser.add_argument("--config", default=str(CONFIGS_DIR / "cpu.yaml"))
    parser.add_argument("--force", action="store_true", help="descargar aunque ya exista")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    log = get_logger("data", LOGS_DIR / "data.log")
    out_dir = raw_dir(cfg.data.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    targets = {"train": cfg.data.train_samples, "val": cfg.data.val_samples}

    if not args.force and is_up_to_date(manifest_path, cfg, targets):
        log.info("El dataset ya está descargado en %s y coincide con la configuración.", out_dir)
        log.info("Usa --force para volver a descargarlo.")
        return 0

    log.info("Dataset: %s [%s] -> %s | train=%d docs | val=%d docs",
             cfg.data.dataset, cfg.data.subset or "-", out_dir, targets["train"], targets["val"])
    if cfg.data.topic_keywords:
        log.info("Filtro temático: %d términos, mínimo %d distintos y %d apariciones",
                 len(cfg.data.topic_keywords), cfg.data.min_topic_terms, cfg.data.min_topic_hits)
    t0 = time.perf_counter()
    finals = {k: out_dir / f"{k}.jsonl" for k in targets}
    tmps = {k: p.with_suffix(".jsonl.tmp") for k, p in finals.items()}
    files = {k: tmps[k].open("w", encoding="utf-8") for k in targets}
    try:
        download(cfg, targets, files, log)
    finally:
        for f in files.values():
            f.close()

    splits = {}
    for key in targets:
        tmps[key].replace(finals[key])  # escritura atómica
        splits[key] = s = file_stats(finals[key])
        log.info("%-5s -> %s docs | %s palabras | %.1f MB | ~%.0f palabras/doc",
                 key, f"{s['documents']:,}", f"{s['words']:,}", s["size_mb"], s["avg_words_per_doc"])

    manifest = {
        "source": source_signature(cfg),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_file": str(args.config),
        "splits": splits,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Manifiesto guardado en %s (%.0f s)", manifest_path, time.perf_counter() - t0)

    with finals["val"].open(encoding="utf-8") as f:
        first = f.readline()
    if first:
        log.info("Ejemplo de documento:\n%s", json.loads(first)["text"][:600])
    return 0


if __name__ == "__main__":
    sys.exit(main())
