"""Etapa 2: descarga de un subconjunto del dataset desde Hugging Face.

- Usa *streaming*: solo descarga los documentos que necesitamos, no el corpus entero.
- Es tolerante a cortes de red: si la conexión falla, reintenta y continúa donde iba.
- Escribe de forma atómica (.tmp -> .jsonl) para no dejar archivos a medias.
- Genera un manifiesto con estadísticas y huella SHA-256 para la trazabilidad.

Uso:
    mini-llm-data
    mini-llm-data --config configs/gpu.yaml --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path

# Tiempos de espera más generosos para conexiones lentas (antes de importar datasets)
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

from tqdm import tqdm

from mini_llm.config import Config, DataConfig
from mini_llm.utils import CONFIGS_DIR, LOGS_DIR, RAW_DIR, get_logger

MAX_RETRIES = 8

StreamFn = Callable[[DataConfig, str, int], Iterator[str]]


def source_split(data_cfg: DataConfig, split_key: str) -> str:
    """Traduce el nombre local ('train' / 'val') al nombre del split en Hugging Face."""
    return {"train": data_cfg.train_split, "val": data_cfg.val_split}[split_key]


def stream_texts(data_cfg: DataConfig, split: str, skip: int = 0) -> Iterator[str]:
    """Recorre el dataset en streaming, saltando los primeros `skip` documentos."""
    from datasets import load_dataset

    ds = load_dataset(data_cfg.dataset, name=data_cfg.subset or None, split=split, streaming=True)
    if skip:
        ds = ds.skip(skip)
    for row in ds:
        if data_cfg.text_field not in row:
            raise KeyError(
                f"La columna '{data_cfg.text_field}' no existe. Columnas: {list(row)}"
            )
        yield row[data_cfg.text_field]


def download_split(
    cfg: Config,
    split_key: str,
    limit: int,
    out_path: Path,
    log: logging.Logger,
    stream_fn: StreamFn = stream_texts,
) -> int:
    """Descarga hasta `limit` documentos no vacíos en `out_path` (JSONL). Devuelve cuántos."""
    consumed = written = attempt = 0
    with out_path.open("w", encoding="utf-8") as f, tqdm(
        total=limit, desc=f"Descargando {split_key}", unit="doc"
    ) as bar:
        while written < limit:
            try:
                stream = stream_fn(cfg.data, source_split(cfg.data, split_key), consumed)
                for text in stream:
                    consumed += 1
                    text = (text or "").strip()
                    if not text:
                        continue
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    written += 1
                    bar.update(1)
                    if written >= limit:
                        break
                else:
                    log.warning("El split '%s' se agotó con %d documentos.", split_key, written)
                    break
            except (KeyError, ValueError):
                raise  # errores de configuración: reintentar no sirve
            except Exception as exc:  # errores de red u otros transitorios
                attempt += 1
                if attempt > MAX_RETRIES:
                    raise RuntimeError(f"Falló la descarga tras {MAX_RETRIES} reintentos") from exc
                wait = min(2**attempt, 60)
                log.warning(
                    "Error de red (%s). Reintento %d/%d en %d s, continuando desde el documento %d.",
                    type(exc).__name__, attempt, MAX_RETRIES, wait, consumed,
                )
                f.flush()
                time.sleep(wait)
    return written


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


def is_up_to_date(manifest_path: Path, cfg: Config, targets: dict[str, int]) -> bool:
    """True si ya existe una descarga completa que coincide con la configuración."""
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != cfg.data.dataset or manifest.get("subset") != cfg.data.subset:
        return False
    for key, limit in targets.items():
        info = manifest.get("splits", {}).get(key)
        if not info or info["documents"] != limit or not (RAW_DIR / info["file"]).exists():
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Etapa 2: descargar el dataset")
    parser.add_argument("--config", default=str(CONFIGS_DIR / "cpu.yaml"))
    parser.add_argument("--force", action="store_true", help="descargar aunque ya exista")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    log = get_logger("data", LOGS_DIR / "data.log")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW_DIR / "manifest.json"
    targets = {"train": cfg.data.train_samples, "val": cfg.data.val_samples}

    if not args.force and is_up_to_date(manifest_path, cfg, targets):
        log.info("El dataset ya está descargado y coincide con la configuración.")
        log.info("Usa --force para volver a descargarlo.")
        return 0

    log.info("Dataset: %s [%s] | train=%d docs | val=%d docs",
             cfg.data.dataset, cfg.data.subset or "-", targets["train"], targets["val"])
    t0 = time.perf_counter()
    splits = {}
    for key, limit in targets.items():
        final = RAW_DIR / f"{key}.jsonl"
        tmp = final.with_suffix(".jsonl.tmp")
        download_split(cfg, key, limit, tmp, log)
        tmp.replace(final)  # escritura atómica
        splits[key] = file_stats(final)
        s = splits[key]
        log.info("%-5s -> %s docs | %s palabras | %.1f MB | ~%.0f palabras/doc",
                 key, f"{s['documents']:,}", f"{s['words']:,}", s["size_mb"], s["avg_words_per_doc"])

    manifest = {
        "dataset": cfg.data.dataset,
        "subset": cfg.data.subset,
        "text_field": cfg.data.text_field,
        "source_splits": {"train": cfg.data.train_split, "val": cfg.data.val_split},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_file": str(args.config),
        "splits": splits,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Manifiesto guardado en %s (%.0f s)", manifest_path, time.perf_counter() - t0)

    with (RAW_DIR / "val.jsonl").open(encoding="utf-8") as f:
        sample = json.loads(f.readline())["text"]
    log.info("Ejemplo de documento:\n%s", sample[:600])
    return 0


if __name__ == "__main__":
    sys.exit(main())
